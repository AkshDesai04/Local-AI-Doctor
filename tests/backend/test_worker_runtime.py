from __future__ import annotations

import json
import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import numpy as np
import pytest

from local_ai_doctor.workers.runtime import (
    WorkerRuntime,
    _decoder_start_token,
    _embedding_dimension_bounds,
    _embedding_model_kwargs,
    _mean_causal_self_attention,
    _reasoning_segmenter,
    _render_messages,
    _safe_error,
)


def test_qwen3_vl_embedding_loader_strips_task_wrapper_prefix(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "qwen3_vl"}), encoding="utf-8")

    original = {"dtype": "sentinel"}
    result = _embedding_model_kwargs(tmp_path, original)

    assert result == {"dtype": "sentinel", "key_mapping": {r"^model\.": ""}}
    assert original == {"dtype": "sentinel"}


def test_worker_error_payload_never_exposes_raw_exception_text_or_paths() -> None:
    private_message = r"failed to open C:\Users\alice\private-model\weights.safetensors"

    error = _safe_error(FileNotFoundError(private_message))

    assert error == {
        "code": "model_worker_error",
        "message": "model worker operation failed",
        "hint": "Review the model diagnostics and worker runtime dependencies.",
        "exception": "FileNotFoundError",
    }
    assert "alice" not in str(error).lower()
    assert "weights.safetensors" not in str(error)


def test_attention_aggregation_retains_exact_top_weights_and_reports_omitted_mass() -> None:
    import torch

    sources = [
        {
            "context_index": index,
            "token_id": index + 10,
            "piece": str(index),
            "display_text": str(index),
            "source_kind": "prompt",
        }
        for index in range(4)
    ]
    attentions = (
        torch.tensor([[[[0.1, 0.2, 0.3, 0.4]], [[0.4, 0.3, 0.2, 0.1]]]]),
        torch.tensor([[[[0.0, 0.2, 0.3, 0.5]], [[0.2, 0.2, 0.2, 0.4]]]]),
    )

    attribution, error = _mean_causal_self_attention(
        torch,
        attentions,
        sources,
        source_limit=2,
    )

    assert error is None
    assert attribution is not None
    assert attribution["method"] == "mean_causal_self_attention"
    assert attribution["semantics"] == "attention_weights_not_causal_contributions"
    assert attribution["scope"] == (
        "decoder_step_context_attention_independent_of_sampled_candidate"
    )
    assert attribution["captured_layers"] == [0, 1]
    assert attribution["captured_heads"] == 2
    assert attribution["total_head_rows"] == 4
    assert attribution["total_source_count"] == 4
    assert attribution["retained_source_count"] == 2
    assert [item["context_index"] for item in attribution["source_tokens"]] == [2, 3]
    assert attribution["retained_weight"] == pytest.approx(0.6)
    assert attribution["omitted_weight"] == pytest.approx(0.4)


def test_attention_capture_falls_back_to_native_model_outputs_without_hooks() -> None:
    import torch

    native_attentions = (torch.ones((1, 1, 1, 2)),)

    class Body:
        @staticmethod
        def __call__(**_kwargs: Any) -> Any:
            return SimpleNamespace(
                last_hidden_state=torch.zeros((1, 1, 2)),
                past_key_values="cache",
                attentions=native_attentions,
            )

    class Head:
        @staticmethod
        def __call__(_hidden: Any) -> Any:
            return torch.zeros((1, 1, 3))

    class NativeAttentionModel:
        config = SimpleNamespace(is_encoder_decoder=False)
        model = Body()
        lm_head = Head()

        @staticmethod
        def named_modules() -> list[tuple[str, Any]]:
            return []

    runtime = WorkerRuntime(cast(Any, FakeQueue()), cast(Any, FakeQueue()), FakeCancelEvent())
    runtime.model = NativeAttentionModel()
    logits, past, attentions = runtime._forward_last(
        torch,
        torch.tensor([[1, 2]]),
        torch.ones((1, 2), dtype=torch.long),
        capture_attention=True,
    )

    assert logits.shape == (1, 3)
    assert past == "cache"
    assert attentions is native_attentions


def test_embedding_loader_does_not_remap_other_architectures(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"model_type": "bert"}), encoding="utf-8")

    assert _embedding_model_kwargs(tmp_path, {"dtype": "sentinel"}) == {"dtype": "sentinel"}


def test_embedding_loader_tolerates_unreadable_metadata(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text("not-json", encoding="utf-8")

    assert _embedding_model_kwargs(tmp_path, {"dtype": "sentinel"}) == {"dtype": "sentinel"}


class FakeTensor:
    def __init__(self, values: Any) -> None:
        self.values = np.asarray(values)
        self.device = "cpu"

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(int(dimension) for dimension in self.values.shape)

    @property
    def dtype(self) -> np.dtype[Any]:
        return np.dtype(self.values.dtype)

    def to(self, *_args: Any, **_kwargs: Any) -> FakeTensor:
        return self

    def clone(self) -> FakeTensor:
        return FakeTensor(self.values.copy())

    def float(self) -> FakeTensor:
        return FakeTensor(self.values.astype(np.float32))

    def item(self) -> Any:
        return self.values.item()

    def tolist(self) -> Any:
        return self.values.tolist()

    def numel(self) -> int:
        return int(self.values.size)

    def __getitem__(self, key: Any) -> FakeTensor:
        return FakeTensor(self.values[key])

    def __setitem__(self, key: Any, value: Any) -> None:
        self.values[key] = value.values if isinstance(value, FakeTensor) else value


class FakeGenerator:
    def __init__(self, _device: str) -> None:
        self.seed: int | None = None

    def manual_seed(self, seed: int) -> None:
        self.seed = seed


class FakeCuda:
    @staticmethod
    def is_available() -> bool:
        return False


class FakeTorch(ModuleType):
    def __init__(self) -> None:
        super().__init__("torch")
        self.long = "long"
        self.inf = float("inf")
        self.cuda = FakeCuda()
        self.backends = SimpleNamespace(cudnn=SimpleNamespace(benchmark=True))
        self.deterministic_calls: list[tuple[bool, bool]] = []

    def use_deterministic_algorithms(self, enabled: bool, *, warn_only: bool) -> None:
        self.deterministic_calls.append((enabled, warn_only))

    @staticmethod
    def Generator(device: str) -> FakeGenerator:
        return FakeGenerator(device)

    @staticmethod
    def inference_mode() -> Any:
        return nullcontext()

    @staticmethod
    def ones_like(tensor: FakeTensor) -> FakeTensor:
        return FakeTensor(np.ones_like(tensor.values))

    @staticmethod
    def ones(shape: tuple[int, int], **kwargs: Any) -> FakeTensor:
        dtype = kwargs.get("dtype", np.int64)
        if not isinstance(dtype, np.dtype) and dtype == "long":
            dtype = np.int64
        return FakeTensor(np.ones(shape, dtype=dtype))

    @staticmethod
    def tensor(values: Any, **_kwargs: Any) -> FakeTensor:
        return FakeTensor(values)

    @staticmethod
    def cat(tensors: list[FakeTensor], dim: int) -> FakeTensor:
        return FakeTensor(np.concatenate([tensor.values for tensor in tensors], axis=dim))

    @staticmethod
    def argmax(tensor: FakeTensor) -> FakeTensor:
        return FakeTensor(np.asarray(np.argmax(tensor.values)))

    @staticmethod
    def full_like(tensor: FakeTensor, value: float) -> FakeTensor:
        return FakeTensor(np.full_like(tensor.values, value, dtype=np.float32))

    @staticmethod
    def arange(start: int, end: int, **_kwargs: Any) -> FakeTensor:
        return FakeTensor(np.arange(start, end))


class FakeQueue:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def put(self, item: dict[str, Any]) -> None:
        self.items.append(item)


class FakeCancelEvent:
    def __init__(self, *, cancelled: bool = False) -> None:
        self.cancelled = cancelled

    def clear(self) -> None:
        pass

    def is_set(self) -> bool:
        return self.cancelled


class FakeTokenizer:
    chat_template = None
    eos_token_id = 3
    bos_token_id = None

    def __call__(self, text: str, **_kwargs: Any) -> dict[str, FakeTensor]:
        assert text == "summarize this"
        return {
            "input_ids": FakeTensor([[5, 6]]),
            "attention_mask": FakeTensor([[1, 1]]),
        }

    @staticmethod
    def convert_ids_to_tokens(token_id: int) -> str:
        return {2: "hello", 3: "</s>", 4: "alternate"}[token_id]

    @staticmethod
    def decode(token_ids: list[int], **_kwargs: Any) -> str:
        return "".join({2: "hello", 3: "</s>", 4: "alternate"}[token_id] for token_id in token_ids)


class FakeEncoder:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []
        self.output = SimpleNamespace(last_hidden_state=FakeTensor([[[1.0]]]))

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        return self.output


class FakeEncoderDecoderModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(
            is_encoder_decoder=True,
            decoder_start_token_id=1,
            bos_token_id=None,
            eos_token_id=3,
        )
        self.generation_config = SimpleNamespace(
            decoder_start_token_id=1,
            bos_token_id=None,
            eos_token_id=3,
        )
        self.encoder = FakeEncoder()
        self.decoder_calls: list[dict[str, Any]] = []

    def get_encoder(self) -> FakeEncoder:
        return self.encoder

    def __call__(self, **kwargs: Any) -> Any:
        self.decoder_calls.append(kwargs)
        if len(self.decoder_calls) == 1:
            logits = FakeTensor([[[0.0, 0.0, 9.0, 1.0, 3.0]]])
        else:
            logits = FakeTensor([[[0.0, 0.0, 1.0, 9.0, 0.0]]])
        return SimpleNamespace(logits=logits, past_key_values=f"cache-{len(self.decoder_calls)}")


class FakeCausalModel:
    def __init__(self) -> None:
        self.config = SimpleNamespace(is_encoder_decoder=False, eos_token_id=3)
        self.generation_config = SimpleNamespace(eos_token_id=3)
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        input_ids = kwargs["input_ids"].tolist()
        past = kwargs["past_key_values"]
        values = np.full((1, 1, 5), -10.0, dtype=np.float32)
        if len(self.calls) == 1:
            assert input_ids == [[5, 6]]
            assert past is None
            # The original path would choose token 2; the branch substitutes 4.
            values[0, 0, 2] = 10.0
            next_cache = "cache-after-prompt"
        elif len(self.calls) == 2:
            assert input_ids == [[4]]
            assert past == "cache-after-prompt"
            values[0, 0, 2] = 10.0
            next_cache = "cache-after-substitution-4"
        else:
            assert input_ids == [[2]]
            assert past == "cache-after-substitution-4"
            values[0, 0, 3] = 10.0
            next_cache = "cache-after-suffix-2"
        return SimpleNamespace(logits=FakeTensor(values), past_key_values=next_cache)


class FakeSentenceModel:
    @staticmethod
    def encode(_values: list[Any], **_kwargs: Any) -> np.ndarray[Any, np.dtype[np.float32]]:
        return np.arange(128, dtype=np.float32).reshape(1, 128)


class FakeReasoningTokenizer:
    eos_token_id = 99
    bos_token_id = None

    def __init__(self, pieces: tuple[str, ...], *, prompt_primed: bool) -> None:
        self._pieces = pieces
        self.chat_template = "reasoning-template" if prompt_primed else None

    def apply_chat_template(self, _messages: Any, **_kwargs: Any) -> str:
        return "Assistant: <think>\n"

    def __call__(self, _text: str, **_kwargs: Any) -> dict[str, FakeTensor]:
        return {
            "input_ids": FakeTensor([[5, 6]]),
            "attention_mask": FakeTensor([[1, 1]]),
        }

    def convert_ids_to_tokens(self, token_id: int) -> str:
        if token_id == self.eos_token_id:
            return "</s>"
        return self._pieces[token_id - 2]

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return "".join(
            "</s>" if token_id == self.eos_token_id else self._pieces[token_id - 2]
            for token_id in token_ids
        )


class FakeReasoningEncoderDecoderModel:
    def __init__(self, token_ids: tuple[int, ...]) -> None:
        self.config = SimpleNamespace(
            is_encoder_decoder=True,
            decoder_start_token_id=1,
            bos_token_id=None,
            eos_token_id=99,
        )
        self.generation_config = SimpleNamespace(
            decoder_start_token_id=1,
            bos_token_id=None,
            eos_token_id=99,
        )
        self.encoder = FakeEncoder()
        self.token_ids = token_ids
        self.decoder_calls: list[dict[str, Any]] = []

    def get_encoder(self) -> FakeEncoder:
        return self.encoder

    def __call__(self, **kwargs: Any) -> Any:
        self.decoder_calls.append(kwargs)
        chosen_id = self.token_ids[len(self.decoder_calls) - 1]
        values = np.full((1, 1, 100), -10.0, dtype=np.float32)
        values[0, 0, chosen_id] = 10.0
        return SimpleNamespace(
            logits=FakeTensor(values),
            past_key_values=f"cache-{len(self.decoder_calls)}",
        )


def generation_command() -> dict[str, Any]:
    return {
        "run_id": "run-1",
        "request_id": "request-1",
        "messages": [{"role": "user", "content": "summarize this"}],
        "effective_seed": 0,
        "instrumentation": "basic",
        "sampling": {
            "max_output_tokens": 4,
            "temperature": 0.0,
            "top_k": 0,
            "top_p": 1.0,
            "min_p": 0.0,
            "repetition_penalty": 1.0,
            "frequency_penalty": 0.0,
            "presence_penalty": 0.0,
            "alternatives": 0,
            "stop_sequences": [],
        },
    }


def configured_encoder_decoder_runtime(
    *, cancelled: bool = False
) -> tuple[WorkerRuntime, Any, FakeQueue]:
    output = FakeQueue()
    runtime = WorkerRuntime(
        cast(Any, FakeQueue()),
        cast(Any, output),
        FakeCancelEvent(cancelled=cancelled),
    )
    model = FakeEncoderDecoderModel()
    runtime.model = model
    runtime.tokenizer = FakeTokenizer()
    runtime.model_info = {
        "id": "tiny-seq2seq",
        "display_name": "Tiny Seq2Seq Fixture",
        "task": "encoder_decoder_generation",
        "effective_context_limit": 32,
    }
    runtime.decoder_start_token_id = 1
    runtime.decoder_start_token_source = "config.decoder_start_token_id"
    return runtime, model, output


def configured_causal_runtime() -> tuple[WorkerRuntime, FakeCausalModel, FakeQueue]:
    output = FakeQueue()
    runtime = WorkerRuntime(cast(Any, FakeQueue()), cast(Any, output), FakeCancelEvent())
    model = FakeCausalModel()
    runtime.model = model
    runtime.tokenizer = FakeTokenizer()
    runtime.model_info = {
        "id": "tiny-causal",
        "display_name": "Tiny Causal Fixture",
        "task": "text_generation",
        "effective_context_limit": 32,
    }
    return runtime, model, output


def configured_reasoning_runtime(
    pieces: tuple[str, ...], *, prompt_primed: bool, token_ids: tuple[int, ...] | None = None
) -> tuple[WorkerRuntime, FakeQueue]:
    output = FakeQueue()
    runtime = WorkerRuntime(cast(Any, FakeQueue()), cast(Any, output), FakeCancelEvent())
    runtime.model = FakeReasoningEncoderDecoderModel(token_ids or tuple(range(2, len(pieces) + 2)))
    runtime.tokenizer = FakeReasoningTokenizer(pieces, prompt_primed=prompt_primed)
    runtime.model_info = {
        "id": "tiny-reasoner",
        "display_name": "Tiny Reasoner Fixture",
        "task": "encoder_decoder_generation",
        "effective_context_limit": 32,
        "reasoning_delimiters": ("<think>", "</think>"),
    }
    runtime.decoder_start_token_id = 1
    runtime.decoder_start_token_source = "config.decoder_start_token_id"
    return runtime, output


class BosTokenizer(FakeTokenizer):
    """A base-model tokenizer that inserts BOS by default, like Llama 3.2's."""

    bos_token_id = 1

    def __init__(self, *, adds_bos: bool = True, chat_template: str | None = None) -> None:
        self.adds_bos = adds_bos
        self.chat_template = chat_template

    def apply_chat_template(self, _messages: Any, **_options: Any) -> str:
        return "summarize this"

    def __call__(self, text: str, **kwargs: Any) -> Any:
        if text == "":
            return {"input_ids": [1] if self.adds_bos and kwargs["add_special_tokens"] else []}
        return super().__call__(text, **kwargs)


class RecordingCausalModel:
    """Greedy fixture that emits token 2 twice, then EOS, recording every call."""

    def __init__(self, *, reject: frozenset[str] = frozenset()) -> None:
        self.config = SimpleNamespace(is_encoder_decoder=False, eos_token_id=3)
        self.generation_config = SimpleNamespace(eos_token_id=3)
        self.calls: list[dict[str, Any]] = []
        self.reject = reject

    def __call__(self, **kwargs: Any) -> Any:
        rejected = self.reject.intersection(kwargs)
        if rejected:
            raise TypeError(f"unexpected keyword argument {sorted(rejected)[0]!r}")
        self.calls.append(kwargs)
        values = np.full((1, 1, 5), -10.0, dtype=np.float32)
        values[0, 0, 3 if len(self.calls) > 2 else 2] = 10.0
        return SimpleNamespace(logits=FakeTensor(values), past_key_values="cache")


@pytest.mark.parametrize(
    ("tokenizer", "expected_ids"),
    [
        (BosTokenizer(), [1, 5, 6]),
        (BosTokenizer(adds_bos=False), [5, 6]),
        # A chat template writes its own BOS, so templated prompts are untouched.
        (BosTokenizer(chat_template="fixture"), [5, 6]),
    ],
)
def test_plain_text_fallback_restores_the_default_bos_token(
    monkeypatch: pytest.MonkeyPatch, tokenizer: BosTokenizer, expected_ids: list[int]
) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    runtime, _model, output = configured_causal_runtime()
    model = RecordingCausalModel()
    runtime.model = model
    runtime.tokenizer = tokenizer

    runtime._generate(generation_command())

    assert model.calls[0]["input_ids"].tolist() == [expected_ids]
    assert model.calls[0]["attention_mask"].tolist() == [[1] * len(expected_ids)]
    stage = next(
        item["payload"]
        for item in output.items
        if item["kind"] == "run_event" and item["event_type"] == "stage"
    )
    assert stage["prompt_tokens"] == len(expected_ids)


def test_decoder_calls_receive_absolute_cache_positions(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    runtime, _model, _output = configured_causal_runtime()
    model = RecordingCausalModel()
    runtime.model = model

    runtime._generate(generation_command())

    assert [call["cache_position"].tolist() for call in model.calls] == [[0, 1], [2], [3]]
    assert all(call["logits_to_keep"] == 1 for call in model.calls)


def test_families_rejecting_new_keywords_still_generate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    runtime, _model, output = configured_causal_runtime()
    model = RecordingCausalModel(reject=frozenset({"cache_position", "logits_to_keep"}))
    runtime.model = model

    runtime._generate(generation_command())

    assert len(model.calls) == 3
    assert not any("cache_position" in call for call in model.calls)
    completed = next(item for item in output.items if item.get("event_type") == "completed")
    assert completed["payload"]["text"] == "hellohello</s>"


class FakeProcessor:
    """Processor fixture: renders content parts and expands one image into two pads."""

    chat_template = "fixture"
    image_processor = SimpleNamespace(merge_size=1)
    video_processor = SimpleNamespace(merge_size=1)

    def __init__(self) -> None:
        self.rendered: list[Any] = []
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, messages: Any, **_options: Any) -> str:
        self.rendered.append(messages)
        return "summarize this"

    def __call__(self, **kwargs: Any) -> dict[str, FakeTensor]:
        self.calls.append(kwargs)
        return {
            "input_ids": FakeTensor([[7, 7, 5, 6]]),
            "attention_mask": FakeTensor([[1, 1, 1, 1]]),
            "pixel_values": FakeTensor([[0.5, 0.5]]),
            "image_grid_thw": FakeTensor([[1, 1, 2]]),
        }


def _media_command(image_path: Path) -> dict[str, Any]:
    command = generation_command()
    command["messages"] = [
        {"role": "user", "content": "earlier", "attachments": []},
        {"role": "assistant", "content": "noted"},
        {
            "role": "user",
            "content": "summarize this",
            "attachments": [{"kind": "image", "path": str(image_path)}],
        },
    ]
    return command


def test_media_prompts_use_processor_content_parts_and_prefill_only_pixels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PIL import Image

    image_path = tmp_path / "red.png"
    Image.new("RGB", (4, 4), (255, 0, 0)).save(image_path)
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    runtime, _model, output = configured_causal_runtime()
    model = RecordingCausalModel()
    model.config.image_token_id = 7
    runtime.model = model
    processor = FakeProcessor()
    runtime.processor = processor

    runtime._generate(_media_command(image_path))

    assert processor.rendered[0][-1]["content"] == [
        {"type": "image"},
        {"type": "text", "text": "summarize this"},
    ]
    assert processor.rendered[0][0]["content"] == [{"type": "text", "text": "earlier"}]
    assert "path" not in json.dumps(processor.rendered[0])
    images = processor.calls[0]["images"]
    assert len(images) == 1 and images[0].mode == "RGB"
    assert processor.calls[0]["add_special_tokens"] is False
    assert "pixel_values" in model.calls[0] and "image_grid_thw" in model.calls[0]
    assert all("pixel_values" not in call for call in model.calls[1:])
    assert model.calls[0]["input_ids"].tolist() == [[7, 7, 5, 6]]
    stage = next(
        item["payload"]
        for item in output.items
        if item["kind"] == "run_event" and item["event_type"] == "stage"
    )
    assert stage["media"] == {"images": 1, "videos": 0, "placeholder_tokens": 2}


def test_media_prompts_require_a_loaded_processor(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    runtime, model, _output = configured_causal_runtime()

    with pytest.raises(ValueError, match="no chat media processor"):
        runtime._generate(_media_command(tmp_path / "unused.png"))
    assert model.calls == []


def test_media_labels_assign_placeholders_to_their_attachment() -> None:
    from local_ai_doctor.workers.runtime import _media_labels

    labels = _media_labels(
        [1, 7, 7, 2, 7, 8, 8, 8, 3],
        {7: "image", 8: "video"},
        {"image": [2, 1], "video": [3]},
    )
    assert labels == [
        None,
        {"kind": "image", "index": 0},
        {"kind": "image", "index": 0},
        None,
        {"kind": "image", "index": 1},
        {"kind": "video", "index": 0},
        {"kind": "video", "index": 0},
        {"kind": "video", "index": 0},
        None,
    ]
    # Counts that do not match the prompt are not trusted for item indices.
    assert _media_labels([7, 7], {7: "image"}, {"image": [3]}) == [
        {"kind": "image"},
        {"kind": "image"},
    ]


def _tiny_qwen3_vl() -> Any:
    import torch
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    config = Qwen3VLConfig(
        text_config={
            "vocab_size": 64,
            "hidden_size": 32,
            "intermediate_size": 64,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 8,
            "max_position_embeddings": 128,
            "rope_scaling": {
                "rope_type": "default",
                "mrope_section": [2, 1, 1],
                "mrope_interleaved": True,
            },
            "bos_token_id": None,
            "eos_token_id": None,
            "pad_token_id": None,
        },
        vision_config={
            "depth": 1,
            "hidden_size": 16,
            "intermediate_size": 32,
            "num_heads": 2,
            "out_hidden_size": 32,
            "patch_size": 4,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "num_position_embeddings": 16,
            "deepstack_visual_indexes": [0],
        },
        image_token_id=60,
        video_token_id=61,
        vision_start_token_id=62,
        vision_end_token_id=63,
    )
    torch.manual_seed(0)
    return Qwen3VLForConditionalGeneration(config).eval()


class TinyIdTokenizer:
    chat_template = None
    eos_token_id = None
    bos_token_id = None

    def __init__(self, prompt_ids: list[int]) -> None:
        self.prompt_ids = prompt_ids

    def __call__(self, _text: str, **_kwargs: Any) -> dict[str, Any]:
        import torch

        return {
            "input_ids": torch.tensor([self.prompt_ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(self.prompt_ids)), dtype=torch.long),
        }

    @staticmethod
    def convert_ids_to_tokens(token_id: int) -> str:
        return f"token-{token_id}"

    @staticmethod
    def decode(token_ids: list[int], **_kwargs: Any) -> str:
        return "".join(f"<{token_id}>" for token_id in token_ids)


@pytest.mark.parametrize("instrumentation", ["token", "full"])
def test_multimodal_rope_decode_matches_reference_generate(instrumentation: str) -> None:
    """Qwen3-VL positions decode steps from cache_position plus prefill rope deltas.

    Without cache_position every decode step sat at position zero, so the reference
    loop drifted from ``generate()`` after a few tokens even for text-only prompts.
    The full tier also covers the split eager prefill used for attention capture.
    """

    import torch

    model = _tiny_qwen3_vl()
    prompt = [5, 6, 7, 8]
    reference = model.generate(
        input_ids=torch.tensor([prompt]),
        attention_mask=torch.ones((1, len(prompt)), dtype=torch.long),
        max_new_tokens=8,
        do_sample=False,
    )[0, len(prompt) :].tolist()
    output = FakeQueue()
    runtime = WorkerRuntime(cast(Any, FakeQueue()), cast(Any, output), FakeCancelEvent())
    runtime.model = model
    runtime.tokenizer = TinyIdTokenizer(prompt)
    runtime.model_info = {
        "id": "tiny-qwen3-vl",
        "display_name": "Tiny Qwen3-VL Fixture",
        "task": "text_generation",
        "effective_context_limit": 64,
    }
    runtime.loaded_attention_implementation = str(model.config._attn_implementation)
    command = generation_command()
    command["instrumentation"] = instrumentation
    command["sampling"]["max_output_tokens"] = 8

    runtime._generate(command)

    tokens = [
        item["payload"]["token_id"]
        for item in output.items
        if item["kind"] == "run_event" and item["event_type"] == "token"
    ]
    assert tokens == reference


def test_prompt_rendering_falls_back_without_chat_template() -> None:
    rendered, renderer = _render_messages(
        FakeTokenizer(), [{"role": "user", "content": "summarize this"}]
    )
    assert rendered == "summarize this"
    assert renderer == "plain_text_fallback"


@pytest.mark.parametrize("reasoning", [True, False])
def test_prompt_rendering_forwards_reasoning_choice_to_chat_template(reasoning: bool) -> None:
    class CapturingTokenizer:
        chat_template = "fixture"

        def __init__(self) -> None:
            self.options: dict[str, Any] = {}

        def apply_chat_template(self, _messages: Any, **options: Any) -> str:
            self.options = options
            return "rendered"

    tokenizer = CapturingTokenizer()
    rendered, renderer = _render_messages(
        tokenizer,
        [{"role": "user", "content": "hello"}],
        reasoning=reasoning,
    )

    assert rendered == "rendered"
    assert renderer == "chat_template"
    assert tokenizer.options["enable_thinking"] is reasoning


def test_prompt_rendering_closes_a_hard_coded_reasoning_prefix_when_disabled() -> None:
    class HardCodedReasoningTokenizer:
        chat_template = "fixture"

        @staticmethod
        def apply_chat_template(_messages: Any, **options: Any) -> str:
            return (
                "User: hello\nAssistant: <think>\n"
                if options["add_generation_prompt"]
                else "User: hello"
            )

    tokenizer = HardCodedReasoningTokenizer()

    rendered, renderer = _render_messages(
        tokenizer,
        [{"role": "user", "content": "hello"}],
        reasoning=False,
        reasoning_delimiters=("<think>", "</think>"),
    )

    assert renderer == "chat_template"
    assert rendered.endswith("<think>\n\n</think>\n\n")


def test_prompt_rendering_does_not_rewrite_a_literal_user_suffix() -> None:
    class LiteralTokenizer:
        chat_template = "fixture"

        @staticmethod
        def apply_chat_template(messages: Any, **_options: Any) -> str:
            return str(messages[-1]["content"])

    rendered, _renderer = _render_messages(
        LiteralTokenizer(),
        [{"role": "user", "content": "literal <think>"}],
        reasoning=False,
        reasoning_delimiters=("<think>", "</think>"),
    )

    assert rendered == "literal <think>"


class TemplateWithoutSystemRole:
    """Chat template that either rejects the system role or silently drops it."""

    chat_template = "fixture"

    def __init__(self, *, raises: bool) -> None:
        self.raises = raises
        self.calls: list[list[dict[str, Any]]] = []

    def apply_chat_template(self, messages: Any, **_options: Any) -> str:
        self.calls.append([dict(message) for message in messages])
        if any(message["role"] == "system" for message in messages):
            if self.raises:
                raise ValueError("System role not supported")
            messages = [message for message in messages if message["role"] != "system"]
        return "".join(f"<{message['role']}>{message['content']}" for message in messages)


@pytest.mark.parametrize("raises", [True, False])
def test_prompt_rendering_merges_system_prompt_when_template_has_no_system_role(
    raises: bool,
) -> None:
    tokenizer = TemplateWithoutSystemRole(raises=raises)
    messages = [
        {"role": "system", "content": "Be brief."},
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
        {"role": "user", "content": "again"},
    ]

    rendered, renderer = _render_messages(tokenizer, messages)

    assert renderer == "chat_template_system_merged"
    assert rendered == "<user>Be brief.\n\nhello<assistant>hi<user>again"
    assert tokenizer.calls[-1][0] == {"role": "user", "content": "Be brief.\n\nhello"}
    assert messages[0] == {"role": "system", "content": "Be brief."}


def test_prompt_rendering_keeps_the_system_role_when_the_template_renders_it() -> None:
    class SystemAwareTokenizer:
        chat_template = "fixture"

        @staticmethod
        def apply_chat_template(messages: Any, **_options: Any) -> str:
            return "".join(f"<{message['role']}>{message['content']}" for message in messages)

    rendered, renderer = _render_messages(
        SystemAwareTokenizer(),
        [{"role": "system", "content": "Be brief."}, {"role": "user", "content": "hello"}],
    )

    assert (rendered, renderer) == ("<system>Be brief.<user>hello", "chat_template")


def test_prompt_rendering_does_not_hide_template_errors_it_cannot_repair() -> None:
    tokenizer = TemplateWithoutSystemRole(raises=True)

    with pytest.raises(ValueError, match="System role not supported"):
        _render_messages(tokenizer, [{"role": "system", "content": "Be brief."}])
    without_system = TemplateWithoutSystemRole(raises=True)
    rendered, renderer = _render_messages(without_system, [{"role": "user", "content": "hi"}])
    assert (rendered, renderer) == ("<user>hi", "chat_template")


def test_generation_warns_when_the_system_prompt_is_merged_into_the_first_user_turn(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class MergingTokenizer(TemplateWithoutSystemRole, FakeTokenizer):
        def __init__(self) -> None:
            TemplateWithoutSystemRole.__init__(self, raises=True)
            self.prompts: list[str] = []

        def __call__(self, text: str, **kwargs: Any) -> dict[str, FakeTensor]:
            self.prompts.append(text)
            return {"input_ids": FakeTensor([[5, 6]]), "attention_mask": FakeTensor([[1, 1]])}

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    runtime, _model, output = configured_encoder_decoder_runtime()
    tokenizer = MergingTokenizer()
    runtime.tokenizer = tokenizer
    command = generation_command()
    command["messages"] = [
        {"role": "system", "content": "Answer in French."},
        {"role": "user", "content": "summarize this"},
    ]

    runtime._generate(command)

    events = [item for item in output.items if item["kind"] == "run_event"]
    warnings = [item["payload"] for item in events if item["event_type"] == "warning"]
    merged = [item for item in warnings if item["code"] == "system_prompt_merged"]
    assert len(merged) == 1
    assert "prepended to the first user message" in merged[0]["message"]
    expected_prompt = "<user>Answer in French.\n\nsummarize this"
    assert tokenizer.prompts == [expected_prompt]
    stage = next(item["payload"] for item in events if item["event_type"] == "stage")
    assert stage["rendered_prompt"] == expected_prompt


def test_runtime_reasoning_segmenter_uses_only_model_declared_delimiters() -> None:
    unknown, primed = _reasoning_segmenter({}, "Assistant: <think>\n")
    tagged, tagged_primed = _reasoning_segmenter(
        {"reasoning_delimiters": ("<think>", "</think>")},
        "Assistant: <think>\n",
    )

    assert unknown.feed(0, "analysis")[0].classification.value == "unknown"
    assert primed is False
    assert tagged.inside_reasoning is True
    assert tagged_primed is True


@pytest.mark.parametrize(
    ("prompt_primed", "pieces", "expected_segments"),
    [
        (
            False,
            ("<thi", "nk>step", "</th", "ink>answer"),
            ["reasoning", "reasoning", "reasoning", "unknown"],
        ),
        (
            True,
            ("step", "</th", "ink>", "answer"),
            ["reasoning", "reasoning", "reasoning", "answer"],
        ),
    ],
)
def test_generation_stream_segments_split_delimiters_and_prompt_priming(
    monkeypatch: pytest.MonkeyPatch,
    prompt_primed: bool,
    pieces: tuple[str, ...],
    expected_segments: list[str],
) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    runtime, output = configured_reasoning_runtime(pieces, prompt_primed=prompt_primed)
    command = generation_command()
    command["sampling"]["max_output_tokens"] = len(pieces)

    runtime._generate(command)

    events = [item for item in output.items if item["kind"] == "run_event"]
    stage = next(item for item in events if item["event_type"] == "stage")
    tokens = [item["payload"] for item in events if item["event_type"] == "token"]
    completed = next(item["payload"] for item in events if item["event_type"] == "completed")

    assert stage["payload"]["reasoning_primed"] is prompt_primed
    assert [item["token_index"] for item in tokens] == list(range(len(pieces)))
    assert [item["display_text"] for item in tokens] == list(pieces)
    assert [item["segment"] for item in tokens] == expected_segments
    assert completed["text"] == "".join(pieces)
    assert sum(
        int(metrics["token_count"]) for metrics in completed["segment_metrics"].values()
    ) == len(pieces)
    if not prompt_primed:
        assert tokens[-1]["reasoning_slices"] == [
            {"start": 0, "end": 4, "classification": "reasoning", "delimiter": True},
            {"start": 4, "end": 10, "classification": "answer", "delimiter": False},
        ]


def test_reasoning_that_fills_the_output_limit_gets_a_bounded_answer_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    runtime, output = configured_reasoning_runtime(
        ("deliberation", "</think>", "Final answer"),
        prompt_primed=True,
        token_ids=(2, 3, 4, 99),
    )
    command = generation_command()
    command["reasoning"] = True
    command["sampling"]["max_output_tokens"] = 2

    runtime._generate(command)

    events = [item for item in output.items if item["kind"] == "run_event"]
    stage = next(item["payload"] for item in events if item["event_type"] == "stage")
    warnings = [item["payload"] for item in events if item["event_type"] == "warning"]
    tokens = [item["payload"] for item in events if item["event_type"] == "token"]
    completed = next(item["payload"] for item in events if item["event_type"] == "completed")

    assert stage["configured_max_output_tokens"] == 2
    assert stage["reasoning_answer_allowance"] == 2
    assert stage["generation_token_limit"] == 4
    assert [item["display_text"] for item in tokens] == [
        "deliberation",
        "</think>",
        "Final answer",
        "</s>",
    ]
    assert [item["segment"] for item in tokens] == [
        "reasoning",
        "reasoning",
        "answer",
        "answer",
    ]
    assert [item["code"] for item in warnings] == ["reasoning_answer_allowance_activated"]
    assert completed["finish_reason"] == "eos"
    assert completed["configured_max_output_tokens"] == 2
    assert completed["reasoning_answer_allowance"] == 2
    assert completed["reasoning_answer_allowance_used"] == 2
    assert completed["text"] == "deliberation</think>Final answer</s>"


def test_unclosed_reasoning_at_the_output_limit_can_close_and_answer_in_allowance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    runtime, output = configured_reasoning_runtime(
        ("first thought", "second thought", "</think>", "Final answer"),
        prompt_primed=True,
    )
    command = generation_command()
    command["reasoning"] = True
    command["sampling"]["max_output_tokens"] = 2

    runtime._generate(command)

    events = [item for item in output.items if item["kind"] == "run_event"]
    warnings = [item["payload"] for item in events if item["event_type"] == "warning"]
    tokens = [item["payload"] for item in events if item["event_type"] == "token"]
    completed = next(item["payload"] for item in events if item["event_type"] == "completed")

    assert [item["display_text"] for item in tokens] == [
        "first thought",
        "second thought",
        "</think>",
        "Final answer",
    ]
    assert [item["segment"] for item in tokens] == [
        "reasoning",
        "reasoning",
        "reasoning",
        "answer",
    ]
    assert [item["code"] for item in warnings] == ["reasoning_answer_allowance_activated"]
    assert completed["finish_reason"] == "length"
    assert completed["reasoning_answer_allowance_used"] == 2
    assert completed["text"] == "first thoughtsecond thought</think>Final answer"


def test_reasoning_answer_allowance_is_not_used_when_answer_starts_within_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    runtime, output = configured_reasoning_runtime(
        ("deliberation", "</think>Final answer", "should not be generated"),
        prompt_primed=True,
    )
    command = generation_command()
    command["reasoning"] = True
    command["sampling"]["max_output_tokens"] = 2

    runtime._generate(command)

    events = [item for item in output.items if item["kind"] == "run_event"]
    tokens = [item["payload"] for item in events if item["event_type"] == "token"]
    warnings = [item["payload"] for item in events if item["event_type"] == "warning"]
    completed = next(item["payload"] for item in events if item["event_type"] == "completed")

    assert [item["display_text"] for item in tokens] == [
        "deliberation",
        "</think>Final answer",
    ]
    assert warnings == []
    assert completed["finish_reason"] == "length"
    assert completed["reasoning_answer_allowance_used"] == 0


def test_reasoning_answer_allowance_respects_an_explicit_disabled_choice(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    runtime, output = configured_reasoning_runtime(
        ("deliberation", "</think>", "must not be generated"),
        prompt_primed=True,
    )
    command = generation_command()
    command["reasoning"] = False
    command["sampling"]["max_output_tokens"] = 2

    runtime._generate(command)

    events = [item for item in output.items if item["kind"] == "run_event"]
    stage = next(item["payload"] for item in events if item["event_type"] == "stage")
    tokens = [item["payload"] for item in events if item["event_type"] == "token"]
    completed = next(item["payload"] for item in events if item["event_type"] == "completed")

    assert stage["reasoning_answer_allowance"] == 0
    assert [item["display_text"] for item in tokens] == ["deliberation", "</think>"]
    assert completed["reasoning_answer_allowance_used"] == 0


def test_decoder_start_token_resolution_and_missing_metadata_diagnostic() -> None:
    explicit = SimpleNamespace(
        generation_config=SimpleNamespace(decoder_start_token_id=0, bos_token_id=7),
        config=SimpleNamespace(decoder_start_token_id=None, bos_token_id=8),
    )
    assert _decoder_start_token(explicit, SimpleNamespace(bos_token_id=9)) == (
        0,
        "generation_config.decoder_start_token_id",
        None,
    )
    fallback = SimpleNamespace(
        generation_config=SimpleNamespace(decoder_start_token_id=None, bos_token_id=None),
        config=SimpleNamespace(decoder_start_token_id=None, bos_token_id=8),
    )
    token_id, source, warning = _decoder_start_token(fallback, SimpleNamespace(bos_token_id=9))
    assert (token_id, source) == (8, "config.bos_token_id")
    assert warning and "decoder_start_token_id" in warning

    missing = SimpleNamespace(generation_config=SimpleNamespace(), config=SimpleNamespace())
    with pytest.raises(ValueError) as caught:
        _decoder_start_token(missing, SimpleNamespace(bos_token_id=None))
    error = _safe_error(caught.value)
    assert error["code"] == "missing_decoder_start_token_id"
    assert "bundled" in error["hint"]


def test_encoder_decoder_generation_encodes_once_and_reuses_cached_decoder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    runtime, model, output = configured_encoder_decoder_runtime()

    runtime._generate(generation_command())

    assert len(model.encoder.calls) == 1
    assert len(model.decoder_calls) == 2
    first, second = model.decoder_calls
    assert first["decoder_input_ids"].tolist() == [[1]]
    assert second["decoder_input_ids"].tolist() == [[2]]
    assert first["past_key_values"] is None
    assert second["past_key_values"] == "cache-1"
    assert first["encoder_outputs"] is second["encoder_outputs"] is model.encoder.output
    assert first["attention_mask"].shape == second["attention_mask"].shape == (1, 2)
    assert first["decoder_attention_mask"].shape == (1, 1)
    assert second["decoder_attention_mask"].shape == (1, 2)

    events = [item for item in output.items if item["kind"] == "run_event"]
    assert [item["event_type"] for item in events] == [
        "warning",
        "stage",
        "metric",
        "token",
        "token",
        "completed",
    ]
    stage = next(item for item in events if item["event_type"] == "stage")
    assert stage["payload"]["architecture_mode"] == "encoder_decoder"
    assert stage["payload"]["prompt_renderer"] == "plain_text_fallback"
    tokens = [item["payload"] for item in events if item["event_type"] == "token"]
    assert [item["token_id"] for item in tokens] == [2, 3]
    assert tokens[0]["includes_prefill"] is True
    assert tokens[1]["includes_prefill"] is False
    completed = events[-1]["payload"]
    assert completed["finish_reason"] == "eos"
    assert completed["text"] == "hello</s>"
    assert fake_torch.deterministic_calls == [(False, False)]
    assert fake_torch.backends.cudnn.benchmark is False


def test_generation_forces_a_selected_prefix_then_continues_from_its_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    runtime, model, output = configured_encoder_decoder_runtime()
    command = generation_command()
    command["forced_prefix_token_ids"] = [4]

    runtime._generate(command)

    events = [item for item in output.items if item["kind"] == "run_event"]
    tokens = [item["payload"] for item in events if item["event_type"] == "token"]
    completed = next(item["payload"] for item in events if item["event_type"] == "completed")
    assert [item["token_id"] for item in tokens] == [4, 3]
    assert tokens[0]["filters"] == ["greedy_argmax", "forced_prefix"]
    assert tokens[0]["sample_probability"] == 0.0
    assert model.decoder_calls[1]["decoder_input_ids"].tolist() == [[4]]
    assert completed["text"] == "alternate</s>"


def test_causal_generation_advances_cache_with_substituted_prefix_before_suffix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    runtime, model, output = configured_causal_runtime()
    command = generation_command()
    command["forced_prefix_token_ids"] = [4]

    runtime._generate(command)

    events = [item for item in output.items if item["kind"] == "run_event"]
    tokens = [item["payload"] for item in events if item["event_type"] == "token"]
    completed = next(item["payload"] for item in events if item["event_type"] == "completed")
    assert [item["token_id"] for item in tokens] == [4, 2, 3]
    assert tokens[0]["filters"] == ["greedy_argmax", "forced_prefix"]
    assert model.calls[1]["input_ids"].tolist() == [[4]]
    assert model.calls[1]["past_key_values"] == "cache-after-prompt"
    assert model.calls[2]["input_ids"].tolist() == [[2]]
    assert model.calls[2]["past_key_values"] == "cache-after-substitution-4"
    assert completed["finish_reason"] == "eos"
    assert completed["text"] == "alternatehello</s>"


def test_full_instrumentation_captures_qwen_causal_attention_and_restores_kernel() -> None:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    class TinyTokenizer:
        chat_template = None
        eos_token_id = None
        bos_token_id = None

        @staticmethod
        def __call__(_text: str, **_kwargs: Any) -> dict[str, Any]:
            return {
                "input_ids": torch.tensor([[5, 6, 7]], dtype=torch.long),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            }

        @staticmethod
        def convert_ids_to_tokens(token_id: int) -> str:
            return f"token-{token_id}"

        @staticmethod
        def decode(token_ids: list[int], **_kwargs: Any) -> str:
            return "".join(f"<{token_id}>" for token_id in token_ids)

    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            bos_token_id=None,
            eos_token_id=None,
        )
    ).eval()
    model.set_attn_implementation("sdpa")
    attention_modules = [
        module for name, module in model.named_modules() if name.rsplit(".", 1)[-1] == "self_attn"
    ]
    initial_hook_counts = [len(module._forward_hooks) for module in attention_modules]
    forward_implementations: list[tuple[int, str]] = []

    def record_attention_implementation(_module: Any, _args: Any, kwargs: dict[str, Any]) -> None:
        forward_implementations.append(
            (
                int(kwargs["input_ids"].shape[-1]),
                str(model.config._attn_implementation),
            )
        )

    implementation_hook = model.model.register_forward_pre_hook(
        record_attention_implementation,
        with_kwargs=True,
    )
    output = FakeQueue()
    runtime = WorkerRuntime(cast(Any, FakeQueue()), cast(Any, output), FakeCancelEvent())
    runtime.model = model
    runtime.tokenizer = TinyTokenizer()
    runtime.model_info = {
        "id": "tiny-qwen",
        "display_name": "Tiny Qwen Fixture",
        "task": "text_generation",
        "effective_context_limit": 32,
    }
    runtime.loaded_attention_implementation = "sdpa"
    command = generation_command()
    command["instrumentation"] = "full"
    command["sampling"]["max_output_tokens"] = 2

    try:
        runtime._generate(command)
    finally:
        implementation_hook.remove()

    events = [item for item in output.items if item["kind"] == "run_event"]
    tokens = [item["payload"] for item in events if item["event_type"] == "token"]
    completed = next(item["payload"] for item in events if item["event_type"] == "completed")
    assert len(tokens) == 2
    first = tokens[0]["attention_attribution"]
    second = tokens[1]["attention_attribution"]
    assert first["total_source_count"] == 3
    assert len(first["context_tokens"]) == 3
    assert first["captured_layers"] == [0, 1]
    assert first["captured_heads"] == 4
    assert sum(item["weight"] for item in first["source_tokens"]) == pytest.approx(1.0)
    assert second["total_source_count"] == 4
    generated_source = next(
        item for item in second["source_tokens"] if item["source_kind"] == "generated"
    )
    assert generated_source["generated_token_index"] == 0
    assert generated_source["context_index"] == 3
    assert completed["attention_capture"] == {
        "requested": True,
        "method": "mean_causal_self_attention",
        "captured_token_count": 2,
        "semantics": "attention_weights_not_causal_contributions",
        "scope": "decoder_step_context_attention_independent_of_sampled_candidate",
    }
    assert forward_implementations == [(2, "sdpa"), (1, "eager"), (1, "eager")]
    assert model.config._attn_implementation == "sdpa"
    assert [len(module._forward_hooks) for module in attention_modules] == initial_hook_counts


def test_full_instrumentation_continues_when_eager_cannot_be_reenabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    class TinyTokenizer:
        chat_template = None
        eos_token_id = None
        bos_token_id = None

        @staticmethod
        def __call__(_text: str, **_kwargs: Any) -> dict[str, Any]:
            return {
                "input_ids": torch.tensor([[5, 6, 7]], dtype=torch.long),
                "attention_mask": torch.ones((1, 3), dtype=torch.long),
            }

        @staticmethod
        def convert_ids_to_tokens(token_id: int) -> str:
            return f"token-{token_id}"

        @staticmethod
        def decode(token_ids: list[int], **_kwargs: Any) -> str:
            return "".join(f"<{token_id}>" for token_id in token_ids)

    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=32,
            bos_token_id=None,
            eos_token_id=None,
        )
    ).eval()
    model.set_attn_implementation("sdpa")
    output = FakeQueue()
    runtime = WorkerRuntime(cast(Any, FakeQueue()), cast(Any, output), FakeCancelEvent())
    runtime.model = model
    runtime.tokenizer = TinyTokenizer()
    runtime.model_info = {
        "id": "tiny-qwen-fallback",
        "display_name": "Tiny Qwen Fallback Fixture",
        "task": "text_generation",
        "effective_context_limit": 32,
    }
    runtime.loaded_attention_implementation = "sdpa"
    select_implementation = runtime._select_attention_implementation
    eager_requests = 0

    def fail_second_eager_selection(implementation: str) -> bool:
        nonlocal eager_requests
        if implementation == "eager":
            eager_requests += 1
            if eager_requests == 2:
                return False
        return select_implementation(implementation)

    monkeypatch.setattr(
        runtime,
        "_select_attention_implementation",
        fail_second_eager_selection,
    )
    command = generation_command()
    command["instrumentation"] = "full"
    command["sampling"]["max_output_tokens"] = 1

    runtime._generate(command)

    events = [item for item in output.items if item["kind"] == "run_event"]
    warnings = [item["payload"] for item in events if item["event_type"] == "warning"]
    tokens = [item["payload"] for item in events if item["event_type"] == "token"]
    completed = next(item["payload"] for item in events if item["event_type"] == "completed")
    assert eager_requests == 2
    assert any(item["code"] == "attention_capture_unavailable" for item in warnings)
    assert len(tokens) == 1
    assert tokens[0]["attention_attribution"] is None
    assert completed["attention_capture"]["requested"] is True
    assert completed["attention_capture"]["captured_token_count"] == 0
    assert completed["attention_capture"]["method"] is None
    assert model.config._attn_implementation == "sdpa"


def test_oversized_full_prompt_is_rejected_before_catalog_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class OversizedTokenizer(FakeTokenizer):
        decoded_token_count = 0

        def __call__(self, text: str, **_kwargs: Any) -> dict[str, FakeTensor]:
            assert text == "summarize this"
            return {
                "input_ids": FakeTensor([[5] * 33]),
                "attention_mask": FakeTensor([[1] * 33]),
            }

        def convert_ids_to_tokens(self, token_id: int) -> str:
            self.decoded_token_count += 1
            return str(token_id)

    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    runtime, _model, _output = configured_causal_runtime()
    tokenizer = OversizedTokenizer()
    runtime.tokenizer = tokenizer
    runtime.model_info = {
        **cast(dict[str, Any], runtime.model_info),
        "effective_context_limit": 8,
    }
    monkeypatch.setattr(runtime, "_select_attention_implementation", lambda _value: True)
    command = generation_command()
    command["instrumentation"] = "full"

    with pytest.raises(ValueError, match="rendered prompt exceeds"):
        runtime._generate(command)

    assert tokenizer.decoded_token_count == 0


def test_generation_rejects_a_forced_prefix_longer_than_the_recorded_output_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    runtime, _model, _output = configured_encoder_decoder_runtime()
    command = generation_command()
    command["sampling"]["max_output_tokens"] = 1
    command["forced_prefix_token_ids"] = [2, 4]

    with pytest.raises(ValueError, match="configured output limit"):
        runtime._generate(command)


def test_deterministic_mode_is_set_explicitly_for_each_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_torch = FakeTorch()
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    runtime, _model, _output = configured_encoder_decoder_runtime()
    deterministic = generation_command()
    deterministic["deterministic_reference_mode"] = True

    runtime._generate(deterministic)
    runtime._generate(generation_command())

    assert fake_torch.deterministic_calls == [(True, True), (False, False)]


def test_qwen3_vl_matryoshka_dimension_floor_is_model_specific() -> None:
    assert _embedding_dimension_bounds("qwen3_vl", 2048) == (64, 2048)
    assert _embedding_dimension_bounds("bert", 768) is None


def test_qwen3_vl_embedding_rejects_dimensions_below_documented_floor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    runtime = WorkerRuntime(cast(Any, FakeQueue()), cast(Any, FakeQueue()), FakeCancelEvent())
    runtime.sentence_model = FakeSentenceModel()
    runtime.model_info = {"id": "qwen-embed", "model_type": "qwen3_vl"}

    with pytest.raises(ValueError, match="between 64 and 128"):
        runtime._embed(
            {
                "inputs": [{"input_id": "one", "modality": "text", "text": "hello"}],
                "dimensions": 63,
                "normalize": True,
                "batch_size": 1,
            }
        )


def test_generic_embedding_rejects_unreviewed_dimension_truncation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    runtime = WorkerRuntime(cast(Any, FakeQueue()), cast(Any, FakeQueue()), FakeCancelEvent())
    runtime.sentence_model = FakeSentenceModel()
    runtime.model_info = {
        "id": "generic-embed",
        "model_type": "bert",
        "embedding_pooling": "mean",
        "joint_embedding_space": False,
    }

    with pytest.raises(ValueError, match="custom embedding dimensions are not supported"):
        runtime._embed(
            {
                "inputs": [{"input_id": "one", "modality": "text", "text": "hello"}],
                "dimensions": 64,
                "normalize": True,
                "batch_size": 1,
            }
        )


def test_encoder_decoder_cancellation_preserves_consistent_terminal_event(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "torch", FakeTorch())
    runtime, model, output = configured_encoder_decoder_runtime(cancelled=True)

    runtime._generate(generation_command())

    assert len(model.encoder.calls) == 1
    assert len(model.decoder_calls) == 1
    events = [item for item in output.items if item["kind"] == "run_event"]
    assert not any(item["event_type"] == "token" for item in events)
    assert events[-1]["event_type"] == "cancelled"
    assert events[-1]["payload"]["finish_reason"] == "cancelled"
    assert events[-1]["payload"]["generated_token_count"] == 0
