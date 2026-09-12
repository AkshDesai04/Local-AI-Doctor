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
        return {2: "hello", 3: "</s>"}[token_id]

    @staticmethod
    def decode(token_ids: list[int], **_kwargs: Any) -> str:
        return "".join({2: "hello", 3: "</s>"}[token_id] for token_id in token_ids)


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
            logits = FakeTensor([[[0.0, 0.0, 9.0, 1.0]]])
        else:
            logits = FakeTensor([[[0.0, 0.0, 1.0, 9.0]]])
        return SimpleNamespace(logits=logits, past_key_values=f"cache-{len(self.decoder_calls)}")


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
        return self._pieces[token_id - 2]

    def decode(self, token_ids: list[int], **_kwargs: Any) -> str:
        return "".join(self._pieces[token_id - 2] for token_id in token_ids)


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


def configured_reasoning_runtime(
    pieces: tuple[str, ...], *, prompt_primed: bool
) -> tuple[WorkerRuntime, FakeQueue]:
    output = FakeQueue()
    runtime = WorkerRuntime(cast(Any, FakeQueue()), cast(Any, output), FakeCancelEvent())
    runtime.model = FakeReasoningEncoderDecoderModel(tuple(range(2, len(pieces) + 2)))
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


def test_prompt_rendering_falls_back_without_chat_template() -> None:
    rendered, renderer = _render_messages(
        FakeTokenizer(), [{"role": "user", "content": "summarize this"}]
    )
    assert rendered == "summarize this"
    assert renderer == "plain_text_fallback"


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
