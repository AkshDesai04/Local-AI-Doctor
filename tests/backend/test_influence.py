"""Token influence math on a tiny random Qwen2 (CPU) plus the grouping helpers."""

from __future__ import annotations

import math
from typing import Any, cast

import pytest

from local_ai_doctor.errors import worker_failure_response
from local_ai_doctor.workers.runtime import (
    WorkerReportedError,
    WorkerRuntime,
    _influence_groups,
    _mean_causal_self_attention,
    _media_labels,
    _ranked_influence,
)

PROMPT = [5, 6, 7, 8]
GENERATED = [9, 10]
TARGET = 11


class _Queue:
    def __init__(self) -> None:
        self.items: list[dict[str, Any]] = []

    def put(self, item: dict[str, Any]) -> None:
        self.items.append(item)


class _NeverCancelled:
    @staticmethod
    def is_set() -> bool:
        return False


class TinyTokenizer:
    chat_template = None
    eos_token_id = None
    bos_token_id = None
    all_special_ids = (5,)

    def __init__(self, prompt_ids: list[int]) -> None:
        self.prompt_ids = prompt_ids

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        import torch

        ids = self.prompt_ids
        if kwargs.get("return_tensors") is None:
            return {"input_ids": ids}
        return {
            "input_ids": torch.tensor([ids], dtype=torch.long),
            "attention_mask": torch.ones((1, len(ids)), dtype=torch.long),
        }

    @staticmethod
    def convert_ids_to_tokens(token_id: int) -> str:
        return f"token-{token_id}"

    @staticmethod
    def decode(token_ids: list[int], **_kwargs: Any) -> str:
        return "".join(f"<{token_id}>" for token_id in token_ids)


def _tiny_qwen(layers: int = 2) -> Any:
    import torch
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(0)
    model = Qwen2ForCausalLM(
        Qwen2Config(
            vocab_size=32,
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=layers,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
            bos_token_id=None,
            eos_token_id=None,
        )
    ).eval()
    model.set_attn_implementation("sdpa")
    return model


def _runtime(prompt: list[int] = PROMPT, tokenizer: Any = None) -> WorkerRuntime:
    runtime = WorkerRuntime(cast(Any, _Queue()), cast(Any, _Queue()), _NeverCancelled())
    runtime.model = _tiny_qwen()
    runtime.tokenizer = tokenizer or TinyTokenizer(prompt)
    runtime.model_info = {
        "id": "tiny-qwen",
        "display_name": "Tiny Qwen",
        "task": "text_generation",
        "effective_context_limit": 64,
    }
    runtime.loaded_attention_implementation = "sdpa"
    return runtime


def _command(method: str, **overrides: Any) -> dict[str, Any]:
    return {
        "op": "analyze_influence",
        "method": method,
        "layers": "mean" if method == "attention" else None,
        "alternative_token_id": None,
        "source_limit": 128,
        "rendered_prompt": "rendered",
        "prompt_renderer": "chat_template",
        "media": [],
        "expected_prompt_token_count": len(PROMPT),
        "generated_token_ids": GENERATED,
        "target_token_id": TARGET,
        "max_gradient_tokens": 2048,
        **overrides,
    }


def _weights(sources: list[dict[str, Any]]) -> list[float]:
    return [item["weight"] for item in sorted(sources, key=lambda item: item["context_index"])]


def test_attention_rows_per_layer_and_mean_each_sum_to_one() -> None:
    import torch

    runtime = _runtime()
    result = runtime._analyze_influence(_command("attention", layers="all"))

    context = len(PROMPT) + len(GENERATED)
    assert result["context_token_count"] == context
    assert result["captured_layers"] == [0, 1]
    assert result["heads_per_layer"] == 4
    assert result["normalization"] == "sum_to_one"
    assert math.fsum(_weights(result["sources"])) == pytest.approx(1.0)
    assert result["retained_weight"] == pytest.approx(1.0)
    assert result["omitted_weight"] == pytest.approx(0.0, abs=1e-9)
    layer_rows = torch.tensor([_weights(layer["sources"]) for layer in result["layers"]])
    assert layer_rows.shape == (2, context)
    assert torch.allclose(layer_rows.sum(dim=1), torch.ones(2, dtype=layer_rows.dtype))
    # With equal heads per layer the mean row is the plain mean of the layer rows.
    assert torch.allclose(layer_rows.mean(dim=0), torch.tensor(_weights(result["sources"])))
    assert "allocation, not causal attribution" in result["semantics"].lower()
    # The kernel the resident was loaded with is restored.
    assert runtime.model.config._attn_implementation == "sdpa"


def test_attention_mean_matches_the_live_capture_for_the_same_token() -> None:
    runtime = _runtime()
    output = cast(Any, runtime.output)
    runtime._generate(
        {
            "run_id": "live",
            "request_id": "request-live",
            "messages": [{"role": "user", "content": "x"}],
            "sampling": {
                "max_output_tokens": 3,
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
            "effective_seed": 0,
            "instrumentation": "full",
            "max_prompt_tokens": 64,
            "reserved_output_tokens": 0,
        }
    )
    tokens = [
        item["payload"]
        for item in output.items
        if item.get("kind") == "run_event" and item["event_type"] == "token"
    ]
    live = tokens[2]["attention_attribution"]["source_tokens"]

    result = runtime._analyze_influence(
        _command(
            "attention",
            generated_token_ids=[tokens[0]["token_id"], tokens[1]["token_id"]],
            target_token_id=tokens[2]["token_id"],
        )
    )

    assert [item["context_index"] for item in result["sources"]] == [
        item["context_index"] for item in live
    ]
    assert _weights(result["sources"]) == pytest.approx(_weights(live), abs=1e-5)


def test_a_selected_layer_reports_that_layer_only() -> None:
    runtime = _runtime()
    every = runtime._analyze_influence(_command("attention", layers="all"))
    one = runtime._analyze_influence(_command("attention", layers=[1]))

    assert [layer["layer"] for layer in one["layers"]] == [1]
    assert _weights(one["sources"]) == pytest.approx(_weights(every["layers"][1]["sources"]))


def test_gradient_x_input_is_finite_non_negative_and_matches_the_embedding_reference() -> None:
    import torch

    runtime = _runtime()
    result = runtime._analyze_influence(_command("gradient_x_input"))

    weights = _weights(result["sources"])
    assert all(math.isfinite(value) and value >= 0.0 for value in weights)
    assert math.fsum(weights) == pytest.approx(1.0)
    assert result["objective"] == "log_probability"
    assert result["captured_layers"] == []
    assert "not causal attribution" in result["semantics"]

    # Textbook gradient x input on the embeddings Qwen2 feeds unchanged into layer 0.
    model = runtime.model
    ids = torch.tensor([PROMPT + GENERATED])
    embeddings = model.model.embed_tokens(ids).detach().requires_grad_(True)
    logits = model(inputs_embeds=embeddings).logits[0, -1].float()
    objective = torch.log_softmax(logits, dim=-1)[TARGET]
    objective.backward()
    reference = (embeddings.grad * embeddings).sum(dim=-1)[0].abs()
    reference = reference / reference.sum()
    assert weights == pytest.approx(reference.tolist(), abs=1e-5)
    assert result["objective_value"] == pytest.approx(float(objective.detach()), abs=1e-5)
    # Weights stay frozen, and nothing accumulates gradients across analyses.
    assert all(
        not parameter.requires_grad and parameter.grad is None for parameter in model.parameters()
    )


def test_an_alternative_target_changes_the_gradient_weights() -> None:
    runtime = _runtime()
    chosen = runtime._analyze_influence(_command("gradient_x_input"))
    contrast = runtime._analyze_influence(_command("gradient_x_input", alternative_token_id=3))

    assert contrast["objective"] == "logit_difference"
    assert contrast["alternative_piece"] == "token-3"
    assert math.fsum(_weights(contrast["sources"])) == pytest.approx(1.0)
    differences = [
        abs(left - right)
        for left, right in zip(
            _weights(chosen["sources"]), _weights(contrast["sources"]), strict=True
        )
    ]
    assert max(differences) > 1e-4


def test_sources_carry_kind_generated_index_and_special_flags() -> None:
    result = _runtime()._analyze_influence(_command("gradient_x_input"))
    sources = sorted(result["sources"], key=lambda item: item["context_index"])

    assert [item["source_kind"] for item in sources] == ["prompt"] * 4 + ["generated"] * 2
    assert [item["generated_token_index"] for item in sources] == [None] * 4 + [0, 1]
    assert [item["is_special"] for item in sources] == [True] + [False] * 5
    assert sources[0]["piece"] == "token-5" and sources[0]["token_count"] == 1
    assert all(item["span"] is None for item in sources)


def test_source_limit_keeps_the_heaviest_sources_and_reports_omitted_mass() -> None:
    result = _runtime()._analyze_influence(_command("attention", source_limit=2))

    assert len(result["sources"]) == 2
    retained = math.fsum(item["weight"] for item in result["sources"])
    assert result["retained_weight"] == pytest.approx(retained)
    assert result["omitted_weight"] == pytest.approx(1.0 - retained)


def test_image_placeholder_runs_group_into_one_source_before_the_top_k_cut() -> None:
    ids = [1, 7, 7, 7, 2, 7, 7, 3]
    labels = _media_labels(ids, {7: "image"}, {"image": [3, 2]})
    groups = _influence_groups(ids, 7, labels)

    assert [(group.start, group.end, group.kind, group.media_index) for group in groups] == [
        (0, 1, "prompt", None),
        (1, 4, "image", 0),
        (4, 5, "prompt", None),
        (5, 7, "image", 1),
        (7, 8, "generated", None),
    ]
    # Each placeholder is light, but the first image's three positions sum to 0.3.
    weights = [0.1, 0.1, 0.1, 0.1, 0.2, 0.05, 0.05, 0.3]
    ranked, retained = _ranked_influence(weights, groups, 2)
    assert [(group.kind, group.start) for group, _weight in ranked] == [
        ("image", 1),
        ("generated", 7),
    ]
    assert [weight for _group, weight in ranked] == pytest.approx([0.3, 0.3])
    assert retained == pytest.approx(0.6)


def test_per_layer_rows_left_pad_sliding_window_layers() -> None:
    import torch

    sources = [{"context_index": index} for index in range(4)]
    attentions = (
        torch.tensor([[[[0.1, 0.2, 0.3, 0.4]], [[0.4, 0.3, 0.2, 0.1]]]]),
        torch.tensor([[[[0.5, 0.5]], [[0.25, 0.75]]]]),
    )
    aggregate, error = _mean_causal_self_attention(torch, attentions, sources, per_layer=True)

    assert error is None and aggregate is not None
    assert aggregate["layer_rows"][0] == pytest.approx([0.25, 0.25, 0.25, 0.25])
    assert aggregate["layer_rows"][1] == pytest.approx([0.0, 0.0, 0.375, 0.625])
    assert aggregate["mean_row"] == pytest.approx([0.125, 0.125, 0.3125, 0.4375])


@pytest.mark.parametrize(
    ("overrides", "placement", "code", "status"),
    [
        (
            {"max_gradient_tokens": 16, "generated_token_ids": [9] * 20},
            "cpu",
            "influence_sequence_too_long",
            409,
        ),
        ({}, "offload", "influence_offload_unsupported", 409),
        ({"expected_prompt_token_count": 5}, "cpu", "influence_prompt_mismatch", 409),
        ({"alternative_token_id": 999}, "cpu", "invalid_request", 422),
    ],
)
def test_gradient_refusals_are_structured_conflicts(
    overrides: dict[str, Any], placement: str, code: str, status: int
) -> None:
    runtime = _runtime()
    assert runtime._active is not None
    runtime._active.placement = placement
    with pytest.raises(WorkerReportedError) as caught:
        runtime._analyze_influence(_command("gradient_x_input", **overrides))

    assert caught.value.error["code"] == code
    http_status, payload = worker_failure_response(caught.value.error)
    assert http_status == status
    assert payload["code"] == code
    if code == "influence_prompt_mismatch":
        assert payload["details"] == {"expected_prompt_tokens": 5, "actual_prompt_tokens": 4}
    if code == "influence_sequence_too_long":
        assert payload["details"] == {"sequence_tokens": 24, "max_sequence_tokens": 16}


def test_an_out_of_range_attention_layer_is_an_invalid_request() -> None:
    with pytest.raises(WorkerReportedError) as caught:
        _runtime()._analyze_influence(_command("attention", layers=[0, 7]))

    status, payload = worker_failure_response(caught.value.error)
    assert status == 422
    assert payload["details"] == {"layer_count": 2}


def test_runs_without_a_recorded_renderer_recover_the_fallback_bos() -> None:
    class BosTokenizer(TinyTokenizer):
        bos_token_id = 1

        def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
            if text == "" and kwargs.get("add_special_tokens"):
                return {"input_ids": [1]}
            return super().__call__(text, **kwargs)

    runtime = _runtime(tokenizer=BosTokenizer(PROMPT))
    result = runtime._analyze_influence(
        _command(
            "attention",
            prompt_renderer=None,
            expected_prompt_token_count=len(PROMPT) + 1,
        )
    )

    assert result["prompt_token_count"] == len(PROMPT) + 1
    assert result["sources"][0]["token_id"] == 1


def test_the_worker_operation_replies_and_reports_failures_safely() -> None:
    runtime = _runtime()
    output = cast(Any, runtime.output)

    runtime._handle({**_command("attention"), "request_id": "ok"})
    runtime._handle({**_command("attention", expected_prompt_token_count=9), "request_id": "bad"})

    replies = {item["request_id"]: item for item in output.items if item["kind"] == "reply"}
    assert replies["ok"]["ok"] is True
    assert replies["ok"]["payload"]["method"] == "attention"
    assert replies["bad"]["ok"] is False
    assert replies["bad"]["error"]["code"] == "influence_prompt_mismatch"
    assert runtime.model.config._attn_implementation == "sdpa"
