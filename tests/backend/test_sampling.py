from __future__ import annotations

import math

import numpy as np
import pytest

from local_ai_doctor.errors import InvalidLogitsError, InvalidSamplingSettingsError
from local_ai_doctor.sampling import (
    MetricsAccumulator,
    SamplingSettings,
    apply_sampling_pipeline,
    compute_distribution_metrics,
    compute_token_metrics,
    effective_seed,
    sample_token,
    score_prompt_perplexity,
)


def test_full_vocabulary_fp32_probability_rank_entropy_and_alternatives() -> None:
    logits = np.log(np.array([1.0, 2.0, 4.0], dtype=np.float32))
    metrics = compute_distribution_metrics(logits, 1, alternatives=2)
    assert metrics.selected_probability == pytest.approx(2 / 7, rel=2e-6)
    assert metrics.selected_log_probability == pytest.approx(math.log(2 / 7), rel=2e-6)
    assert metrics.selected_rank == 2
    expected_entropy = -sum(
        probability * math.log(probability) for probability in (1 / 7, 2 / 7, 4 / 7)
    )
    assert metrics.entropy_nats == pytest.approx(expected_entropy, rel=2e-6)
    assert [item.token_id for item in metrics.alternatives] == [2, 1]
    assert sum(item.probability for item in metrics.alternatives) < 1.0
    assert metrics.exact_full_vocabulary


def test_rank_has_deterministic_ascending_token_id_tie_rule() -> None:
    assert compute_distribution_metrics([1.0, 1.0, 1.0], 0).selected_rank == 1
    assert compute_distribution_metrics([1.0, 1.0, 1.0], 2).selected_rank == 3


def test_raw_and_sampler_distributions_are_distinct() -> None:
    metrics = compute_token_metrics(
        [1.0, 2.0, 3.0, 4.0],
        [-np.inf, -np.inf, 3.0, 4.0],
        3,
        alternatives=4,
    )
    assert metrics.raw.selected_probability < metrics.sampling.selected_probability
    survivors = {
        item.token_id: item.survived_sampler_filtering for item in metrics.raw.alternatives
    }
    assert survivors == {3: True, 2: True, 1: False, 0: False}
    assert all(item.survived_sampler_filtering for item in metrics.sampling.alternatives)


def test_reference_pipeline_records_exact_operation_order_and_does_not_mutate_input() -> None:
    source = np.array([-1.0, 0.0, 1.0, 2.0], dtype=np.float32)
    original = source.copy()
    result = apply_sampling_pipeline(
        source,
        SamplingSettings(
            temperature=0.5,
            top_k=3,
            top_p=0.9,
            min_p=0.1,
            repetition_penalty=1.1,
            frequency_penalty=0.2,
            presence_penalty=0.1,
        ),
        token_counts={3: 2},
    )
    assert np.array_equal(source, original)
    assert [item.name for item in result.operations] == [
        "repetition_penalty",
        "frequency_penalty",
        "presence_penalty",
        "constraints",
        "temperature",
        "top_k",
        "top_p",
        "min_p",
    ]
    assert np.isfinite(result.logits).any()


def test_greedy_tie_is_deterministic_and_seed_zero_is_valid() -> None:
    result = apply_sampling_pipeline([1.0, 2.0, 2.0], SamplingSettings(temperature=0.0))
    assert np.flatnonzero(np.isfinite(result.logits)).tolist() == [1]
    token, algorithm = sample_token(result.logits, 0)
    assert token == 1
    assert "seed did not affect" in algorithm
    assert effective_seed(0) == 0


def test_metrics_accumulator_reports_raw_and_post_sampler_perplexity_separately() -> None:
    base = compute_token_metrics([0.0, 1.0], [0.0, 2.0], 1)
    accumulator = MetricsAccumulator()
    first = accumulator.observe(base, segment="reasoning")
    second = accumulator.observe(base, segment="answer")
    assert second.cumulative_raw_log_probability == pytest.approx(
        2 * base.raw.selected_log_probability
    )
    summary = accumulator.summary()
    assert summary["conditional_generated_response_perplexity"] == pytest.approx(
        math.exp(-base.raw.selected_log_probability)
    )
    assert (
        summary["post_sampler_perplexity"] != summary["conditional_generated_response_perplexity"]
    )
    assert first.running_response_perplexity == pytest.approx(second.running_response_perplexity)


def test_prompt_perplexity_uses_shifted_logits_and_documents_exclusions() -> None:
    logits = np.array(
        [
            [0.0, 3.0, 0.0],  # predicts token 1 at position 1
            [0.0, 0.0, 3.0],  # position 2 is a masked multimodal token
            [3.0, 0.0, 0.0],  # predicts token 0 at position 3
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    result = score_prompt_perplexity(
        logits,
        [2, 1, 2, 0],
        attention_mask=[True, True, True, True],
        text_position_mask=[True, True, False, True],
    )
    expected_log_probability = 3.0 - math.log(math.exp(3.0) + 2)
    assert result.included_token_count == 2
    assert result.excluded_first_token == 1
    assert result.excluded_non_text_tokens == 1
    assert result.mean_log_probability == pytest.approx(expected_log_probability, rel=2e-6)
    assert result.token_log_probabilities[0] is None
    assert result.token_log_probabilities[2] is None


def test_invalid_distribution_and_constraint_errors_are_structured() -> None:
    with pytest.raises(InvalidLogitsError) as logits_error:
        compute_distribution_metrics([float("nan"), 0.0], 1)
    assert logits_error.value.to_dict()["code"] == "invalid_logits"
    with pytest.raises(InvalidSamplingSettingsError):
        apply_sampling_pipeline([0.0, 1.0], SamplingSettings(), allowed_token_ids=[])
