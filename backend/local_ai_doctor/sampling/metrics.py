"""Numerically stable fp32 sampling and observability calculations.

All probabilities and ranks are computed over the complete vocabulary.  Top-K
lists are bounded views of those distributions and are never renormalized or
described as exact distributions themselves.
"""

from __future__ import annotations

import math
import secrets
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from ..errors import InvalidLogitsError, InvalidSamplingSettingsError

UINT64_MAX = 2**64 - 1


class SamplingSettings(BaseModel):
    model_config = ConfigDict(frozen=True)

    temperature: float = Field(default=1.0, ge=0.0, le=10.0)
    top_k: int = Field(default=0, ge=0, le=1_000_000)
    top_p: float = Field(default=1.0, gt=0.0, le=1.0)
    min_p: float = Field(default=0.0, ge=0.0, le=1.0)
    repetition_penalty: float = Field(default=1.0, gt=0.0, le=10.0)
    frequency_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    presence_penalty: float = Field(default=0.0, ge=-2.0, le=2.0)
    minimum_tokens_to_keep: int = Field(default=1, ge=1, le=1_000_000)


class SamplingOperation(BaseModel):
    model_config = ConfigDict(frozen=True)

    order: int = Field(ge=0)
    name: str
    applied: bool
    parameters: dict[str, Any] = Field(default_factory=dict)
    surviving_tokens: int = Field(ge=0)


class Alternative(BaseModel):
    model_config = ConfigDict(frozen=True)

    token_id: int = Field(ge=0)
    logit: float
    log_probability: float
    probability: float = Field(ge=0.0, le=1.0)
    rank: int = Field(ge=1)
    survived_sampler_filtering: bool | None = None
    chosen: bool = False


class DistributionMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    selected_token_id: int = Field(ge=0)
    selected_logit: float
    selected_log_probability: float
    selected_probability: float = Field(ge=0.0, le=1.0)
    selected_rank: int = Field(ge=1)
    entropy_nats: float = Field(ge=0.0)
    log_normalizer: float
    vocabulary_size: int = Field(ge=1)
    surviving_tokens: int = Field(ge=1)
    exact_full_vocabulary: Literal[True] = True
    alternatives: tuple[Alternative, ...] = ()


class TokenProbabilityMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    token_id: int = Field(ge=0)
    raw: DistributionMetrics
    sampling: DistributionMetrics
    processed_logit: float
    raw_surprise_nats: float = Field(ge=0.0)
    sampling_surprise_nats: float = Field(ge=0.0)
    cumulative_raw_log_probability: float | None = None
    cumulative_sampling_log_probability: float | None = None
    running_response_perplexity: float | None = Field(default=None, ge=0.0)
    running_post_sampler_perplexity: float | None = Field(default=None, ge=0.0)


class PromptPerplexityResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    perplexity: float | None = Field(default=None, ge=0.0)
    mean_log_probability: float | None = None
    included_token_count: int = Field(ge=0)
    excluded_first_token: int = Field(default=1, ge=0, le=1)
    excluded_masked_tokens: int = Field(ge=0)
    excluded_non_text_tokens: int = Field(ge=0)
    token_log_probabilities: tuple[float | None, ...]


@dataclass(frozen=True, slots=True)
class SamplingPipelineResult:
    logits: np.ndarray
    operations: tuple[SamplingOperation, ...]
    greedy: bool


def _as_fp32_logits(values: Any, *, allow_negative_infinity: bool = True) -> np.ndarray:
    # Torch tensors, including CUDA tensors, deliberately cross one explicit
    # synchronization boundary here.  Production adapters may keep the same
    # fp32 operations on-device and serialize the bounded result instead.
    if hasattr(values, "detach") and hasattr(values, "cpu"):
        torch_module = __import__("torch")
        values = values.detach().to(dtype=torch_module.float32).cpu().numpy()
    try:
        logits = np.asarray(values, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise InvalidLogitsError(
            "logits must be convertible to a one-dimensional fp32 array"
        ) from exc
    if logits.ndim != 1 or logits.size == 0:
        raise InvalidLogitsError(
            "logits must be a non-empty one-dimensional vocabulary vector",
            details={"shape": tuple(logits.shape)},
        )
    if np.isnan(logits).any() or np.isposinf(logits).any():
        raise InvalidLogitsError("logits contain NaN or positive infinity")
    if not allow_negative_infinity and np.isneginf(logits).any():
        raise InvalidLogitsError("raw model logits contain negative infinity")
    if not np.isfinite(logits).any():
        raise InvalidLogitsError(
            "all tokens were filtered; no valid probability distribution remains"
        )
    return logits.copy()


def _stable_log_softmax(logits: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.float32]:
    finite = np.isfinite(logits)
    maximum = np.max(logits[finite]).astype(np.float32)
    shifted = np.full_like(logits, -np.inf, dtype=np.float32)
    shifted[finite] = np.subtract(logits[finite], maximum, dtype=np.float32)
    exponentials = np.zeros_like(logits, dtype=np.float32)
    exponentials[finite] = np.exp(shifted[finite], dtype=np.float32)
    total = np.sum(exponentials, dtype=np.float32)
    if not np.isfinite(total) or total <= 0:
        raise InvalidLogitsError("fp32 softmax normalization failed")
    log_total = np.log(total, dtype=np.float32)
    log_normalizer = np.add(maximum, log_total, dtype=np.float32)
    log_probabilities = np.subtract(logits, log_normalizer, dtype=np.float32)
    probabilities = np.zeros_like(logits, dtype=np.float32)
    probabilities[finite] = np.exp(log_probabilities[finite], dtype=np.float32)
    return log_probabilities, probabilities, log_normalizer


def _ordered_token_ids(logits: np.ndarray) -> np.ndarray:
    token_ids = np.arange(logits.size, dtype=np.int64)
    finite_ids = token_ids[np.isfinite(logits)]
    # Last key is primary: descending logit, then ascending token ID.  This tie
    # rule is stable across platforms and is also used for exact ranks.
    order = np.lexsort((finite_ids, -logits[finite_ids]))
    return finite_ids[order]


def _rank(logits: np.ndarray, token_id: int) -> int:
    chosen = logits[token_id]
    token_ids = np.arange(logits.size)
    return int(
        1
        + np.count_nonzero(logits > chosen)
        + np.count_nonzero((logits == chosen) & (token_ids < token_id))
    )


def compute_distribution_metrics(
    logits: Any,
    selected_token_id: int,
    *,
    alternatives: int = 10,
    sampler_logits: Any | None = None,
) -> DistributionMetrics:
    values = _as_fp32_logits(logits)
    if selected_token_id < 0 or selected_token_id >= values.size:
        raise InvalidLogitsError(
            "selected token ID is outside the vocabulary",
            details={"token_id": selected_token_id, "vocabulary_size": int(values.size)},
        )
    if not np.isfinite(values[selected_token_id]):
        raise InvalidLogitsError("selected token was filtered out of this distribution")
    if alternatives < 0:
        raise InvalidSamplingSettingsError("alternatives must be non-negative")
    log_probabilities, probabilities, log_normalizer = _stable_log_softmax(values)
    entropy_terms = np.zeros_like(probabilities, dtype=np.float32)
    positive = probabilities > 0
    entropy_terms[positive] = np.multiply(
        probabilities[positive], log_probabilities[positive], dtype=np.float32
    )
    entropy = float(-np.sum(entropy_terms, dtype=np.float32))
    if entropy < 0 and entropy > -1e-6:
        entropy = 0.0

    survivor_mask: np.ndarray | None = None
    if sampler_logits is not None:
        sampled = _as_fp32_logits(sampler_logits)
        if sampled.size != values.size:
            raise InvalidLogitsError("raw and sampler vocabularies have different sizes")
        survivor_mask = np.isfinite(sampled)

    ordered = _ordered_token_ids(values)
    top_ids = ordered[:alternatives]
    alternatives_result = tuple(
        Alternative(
            token_id=int(token_id),
            logit=float(values[token_id]),
            log_probability=float(log_probabilities[token_id]),
            probability=float(probabilities[token_id]),
            rank=index + 1,
            survived_sampler_filtering=(
                bool(survivor_mask[token_id]) if survivor_mask is not None else None
            ),
            chosen=int(token_id) == selected_token_id,
        )
        for index, token_id in enumerate(top_ids)
    )
    return DistributionMetrics(
        selected_token_id=selected_token_id,
        selected_logit=float(values[selected_token_id]),
        selected_log_probability=float(log_probabilities[selected_token_id]),
        selected_probability=float(probabilities[selected_token_id]),
        selected_rank=_rank(values, selected_token_id),
        entropy_nats=entropy,
        log_normalizer=float(log_normalizer),
        vocabulary_size=int(values.size),
        surviving_tokens=int(np.count_nonzero(np.isfinite(values))),
        alternatives=alternatives_result,
    )


def compute_token_metrics(
    raw_logits: Any,
    processed_logits: Any,
    selected_token_id: int,
    *,
    alternatives: int = 10,
) -> TokenProbabilityMetrics:
    raw_values = _as_fp32_logits(raw_logits, allow_negative_infinity=False)
    processed_values = _as_fp32_logits(processed_logits)
    if raw_values.size != processed_values.size:
        raise InvalidLogitsError("raw and processed logits have different vocabulary sizes")
    raw = compute_distribution_metrics(
        raw_values,
        selected_token_id,
        alternatives=alternatives,
        sampler_logits=processed_values,
    )
    sampling = compute_distribution_metrics(
        processed_values,
        selected_token_id,
        alternatives=alternatives,
        sampler_logits=processed_values,
    )
    return TokenProbabilityMetrics(
        token_id=selected_token_id,
        raw=raw,
        sampling=sampling,
        processed_logit=float(processed_values[selected_token_id]),
        raw_surprise_nats=-raw.selected_log_probability,
        sampling_surprise_nats=-sampling.selected_log_probability,
    )


def _survivor_count(logits: np.ndarray) -> int:
    return int(np.count_nonzero(np.isfinite(logits)))


def _operation(
    operations: list[SamplingOperation],
    name: str,
    applied: bool,
    logits: np.ndarray,
    parameters: Mapping[str, Any],
) -> None:
    operations.append(
        SamplingOperation(
            order=len(operations),
            name=name,
            applied=applied,
            parameters=dict(parameters),
            surviving_tokens=_survivor_count(logits),
        )
    )


def apply_sampling_pipeline(
    raw_logits: Any,
    settings: SamplingSettings,
    *,
    token_counts: Mapping[int, int] | None = None,
    allowed_token_ids: Sequence[int] | None = None,
) -> SamplingPipelineResult:
    logits = _as_fp32_logits(raw_logits, allow_negative_infinity=False)
    vocabulary_size = int(logits.size)
    operations: list[SamplingOperation] = []
    counts = dict(token_counts or {})
    for token_id, count in counts.items():
        if token_id < 0 or token_id >= vocabulary_size or count < 0:
            raise InvalidSamplingSettingsError(
                "token frequency map contains an invalid token ID or count",
                details={"token_id": token_id, "count": count, "vocabulary_size": vocabulary_size},
            )

    repetition_applied = settings.repetition_penalty != 1.0 and bool(counts)
    if repetition_applied:
        ids = np.fromiter(counts.keys(), dtype=np.int64)
        values = logits[ids]
        logits[ids] = np.where(
            values < 0,
            np.multiply(values, np.float32(settings.repetition_penalty), dtype=np.float32),
            np.divide(values, np.float32(settings.repetition_penalty), dtype=np.float32),
        )
    _operation(
        operations,
        "repetition_penalty",
        repetition_applied,
        logits,
        {"penalty": settings.repetition_penalty},
    )

    frequency_applied = settings.frequency_penalty != 0.0 and bool(counts)
    if frequency_applied:
        for token_id, count in counts.items():
            logits[token_id] = np.subtract(
                logits[token_id],
                np.float32(settings.frequency_penalty * count),
                dtype=np.float32,
            )
    _operation(
        operations,
        "frequency_penalty",
        frequency_applied,
        logits,
        {"penalty": settings.frequency_penalty},
    )

    presence_applied = settings.presence_penalty != 0.0 and bool(counts)
    if presence_applied:
        for token_id, count in counts.items():
            if count:
                logits[token_id] = np.subtract(
                    logits[token_id],
                    np.float32(settings.presence_penalty),
                    dtype=np.float32,
                )
    _operation(
        operations,
        "presence_penalty",
        presence_applied,
        logits,
        {"penalty": settings.presence_penalty},
    )

    constraints_applied = allowed_token_ids is not None
    if constraints_applied:
        allowed = np.zeros(vocabulary_size, dtype=np.bool_)
        for token_id in allowed_token_ids or ():
            if token_id < 0 or token_id >= vocabulary_size:
                raise InvalidSamplingSettingsError(
                    "allowed token ID is outside the vocabulary",
                    details={"token_id": token_id, "vocabulary_size": vocabulary_size},
                )
            allowed[token_id] = True
        logits[~allowed] = -np.inf
        if not allowed.any():
            raise InvalidSamplingSettingsError("token constraints removed every vocabulary item")
    _operation(
        operations,
        "constraints",
        constraints_applied,
        logits,
        {"allowed_token_count": None if allowed_token_ids is None else len(set(allowed_token_ids))},
    )

    greedy = settings.temperature == 0.0
    if greedy:
        winner = int(_ordered_token_ids(logits)[0])
        logits[np.arange(vocabulary_size) != winner] = -np.inf
    elif settings.temperature != 1.0:
        finite = np.isfinite(logits)
        logits[finite] = np.divide(
            logits[finite], np.float32(settings.temperature), dtype=np.float32
        )
    _operation(
        operations,
        "temperature",
        greedy or settings.temperature != 1.0,
        logits,
        {"temperature": settings.temperature, "greedy": greedy},
    )

    current_count = _survivor_count(logits)
    top_k_applied = 0 < settings.top_k < current_count
    if top_k_applied:
        keep = _ordered_token_ids(logits)[: max(settings.top_k, settings.minimum_tokens_to_keep)]
        mask = np.ones(vocabulary_size, dtype=np.bool_)
        mask[keep] = False
        logits[mask] = -np.inf
    _operation(
        operations,
        "top_k",
        top_k_applied,
        logits,
        {"top_k": settings.top_k, "minimum_tokens_to_keep": settings.minimum_tokens_to_keep},
    )

    top_p_applied = (
        settings.top_p < 1.0 and _survivor_count(logits) > settings.minimum_tokens_to_keep
    )
    if top_p_applied:
        _logs, probabilities, _normalizer = _stable_log_softmax(logits)
        ordered = _ordered_token_ids(logits)
        cumulative = np.cumsum(probabilities[ordered], dtype=np.float32)
        crossing = int(np.searchsorted(cumulative, np.float32(settings.top_p), side="left")) + 1
        keep_count = min(len(ordered), max(crossing, settings.minimum_tokens_to_keep))
        keep = ordered[:keep_count]
        mask = np.ones(vocabulary_size, dtype=np.bool_)
        mask[keep] = False
        logits[mask] = -np.inf
    _operation(
        operations,
        "top_p",
        top_p_applied,
        logits,
        {"top_p": settings.top_p, "minimum_tokens_to_keep": settings.minimum_tokens_to_keep},
    )

    min_p_applied = (
        settings.min_p > 0.0 and _survivor_count(logits) > settings.minimum_tokens_to_keep
    )
    if min_p_applied:
        _logs, probabilities, _normalizer = _stable_log_softmax(logits)
        threshold = np.float32(settings.min_p) * np.max(probabilities)
        eligible = np.flatnonzero(probabilities >= threshold)
        if len(eligible) < settings.minimum_tokens_to_keep:
            eligible = _ordered_token_ids(logits)[: settings.minimum_tokens_to_keep]
        mask = np.ones(vocabulary_size, dtype=np.bool_)
        mask[eligible] = False
        logits[mask] = -np.inf
    _operation(
        operations,
        "min_p",
        min_p_applied,
        logits,
        {"min_p": settings.min_p, "minimum_tokens_to_keep": settings.minimum_tokens_to_keep},
    )

    if not np.isfinite(logits).any():
        raise InvalidSamplingSettingsError("sampling pipeline removed every vocabulary item")
    return SamplingPipelineResult(logits=logits, operations=tuple(operations), greedy=greedy)


def effective_seed(requested: int | None) -> int:
    if requested is None:
        return secrets.randbits(64)
    if (
        isinstance(requested, bool)
        or not isinstance(requested, int)
        or not 0 <= requested <= UINT64_MAX
    ):
        raise InvalidSamplingSettingsError("seed must be an unsigned 64-bit integer")
    return requested


def sample_token(processed_logits: Any, seed: int) -> tuple[int, str]:
    values = _as_fp32_logits(processed_logits)
    seed = effective_seed(seed)
    finite_ids = np.flatnonzero(np.isfinite(values))
    if len(finite_ids) == 1:
        return int(finite_ids[0]), "greedy (seed did not affect token selection)"
    _logs, probabilities, _normalizer = _stable_log_softmax(values)
    generator = np.random.Generator(np.random.PCG64(seed))
    token_id = int(generator.choice(values.size, p=probabilities.astype(np.float64)))
    return token_id, "NumPy PCG64"


def _perplexity(log_probability_sum: float, count: int) -> float | None:
    if count == 0:
        return None
    exponent = -log_probability_sum / count
    try:
        return math.exp(exponent)
    except OverflowError:
        return math.inf


class MetricsAccumulator:
    """Stateful cumulative and segment perplexity accounting."""

    def __init__(self) -> None:
        self._count = 0
        self._raw_sum = 0.0
        self._sampling_sum = 0.0
        self._segment_sums: dict[str, float] = {"reasoning": 0.0, "answer": 0.0}
        self._segment_counts: dict[str, int] = {"reasoning": 0, "answer": 0}
        self._unknown_segment_tokens = 0

    def observe(
        self, metrics: TokenProbabilityMetrics, *, segment: str = "unknown"
    ) -> TokenProbabilityMetrics:
        self._count += 1
        self._raw_sum += metrics.raw.selected_log_probability
        self._sampling_sum += metrics.sampling.selected_log_probability
        if segment in self._segment_sums:
            self._segment_sums[segment] += metrics.raw.selected_log_probability
            self._segment_counts[segment] += 1
        else:
            self._unknown_segment_tokens += 1
        return metrics.model_copy(
            update={
                "cumulative_raw_log_probability": self._raw_sum,
                "cumulative_sampling_log_probability": self._sampling_sum,
                "running_response_perplexity": _perplexity(self._raw_sum, self._count),
                "running_post_sampler_perplexity": _perplexity(self._sampling_sum, self._count),
            }
        )

    def summary(self) -> dict[str, Any]:
        return {
            "token_count": self._count,
            "conditional_generated_response_perplexity": _perplexity(self._raw_sum, self._count),
            "post_sampler_perplexity": _perplexity(self._sampling_sum, self._count),
            "reasoning_segment_perplexity": _perplexity(
                self._segment_sums["reasoning"], self._segment_counts["reasoning"]
            ),
            "answer_segment_perplexity": _perplexity(
                self._segment_sums["answer"], self._segment_counts["answer"]
            ),
            "segment_token_counts": dict(self._segment_counts),
            "unknown_segment_tokens_excluded": self._unknown_segment_tokens,
            "definition": "exp(-mean(raw full-vocabulary model log probability))",
        }


def score_prompt_perplexity(
    logits: Any,
    token_ids: Sequence[int],
    *,
    attention_mask: Sequence[bool] | None = None,
    text_position_mask: Sequence[bool] | None = None,
) -> PromptPerplexityResult:
    try:
        values = np.asarray(logits, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise InvalidLogitsError("prompt logits must be convertible to fp32") from exc
    if values.ndim != 2 or values.shape[0] != len(token_ids) or values.shape[1] == 0:
        raise InvalidLogitsError(
            "prompt logits must have shape [sequence_length, vocabulary_size]",
            details={"shape": tuple(values.shape), "token_count": len(token_ids)},
        )
    if np.isnan(values).any() or np.isinf(values).any():
        raise InvalidLogitsError("prompt logits contain a non-finite value")
    length, vocabulary_size = values.shape
    for name, mask in (
        ("attention_mask", attention_mask),
        ("text_position_mask", text_position_mask),
    ):
        if mask is not None and len(mask) != length:
            raise InvalidLogitsError(f"{name} length does not match prompt token count")

    log_probabilities: list[float | None] = [None] * length
    included: list[float] = []
    excluded_masked = 0
    excluded_non_text = 0
    for position in range(1, length):
        if attention_mask is not None and not attention_mask[position]:
            excluded_masked += 1
            continue
        if text_position_mask is not None and not text_position_mask[position]:
            excluded_non_text += 1
            continue
        target = token_ids[position]
        if target < 0 or target >= vocabulary_size:
            raise InvalidLogitsError(
                "prompt token ID is outside the vocabulary",
                details={
                    "position": position,
                    "token_id": target,
                    "vocabulary_size": vocabulary_size,
                },
            )
        row = _as_fp32_logits(values[position - 1], allow_negative_infinity=False)
        row_logs, _row_probs, _normalizer = _stable_log_softmax(row)
        selected = float(row_logs[target])
        log_probabilities[position] = selected
        included.append(selected)

    total = math.fsum(included)
    count = len(included)
    return PromptPerplexityResult(
        perplexity=_perplexity(total, count),
        mean_log_probability=(total / count if count else None),
        included_token_count=count,
        excluded_first_token=1 if length else 0,
        excluded_masked_tokens=excluded_masked,
        excluded_non_text_tokens=excluded_non_text,
        token_log_probabilities=tuple(log_probabilities),
    )
