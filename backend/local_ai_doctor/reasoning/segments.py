"""Token-preserving streaming segmentation for emitted reasoning tags."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, model_validator

from ..adapters.base import ReasoningSegmentAdapter


class SegmentClass(StrEnum):
    REASONING = "reasoning"
    ANSWER = "answer"
    UNKNOWN = "unknown"


class ReasoningSlice(BaseModel):
    """A half-open character range within one exact decoded token piece."""

    model_config = ConfigDict(frozen=True)

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    classification: SegmentClass
    delimiter: bool = False

    @model_validator(mode="after")
    def non_empty(self) -> ReasoningSlice:
        if self.end <= self.start:
            raise ValueError("reasoning slice must be non-empty")
        return self


class SegmentedToken(BaseModel):
    model_config = ConfigDict(frozen=True)

    token_index: int = Field(ge=0)
    text: str
    classification: SegmentClass
    slices: tuple[ReasoningSlice, ...] = ()


@dataclass(frozen=True, slots=True)
class _Character:
    token_index: int
    offset: int
    value: str


def _longest_delimiter_prefix_suffix(text: str, delimiter: str) -> int:
    maximum = min(len(text), len(delimiter) - 1)
    for length in range(maximum, 0, -1):
        if delimiter.startswith(text[-length:]):
            return length
    return 0


class TagReasoningSegmenter(ReasoningSegmentAdapter):
    """Recognize configured tags even when either tag spans token boundaries.

    Tokens are never rewritten or split.  Character slices retain exact tag
    boundaries; a token spanning reasoning and answer text is honestly marked
    ``unknown`` at token level while its slices remain precise.
    """

    def __init__(self, opening_tag: str = "<think>", closing_tag: str = "</think>") -> None:
        if not opening_tag or not closing_tag or opening_tag == closing_tag:
            raise ValueError("reasoning delimiters must be non-empty and distinct")
        self._opening_tag = opening_tag
        self._closing_tag = closing_tag
        self._inside = False
        self._pending: list[_Character] = []
        self._texts: dict[int, str] = {}
        self._labels: dict[int, list[tuple[SegmentClass, bool] | None]] = {}
        self._empty_classes: dict[int, SegmentClass] = {}
        self._next_output = 0
        self._last_index = -1
        self._finalized = False
        self._warnings: list[str] = []

    @property
    def warnings(self) -> tuple[str, ...]:
        return tuple(self._warnings)

    @property
    def inside_reasoning(self) -> bool:
        return self._inside

    def _classify(
        self,
        characters: list[_Character],
        classification: SegmentClass,
        *,
        delimiter: bool,
    ) -> None:
        for character in characters:
            self._labels[character.token_index][character.offset] = (classification, delimiter)

    def _drain(self, *, final: bool) -> None:
        while self._pending:
            delimiter = self._closing_tag if self._inside else self._opening_tag
            text = "".join(character.value for character in self._pending)
            position = text.find(delimiter)
            current = SegmentClass.REASONING if self._inside else SegmentClass.ANSWER
            if position >= 0:
                before = self._pending[:position]
                marker = self._pending[position : position + len(delimiter)]
                self._classify(before, current, delimiter=False)
                # Delimiters are part of the visible reasoning protocol.  This
                # keeps tag-only tokens in the reasoning count without claiming
                # any invisible model state.
                self._classify(marker, SegmentClass.REASONING, delimiter=True)
                del self._pending[: position + len(delimiter)]
                self._inside = not self._inside
                continue
            retained = 0 if final else _longest_delimiter_prefix_suffix(text, delimiter)
            stable_count = len(self._pending) - retained
            if stable_count:
                self._classify(self._pending[:stable_count], current, delimiter=False)
                del self._pending[:stable_count]
            break

    def _make_token(self, token_index: int) -> SegmentedToken:
        text = self._texts[token_index]
        labels = self._labels[token_index]
        if not text:
            return SegmentedToken(
                token_index=token_index,
                text=text,
                classification=self._empty_classes[token_index],
            )
        if any(label is None for label in labels):
            raise RuntimeError(
                "attempted to emit a token before delimiter classification completed"
            )
        resolved = [label for label in labels if label is not None]
        slices: list[ReasoningSlice] = []
        start = 0
        previous = resolved[0]
        for offset, label in enumerate(resolved[1:], start=1):
            if label != previous:
                slices.append(
                    ReasoningSlice(
                        start=start,
                        end=offset,
                        classification=previous[0],
                        delimiter=previous[1],
                    )
                )
                start = offset
                previous = label
        slices.append(
            ReasoningSlice(
                start=start,
                end=len(text),
                classification=previous[0],
                delimiter=previous[1],
            )
        )
        classes = {item.classification for item in slices}
        classification = classes.pop() if len(classes) == 1 else SegmentClass.UNKNOWN
        return SegmentedToken(
            token_index=token_index,
            text=text,
            classification=classification,
            slices=tuple(slices),
        )

    def _ready(self) -> list[SegmentedToken]:
        result: list[SegmentedToken] = []
        while self._next_output <= self._last_index:
            labels = self._labels[self._next_output]
            if any(label is None for label in labels):
                break
            result.append(self._make_token(self._next_output))
            self._next_output += 1
        return result

    def feed(self, token_index: int, text: str) -> tuple[SegmentedToken, ...]:
        if self._finalized:
            raise RuntimeError("reasoning segmenter has already been finalized")
        if token_index != self._last_index + 1:
            raise ValueError("generated token indexes must be contiguous and start at zero")
        self._last_index = token_index
        self._texts[token_index] = text
        self._labels[token_index] = [None] * len(text)
        self._empty_classes[token_index] = (
            SegmentClass.REASONING if self._inside else SegmentClass.ANSWER
        )
        self._pending.extend(
            _Character(token_index=token_index, offset=offset, value=value)
            for offset, value in enumerate(text)
        )
        self._drain(final=False)
        return tuple(self._ready())

    def finalize(self) -> tuple[SegmentedToken, ...]:
        if self._finalized:
            return ()
        self._drain(final=True)
        if self._inside:
            self._warnings.append(
                "reasoning opening delimiter was emitted without a matching closing delimiter"
            )
        self._finalized = True
        return tuple(self._ready())


class UnknownReasoningSegmenter(ReasoningSegmentAdapter):
    """Used when a model/backend exposes no explicit reasoning channel."""

    def __init__(self) -> None:
        self._next_index = 0
        self._finalized = False

    def feed(self, token_index: int, text: str) -> tuple[SegmentedToken, ...]:
        if self._finalized:
            raise RuntimeError("reasoning segmenter has already been finalized")
        if token_index != self._next_index:
            raise ValueError("generated token indexes must be contiguous and start at zero")
        self._next_index += 1
        slices = (
            (ReasoningSlice(start=0, end=len(text), classification=SegmentClass.UNKNOWN),)
            if text
            else ()
        )
        return (
            SegmentedToken(
                token_index=token_index,
                text=text,
                classification=SegmentClass.UNKNOWN,
                slices=slices,
            ),
        )

    def finalize(self) -> tuple[SegmentedToken, ...]:
        self._finalized = True
        return ()


def segment_tagged_tokens(
    tokens: list[str] | tuple[str, ...],
    *,
    opening_tag: str = "<think>",
    closing_tag: str = "</think>",
) -> tuple[SegmentedToken, ...]:
    segmenter = TagReasoningSegmenter(opening_tag, closing_tag)
    output: list[SegmentedToken] = []
    for index, token in enumerate(tokens):
        output.extend(segmenter.feed(index, token))
    output.extend(segmenter.finalize())
    return tuple(output)


class SegmentAccounting:
    """Separate observable statistics without assigning mixed/unknown tokens."""

    def __init__(self) -> None:
        self._stats: dict[SegmentClass, dict[str, float | int]] = {
            SegmentClass.REASONING: {"tokens": 0, "duration": 0.0, "log_probability": 0.0},
            SegmentClass.ANSWER: {"tokens": 0, "duration": 0.0, "log_probability": 0.0},
        }
        self._unknown = 0

    def add(
        self,
        classification: SegmentClass,
        *,
        raw_log_probability: float,
        duration_seconds: float,
    ) -> None:
        if duration_seconds < 0:
            raise ValueError("token duration must be non-negative")
        if classification not in self._stats:
            self._unknown += 1
            return
        stats = self._stats[classification]
        stats["tokens"] = int(stats["tokens"]) + 1
        stats["duration"] = float(stats["duration"]) + duration_seconds
        stats["log_probability"] = float(stats["log_probability"]) + raw_log_probability

    def summary(self) -> dict[str, object]:
        result: dict[str, object] = {"unknown_tokens_excluded": self._unknown}
        for classification, stats in self._stats.items():
            count = int(stats["tokens"])
            duration = float(stats["duration"])
            log_probability = float(stats["log_probability"])
            perplexity = math.exp(-log_probability / count) if count else None
            result[classification.value] = {
                "token_count": count,
                "duration_seconds": duration,
                "raw_log_probability": log_probability if count else None,
                "perplexity": perplexity,
                "tokens_per_second": count / duration if duration > 0 else None,
            }
        return result
