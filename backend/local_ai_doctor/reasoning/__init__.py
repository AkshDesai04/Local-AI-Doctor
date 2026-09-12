"""Explicit emitted-reasoning segmentation; no hidden-CoT inference."""

from .segments import (
    ReasoningSlice,
    SegmentAccounting,
    SegmentClass,
    SegmentedToken,
    TagReasoningSegmenter,
    UnknownReasoningSegmenter,
    segment_tagged_tokens,
)

__all__ = [
    "ReasoningSlice",
    "SegmentAccounting",
    "SegmentClass",
    "SegmentedToken",
    "TagReasoningSegmenter",
    "UnknownReasoningSegmenter",
    "segment_tagged_tokens",
]
