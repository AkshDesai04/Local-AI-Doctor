from __future__ import annotations

from local_ai_doctor.reasoning import (
    SegmentAccounting,
    SegmentClass,
    TagReasoningSegmenter,
    UnknownReasoningSegmenter,
    segment_tagged_tokens,
)


def test_tags_split_across_tokens_preserve_exact_text_and_boundaries() -> None:
    raw = ("<thi", "nk>step", " one</th", "ink>answer")
    segmented = segment_tagged_tokens(raw)
    assert tuple(item.text for item in segmented) == raw
    assert "".join(item.text for item in segmented) == "<think>step one</think>answer"
    assert [item.classification for item in segmented] == [
        SegmentClass.REASONING,
        SegmentClass.REASONING,
        SegmentClass.REASONING,
        SegmentClass.UNKNOWN,
    ]
    final = segmented[-1]
    assert final.slices[0].delimiter
    assert final.slices[0].classification is SegmentClass.REASONING
    assert final.slices[1].classification is SegmentClass.ANSWER


def test_streaming_segmenter_delays_only_ambiguous_tag_prefix() -> None:
    segmenter = TagReasoningSegmenter()
    assert segmenter.feed(0, "plain <thi") == ()
    emitted = segmenter.feed(1, "nk>x")
    assert [item.token_index for item in emitted] == [0, 1]
    assert emitted[0].classification is SegmentClass.UNKNOWN
    assert emitted[1].classification is SegmentClass.REASONING
    assert segmenter.finalize() == ()


def test_unclosed_reasoning_is_reported_without_reclassifying_visible_tokens() -> None:
    segmenter = TagReasoningSegmenter()
    output = list(segmenter.feed(0, "<think>unfinished"))
    output.extend(segmenter.finalize())
    assert output[0].classification is SegmentClass.REASONING
    assert segmenter.warnings


def test_unknown_adapter_never_claims_hidden_reasoning() -> None:
    segmenter = UnknownReasoningSegmenter()
    token = segmenter.feed(0, "analysis-like prose")[0]
    assert token.classification is SegmentClass.UNKNOWN


def test_segment_accounting_excludes_mixed_unknown_tokens() -> None:
    accounting = SegmentAccounting()
    accounting.add(SegmentClass.REASONING, raw_log_probability=-1.0, duration_seconds=0.5)
    accounting.add(SegmentClass.ANSWER, raw_log_probability=-2.0, duration_seconds=1.0)
    accounting.add(SegmentClass.UNKNOWN, raw_log_probability=-10.0, duration_seconds=2.0)
    summary = accounting.summary()
    assert summary["unknown_tokens_excluded"] == 1
    assert summary["reasoning"]["perplexity"] > 1
    assert summary["answer"]["token_count"] == 1
