"""Fingerprint-pinned exceptions to the global ``trust_remote_code=False`` policy.

``inference.trust_remote_code`` cannot be set ``True`` globally (``config.py``
rejects it at validation time): loading a checkpoint's bundled Python still
means executing third-party code inside the worker process, and that decision
must stay narrow and reviewed rather than a blanket opt-in. This module is the
only place that narrow exception is allowed to live.

An entry pins one manifest fingerprint to one model directory name. The
fingerprint hashes metadata files fully plus SafeTensors headers and sizes
(``discovery/fingerprint.py``'s quick policy), so any change to the bundled
`.py` files, `config.json`, or the weight headers produces a different
fingerprint and silently drops the checkpoint back to rejection. Review does
not transitively cover a payload swap at the same path.

Review checklist before adding an entry: read every bundled `.py` file end to
end. Confirm there is no network I/O, no `subprocess`/`os.system`/`eval`/
`exec`/`__import__`/`ctypes`, no writes outside paths the loader itself
manages (the HF/torch cache), and no obfuscation. Record what the code
actually does in `reason`, not just that it "looked fine".
"""

from __future__ import annotations

from typing import Final, NamedTuple


class ReviewedCheckpoint(NamedTuple):
    fingerprint: str
    directory_name: str
    reason: str


_REVIEWED: Final[tuple[ReviewedCheckpoint, ...]] = (
    ReviewedCheckpoint(
        fingerprint="ec345828c936fa9da61111b7eab8a40fc45145c26c1f590c2419d9f9ff35fb9b",
        directory_name="Krutrim-1-instruct",
        reason=(
            "Bundled code is the unmodified MosaicML llm-foundry MPT reference "
            "implementation (attention.py, blocks.py, modeling_mpt.py, "
            "configuration_mpt.py, norm.py, ffn.py, fc.py, param_init_fns.py, "
            "meta_init_context.py, act_ckpt.py, custom_embedding.py, "
            "hf_prefixlm_converter.py, adapt_tokenizer.py, flash_attn_triton.py, "
            "warnings.py); reviewed 2026-09-19. It adds grouped_query_attention "
            "support that upstream Transformers' built-in MPT config does not "
            "accept (`attn_type` is restricted to multihead/multiquery "
            "attention there). No network I/O, subprocess, eval/exec, or "
            "filesystem writes outside the standard torch/HF cache calls."
        ),
    ),
)


def reviewed_reason(fingerprint: str, directory_name: str) -> str | None:
    """Return the recorded review reason, or None if this checkpoint isn't pinned."""

    for entry in _REVIEWED:
        if entry.fingerprint == fingerprint and entry.directory_name == directory_name:
            return entry.reason
    return None
