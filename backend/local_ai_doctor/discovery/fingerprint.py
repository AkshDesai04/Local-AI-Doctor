"""Stable model fingerprints designed for multi-gigabyte local checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Final

from ..domain.models import ModelFingerprint

_WEIGHT_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".safetensors", ".bin", ".pt", ".pth", ".gguf"}
)
_METADATA_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".json", ".jinja", ".txt", ".model", ".py", ".md"}
)
_SAMPLE_BYTES: Final[int] = 1024 * 1024
_MAX_HEADER_BYTES: Final[int] = 256 * 1024**2


class FingerprintMode(StrEnum):
    QUICK = "quick"
    FULL = "full"


@dataclass(frozen=True, slots=True)
class FingerprintPolicy:
    mode: FingerprintMode = FingerprintMode.QUICK
    include_documentation: bool = False


def _hash_stream(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(4 * 1024**2):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_hash(path: Path) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as stream:
        digest.update(stream.read(_SAMPLE_BYTES))
        if size > _SAMPLE_BYTES:
            stream.seek(max(0, size - _SAMPLE_BYTES))
            digest.update(stream.read(_SAMPLE_BYTES))
    digest.update(struct.pack("<Q", size))
    return digest.hexdigest()


def _safetensors_header_hash(path: Path) -> str:
    digest = hashlib.sha256()
    size = path.stat().st_size
    with path.open("rb") as stream:
        prefix = stream.read(8)
        if len(prefix) != 8:
            return _sample_hash(path)
        (header_length,) = struct.unpack("<Q", prefix)
        if not 0 < header_length <= _MAX_HEADER_BYTES or 8 + header_length > size:
            return _sample_hash(path)
        digest.update(prefix)
        remaining = header_length
        while remaining:
            chunk = stream.read(min(4 * 1024**2, remaining))
            if not chunk:
                return _sample_hash(path)
            digest.update(chunk)
            remaining -= len(chunk)
    digest.update(struct.pack("<Q", size))
    return digest.hexdigest()


def _candidate_files(root: Path, policy: FingerprintPolicy) -> tuple[list[Path], list[Path]]:
    metadata: list[Path] = []
    weights: list[Path] = []
    for current, directories, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        directories[:] = [
            name
            for name in directories
            if not (current_path / name).is_symlink() and not name.startswith(".")
        ]
        for filename in filenames:
            path = current_path / filename
            if path.is_symlink():
                continue
            suffix = path.suffix.lower()
            if suffix in _WEIGHT_SUFFIXES:
                weights.append(path)
            elif suffix in _METADATA_SUFFIXES and (policy.include_documentation or suffix != ".md"):
                metadata.append(path)

    def key(item: Path) -> str:
        return item.relative_to(root).as_posix().casefold()

    return sorted(metadata, key=key), sorted(weights, key=key)


def _manifest_digest(
    root: Path, files: list[Path], policy: FingerprintPolicy, *, weight: bool
) -> str:
    records: list[dict[str, object]] = []
    for path in files:
        relative = path.relative_to(root).as_posix()
        if not weight or policy.mode is FingerprintMode.FULL:
            content_digest = _hash_stream(path)
        elif path.suffix.lower() == ".safetensors":
            content_digest = _safetensors_header_hash(path)
        else:
            content_digest = _sample_hash(path)
        records.append(
            {
                "path": relative,
                "size": path.stat().st_size,
                "content_sha256": content_digest,
            }
        )
    canonical = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def fingerprint_model_directory(
    root: Path,
    policy: FingerprintPolicy | None = None,
) -> ModelFingerprint:
    policy = policy or FingerprintPolicy()
    resolved = root.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError("model fingerprint target must be a directory")
    metadata, weights = _candidate_files(resolved, policy)
    metadata_digest = _manifest_digest(resolved, metadata, policy, weight=False)
    weights_digest = _manifest_digest(resolved, weights, policy, weight=True)
    identity = {
        "version": 1,
        "metadata": metadata_digest,
        "weights": weights_digest,
        "mode": policy.mode.value,
    }
    value = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return ModelFingerprint(
        value=value,
        strength="full-content"
        if policy.mode is FingerprintMode.FULL
        else "metadata-and-weight-headers",
        file_count=len(metadata) + len(weights),
        total_weight_bytes=sum(path.stat().st_size for path in weights),
        metadata_digest=metadata_digest,
        weights_digest=weights_digest,
    )
