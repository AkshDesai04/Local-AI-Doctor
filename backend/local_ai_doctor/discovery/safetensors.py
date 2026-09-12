"""Bounded SafeTensors header inspection without loading tensor data."""

from __future__ import annotations

import json
import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

MAX_HEADER_BYTES: Final[int] = 256 * 1024**2
MAX_TENSORS: Final[int] = 1_000_000


class SafeTensorHeaderError(ValueError):
    pass


@dataclass(frozen=True, slots=True)
class SafeTensorSummary:
    files: tuple[str, ...]
    parameter_count: int
    dtype_parameter_counts: dict[str, int]
    tensor_names: frozenset[str]
    errors: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.errors


def read_safetensors_header(path: Path) -> dict[str, Any]:
    try:
        file_size = path.stat().st_size
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise SafeTensorHeaderError("file is shorter than the 8-byte header prefix")
            (header_length,) = struct.unpack("<Q", prefix)
            if header_length == 0 or header_length > MAX_HEADER_BYTES:
                raise SafeTensorHeaderError(
                    f"declared header length {header_length} is outside the allowed range"
                )
            if 8 + header_length > file_size:
                raise SafeTensorHeaderError("declared header extends beyond the file")
            header_bytes = stream.read(header_length)
    except OSError as exc:
        raise SafeTensorHeaderError(f"header could not be read ({type(exc).__name__})") from exc
    try:
        header = json.loads(header_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SafeTensorHeaderError("header is not valid UTF-8 JSON") from exc
    if not isinstance(header, dict):
        raise SafeTensorHeaderError("header JSON root is not an object")
    if len(header) > MAX_TENSORS + 1:
        raise SafeTensorHeaderError("header contains too many tensor records")

    payload_size = file_size - 8 - header_length
    for name, record in header.items():
        if name == "__metadata__":
            if not isinstance(record, dict):
                raise SafeTensorHeaderError("__metadata__ must be an object")
            continue
        if not isinstance(name, str) or not isinstance(record, dict):
            raise SafeTensorHeaderError("tensor entry is malformed")
        shape = record.get("shape")
        offsets = record.get("data_offsets")
        dtype = record.get("dtype")
        if (
            not isinstance(shape, list)
            or not all(isinstance(value, int) and value >= 0 for value in shape)
            or not isinstance(dtype, str)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(isinstance(value, int) for value in offsets)
            or offsets[0] < 0
            or offsets[1] < offsets[0]
            or offsets[1] > payload_size
        ):
            raise SafeTensorHeaderError(f"tensor metadata is invalid for {name!r}")
    return header


def inspect_safetensors(paths: list[Path] | tuple[Path, ...]) -> SafeTensorSummary:
    names: set[str] = set()
    dtype_counts: dict[str, int] = {}
    total_parameters = 0
    errors: list[str] = []
    files: list[str] = []
    for path in sorted(paths, key=lambda item: item.name.casefold()):
        files.append(path.name)
        try:
            header = read_safetensors_header(path)
        except SafeTensorHeaderError as exc:
            errors.append(f"{path.name}: {exc}")
            continue
        for name, record in header.items():
            if name == "__metadata__":
                continue
            if name in names:
                errors.append(f"{path.name}: duplicate tensor name {name!r} across shards")
                continue
            names.add(name)
            shape = record["shape"]
            parameters = math.prod(shape)
            dtype = str(record["dtype"])
            total_parameters += parameters
            dtype_counts[dtype] = dtype_counts.get(dtype, 0) + parameters
    return SafeTensorSummary(
        files=tuple(files),
        parameter_count=total_parameters,
        dtype_parameter_counts=dict(sorted(dtype_counts.items())),
        tensor_names=frozenset(names),
        errors=tuple(errors),
    )
