"""Immutable hardware discovery records."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from ..config import DeviceMode, DType


class BackendKind(StrEnum):
    CPU = "cpu"
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"


class MemoryInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    total_bytes: int | None = Field(default=None, ge=0)
    available_bytes: int | None = Field(default=None, ge=0)
    source: str


class CPUInfo(BaseModel):
    model_config = ConfigDict(frozen=True)

    logical_cores: int = Field(ge=1)
    physical_cores: int | None = Field(default=None, ge=1)
    architecture: str
    processor: str | None = None


class AcceleratorDevice(BaseModel):
    model_config = ConfigDict(frozen=True)

    backend: BackendKind
    index: int = Field(ge=0)
    name: str
    identifier: str
    total_memory_bytes: int | None = Field(default=None, ge=0)
    free_memory_bytes: int | None = Field(default=None, ge=0)
    compute_capability: tuple[int, int] | None = None
    runtime_available: bool
    runtime_reason: str | None = None


class HardwareInventory(BaseModel):
    model_config = ConfigDict(frozen=True)

    operating_system: str
    os_release: str
    python_version: str
    cpu: CPUInfo
    memory: MemoryInfo
    accelerators: tuple[AcceleratorDevice, ...] = ()
    software_versions: dict[str, str] = Field(default_factory=dict)
    discovery_warnings: tuple[str, ...] = ()

    def usable(self, backend: BackendKind) -> tuple[AcceleratorDevice, ...]:
        return tuple(
            device
            for device in self.accelerators
            if device.backend is backend and device.runtime_available
        )


class HardwareSelection(BaseModel):
    model_config = ConfigDict(frozen=True)

    requested: DeviceMode
    selected_backend: BackendKind
    device_identifier: str
    effective_dtype: DType
    used_fallback: bool = False
    reason: str
    warnings: tuple[str, ...] = ()
