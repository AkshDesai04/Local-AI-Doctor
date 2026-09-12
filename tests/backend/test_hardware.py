from __future__ import annotations

import pytest

from local_ai_doctor.config import DeviceMode, DType
from local_ai_doctor.errors import BackendUnavailableError
from local_ai_doctor.hardware import (
    AcceleratorDevice,
    BackendKind,
    CPUInfo,
    HardwareInventory,
    MemoryInfo,
    select_hardware,
)


def inventory(*devices: AcceleratorDevice) -> HardwareInventory:
    return HardwareInventory(
        operating_system="TestOS",
        os_release="1",
        python_version="3.12",
        cpu=CPUInfo(logical_cores=8, physical_cores=4, architecture="x86_64"),
        memory=MemoryInfo(
            total_bytes=16_000_000_000, available_bytes=8_000_000_000, source="fixture"
        ),
        accelerators=devices,
    )


def test_auto_prefers_usable_cuda_and_selects_device_aware_dtype() -> None:
    cuda = AcceleratorDevice(
        backend=BackendKind.CUDA,
        index=0,
        name="Fixture GPU",
        identifier="GPU-fixture",
        total_memory_bytes=8_000_000_000,
        compute_capability=(8, 9),
        runtime_available=True,
    )
    selected = select_hardware(inventory(cuda), DeviceMode.AUTO)
    assert selected.selected_backend is BackendKind.CUDA
    assert selected.device_identifier == "cuda:0"
    assert selected.effective_dtype is DType.BFLOAT16
    assert not selected.used_fallback


def test_requested_cuda_falls_back_only_when_allowed() -> None:
    physical_only = AcceleratorDevice(
        backend=BackendKind.CUDA,
        index=0,
        name="GPU without runtime",
        identifier="GPU-fixture",
        runtime_available=False,
        runtime_reason="runtime missing",
    )
    selected = select_hardware(inventory(physical_only), DeviceMode.CUDA, allow_cpu_fallback=True)
    assert selected.selected_backend is BackendKind.CPU
    assert selected.used_fallback
    with pytest.raises(BackendUnavailableError):
        select_hardware(inventory(physical_only), DeviceMode.CUDA, allow_cpu_fallback=False)


def test_cpu_rejects_unsafe_float16_request() -> None:
    with pytest.raises(BackendUnavailableError):
        select_hardware(inventory(), DeviceMode.CPU, requested_dtype=DType.FLOAT16)
