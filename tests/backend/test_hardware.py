from __future__ import annotations

import shutil
import subprocess

import pytest

from local_ai_doctor.config import DeviceMode, DType
from local_ai_doctor.errors import BackendUnavailableError
from local_ai_doctor.hardware import (
    AcceleratorDevice,
    BackendKind,
    CPUInfo,
    HardwareInventory,
    MemoryInfo,
    SystemHardwareProbe,
    select_hardware,
)
from local_ai_doctor.hardware import probe as hardware_probe


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


def test_explicit_cpu_probe_does_not_import_accelerator_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_torch_probe() -> tuple[list[AcceleratorDevice], list[str]]:
        raise AssertionError("CPU-only discovery must not import the accelerator runtime")

    monkeypatch.setattr(hardware_probe, "_torch_devices", unexpected_torch_probe)
    monkeypatch.setattr(hardware_probe, "_nvidia_smi_devices", lambda: ([], []))

    discovered = SystemHardwareProbe().discover(probe_runtime=False)

    assert discovered.accelerators == ()
    assert "CPU execution is configured" in discovered.discovery_warnings[0]


def _nvidia_smi_reports_a_gpu() -> bool:
    path = shutil.which("nvidia-smi")
    if path is None:
        return False
    try:
        result = subprocess.run(
            [path, "--query-gpu=index", "--format=csv,noheader"],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0 and bool(result.stdout.strip())


@pytest.mark.skipif(
    not _nvidia_smi_reports_a_gpu(), reason="requires a real machine with a working nvidia-smi"
)
def test_real_hardware_detects_gpu_when_nvidia_smi_works() -> None:
    """Regression guard: a machine where nvidia-smi sees a GPU must end up with a

    usable CUDA device and an auto selection that actually picks it. This catches
    silent GPU loss (e.g. a CPU-only torch build shadowing a CUDA one) that unit
    tests with fixture inventories can't see.
    """

    discovered = SystemHardwareProbe().discover()
    cuda_devices = discovered.usable(BackendKind.CUDA)
    assert cuda_devices, (
        f"nvidia-smi reports a GPU but no usable CUDA device was discovered: "
        f"accelerators={discovered.accelerators!r} "
        f"warnings={discovered.discovery_warnings!r} "
        f"software_versions={discovered.software_versions!r}"
    )

    selected = select_hardware(discovered, DeviceMode.AUTO)
    assert selected.selected_backend is BackendKind.CUDA
    assert not selected.used_fallback
