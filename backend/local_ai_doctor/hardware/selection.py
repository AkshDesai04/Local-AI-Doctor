"""Deterministic device and dtype selection with explainable fallback."""

from __future__ import annotations

from ..config import DeviceMode, DType
from ..errors import BackendUnavailableError
from .models import BackendKind, HardwareInventory, HardwareSelection


def _cuda_dtype(
    device_capability: tuple[int, int] | None, requested: DType
) -> tuple[DType, list[str]]:
    warnings: list[str] = []
    if requested is DType.AUTO:
        if device_capability and device_capability[0] >= 8:
            return DType.BFLOAT16, warnings
        return DType.FLOAT16, warnings
    if requested is DType.BFLOAT16 and (not device_capability or device_capability[0] < 8):
        warnings.append(
            "bfloat16 support could not be verified for this GPU; model loading may reject it"
        )
    return requested, warnings


def _cpu_dtype(requested: DType) -> tuple[DType, list[str]]:
    if requested is DType.AUTO:
        return DType.FLOAT32, []
    if requested is DType.FLOAT16:
        raise BackendUnavailableError(
            "float16 was requested for the CPU backend",
            hint="Use float32, bfloat16 on supported CPUs, or select CUDA.",
        )
    warnings = []
    if requested is DType.BFLOAT16:
        warnings.append("CPU bfloat16 kernel support is architecture and operation dependent")
    return requested, warnings


def select_hardware(
    inventory: HardwareInventory,
    requested: DeviceMode,
    *,
    requested_dtype: DType = DType.AUTO,
    allow_cpu_fallback: bool = True,
) -> HardwareSelection:
    cuda_devices = inventory.usable(BackendKind.CUDA)
    if requested is DeviceMode.CPU:
        dtype, warnings = _cpu_dtype(requested_dtype)
        return HardwareSelection(
            requested=requested,
            selected_backend=BackendKind.CPU,
            device_identifier="cpu",
            effective_dtype=dtype,
            reason="CPU execution was explicitly requested",
            warnings=tuple(warnings),
        )

    if cuda_devices:
        device = cuda_devices[0]
        dtype, warnings = _cuda_dtype(device.compute_capability, requested_dtype)
        reason = (
            "selected the first usable CUDA device discovered by the runtime"
            if requested is DeviceMode.AUTO
            else "CUDA execution was explicitly requested and is available"
        )
        return HardwareSelection(
            requested=requested,
            selected_backend=BackendKind.CUDA,
            device_identifier=f"cuda:{device.index}",
            effective_dtype=dtype,
            reason=reason,
            warnings=tuple(warnings),
        )

    if requested is DeviceMode.CUDA and not allow_cpu_fallback:
        found = [
            device.name for device in inventory.accelerators if device.backend is BackendKind.CUDA
        ]
        raise BackendUnavailableError(
            "CUDA was requested but no usable CUDA runtime was discovered",
            hint="Install a compatible GPU runtime, select cpu, or enable CPU fallback.",
            details={
                "physical_cuda_devices": found,
                "discovery_warnings": inventory.discovery_warnings,
            },
        )

    dtype, warnings = _cpu_dtype(requested_dtype)
    fallback = requested is DeviceMode.CUDA
    reason = (
        "CUDA was unavailable, so the configured CPU fallback was selected"
        if fallback
        else "no usable accelerator runtime was found; selected the portable CPU backend"
    )
    if fallback:
        warnings.append("requested CUDA execution is not in effect")
    return HardwareSelection(
        requested=requested,
        selected_backend=BackendKind.CPU,
        device_identifier="cpu",
        effective_dtype=dtype,
        used_fallback=fallback,
        reason=reason,
        warnings=tuple(warnings),
    )
