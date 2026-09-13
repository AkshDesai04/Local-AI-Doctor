"""Best-effort hardware discovery with no mandatory ML runtime import."""

from __future__ import annotations

import csv
import importlib.metadata
import os
import platform
import subprocess
import sys
from collections.abc import Sequence
from typing import Protocol

from .models import AcceleratorDevice, BackendKind, CPUInfo, HardwareInventory, MemoryInfo


class HardwareProbe(Protocol):
    def discover(self) -> HardwareInventory: ...


def _memory_info() -> MemoryInfo:
    try:
        import psutil

        memory = psutil.virtual_memory()
        return MemoryInfo(
            total_bytes=int(memory.total),
            available_bytes=int(memory.available),
            source="psutil",
        )
    except ImportError:
        pass

    if sys.platform == "win32":
        try:
            import ctypes

            class MemoryStatus(ctypes.Structure):
                _fields_ = [
                    ("length", ctypes.c_ulong),
                    ("memory_load", ctypes.c_ulong),
                    ("total_physical", ctypes.c_ulonglong),
                    ("available_physical", ctypes.c_ulonglong),
                    ("total_page_file", ctypes.c_ulonglong),
                    ("available_page_file", ctypes.c_ulonglong),
                    ("total_virtual", ctypes.c_ulonglong),
                    ("available_virtual", ctypes.c_ulonglong),
                    ("available_extended_virtual", ctypes.c_ulonglong),
                ]

            status = MemoryStatus()
            status.length = ctypes.sizeof(status)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
                return MemoryInfo(
                    total_bytes=int(status.total_physical),
                    available_bytes=int(status.available_physical),
                    source="win32",
                )
        except (AttributeError, OSError):
            pass
    else:
        try:
            page_size = os.sysconf("SC_PAGE_SIZE")
            pages = os.sysconf("SC_PHYS_PAGES")
            available_pages = os.sysconf("SC_AVPHYS_PAGES")
            return MemoryInfo(
                total_bytes=int(page_size * pages),
                available_bytes=int(page_size * available_pages),
                source="sysconf",
            )
        except (AttributeError, OSError, ValueError):
            pass
    return MemoryInfo(source="unavailable")


def _physical_cores() -> int | None:
    try:
        import psutil

        count = psutil.cpu_count(logical=False)
        return int(count) if count is not None else None
    except ImportError:
        return None


def _software_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in ("torch", "transformers", "tokenizers", "safetensors", "numpy"):
        try:
            versions[distribution] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            versions[distribution] = "not-installed"
    return versions


def _torch_devices() -> tuple[list[AcceleratorDevice], list[str]]:
    warnings: list[str] = []
    try:
        import torch
    except (ImportError, OSError) as exc:
        return [], [f"PyTorch runtime unavailable ({type(exc).__name__})"]

    devices: list[AcceleratorDevice] = []
    hip_version = getattr(getattr(torch, "version", None), "hip", None)
    backend = BackendKind.ROCM if hip_version else BackendKind.CUDA
    try:
        cuda_available = bool(torch.cuda.is_available())
        count = int(torch.cuda.device_count()) if cuda_available else 0
        for index in range(count):
            properties = torch.cuda.get_device_properties(index)
            try:
                free_bytes, total_bytes = torch.cuda.mem_get_info(index)
            except (AttributeError, RuntimeError):
                free_bytes = None
                total_bytes = int(properties.total_memory)
            capability = torch.cuda.get_device_capability(index)
            devices.append(
                AcceleratorDevice(
                    backend=backend,
                    index=index,
                    name=str(properties.name),
                    identifier=f"{backend.value}:{index}",
                    total_memory_bytes=int(total_bytes),
                    free_memory_bytes=int(free_bytes) if free_bytes is not None else None,
                    compute_capability=(int(capability[0]), int(capability[1])),
                    runtime_available=True,
                )
            )
        if not cuda_available:
            reason = "PyTorch reports no compatible CUDA/ROCm runtime"
            warnings.append(reason)
    except (RuntimeError, AssertionError, OSError) as exc:
        warnings.append(f"PyTorch accelerator discovery failed ({type(exc).__name__})")

    mps = getattr(getattr(torch, "backends", None), "mps", None)
    try:
        if mps is not None and mps.is_available():
            devices.append(
                AcceleratorDevice(
                    backend=BackendKind.MPS,
                    index=0,
                    name="Apple Metal Performance Shaders",
                    identifier="mps:0",
                    runtime_available=True,
                )
            )
    except (RuntimeError, AttributeError):
        warnings.append("PyTorch MPS discovery failed")
    return devices, warnings


def _run_nvidia_smi(arguments: Sequence[str]) -> subprocess.CompletedProcess[str] | None:
    flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    try:
        return subprocess.run(
            ["nvidia-smi", *arguments],
            check=False,
            capture_output=True,
            text=True,
            timeout=3.0,
            creationflags=flags,
        )
    except (FileNotFoundError, OSError, subprocess.TimeoutExpired):
        return None


def _nvidia_smi_devices() -> tuple[list[AcceleratorDevice], list[str]]:
    """Report physical NVIDIA devices even when PyTorch is not installed."""

    fields = "index,name,memory.total,memory.free,compute_cap"
    process = _run_nvidia_smi([f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
    include_capability = True
    if process is not None and process.returncode != 0:
        fields = "index,name,memory.total,memory.free"
        process = _run_nvidia_smi([f"--query-gpu={fields}", "--format=csv,noheader,nounits"])
        include_capability = False
    if process is None or process.returncode != 0:
        return [], []

    devices: list[AcceleratorDevice] = []
    for row in csv.reader(process.stdout.splitlines(), skipinitialspace=True):
        expected = 5 if include_capability else 4
        if len(row) != expected:
            continue
        try:
            index = int(row[0])
            total = int(float(row[2])) * 1024**2
            free = int(float(row[3])) * 1024**2
            capability = None
            if include_capability:
                major, minor = row[4].split(".", maxsplit=1)
                capability = (int(major), int(minor))
        except (ValueError, IndexError):
            continue
        devices.append(
            AcceleratorDevice(
                backend=BackendKind.CUDA,
                index=index,
                name=row[1].strip(),
                identifier=f"cuda:{index}",
                total_memory_bytes=total,
                free_memory_bytes=free,
                compute_capability=capability,
                runtime_available=False,
                runtime_reason="device found by nvidia-smi but no usable PyTorch CUDA runtime was detected",
            )
        )
    warning = (
        "NVIDIA hardware was detected, but CUDA inference is unavailable until a compatible "
        "PyTorch runtime is installed"
    )
    return devices, [warning] if devices else []


class SystemHardwareProbe:
    """Discover host resources without assuming that CUDA or PyTorch exists."""

    def discover(self, *, probe_runtime: bool = True) -> HardwareInventory:
        if probe_runtime:
            torch_devices, warnings = _torch_devices()
        else:
            torch_devices = []
            warnings = ["accelerator runtime probing skipped because CPU execution is configured"]
        devices = torch_devices
        if not any(device.backend is BackendKind.CUDA for device in devices):
            smi_devices, smi_warnings = _nvidia_smi_devices()
            devices.extend(smi_devices)
            warnings.extend(smi_warnings)
        return HardwareInventory(
            operating_system=platform.system(),
            os_release=platform.release(),
            python_version=platform.python_version(),
            cpu=CPUInfo(
                logical_cores=max(1, os.cpu_count() or 1),
                physical_cores=_physical_cores(),
                architecture=platform.machine() or "unknown",
                processor=platform.processor() or None,
            ),
            memory=_memory_info(),
            accelerators=tuple(devices),
            software_versions=_software_versions(),
            discovery_warnings=tuple(dict.fromkeys(warnings)),
        )
