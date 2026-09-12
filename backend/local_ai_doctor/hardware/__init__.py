"""Hardware inventory and explicit backend-selection policy."""

from .models import (
    AcceleratorDevice,
    BackendKind,
    CPUInfo,
    HardwareInventory,
    HardwareSelection,
    MemoryInfo,
)
from .probe import HardwareProbe, SystemHardwareProbe
from .selection import select_hardware

__all__ = [
    "AcceleratorDevice",
    "BackendKind",
    "CPUInfo",
    "HardwareInventory",
    "HardwareProbe",
    "HardwareSelection",
    "MemoryInfo",
    "SystemHardwareProbe",
    "select_hardware",
]
