"""Validated domain objects shared by discovery, adapters, and transports."""

from .capabilities import Capability, CapabilityMatrix, CapabilityState, CapabilitySupport
from .models import (
    Diagnostic,
    DiagnosticSeverity,
    ModelComponents,
    ModelDescriptor,
    ModelFingerprint,
    ModelTask,
    TrustDecision,
)

__all__ = [
    "Capability",
    "CapabilityMatrix",
    "CapabilityState",
    "CapabilitySupport",
    "Diagnostic",
    "DiagnosticSeverity",
    "ModelComponents",
    "ModelDescriptor",
    "ModelFingerprint",
    "ModelTask",
    "TrustDecision",
]
