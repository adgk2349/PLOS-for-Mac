from __future__ import annotations

from ..models import CapabilityRequestEnvelope, CapabilityResponseEnvelope, PluginErrorCode

PROTOCOL_VERSION = "v1"

OUT_OF_PROCESS_ERROR_CODES = (
    PluginErrorCode.PLUGIN_TIMEOUT,
    PluginErrorCode.PLUGIN_UNAVAILABLE,
    PluginErrorCode.PLUGIN_VALIDATION_ERROR,
    PluginErrorCode.PLUGIN_PERMISSION_DENIED,
)

__all__ = [
    "PROTOCOL_VERSION",
    "OUT_OF_PROCESS_ERROR_CODES",
    "CapabilityRequestEnvelope",
    "CapabilityResponseEnvelope",
]
