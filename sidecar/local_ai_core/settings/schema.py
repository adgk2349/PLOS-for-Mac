"""Canonical settings schema module.

The current SettingsModel remains defined in local_ai_core.models for backward
compatibility; this module provides the domain-classified import path.
"""

from ..models import SettingsModel

__all__ = ["SettingsModel"]

