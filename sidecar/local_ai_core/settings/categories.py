from __future__ import annotations

from enum import Enum


class SettingsCategory(str, Enum):
    PRIVACY = "privacy"
    RUNTIME = "runtime"
    MEMORY = "memory"
    SEARCH = "search"
    INDEXING = "indexing"
    PERMISSIONS = "permissions"


SETTINGS_CATEGORIES: tuple[SettingsCategory, ...] = (
    SettingsCategory.PRIVACY,
    SettingsCategory.RUNTIME,
    SettingsCategory.MEMORY,
    SettingsCategory.SEARCH,
    SettingsCategory.INDEXING,
    SettingsCategory.PERMISSIONS,
)
