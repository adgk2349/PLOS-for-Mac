from __future__ import annotations

import logging

from ..models import (
    ExtensionCapability,
    PluginManifestV1,
    PluginPrivacyMode,
    PrivacyMode,
)

logger = logging.getLogger(__name__)


class PluginManifestValidator:
    @staticmethod
    def _privacy_rank(mode: PluginPrivacyMode) -> int:
        if mode == PluginPrivacyMode.LOCAL_ONLY:
            return 0
        if mode == PluginPrivacyMode.HYBRID:
            return 1
        return 2

    @staticmethod
    def _plugin_mode_from_app(mode: PrivacyMode) -> PluginPrivacyMode:
        if mode == PrivacyMode.LOCAL_ONLY:
            return PluginPrivacyMode.LOCAL_ONLY
        if mode == PrivacyMode.HYBRID:
            return PluginPrivacyMode.HYBRID
        return PluginPrivacyMode.EXTERNAL_ALLOWED

    @classmethod
    def effective_privacy_mode(cls, *, app_mode: PrivacyMode, plugin_mode: PluginPrivacyMode) -> PluginPrivacyMode:
        app_bound = cls._plugin_mode_from_app(app_mode)
        if cls._privacy_rank(plugin_mode) <= cls._privacy_rank(app_bound):
            return plugin_mode
        return app_bound

    @classmethod
    def validate(
        cls,
        *,
        manifest: PluginManifestV1,
        app_privacy_mode: PrivacyMode,
    ) -> PluginManifestV1:
        # Pydantic-level schema validation is already applied; enforce policy/compat here.
        declared = set(manifest.capabilities)
        supported = set(ExtensionCapability)
        unsupported = sorted(item.value for item in declared.difference(supported))
        if unsupported:
            raise ValueError(f"PLUGIN_VALIDATION_ERROR: unsupported capabilities: {', '.join(unsupported)}")

        effective = cls.effective_privacy_mode(app_mode=app_privacy_mode, plugin_mode=manifest.privacy_mode)
        if effective != manifest.privacy_mode:
            raise ValueError(
                "PLUGIN_PERMISSION_DENIED: "
                f"plugin_privacy_mode={manifest.privacy_mode.value}; "
                f"effective_privacy_mode={effective.value}"
            )

        logger.info(
            "plugin_validate:ok plugin_id=%s capabilities=%s privacy_mode=%s effective_privacy_mode=%s",
            manifest.plugin_id,
            ",".join(item.value for item in manifest.capabilities),
            manifest.privacy_mode.value,
            effective.value,
        )
        return manifest
