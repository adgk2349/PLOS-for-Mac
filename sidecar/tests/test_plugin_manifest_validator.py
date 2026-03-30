from __future__ import annotations

import pytest

from local_ai_core.models import (
    ExtensionCapability,
    PluginManifestV1,
    PluginPrivacyMode,
    PrivacyMode,
)
from local_ai_core.plugins.validator import PluginManifestValidator


def _manifest(*, privacy_mode: PluginPrivacyMode) -> PluginManifestV1:
    return PluginManifestV1(
        plugin_id="sample.plugin",
        version="0.1.0",
        api_version="v1",
        capabilities=[ExtensionCapability.RETRIEVER_SEARCH],
        privacy_mode=privacy_mode,
        permissions=["fs.read"],
        entrypoint="python -m sample",
    )


def test_validator_accepts_compatible_privacy_mode():
    manifest = _manifest(privacy_mode=PluginPrivacyMode.HYBRID)
    out = PluginManifestValidator.validate(manifest=manifest, app_privacy_mode=PrivacyMode.HYBRID)
    assert out.plugin_id == "sample.plugin"


def test_validator_rejects_privacy_mode_above_app_limit():
    manifest = _manifest(privacy_mode=PluginPrivacyMode.EXTERNAL_ALLOWED)
    with pytest.raises(ValueError):
        PluginManifestValidator.validate(manifest=manifest, app_privacy_mode=PrivacyMode.HYBRID)
