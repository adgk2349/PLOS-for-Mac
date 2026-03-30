from __future__ import annotations

from local_ai_core.extensions.kernel import CapabilityRouter as LegacyCapabilityRouter
from local_ai_core.extensions.kernel import ExtensionKernel as LegacyExtensionKernel
from local_ai_core.extensions.service import FinetuneJobService as LegacyFinetuneJobService
from local_ai_core.extensions.service import PluginRegistryService as LegacyPluginRegistryService
from local_ai_core.plugins.kernel import CapabilityRouter as NewCapabilityRouter
from local_ai_core.plugins.kernel import ExtensionKernel as NewExtensionKernel
from local_ai_core.plugins.service import FinetuneJobService as NewFinetuneJobService
from local_ai_core.plugins.service import PluginRegistryService as NewPluginRegistryService


def test_legacy_extension_imports_alias_plugins():
    assert LegacyCapabilityRouter is NewCapabilityRouter
    assert LegacyExtensionKernel is NewExtensionKernel
    assert LegacyPluginRegistryService is NewPluginRegistryService
    assert LegacyFinetuneJobService is NewFinetuneJobService

