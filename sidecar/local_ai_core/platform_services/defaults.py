from __future__ import annotations

import importlib
import os
from typing import Any, Callable

from .contracts import AuditSink, AuthProvider, LicenseGate, PlatformServices, PolicyProvider
from ..auth import SessionAuth


class SessionTokenAuthProvider(AuthProvider):
    def __init__(self, session_token: str):
        self._delegate = SessionAuth(session_token)

    def verify_request(self, request) -> None:
        self._delegate.verify_request(request)


class AllowAllLicenseGate(LicenseGate):
    def is_allowed(self, *, feature: str, context: dict[str, Any] | None = None) -> bool:
        _ = feature
        _ = context
        return True


class NoOpAuditSink(AuditSink):
    def emit(self, *, event: str, payload: dict[str, Any]) -> None:
        _ = event
        _ = payload


class LocalStaticPolicyProvider(PolicyProvider):
    def get_policy(self) -> dict[str, Any]:
        return {
            "mode": "community",
            "external_policy": "local_static",
            "plugin_runtime": "disabled",
        }


import sys


def default_platform_services(session_token: str) -> PlatformServices:
    system_tools = None
    if sys.platform == "darwin":
        try:
            from .mac_tools import MacSystemTools
            system_tools = MacSystemTools()
        except Exception:
            system_tools = None

    return PlatformServices(
        auth_provider=SessionTokenAuthProvider(session_token),
        license_gate=AllowAllLicenseGate(),
        audit_sink=NoOpAuditSink(),
        policy_provider=LocalStaticPolicyProvider(),
        system_tools=system_tools,
    )


def load_platform_services(session_token: str) -> PlatformServices:
    defaults = default_platform_services(session_token)
    factory_path = (os.getenv("LOCAL_AI_PLATFORM_FACTORY") or "").strip()
    if not factory_path:
        return defaults

    try:
        module_name, func_name = factory_path.split(":", 1)
        module = importlib.import_module(module_name)
        factory = getattr(module, func_name)
        if not callable(factory):
            return defaults
        built = factory(session_token=session_token, defaults=defaults)
        if isinstance(built, PlatformServices):
            return built
    except Exception:
        # Extension hook failures should never break startup in community mode.
        return defaults

    return defaults
