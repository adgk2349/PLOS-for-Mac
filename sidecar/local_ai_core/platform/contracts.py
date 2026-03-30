from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from fastapi import Request
from ..models import SystemFilePermission


class AuthProvider(Protocol):
    def verify_request(self, request: Request) -> None: ...


class LicenseGate(Protocol):
    def is_allowed(self, *, feature: str, context: dict[str, Any] | None = None) -> bool: ...


class AuditSink(Protocol):
    def emit(self, *, event: str, payload: dict[str, Any]) -> None: ...


class PolicyProvider(Protocol):
    def get_policy(self) -> dict[str, Any]: ...


@runtime_checkable
class SystemToolProvider(Protocol):
    def spotlight_search(self, query: str) -> list[str]: ...
    def get_metadata(self, path: str) -> dict[str, Any]: ...
    def execute_command(self, command: str, permission_level: SystemFilePermission) -> str: ...


@dataclass(frozen=True)
class PlatformServices:
    auth_provider: AuthProvider
    license_gate: LicenseGate
    audit_sink: AuditSink
    policy_provider: PolicyProvider
    system_tools: SystemToolProvider | None = None
