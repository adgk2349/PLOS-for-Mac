from __future__ import annotations

from dataclasses import dataclass

from ..models import WorkspaceMemoryMode


@dataclass(slots=True)
class ResolvedMemoryPreferences:
    response_length: str = "long"
    show_citations: bool = True
    confirm_external_calls: bool = False
    prefer_action_suggestions: bool = True
    default_action_order: list[str] = None  # type: ignore[assignment]
    default_mode: str | None = None
    workspace_weights: dict[str, float] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.default_action_order is None:
            self.default_action_order = []
        if self.workspace_weights is None:
            self.workspace_weights = {}


def episodic_disabled(mode: WorkspaceMemoryMode) -> bool:
    """Return True when episodic/workspace memory should not be used."""
    return mode in {WorkspaceMemoryMode.DISABLED, WorkspaceMemoryMode.PINNED_ONLY}

