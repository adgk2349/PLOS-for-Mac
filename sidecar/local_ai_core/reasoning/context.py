from dataclasses import dataclass
from typing import Any

from ..models import (
    LocalChatRequestV2,
    ParsedIntent,
    RelevantMemoryBundle,
    SettingsModel,
    WorkspaceIdentity,
    WorkspaceResponse,
)
from ..nlu.followup_resolver import FollowUpResolution


@dataclass
class ReasoningContext:
    """
    Encapsulates all necessary state required for a ReasoningStrategy to execute.
    """
    req: LocalChatRequestV2
    workspace: WorkspaceResponse
    workspace_identity: WorkspaceIdentity
    settings: SettingsModel
    session_id: str
    response_language: str
    parsed_intent: ParsedIntent
    followup_resolution: FollowUpResolution
    memory_bundle: RelevantMemoryBundle
    behavior_policy: dict[str, Any]
    memory_prefs: Any
    last_context: dict[str, Any] | None
    session_digest: str | None
    effective_query: str
    force_web_search: bool = False
