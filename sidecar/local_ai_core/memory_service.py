from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
import re
import time
from typing import Any, Callable, Literal

from .db import Database
from .models import (
    MemoryClearResponse,
    MemoryClearScope,
    MemoryEventRequest,
    MemoryEventResponse,
    PinnedMemoryItem,
    RelevantMemoryBundle,
    UserPreferenceItem,
    WorkspaceIdentity,
    WorkspaceMemoryMode,
)
from .response_composer import ResponseComposer


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class ResolvedMemoryPreferences:
    response_length: str = "medium"
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


def _episodic_disabled(mode: WorkspaceMemoryMode) -> bool:
    """Return True when episodic/workspace memory should not be used."""
    return mode in {WorkspaceMemoryMode.DISABLED, WorkspaceMemoryMode.PINNED_ONLY}


from .memory import service_mixins as _memory_service_mixins

class MemoryService(_memory_service_mixins.MemoryServiceMethodsMixin):
    _DIGEST_KEY = "conversation_digest_v1"
    _DIGEST_VERSION = "v1"
    _DIGEST_RECENT_TURNS_CAP = 8
    _DIGEST_TOPICS_CAP = 8
    _DIGEST_FACTS_CAP = 10
    _DIGEST_OPEN_LOOPS_CAP = 6
    _DIGEST_RECENT_TURN_MAX_CHARS = 220
    _DIGEST_ITEM_MAX_CHARS = 180
    _TOPIC_STOPWORDS = {
        "and",
        "the",
        "that",
        "this",
        "with",
        "from",
        "what",
        "where",
        "when",
        "which",
        "about",
        "please",
        "chat",
        "talk",
        "file",
        "files",
        "document",
        "documents",
        "summary",
        "summarize",
        "search",
        "find",
        "list",
        "all",
        "every",
        "안녕",
        "근데",
        "그냥",
        "근거",
        "파일",
        "문서",
        "자료",
        "요약",
        "정리",
        "검색",
        "찾아",
        "전부",
        "전체",
        "모두",
        "대화",
        "질문",
        "답변",
    }

    def __init__(self, db: Database):
        self._db = db
        self._last_inferred_refresh_by_workspace: dict[str, float] = {}
        self._digest_model_refresher: Callable[[str, dict[str, Any]], dict[str, Any] | None] | None = None

    def get_workspace_identity(self) -> WorkspaceIdentity:
        workspace = self._db.get_workspace()
        return self._db.get_workspace_identity(workspace)

    # ------------------------------------------------------------------
    # Session memory
    # ------------------------------------------------------------------


_memory_service_mixins.ResolvedMemoryPreferences = ResolvedMemoryPreferences
_memory_service_mixins._episodic_disabled = _episodic_disabled

