from __future__ import annotations

import logging
from typing import Any, Callable

from .db import Database
from .models import (
    WorkspaceIdentity,
)
from .memory.preferences import ResolvedMemoryPreferences
from .memory import service_mixins as _memory_service_mixins

logger = logging.getLogger(__name__)

class MemoryService(_memory_service_mixins.MemoryServiceMethodsMixin):
    _DIGEST_KEY = "conversation_digest_v1"
    _DIGEST_VERSION = "v1"
    # DB에 저장하는 최근 턴 수 (raw turns buffer before compression)
    _DIGEST_RECENT_TURNS_CAP = 20
    _DIGEST_TOPICS_CAP = 8
    _DIGEST_FACTS_CAP = 10
    _DIGEST_OPEN_LOOPS_CAP = 6
    _DIGEST_RECENT_TURN_MAX_CHARS = 300
    _DIGEST_ITEM_MAX_CHARS = 240
    # L1: MessageState에 원문 그대로 주입할 최근 턴 수 (user+assistant 각 1개 = 2개)
    _DIGEST_WINDOW_VERBATIM = 10
    # L2: rolling_summary 최대 글자수
    _DIGEST_ROLLING_SUMMARY_MAX_CHARS = 600
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
        self._vector_store: Any = None
        self._embedding_service: Any = None

    def set_dependencies(self, *, vector_store: Any, embedding_service: Any) -> None:
        self._vector_store = vector_store
        self._embedding_service = embedding_service

    def get_workspace_identity(self) -> WorkspaceIdentity:
        workspace = self._db.get_workspace()
        return self._db.get_workspace_identity(workspace)

    # ------------------------------------------------------------------
    # Session memory
    # ------------------------------------------------------------------
