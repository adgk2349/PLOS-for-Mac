from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Callable

from ...language_utils import resolve_response_language
from ...models import LocalChatRequestV2, ReasoningIntent, WorkspaceIdentity, WorkspaceResponse
from .. import utils
from ..context import ReasoningContext, RelevantMemoryBundle
from ..helpers.web.general_chat_web_gate_helpers import GeneralChatWebGateHelpers


@dataclass(slots=True)
class PipelineRunContext:
    context: ReasoningContext
    memory: Any
    memory_bundle: RelevantMemoryBundle
    response_language: str
    session_id: str
    session_digest_text: str
    force_general_chat: bool
    web_auto_triggered: bool


class ContextLoader:
    def __init__(self, *, intent_parser, followup_resolver, digest_to_text: Callable[..., str]):
        self._intent_parser = intent_parser
        self._followup_resolver = followup_resolver
        self._digest_to_text = digest_to_text

    @staticmethod
    def _safe_recent_session_state(memory: Any, session_id: str) -> tuple[dict[str, Any], list[str], str | None, list[str], dict[str, Any] | None]:
        last_context_data: dict[str, Any] = {}
        last_candidates: list[str] = []
        last_selected_file: str | None = None
        last_actions: list[str] = []
        session_digest_payload: dict[str, Any] | None = None
        if memory is None:
            return last_context_data, last_candidates, last_selected_file, last_actions, session_digest_payload
        try:
            raw_last_context = memory.get_last_conversational_context(session_id)
            if isinstance(raw_last_context, dict):
                last_context_data = dict(raw_last_context)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            last_context_data = {}
        try:
            last_candidates = list(memory.get_last_candidate_set(session_id) or [])[:8]
        except (AttributeError, TypeError, ValueError, RuntimeError):
            last_candidates = []
        try:
            raw_selected = memory.get_last_selected_file(session_id)
            if isinstance(raw_selected, str) and raw_selected.strip():
                last_selected_file = raw_selected.strip()
        except (AttributeError, TypeError, ValueError, RuntimeError):
            last_selected_file = None
        try:
            last_actions = list(memory.get_last_shown_actions(session_id) or [])[:8]
        except (AttributeError, TypeError, ValueError, RuntimeError):
            last_actions = []
        try:
            raw_digest = memory.get_session_digest(session_id)
            if isinstance(raw_digest, dict):
                session_digest_payload = dict(raw_digest)
        except (AttributeError, TypeError, ValueError, RuntimeError):
            session_digest_payload = None
        return last_context_data, last_candidates, last_selected_file, last_actions, session_digest_payload

    @staticmethod
    def _empty_memory_bundle(workspace_identity: WorkspaceIdentity) -> RelevantMemoryBundle:
        return RelevantMemoryBundle(
            workspace_identity=workspace_identity,
            session_items=[],
            workspace_items=[],
            preference_items=[],
            episodic_items=[],
            pinned_items=[],
            semantic_memories=[],
        )

    @staticmethod
    def _room_memory_isolation_enabled() -> bool:
        raw = str(os.getenv("LOCAL_AI_ROOM_MEMORY_ISOLATION", "1") or "1").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @classmethod
    def _session_only_bundle(cls, *, memory_bundle: RelevantMemoryBundle, session_id: str) -> RelevantMemoryBundle:
        semantic_rows: list[dict[str, Any]] = []
        for row in list(getattr(memory_bundle, "semantic_memories", []) or []):
            if not isinstance(row, dict):
                continue
            scope = str(row.get("scope") or "").strip().lower()
            row_session = str(row.get("session_id") or "").strip()
            if scope == "session" or (row_session and row_session == session_id):
                semantic_rows.append(dict(row))
        pinned_rows = []
        for row in list(getattr(memory_bundle, "pinned_items", []) or []):
            scope = str(getattr(row, "scope", "") or "").strip().lower()
            if scope == "global":
                pinned_rows.append(row)
        return RelevantMemoryBundle(
            workspace_identity=memory_bundle.workspace_identity,
            session_items=list(getattr(memory_bundle, "session_items", []) or []),
            workspace_items=[],
            preference_items=[],
            episodic_items=[],
            pinned_items=pinned_rows,
            semantic_memories=semantic_rows,
        )

    @staticmethod
    def _is_conversational_path(path: str) -> bool:
        normalized = str(path or "").strip().lower()
        if not normalized:
            return False
        return (
            normalized.startswith("local_conversation")
            or normalized.startswith("external_web_search")
            or normalized == "session_web_memory_reused"
            or normalized.startswith("general_chat")
        )

    @classmethod
    def _should_route_continue_to_general_chat(
        cls,
        *,
        parsed_intent: Any,
        followup_resolution: Any,
        last_context: dict[str, Any],
        query: str,
    ) -> bool:
        if getattr(parsed_intent, "intent", None) != ReasoningIntent.CONTINUE_PREVIOUS_RESULT:
            return False

        text = str(query or "").strip().lower()
        if not text:
            return False
        followup_tokens = ("하나 더", "더 보여", "계속", "show more", "one more", "another")
        if not any(token in text for token in followup_tokens):
            return False

        entities = getattr(parsed_intent, "entities", None)
        if entities is not None:
            if getattr(entities, "file_names", None) or getattr(entities, "projects", None) or getattr(entities, "tags", None):
                return False

        path = str((last_context or {}).get("conversation_path") or "")
        if path and not cls._is_conversational_path(path):
            return False

        resolved_targets = list(getattr(followup_resolution, "resolved_target_files", []) or [])
        if resolved_targets and not cls._is_conversational_path(path):
            return False

        return True

    def load(self, *, req: LocalChatRequestV2, dependencies: dict[str, Any]) -> PipelineRunContext:
        db = dependencies.get("db")
        memory = dependencies.get("memory") or dependencies.get("memory_service")

        workspace = (
            db.get_workspace()
            if db
            else WorkspaceResponse(included_paths=[], excluded_paths=[], startup_profile="fast", default_mode="auto", updated_at=None)
        )
        workspace_identity = (
            memory.get_workspace_identity()
            if memory
            else WorkspaceIdentity(workspace_id="default", included_paths_hash="", version=1)
        )
        settings = db.get_settings() if db else None
        response_language = resolve_response_language(req.query, getattr(settings, "language", None) if settings else None)

        session_id = str(req.session_id or req.conversation_id or "default-session")
        parsed_intent = self._intent_parser.parse(query=req.query, mode=req.mode, workspace=workspace)
        if GeneralChatWebGateHelpers.is_explicit_web_search_request(req.query):
            parsed_intent = parsed_intent.model_copy(
                update={
                    "intent": ReasoningIntent.GENERAL_CHAT,
                    "operation": "chat",
                    "target": None,
                    "scope": "single",
                    "ambiguity": "clear",
                }
            )
        (
            last_context_data,
            last_candidates,
            last_selected_file,
            last_actions,
            session_digest_payload,
        ) = self._safe_recent_session_state(memory, session_id)

        followup_resolution = self._followup_resolver.resolve(
            query=req.query,
            parsed_intent=parsed_intent,
            mode=req.mode,
            last_context=last_context_data,
            last_candidates=last_candidates,
            last_selected_file=last_selected_file,
            last_actions=last_actions,
        )

        if self._should_route_continue_to_general_chat(
            parsed_intent=parsed_intent,
            followup_resolution=followup_resolution,
            last_context=last_context_data,
            query=req.query,
        ):
            parsed_intent = parsed_intent.model_copy(
                update={
                    "intent": ReasoningIntent.GENERAL_CHAT,
                    "operation": "chat",
                    "target": None,
                    "scope": "single",
                    "ambiguity": "clear",
                }
            )

        memory_bundle: RelevantMemoryBundle
        memory_prefs = None
        if memory and settings is not None:
            try:
                memory_bundle = memory.get_relevant_memory_bundle(
                    session_id=session_id,
                    workspace_id=workspace_identity.workspace_id,
                    intent=parsed_intent.intent.value,
                    related_file_ids=[],
                    query=req.query,
                )
                if (
                    self._room_memory_isolation_enabled()
                    and getattr(parsed_intent, "intent", None) == ReasoningIntent.GENERAL_CHAT
                ):
                    memory_bundle = self._session_only_bundle(memory_bundle=memory_bundle, session_id=session_id)
                memory_prefs = memory.resolve_preferences(memory_bundle)
            except (AttributeError, TypeError, ValueError, RuntimeError):
                memory_bundle = self._empty_memory_bundle(workspace_identity)
        else:
            memory_bundle = self._empty_memory_bundle(workspace_identity)

        session_digest_text = self._digest_to_text(
            digest=session_digest_payload,
            last_context=last_context_data,
            max_chars=900,
        )

        context = ReasoningContext(
            req=req,
            workspace=workspace,
            workspace_identity=workspace_identity,
            settings=settings,
            session_id=session_id,
            response_language=response_language,
            parsed_intent=parsed_intent,
            followup_resolution=followup_resolution,
            memory_bundle=memory_bundle,
            behavior_policy={},
            memory_prefs=memory_prefs,
            last_context=last_context_data,
            session_digest=session_digest_text or None,
            effective_query=req.query,
            force_web_search=False,
            session_digest_payload=session_digest_payload,
        )

        force_general_chat = False
        web_auto_triggered = False
        if GeneralChatWebGateHelpers.is_explicit_web_search_request(req.query):
            context.force_web_search = True
            force_general_chat = True
        elif GeneralChatWebGateHelpers.should_auto_web_search(
            query=req.query,
            parsed_intent=parsed_intent,
            last_context=last_context_data,
        ):
            context.force_web_search = True
            force_general_chat = True
            web_auto_triggered = True
            if context.parsed_intent.intent != ReasoningIntent.GENERAL_CHAT:
                context.parsed_intent = context.parsed_intent.model_copy(
                    update={
                        "intent": ReasoningIntent.GENERAL_CHAT,
                        "operation": "chat",
                        "target": None,
                        "scope": "single",
                        "ambiguity": "clear",
                    }
                )

        return PipelineRunContext(
            context=context,
            memory=memory,
            memory_bundle=memory_bundle,
            response_language=response_language,
            session_id=session_id,
            session_digest_text=session_digest_text,
            force_general_chat=force_general_chat,
            web_auto_triggered=web_auto_triggered,
        )
