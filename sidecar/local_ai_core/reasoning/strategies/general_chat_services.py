from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any, Protocol

from ..helpers.web.general_chat_web_gate_helpers import GeneralChatWebGateHelpers
from ..conversation_context_policy import resolve_conversation_context_policy


class WebSearchGate(Protocol):
    def is_followup_web_search_request(self, *, query: str, last_context: dict[str, Any] | None) -> bool: ...


class MemoryRecallRouter(Protocol):
    def has_memory_recall_cue(self, query: str) -> bool: ...
    def recall_from_fact_store(
        self,
        *,
        query: str,
        response_language: str,
        memory_bundle: Any,
    ) -> dict[str, str]: ...

class ConversationInputBuilder(Protocol):
    def build(
        self,
        *,
        context: Any,
        runtime_context: Any,
        runtime_notes: list[Any],
    ) -> tuple[str, str | None]: ...


@dataclass(slots=True)
class DefaultWebSearchGate:
    def is_followup_web_search_request(self, *, query: str, last_context: dict[str, Any] | None) -> bool:
        return GeneralChatWebGateHelpers.is_followup_web_search_request(
            query=query,
            last_context=last_context,
        )


@dataclass(slots=True)
class DefaultMemoryRecallRouter:
    strategy: Any

    def has_memory_recall_cue(self, query: str) -> bool:
        return bool(self.strategy._has_memory_recall_cue(query))

    def recall_from_fact_store(
        self,
        *,
        query: str,
        response_language: str,
        memory_bundle: Any,
    ) -> dict[str, str]:
        return self.strategy._memory_recall_response_from_fact_store(
            query=query,
            response_language=response_language,
            memory_bundle=memory_bundle,
        )


@dataclass(slots=True)
class DefaultConversationInputBuilder:
    strategy: Any

    def build(
        self,
        *,
        context: Any,
        runtime_context: Any,
        runtime_notes: list[Any],
    ) -> tuple[str, str | None]:
        followup_resolution = getattr(context, "followup_resolution", None)
        conversation_query = self.strategy._conversation_query_with_context(
            query=context.req.query,
            response_language=context.response_language,
            followup_resolution=followup_resolution,
            last_context=context.last_context,
        )

        memory_injection_enabled = str(
            os.getenv("LOCAL_AI_CONVERSATION_MEMORY_CONTEXT_INJECTION_ENABLED", "0") or "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if memory_injection_enabled:
            memory_context = self.strategy._build_conversation_memory_context(
                context.memory_bundle,
                response_language=context.response_language,
            )
            if memory_context:
                conversation_query = f"{memory_context}\n{conversation_query}"

        followup_hint_enabled = str(
            os.getenv("LOCAL_AI_CONVERSATION_FOLLOWUP_HINT_ENABLED", "0") or "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        followup_hint = ""
        if followup_hint_enabled:
            followup_hint = self.strategy._followup_memory_hint(
                query=context.req.query,
                response_language=context.response_language,
                followup_resolution=followup_resolution,
                last_context=context.last_context,
            )
        context_policy = resolve_conversation_context_policy(
            startup_profile=getattr(getattr(context, "workspace", None), "startup_profile", "RECOMMENDED"),
            model_path=(
                getattr(getattr(context, "settings", None), "mlx_model_path", None)
                or getattr(getattr(context, "settings", None), "llama_model_path", None)
            ),
            digest=getattr(context, "session_digest_payload", None),
        )
        # MessageState already carries digest history for chat-template engines.
        # Only use the compact summary as a legacy-prompt fallback when history
        # is unavailable; runtime context and explicit hints still pass through.
        digest_fallback = context.session_digest if context_policy.inject_digest_into_legacy_prompt else ""
        session_summary_override = self.strategy._merge_session_summary_with_hint(
            session_summary=digest_fallback,
            followup_hint=followup_hint,
        )
        session_summary_override = self.strategy._merge_session_summary_with_runtime_context(
            session_summary=session_summary_override,
            multimodal_context=runtime_context,
            multimodal_notes=runtime_notes,
        )
        return conversation_query, session_summary_override
