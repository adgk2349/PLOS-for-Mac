from __future__ import annotations

from typing import Any

from .. import utils
from ..helpers.system.settings_sys_helpers import SettingsSysHelpers


class PipelineCompatDelegates:
    @staticmethod
    def _normalized_match_text(query: str) -> str:
        return utils._normalized_match_text(query)

    @staticmethod
    def _is_explicit_web_search_request(query: str) -> bool:
        return utils._is_explicit_web_search_request(query)

    @staticmethod
    def _should_auto_web_search(*, query: str, parsed_intent, last_context: dict | None) -> bool:
        return utils._should_auto_web_search(query=query, parsed_intent=parsed_intent, last_context=last_context)

    @staticmethod
    def _is_followup_web_search_request(*, query: str, last_context: dict | None) -> bool:
        return utils._is_followup_web_search_request(query=query, last_context=last_context)

    @staticmethod
    def _has_local_file_target_cues(query: str) -> bool:
        return utils._has_local_file_target_cues(query)

    @staticmethod
    def _has_explicit_retrieval_request(query: str, *, target_hint: str | None = None) -> bool:
        return utils._has_explicit_retrieval_request(query, target_hint=target_hint)

    @staticmethod
    def _contains_task_cues(lowered: str) -> bool:
        return utils._contains_task_cues(lowered)

    @staticmethod
    def _is_greeting_query(query: str) -> bool:
        return utils._is_greeting_query(query)

    @staticmethod
    def _is_brief_chat_query(query: str) -> bool:
        return utils._is_brief_chat_query(query)

    @staticmethod
    def _has_followup_context_signal(query: str) -> bool:
        return utils._has_followup_context_signal(query)

    @staticmethod
    def _has_strong_followup_context_signal(query: str) -> bool:
        return utils._has_strong_followup_context_signal(query)

    @staticmethod
    def _conversation_context_relevance(*, query: str, session_digest: dict[str, Any] | None, last_context: dict[str, Any] | None) -> float:
        return utils._conversation_context_relevance(query=query, session_digest=session_digest, last_context=last_context)

    @staticmethod
    def _conversation_context_budget_tokens(response_length: str, model_profile: str = "recommended") -> int:
        return utils._conversation_context_budget_tokens(response_length, model_profile=model_profile)

    @staticmethod
    def _estimate_context_tokens(text: str) -> int:
        return utils._estimate_context_tokens(text)

    @staticmethod
    def _system_memory_gb() -> int:
        return utils._system_memory_gb()

    @staticmethod
    def _model_size_b(reference: str) -> int | None:
        return utils._model_size_b(reference)

    @staticmethod
    def _extract_excluded_weeks(query: str) -> list[int]:
        return utils._extract_excluded_weeks(query)

    @staticmethod
    def _has_token_overlap(query_text: str, reference_text: str, *, min_overlap: int = 1) -> bool:
        return utils._has_token_overlap(query_text, reference_text, min_overlap=min_overlap)

    @staticmethod
    def _general_chat_assist_mode(*, parsed_intent, query):
        from ..helpers.chat.core_chat_helpers import CoreChatHelpers

        return CoreChatHelpers._general_chat_assist_mode(parsed_intent=parsed_intent, query=query)

    @staticmethod
    def _conversation_max_tokens(response_length: str, model_profile: str = "recommended", query: str = "") -> int:
        return SettingsSysHelpers.conversation_max_tokens(response_length, model_profile=model_profile, query=query)

    @staticmethod
    def _adaptive_response_length_for_query(
        *,
        query: str,
        base_response_length: str,
        explicit_web_search_request: bool,
        last_context: dict[str, Any] | None,
    ) -> str:
        _ = last_context
        text = str(query or "").lower()
        if explicit_web_search_request:
            return "medium"
        if any(token in text for token in ("자세히", "상세", "길게", "deep", "detail")):
            return "long"
        if any(token in text for token in ("한 줄", "짧게", "간단히", "brief", "short")):
            return "short"
        base = str(base_response_length or "medium").lower()
        return base if base in {"short", "medium", "long"} else "medium"

    @staticmethod
    def _conversation_session_summary(**kwargs):
        from ..helpers.format.formatting_helpers import FormattingHelpers

        return FormattingHelpers._conversation_session_summary(**kwargs)

    @staticmethod
    def _quality_rollup_summary(events):
        from ..helpers.format.formatting_helpers import FormattingHelpers

        return FormattingHelpers._quality_rollup_summary(events)

    @staticmethod
    def _filter_doc_ids_by_path_focus(doc_ids, metadata_map, focus_terms):
        from ..helpers.retrieval.retrieval_helpers import RetrievalHelpers

        return RetrievalHelpers({}).filter_doc_ids_by_path_focus(
            doc_ids=doc_ids,
            metadata_map=metadata_map,
            focus_terms=focus_terms,
        )

    @classmethod
    def _should_force_general_chat(cls, *, query: str, parsed_intent, last_context: dict | None = None) -> bool:
        if cls._is_greeting_query(query):
            return True
        if cls._should_auto_web_search(query=query, parsed_intent=parsed_intent, last_context=last_context):
            return True
        return False

    @classmethod
    def _should_apply_conversation_context(cls, query: str, has_session_digest: bool, has_last_context: bool) -> bool:
        if not has_session_digest and not has_last_context:
            return False
        if cls._is_greeting_query(query):
            return False
        return True

    @staticmethod
    def _looks_repetitive_conversation_output(*, query: str, answer: str, session_digest: dict | None, last_context: dict | None) -> bool:
        from ..helpers.chat.core_chat_helpers import CoreChatHelpers

        return CoreChatHelpers._looks_repetitive_conversation_output(
            query=query,
            answer=answer,
            session_digest=session_digest,
            last_context=last_context,
        )

    @staticmethod
    def _web_search_query_for_turn(*, query: str, last_context: dict | None, is_followup_web_search: bool) -> str:
        return utils._web_search_query_for_turn(
            query=query,
            last_context=last_context,
            is_followup_web_search=is_followup_web_search,
        )
