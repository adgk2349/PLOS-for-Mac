from __future__ import annotations

from ....models import ExecutionResult, PrivacyMode
from ... import utils


class GeneralChatWebGateHelpers:
    @staticmethod
    def is_explicit_web_search_request(query: str) -> bool:
        return bool(utils._is_explicit_web_search_request(query))

    @staticmethod
    def is_followup_web_search_request(*, query: str, last_context: dict | None) -> bool:
        return bool(utils._is_followup_web_search_request(query=query, last_context=last_context))

    @staticmethod
    def should_auto_web_search(*, query: str, parsed_intent, last_context: dict | None) -> bool:
        return bool(utils._should_auto_web_search(query=query, parsed_intent=parsed_intent, last_context=last_context))

    @staticmethod
    def blocked_execution(
        *,
        response_language: str,
        privacy_mode: PrivacyMode,
        hybrid_web_search_enabled: bool,
    ) -> ExecutionResult:
        if response_language == "ko":
            blocked_text = (
                "현재 프라이버시 모드가 로컬 전용이라 인터넷 검색을 실행할 수 없습니다. 설정에서 하이브리드로 전환해 주세요."
                if privacy_mode == PrivacyMode.LOCAL_ONLY
                else "하이브리드 모드지만 웹검색(인터넷 경로)이 꺼져 있어 인터넷 검색을 실행할 수 없습니다. 프라이버시 설정에서 웹검색 허용을 켜주세요."
            )
        else:
            blocked_text = (
                "Internet search is blocked because privacy mode is LOCAL_ONLY. Switch to HYBRID in settings."
                if privacy_mode == PrivacyMode.LOCAL_ONLY
                else "Internet search is disabled in current privacy settings. Enable hybrid web search."
            )

        # Guard rail: if this helper is called in an enabled state by mistake, still
        # preserve the same blocked payload shape for compatibility.
        if privacy_mode not in {PrivacyMode.LOCAL_ONLY, PrivacyMode.HYBRID}:
            blocked_text = "Internet search is disabled in current privacy settings."
        elif privacy_mode == PrivacyMode.HYBRID and bool(hybrid_web_search_enabled):
            blocked_text = "Internet search is disabled in current privacy settings."

        return ExecutionResult(
            result_type="conversation",
            structured_payload={"web_path": "blocked", "ungrounded_allowed": True},
            citations=[],
            tool_logs=["web_search:blocked:privacy"],
            generated_text=blocked_text,
            engine_used=None,
            used_fallback=False,
            runtime_detail="web_search_blocked",
        )

