from __future__ import annotations

import asyncio
import json
import os
import re
import socket
import time
import unicodedata
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, Optional
from urllib.parse import urlencode, urlparse
from urllib.request import Request, urlopen

from ... import utils
from ....models import (
    BehaviorPolicy,
    Citation,
    ComposedChatResponseV2,
    ExecutionResult,
    LocalPlan,
    ParsedIntent,
    PrivacyMode,
    ReasoningIntent,
    SuggestedActionKind,
    VerificationResult,
)
from ....nlu.followup_resolver import FollowUpResolution
from ...context import ReasoningContext
from ...message_state import MessageState
from ...executor_contract import bind_async_executor_contract, require_executor_methods
from ....web_retrieval import WebRetrievalReport, WebRetriever
from ....language_utils import detect_query_language, normalize_language_code
from ...helpers.web.general_chat_web_gate_helpers import GeneralChatWebGateHelpers
from ...helpers.web.general_chat_web_execution_helpers import GeneralChatWebExecutionHelpers
from ...helpers.chat.general_chat_recall_gate_helpers import GeneralChatRecallGateHelpers
from ...helpers.chat.general_chat_recall_execution_helpers import GeneralChatRecallExecutionHelpers
from ...helpers.chat.general_chat_conversation_execution_helpers import GeneralChatConversationExecutionHelpers
from ...answer_contract import (
    coerce_answer_type_hint,
    extract_contract_response,
    infer_answer_type_hint,
    validate_contract_response,
)


class GeneralChatConversationMixin:
    @staticmethod
    def _is_fast_chat_query(query: str) -> bool:
        text = str(query or "").strip()
        if not text:
            return True
        lowered = text.lower()
        token_count = len(re.findall(r"[A-Za-z가-힣0-9_]+", lowered))
        if token_count > 12:
            return False
        if any(token in lowered for token in ("코드", "python", "swift", "debug", "분석", "자세히", "explain", "analyze")):
            return False
        return True

    @staticmethod
    def _should_force_detailed_grounding(query: str) -> bool:
        lowered = str(query or "").strip().lower()
        if not lowered:
            return False
        detail_tokens = ("자세히", "상세", "깊게", "분석", "근거", "explain", "detail", "analyze")
        return any(token in lowered for token in detail_tokens)

    @staticmethod
    def _model_family_from_settings(settings: Any) -> str:
        model_ref = str(getattr(settings, "mlx_model_path", "") or getattr(settings, "llama_model_path", "") or "").lower()
        compact = re.sub(r"[^a-z0-9]+", "", model_ref)
        if "deepseek" in compact:
            return "deepseek"
        if "qwen" in compact:
            return "qwen"
        if "gemma" in compact:
            return "gemma"
        if "llama" in compact or "mistral" in compact:
            return "llama"
        return "default"

    @staticmethod
    def _model_family_prompt_hint(*, family: str, response_language: str) -> str:
        _ = family
        _ = response_language
        # Disable model-family hard rules; keep model-native behavior.
        return ""

    @staticmethod
    def _extract_memory_lifestyle(text: str) -> str:
        lowered = unicodedata.normalize("NFC", str(text or "").strip()).lower()
        if not lowered:
            return ""
        if "아침형" in lowered:
            return "아침형 인간"
        if "저녁형" in lowered or "야행성" in lowered:
            return "저녁형 성향"
        return ""

    @staticmethod
    def _contract_regen_attempts() -> int:
        try:
            raw = int(float(str(os.getenv("LOCAL_AI_CONTRACT_REGEN_ATTEMPTS", "2")).strip() or "2"))
            return max(0, min(4, raw))
        except Exception:
            return 2

    @classmethod
    def _contract_regen_prompt(
        cls,
        *,
        query: str,
        response_language: str,
        answer_type_hint: str,
        draft_answer: str,
        fail_reasons: list[str],
    ) -> str:
        return GeneralChatConversationExecutionHelpers.contract_regen_prompt(
            query=query,
            response_language=response_language,
            answer_type_hint=answer_type_hint,
            draft_answer=draft_answer,
            fail_reasons=fail_reasons,
        )

    async def _enforce_output_contract(
        self,
        *,
        executor: Any,
        context: ReasoningContext,
        execution: ExecutionResult,
        query: str,
        answer_type_hint: str,
        style_profile: dict[str, Any] | None,
    ) -> tuple[ExecutionResult, dict[str, Any]]:
        return await GeneralChatConversationExecutionHelpers.enforce_output_contract(
            self,
            executor=executor,
            context=context,
            execution=execution,
            query=query,
            answer_type_hint=answer_type_hint,
            style_profile=style_profile,
        )

    async def _run_conversation_inference(
        self,
        *,
        executor,
        query: str,
        context: ReasoningContext,
        max_tokens: int,
        generation_style: str = "conversation",
        sampling_overrides: dict[str, float | int] | None = None,
        timeout_seconds: float | None = None,
        session_summary_override: str | None = None,
        style_profile: dict[str, Any] | None = None,
        response_language_override: str | None = None,
        language_preference_override: str | None = None,
    ) -> ExecutionResult:
        # Disable timeout in conversation path to avoid premature runtime-error fallback
        # on low-spec hardware. Generation should continue until model returns.
        del timeout_seconds
        timeout = None
        executor = bind_async_executor_contract(executor)
        require_executor_methods(executor, "execute_conversation_async")
        explicit_response_language = normalize_language_code(response_language_override)
        explicit_language_preference = normalize_language_code(language_preference_override)
        query_majority_language = detect_query_language(query)
        # Conversation output should follow the current user message language first.
        effective_response_language = (
            query_majority_language
            or explicit_response_language
            or explicit_language_preference
            or normalize_language_code(context.response_language)
            or "en"
        )
        effective_language_preference = (
            query_majority_language
            or explicit_language_preference
            or explicit_response_language
            or normalize_language_code(context.settings.language)
            or normalize_language_code(context.response_language)
            or "en"
        )

        state = MessageState()
        reflection_text = getattr(context, "global_reflection", "") or ""
        if reflection_text:
            state.add_system(reflection_text)

        # Inject conversation history from session_digest into MessageState.
        # L2: rolling_summary (compressed older turns) → injected as system message first.
        # L1: recent verbatim turns → injected as user/assistant messages.
        # Default: keep last 4 turn-pairs (= 8 messages). Tunable via env var.
        _max_history_turns = int(str(os.getenv("LOCAL_AI_HISTORY_TURNS", "4")).strip() or "4")
        dangling_user_text = ""
        try:
            digest_obj = getattr(context, "session_digest_payload", None)
            if not digest_obj:
                raw_digest = getattr(context, "session_digest", None) or {}
                if isinstance(raw_digest, str):
                    import json as _json
                    try:
                        digest_obj = _json.loads(raw_digest)
                    except Exception:
                        digest_obj = {}
                else:
                    digest_obj = raw_digest or {}
            if isinstance(digest_obj, dict):
                rolling_summary = str(digest_obj.get("rolling_summary") or "").strip()
                if rolling_summary:
                    compact_summary = re.sub(r"\s+", " ", rolling_summary).strip()
                    if compact_summary:
                        state.add_system(f"Previous conversation summary: {compact_summary[:420]}")

                # L1: verbatim recent turns
                raw_turns = digest_obj.get("recent_turns") or []
                if isinstance(raw_turns, list) and raw_turns:
                    eligible = [
                        t for t in raw_turns
                        if isinstance(t, dict)
                        and str(t.get("role") or "").strip().lower() in {"user", "assistant"}
                        and str(t.get("text") or "").strip()
                    ]
                    # Keep the most recent N turn-pairs (2 messages per pair).
                    eligible = eligible[-(_max_history_turns * 2):]

                    # Merge consecutive user messages and consecutive assistant messages
                    # to keep the alternating structure valid for the chat template.
                    merged_eligible = []
                    for turn in eligible:
                        if not merged_eligible:
                            merged_eligible.append(dict(turn))
                        else:
                            last = merged_eligible[-1]
                            last_role = str(last.get("role") or "").strip().lower()
                            curr_role = str(turn.get("role") or "").strip().lower()
                            if last_role == curr_role:
                                last_text = str(last.get("text") or "").strip()
                                curr_text = str(turn.get("text") or "").strip()
                                last["text"] = last_text + "\n\n" + curr_text
                            else:
                                merged_eligible.append(dict(turn))
                    eligible = merged_eligible

                    # Ensure history ends on an assistant turn so the chat template
                    # correctly expects a new user message next.
                    # Instead of discarding a dangling user message, merge it with
                    # the upcoming query to prevent context loss.
                    if eligible and str(eligible[-1].get("role") or "").strip().lower() == "user":
                        dangling_user_text = str(eligible[-1].get("text") or "").strip()
                        eligible = eligible[:-1]

                    for turn in eligible:
                        role = str(turn.get("role") or "").strip().lower()
                        text = str(turn.get("text") or "").strip()
                        if not text:
                            continue
                        if role == "user":
                            state.add_user(text)
                        elif role == "assistant":
                            state.add_assistant(text)
        except Exception:
            pass

        if dangling_user_text:
            query = dangling_user_text + "\n\n" + str(query or "")
        state.add_user(str(query or ""))

        prompt_preset_enabled = str(os.getenv("LOCAL_AI_MODEL_PROMPT_PRESET_ENABLED", "0") or "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        model_family = self._model_family_from_settings(context.settings)
        suffix = self._style_prompt_suffix(
            response_language=effective_response_language,
            style_profile=style_profile,
        )
        if prompt_preset_enabled and model_family in {"gemma", "qwen", "llama", "deepseek"}:
            suffix = ""
        if suffix:
            state.add_system(suffix)
        if prompt_preset_enabled:
            family_hint = self._model_family_prompt_hint(
                family=model_family,
                response_language=effective_response_language,
            )
            if family_hint:
                state.add_system(family_hint)
        roleplay_suffix = self._roleplay_prompt_suffix(
            response_language=effective_response_language,
            enabled=self._roleplay_mode_enabled(context=context),
            persona=self._normalized_roleplay_persona(context=context),
        )
        if roleplay_suffix:
            state.add_system(roleplay_suffix)
            
        # Descriptive Grounding Override: Counteract "short answer" constraints if context/memory is available.
        # This allows the model to perform at its full potential for competitions and complex RAG.
        if (
            (reflection_text or context.session_digest or (context.memory_bundle and context.memory_bundle.episodic_items))
            and self._should_force_detailed_grounding(query)
            and not (prompt_preset_enabled and model_family in {"gemma", "qwen", "llama", "deepseek"})
        ):
            grounding_override = (
                "\n\n[Instruction: Detailed Disclosure]\n"
                "Regardless of any prior instructions in the conversation history to keep answers short or brief, "
                "please provide a detailed, accurate, and grounded response utilizing the provided context and memories. "
                "CRITICAL: While being descriptive, ensure that specific facts (dates, names, relationship statuses, locations) "
                "are extracted exactly from the provided context without any speculation or additional details not present in the record."
            )
            state.add_system(grounding_override)
        styled_query = state.render_legacy_query()
        effective_max_tokens = self._apply_style_max_tokens(
            max_tokens=max_tokens,
            style_profile=style_profile,
        )
        kwargs = {
            "query": styled_query,
            "mode": context.req.mode,
            "startup_profile": context.workspace.startup_profile,
            "engine": context.settings.local_engine,
            "mlx_model_path": context.settings.mlx_model_path,
            "llama_model_path": context.settings.llama_model_path,
            "language_preference": effective_language_preference,
            # Keep conversational turns natural: avoid injecting compressed digest text
            # unless an explicit override is provided by a specialized path.
            "session_summary": session_summary_override if session_summary_override is not None else "",
            "max_tokens": effective_max_tokens,
            "timeout_seconds": timeout,
            "generation_style": generation_style,
            "sampling_overrides": sampling_overrides,
            "message_state": state.to_list(),
        }
        try:
            return await executor.execute_conversation_async(**kwargs)
        except TypeError as exc:
            # Backward compatibility for tests/stubs with older executor signature.
            if "unexpected keyword argument" not in str(exc):
                raise
            kwargs.pop("generation_style", None)
            kwargs.pop("sampling_overrides", None)
            kwargs.pop("message_state", None)
            return await executor.execute_conversation_async(**kwargs)

    @staticmethod
    def _generation_retry_max_attempts(*, context: ReasoningContext | None = None) -> int:
        adaptive = getattr(context, "adaptive_runtime", None) if context is not None else None
        if isinstance(adaptive, dict):
            try:
                override_value = int(float(str(adaptive.get("gen_retry_max_attempts", "")).strip()))
                if override_value > 0:
                    return max(1, min(6, override_value))
            except Exception:
                pass
        try:
            return max(1, min(6, int(str(os.getenv("GEN_RETRY_MAX_ATTEMPTS", "2")).strip() or "2")))
        except Exception:
            return 2

    @staticmethod
    def _generation_retry_total_budget_ms() -> int:
        try:
            return max(2000, min(120000, int(str(os.getenv("GEN_RETRY_TOTAL_BUDGET_MS", "30000")).strip() or "30000")))
        except Exception:
            return 30000

    @staticmethod
    def _retry_until_cancel_enabled() -> bool:
        # Default to keep-generating-until-user-cancels mode for local chat.
        # This avoids synthetic timeout fallback messages on slower devices.
        raw = str(os.getenv("GEN_RETRY_UNTIL_CANCEL", "1") or "1").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def _no_fallback_mode_enabled() -> bool:
        raw = str(
            os.getenv(
                "GEN_RETRY_NO_FALLBACK",
                os.getenv("LOCAL_AI_NO_FALLBACK_MODE", "0"),
            ) or "0"
        ).strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def _retry_until_cancel_max_attempts() -> int:
        try:
            return max(1, min(64, int(str(os.getenv("GEN_RETRY_UNTIL_CANCEL_MAX_ATTEMPTS", "8")).strip() or "8")))
        except Exception:
            return 8

    @staticmethod
    def _retry_until_cancel_max_seconds() -> float:
        try:
            return max(5.0, min(3600.0, float(str(os.getenv("GEN_RETRY_UNTIL_CANCEL_MAX_SECONDS", "600")).strip() or "600")))
        except Exception:
            return 600.0

    @staticmethod
    def _generation_backoff_steps() -> list[float]:
        raw = str(os.getenv("GEN_RETRY_TOKEN_BACKOFF_STEPS", "1.0,0.7") or "1.0,0.7").strip()
        values: list[float] = []
        for token in raw.split(","):
            item = token.strip()
            if not item:
                continue
            try:
                values.append(float(item))
            except Exception:
                continue
        if not values:
            values = [1.0, 0.7]
        if values[0] < 0.95:
            values.insert(0, 1.0)
        return [max(0.2, min(1.0, v)) for v in values[:4]]

    @staticmethod
    def _retry_timeouts(*, total_budget_ms: int, slots: int) -> list[float]:
        base = max(1, int(total_budget_ms))
        count = max(1, slots)
        per = max(3.0, (base / count) / 1000.0)
        return [per] * count

    @staticmethod
    def _retry_min_tokens_floor() -> int:
        try:
            raw = int(float(str(os.getenv("GEN_RETRY_MIN_TOKENS", "128")).strip() or "128"))
            return max(96, min(256, raw))
        except Exception:
            return 128

    @staticmethod
    def _retry_tokens_cap() -> int:
        try:
            raw = int(float(str(os.getenv("GEN_RETRY_MAX_TOKENS_CAP", "2048")).strip() or "2048"))
            return max(512, min(8192, raw))
        except Exception:
            return 2048

    @staticmethod
    def _conversation_decode_profile(*, query: str) -> str:
        lowered = str(query or "").strip().lower()
        if not lowered:
            return "balanced"
        coding_pattern = r"(?is)\b(code|python|swift|javascript|typescript|java|c\+\+|algorithm|leetcode|debug|bug|함수|코드|문제\s*풀|풀이)\b"
        if re.search(coding_pattern, lowered):
            return "coding"
        concise_tokens = ("짧게", "간단히", "한 줄", "요약", "brief", "short", "quick")
        if any(token in lowered for token in concise_tokens):
            return "concise"
        analytic_tokens = ("자세히", "상세", "깊게", "비교", "why", "how", "detail", "explain")
        if any(token in lowered for token in analytic_tokens):
            return "analytic"
        return "balanced"

    @staticmethod
    def _resolve_effective_style_profile(
        *,
        context: ReasoningContext,
    ) -> tuple[dict[str, Any] | None, str, str]:
        turn_profile = utils._normalize_response_style_profile(
            getattr(context, "turn_response_style_profile", None)
        )
        if turn_profile is not None:
            return turn_profile, "turn_explicit", "current_turn_explicit"
        session_profile = utils._normalize_response_style_profile(
            getattr(context, "response_style_profile", None)
        )
        if session_profile is not None:
            return session_profile, "session", "session_profile"
        global_profile = utils._normalize_response_style_profile(
            getattr(context, "global_response_style_profile", None)
        )
        if global_profile is not None:
            return global_profile, "global_preference", "global_profile"
        return None, "default", "default"

    @staticmethod
    def _decode_profile_with_style(
        *,
        fallback_profile: str,
        style_profile: dict[str, Any] | None,
    ) -> str:
        if fallback_profile == "coding":
            return "coding"
        normalized = utils._normalize_response_style_profile(style_profile)
        if normalized is None:
            return fallback_profile
        tone = str(normalized.get("tone") or "").strip().lower()
        if tone == "direct":
            return "concise"
        if tone == "analytic":
            return "analytic"
        return "balanced"

    @staticmethod
    def _apply_style_max_tokens(*, max_tokens: int, style_profile: dict[str, Any] | None) -> int:
        normalized = utils._normalize_response_style_profile(style_profile)
        if normalized is None:
            return int(max_tokens)
        base = max(72, int(max_tokens))
        verbosity = str(normalized.get("verbosity") or "").strip().lower()
        if verbosity == "short":
            return min(base, 420)
        if verbosity == "medium":
            return min(base, 960)
        return base

    @staticmethod
    def _style_prompt_suffix(
        *,
        response_language: str,
        style_profile: dict[str, Any] | None,
    ) -> str:
        normalized = utils._normalize_response_style_profile(style_profile)
        if normalized is None:
            return ""
        verbosity = str(normalized.get("verbosity") or "medium")
        tone = str(normalized.get("tone") or "balanced")
        fmt = str(normalized.get("format") or "paragraph")
        if response_language == "ko":
            return (
                "\n\n[응답 스타일 프로필]\n"
                f"- 길이: {verbosity}\n"
                f"- 톤: {tone}\n"
                f"- 형식: {fmt}\n"
                "- 규칙: 질의가 모호할 때만 재질문하고, 명확하면 바로 답변하세요."
            )
        if response_language == "ja":
            return (
                "\n\n[応答スタイルプロファイル]\n"
                f"- 長さ: {verbosity}\n"
                f"- トーン: {tone}\n"
                f"- 形式: {fmt}\n"
                "- ルール: 曖昧な場合のみ確認質問し、明確なら直接回答すること。"
            )
        return (
            "\n\n[Response style profile]\n"
            f"- Verbosity: {verbosity}\n"
            f"- Tone: {tone}\n"
            f"- Format: {fmt}\n"
            "- Rule: ask follow-up only when ambiguous; otherwise answer directly."
        )

    @staticmethod
    def _roleplay_mode_enabled(*, context: ReasoningContext) -> bool:
        try:
            return bool(getattr(context.req, "roleplay_mode", False))
        except Exception:
            return False

    @staticmethod
    def _normalized_roleplay_persona(*, context: ReasoningContext) -> str:
        try:
            raw = str(getattr(context.req, "roleplay_persona", "") or "").strip()
        except Exception:
            raw = ""
        if not raw:
            return ""
        cleaned = re.sub(r"\s+", " ", raw).strip()
        return cleaned[:48]

    @staticmethod
    def _roleplay_prompt_suffix(*, response_language: str, enabled: bool, persona: str) -> str:
        if not enabled:
            return ""
        if response_language == "ko":
            persona_line = f"- 캐릭터: {persona}\n" if persona else ""
            return (
                "\n\n[역할극 모드]\n"
                f"{persona_line}"
                "- 규칙: 캐릭터 말투/관점을 유지하세요.\n"
                "- 규칙: 메타 발언(역할을 깨는 설명)은 하지 마세요.\n"
                "- 규칙: 사용자가 역할극 종료를 말하기 전까지 동일 캐릭터를 유지하세요.\n"
                "- 규칙: 질문이 모호해도 캐릭터를 유지한 채 짧게 확인 질문하세요."
            )
        if response_language == "ja":
            persona_line = f"- キャラクター: {persona}\n" if persona else ""
            return (
                "\n\n[ロールプレイモード]\n"
                f"{persona_line}"
                "- ルール: キャラクターの口調と視点を維持すること。\n"
                "- ルール: 役割を壊すメタ説明はしないこと。\n"
                "- ルール: ユーザーが終了を明示するまで同じキャラクターを維持すること。\n"
                "- ルール: 曖昧な質問でも、キャラを保った短い確認質問にすること。"
            )
        persona_line = f"- Character: {persona}\n" if persona else ""
        return (
            "\n\n[Roleplay mode]\n"
            f"{persona_line}"
            "- Rule: stay in-character in tone and perspective.\n"
            "- Rule: avoid meta statements that break character.\n"
            "- Rule: keep the same character until the user explicitly ends roleplay.\n"
            "- Rule: if ambiguous, ask a short in-character clarification."
        )

    @staticmethod
    def _conversation_answer_ready(execution: ExecutionResult | None) -> bool:
        if execution is None:
            return False
        if execution.result_type != "conversation":
            return False
        if execution.used_fallback:
            return False
        text = str(execution.generated_text or "").strip()
        if not text:
            return False
        lowered = text.lower()
        if re.search(r"</?(conversation_memory|runtime_context|runtime_context_notes|response_style_profile)\b", lowered):
            return False
        if any(marker in lowered for marker in ("[응답]", "[최종 응답]", "[response]", "[final response]")):
            return False
        if lowered in {"응답", "response", "[응답].", "[response]."}:
            return False
        visible = re.sub(r"[\s`<>\[\]/:;,.!?_\-]+", "", text)
        if len(visible) < 2:
            return False
        return True

    async def _stabilize_conversation_completion(
        self,
        *,
        executor: Any,
        context: ReasoningContext,
        query: str,
        execution: ExecutionResult,
        base_max_tokens: int,
        timeout_seconds: float,
        session_summary_override: str | None,
        style_profile: dict[str, Any] | None,
    ) -> ExecutionResult:
        return await GeneralChatConversationExecutionHelpers.stabilize_conversation_completion(
            self,
            executor=executor,
            context=context,
            query=query,
            execution=execution,
            base_max_tokens=base_max_tokens,
            timeout_seconds=timeout_seconds,
            session_summary_override=session_summary_override,
            style_profile=style_profile,
        )

    @staticmethod
    def _general_clarify_focus(*, query: str, last_context: dict[str, Any] | None) -> str:
        lowered = str(query or "").strip().lower()
        if not lowered:
            return "target"
        if re.search(r"\b\d{1,2}\s*(월|month|months?)\s*쯤", lowered) and not re.search(r"\b(20\d{2}|19\d{2})\b", lowered):
            return "time"
        if any(token in lowered for token in ("그때", "그거", "이거", "저거", "that", "this", "previous", "similar")):
            overlap = 0.0
            if isinstance(last_context, dict):
                context_text = " ".join(
                    [
                        str(last_context.get("last_user_query") or ""),
                        str(last_context.get("result_summary") or ""),
                    ]
                )
                q_terms = {token for token in re.findall(r"[A-Za-z가-힣0-9_]{2,32}", lowered)}
                c_terms = {token for token in re.findall(r"[A-Za-z가-힣0-9_]{2,32}", context_text.lower())}
                if q_terms and c_terms:
                    overlap = len(q_terms.intersection(c_terms)) / max(1, len(q_terms))
            if overlap < 0.2:
                return "target"
        return "scope"

    @staticmethod
    def _default_last_resort_clarify_question(
        *,
        query: str,
        response_language: str,
        last_context: dict[str, Any] | None = None,
    ) -> str:
        trimmed = str(query or "").strip()
        focus = GeneralChatConversationMixin._general_clarify_focus(query=query, last_context=last_context)
        if response_language == "ja":
            if focus == "time":
                return "時期（年/月/日）を1つだけ具体化していただけますか？"
            if focus == "target":
                return "対象を1つだけ具体化していただけますか？"
            if trimmed:
                return f"「{trimmed}」について、対象や条件を1つだけ具体的に教えていただけますか？"
            return "対象や条件を1つだけ具体的に教えていただけますか？"
        if response_language == "en":
            if focus == "time":
                return "Can you specify one time hint (year/month/date)?"
            if focus == "target":
                return "Can you specify one concrete target so I can answer precisely?"
            if trimmed:
                return f"For \"{trimmed}\", can you specify one concrete target or condition?"
            return "Can you specify one concrete target or condition?"
        if focus == "time":
            return "시점(연/월/일) 한 가지만 더 알려주실래요?"
        if focus == "target":
            return "대상이 되는 항목을 1개만 구체적으로 알려주실래요?"
        if trimmed:
            return f"\"{trimmed}\"에서 대상이나 조건 한 가지만 더 구체적으로 알려주실래요?"
        return "대상이나 조건 한 가지만 더 구체적으로 알려주실래요?"

    @staticmethod
    def _topic_bridge_prefix(*, response_language: str) -> str:
        if response_language == "ko":
            return "주제를 바꿔 말씀드리면"
        if response_language == "ja":
            return "話題を変えると"
        return "Moving on to a new topic,"

    async def _run_conversation_with_recovery(
        self,
        *,
        executor: Any,
        context: ReasoningContext,
        query: str,
        base_max_tokens: int,
        style_profile: dict[str, Any] | None,
        session_summary_override: str | None = None,
        generation_style: str = "conversation",
    ) -> tuple[ExecutionResult, dict[str, Any]]:
        return await GeneralChatConversationExecutionHelpers.run_conversation_with_recovery(
            self,
            executor=executor,
            context=context,
            query=query,
            base_max_tokens=base_max_tokens,
            style_profile=style_profile,
            session_summary_override=session_summary_override,
            generation_style=generation_style,
        )

    @staticmethod
    def _normalize_space(text: str) -> str:
        return " ".join(str(text or "").split()).strip()

    @staticmethod
    def _conversation_query_with_context(
        *,
        query: str,
        response_language: str,
        followup_resolution: FollowUpResolution | None,
        last_context: dict[str, Any] | None,
    ) -> str:
        text = str(query or "").strip()
        _ = response_language
        _ = followup_resolution
        _ = last_context
        # Keep user query unchanged. Context behavior should be controlled by
        # hidden system prompt/session-memory, not by rewriting user text.
        return text
