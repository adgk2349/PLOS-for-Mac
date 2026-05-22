from __future__ import annotations

from collections import Counter, deque
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
from typing import Any

from ....composition.composer import ResponseComposer
from ....models import ExecutionResult, LocalEngine, MemoryEventType, ParsedIntent


class CoreChatPostHelpers:
    @staticmethod
    def _scope_target_clarification_prompt(*, response_language: str, operation: str) -> str:
        op = (operation or "find").lower()
        if response_language == "ko":
            if op == "summarize":
                return "좋아요. 어떤 자료를 기준으로 요약할까요? 파일명 일부나 폴더명을 한 번만 알려주세요."
            if op == "open":
                return "좋아요. 어떤 파일을 열면 될까요? 파일명 일부만 알려주시면 바로 찾을게요."
            return "좋아요. 어떤 대상을 기준으로 찾을까요? 예: 데통 파일 전부 / 자료구조 폴더 전부"
        if op == "summarize":
            return "Got it. Which material should I summarize? Share a file name fragment or folder name."
        if op == "open":
            return "Got it. Which file should I open? A partial file name is enough."
        return "Got it. What target should I search? Example: all files under a specific topic or folder."

    @staticmethod
    def _should_force_general_chat(
        *,
        query: str,
        parsed_intent: ParsedIntent,
        last_context: dict | None = None,
    ) -> bool:
        from ... import utils

        if utils._is_explicit_web_search_request(query):
            return True
        if utils._is_followup_web_search_request(query=query, last_context=last_context):
            return True
        if utils._looks_general_chat_query(query):
            return True
        if utils._is_greeting_query(query):
            return True
        if str(getattr(parsed_intent, "operation", "chat") or "chat") != "chat":
            return False
        return False

    async def _repair_repetitive_conversation_response(
        self,
        *,
        query: str,
        execution: ExecutionResult,
        session_digest: dict[str, Any] | None,
        last_context: dict[str, Any] | None,
        response_language: str,
        mode,
        workspace,
        settings,
        response_length: str,
    ) -> ExecutionResult | None:
        answer = str(execution.generated_text or "").strip()
        if not answer:
            return None
        if not self._looks_repetitive_conversation_output(
            query=query,
            answer=answer,
            session_digest=session_digest,
            last_context=last_context,
        ):
            return None

        repair_query = self._anti_repeat_query(
            query=query,
            previous_answer=answer,
            response_language=response_language,
        )
        repaired = await self._executor.execute_conversation_async(
            query=repair_query,
            mode=mode,
            startup_profile=workspace.startup_profile,
            engine=settings.local_engine,
            mlx_model_path=settings.mlx_model_path,
            llama_model_path=settings.llama_model_path,
            language_preference=settings.language,
            session_summary=None,
            max_tokens=self._conversation_max_tokens(
                response_length,
                model_profile=getattr(settings, "model_profile", "recommended"),
                query=query,
            ),
            timeout_seconds=float(os.getenv("LOCAL_AI_INFERENCE_TIMEOUT_SECONDS", "40")),
        )
        repaired_text = str(repaired.generated_text or "").strip()
        if not repaired_text:
            return None
        if self._looks_repetitive_conversation_output(
            query=query,
            answer=repaired_text,
            session_digest=session_digest,
            last_context=last_context,
        ):
            return None
        return repaired

    @staticmethod
    def _anti_repeat_query(*, query: str, previous_answer: str, response_language: str) -> str:
        prior = re.sub(r"\s+", " ", (previous_answer or "").strip())[:140]
        if response_language == "ko":
            return (
                f"{query}\n    \n    "
                "바로 답변만 작성해줘. "
                "직전 답변과 겹치는 표현은 피하고, 자연스러운 한국어 존댓말 한두 문장으로 답해줘. "
                "역할 라벨이나 규칙 문장은 쓰지 마.\n    "
                f"이전 답변 핵심: {prior}"
            )
        return (
            f"{query}\n    \n    "
            "Answer directly in one or two natural sentences. "
            "Avoid repeating wording from the previous answer. "
            "Do not output role labels or rule text.\n    "
            f"Previous answer core: {prior}"
        )

    @staticmethod
    def _looks_repetitive_conversation_output(
        *,
        query: str,
        answer: str,
        session_digest: dict[str, Any] | None,
        last_context: dict[str, Any] | None,
    ) -> bool:
        cleaned_answer = ResponseComposer._strip_instruction_leakage(answer or "")
        cleaned_answer = re.sub(r"\s+", " ", cleaned_answer).strip()
        if not cleaned_answer:
            return True

        if CoreChatPostHelpers._has_duplicate_sentences(cleaned_answer):
            return True

        previous_assistant_texts: list[str] = []
        digest = session_digest or {}
        raw_turns = digest.get("recent_turns")
        if isinstance(raw_turns, list):
            for entry in raw_turns:
                if not isinstance(entry, dict):
                    continue
                role = str(entry.get("role") or "").strip().lower()
                if role != "assistant":
                    continue
                text = str(entry.get("text") or "").strip()
                if text:
                    previous_assistant_texts.append(text)
        context = last_context or {}
        recent_summary = str(context.get("result_summary") or "").strip()
        if recent_summary:
            previous_assistant_texts.append(recent_summary)

        for prev in previous_assistant_texts[-3:]:
            if CoreChatPostHelpers._text_similarity(cleaned_answer, prev) >= 0.76:
                return True

        if CoreChatPostHelpers._text_similarity(cleaned_answer, query) >= 0.86 and len(cleaned_answer) <= 120:
            return True
        return False

    @staticmethod
    def _has_duplicate_sentences(text: str) -> bool:
        parts = [
            seg.strip()
            for seg in re.split(r"(?:\n    +|(?<=[.!?。！？])\s+)", text or "")
            if seg.strip()
        ]
        if len(parts) < 2:
            return False
        seen: set[str] = set()
        for seg in parts:
            key = re.sub(r"[^\w가-힣]+", "", seg).casefold()
            if not key:
                continue
            if key in seen:
                return True
            seen.add(key)
        return False

    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        a_tokens = set(re.findall(r"[A-Za-z가-힣0-9_]+", (a or "").lower()))
        b_tokens = set(re.findall(r"[A-Za-z가-힣0-9_]+", (b or "").lower()))
        if not a_tokens or not b_tokens:
            return 0.0
        inter = len(a_tokens.intersection(b_tokens))
        union = len(a_tokens.union(b_tokens))
        if union <= 0:
            return 0.0
        return inter / union

    @staticmethod
    def _conversation_quality_debug_from_detail(detail: str | None) -> dict[str, Any]:
        value = str(detail or "").strip()
        if not value:
            return {
                "korean_rewrite_used": False,
                "quality_repair_reason": "",
                "repair_triggered": False,
                "repair_success": False,
                "leak_blocked": False,
                "direct_first_applied": False,
                "question_count_after_postprocess": 0,
                "recommendation_shape": "",
            }
        lowered = value.lower()
        rewrite_used = "korean_rewrite_used=1" in lowered
        repair_triggered = "repair_triggered=1" in lowered
        repair_success = "repair_success=1" in lowered
        leak_blocked = "leak_blocked=1" in lowered
        match = re.search(r"quality_repair_reason=([a-z0-9_\-|]+)", lowered)
        reason = match.group(1) if match else ""
        direct_first_applied = "direct_first_applied=1" in lowered
        question_match = re.search(r"question_count_after_postprocess=([0-9]+)", lowered)
        question_count_after_postprocess = int(question_match.group(1)) if question_match else 0
        shape_match = re.search(r"recommendation_shape=([a-z0-9_\-]+)", lowered)
        recommendation_shape = shape_match.group(1) if shape_match else ""
        return {
            "korean_rewrite_used": rewrite_used,
            "quality_repair_reason": reason,
            "repair_triggered": repair_triggered,
            "repair_success": repair_success,
            "leak_blocked": leak_blocked,
            "direct_first_applied": direct_first_applied,
            "question_count_after_postprocess": question_count_after_postprocess,
            "recommendation_shape": recommendation_shape,
        }

    @staticmethod
    def _flag_enabled(value: str | None, *, default: bool) -> bool:
        if value is None:
            return default
        lowered = str(value).strip().lower()
        if lowered in {"0", "false", "off", "no", "n"}:
            return False
        if lowered in {"1", "true", "on", "yes", "y"}:
            return True
        return default

    @staticmethod
    def _parse_positive_int(
        value: str | None,
        *,
        default: int,
        minimum: int,
        maximum: int,
    ) -> int:
        try:
            parsed = int(str(value).strip()) if value is not None else default
        except Exception:
            parsed = default
        return max(minimum, min(maximum, parsed))

    @staticmethod
    def _load_recent_quality_events(path: Path, *, limit: int) -> list[dict[str, Any]]:
        if limit <= 0:
            return []
        recent: deque[dict[str, Any]] = deque(maxlen=limit)
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    raw = line.strip()
                    if not raw:
                        continue
                    try:
                        payload = json.loads(raw)
                    except Exception:
                        continue
                    if isinstance(payload, dict):
                        recent.append(payload)
        except Exception:
            return []
        return list(recent)

    def _record_conversation_quality_event(
        self,
        *,
        session_id: str,
        query: str,
        execution: ExecutionResult,
        metadata: dict[str, Any],
    ) -> dict[str, Any]:
        if not self._quality_log_enabled:
            return {}

        quality_repair_reason = str(metadata.get("quality_repair_reason") or "").strip().lower()
        payload = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "session_id": session_id,
            "engine": execution.engine_used.value if execution.engine_used else "none",
            "result_type": execution.result_type,
            "query_length": len(str(query or "")),
            "answer_length": len(str(execution.generated_text or "")),
            "korean_rewrite_used": bool(metadata.get("korean_rewrite_used", False)),
            "quality_repair_reason": quality_repair_reason,
            "assistive_retrieval_suppressed": bool(metadata.get("assistive_retrieval_suppressed", False)),
            "assist_mode": str(metadata.get("assist_mode") or "none"),
            "repair_triggered": bool(metadata.get("repair_triggered", False)),
            "repair_success": bool(metadata.get("repair_success", False)),
            "leak_blocked": bool(metadata.get("leak_blocked", False)),
        }

        self._quality_rollup_window.append(payload)
        try:
            self._quality_log_path.parent.mkdir(parents=True, exist_ok=True)
            with self._quality_log_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(payload, ensure_ascii=False) + "\n    ")
        except Exception:
            pass
        return self._quality_rollup_summary(list(self._quality_rollup_window))

    def _escalate_general_chat(
        self,
        *,
        query: str,
        mode,
        context_summary: str,
        last_context: dict,
        settings,
        allow_web_search: bool = False,
    ) -> tuple[ExecutionResult, str] | None:
        if self._providers is None:
            return None
        provider_order = ["anthropic", "openai"]
        mini_citations = self._context_citation_summaries(context_summary=context_summary, last_context=last_context)
        payload_query = query
        if context_summary:
            payload_query = f"{query}\n    \n    Session summary:\n    {context_summary}"
        for provider in provider_order:
            if not self._providers.provider_has_key(provider):
                continue
            endpoint = self._external_provider_endpoint(provider)
            try:
                result = self._providers.analyze_sync(
                    provider=provider,
                    query=payload_query,
                    mode=mode,
                    citations=mini_citations,
                    language_preference=settings.language,
                    allow_web_search=allow_web_search,
                )
            except Exception:
                continue
            if not result.answer.strip():
                continue
            trace_logs: list[str] = []
            if endpoint:
                trace_logs.append(f"retrieving:{endpoint}")
                trace_logs.append(f"retrieved:{endpoint}")
            execution = ExecutionResult(
                result_type="conversation",
                structured_payload={
                    "style": "general_chat",
                    "source": "external_escalated",
                    "provider": provider,
                    "ungrounded_allowed": True,
                },
                citations=[],
                tool_logs=[*trace_logs, f"external_escalated:{provider}"],
                generated_text=result.answer.strip(),
                engine_used=None,
                used_fallback=False,
                runtime_detail=f"external_escalated_provider={provider}",
            )
            return execution, provider
        return None

    @staticmethod
    def _runtime_error_execution(response_language: str, detail: str | None) -> ExecutionResult:
        return ExecutionResult(
            result_type="conversation",
            structured_payload={
                "style": "runtime_error",
                "reason": "conversation_engine_unavailable",
                "ungrounded_allowed": True,
                "offer_regenerate": True,
            },
            citations=[],
            tool_logs=[],
            generated_text="",
            engine_used=None,
            used_fallback=False,
            runtime_detail=detail,
        )

    @staticmethod
    def _is_16gb_tier_model(settings) -> bool:
        from ... import utils

        reference = ""
        try:
            if settings.local_engine == LocalEngine.MLX:
                reference = str(settings.mlx_model_path or "")
            else:
                reference = str(settings.llama_model_path or "")
        except Exception:
            reference = ""
        if not reference:
            return False
        size_b = utils._model_size_b(reference)
        if size_b is None:
            return False
        return size_b <= 8
