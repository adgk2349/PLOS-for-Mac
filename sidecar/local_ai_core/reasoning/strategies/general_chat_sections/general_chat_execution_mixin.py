from __future__ import annotations

import asyncio
import json
import os
import re
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


class GeneralChatExecutionMixin:
    @staticmethod
    def _text_similarity(a: str, b: str) -> float:
        a_tokens = set(re.findall(r"[A-Za-z가-힣0-9_]+", str(a or "").lower()))
        b_tokens = set(re.findall(r"[A-Za-z가-힣0-9_]+", str(b or "").lower()))
        if not a_tokens or not b_tokens:
            return 0.0
        inter = len(a_tokens.intersection(b_tokens))
        union = len(a_tokens.union(b_tokens))
        return inter / union if union > 0 else 0.0

    @classmethod
    def _looks_stale_prev_user_echo(
        cls,
        *,
        answer: str,
        query: str,
        last_context: dict[str, Any] | None,
    ) -> bool:
        if not isinstance(last_context, dict):
            return False
        prev_user = str(last_context.get("last_user_query") or "").strip()
        if not prev_user:
            return False
        ans = str(answer or "").strip()
        cur = str(query or "").strip()
        if not ans or not cur:
            return False
        sim_prev = cls._text_similarity(ans, prev_user)
        sim_curr = cls._text_similarity(ans, cur)
        return sim_prev >= 0.62 and sim_curr <= 0.52

    def handles_intent(self, intent: ParsedIntent, followup: FollowUpResolution | None) -> bool:
        return intent.intent == ReasoningIntent.GENERAL_CHAT

    async def execute(
        self,
        *,
        context: ReasoningContext,
        dependencies: dict[str, Any],
    ) -> ComposedChatResponseV2:
        executor = dependencies["executor"]
        composer = dependencies["composer"]

        plan = LocalPlan(
            plan_type="conversation",
            selected_files=[],
            selected_chunks=[],
            response_strategy="conversational_assistant",
            allowed_actions=[SuggestedActionKind.ASK_FOLLOWUP],
            external_reasoning_needed=False,
        )

        execution: ExecutionResult | None = None
        conversation_path = "local_conversation"
        is_local = True
        response_length = getattr(context.memory_prefs, "response_length", "long") if context.memory_prefs else "long"
        model_profile = str(getattr(context.settings, "model_profile", "recommended") or "recommended")
        max_tokens = 2048 if response_length == "long" else 1280 if response_length == "medium" else 640
        adaptive_scale_override: float | None = None
        adaptive_runtime = getattr(context, "adaptive_runtime", None)
        if isinstance(adaptive_runtime, dict):
            try:
                adaptive_scale_override = float(str(adaptive_runtime.get("conversation_max_tokens_scale", "")).strip())
            except Exception:
                adaptive_scale_override = None
        sys_helpers = dependencies.get("sys_helpers")
        if sys_helpers is not None:
            try:
                max_tokens = int(
                    sys_helpers.conversation_max_tokens(
                        response_length,
                        model_profile=model_profile,
                        query=context.req.query,
                        adaptive_scale_override=adaptive_scale_override,
                    )
                )
            except Exception:
                pass
        effective_style_profile, style_profile_source, style_override_reason = self._resolve_effective_style_profile(
            context=context
        )
        roleplay_mode = self._roleplay_mode_enabled(context=context)
        roleplay_persona = self._normalized_roleplay_persona(context=context)
        max_tokens = self._apply_style_max_tokens(
            max_tokens=max_tokens,
            style_profile=effective_style_profile,
        )
        answer_type_hint = self._answer_type_hint(
            query=context.req.query,
            request_hint=getattr(context.req, "answer_type_hint", None),
        )
        recall_answer_type_hint = self._recall_answer_type_hint(
            query=context.req.query,
            request_hint=getattr(context.req, "answer_type_hint", None),
        )
        web_memory_for_metadata: dict[str, Any] = {}
        web_memory_reused = False
        web_memory_rank_score = 0.0
        last_context = context.last_context if isinstance(context.last_context, dict) else {}
        memory_service = dependencies.get("memory") or dependencies.get("memory_service")
        freshness_sensitive_query = utils._is_freshness_sensitive_query(context.req.query)
        web_gate = getattr(self, "web_search_gate", None)
        if web_gate is not None and hasattr(web_gate, "is_followup_web_search_request"):
            followup_web_search = bool(
                web_gate.is_followup_web_search_request(
                    query=context.req.query,
                    last_context=last_context,
                )
            )
        else:
            followup_web_search = GeneralChatWebGateHelpers.is_followup_web_search_request(
                query=context.req.query,
                last_context=last_context,
            )
        
        # Calculate routing scores to detect social/casual intent
        routing_scores = utils._intent_routing_scores(
            query=context.req.query,
            parsed_intent=context.parsed_intent,
            last_context=last_context
        )
        conversational_score = routing_scores.get("conversational_chat", 0.0)
        social_intent = conversational_score >= 0.7
        decode_profile_hint = self._conversation_decode_profile(query=context.req.query)
        # Latency-oriented token cap by query shape.
        # Keep answers natural while preventing oversized decode budgets for simple turns.
        if decode_profile_hint == "concise":
            max_tokens = min(max_tokens, 320)
        elif social_intent:
            max_tokens = min(max_tokens, 512)
        elif decode_profile_hint == "balanced":
            if model_profile.lower() != "advanced":
                max_tokens = min(max_tokens, 768)
        generation_style = "casual" if social_intent else "conversation"
        
        # Routing authority should primarily come from context-loader gates.
        web_search_triggered = bool(context.force_web_search)
        if roleplay_mode:
            web_search_triggered = False

        runtime_context = getattr(context.req, "multimodal_context", None)
        runtime_notes = getattr(context.req, "multimodal_notes", [])
        runtime_context_qa_mode = self._is_runtime_context_qa_mode(
            query=context.req.query,
            multimodal_context=runtime_context,
            multimodal_notes=runtime_notes,
        )
        runtime_context_qa_response_language = self._recall_response_language(
            query=context.req.query,
            default_language=context.response_language,
            multimodal_notes=runtime_notes,
        )
        # Strict gate: only explicit recall cues should activate deterministic fact recall.
        # Subject parsing alone must not hijack normal conversation turns.
        memory_router = getattr(self, "memory_recall_router", None)
        if memory_router is not None and hasattr(memory_router, "has_memory_recall_cue"):
            memory_recall_mode = bool(memory_router.has_memory_recall_cue(context.req.query))
        else:
            memory_recall_mode = bool(self._has_memory_recall_cue(context.req.query))
        recall_orchestration_mode = bool(
            (runtime_context_qa_mode or memory_recall_mode)
            and (not web_search_triggered)
            and (not roleplay_mode)
            and (not bool(getattr(context, "force_local_rag", False)))
        )
        runtime_context_qa_evidence_payload: dict[str, Any] | None = None
        early_memory_recall_path = ""
        recall_fact_hit_subject = ""
        recall_fact_miss_reason = ""

        direct_fact_recall_enabled = str(
            os.getenv("LOCAL_AI_DIRECT_FACT_RECALL_ENABLED", "1") or "1"
        ).strip().lower() in {"1", "true", "yes", "on"}

        # Optional deterministic memory recall path (disabled by default).
        # Default behavior is model-native generation with memory context injection.
        if (
            direct_fact_recall_enabled
            and execution is None
            and memory_recall_mode
            and (not web_search_triggered)
            and (not roleplay_mode)
        ):
            if memory_router is not None and hasattr(memory_router, "recall_from_fact_store"):
                fact_recall = memory_router.recall_from_fact_store(
                    query=context.req.query,
                    response_language=context.response_language,
                    memory_bundle=context.memory_bundle,
                )
            else:
                fact_recall = self._memory_recall_response_from_fact_store(
                    query=context.req.query,
                    response_language=context.response_language,
                    memory_bundle=context.memory_bundle,
                )
            deterministic_memory_answer = str(fact_recall.get("answer") or "").strip()
            recall_fact_hit_subject = str(fact_recall.get("hit_subject") or "").strip()
            recall_fact_miss_reason = str(fact_recall.get("miss_reason") or "").strip()
            if deterministic_memory_answer:
                execution = ExecutionResult(
                    result_type="conversation",
                    structured_payload={
                        "style": "general_chat",
                        "memory_recall": "fact_store_direct",
                        "answer_type": coerce_answer_type_hint(answer_type_hint),
                        "contract_format": "plain",
                        "ungrounded_allowed": True,
                    },
                    citations=[],
                    tool_logs=[
                        "memory_recall:fact_store_direct",
                        "memory_recall:fact_store_injected_generation",
                    ],
                    generated_text=deterministic_memory_answer,
                    engine_used=context.settings.local_engine,
                    used_fallback=False,
                    runtime_detail="memory_recall_fact_store_direct",
                )
                early_memory_recall_path = "memory_recall_fact_store"
            elif recall_fact_miss_reason == "no_fact_for_subject":
                # Keep generating via model path even when fact subject is missing.
                # This avoids deterministic fallback wording and keeps natural conversation.
                pass

        if web_search_triggered:
            mode = context.settings.privacy_mode
            if mode == PrivacyMode.LOCAL_ONLY or (
                mode == PrivacyMode.HYBRID and not bool(getattr(context.settings, "hybrid_web_search_enabled", False))
            ):
                execution = GeneralChatWebGateHelpers.blocked_execution(
                    response_language=context.response_language,
                    privacy_mode=mode,
                    hybrid_web_search_enabled=bool(getattr(context.settings, "hybrid_web_search_enabled", False)),
                )
                conversation_path = "external_web_search_blocked"
            else:
                retriever = WebRetriever()
                configured_searxng_url = str(getattr(context.settings, "searxng_url", "") or "").strip()
                if not configured_searxng_url:
                    configured_searxng_url = str(os.getenv("LOCAL_AI_SEARXNG_URL", "") or "").strip()
                if not configured_searxng_url:
                    configured_searxng_url = "http://127.0.0.1:8080/search"

                docker_service = dependencies.get("docker_service")
                keep_running = bool(getattr(context.settings, "auto_start_searxng", False))
                allow_auto_stop = not keep_running
                parsed_searx = urlparse(configured_searxng_url)
                is_local_searx = (
                    parsed_searx.scheme in {"http", "https"}
                    and (parsed_searx.hostname or "").strip().lower() in {"localhost", "127.0.0.1"}
                    and (parsed_searx.port in {None, 8080})
                )
                searxng_ready = True
                readiness_logs: list[str] = []
                if is_local_searx and docker_service is not None:
                    try:
                        host = (parsed_searx.hostname or "127.0.0.1").strip() or "127.0.0.1"
                        port = int(parsed_searx.port or 8080)
                        searxng_ready, readiness_logs = await asyncio.wait_for(
                            self.ensure_local_searxng_ready_async(
                                docker_service=docker_service,
                                keep_running=keep_running,
                                allow_auto_stop=allow_auto_stop,
                                host=host,
                                port=port,
                                searxng_url=configured_searxng_url,
                                port_timeout_seconds=6.0,
                                http_timeout_seconds=8.0,
                            ),
                            timeout=20.0,
                        )
                    except asyncio.TimeoutError:
                        searxng_ready = False
                        readiness_logs = ["web_search:searxng_ready_timeout"]
                    except Exception:
                        searxng_ready = False
                        readiness_logs = ["web_search:searxng_ready_unavailable"]

                effective_web_query = utils._web_search_query_for_turn(
                    query=context.req.query,
                    last_context=last_context,
                    is_followup_web_search=followup_web_search,
                )
                if not searxng_ready:
                    source_rows = []
                    web_loop_logs = list(readiness_logs)
                    web_loop_logs.append("web_search:unavailable:searxng_not_ready")
                    web_loop_meta = {
                        "web_loop_rounds": 0,
                        "web_loop_converged": False,
                        "web_loop_quality_score": 0.0,
                        "web_loop_queries": [effective_web_query or context.req.query],
                        "web_loop_timed_out": False,
                        "round_timeout_seconds": 6.0,
                    }
                else:
                    source_rows, web_loop_logs, web_loop_meta = await asyncio.to_thread(
                        self._run_web_reasoning_loop,
                        retriever=retriever,
                        base_query=effective_web_query or context.req.query,
                        freshness_sensitive_query=bool(freshness_sensitive_query),
                        searxng_url=configured_searxng_url or None,
                        prefer_searxng=True,
                        max_rounds=3,
                        max_total_seconds=18.0,
                        round_timeout_seconds=6.0,
                    )
                if (
                    not source_rows
                    and is_local_searx
                    and docker_service is not None
                    and self._searxng_connection_refused(web_loop_logs)
                ):
                    try:
                        host = (parsed_searx.hostname or "127.0.0.1").strip() or "127.0.0.1"
                        port = int(parsed_searx.port or 8080)
                        retry_ready, retry_ready_logs = await asyncio.wait_for(
                            self.ensure_local_searxng_ready_async(
                                docker_service=docker_service,
                                keep_running=keep_running,
                                allow_auto_stop=allow_auto_stop,
                                host=host,
                                port=port,
                                searxng_url=configured_searxng_url,
                                port_timeout_seconds=20.0,
                                http_timeout_seconds=20.0,
                            ),
                            timeout=45.0,
                        )
                        web_loop_logs.extend(retry_ready_logs)
                        if retry_ready:
                            retriever_retry = WebRetriever()
                            source_rows, retried_logs, retried_meta = await asyncio.to_thread(
                                self._run_web_reasoning_loop,
                                retriever=retriever_retry,
                                base_query=effective_web_query or context.req.query,
                                freshness_sensitive_query=bool(freshness_sensitive_query),
                                searxng_url=configured_searxng_url or None,
                                prefer_searxng=True,
                                max_rounds=3,
                                max_total_seconds=18.0,
                                round_timeout_seconds=6.0,
                            )
                            web_loop_logs.extend(retried_logs)
                            web_loop_meta = retried_meta
                        else:
                            web_loop_logs.append("web_search:unavailable:searxng_not_ready")
                    except asyncio.TimeoutError:
                        web_loop_logs.append("web_search:searxng_retry_timeout")
                    except Exception:
                        pass
                if is_local_searx and docker_service is not None:
                    try:
                        docker_service.mark_usage(allow_auto_stop=allow_auto_stop)
                    except Exception:
                        pass
                if source_rows:
                    now = datetime.now(timezone.utc)
                    citations: list[Citation] = []
                    for idx, source in enumerate(source_rows[:3], start=1):
                        citations.append(
                            Citation(
                                doc_id=f"web:{idx}",
                                chunk_id=f"web:{idx}:chunk",
                                file_path=str(source.get("url") or ""),
                                snippet=str(source.get("snippet") or "")[:320],
                                score=max(0.2, 0.9 - (idx * 0.15)),
                                modified_at=now,
                                category="web",
                                subcategory="internet",
                                tags=["web"],
                                document_type="web_page",
                                importance=0.6,
                            )
                        )

                    synth_prompt = (
                        "아래 웹 근거를 바탕으로 사용자의 질문에 직접 답하세요.\n"
                        "- 한국어로 자연스럽고 간결하게 작성\n"
                        "- 핵심 주장 2~4개로 정리\n"
                        "- 불확실하면 단정하지 말 것\n"
                        "- 출처 번호를 [1], [2] 형태로 문장 끝에 표기\n\n"
                        f"사용자 질문:\n{context.req.query}\n\n"
                        f"웹 근거:\n{self._web_sources_for_prompt(source_rows)}"
                    ) if context.response_language == "ko" else (
                        "Answer the user question directly using only the web evidence below.\n"
                        "- Be concise and natural\n"
                        "- Keep 2-4 key points\n"
                        "- Do not overclaim uncertain facts\n"
                        "- Cite with [1], [2] markers\n\n"
                        f"User question:\n{context.req.query}\n\n"
                        f"Web evidence:\n{self._web_sources_for_prompt(source_rows)}"
                    )
                    try:
                        synthesized = await self._run_conversation_inference(
                            executor=executor,
                            query=synth_prompt,
                            context=context,
                            max_tokens=max_tokens,
                            style_profile=effective_style_profile,
                        )
                        answer_text = str(synthesized.generated_text or "").strip()
                    except Exception as exc:
                        synthesized = ExecutionResult(
                            result_type="conversation",
                            structured_payload={"reason": "web_synthesis_inference_failed", "ungrounded_allowed": True},
                            citations=[],
                            tool_logs=["web_answer:synthesis_failed"],
                            generated_text="",
                            engine_used=context.settings.local_engine,
                            used_fallback=True,
                            runtime_detail=str(exc),
                        )
                        answer_text = ""
                    if not answer_text:
                        answer_text = self._deterministic_web_summary(
                            query=context.req.query,
                            sources=source_rows,
                            language=context.response_language,
                        )
                    web_summary = answer_text[:500]
                    web_memory_for_metadata = {
                        "web_query": (effective_web_query or context.req.query)[:260],
                        "web_summary": web_summary,
                        "web_sources": source_rows,
                    }
                    execution = ExecutionResult(
                        result_type="conversation",
                        structured_payload={
                            "web_path": "direct",
                            "web_sources_count": len(citations),
                            "ungrounded_allowed": True,
                            "web_summary": web_summary,
                            "web_loop_rounds": int(web_loop_meta.get("web_loop_rounds") or 0),
                            "web_loop_converged": bool(web_loop_meta.get("web_loop_converged")),
                            "web_loop_quality_score": float(web_loop_meta.get("web_loop_quality_score") or 0.0),
                            "web_loop_queries": list(web_loop_meta.get("web_loop_queries") or [])[:3],
                        },
                        citations=citations,
                        tool_logs=[*web_loop_logs, "web_search:direct", "web_answer:synthesized"],
                        generated_text=answer_text,
                        engine_used=None,
                        used_fallback=False,
                        runtime_detail="web_search_direct",
                    )
                    conversation_path = "external_web_search_direct"
                    is_local = False
                else:
                    unavailable_text = (
                        "인터넷에서 신뢰 가능한 근거를 수집하지 못해 답을 확정할 수 없습니다. 잠시 후 다시 시도해 주세요."
                        if context.response_language == "ko"
                        else "I couldn't gather reliable web results right now. Please try again shortly."
                    )
                    execution = ExecutionResult(
                        result_type="conversation",
                        structured_payload={
                            "web_path": "unavailable",
                            "ungrounded_allowed": True,
                            "web_loop_rounds": int(web_loop_meta.get("web_loop_rounds") or 0),
                            "web_loop_converged": bool(web_loop_meta.get("web_loop_converged")),
                            "web_loop_quality_score": float(web_loop_meta.get("web_loop_quality_score") or 0.0),
                            "web_loop_queries": list(web_loop_meta.get("web_loop_queries") or [])[:3],
                        },
                        citations=[],
                        tool_logs=["web_search:unavailable", *web_loop_logs],
                        generated_text=unavailable_text,
                        engine_used=None,
                        used_fallback=False,
                        runtime_detail="web_search_unavailable",
                    )
                    conversation_path = "external_web_search_unavailable"

        if execution is None and not web_search_triggered:
            ranked_entries: list[dict[str, Any]] = []
            if memory_service is not None:
                try:
                    ranked_entries = list(
                        memory_service.get_ranked_web_memory_entries(
                            session_id=context.session_id,
                            query=context.req.query,
                            limit=4,
                        )
                    )
                except Exception:
                    ranked_entries = []

            selected_entry: dict[str, Any] | None = None
            if ranked_entries:
                top = ranked_entries[0]
                top_confidence = float(top.get("confidence") or 0.0)
                top_source_count = int(top.get("source_count") or 0)
                if (
                    top_confidence >= 0.60
                    and top_source_count >= 1
                    and self._is_relevant_web_memory_entry(
                        query=context.req.query,
                        entry=top,
                        last_context=last_context,
                        followup_web_search=followup_web_search,
                    )
                ):
                    selected_entry = top
                    web_memory_rank_score = max(0.0, min(1.0, top_confidence))

            if selected_entry is not None:
                selected_sources = selected_entry.get("sources")
                if not isinstance(selected_sources, list):
                    selected_sources = []
                memory_prompt = (
                    "아래는 같은 채팅방에서 저장된 웹 검색 메모리입니다.\n"
                    "새 검색 없이 이 메모리만으로 질문 중심으로 답하세요.\n"
                    "- 한국어 자연어 요약으로 답할 것\n"
                    "- 핵심 2~4개를 정리할 것\n"
                    "- 메모리에 없는 내용은 추측하지 말 것\n"
                    "- 출처 번호 [1], [2]를 문장 끝에 유지할 것\n\n"
                    f"사용자 질문:\n{context.req.query}\n\n"
                    f"저장된 웹 질의:\n{selected_entry.get('query','')}\n\n"
                    f"저장된 웹 요약:\n{selected_entry.get('answer_summary','')}\n\n"
                    f"저장된 출처:\n{self._web_sources_for_prompt(selected_sources)}"
                ) if context.response_language == "ko" else (
                    "Use only the saved web memory from this same chat session.\n"
                    "Do not run a new search and do not hallucinate.\n"
                    "- Respond with natural language summary\n"
                    "- Keep 2-4 key points\n"
                    "- Keep citation markers like [1], [2]\n\n"
                    f"User question:\n{context.req.query}\n\n"
                    f"Saved web query:\n{selected_entry.get('query','')}\n\n"
                    f"Saved summary:\n{selected_entry.get('answer_summary','')}\n\n"
                    f"Saved sources:\n{self._web_sources_for_prompt(selected_sources)}"
                )
                try:
                    memory_inference = await self._run_conversation_inference(
                        executor=executor,
                        query=memory_prompt,
                        context=context,
                        max_tokens=max_tokens,
                        style_profile=effective_style_profile,
                    )
                    memory_answer = str(memory_inference.generated_text or "").strip()
                except Exception as exc:
                    memory_inference = ExecutionResult(
                        result_type="conversation",
                        structured_payload={"reason": "session_web_memory_inference_failed", "ungrounded_allowed": True},
                        citations=[],
                        tool_logs=["web_memory:inference_failed"],
                        generated_text="",
                        engine_used=context.settings.local_engine,
                        used_fallback=True,
                        runtime_detail=str(exc),
                    )
                    memory_answer = ""
                if not memory_answer:
                    memory_answer = str(selected_entry.get("answer_summary") or "").strip()
                if memory_answer:
                    web_memory_for_metadata = {
                        "web_query": str(selected_entry.get("query") or "")[:260],
                        "web_summary": memory_answer[:500],
                        "web_sources": selected_sources[:4],
                    }
                    now = datetime.now(timezone.utc)
                    citations: list[Citation] = []
                    for idx, source in enumerate(selected_sources[:3], start=1):
                        if not isinstance(source, dict):
                            continue
                        url = str(source.get("url") or "").strip()
                        if not url:
                            continue
                        citations.append(
                            Citation(
                                doc_id=f"web-memory:{idx}",
                                chunk_id=f"web-memory:{idx}:chunk",
                                file_path=url,
                                snippet=str(source.get("snippet") or "")[:320],
                                score=max(0.2, 0.9 - (idx * 0.15)),
                                modified_at=now,
                                category="web",
                                subcategory="session_memory",
                                tags=["web", "session-memory"],
                                document_type="web_page",
                                importance=0.55,
                            )
                        )
                    execution = ExecutionResult(
                        result_type="conversation",
                        structured_payload={"web_path": "session_memory", "ungrounded_allowed": True},
                        citations=citations,
                        tool_logs=[
                            f"web_memory:candidates={len(ranked_entries)}",
                            f"web_memory:selected_score={web_memory_rank_score:.3f}",
                            "web_memory:reuse",
                            "web_answer:session_memory",
                        ],
                        generated_text=memory_answer,
                        engine_used=memory_inference.engine_used,
                        used_fallback=memory_inference.used_fallback,
                        runtime_detail=memory_inference.runtime_detail or "web_memory_reused",
                    )
                    conversation_path = "session_web_memory_reused"
                    is_local = True
                    web_memory_reused = True

        recovery_metadata: dict[str, Any] = {
            "generation_retry_count": 0,
            "generation_backoff_profile": "",
            "recovery_path": (early_memory_recall_path or "none"),
            "degraded_internal": False,
        }
        contract_metadata: dict[str, Any] = {
            "validation_passed": True,
            "validation_fail_reasons": [],
            "regeneration_attempts": 0,
            "contract_format": "plain",
        }
        recall_pipeline_metadata: dict[str, Any] = {}
        if execution is None:
            if recall_orchestration_mode:
                if runtime_context_qa_mode:
                    runtime_context_qa_evidence_payload = self._runtime_context_qa_evidence_payload(
                        query=context.req.query,
                        multimodal_context=str(runtime_context or ""),
                        response_language=runtime_context_qa_response_language,
                        answer_type_hint=recall_answer_type_hint,
                    )
                    recall_evidence_payload = dict(runtime_context_qa_evidence_payload or {})
                    if self._runtime_context_qa_deterministic_first_enabled():
                        deterministic_answer = self._runtime_context_qa_deterministic_answer(
                            query=context.req.query,
                            multimodal_context=str(runtime_context or ""),
                            response_language=runtime_context_qa_response_language,
                        ).strip()
                        if deterministic_answer:
                            execution = ExecutionResult(
                                result_type="conversation",
                                structured_payload={
                                    "style": "general_chat",
                                    "ungrounded_allowed": True,
                                    "runtime_context_qa": "deterministic",
                                    "runtime_context_qa_evidence_count": int(
                                        (runtime_context_qa_evidence_payload or {}).get("candidate_count") or 0
                                    ),
                                    "answer_type": str(recall_answer_type_hint or "freeform"),
                                    "contract_format": "plain",
                                },
                                citations=[],
                                tool_logs=[
                                    "recall:runtime_context_qa_deterministic",
                                    "runtime_context_qa:deterministic",
                                ],
                                generated_text=deterministic_answer,
                                engine_used=context.settings.local_engine,
                                used_fallback=False,
                                runtime_detail="runtime_context_qa_deterministic",
                            )
                            contract_metadata = {
                                "validation_passed": True,
                                "validation_fail_reasons": [],
                                "regeneration_attempts": 0,
                                "contract_format": "plain",
                                "recall_pipeline_version": "recall.v2.runtime_context.deterministic",
                                "pass1_candidate_count": int((runtime_context_qa_evidence_payload or {}).get("candidate_count") or 0),
                                "pass1_coverage": float((runtime_context_qa_evidence_payload or {}).get("coverage") or 0.0),
                                "elapsed_ms": 0,
                                "answer_type": str(recall_answer_type_hint or "freeform"),
                            }
                            recall_pipeline_metadata = {
                                "recall_pipeline_version": "recall.v2.runtime_context.deterministic",
                                "pass1_candidate_count": int((runtime_context_qa_evidence_payload or {}).get("candidate_count") or 0),
                                "pass1_coverage": float((runtime_context_qa_evidence_payload or {}).get("coverage") or 0.0),
                                "elapsed_ms": 0,
                            }
                            recovery_metadata["recovery_path"] = "runtime_context_qa_deterministic"
                            recovery_metadata["degraded_internal"] = False
                else:
                    recall_evidence_payload = self._memory_recall_evidence_payload(
                        query=context.req.query,
                        response_language=context.response_language,
                        last_context=last_context,
                        session_digest=context.session_digest,
                        memory_bundle=context.memory_bundle,
                        answer_type_hint=recall_answer_type_hint,
                    )
                    runtime_context_qa_evidence_payload = dict(recall_evidence_payload)
                if execution is None:
                    if memory_recall_mode and not runtime_context_qa_mode:
                        evidence_rows = list((recall_evidence_payload or {}).get("candidate_evidence") or [])
                        evidence_lines: list[str] = []
                        for idx, row in enumerate(evidence_rows[:8], 1):
                            content = str((row or {}).get("content") or "").strip()
                            if not content:
                                continue
                            evidence_lines.append(f"{idx}. {content[:220]}")
                        evidence_block = "\n".join(evidence_lines) if evidence_lines else "(no evidence)"
                        if context.response_language == "ko":
                            injected_query = (
                                "메모리 회수 질문입니다. 아래 근거만 사용해 답하세요.\n"
                                "추측/환각 금지. 근거가 없으면 없다고 말하세요.\n\n"
                                f"[질문]\n{context.req.query}\n\n[메모리 근거]\n{evidence_block}\n\n[답변]"
                            )
                        else:
                            injected_query = (
                                "Memory recall question. Use only the evidence below.\n"
                                "Do not guess. If missing, say unknown.\n\n"
                                f"[Question]\n{context.req.query}\n\n[Evidence]\n{evidence_block}\n\n[Answer]"
                            )
                        execution, recovery_metadata = await self._run_conversation_with_recovery(
                            executor=executor,
                            query=injected_query,
                            context=context,
                            base_max_tokens=max_tokens,
                            session_summary_override=context.session_digest,
                            style_profile=effective_style_profile,
                            generation_style="conversation",
                        )
                        recovery_metadata["recovery_path"] = "memory_recall_injected_generation"
                        recovery_metadata["degraded_internal"] = bool(execution.used_fallback)
                        recall_pipeline_metadata = {
                            "recall_pipeline_version": "recall.v3.memory_injected_generation",
                            "pass1_candidate_count": int((recall_evidence_payload or {}).get("candidate_count") or 0),
                            "pass1_coverage": float((recall_evidence_payload or {}).get("coverage") or 0.0),
                            "elapsed_ms": 0,
                        }
                        execution.structured_payload["memory_recall"] = "injected_generation"
                        execution.tool_logs.append("memory_recall:injected_generation")
                    else:
                        execution, contract_metadata = await self._run_recall_two_pass_orchestration(
                            executor=executor,
                            context=context,
                            query=context.req.query,
                            answer_type_hint=recall_answer_type_hint,
                            evidence_payload=recall_evidence_payload,
                            response_language=(
                                runtime_context_qa_response_language
                                if runtime_context_qa_mode
                                else self._recall_response_language(
                                    query=context.req.query,
                                    default_language=context.response_language,
                                    multimodal_notes=getattr(context.req, "multimodal_notes", None),
                                )
                            ),
                            style_profile=effective_style_profile,
                        )
                        recall_pipeline_metadata = {
                            "recall_pipeline_version": str(contract_metadata.get("recall_pipeline_version") or ""),
                            "pass1_candidate_count": int(contract_metadata.get("pass1_candidate_count") or 0),
                            "pass1_coverage": float(contract_metadata.get("pass1_coverage") or 0.0),
                            "elapsed_ms": int(contract_metadata.get("elapsed_ms") or 0),
                        }
                        recovery_metadata["recovery_path"] = "recall_two_pass"
                        recovery_metadata["degraded_internal"] = not bool(contract_metadata.get("validation_passed", True))
                        recovery_metadata["generation_retry_count"] = int(contract_metadata.get("regeneration_attempts") or 0)
                        if runtime_context_qa_mode:
                            execution.tool_logs.append("runtime_context_qa:strict")
                            evidence_count = int((runtime_context_qa_evidence_payload or {}).get("candidate_count") or 0)
                            execution.structured_payload["runtime_context_qa"] = "two_pass"
                            execution.structured_payload["runtime_context_qa_evidence_count"] = evidence_count
            else:
                input_builder = getattr(self, "conversation_input_builder", None)
                if input_builder is not None and hasattr(input_builder, "build"):
                    conversation_query, session_summary_override = input_builder.build(
                        context=context,
                        runtime_context=runtime_context,
                        runtime_notes=runtime_notes,
                    )
                else:
                    conversation_query = self._conversation_query_with_context(
                        query=context.req.query,
                        response_language=context.response_language,
                        followup_resolution=context.followup_resolution,
                        last_context=context.last_context,
                    )
                    session_summary_override = None
                runtime_style_profile = effective_style_profile
                runtime_max_tokens = max_tokens
                # Optional concise-followup shortcut.
                # Default is disabled to avoid stale copy/reuse of previous answer text.
                try:
                    q_raw = str(context.req.query or "").strip().lower()
                    concise_shortcut_enabled = str(
                        os.getenv("LOCAL_AI_DIRECT_ONE_LINE_SHORTCUT_ENABLED", "0") or "0"
                    ).strip().lower() in {"1", "true", "yes", "on"}
                    concise_terms = ("한 줄", "한줄", "한 문장", "한문장", "짧게", "짧은")
                    refer_terms = ("방금", "아까", "직전", "그거", "그 답변", "that", "previous")
                    concise_followup_requested = (
                        any(token in q_raw for token in concise_terms)
                        and any(token in q_raw for token in refer_terms)
                    )
                    if concise_followup_requested:
                        conversation_query = (
                            f"{conversation_query}\n\n"
                            "[응답 형식]\n"
                            "- 바로 이전 답변을 한국어 한 문장으로만 요약하세요.\n"
                            "- 불필요한 서론/목록/설명 없이 결과 문장만 출력하세요."
                        )
                    concise_followup = (
                        concise_shortcut_enabled
                        and
                        concise_followup_requested
                    )
                    last_summary = ""
                    if isinstance(context.last_context, dict):
                        last_summary = str(
                            context.last_context.get("generated_text")
                            or context.last_context.get("result_summary")
                            or ""
                        ).strip()
                    if not last_summary:
                        digest_payload = getattr(context, "session_digest_payload", None)
                        if isinstance(digest_payload, dict):
                            turns = digest_payload.get("recent_turns") if isinstance(digest_payload.get("recent_turns"), list) else []
                            for item in reversed(turns):
                                if not isinstance(item, dict):
                                    continue
                                if str(item.get("role") or "").strip().lower() != "assistant":
                                    continue
                                txt = str(item.get("text") or "").strip()
                                if txt:
                                    last_summary = txt
                                    break
                    if concise_followup and last_summary:
                        one_line = re.sub(r"\s+", " ", last_summary).strip()
                        one_line = one_line.split("\n")[0].strip()
                        if not one_line:
                            one_line = re.sub(r"\s+", " ", str(last_summary)).strip()
                        if len(one_line) > 180:
                            cut = max(one_line.rfind(". ", 0, 180), one_line.rfind("! ", 0, 180), one_line.rfind("? ", 0, 180))
                            if cut >= 40:
                                one_line = one_line[: cut + 1].strip()
                            else:
                                one_line = one_line[:180].rstrip() + "..."
                        execution = ExecutionResult(
                            result_type="conversation",
                            structured_payload={
                                "style": "general_chat",
                                "ungrounded_allowed": True,
                                "answer_type": "summary",
                                "contract_format": "plain",
                                "response_mode": "conversational_direct",
                            },
                            citations=[],
                            tool_logs=["conversation:direct_one_line_summary"],
                            generated_text=one_line,
                            engine_used=context.settings.local_engine,
                            used_fallback=False,
                            runtime_detail="direct_one_line_summary",
                        )
                        recovery_metadata = {
                            "generation_retry_count": 0,
                            "generation_backoff_profile": "",
                            "recovery_path": "direct_one_line_summary",
                            "degraded_internal": False,
                            "keep_waiting_mode": True,
                            "fast_chat_mode": False,
                        }
                    else:
                        execution, recovery_metadata = await self._run_conversation_with_recovery(
                            executor=executor,
                            query=conversation_query,
                            context=context,
                            base_max_tokens=runtime_max_tokens,
                            session_summary_override=session_summary_override,
                            style_profile=runtime_style_profile,
                            generation_style=generation_style,
                        )
                except Exception:
                    execution, recovery_metadata = await self._run_conversation_with_recovery(
                        executor=executor,
                        query=conversation_query,
                        context=context,
                        base_max_tokens=runtime_max_tokens,
                        session_summary_override=session_summary_override,
                        style_profile=runtime_style_profile,
                        generation_style=generation_style,
                    )
                # Keep model-native conversation by default.
                # Optional extra post-guards can be toggled for debugging only.
                extra_post_guards_enabled = str(
                    os.getenv("LOCAL_AI_CONVERSATION_EXTRA_POST_GUARDS", "0") or "0"
                ).strip().lower() in {"1", "true", "yes", "on"}
                if extra_post_guards_enabled:
                    if self._looks_stale_prev_user_echo(
                        answer=str(getattr(execution, "generated_text", "") or ""),
                        query=str(context.req.query or ""),
                        last_context=context.last_context,
                    ):
                        retry_execution = await self._run_conversation_inference(
                            executor=executor,
                            query=conversation_query,
                            context=context,
                            max_tokens=runtime_max_tokens,
                            generation_style=generation_style,
                            sampling_overrides={"temperature": 0.18, "top_p": 0.78, "top_k": 20},
                            timeout_seconds=None,
                            session_summary_override=session_summary_override,
                            style_profile=runtime_style_profile,
                        )
                        if self._conversation_answer_ready(retry_execution):
                            execution = retry_execution
                            execution.tool_logs.append("conversation:stale_prev_user_echo_retry")
            execution.tool_logs.insert(0, f"router:intent={ReasoningIntent.GENERAL_CHAT.value}")
            execution.tool_logs.append("agent:conversation_assistant")

        execution.tool_logs.append(f"conversation:max_tokens={max_tokens}")
        detail = str(execution.runtime_detail or "").strip()
        if detail:
            if "max_tokens=" not in detail:
                execution.runtime_detail = f"{detail};max_tokens={max_tokens}"
        else:
            execution.runtime_detail = f"max_tokens={max_tokens}"

        opening_trim_enabled = False
        original_text = str(execution.generated_text or "")
        execution_text = original_text
        if context.response_language == "ko":
            # Remove accidental leading non-Korean token fragments (e.g., stray "い").
            execution_text = re.sub(r"^\s*[ぁ-ゟ゠-ヿ]+\s*", "", execution_text).strip()
            execution_text = re.sub(r'^\s*까요\?\s*', "", execution_text).strip()
            execution_text = re.sub(r"^\s*께서는\s+", "", execution_text).strip()
            execution_text = re.sub(r"^\s*당신은\s+", "", execution_text).strip()
        opening_trim_applied = False
        if opening_trim_enabled:
            trimmed_text = self._trim_redundant_opening_from_last_context(
                answer=execution_text,
                last_context=context.last_context,
            )
            if trimmed_text != execution_text:
                opening_trim_applied = True
            execution_text = trimmed_text
        if memory_recall_mode and execution.result_type == "conversation":
            execution_text = self._normalize_memory_recall_surface(
                text=execution_text,
                response_language=context.response_language,
            )
        # Fragment retry is disabled by default to avoid response rewriting side effects.
        # Enforce final output language to follow current user query majority language.
        if execution.result_type == "conversation" and execution_text.strip():
            query_lang = detect_query_language(str(context.req.query or ""))
            ans = execution_text.strip()
            ko_chars = len(re.findall(r"[가-힣]", ans))
            ja_chars = len(re.findall(r"[\u3040-\u30ff]", ans))
            en_chars = len(re.findall(r"[A-Za-z]", ans))
            mismatch = False
            if query_lang == "ko" and ja_chars >= 6 and ko_chars <= ja_chars:
                mismatch = True
            elif query_lang == "ja" and ko_chars >= 6 and ja_chars <= ko_chars:
                mismatch = True
            elif query_lang == "en" and en_chars < 6 and (ko_chars >= 6 or ja_chars >= 6):
                mismatch = True
            if mismatch:
                lang_retry = await self._run_conversation_inference(
                    executor=executor,
                    query=context.req.query,
                    context=context,
                    max_tokens=runtime_max_tokens,
                    generation_style=generation_style,
                    sampling_overrides={"temperature": 0.18, "top_p": 0.78, "top_k": 20},
                    timeout_seconds=None,
                    session_summary_override=session_summary_override,
                    style_profile=runtime_style_profile,
                    response_language_override=query_lang,
                    language_preference_override=query_lang,
                )
                lang_retry_text = str(getattr(lang_retry, "generated_text", "") or "").strip()
                if lang_retry_text:
                    execution = lang_retry
                    original_text = str(execution.generated_text or "")
                    execution_text = original_text
                    execution.tool_logs.append(f"conversation:language_retry={query_lang}")
                    ans2 = execution_text.strip()
                    ko2 = len(re.findall(r"[가-힣]", ans2))
                    ja2 = len(re.findall(r"[\u3040-\u30ff]", ans2))
                    en2 = len(re.findall(r"[A-Za-z]", ans2))
                    mismatch2 = False
                    if query_lang == "ko" and ja2 >= 6 and ko2 <= ja2:
                        mismatch2 = True
                    elif query_lang == "ja" and ko2 >= 6 and ja2 <= ko2:
                        mismatch2 = True
                    elif query_lang == "en" and en2 < 6 and (ko2 >= 6 or ja2 >= 6):
                        mismatch2 = True
                    # Keep only one language retry guard (no rewrite pass).
        if execution_text != original_text:
            updated_detail = str(execution.runtime_detail or "")
            if opening_trim_applied:
                updated_detail = (
                    f"{updated_detail};deduped_opening=True"
                    if updated_detail
                    else "deduped_opening=True"
                )
            execution = execution.model_copy(
                update={
                    "generated_text": execution_text,
                    "runtime_detail": updated_detail,
                }
            )
            if opening_trim_applied:
                execution.tool_logs.append("conversation:deduped_opening")

        should_enforce_contract = str(
            os.getenv("LOCAL_AI_CONVERSATION_CONTRACT_ENFORCE_ENABLED", "0") or "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if (
            execution.result_type == "conversation"
            and str(execution.generated_text or "").strip()
            and execution.engine_used is not None
            and (not recall_orchestration_mode)
            and (not (generation_style == "casual"))
            and should_enforce_contract
        ):
            execution, contract_metadata = await self._enforce_output_contract(
                executor=executor,
                context=context,
                execution=execution,
                query=context.req.query,
                answer_type_hint=answer_type_hint,
                style_profile=effective_style_profile,
            )
            execution.structured_payload["answer_type"] = coerce_answer_type_hint(answer_type_hint)
            execution.structured_payload["contract_format"] = str(contract_metadata.get("contract_format") or "plain")
        elif execution.result_type == "conversation" and str(execution.generated_text or "").strip():
            execution.structured_payload["answer_type"] = coerce_answer_type_hint(answer_type_hint)
            execution.structured_payload["contract_format"] = "plain"

        if recall_orchestration_mode and execution.result_type == "conversation":
            execution.structured_payload["answer_type"] = str(
                contract_metadata.get("answer_type")
                or execution.structured_payload.get("answer_type")
                or recall_answer_type_hint
            )
            execution.structured_payload["contract_format"] = str(contract_metadata.get("contract_format") or "plain")
            if runtime_context_qa_mode:
                existing_runtime_qa_mode = str(execution.structured_payload.get("runtime_context_qa") or "").strip()
                if not existing_runtime_qa_mode:
                    execution.structured_payload["runtime_context_qa"] = "two_pass"
            if memory_recall_mode and not runtime_context_qa_mode:
                execution.structured_payload["memory_recall"] = "two_pass"

        if runtime_context_qa_mode and execution.result_type == "conversation" and (not recall_orchestration_mode):
            qa_valid = self._runtime_context_qa_two_pass_valid_answer(
                answer=str(execution.generated_text or ""),
                evidence_payload=runtime_context_qa_evidence_payload or {},
                response_language=context.response_language,
            )
            if qa_valid:
                execution.structured_payload["runtime_context_qa"] = "two_pass"
            else:
                runtime_context_qa_evidence_payload = self._runtime_context_qa_evidence_payload(
                    query=context.req.query,
                    multimodal_context=str(runtime_context or ""),
                    response_language=context.response_language,
                    answer_type_hint=recall_answer_type_hint,
                )
                execution.structured_payload["runtime_context_qa"] = "two_pass_best_effort"

        generated = str(execution.generated_text or "").strip()
        if generated.lower().startswith("error:"):
            detail = str(execution.runtime_detail or "").strip()
            execution.generated_text = ""
            execution.tool_logs = [*list(execution.tool_logs or []), "conversation:error_text_filtered"]
            execution.runtime_detail = (
                f"{detail}; generated_error_filtered={generated[:160]}"
                if detail
                else f"generated_error_filtered={generated[:160]}"
            )

        if not str(execution.generated_text or "").strip() and execution.result_type != "insufficient":
            execution = execution.model_copy(
                update={
                    "result_type": "conversation",
                    "structured_payload": {
                        "style": "general_chat",
                        "reason": "conversation_generation_failed",
                        "ungrounded_allowed": True,
                        "offer_regenerate": True,
                    },
                    "tool_logs": [*list(execution.tool_logs or []), "recovery:conversation_generation_failed"],
                    "generated_text": "",
                    "used_fallback": False,
                }
            )
        if str(getattr(context, "topic_switch_state", "stable") or "stable") == "confirmed":
            execution.structured_payload["transition_applied"] = True
            execution.structured_payload["transition_style"] = "llm_bridge"

        assist_citations = execution.citations if (
            conversation_path.startswith("external_web_search")
            or conversation_path == "session_web_memory_reused"
        ) else []

        verification = VerificationResult(
            is_valid=(execution.result_type == "conversation"),
            confidence=(0.84 if execution.result_type == "conversation" else 0.2),
            issues=([] if execution.result_type == "conversation" else ["runtime_unavailable"]),
            ambiguity_level=(0.16 if execution.result_type == "conversation" else 0.8),
            candidate_mode=False,
        )

        composed = composer.compose_v2(
            query=context.req.query,
            mode=context.req.mode,
            response_language=context.response_language,
            parsed_intent=context.parsed_intent,
            plan=plan,
            execution_result=execution,
            verification=verification,
            behavior_policy=(
                context.behavior_policy
                if hasattr(context.behavior_policy, "preferred_action_order")
                else BehaviorPolicy()
            ),
            response_length=getattr(context.memory_prefs, "response_length", "long") if context.memory_prefs else "long",
            show_citations=bool(assist_citations),
            prefer_action_suggestions=getattr(context.memory_prefs, "prefer_action_suggestions", True) if context.memory_prefs else True,
            used_profile=context.workspace.startup_profile,
            engine_used=execution.engine_used,
            used_fallback=execution.used_fallback,
            runtime_detail=execution.runtime_detail,
            followup_resolution=context.followup_resolution,
            allow_clarification=None,
            conversation_path=conversation_path,
            is_local=is_local,
            prompt_cache_hit=False,
        )
        composed.metadata["topic_switch_state"] = str(getattr(context, "topic_switch_state", "stable") or "stable")
        composed.metadata["topic_switch_score"] = float(max(0.0, min(1.0, float(getattr(context, "topic_switch_score", 0.0) or 0.0))))
        composed.metadata["memory_scope_mode"] = str(getattr(context, "memory_scope_mode", "full") or "full")
        composed.metadata["generation_retry_count"] = int(recovery_metadata.get("generation_retry_count") or 0)
        composed.metadata["generation_backoff_profile"] = str(recovery_metadata.get("generation_backoff_profile") or "")
        composed.metadata["recovery_path"] = str(recovery_metadata.get("recovery_path") or "none")
        composed.metadata["degraded_internal"] = bool(recovery_metadata.get("degraded_internal"))
        composed.metadata["keep_waiting_mode"] = bool(recovery_metadata.get("keep_waiting_mode"))
        composed.metadata["fast_chat_mode"] = bool(recovery_metadata.get("fast_chat_mode"))
        composed.metadata["clarify_gate_reason"] = str(recovery_metadata.get("clarify_gate_reason") or "general_chat_path")
        composed.metadata["aux_timeout_hit"] = bool(recovery_metadata.get("aux_timeout_hit") or False)
        composed.metadata["conversation_decode_profile"] = str(recovery_metadata.get("conversation_decode_profile") or "balanced")
        composed.metadata["style_profile_applied"] = bool(effective_style_profile is not None)
        composed.metadata["style_profile_source"] = str(style_profile_source or "default")
        composed.metadata["style_override_reason"] = str(style_override_reason or "default")
        composed.metadata["roleplay_mode"] = bool(roleplay_mode)
        composed.metadata["roleplay_persona"] = str(roleplay_persona or "")
        if recall_orchestration_mode:
            composed.metadata["answer_type_hint"] = str(
                execution.structured_payload.get("answer_type")
                or contract_metadata.get("answer_type")
                or recall_answer_type_hint
            )
        else:
            composed.metadata["answer_type_hint"] = coerce_answer_type_hint(answer_type_hint)
        composed.metadata["validation_passed"] = bool(contract_metadata.get("validation_passed", True))
        composed.metadata["validation_fail_reasons"] = list(contract_metadata.get("validation_fail_reasons") or [])
        composed.metadata["regeneration_attempts"] = int(contract_metadata.get("regeneration_attempts") or 0)
        composed.metadata["contract_format"] = str(contract_metadata.get("contract_format") or "plain")
        composed.metadata["contract_auto_wrapped"] = bool(contract_metadata.get("contract_auto_wrapped") or False)
        recovery_path = str(recovery_metadata.get("recovery_path") or "none")
        if recovery_path in {"memory_recall_fact_store", "memory_recall_fact_store_generated"}:
            composed.metadata["recall_path"] = "fact_store"
        elif recovery_path == "memory_recall_injected_generation":
            composed.metadata["recall_path"] = "memory_injected_generation"
        elif recovery_path == "recall_two_pass":
            composed.metadata["recall_path"] = "recall_two_pass"
        else:
            composed.metadata["recall_path"] = "none"
        if recall_fact_hit_subject:
            composed.metadata["fact_hit_subject"] = recall_fact_hit_subject
        if recall_fact_miss_reason:
            composed.metadata["fact_miss_reason"] = recall_fact_miss_reason
        composed.metadata["fact_overwrite_blocked"] = int(composed.metadata.get("fact_overwrite_blocked") or 0)
        logs_joined = " ".join(str(item) for item in list(execution.tool_logs or []))
        composed.metadata["repetition_mitigated"] = bool(
            ("conversation:deduped_opening" in logs_joined)
            or ("recovery:trim_incomplete_tail" in logs_joined)
            or ("recovery:continuation_append" in logs_joined)
        )
        if recall_orchestration_mode:
            composed.metadata["recall_pipeline_version"] = str(
                recall_pipeline_metadata.get("recall_pipeline_version")
                or contract_metadata.get("recall_pipeline_version")
                or self._recall_pipeline_version()
            )
            composed.metadata["pass1_candidate_count"] = int(
                recall_pipeline_metadata.get("pass1_candidate_count")
                or contract_metadata.get("pass1_candidate_count")
                or 0
            )
            composed.metadata["pass1_coverage"] = float(
                recall_pipeline_metadata.get("pass1_coverage")
                or contract_metadata.get("pass1_coverage")
                or 0.0
            )
            composed.metadata["elapsed_ms"] = int(
                recall_pipeline_metadata.get("elapsed_ms")
                or contract_metadata.get("elapsed_ms")
                or 0
            )
            composed.metadata["pass1_confidence"] = float(contract_metadata.get("pass1_confidence") or 0.0)
            composed.metadata["pass1_answer_type"] = str(
                contract_metadata.get("pass1_answer_type")
                or contract_metadata.get("answer_type")
                or execution.structured_payload.get("answer_type")
                or "freeform"
            )
            composed.metadata["pass1_top_indices"] = list(contract_metadata.get("pass1_top_indices") or [])[:5]
            composed.metadata["pass1_scored_candidates"] = list(contract_metadata.get("pass1_scored_candidates") or [])[:5]
            composed.metadata["selected_candidate_index"] = int(
                contract_metadata.get("selected_candidate_index")
                if contract_metadata.get("selected_candidate_index") is not None
                else execution.structured_payload.get("selected_candidate_index") or -1
            )
            composed.metadata["selected_candidate_score"] = float(contract_metadata.get("selected_candidate_score") or 0.0)
            composed.metadata["selected_candidate_preview"] = str(contract_metadata.get("selected_candidate_preview") or "")
            composed.metadata["best_validation_score"] = float(contract_metadata.get("best_validation_score") or 0.0)
            composed.metadata["best_validation_reasons"] = list(contract_metadata.get("best_validation_reasons") or [])
        if effective_style_profile is not None:
            composed.metadata["response_style_profile"] = dict(effective_style_profile)
        composed.metadata.setdefault("engine_recovery_attempt", 0)
        composed.metadata.setdefault("native_crash_detected", False)
        web_path = execution.structured_payload.get("web_path")
        if isinstance(web_path, str) and web_path:
            composed.metadata["web_path"] = web_path
            composed.metadata["web_sources_count"] = int(execution.structured_payload.get("web_sources_count") or 0)
            composed.metadata["web_fetch_failures"] = int(execution.structured_payload.get("web_fetch_failures") or 0)
            composed.metadata["web_loop_rounds"] = int(execution.structured_payload.get("web_loop_rounds") or 0)
            composed.metadata["web_loop_converged"] = bool(execution.structured_payload.get("web_loop_converged"))
            composed.metadata["web_loop_quality_score"] = float(
                max(0.0, min(1.0, float(execution.structured_payload.get("web_loop_quality_score") or 0.0)))
            )
            composed.metadata["web_loop_queries"] = list(execution.structured_payload.get("web_loop_queries") or [])[:3]
        if web_memory_for_metadata:
            composed.metadata["web_query"] = str(web_memory_for_metadata.get("web_query") or "")[:260]
            composed.metadata["web_summary"] = str(web_memory_for_metadata.get("web_summary") or "")[:500]
            raw_sources = web_memory_for_metadata.get("web_sources")
            if isinstance(raw_sources, list):
                composed.metadata["web_sources"] = raw_sources[:4]
        composed.metadata["web_memory_reused"] = bool(web_memory_reused)
        composed.metadata["web_memory_rank_score"] = (
            float(max(0.0, min(1.0, web_memory_rank_score)))
            if web_memory_reused
            else 0.0
        )
        return composed
