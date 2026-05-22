from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from typing import Any

from ....models import ExecutionResult
from ...answer_contract import (
    coerce_answer_type_hint,
    extract_contract_response,
    validate_contract_response,
)


class GeneralChatConversationExecutionHelpers:
    logger = logging.getLogger(__name__)
    @staticmethod
    def contract_regen_prompt(
        *,
        query: str,
        response_language: str,
        answer_type_hint: str,
        draft_answer: str,
        fail_reasons: list[str],
    ) -> str:
        reasons = ", ".join(str(item) for item in fail_reasons[:5]) or "contract_violation"
        if response_language == "ko":
            return (
                "아래 질문에 대해 엄격 JSON만 출력하세요.\n"
                "출력 스키마: {\"answer\":\"...\",\"answer_type\":\"date|number|boolean|entity|freeform\",\"language\":\"ko|en|ja\"}\n"
                "규칙:\n"
                "1) JSON 외 텍스트 금지\n"
                "2) 인사/메타/태그 금지\n"
                "3) answer는 질문에 직접 답변 1~2문장\n"
                f"4) answer_type은 {answer_type_hint}\n\n"
                f"질문: {query}\n"
                f"이전 초안: {draft_answer[:320]}\n"
                f"실패 사유: {reasons}\n"
                "JSON:"
            )
        if response_language == "ja":
            return (
                "次の質問に対して厳密なJSONのみを出力してください。\n"
                "スキーマ: {\"answer\":\"...\",\"answer_type\":\"date|number|boolean|entity|freeform\",\"language\":\"ko|en|ja\"}\n"
                "ルール:\n"
                "1) JSON以外の出力禁止\n"
                "2) 挨拶/メタ/タグ禁止\n"
                "3) answerは質問に直接答える1〜2文\n"
                f"4) answer_typeは{answer_type_hint}\n\n"
                f"質問: {query}\n"
                f"前回草案: {draft_answer[:320]}\n"
                f"失敗理由: {reasons}\n"
                "JSON:"
            )
        return (
            "Return strict JSON only for the question below.\n"
            "Schema: {\"answer\":\"...\",\"answer_type\":\"date|number|boolean|entity|freeform\",\"language\":\"ko|en|ja\"}\n"
            "Rules:\n"
            "1) No text outside JSON\n"
            "2) No greeting/meta/tag leakage\n"
            "3) answer must directly answer the question in 1-2 sentences\n"
            f"4) answer_type must be {answer_type_hint}\n\n"
            f"Question: {query}\n"
            f"Previous draft: {draft_answer[:320]}\n"
            f"Failure reasons: {reasons}\n"
            "JSON:"
        )

    @staticmethod
    async def enforce_output_contract(
        strategy,
        *,
        executor: Any,
        context: ReasoningContext,
        execution: ExecutionResult,
        query: str,
        answer_type_hint: str,
        style_profile: dict[str, Any] | None,
    ) -> tuple[ExecutionResult, dict[str, Any]]:
        raw_text = str(getattr(execution, "generated_text", "") or "").strip()
        parsed = extract_contract_response(raw_text) or {}
        answer = str(parsed.get("answer") or "").strip()
        contract_format = str(parsed.get("contract_format") or "plain")
        declared_answer_type = parsed.get("declared_answer_type")
        declared_language = parsed.get("declared_language")
        validation_fail_reasons = validate_contract_response(
            answer=answer,
            raw_text=raw_text,
            expected_language=context.response_language,
            answer_type_hint=coerce_answer_type_hint(answer_type_hint),
            declared_answer_type=str(declared_answer_type) if declared_answer_type else None,
            declared_language=str(declared_language) if declared_language else None,
        )
        if contract_format == "plain":
            validation_fail_reasons = ["missing_contract", *validation_fail_reasons]

        if not validation_fail_reasons and answer:
            return execution.model_copy(update={"generated_text": answer}), {
                "validation_passed": True,
                "validation_fail_reasons": [],
                "regeneration_attempts": 0,
                "contract_format": contract_format,
            }

        retry_attempts = strategy._contract_regen_attempts()
        last_answer = answer
        last_format = contract_format
        reasons = list(validation_fail_reasons)
        for attempt in range(1, retry_attempts + 1):
            prompt = strategy._contract_regen_prompt(
                query=query,
                response_language=context.response_language,
                answer_type_hint=coerce_answer_type_hint(answer_type_hint),
                draft_answer=last_answer or raw_text,
                fail_reasons=reasons,
            )
            try:
                regen = await strategy._run_conversation_inference(
                    executor=executor,
                    query=prompt,
                    context=context,
                    max_tokens=220,
                    generation_style="rewrite",
                    sampling_overrides={"temperature": 0.18, "top_p": 0.85, "top_k": 22},
                    timeout_seconds=None,
                    session_summary_override="",
                    style_profile=style_profile,
                )
            except asyncio.CancelledError:
                raise
            except Exception:
                continue
            regen_raw = str(getattr(regen, "generated_text", "") or "").strip()
            regen_parsed = extract_contract_response(regen_raw) or {}
            regen_answer = str(regen_parsed.get("answer") or "").strip()
            regen_format = str(regen_parsed.get("contract_format") or "plain")
            regen_reasons = validate_contract_response(
                answer=regen_answer,
                raw_text=regen_raw,
                expected_language=context.response_language,
                answer_type_hint=coerce_answer_type_hint(answer_type_hint),
                declared_answer_type=str(regen_parsed.get("declared_answer_type") or ""),
                declared_language=str(regen_parsed.get("declared_language") or ""),
            )
            if regen_format == "plain":
                regen_reasons = ["missing_contract", *regen_reasons]
            last_answer = regen_answer or last_answer
            last_format = regen_format or last_format
            reasons.extend(regen_reasons)
            if not regen_reasons and regen_answer:
                merged_logs = [*list(execution.tool_logs or []), f"contract:regen_success:attempt={attempt}"]
                return execution.model_copy(
                    update={
                        "generated_text": regen_answer,
                        "tool_logs": merged_logs,
                    }
                ), {
                    "validation_passed": True,
                    "validation_fail_reasons": [],
                    "regeneration_attempts": attempt,
                    "contract_format": regen_format,
                }

        fallback_answer = last_answer or answer or raw_text
        failure_logs = [*list(execution.tool_logs or []), "contract:validation_failed"]
        return execution.model_copy(
            update={
                "generated_text": fallback_answer,
                "tool_logs": failure_logs,
            }
        ), {
            "validation_passed": False,
            "validation_fail_reasons": sorted({str(item) for item in reasons if str(item).strip()}),
            "regeneration_attempts": retry_attempts,
            "contract_format": last_format or contract_format or "plain",
        }

    @staticmethod
    async def stabilize_conversation_completion(
        strategy,
        *,
        executor,
        context: ReasoningContext,
        query: str,
        execution: ExecutionResult,
        base_max_tokens: int,
        timeout_seconds: float,
        session_summary_override: str | None = None,
        style_profile: dict[str, Any] | None = None,
    ) -> ExecutionResult:
        if not strategy._conversation_answer_ready(execution):
            return execution
        base_text = str(execution.generated_text or "").strip()
        if not strategy._looks_incomplete_answer(base_text):
            return execution
        # Keep model-native conversational output. Rewrite-style continuation can
        # introduce unnatural tone drift and fragment artifacts.
        return execution

    @staticmethod
    async def run_conversation_with_recovery(
        strategy,
        *,
        executor,
        context: ReasoningContext,
        query: str,
        base_max_tokens: int,
        session_summary_override: Optional[str] = None,
        style_profile: Optional[Dict[str, Any]] = None,
        generation_style: str = "conversation",
    ) -> tuple[ExecutionResult, Dict[str, Any]]:
        # Simplified Ollama-like path (default):
        # - keep model-native output
        # - run at most one retry
        # - avoid salvage/rewriter-heavy recovery branches
        simplified_path_enabled = str(
            os.getenv("LOCAL_AI_CONVERSATION_SIMPLE_RECOVERY_ENABLED", "1") or "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        if simplified_path_enabled:
            decode_profile = strategy._decode_profile_with_style(
                fallback_profile=strategy._conversation_decode_profile(query=context.req.query),
                style_profile=style_profile,
            )
            if strategy._roleplay_mode_enabled(context=context):
                decode_profile = "roleplay"
            metadata: dict[str, Any] = {
                "generation_retry_count": 0,
                "generation_backoff_profile": "1.00,0.82",
                "recovery_path": "none",
                "degraded_internal": False,
                "keep_waiting_mode": True,
                "fast_chat_mode": False,
                "conversation_decode_profile": decode_profile,
            }
            retry_cap = max(strategy._retry_tokens_cap(), int(base_max_tokens))
            first_tokens = min(retry_cap, max(strategy._retry_min_tokens_floor(), int(base_max_tokens)))
            first_sampling = strategy._sampling_for_attempt(
                profile=decode_profile,
                attempt_index=0,
                generation_style="conversation",
            )
            try:
                first = await strategy._run_conversation_inference(
                    executor=executor,
                    query=str(query or "").strip(),
                    context=context,
                    max_tokens=first_tokens,
                    generation_style="conversation",
                    sampling_overrides=first_sampling,
                    timeout_seconds=None,
                    session_summary_override=session_summary_override,
                    style_profile=style_profile,
                )
                if strategy._conversation_answer_ready(first):
                    first.tool_logs.append(f"recovery:attempt=1;max_tokens={first_tokens};style=conversation")
                    return first, metadata
            except asyncio.CancelledError:
                raise
            except Exception:
                pass

            retry_tokens = min(retry_cap, max(strategy._retry_min_tokens_floor(), int(base_max_tokens * 0.82)))
            retry_sampling = {"temperature": 0.20, "top_p": 0.80, "top_k": 20}
            try:
                second = await strategy._run_conversation_inference(
                    executor=executor,
                    query=str(query or "").strip(),
                    context=context,
                    max_tokens=retry_tokens,
                    generation_style="conversation",
                    sampling_overrides=retry_sampling,
                    timeout_seconds=None,
                    session_summary_override=session_summary_override,
                    style_profile=style_profile,
                )
                if strategy._conversation_answer_ready(second):
                    second.tool_logs.append(f"recovery:attempt=2;max_tokens={retry_tokens};style=conversation")
                    metadata["generation_retry_count"] = 1
                    metadata["recovery_path"] = "single_retry"
                    return second, metadata
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                metadata["last_error"] = str(exc)

            metadata["generation_retry_count"] = 2
            metadata["recovery_path"] = "regenerate_required"
            metadata["degraded_internal"] = True
            return (
                ExecutionResult(
                    result_type="conversation",
                    structured_payload={
                        "style": "general_chat",
                        "reason": "generation_retry_exhausted",
                        "ungrounded_allowed": True,
                        "offer_regenerate": True,
                        "regenerate_prompt": str(query or "").strip(),
                    },
                    citations=[],
                    tool_logs=["recovery:generation_retry_exhausted"],
                    generated_text="",
                    engine_used=context.settings.local_engine,
                    used_fallback=False,
                    runtime_detail="generation_retry_exhausted",
                ),
                metadata,
            )

        # Policy simplification:
        # - Do not apply fast-chat short-circuit budgets.
        # - Prefer keep-generating flow over timeout-style early failure.
        fast_chat = False
        keep_waiting_mode = True
        attempts = strategy._generation_retry_max_attempts(context=context)
        backoff = strategy._generation_backoff_steps()[: max(1, attempts)]
        total_budget_ms = strategy._generation_retry_total_budget_ms()
        slot_count = len(backoff) + 2  # + split generation + clarify
        timeouts = strategy._retry_timeouts(total_budget_ms=total_budget_ms, slots=slot_count)
        metadata: dict[str, Any] = {
            "generation_retry_count": 0,
            "generation_backoff_profile": ",".join(f"{step:.2f}" for step in backoff),
            "recovery_path": "none",
            "degraded_internal": False,
            "keep_waiting_mode": bool(keep_waiting_mode),
            "fast_chat_mode": bool(fast_chat),
        }
        decode_profile = strategy._decode_profile_with_style(
            fallback_profile=strategy._conversation_decode_profile(query=context.req.query),
            style_profile=style_profile,
        )
        turn_started = time.time()
        if strategy._roleplay_mode_enabled(context=context):
            decode_profile = "roleplay"
        metadata["conversation_decode_profile"] = decode_profile

        require_bridge = False  # Disabled by user request to prevent repetition
        bridge_prefix = strategy._topic_bridge_prefix(response_language=context.response_language)
        base_query = str(query or "").strip()
        if require_bridge:
            base_query = (
                f"{base_query}\n\n"
                f"출력 규칙: 첫 문장은 '{bridge_prefix}'와 같은 자연스런 전환 문장 1개로 시작하고, "
                "바로 이어서 본답을 작성하세요. 전환 문장만 출력하지 마세요."
            ) if context.response_language == "ko" else (
                f"{base_query}\n\n"
                f"Output rule: Start with one natural transition sentence like '{bridge_prefix}', "
                "then immediately provide the full answer body. Do not output bridge-only text."
            )

        last_failure_detail = ""
        last_failed_text = ""
        retry_floor = strategy._retry_min_tokens_floor()
        retry_cap = strategy._retry_tokens_cap()
        split_cap = strategy._split_answer_tokens_cap()
        retry_cap = max(retry_cap, int(base_max_tokens))
        split_cap = max(split_cap, int(base_max_tokens))
        for idx, scale in enumerate(backoff):
            max_tokens = min(retry_cap, max(retry_floor, int(base_max_tokens * scale)))
            timeout = timeouts[min(idx, len(timeouts) - 1)]
            sampling = strategy._sampling_for_attempt(
                profile=decode_profile,
                attempt_index=idx,
                generation_style="conversation",
            )
            style = "conversation"
            try:
                execution = await strategy._run_conversation_inference(
                    executor=executor,
                    query=base_query,
                    context=context,
                    max_tokens=max_tokens,
                    generation_style="conversation",
                    sampling_overrides=sampling,
                    timeout_seconds=None,
                    session_summary_override=session_summary_override,
                    style_profile=style_profile,
                )
                metadata["generation_retry_count"] = idx
                if strategy._conversation_answer_ready(execution):
                    execution = await strategy._stabilize_conversation_completion(
                        executor=executor,
                        context=context,
                        query=query,
                        execution=execution,
                        base_max_tokens=base_max_tokens,
                        timeout_seconds=None,
                        session_summary_override=session_summary_override,
                        style_profile=style_profile,
                    )
                    execution.tool_logs.append(f"recovery:attempt={idx + 1};max_tokens={max_tokens};style={style}")
                    if idx > 0:
                        metadata["recovery_path"] = "token_backoff"
                    return execution, metadata
                last_failure_detail = str(execution.runtime_detail or "").strip()
                last_failed_text = str(execution.generated_text or "").strip()
            except asyncio.TimeoutError:
                last_failure_detail = f"inference_timeout:attempt{idx + 1}"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_failure_detail = str(exc)

        # Split generation path is intentionally disabled to avoid policy-heavy rewriting.
        try:
            raise RuntimeError("split_generate_disabled")
            split_timeout = timeouts[min(len(backoff), len(timeouts) - 1)]
            bridge_prompt = (
                f"Task: Create a single natural bridge sentence for a topic switch in {context.response_language}.\n"
                f"Example Start: '{bridge_prefix}'\n"
                f"User Topic: {query}\n"
                "Rule: Output ONLY the sentence. No explanation or reasoning. If no switch is needed, output an empty string.\n\n"
                "Bridge Sentence:"
            )
            bridge_execution = await strategy._run_conversation_inference(
                executor=executor,
                query=bridge_prompt,
                context=context,
                max_tokens=72,
                generation_style="rewrite",
                sampling_overrides={"temperature": 0.28, "top_p": 0.86, "top_k": 24},
                timeout_seconds=max(4.0, split_timeout * 0.5),
                session_summary_override=session_summary_override,
                style_profile=style_profile,
            )
            answer_prompt = (
                f"{query}\n\n"
                "Output only the direct final answer body."
            )
            answer_execution = await strategy._run_conversation_inference(
                executor=executor,
                query=answer_prompt,
                context=context,
                max_tokens=min(split_cap, max(192, int(base_max_tokens * 0.65))),
                generation_style="rewrite",
                sampling_overrides={"temperature": 0.30, "top_p": 0.84, "top_k": 24},
                timeout_seconds=max(4.0, split_timeout * 0.5),
                session_summary_override=session_summary_override,
                style_profile=style_profile,
            )
            bridge_text = str(bridge_execution.generated_text or "").strip()
            answer_text = str(answer_execution.generated_text or "").strip()
            if not bridge_execution.used_fallback and not answer_execution.used_fallback and bridge_text and answer_text:
                merged = f"{bridge_text}\n\n{answer_text}" if require_bridge else answer_text
                execution = answer_execution.model_copy(
                    update={
                        "generated_text": merged,
                        "tool_logs": [*list(answer_execution.tool_logs or []), "recovery:split_generate"],
                    }
                )
                execution = await strategy._stabilize_conversation_completion(
                    executor=executor,
                    context=context,
                    query=query,
                    execution=execution,
                    base_max_tokens=base_max_tokens,
                    timeout_seconds=None,
                    session_summary_override=session_summary_override,
                    style_profile=style_profile,
                )
                metadata["generation_retry_count"] = len(backoff)
                metadata["recovery_path"] = "split_generate"
                return execution, metadata
            last_failure_detail = str(answer_execution.runtime_detail or bridge_execution.runtime_detail or "").strip()
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            detail = str(exc).strip()
            # Preserve earlier concrete failure cause; split-disable marker is informational only.
            if detail and detail != "split_generate_disabled":
                last_failure_detail = detail

        no_fallback_mode = strategy._no_fallback_mode_enabled()
        if keep_waiting_mode:
            steady_scale = backoff[-1] if backoff else 0.45
            steady_tokens = min(retry_cap, max(retry_floor, int(base_max_tokens * steady_scale)))
            steady_timeout = timeouts[-1] if timeouts else 12.0
            steady_sampling = {"temperature": 0.28, "top_p": 0.84, "top_k": 20}
            steady_started = time.time()
            try:
                max_extra_attempts = max(
                    1,
                    min(4, int(float(str(os.getenv("LOCAL_AI_RETRY_UNTIL_CANCEL_MAX_ATTEMPTS", "1")).strip() or "1"))),
                )
            except Exception:
                max_extra_attempts = 1
            try:
                max_extra_seconds = max(
                    6.0,
                    min(60.0, float(str(os.getenv("LOCAL_AI_RETRY_UNTIL_CANCEL_MAX_SECONDS", "18")).strip() or "18")),
                )
            except Exception:
                max_extra_seconds = 18.0
            extra_attempt = 0
            while extra_attempt < max_extra_attempts and (time.time() - steady_started) < max_extra_seconds:
                extra_attempt += 1
                try:
                    execution = await strategy._run_conversation_inference(
                        executor=executor,
                        query=base_query,
                        context=context,
                        max_tokens=steady_tokens,
                        generation_style="conversation",
                        sampling_overrides=steady_sampling,
                        timeout_seconds=None,
                        session_summary_override=session_summary_override,
                        style_profile=style_profile,
                    )
                    if strategy._conversation_answer_ready(execution):
                        execution = await strategy._stabilize_conversation_completion(
                            executor=executor,
                            context=context,
                            query=query,
                            execution=execution,
                            base_max_tokens=base_max_tokens,
                            timeout_seconds=None,
                            session_summary_override=session_summary_override,
                            style_profile=style_profile,
                        )
                        execution.tool_logs.append(
                            f"recovery:retry_until_cancel:attempt={extra_attempt};max_tokens={steady_tokens}"
                        )
                        metadata["generation_retry_count"] = len(backoff) + extra_attempt
                        metadata["recovery_path"] = "retry_until_cancel"
                        metadata["degraded_internal"] = True
                        if no_fallback_mode:
                            metadata["no_fallback_mode"] = True
                        return execution, metadata
                    last_failure_detail = str(execution.runtime_detail or "").strip()
                    last_failed_text = str(execution.generated_text or "").strip()
                    # Deterministic engine/runtime failures should not spin forever.
                    lowered_detail = last_failure_detail.lower()
                    if any(
                        marker in lowered_detail
                        for marker in (
                            "mlx_isolated_failed_no_inprocess_fallback",
                            "worker_start_",
                            "워커 응답 타임아웃",
                            "워커 추론 실패",
                            "모델 경로가 비어",
                            "runtime_error",
                        )
                    ):
                        break
                except asyncio.TimeoutError:
                    last_failure_detail = f"inference_timeout:steady{extra_attempt}"
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    last_failure_detail = str(exc)
                await asyncio.sleep(0.15)
            # Exit keep-waiting loop and return regenerate-required runtime error below.

        # Fail-open path: prefer returning minimally sanitized model text over empty runtime error.
        salvage_text = str(last_failed_text or "").strip()
        # Never salvage from runtime-detail raw_preview fragments.
        # They may contain internal diagnostics (route/quality markers).
        if salvage_text:
            cleaned = ""
            compact = re.sub(r"\s+", " ", salvage_text).strip()
            compact = re.sub(r"(?im)^\s*(?:user|assistant|answer|response|답변)\s*[:：]\s*", "", compact).strip()
            compact = re.sub(r"(?i)\b(?:do not repeat instructions|respond with plain answer text only)\b\.?", "", compact).strip()
            lines = [ln.strip() for ln in compact.splitlines() if ln.strip()]
            if len(lines) >= 4 and len(set(lines)) == 1:
                compact = lines[0]
            cleaned = compact.strip()
            q_compact = re.sub(r"\s+", "", str(query or "")).lower()
            c_compact = re.sub(r"\s+", "", cleaned).lower()
            if q_compact and c_compact:
                # Drop obvious query-echo loops from fail-open salvage.
                if c_compact.startswith(q_compact) and len(c_compact) >= max(18, len(q_compact) * 2):
                    cleaned = ""
                elif len(set(re.findall(r"[A-Za-z가-힣0-9_]+", c_compact))) <= 2 and len(c_compact) >= 24:
                    cleaned = ""
            if cleaned:
                token_count = len(re.findall(r"\S+", cleaned))
                if token_count <= 2 or strategy._looks_leading_fragment(cleaned):
                    cleaned = ""
            if cleaned:
                metadata["generation_retry_count"] = len(backoff) + 1
                metadata["recovery_path"] = "fail_open_last_text"
                metadata["degraded_internal"] = True
                GeneralChatConversationExecutionHelpers.logger.warning(
                    "[GeneralChat] fail-open applied after retry exhaustion; detail=%s text=%s",
                    (last_failure_detail or "")[:240],
                    cleaned[:120],
                )
                return (
                    ExecutionResult(
                        result_type="conversation",
                        structured_payload={
                            "style": "general_chat",
                            "ungrounded_allowed": True,
                            "answer_type": "medium",
                            "contract_format": "plain",
                            "response_mode": "conversational_direct",
                        },
                        citations=[],
                        tool_logs=["recovery:fail_open_last_text"],
                        generated_text=cleaned,
                        engine_used=context.settings.local_engine,
                        used_fallback=False,
                        runtime_detail=last_failure_detail or "fail_open_last_text",
                    ),
                    metadata,
                )

        # Last attempt: run once more without timeout and with full token budget.
        try:
            final_execution = await strategy._run_conversation_inference(
                executor=executor,
                query=base_query,
                context=context,
                max_tokens=retry_cap,
                generation_style="conversation",
                sampling_overrides={"temperature": 0.26, "top_p": 0.86, "top_k": 24},
                timeout_seconds=None,
                session_summary_override=session_summary_override,
                style_profile=style_profile,
            )
            if strategy._conversation_answer_ready(final_execution):
                final_execution = await strategy._stabilize_conversation_completion(
                    executor=executor,
                    context=context,
                    query=query,
                    execution=final_execution,
                    base_max_tokens=base_max_tokens,
                    timeout_seconds=None,
                    session_summary_override=session_summary_override,
                    style_profile=style_profile,
                )
                final_execution.tool_logs.append("recovery:last_unbounded_attempt")
                metadata["generation_retry_count"] = len(backoff) + 1
                metadata["recovery_path"] = "last_unbounded_attempt"
                metadata["degraded_internal"] = True
                return final_execution, metadata
            last_failure_detail = str(final_execution.runtime_detail or last_failure_detail or "").strip()
            final_failed_text = str(final_execution.generated_text or "").strip()
            if final_failed_text:
                last_failed_text = final_failed_text
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            last_failure_detail = str(exc).strip() or last_failure_detail

        # If final attempt produced text but quality gate still rejected it,
        # return that text as degraded output instead of blank response.
        if last_failed_text:
            metadata["generation_retry_count"] = len(backoff) + 1
            metadata["recovery_path"] = "last_unbounded_degraded_text"
            metadata["degraded_internal"] = True
            return (
                ExecutionResult(
                    result_type="conversation",
                    structured_payload={
                        "style": "general_chat",
                        "ungrounded_allowed": True,
                        "answer_type": "medium",
                        "contract_format": "plain",
                        "response_mode": "conversational_direct",
                    },
                    citations=[],
                    tool_logs=["recovery:last_unbounded_degraded_text"],
                    generated_text=last_failed_text,
                    engine_used=context.settings.local_engine,
                    used_fallback=False,
                    runtime_detail=last_failure_detail or "last_unbounded_degraded_text",
                ),
                metadata,
            )

        # Return runtime error only when generation truly failed after all retries and final unbounded attempt.
        metadata["generation_retry_count"] = len(backoff) + 1
        metadata["recovery_path"] = "regenerate_required"
        metadata["degraded_internal"] = True
        regenerate_message = ""
        return (
            ExecutionResult(
                result_type="conversation",
                structured_payload={
                    "style": "general_chat",
                    "reason": "generation_retry_exhausted",
                    "ungrounded_allowed": True,
                    "offer_regenerate": True,
                    "regenerate_prompt": str(query or "").strip(),
                },
                citations=[],
                tool_logs=["recovery:generation_retry_exhausted"],
                generated_text=regenerate_message,
                engine_used=context.settings.local_engine,
                used_fallback=False,
                runtime_detail=last_failure_detail or "generation_retry_exhausted",
            ),
            metadata,
        )
