from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from ....models import ExecutionResult
from ...answer_contract import (
    coerce_answer_type_hint,
    extract_contract_response,
    validate_contract_response,
)


class GeneralChatConversationExecutionHelpers:
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
                    timeout_seconds=max(6.0, float(os.getenv("LOCAL_AI_AUX_TIMEOUT_SECONDS", "25") or "25")),
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
        continuation_prompt = (
            "Continue only the unfinished tail of the previous answer.\n"
            "Do not restart from the beginning.\n"
            "Do not repeat already written content.\n"
            f"Language: {context.response_language}\n"
            f"User request: {query}\n"
            f"Previous answer:\n{base_text}\n\n"
            "Continuation:"
        )
        try:
            continuation = await strategy._run_conversation_inference(
                executor=executor,
                query=continuation_prompt,
                context=context,
                max_tokens=max(192, min(max(640, int(base_max_tokens * 0.8)), int(base_max_tokens))),
                generation_style="rewrite",
                sampling_overrides={"temperature": 0.22, "top_p": 0.82, "top_k": 16, "repeat_penalty": 1.12},
                timeout_seconds=max(4.0, min(14.0, float(timeout_seconds))),
                session_summary_override=session_summary_override,
                style_profile=style_profile,
            )
            if not strategy._conversation_answer_ready(continuation):
                trimmed = strategy._trim_incomplete_tail(base_text)
                if trimmed != base_text:
                    logs = [*list(execution.tool_logs or []), "recovery:trim_incomplete_tail"]
                    return execution.model_copy(update={"generated_text": trimmed, "tool_logs": logs})
                return execution
            tail_text = str(continuation.generated_text or "").strip()
            if not tail_text:
                trimmed = strategy._trim_incomplete_tail(base_text)
                if trimmed != base_text:
                    logs = [*list(execution.tool_logs or []), "recovery:trim_incomplete_tail"]
                    return execution.model_copy(update={"generated_text": trimmed, "tool_logs": logs})
                return execution
            merged = strategy._merge_continuation(base_text, tail_text)
            if merged == base_text:
                return execution
            logs = [*list(execution.tool_logs or []), "recovery:continuation_append"]
            return execution.model_copy(update={"generated_text": merged, "tool_logs": logs})
        except asyncio.CancelledError:
            raise
        except Exception:
            trimmed = strategy._trim_incomplete_tail(base_text)
            if trimmed != base_text:
                logs = [*list(execution.tool_logs or []), "recovery:trim_incomplete_tail"]
                return execution.model_copy(update={"generated_text": trimmed, "tool_logs": logs})
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
        }
        decode_profile = strategy._decode_profile_with_style(
            fallback_profile=strategy._conversation_decode_profile(query=context.req.query),
            style_profile=style_profile,
        )
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
        retry_floor = strategy._retry_min_tokens_floor()
        retry_cap = strategy._retry_tokens_cap()
        split_cap = strategy._split_answer_tokens_cap()
        retry_cap = max(retry_cap, int(base_max_tokens))
        split_cap = max(split_cap, int(base_max_tokens))
        for idx, scale in enumerate(backoff):
            max_tokens = min(retry_cap, max(retry_floor, int(base_max_tokens * scale)))
            timeout = timeouts[min(idx, len(timeouts) - 1)]
            if idx <= 0:
                sampling = strategy._sampling_for_attempt(
                    profile=decode_profile,
                    attempt_index=idx,
                    generation_style="conversation",
                )
                style = "conversation"
            elif idx == 1:
                sampling = strategy._sampling_for_attempt(
                    profile=decode_profile,
                    attempt_index=idx,
                    generation_style="conversation",
                )
                style = "conversation"
            else:
                sampling = strategy._sampling_for_attempt(
                    profile=decode_profile,
                    attempt_index=idx,
                    generation_style="rewrite",
                )
                style = "rewrite"
            try:
                execution = await strategy._run_conversation_inference(
                    executor=executor,
                    query=base_query,
                    context=context,
                    max_tokens=max_tokens,
                    generation_style=generation_style if idx <= 1 else "rewrite",
                    sampling_overrides=sampling,
                    timeout_seconds=timeout,
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
                        timeout_seconds=timeout,
                        session_summary_override=session_summary_override,
                        style_profile=style_profile,
                    )
                    execution.tool_logs.append(f"recovery:attempt={idx + 1};max_tokens={max_tokens};style={style}")
                    if idx > 0:
                        metadata["recovery_path"] = "token_backoff"
                    return execution, metadata
                last_failure_detail = str(execution.runtime_detail or "").strip()
            except asyncio.TimeoutError:
                last_failure_detail = f"inference_timeout:attempt{idx + 1}"
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                last_failure_detail = str(exc)

        # Attempt 4: split generation (bridge + answer), both LLM-generated.
        try:
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
                    timeout_seconds=split_timeout,
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
            last_failure_detail = str(exc)

        no_fallback_mode = strategy._no_fallback_mode_enabled()
        if strategy._retry_until_cancel_enabled() or no_fallback_mode:
            steady_scale = backoff[-1] if backoff else 0.45
            steady_tokens = min(retry_cap, max(retry_floor, int(base_max_tokens * steady_scale)))
            steady_timeout = timeouts[-1] if timeouts else 12.0
            steady_sampling = {"temperature": 0.28, "top_p": 0.84, "top_k": 20}
            steady_started = time.time()
            max_extra_attempts = 1_000_000_000 if no_fallback_mode else strategy._retry_until_cancel_max_attempts()
            max_extra_seconds = float("inf") if no_fallback_mode else strategy._retry_until_cancel_max_seconds()
            extra_attempt = 0
            while extra_attempt < max_extra_attempts and (time.time() - steady_started) < max_extra_seconds:
                extra_attempt += 1
                try:
                    execution = await strategy._run_conversation_inference(
                        executor=executor,
                        query=base_query,
                        context=context,
                        max_tokens=steady_tokens,
                        generation_style="rewrite",
                        sampling_overrides=steady_sampling,
                        timeout_seconds=steady_timeout,
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
                            timeout_seconds=steady_timeout,
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
                except asyncio.TimeoutError:
                    last_failure_detail = f"inference_timeout:steady{extra_attempt}"
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    last_failure_detail = str(exc)
                await asyncio.sleep(0.15)
            if no_fallback_mode:
                # Safety net: keep waiting until caller cancellation rather than emitting fallback text.
                while True:
                    await asyncio.sleep(0.5)
            metadata["degraded_internal"] = True
            metadata["retry_until_cancel_exhausted"] = True

        # Do not emit synthetic fallback chat text on exhaustion.
        # Return a runtime error and let UI provide explicit regenerate action.
        metadata["generation_retry_count"] = len(backoff) + 1
        metadata["recovery_path"] = "regenerate_required"
        metadata["degraded_internal"] = True
        if context.response_language == "ja":
            regenerate_message = "ローカル生成が時間内に完了しませんでした。『再生成』で再試行してください。"
        elif context.response_language == "en":
            regenerate_message = "Local generation did not finish in time. Use \"Regenerate\" to retry."
        else:
            regenerate_message = "로컬 생성이 시간 내 완료되지 않았습니다. '응답 다시 생성'으로 재시도해 주세요."
        return (
            ExecutionResult(
                result_type="conversation",
                structured_payload={
                    "style": "runtime_error",
                    "reason": "generation_retry_exhausted",
                    "ungrounded_allowed": True,
                    "offer_regenerate": True,
                    "regenerate_prompt": str(query or "").strip(),
                },
                citations=[],
                tool_logs=["runtime_error:generation_retry_exhausted"],
                generated_text=regenerate_message,
                engine_used=context.settings.local_engine,
                used_fallback=False,
                runtime_detail=last_failure_detail or "generation_retry_exhausted",
            ),
            metadata,
        )
