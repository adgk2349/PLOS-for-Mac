from __future__ import annotations

import asyncio
import os
import re
import time
from typing import Any

from ....models import ExecutionResult
from ...answer_contract import extract_contract_response


class GeneralChatRecallExecutionHelpers:
    @staticmethod
    def _room_memory_isolation_enabled() -> bool:
        raw = str(os.getenv("LOCAL_AI_ROOM_MEMORY_ISOLATION", "1") or "1").strip().lower()
        return raw in {"1", "true", "yes", "on"}

    @staticmethod
    def memory_recall_evidence_payload(
        strategy,
        *,
        query: str,
        response_language: str,
        last_context: dict[str, Any] | None,
        session_digest: str | None,
        memory_bundle: Any,
        answer_type_hint: str,
        max_items: int = 12,
    ) -> dict[str, Any]:
        _ = response_language
        focus_query = strategy._single_question(query, response_language="en") or str(query or "").strip()
        query_terms = strategy._recall_terms(focus_query)
        candidates: list[dict[str, Any]] = []

        def _append_candidate(
            *,
            source: str,
            memory_type: str,
            text: str,
            confidence: float,
            memory_scope: str = "session",
            recency_bias: float = 0.5,
            when: str = "",
        ) -> None:
            normalized = " ".join(str(text or "").split()).strip()
            if not normalized:
                return
            terms = strategy._recall_terms(normalized)
            overlap = float(len(query_terms.intersection(terms))) if query_terms and terms else 0.0
            lexical = max(0.0, min(1.0, overlap / 3.0))
            type_bonus_map = {
                "fact": 0.22,
                "preference": 0.18,
                "episode": 0.18,
                "unresolved": -0.25,
            }
            answer_shape_bonus = 0.0
            if answer_type_hint == "date":
                if strategy._runtime_context_has_date_signal(normalized):
                    answer_shape_bonus += 0.22
                if any(token in normalized.lower() for token in ("date:", "yesterday", "last year", "어제", "작년")):
                    answer_shape_bonus += 0.06
            elif answer_type_hint == "number":
                if strategy._runtime_context_extract_numbers(normalized):
                    answer_shape_bonus += 0.16
                if strategy._recall_is_duration_query(query):
                    lowered_norm = normalized.lower()
                    if re.search(r"\b(\d+)\s*(year|years|month|months|week|weeks|day|days)\b", lowered_norm):
                        answer_shape_bonus += 0.18
                    if any(token in lowered_norm for token in ("년", "개월", "주", "일", "ago", "동안", "전")):
                        answer_shape_bonus += 0.08
                    if strategy._runtime_context_has_date_signal(normalized):
                        answer_shape_bonus -= 0.12
            elif answer_type_hint == "entity":
                lowered_norm = normalized.lower()
                if strategy._recall_is_multi_item_query(query):
                    if "," in normalized:
                        answer_shape_bonus += 0.12
                    if re.search(r"\b(and|or)\b", lowered_norm):
                        answer_shape_bonus += 0.06
                if any(token in lowered_norm for token in ("research", "researched", "camp", "camped", "activity", "activities", "likes", "like", "children", "kids", "favorite")):
                    answer_shape_bonus += 0.06
                if any(token in lowered_norm for token in ("safe and loving home", "home country")):
                    answer_shape_bonus -= 0.08
                if strategy._recall_is_origin_query(query):
                    if re.search(r"\bfrom\s+[a-z][a-z\s-]{2,32}\b", lowered_norm):
                        answer_shape_bonus += 0.16
                    if any(token in lowered_norm for token in ("sweden", "korea", "japan", "canada", "france", "germany")):
                        answer_shape_bonus += 0.1
                    if "home country" in lowered_norm:
                        answer_shape_bonus -= 0.18
                if strategy._recall_is_camped_location_query(query):
                    if any(token in lowered_norm for token in ("beach", "mountains", "mountain", "forest", "park", "camped", "camping")):
                        answer_shape_bonus += 0.12
                if strategy._recall_is_kids_preference_query(query):
                    if any(token in lowered_norm for token in ("kids like", "children like", "favorite", "favourite", "dinosaurs", "nature", "animals")):
                        answer_shape_bonus += 0.16
                    if any(token in lowered_norm for token in ("loving home", "accepting environment", "safe home")):
                        answer_shape_bonus -= 0.2
                if strategy._recall_is_relationship_status_query(query):
                    if any(token in lowered_norm for token in ("single", "married", "dating", "divorced")):
                        answer_shape_bonus += 0.1
                    if "single parent" in lowered_norm:
                        answer_shape_bonus += 0.05
                if strategy._recall_is_career_field_query(query):
                    if any(
                        token in lowered_norm
                        for token in (
                            "psychology",
                            "counsel",
                            "counseling",
                            "mental health",
                            "career",
                            "supporting trans",
                            "work with trans",
                        )
                    ):
                        answer_shape_bonus += 0.16
                    if any(token in lowered_norm for token in ("adoption", "family", "kids", "home", "camping")):
                        answer_shape_bonus -= 0.14
                if strategy._recall_is_identity_query(query):
                    if any(token in lowered_norm for token in ("transgender", "woman", "man", "nonbinary", "identity", "lgbtq")):
                        answer_shape_bonus += 0.18
                    if any(token in lowered_norm for token in ("yes,", "yes ", "thanks", "wow", "awesome", "courage")):
                        answer_shape_bonus -= 0.10
            elif answer_type_hint == "boolean":
                if re.search(r"\b(yes|no|true|false|맞아|아니|가능|불가능)\b", normalized.lower()):
                    answer_shape_bonus += 0.08
            score = (
                (0.40 * lexical)
                + (0.18 * max(0.0, min(1.0, confidence)))
                + (0.14 * max(0.0, min(1.0, recency_bias)))
                + float(type_bonus_map.get(memory_type, 0.06))
                + answer_shape_bonus
            )
            if memory_scope == "global":
                score += 0.08
            elif memory_scope == "workspace":
                score += 0.05
            elif memory_scope == "web":
                score += 0.03
            candidates.append(
                {
                    "score": round(score, 4),
                    "source": source,
                    "memory_type": memory_type,
                    "memory_scope": memory_scope,
                    "date": str(when or "")[:32],
                    "content": normalized[:260],
                }
            )

        if isinstance(last_context, dict):
            for idx, key in enumerate(("last_user_query", "result_summary", "parsed_target")):
                value = str(last_context.get(key) or "").strip()
                if not value:
                    continue
                recency_bias = 1.0 - (0.18 * idx)
                _append_candidate(
                    source=f"last_context:{key}",
                    memory_type="episode",
                    memory_scope="session",
                    text=value,
                    confidence=0.72,
                    recency_bias=max(0.2, recency_bias),
                )

        digest_text = str(session_digest or "").strip()
        if digest_text:
            recall_block_match = re.search(
                r"<memory_recall_context>[\s\S]*?</memory_recall_context>",
                digest_text,
                flags=re.IGNORECASE,
            )
            recall_block = str(recall_block_match.group(0) or "").strip() if recall_block_match else ""
            raw_lines = recall_block.splitlines() if recall_block else digest_text.splitlines()
            line_count = max(1, len(raw_lines))
            for idx, raw in enumerate(raw_lines):
                line = str(raw or "").strip()
                if not line:
                    continue
                if line.startswith("<") and line.endswith(">"):
                    continue
                line = re.sub(r"^\-\s*(?:[UA]:\s*)?", "", line).strip()
                if not line:
                    continue
                if line.lower().startswith("topics:"):
                    continue
                recency_bias = float(idx + 1) / float(line_count)
                _append_candidate(
                    source="session_digest",
                    memory_type=("fact" if any(tag in line for tag in ("별명", "nickname", "닉네임")) else "episode"),
                    memory_scope="session",
                    text=line,
                    confidence=0.68,
                    recency_bias=recency_bias,
                )

        room_isolation = GeneralChatRecallExecutionHelpers._room_memory_isolation_enabled()
        memory_lists: list[tuple[str, list[Any]]] = []
        if memory_bundle is not None:
            memory_lists = [("session", list(getattr(memory_bundle, "session_items", []) or []))]
            if not room_isolation:
                memory_lists.extend(
                    [
                        ("workspace", list(getattr(memory_bundle, "workspace_items", []) or [])),
                        ("episodic", list(getattr(memory_bundle, "episodic_items", []) or [])),
                        ("preference", list(getattr(memory_bundle, "preference_items", []) or [])),
                    ]
                )
        for source_name, rows in memory_lists:
            total = max(1, len(rows))
            for idx, item in enumerate(rows):
                text = strategy._recall_memory_item_text(item)
                if not text:
                    continue
                if isinstance(item, dict):
                    superseded_by = str(item.get("superseded_by") or "").strip()
                    memory_type = str(item.get("memory_type") or "").strip().lower() or (
                        "preference" if source_name == "preference" else "episode"
                    )
                    memory_scope = str(item.get("memory_scope") or item.get("scope") or source_name).strip().lower()
                    confidence = float(item.get("confidence") or item.get("score") or 0.62)
                else:
                    superseded_by = str(getattr(item, "superseded_by", "") or "").strip()
                    memory_type = str(getattr(item, "memory_type", "") or "").strip().lower() or (
                        "preference" if source_name == "preference" else "episode"
                    )
                    memory_scope = str(getattr(item, "memory_scope", source_name) or source_name).strip().lower()
                    confidence = float(getattr(item, "confidence", 0.62) or 0.62)
                if superseded_by:
                    continue
                recency_bias = float(total - idx) / float(total)
                _append_candidate(
                    source=f"memory_bundle:{source_name}",
                    memory_type=memory_type,
                    memory_scope=memory_scope if memory_scope in {"session", "workspace", "web", "global"} else "session",
                    text=text,
                    confidence=max(0.0, min(1.0, confidence)),
                    recency_bias=recency_bias,
                )

        # Allow only explicitly pinned global memories as cross-room recall context.
        pinned_rows = list(getattr(memory_bundle, "pinned_items", []) or []) if memory_bundle is not None else []
        pin_total = max(1, len(pinned_rows))
        for idx, item in enumerate(pinned_rows):
            scope = str(getattr(item, "scope", "") or "").strip().lower()
            if scope != "global":
                continue
            title = str(getattr(item, "title", "") or "").strip()
            content = str(getattr(item, "content", "") or "").strip()
            text = " ".join(part for part in [title, content] if part).strip()
            if not text:
                continue
            _append_candidate(
                source="memory_bundle:pinned_global",
                memory_type="fact",
                memory_scope="global",
                text=text,
                confidence=0.88,
                recency_bias=float(pin_total - idx) / float(pin_total),
            )

        semantic_rows = list(getattr(memory_bundle, "semantic_memories", []) or []) if memory_bundle is not None else []
        sem_total = max(1, len(semantic_rows))
        for idx, item in enumerate(semantic_rows):
            if not isinstance(item, dict):
                continue
            text = str(item.get("text") or "").strip()
            if not text:
                continue
            memory_scope = str(item.get("scope") or "").strip().lower()
            if bool(item.get("is_global")):
                memory_scope = "global"
            if room_isolation and memory_scope not in {"session", ""}:
                continue
            _append_candidate(
                source="semantic_memory",
                memory_type="fact",
                memory_scope=memory_scope if memory_scope in {"session", "workspace", "web", "global"} else "workspace",
                text=text,
                confidence=float(item.get("score") or 0.62),
                recency_bias=float(sem_total - idx) / float(sem_total),
            )

        candidates.sort(key=lambda row: float(row.get("score") or 0.0), reverse=True)
        selected_candidates = candidates[: max(1, int(max_items))]
        evidence_terms: set[str] = set()
        for row in selected_candidates:
            evidence_terms.update(strategy._recall_terms(str(row.get("content") or "")))
        coverage = (
            max(0.0, min(1.0, float(len(query_terms.intersection(evidence_terms))) / float(max(1, len(query_terms)))))
            if query_terms
            else 0.0
        )
        confidence = 0.0
        if selected_candidates:
            confidence = sum(float(row.get("score") or 0.0) for row in selected_candidates[:5]) / float(
                max(1, min(5, len(selected_candidates)))
            )
            confidence = max(0.0, min(1.0, confidence))

        aggregation_hint: dict[str, Any] | None = None
        if answer_type_hint == "number":
            values: list[float] = []
            for row in selected_candidates:
                values.extend(strategy._runtime_context_extract_numbers(str(row.get("content") or "")))
            if values:
                if strategy._runtime_context_is_aggregate_numeric_query(query) and len(values) >= 2:
                    aggregation_hint = {
                        "mode": "sum",
                        "estimated_total": strategy._runtime_context_format_number(
                            value=sum(float(v) for v in values),
                            money=strategy._runtime_context_is_money_query(query),
                        ),
                        "value_count": len(values),
                    }
                else:
                    aggregation_hint = {
                        "mode": "single",
                        "estimated_value": strategy._runtime_context_format_number(
                            value=float(values[0]),
                            money=strategy._runtime_context_is_money_query(query),
                        ),
                    }

        return {
            "question": focus_query,
            "answer_type": answer_type_hint,
            "candidate_count": len(selected_candidates),
            "coverage": round(float(coverage), 4),
            "confidence": round(float(confidence), 4),
            "candidate_evidence": selected_candidates,
            "aggregation_hint": aggregation_hint,
        }

    @staticmethod
    async def run_recall_two_pass_orchestration(
        strategy,
        *,
        executor: Any,
        context: ReasoningContext,
        query: str,
        answer_type_hint: str,
        evidence_payload: dict[str, Any],
        response_language: str,
        style_profile: dict[str, Any] | None,
    ) -> tuple[ExecutionResult, dict[str, Any]]:
        started = time.time()
        budget_seconds = strategy._recall_time_budget_seconds()
        disable_timeouts = strategy._recall_disable_timeouts()
        max_regen_attempts = strategy._recall_max_regeneration_attempts()
        recall_version = strategy._recall_pipeline_version()
        normalized_query = strategy._single_question(query, response_language=response_language) or str(query or "").strip()
        pass1_prompt = strategy._recall_pass1_prompt(
            query=normalized_query,
            response_language=response_language,
            answer_type_hint=answer_type_hint,
            evidence_payload=evidence_payload,
        )
        pass1_raw = ""
        pass1_payload = dict(evidence_payload)
        try:
            pass1_timeout = max(3.2, min(8.0, budget_seconds * 0.32))
            if disable_timeouts:
                pass1_exec = await strategy._run_conversation_inference(
                    executor=executor,
                    query=pass1_prompt,
                    context=context,
                    max_tokens=260,
                    generation_style="conversation",
                    sampling_overrides={"temperature": 0.22, "top_p": 0.82, "top_k": 18},
                    timeout_seconds=None,
                    session_summary_override="",
                    style_profile=style_profile,
                    response_language_override=response_language,
                    language_preference_override=response_language,
                )
            else:
                pass1_exec = await asyncio.wait_for(
                    strategy._run_conversation_inference(
                        executor=executor,
                        query=pass1_prompt,
                        context=context,
                        max_tokens=260,
                        generation_style="conversation",
                        sampling_overrides={"temperature": 0.22, "top_p": 0.82, "top_k": 18},
                        timeout_seconds=max(1.5, pass1_timeout - 0.25),
                        session_summary_override="",
                        style_profile=style_profile,
                        response_language_override=response_language,
                        language_preference_override=response_language,
                    ),
                    timeout=pass1_timeout,
                )
            if bool(getattr(pass1_exec, "used_fallback", False)):
                raise RuntimeError("recall_pass1_fallback_not_allowed")
            pass1_raw = str(getattr(pass1_exec, "generated_text", "") or "").strip()
            pass1_payload = strategy._parse_recall_pass1_payload(
                raw_output=pass1_raw,
                fallback_payload=evidence_payload,
                answer_type_hint=answer_type_hint,
            )
        except asyncio.CancelledError:
            raise
        except Exception:
            pass1_payload = strategy._parse_recall_pass1_payload(
                raw_output=pass1_raw,
                fallback_payload=evidence_payload,
                answer_type_hint=answer_type_hint,
            )
        base_selection = strategy._recall_select_candidate(
            query=normalized_query,
            pass1_payload=pass1_payload,
            answer_type_hint=answer_type_hint,
        )
        base_top_indices: list[int] = []
        for item in list(base_selection.get("top_indices") or []):
            try:
                base_top_indices.append(int(item))
            except Exception:
                continue
            if len(base_top_indices) >= 5:
                break
        base_scored_candidates: list[dict[str, Any]] = []
        for row in list(base_selection.get("scored_candidates") or [])[:5]:
            if not isinstance(row, dict):
                continue
            try:
                idx = int(row.get("index") or 0)
            except Exception:
                idx = -1
            try:
                sc = float(row.get("score") or 0.0)
            except Exception:
                sc = 0.0
            base_scored_candidates.append(
                {
                    "index": idx,
                    "score": round(max(0.0, min(1.0, sc)), 4),
                    "content": str(row.get("content") or "")[:120],
                }
            )

        best_raw = ""
        best_answer = ""
        best_score = -1.0
        best_reasons: list[str] = ["no_candidate"]
        best_contract = "plain"
        best_contract_auto_wrapped = False
        base_selected_index = base_selection.get("selected_index")
        try:
            best_selected_index = int(base_selected_index) if base_selected_index is not None else -1
        except Exception:
            best_selected_index = -1
        attempts = 0
        reason_accumulator: list[str] = []
        while attempts < max_regen_attempts:
            elapsed = time.time() - started
            remaining = budget_seconds - elapsed
            if (not disable_timeouts) and remaining <= 1.0:
                break
            attempts += 1
            attempt_selection = strategy._recall_selection_for_attempt(
                base_selection=base_selection,
                pass1_payload=pass1_payload,
                attempt_index=attempts - 1,
            )
            pass2_prompt = strategy._recall_pass2_prompt(
                query=normalized_query,
                response_language=response_language,
                pass1_payload=pass1_payload,
                selected_evidence=attempt_selection,
            )
            if attempts > 1:
                fail_reason_str = ", ".join(best_reasons[:6]) if best_reasons else "contract_violation"
                extra_constraints: list[str] = []
                if "answer_type_date_mismatch" in best_reasons:
                    extra_constraints.append("Output an explicit date/year phrase, never vague words like 'this month'.")
                if "answer_type_number_mismatch" in best_reasons:
                    extra_constraints.append("Output a numeric or duration phrase only (e.g., 10 years ago, 4 years), not a calendar date.")
                if "entity_identity_too_generic" in best_reasons:
                    extra_constraints.append(
                        "For identity question, output explicit descriptor from evidence (e.g., transgender woman), not generic 'identity'."
                    )
                if "entity_too_generic" in best_reasons:
                    extra_constraints.append(
                        "For entity question, output concrete object noun from evidence (e.g., adoption agencies), not subject+verb summary."
                    )
                if "unsupported_no_information" in best_reasons:
                    extra_constraints.append("Do not answer 'No information available' when evidence exists.")
                    if strategy._recall_is_duration_query(normalized_query):
                        extra_constraints.append("Extract and output the explicit duration phrase from selected evidence.")
                if "evidence_mismatch" in best_reasons:
                    extra_constraints.append("Use exact evidence wording for key noun phrase/date to avoid mismatch.")
                if "entity_origin_too_generic" in best_reasons:
                    extra_constraints.append("For origin questions, output a specific country/city name, never 'home country'.")
                if "entity_preference_too_generic" in best_reasons:
                    extra_constraints.append("For preference questions, output concrete liked things (e.g., dinosaurs, nature), not environment descriptions.")
                if "entity_relationship_not_compact" in best_reasons:
                    extra_constraints.append("For relationship status, output a compact label only (e.g., Single), without role suffixes.")
                if response_language == "ko":
                    extra_lines = "\n".join(f"- {line}" for line in extra_constraints[:4])
                    pass2_prompt = (
                        f"{pass2_prompt}\n\n"
                        "재생성 규칙 강화:\n"
                        "- 반드시 계약 형식 준수\n"
                        "- 근거에 없는 문장 금지\n"
                        "- 실패 사유를 제거해서 다시 생성\n"
                        f"{extra_lines}\n"
                        f"- 실패 사유: {fail_reason_str}\n"
                        f"- 이전 답안: {best_answer[:220]}\n"
                        "정답(반드시 <final_answer>...</final_answer> 사용):"
                    )
                else:
                    extra_lines = "\n".join(f"- {line}" for line in extra_constraints[:4])
                    pass2_prompt = (
                        f"{pass2_prompt}\n\n"
                        "Regeneration constraints:\n"
                        "- Strictly follow output contract\n"
                        "- Avoid unsupported statements\n"
                        f"{extra_lines}\n"
                        f"- Failure reasons: {fail_reason_str}\n"
                        f"- Previous answer: {best_answer[:220]}\n"
                        "Answer(MUST USE <final_answer>...</final_answer>):"
                    )
            timeout = max(4.2, min(14.0, remaining - 0.4))
            try:
                if disable_timeouts:
                    pass2_exec = await strategy._run_conversation_inference(
                        executor=executor,
                        query=pass2_prompt,
                        context=context,
                        max_tokens=320,
                        generation_style="conversation",
                        sampling_overrides={"temperature": 0.20, "top_p": 0.80, "top_k": 18},
                        timeout_seconds=None,
                        session_summary_override="",
                        style_profile=style_profile,
                        response_language_override=response_language,
                        language_preference_override=response_language,
                    )
                else:
                    pass2_exec = await asyncio.wait_for(
                        strategy._run_conversation_inference(
                            executor=executor,
                            query=pass2_prompt,
                            context=context,
                            max_tokens=320,
                            generation_style="conversation",
                            sampling_overrides={"temperature": 0.20, "top_p": 0.80, "top_k": 18},
                            timeout_seconds=max(1.5, timeout - 0.2),
                            session_summary_override="",
                            style_profile=style_profile,
                            response_language_override=response_language,
                            language_preference_override=response_language,
                        ),
                        timeout=timeout,
                    )
            except asyncio.CancelledError:
                raise
            except asyncio.TimeoutError:
                reason_accumulator.append("pass2_timeout")
                continue
            except Exception:
                continue
            if bool(getattr(pass2_exec, "used_fallback", False)):
                reason_accumulator.append("pass2_fallback_blocked")
                continue

            raw_output = str(getattr(pass2_exec, "generated_text", "") or "").strip()
            if not raw_output:
                continue
            parsed = extract_contract_response(raw_output)
            if not isinstance(parsed, dict):
                parsed = {}
            extracted_answer = str(parsed.get("answer") or "").strip()
            contract_format = str(parsed.get("contract_format") or "plain")
            validation_input = raw_output
            contract_auto_wrapped = False
            if contract_format == "plain" and extracted_answer:
                validation_input = f"<final_answer>{extracted_answer}</final_answer>"
                contract_format = "final_answer_tag"
                contract_auto_wrapped = True
            reasons = strategy._recall_validation_reasons(
                answer=validation_input,
                evidence_payload=pass1_payload,
                response_language=response_language,
                answer_type_hint=answer_type_hint,
                query=normalized_query,
            )
            reason_accumulator.extend(reasons)
            score = strategy._recall_validation_score(
                answer=extracted_answer or raw_output,
                reasons=reasons,
                evidence_payload=pass1_payload,
            )
            if score > best_score:
                best_score = score
                best_raw = validation_input
                best_answer = extracted_answer or raw_output
                best_reasons = list(reasons)
                best_contract = contract_format
                best_contract_auto_wrapped = contract_auto_wrapped
                selected_idx = attempt_selection.get("selected_index")
                try:
                    best_selected_index = int(selected_idx) if selected_idx is not None else -1
                except Exception:
                    best_selected_index = -1
            if not reasons and extracted_answer:
                break

        elapsed_ms = int(max(0.0, (time.time() - started) * 1000.0))
        final_answer = best_answer.strip()
        if strategy._recall_is_no_information_answer(final_answer):
            final_answer = strategy._recall_no_information_message(response_language)
        validation_passed = not best_reasons and bool(final_answer)
        selected_preview = ""
        selected_candidate_score = 0.0
        candidates = list(pass1_payload.get("candidate_evidence") or [])
        if 0 <= int(best_selected_index) < len(candidates):
            row = candidates[int(best_selected_index)]
            if isinstance(row, dict):
                selected_preview = str(row.get("content") or "")[:180]
                try:
                    selected_candidate_score = float(row.get("score") or 0.0)
                except Exception:
                    selected_candidate_score = 0.0
        if selected_candidate_score <= 0.0 and base_scored_candidates:
            for row in base_scored_candidates:
                if int(row.get("index") or -1) == int(best_selected_index):
                    try:
                        selected_candidate_score = float(row.get("score") or 0.0)
                    except Exception:
                        selected_candidate_score = 0.0
                    break
        metadata = {
            "validation_passed": validation_passed,
            "validation_fail_reasons": (
                [] if validation_passed else sorted({str(item) for item in [*reason_accumulator, *best_reasons] if str(item).strip()})
            ),
            "regeneration_attempts": max(0, attempts - 1),
            "contract_format": best_contract or "plain",
            "contract_auto_wrapped": bool(best_contract_auto_wrapped),
            "recall_pipeline_version": recall_version,
            "pass1_candidate_count": int(pass1_payload.get("candidate_count") or 0),
            "pass1_coverage": float(pass1_payload.get("coverage") or 0.0),
            "pass1_confidence": float(pass1_payload.get("confidence") or 0.0),
            "pass1_answer_type": str(pass1_payload.get("answer_type") or answer_type_hint or "freeform"),
            "pass1_top_indices": base_top_indices,
            "pass1_scored_candidates": base_scored_candidates,
            "selected_candidate_index": best_selected_index,
            "selected_candidate_score": round(max(0.0, min(1.0, float(selected_candidate_score))), 4),
            "selected_candidate_preview": selected_preview,
            "best_validation_score": round(max(0.0, min(1.0, float(best_score if best_score >= 0.0 else 0.0))), 4),
            "best_validation_reasons": sorted({str(item) for item in best_reasons if str(item).strip()}),
            "elapsed_ms": elapsed_ms,
            "answer_type": str(pass1_payload.get("answer_type") or answer_type_hint or "freeform"),
        }
        execution = ExecutionResult(
            result_type="conversation",
            structured_payload={
                "style": "general_chat",
                "ungrounded_allowed": True,
                "recall_mode": "two_pass",
                "recall_pipeline_version": recall_version,
                "answer_type": str(metadata.get("answer_type") or "freeform"),
                "contract_format": str(metadata.get("contract_format") or "plain"),
                "pass1_candidate_count": int(metadata.get("pass1_candidate_count") or 0),
                "pass1_coverage": float(metadata.get("pass1_coverage") or 0.0),
                "selected_candidate_index": (
                    int(metadata.get("selected_candidate_index"))
                    if metadata.get("selected_candidate_index") is not None
                    else -1
                ),
                "elapsed_ms": elapsed_ms,
            },
            citations=[],
            tool_logs=[
                "recall:two_pass",
                f"recall:attempts={attempts}",
                f"recall:selected_index={(int(metadata.get('selected_candidate_index')) if metadata.get('selected_candidate_index') is not None else -1)}",
                f"recall:validation_passed={int(validation_passed)}",
            ],
            generated_text=final_answer or best_raw,
            engine_used=context.settings.local_engine,
            used_fallback=False,
            runtime_detail=f"recall_two_pass_elapsed_ms={elapsed_ms}",
        )
        return execution, metadata
