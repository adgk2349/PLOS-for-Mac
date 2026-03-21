from __future__ import annotations

from pathlib import Path
import re
from typing import Any

from .language_utils import insufficient_evidence_message
from .models import (
    ActionExecutionMode,
    BehaviorPolicy,
    ChatIntent,
    Citation,
    ComposedChatResponseV2,
    ExecutionResult,
    LocalPlan,
    ParsedIntent,
    ReasoningIntent,
    StructuredResult,
    SuggestedAction,
    SuggestedActionKind,
    VerificationResult,
    WorkMode,
)


class ResponseComposer:
    _FILE_SEARCH_KEYWORDS = (
        "찾아",
        "파일",
        "문서",
        "어디",
        "경로",
        "열어",
        "find",
        "file",
        "document",
        "path",
        "locate",
        "where",
    )
    _TASK_REQUEST_KEYWORDS = (
        "요약해",
        "정리해",
        "비교",
        "작성",
        "만들어",
        "계획",
        "summarize",
        "summary",
        "compare",
        "draft",
        "write",
        "plan",
        "organize",
    )
    _KOREAN_ENDING_REGEX = r"(습니다|입니다|해요|해줘요|해줄게요|할게요|이에요|예요|됐어요|됐습니다)[.!?]?$"

    def classify_intent(self, query: str, mode: WorkMode, citations: list[Citation]) -> ChatIntent:
        text = (query or "").strip()
        lowered = text.lower()
        if not text:
            return ChatIntent.AMBIGUOUS

        if mode in {WorkMode.SUMMARY, WorkMode.PLANNING, WorkMode.WRITING}:
            return ChatIntent.TASK_REQUEST

        if any(keyword in lowered for keyword in self._TASK_REQUEST_KEYWORDS):
            return ChatIntent.TASK_REQUEST

        if any(keyword in lowered for keyword in self._FILE_SEARCH_KEYWORDS):
            return ChatIntent.FILE_SEARCH

        if len(text) <= 4 and not citations:
            return ChatIntent.AMBIGUOUS

        return ChatIntent.DOCUMENT_QA

    def compose(
        self,
        *,
        query: str,
        mode: WorkMode,
        response_language: str,
        citations: list[Citation],
        result_summary: str,
        insufficient: bool = False,
    ) -> tuple[ChatIntent, str, str, list[SuggestedAction]]:
        intent = self.classify_intent(query=query, mode=mode, citations=citations)
        lead = self._lead(intent=intent, citations=citations, response_language=response_language, insufficient=insufficient)
        if insufficient:
            summary = insufficient_evidence_message(response_language)
        else:
            summary = result_summary.strip()
        actions = self._actions(
            intent=intent,
            response_language=response_language,
            query=query,
            citations=citations,
            insufficient=insufficient,
        )
        return intent, lead, summary, actions

    def _lead(
        self,
        *,
        intent: ChatIntent,
        citations: list[Citation],
        response_language: str,
        insufficient: bool,
    ) -> str:
        if response_language == "ko":
            if insufficient:
                return "현재 자료에서는 답변 근거가 부족합니다."
            if not citations:
                return "관련 근거를 추가로 찾지 못했습니다."
            top_name = Path(citations[0].file_path).name
            if intent == ChatIntent.FILE_SEARCH:
                return f"관련 파일 {len(citations)}개를 찾았습니다. 가장 관련성이 높은 파일은 {top_name}입니다."
            if intent == ChatIntent.TASK_REQUEST:
                return f"요청 작업에 맞는 근거 {len(citations)}개를 찾았습니다. 바로 이어서 처리할 수 있습니다."
            if intent == ChatIntent.AMBIGUOUS:
                return f"질문을 여러 방식으로 해석할 수 있습니다. 우선 관련 근거 {len(citations)}개를 수집했습니다."
            return f"질문과 관련된 근거 {len(citations)}개를 바탕으로 정리했습니다."

        if insufficient:
            return "There is not enough grounded evidence in the current sources."
        if not citations:
            return "I could not find enough relevant evidence."
        top_name = Path(citations[0].file_path).name
        if intent == ChatIntent.FILE_SEARCH:
            return f"I found {len(citations)} related files. The top match is {top_name}."
        if intent == ChatIntent.TASK_REQUEST:
            return f"I found {len(citations)} pieces of evidence for this task and can continue from here."
        if intent == ChatIntent.AMBIGUOUS:
            return f"The request can be interpreted in multiple ways. I first gathered {len(citations)} relevant sources."
        return f"I summarized your question from {len(citations)} grounded citations."

    def _actions(
        self,
        *,
        intent: ChatIntent,
        response_language: str,
        query: str,
        citations: list[Citation],
        insufficient: bool,
    ) -> list[SuggestedAction]:
        actions: list[SuggestedAction] = []

        def label(ko: str, en: str) -> str:
            return ko if response_language == "ko" else en

        if citations:
            top = citations[0]
            top_name = Path(top.file_path).name
            actions.append(
                SuggestedAction(
                    action_id="open_top_file",
                    kind=SuggestedActionKind.OPEN_FILE,
                    label=label("파일 열기", "Open File"),
                    execution_mode=ActionExecutionMode.SYSTEM,
                    payload={"file_path": top.file_path},
                )
            )

            summarize_prompt = (
                f"{top_name} 핵심만 5줄로 요약해줘. 근거는 로컬 출처로만 써줘."
                if response_language == "ko"
                else f"Summarize only the key points of {top_name} in 5 lines using local citations only."
            )
            actions.append(
                SuggestedAction(
                    action_id="summarize_top",
                    kind=SuggestedActionKind.SUMMARIZE_TOP,
                    label=label("핵심 요약", "Summarize"),
                    execution_mode=ActionExecutionMode.PROMPT_INJECTION,
                    payload={"prompt": summarize_prompt, "file_path": top.file_path},
                )
            )

            if len(citations) >= 2:
                second = citations[1]
                compare_prompt = (
                    f"{Path(top.file_path).name}와 {Path(second.file_path).name}를 비교해 공통점/차이점을 표로 정리해줘."
                    if response_language == "ko"
                    else f"Compare {Path(top.file_path).name} and {Path(second.file_path).name} with similarities and differences in a table."
                )
                actions.append(
                    SuggestedAction(
                        action_id="compare_top_two",
                        kind=SuggestedActionKind.COMPARE_TOP,
                        label=label("비교 정리", "Compare"),
                        execution_mode=ActionExecutionMode.PROMPT_INJECTION,
                        payload={
                            "prompt": compare_prompt,
                            "file_path": top.file_path,
                            "secondary_file_path": second.file_path,
                        },
                    )
                )

        if insufficient:
            followup_prompt = (
                "질문 범위를 좁혀서 다시 물어볼게. 파일명/연도/태그를 포함해서 질의 예시 3개를 만들어줘."
                if response_language == "ko"
                else "Help me narrow the question. Create 3 follow-up query examples including file name, year, or tag."
            )
        elif intent == ChatIntent.FILE_SEARCH:
            followup_prompt = (
                "방금 찾은 파일들을 기준으로 핵심만 짧게 요약해줘."
                if response_language == "ko"
                else "Using the files just found, provide a short key summary."
            )
        elif intent == ChatIntent.TASK_REQUEST:
            followup_prompt = (
                f"지금 질문({query})을 바로 실행 가능한 단계로 나눠서 정리해줘."
                if response_language == "ko"
                else f"Break this request into executable steps: {query}"
            )
        else:
            followup_prompt = (
                "근거 중심으로 다음 확인 질문 3개를 추천해줘."
                if response_language == "ko"
                else "Suggest 3 evidence-oriented follow-up questions."
            )

        actions.append(
            SuggestedAction(
                action_id="ask_followup",
                kind=SuggestedActionKind.ASK_FOLLOWUP,
                label=label("다음 질문", "Follow-up"),
                execution_mode=ActionExecutionMode.PROMPT_INJECTION,
                payload={"prompt": followup_prompt},
            )
        )
        return actions

    def compose_v2(
        self,
        *,
        query: str,
        mode: WorkMode,
        response_language: str,
        parsed_intent: ParsedIntent,
        plan: LocalPlan,
        execution_result: ExecutionResult,
        verification: VerificationResult,
        behavior_policy: BehaviorPolicy | None,
        response_length: str = "medium",
        show_citations: bool = True,
        prefer_action_suggestions: bool = True,
        used_profile,
        engine_used,
        used_fallback: bool,
        runtime_detail: str | None,
        followup_resolution=None,
        allow_clarification: bool = True,
        conversation_path: str = "local_rag",
        escalated_provider: str | None = None,
        is_local: bool = True,
    ) -> ComposedChatResponseV2:
        citations = execution_result.citations
        confidence_band = self._confidence_band(verification.confidence)
        ungrounded_allowed = bool(execution_result.structured_payload.get("ungrounded_allowed", False))
        response_mode = self._select_response_mode(
            query=query,
            parsed_intent=parsed_intent,
            verification=verification,
            execution_result=execution_result,
            citations=citations,
            followup_resolution=followup_resolution,
            allow_clarification=allow_clarification,
        )
        lead = self._lead_v2(
            parsed_intent=parsed_intent,
            verification=verification,
            citations=citations,
            response_language=response_language,
            result_type=execution_result.result_type,
            response_mode=response_mode,
            followup_resolution=followup_resolution,
            ungrounded_allowed=ungrounded_allowed,
        )
        structured_result = self._structured_result_v2(
            query=query,
            execution_result=execution_result,
            verification=verification,
            response_language=response_language,
            response_length=response_length,
            parsed_intent=parsed_intent,
            allow_clarification=allow_clarification,
            response_mode=response_mode,
            ungrounded_allowed=ungrounded_allowed,
        )
        actions = self._actions_v2(
            query=query,
            parsed_intent=parsed_intent,
            response_language=response_language,
            citations=citations,
            plan=plan,
            behavior_policy=behavior_policy,
            candidate_mode=verification.candidate_mode,
            response_mode=response_mode,
        )
        if not prefer_action_suggestions:
            actions = []
        if len(actions) > 3:
            actions = actions[:3]
        visible_citations = citations if show_citations else []
        reasoning_hidden = not self._wants_reasoning_details(query)
        used_clarification = bool(
            allow_clarification
            and response_mode == "conversational_clarify"
            and "확인 질문" in structured_result.summary
        )
        detail_lower = str(runtime_detail or "").lower()
        direct_first_applied = bool(
            execution_result.structured_payload.get("direct_first_applied")
            or ("direct_first_applied=1" in detail_lower)
        )
        recommendation_shape = str(
            execution_result.structured_payload.get("recommendation_shape")
            or ResponseComposer._extract_detail_token(runtime_detail, key="recommendation_shape")
            or ""
        )
        question_count_after = ResponseComposer._extract_detail_int(
            runtime_detail,
            key="question_count_after_postprocess",
        )
        if question_count_after is None:
            question_count_after = ResponseComposer._question_sentence_count(structured_result.summary)
        return ComposedChatResponseV2(
            response_mode=response_mode,
            lead=lead,
            structured_result=structured_result,
            citations=visible_citations,
            actions=actions,
            metadata={
                "used_followup_resolution": bool(getattr(followup_resolution, "is_followup", False)),
                "used_clarification": used_clarification,
                "confidence_band": confidence_band,
                "conversation_path": conversation_path,
                "escalated_provider": escalated_provider,
                "reasoning_hidden": reasoning_hidden,
                "direct_first_applied": direct_first_applied,
                "question_count_after_postprocess": max(0, int(question_count_after)),
                "recommendation_shape": recommendation_shape,
            },
            parsed_intent=parsed_intent,
            plan=plan,
            verification=verification,
            mode=mode,
            used_profile=used_profile,
            is_local=is_local,
            engine_used=engine_used,
            used_fallback=used_fallback,
            runtime_detail=runtime_detail,
        )

    @staticmethod
    def _lead_v2(
        *,
        parsed_intent: ParsedIntent,
        verification: VerificationResult,
        citations: list[Citation],
        response_language: str,
        result_type: str,
        response_mode: str,
        followup_resolution=None,
        ungrounded_allowed: bool = False,
    ) -> str:
        if result_type == "runtime_error":
            if response_language == "ko":
                return "지금은 로컬 실행 환경 점검이 먼저 필요해요."
            return "The local runtime needs a quick check before I can continue."
        if result_type == "conversation" or ungrounded_allowed:
            return ""

        if response_mode == "conversational_clarify":
            return (
                "비슷한 후보가 둘 있어서, 먼저 하나 기준으로 보면 빠를 것 같아."
                if response_language == "ko"
                else "Two close candidates are competing, so a quick choice will speed this up."
            )
        if response_mode == "conversational_candidate" and not citations:
            return (
                "완전히 확실하진 않지만, 지금은 이쪽이 가장 가까워 보여."
                if response_language == "ko"
                else "This is still ambiguous, but this looks like the closest match for now."
            )
        if response_mode in {"task_confirm_execute", "conversational_soft_confirm"}:
            return ""

        if not citations:
            return ""
        return ""

    @staticmethod
    def _structured_result_v2(
        *,
        query: str,
        execution_result: ExecutionResult,
        verification: VerificationResult,
        response_language: str,
        response_length: str,
        parsed_intent: ParsedIntent,
        allow_clarification: bool,
        response_mode: str,
        ungrounded_allowed: bool,
    ) -> StructuredResult:
        summary = execution_result.generated_text.strip()
        if response_language == "ko":
            summary = ResponseComposer._normalize_korean_summary(
                summary,
                conversation_mode=(execution_result.result_type == "conversation"),
                query=query,
            )
        if (
            parsed_intent.intent == ReasoningIntent.GENERAL_CHAT
            and execution_result.result_type == "conversation"
        ):
            summary = ResponseComposer._enforce_direct_first_summary(
                summary=summary,
                response_language=response_language,
            )
            if ResponseComposer._is_recommendation_chat_query(query):
                summary = ResponseComposer._normalize_recommendation_three_options(
                    summary=summary,
                    response_language=response_language,
                )
        if execution_result.result_type == "file_list":
            summary = ResponseComposer._file_list_summary(
                query=query,
                payload=execution_result.structured_payload,
                response_language=response_language,
                candidate_mode=verification.candidate_mode,
                fallback=summary,
                requested_scope=str(getattr(parsed_intent, "scope", "") or "") or None,
            )
            if execution_result.structured_payload.get("auto_indexed"):
                if response_language == "ko":
                    summary = f"요청하신 대로 방금 변경분까지 자동 인덱싱해 반영했습니다.\n{summary}"
                else:
                    summary = f"I auto-indexed recent workspace changes before answering.\n{summary}"
        if verification.candidate_mode and not ungrounded_allowed:
            if not summary or ResponseComposer._is_noisy_generated_text(summary):
                summary = ResponseComposer._candidate_summary_from_citations(
                    citations=execution_result.citations,
                    confidence=verification.confidence,
                    response_language=response_language,
                )
        if (
            parsed_intent.intent == ReasoningIntent.SUMMARIZE_FILE
            and execution_result.result_type in {"answer", "summary"}
            and summary
        ):
            summary = ResponseComposer._format_summary_points(
                summary=summary,
                query=query,
                response_language=response_language,
            )
        summary = ResponseComposer._naturalize_summary_text(
            summary=summary,
            query=query,
            response_language=response_language,
            result_type=execution_result.result_type,
            intent=parsed_intent.intent,
        )
        if execution_result.result_type in {"answer", "summary", "comparison", "classification"} and execution_result.citations:
            summary = ResponseComposer._append_compact_source_line(
                summary=summary,
                citations=execution_result.citations,
                response_language=response_language,
            )

        if not ungrounded_allowed and response_mode != "conversational_direct":
            clarification = ResponseComposer._clarifying_questions(
                parsed_intent=parsed_intent,
                verification=verification,
                execution_result=execution_result,
                response_language=response_language,
            )
            if clarification and allow_clarification:
                if response_language == "ko":
                    summary = f"{summary}\n\n확인 질문:\n" + "\n".join(f"- {item}" for item in clarification[:2])
                else:
                    summary = f"{summary}\n\nClarifying questions:\n" + "\n".join(f"- {item}" for item in clarification[:2])
            elif clarification and not allow_clarification and response_language == "ko":
                summary = f"{summary}\n\n원하면 바로 다른 후보도 이어서 보여줄게."

        if response_length == "short" and len(summary) > 360:
            summary = summary[:360].rstrip() + "..."
        elif response_length == "medium" and len(summary) > 900:
            summary = summary[:900].rstrip() + "..."

        details: list[str] = []
        if verification.issues:
            if response_language == "ko":
                details.append(f"검증 이슈: {', '.join(verification.issues)}")
            else:
                details.append(f"Verification issues: {', '.join(verification.issues)}")

        return StructuredResult(
            result_type=execution_result.result_type,
            summary=summary or (insufficient_evidence_message(response_language)),
            details=details,
            data={**execution_result.structured_payload, "response_mode": response_mode},
        )

    @staticmethod
    def _append_compact_source_line(*, summary: str, citations: list[Citation], response_language: str) -> str:
        text = (summary or "").strip()
        if not text or not citations:
            return text
        names: list[str] = []
        seen: set[str] = set()
        for citation in citations:
            name = Path(citation.file_path).name
            if not name or name in seen:
                continue
            seen.add(name)
            names.append(name)
            if len(names) >= 3:
                break
        if not names:
            return text
        suffix = (
            f"참고 자료: {', '.join(names)}"
            if response_language == "ko"
            else f"References: {', '.join(names)}"
        )
        if suffix in text:
            return text
        return f"{text}\n\n{suffix}"

    @staticmethod
    def _clarifying_questions(
        *,
        parsed_intent: ParsedIntent,
        verification: VerificationResult,
        execution_result: ExecutionResult,
        response_language: str,
    ) -> list[str]:
        needs_clarification = (
            (verification.candidate_mode and not execution_result.citations)
            or verification.ambiguity_level >= 0.72
            or parsed_intent.confidence < 0.45
        )
        if not needs_clarification:
            return []

        has_citations = bool(execution_result.citations)
        intent = parsed_intent.intent
        if response_language == "ko":
            if intent == ReasoningIntent.FIND_FILE:
                questions: list[str] = []
                if not has_citations:
                    questions.append("정확한 범위를 위해 폴더명이나 파일명 일부를 알려주실 수 있나요?")
                if parsed_intent.time_filters.year is None and parsed_intent.time_filters.relative_days is None:
                    questions.append("특정 주차나 기간이 있나요? (예: 1~4주차, 최근 2주)")
                questions.append("찾은 뒤에 바로 무엇을 할까요? (파일 목록 / 핵심 요약 / 시험 포인트)")
                return questions
            if intent in {ReasoningIntent.SUMMARIZE_FILE, ReasoningIntent.EXPLAIN_CONTENT}:
                return [
                    "어떤 파일 기준으로 정리하면 좋을까요? 파일명 일부만 알려주셔도 됩니다.",
                    "시험 대비용 핵심만 원하시나요, 아니면 개념 설명까지 포함할까요?",
                ]
            if intent == ReasoningIntent.COMPARE_FILES:
                return [
                    "비교할 두 파일(또는 주제)을 지정해주실 수 있나요?",
                    "비교 기준은 어떤 쪽이 좋을까요? (공통점/차이점/시험 포인트)",
                ]
            return [
                "원하는 결과 형태를 한 번 더 좁혀주실 수 있나요?",
                "범위를 파일명/폴더명/기간 중 하나로 지정해주시면 정확도가 올라갑니다.",
            ]

        if intent == ReasoningIntent.FIND_FILE:
            questions = []
            if not has_citations:
                questions.append("Could you share a folder name or part of the file name to narrow scope?")
            questions.append("Do you want a file list, week-by-week view, or exam-focused summary?")
            return questions
        return [
            "Could you narrow the target file or time range?",
            "What output format do you prefer for this answer?",
        ]

    @staticmethod
    def _file_list_summary(
        *,
        query: str,
        payload: dict,
        response_language: str,
        candidate_mode: bool,
        fallback: str,
        requested_scope: str | None = None,
    ) -> str:
        items = payload.get("items")
        if not isinstance(items, list) or not items:
            weeks = payload.get("requested_weeks")
            if isinstance(weeks, list):
                week_values = [str(int(item)) for item in weeks if isinstance(item, int)]
                if week_values:
                    joined = ", ".join(f"{value}주차" for value in week_values)
                    if response_language == "ko":
                        return f"요청하신 {joined} 파일은 현재 인덱싱된 자료에서 찾지 못했습니다."
                    return f"I could not find files for {joined} in the currently indexed documents."
            return fallback

        names: list[str] = []
        for row in items:
            if not isinstance(row, dict):
                continue
            path = str(row.get("file_path") or "").strip()
            if not path:
                continue
            name = Path(path).name
            if name and name not in names:
                names.append(name)

        if not names:
            return fallback
        total = len(names)
        preview = names[:8]
        lowered_query = (query or "").lower()
        wants_full_list = (requested_scope == "all") or any(token in lowered_query for token in ("전부", "모두", "전체", "모든", "all", "every"))
        list_limit = 30 if wants_full_list else 48
        full_list = names[:list_limit]
        suffix = f" ... 외 {total - list_limit}개" if total > list_limit else ""

        if response_language == "ko":
            if wants_full_list:
                return (
                    f"요청하신 조건에 맞는 파일을 총 {total}개 찾았습니다.\n"
                    + "\n".join(f"{idx}. {name}" for idx, name in enumerate(full_list, start=1))
                    + (
                        f"\n... 외 {total - list_limit}개\n원하면 '계속 보여줘'라고 하면 이어서 보여드릴게요."
                        if total > list_limit
                        else ""
                    )
                )
            if candidate_mode:
                return (
                    f"요청하신 내용 기준으로 후보 파일 {total}개를 찾았습니다. "
                    f"우선 {names[0]}부터 확인해보시고, 필요하면 제가 바로 핵심만 정리해드리겠습니다."
                )
            if total <= 12:
                return (
                    f"요청하신 내용과 맞는 파일을 찾았습니다. 총 {total}개입니다.\n"
                    + "\n".join(f"{idx}. {name}" for idx, name in enumerate(names, start=1))
                )
            return (
                f"요청하신 내용과 맞는 파일을 찾았습니다. 가장 관련도가 높은 항목은 {preview[0]}이고, 총 {total}개 후보가 있습니다.\n"
                f"파일 목록: {', '.join(full_list)}{suffix}"
            )
        if wants_full_list:
            return (
                f"I found {total} matching files for your request.\n"
                + "\n".join(f"{idx}. {name}" for idx, name in enumerate(full_list, start=1))
                + (
                    f"\n... plus {total - list_limit} more\nSay 'show more' and I will continue."
                    if total > list_limit
                    else ""
                )
            )
        if candidate_mode:
            return (
                f"I found {total} candidate files for your request. "
                f"Start with {names[0]}, then I can summarize or compare the rest."
            )
        if total <= 12:
            return "I found matching files:\n" + "\n".join(f"{idx}. {name}" for idx, name in enumerate(names, start=1))
        return f"I found matching files. The top result is {preview[0]}, with {total} strong candidates.\nFiles: {', '.join(full_list)}{suffix}"

    @staticmethod
    def _is_noisy_generated_text(text: str) -> bool:
        compact = " ".join((text or "").split()).lower()
        if not compact:
            return True
        if "console.log" in compact or "#include" in compact:
            return True
        if compact.count("{") + compact.count("}") >= 6:
            return True
        if len(compact) > 420 and compact.count("http://") + compact.count("https://") >= 2:
            return True
        tokens = re.findall(r"[A-Za-z0-9가-힣_]+", compact)
        if len(tokens) >= 40:
            unique_ratio = len(set(tokens)) / max(1, len(tokens))
            if unique_ratio < 0.34:
                return True
        return False

    @staticmethod
    def _naturalize_summary_text(
        *,
        summary: str,
        query: str,
        response_language: str,
        result_type: str,
        intent: ReasoningIntent,
    ) -> str:
        text = (summary or "").strip()
        if not text:
            return ""
        if result_type == "file_list":
            return text

        text = ResponseComposer._strip_instruction_leakage(text)
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"[ \t]{2,}", " ", text).strip()
        text = ResponseComposer._collapse_repeated_word_chunks(text)
        text = ResponseComposer._dedupe_sentence_lines(
            text,
            aggressive=(result_type != "conversation"),
        )

        if intent == ReasoningIntent.SUMMARIZE_FILE:
            looks_like_points = bool(re.search(r"(?m)^\s*1\.\s+", text))
            if not looks_like_points:
                text = ResponseComposer._format_summary_points(
                    summary=text,
                    query=query,
                    response_language=response_language,
                )

        if ResponseComposer._is_noisy_generated_text(text):
            clauses = ResponseComposer._extract_summary_point_candidates(text, response_language=response_language)
            if not clauses:
                clauses = ResponseComposer._extract_clause_candidates(text, response_language=response_language)
            if clauses:
                text = "\n".join(f"{idx}. {item}" for idx, item in enumerate(clauses[:5], start=1))

        return text.strip()

    @staticmethod
    def _strip_instruction_leakage(text: str) -> str:
        cleaned = (text or "").strip()
        if not cleaned:
            return ""
        cleaned = cleaned.replace("\r\n", "\n").replace("\r", "\n")
        # Remove leaked role markers and policy-like lines.
        cleaned = re.sub(r"(?i)\buser\s*:\s*", "", cleaned)
        cleaned = re.sub(r"(?i)\bassistant\s*:\s*", "", cleaned)
        cleaned = re.sub(r"(?i)\bfollow-?up question\s*:\s*", "", cleaned)
        cleaned = re.sub(r"(?i)\bthink step by step\b[:：]?\s*", "", cleaned)
        cleaned = re.sub(r"(?i)\bokay,\s*let'?s see\.?", "", cleaned).strip()
        cleaned = re.sub(r"(?i)ask at most one follow-up question\.?", "", cleaned).strip()
        cleaned = re.sub(r"(?i)keep response to 1-3 sentences\.?", "", cleaned).strip()
        cleaned = re.sub(r"최대한\s*한\s*번만\s*물어보세요\.?", "", cleaned).strip()
        cleaned = re.sub(r"최대한\s*1[-~]\s*3문장으로만\s*답하세요\.?", "", cleaned).strip()
        cleaned = re.sub(r"(?im)^\s*(?:최종\s*답변|final\s*answer)\s*[:：]\s*", "", cleaned).strip()
        cleaned = re.sub(
            r"(?im)\b사용자에게\s*물어볼\s*때는\s*반드시\s*['\"“”]?\?['\"“”]?\s*를?\s*붙여주세요\.?\s*",
            "",
            cleaned,
        ).strip()
        cleaned = re.sub(
            r"(?im)\b(?:단,\s*)?사용자의?\s*질문에\s*대한\s*(?:답변|답이)\s*(?:부족|충분하지\s*않|명확하지\s*않)[^.!?\n]{0,120}\s*추가(?:적인)?\s*(?:질문|설명)[^.!?\n]{0,120}(?:가능(?:합니다|해요)|할\s*수\s*있습니다|주세요)?\.?\s*",
            "",
            cleaned,
        ).strip()
        cleaned = re.sub(
            r"(?im)^\s*사용자에게\s*(?:직접\s*)?(?:도움을?|답변을?|질문을?)\s*(?:주세요|주십시오|제공하세요)\.?\s*",
            "",
            cleaned,
        ).strip()
        cleaned = re.sub(
            r"(?im)^\s*사용자\s*메시지에\s*바로\s*반응하(?:세요|십시오)\.?\s*",
            "",
            cleaned,
        ).strip()
        cleaned = re.sub(
            r"(?im)^\s*사용자\s*메시지에\s*(?:명확한\s*)?답(?:변)?을?\s*하(?:세요|십시오)\.?\s*",
            "",
            cleaned,
        ).strip()
        cleaned = re.sub(
            r"(?im)\b사용자\s*메시지에\s*(?:명확한\s*)?답(?:변)?을?\s*하(?:세요|십시오)\.?\s*",
            "",
            cleaned,
        ).strip()
        cleaned = re.sub(
            r"(?im)^\s*(?:좋아|좋아요|알겠어|알겠어요|알겠습니다|응)\s*,?\s*(?:이|그)?\s*맥락[^.!?\n]{0,60}(?:볼게|해볼게|정리해볼게|이어서\s*볼게)\.?\s*",
            "",
            cleaned,
        ).strip()
        cleaned = re.sub(r"(?i)\bokay,\s*[.!,]*\s*$", "", cleaned).strip()
        cleaned = re.sub(r"[ \t]{2,}", " ", cleaned)
        cleaned = re.sub(r"\n{3,}", "\n\n", cleaned).strip()
        return cleaned

    @staticmethod
    def _collapse_repeated_word_chunks(text: str) -> str:
        words = (text or "").split()
        if len(words) < 12:
            return text
        out: list[str] = []
        i = 0
        min_chunk = 3
        max_chunk = 12
        min_repeats = 3
        while i < len(words):
            matched = False
            max_try = min(max_chunk, (len(words) - i) // min_repeats)
            for size in range(max_try, min_chunk - 1, -1):
                chunk = words[i : i + size]
                repeats = 1
                while i + (repeats + 1) * size <= len(words):
                    next_chunk = words[i + repeats * size : i + (repeats + 1) * size]
                    if next_chunk != chunk:
                        break
                    repeats += 1
                if repeats >= min_repeats:
                    out.extend(chunk)
                    i += size * repeats
                    matched = True
                    break
            if not matched:
                out.append(words[i])
                i += 1
        return " ".join(out).strip()

    @staticmethod
    def _dedupe_sentence_lines(text: str, *, aggressive: bool = True) -> str:
        if not text:
            return ""
        normalized = re.sub(r"\s+(?=\d+\.\s+)", "\n", text)
        lines = [line.strip() for line in normalized.splitlines() if line.strip()]
        is_numbered = all(re.match(r"^\d+\.\s+", line) for line in lines) if lines else False
        if is_numbered:
            deduped_points: list[str] = []
            for line in lines:
                value = re.sub(r"^\d+\.\s+", "", line).strip()
                if any(ResponseComposer._is_near_duplicate_point(value, prior) for prior in deduped_points):
                    continue
                deduped_points.append(value)
            return "\n".join(f"{idx}. {item}" for idx, item in enumerate(deduped_points, start=1)).strip()

        parts = [
            seg.strip()
            for seg in re.split(r"(?:\n+|(?<=[.!?。！？])\s+)", text)
            if seg.strip()
        ]
        deduped: list[str] = []
        seen_keys: set[str] = set()
        for seg in parts:
            normalized = re.sub(r"\s{2,}", " ", seg).strip()
            if len(normalized) < 8:
                continue
            key = re.sub(r"[^\w가-힣]+", "", normalized).casefold()
            if not key:
                continue
            if key in seen_keys:
                continue
            seen_keys.add(key)
            if aggressive and any(ResponseComposer._is_near_duplicate_point(normalized, prior) for prior in deduped):
                continue
            deduped.append(normalized)
        if not deduped:
            return text.strip()
        joined = " ".join(deduped).strip()
        return re.sub(r"\s{2,}", " ", joined).strip()

    @staticmethod
    def _candidate_summary_from_citations(
        *,
        citations: list[Citation],
        confidence: float,
        response_language: str,
    ) -> str:
        names: list[str] = []
        seen: set[str] = set()
        for citation in citations:
            name = Path(citation.file_path).name
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
            if len(names) >= 3:
                break
        if response_language == "ko":
            uncertainty = ResponseComposer.render_uncertainty(
                confidence=confidence,
                ambiguity_type="candidate",
                risk_level="low",
                response_language=response_language,
            )
            if names:
                return f"{uncertainty} 우선 {', '.join(names)} 순서로 확인하면 가장 빠르게 맞출 수 있어요."
            return "완전히 확실하진 않지만, 지금은 이쪽이 가장 가능성이 높아 보여요. 파일명/주차/태그를 붙이면 더 정확해져요."
        if names:
            return (
                "This is not fully certain yet, but these are the most likely candidates right now: "
                f"{', '.join(names)}."
            )
        return "This is still ambiguous. Please add one more clue such as file name, year, or tag."

    @staticmethod
    def _normalize_korean_summary(summary: str, *, conversation_mode: bool = False, query: str = "") -> str:
        text = (summary or "").strip()
        if not text:
            return ""
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        parts = [segment.strip() for segment in text.split("\n") if segment.strip()]
        if not parts:
            return ""
        compact = " ".join(parts)
        compact = " ".join(compact.split())
        if conversation_mode:
            if ResponseComposer._is_recommendation_chat_query(query):
                compact = compact.strip()
            else:
                trimmed = ResponseComposer._trim_conversation_summary(compact, query=query)
                if trimmed:
                    compact = trimmed
        # Avoid unfinished, clipped endings.
        if compact and not re.search(ResponseComposer._KOREAN_ENDING_REGEX, compact):
            if compact[-1] not in {".", "!", "?"}:
                compact += "."
        return compact

    @staticmethod
    def _format_summary_points(*, summary: str, query: str, response_language: str) -> str:
        text = (summary or "").strip()
        if not text:
            return ""
        desired = ResponseComposer._requested_point_count(query=query)
        candidates = ResponseComposer._extract_summary_point_candidates(text, response_language=response_language)
        if not candidates:
            return text
        if len(candidates) < desired:
            extra = ResponseComposer._extract_clause_candidates(text, response_language=response_language)
            for item in extra:
                if item in candidates:
                    continue
                candidates.append(item)
                if len(candidates) >= desired:
                    break
        points = candidates[:desired]
        if not points:
            return text
        lines = [f"{idx}. {item}" for idx, item in enumerate(points, start=1)]
        return "\n".join(lines).strip()

    @staticmethod
    def _requested_point_count(*, query: str) -> int:
        text = (query or "").lower()
        match = re.search(r"([3-7])\s*(?:줄|개|포인트|문장|lines?|points?)", text)
        if match:
            return max(3, min(7, int(match.group(1))))
        return 5

    @staticmethod
    def _extract_summary_point_candidates(text: str, *, response_language: str) -> list[str]:
        compact = text.replace("\r\n", "\n").replace("\r", "\n")
        compact = re.sub(r"\(\s*[^()\n]{1,60}\.(?:txt|md|markdown|pdf|docx|py|swift|json|ya?ml)\s*\)", " ", compact, flags=re.IGNORECASE)
        compact = re.sub(r"\b\d{1,2}[:：]\d{2}\b", "\n", compact)
        compact = re.sub(r"(?m)^\s*\d{1,2}[:：]\d{2}\s*", "", compact)
        compact = re.sub(r"(?m)^\s*\d+\s*[\.\)]\s*", "", compact)
        compact = re.sub(r"[•·■◆▶]", "\n", compact)
        compact = re.sub(r"\s{2,}", " ", compact).strip()
        if not compact:
            return []
        parts = [
            seg.strip(" \t-")
            for seg in re.split(r"(?:\n+|(?<=[.!?。！？])\s+|;\s+)", compact)
            if seg.strip(" \t-")
        ]
        output: list[str] = []
        seen: set[str] = set()
        for seg in parts:
            line = re.sub(r"^\d+\s*[\.\)]\s*", "", seg).strip()
            line = re.sub(r"(?i)^(?:좋아|좋아요|알겠어|알겠어요|알겠습니다|오케이|okay|alright)\s*,?\s*", "", line).strip()
            line = re.sub(r"\s{2,}", " ", line).strip(" .")
            if len(line) < 10:
                continue
            key = re.sub(r"[^\w가-힣]+", "", line).lower()
            if not key or key in seen:
                continue
            if any(ResponseComposer._is_near_duplicate_point(line, prior) for prior in output):
                continue
            seen.add(key)
            if response_language == "ko":
                line = ResponseComposer._trim_point_length(line, max_chars=90)
            else:
                line = ResponseComposer._trim_point_length(line, max_chars=120)
            output.append(line)
        return output

    @staticmethod
    def _extract_clause_candidates(text: str, *, response_language: str) -> list[str]:
        clauses = [
            seg.strip(" \t-")
            for seg in re.split(r"[,\n]", text or "")
            if seg.strip(" \t-")
        ]
        output: list[str] = []
        for clause in clauses:
            line = re.sub(r"\(\s*[^()\n]{1,60}\.(?:txt|md|markdown|pdf|docx|py|swift|json|ya?ml)\s*\)", "", clause, flags=re.IGNORECASE)
            line = re.sub(r"\b\d{1,2}[:：]\d{2}\b", " ", line)
            line = re.sub(r"\s{2,}", " ", line).strip(" .")
            if len(line) < 10:
                continue
            if response_language == "ko":
                line = ResponseComposer._trim_point_length(line, max_chars=90)
            else:
                line = ResponseComposer._trim_point_length(line, max_chars=120)
            if line and line not in output:
                output.append(line)
        return output

    @staticmethod
    def _is_near_duplicate_point(a: str, b: str) -> bool:
        normalized_a = re.sub(r"\b\d{1,2}[:：]\d{2}\b", " ", a.lower())
        normalized_b = re.sub(r"\b\d{1,2}[:：]\d{2}\b", " ", b.lower())
        normalized_a = re.sub(r"\b\d+\b", " ", normalized_a)
        normalized_b = re.sub(r"\b\d+\b", " ", normalized_b)
        tokens_a = set(re.findall(r"[A-Za-z0-9가-힣_]+", normalized_a))
        tokens_b = set(re.findall(r"[A-Za-z0-9가-힣_]+", normalized_b))
        if not tokens_a or not tokens_b:
            return False
        overlap = len(tokens_a.intersection(tokens_b))
        union = len(tokens_a.union(tokens_b))
        if union <= 0:
            return False
        return (overlap / union) >= 0.82

    @staticmethod
    def _trim_point_length(text: str, *, max_chars: int) -> str:
        value = (text or "").strip()
        if len(value) <= max_chars:
            return value
        head = value[:max_chars]
        cut = head.rsplit(" ", 1)[0].strip()
        if not cut:
            cut = head.strip()
        return cut + "..."

    @staticmethod
    def _trim_conversation_summary(summary: str, *, query: str) -> str:
        if not summary:
            return ""
        text = ResponseComposer._strip_instruction_leakage(summary)
        text = re.sub(
            r"(?im)^\s*(?:사용자의?\s*(?:말|질문|요청)|사용자\s*메시지)에\s*바로\s*반응하(?:세요|십시오)\.?\s*",
            "",
            text,
        ).strip()
        text = re.sub(
            r"(?im)^\s*사용자\s*메시지에\s*(?:명확한\s*)?답(?:변)?을?\s*하(?:세요|십시오)\.?\s*",
            "",
            text,
        ).strip()
        text = re.sub(r"\s{2,}", " ", text).strip()
        if not text:
            return ""

        sentence_candidates = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
        if not sentence_candidates:
            return text[:180].strip()

        deduped: list[str] = []
        seen: set[str] = set()
        for sentence in sentence_candidates:
            key = re.sub(r"[^\w가-힣]+", "", sentence).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            deduped.append(sentence)
            if len(deduped) >= 2:
                break

        output = " ".join(deduped).strip()
        # If this is a plain greeting query, avoid rambling fabricated context.
        lowered_query = (query or "").lower()
        if any(token in lowered_query for token in ("안녕", "hello", "hi", "hey")):
            if len(output) > 90:
                output = deduped[0] if deduped else output[:90]
        return output.strip()

    @staticmethod
    def _looks_question_sentence(sentence: str) -> bool:
        value = str(sentence or "").strip()
        if not value:
            return False
        if "?" in value:
            return True
        return re.search(r"(까요|나요|인가요|어때요|어떨까요|할까요|될까요)\s*[.!?]?$", value) is not None

    @staticmethod
    def _question_sentence_count(text: str) -> int:
        value = str(text or "").strip()
        if not value:
            return 0
        count = 0
        for line in value.splitlines():
            line = line.strip()
            if not line:
                continue
            for sentence in re.split(r"(?<=[.!?。！？])\s+", line):
                sentence = sentence.strip()
                if not sentence:
                    continue
                if ResponseComposer._looks_question_sentence(sentence):
                    count += 1
        return count

    @staticmethod
    def _enforce_direct_first_summary(*, summary: str, response_language: str) -> str:
        text = str(summary or "").strip()
        if not text:
            return ""
        sentences = [s.strip() for s in re.split(r"(?<=[.!?。！？])\s+", text) if s.strip()]
        if not sentences:
            return text
        first = sentences[0]
        if not ResponseComposer._looks_question_sentence(first):
            return text
        for idx, sentence in enumerate(sentences[1:], start=1):
            if ResponseComposer._looks_question_sentence(sentence):
                continue
            reordered = [sentence] + [s for i, s in enumerate(sentences) if i != idx]
            candidate = " ".join(reordered).strip()
            if response_language == "ko" and candidate and candidate[-1] not in {".", "!", "?"}:
                candidate += "."
            return candidate
        return text

    @staticmethod
    def _is_recommendation_chat_query(query: str) -> bool:
        lowered = str(query or "").strip().lower()
        if not lowered:
            return False
        task_cues = (
            "파일",
            "문서",
            "요약",
            "정리해",
            "찾아",
            "열어",
            "비교",
            "주차",
            "file",
            "document",
            "summary",
            "summarize",
            "find",
            "open",
            "compare",
        )
        if any(token in lowered for token in task_cues):
            return False
        recommendation_cues = (
            "추천",
            "메뉴",
            "뭐 먹",
            "어때",
            "골라",
            "선택",
            "뭐가 좋아",
            "뭐가 나아",
            "뭐할까",
            "어떤 게 좋아",
            "recommend",
            "suggest",
            "what should i",
            "which one",
            "choice",
        )
        return any(token in lowered for token in recommendation_cues)

    @staticmethod
    def _normalize_recommendation_three_options(*, summary: str, response_language: str) -> str:
        text = str(summary or "").strip()
        if not text:
            return ""
        numbered = re.findall(r"(?m)^\s*[1-3]\.\s+.+$", text)
        if len(numbered) >= 3:
            return "\n".join([line.strip() for line in numbered[:3]]).strip()

        items: list[str] = []
        seen: set[str] = set()
        inline_numbered = re.findall(
            r"(?:^|\s)([1-3])\.\s*([^0-9]+?)(?=(?:\s[1-3]\.\s)|$)",
            text,
        )
        for _, raw in inline_numbered:
            item = re.sub(r"\s+", " ", str(raw or "").strip()).strip(" .")
            if len(item) < 6:
                continue
            key = re.sub(r"[^\w가-힣]+", "", item).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            items.append(item)
            if len(items) >= 3:
                break
        candidates = re.findall(r"(?m)^\s*(?:\d+[.)]|[-•·])\s*(.+?)\s*$", text)
        if not candidates:
            candidates = [
                seg.strip()
                for seg in re.split(r"(?<=[.!?。！？])\s+|,\s+|\n+", text)
                if seg.strip()
            ]
        for raw in candidates:
            item = re.sub(r"\s+", " ", str(raw or "").strip()).strip(" .")
            if len(item) < 6:
                continue
            key = re.sub(r"[^\w가-힣]+", "", item).lower()
            if not key or key in seen:
                continue
            seen.add(key)
            items.append(item)
            if len(items) >= 3:
                break
        if len(items) < 3:
            return text
        lines = [f"{idx}. {item}" for idx, item in enumerate(items[:3], start=1)]
        if response_language == "ko":
            return "\n".join(lines).strip()
        return "\n".join(lines).strip()

    @staticmethod
    def _extract_detail_token(detail: str | None, *, key: str) -> str | None:
        lowered = str(detail or "").strip().lower()
        if not lowered:
            return None
        match = re.search(rf"{re.escape(key.lower())}=([a-z0-9_\-]+)", lowered)
        if not match:
            return None
        return match.group(1)

    @staticmethod
    def _extract_detail_int(detail: str | None, *, key: str) -> int | None:
        lowered = str(detail or "").strip().lower()
        if not lowered:
            return None
        match = re.search(rf"{re.escape(key.lower())}=([0-9]+)", lowered)
        if not match:
            return None
        try:
            return int(match.group(1))
        except Exception:
            return None

    @staticmethod
    def _confidence_band(confidence: float) -> str:
        if confidence >= 0.74:
            return "high"
        if confidence >= 0.48:
            return "medium"
        return "low"

    @staticmethod
    def _wants_reasoning_details(query: str) -> bool:
        lowered = (query or "").strip().lower()
        if not lowered:
            return False
        cues = (
            "왜",
            "근거",
            "판단 과정",
            "어떻게 판단",
            "reasoning",
            "why",
            "confidence",
            "explain your decision",
        )
        return any(token in lowered for token in cues)

    @staticmethod
    def render_uncertainty(
        *,
        confidence: float,
        ambiguity_type: str,
        risk_level: str,
        response_language: str,
    ) -> str:
        if response_language != "ko":
            if confidence >= 0.74:
                return "This is the strongest match."
            if confidence >= 0.48:
                return "Not perfect, but this is likely the best match right now."
            if risk_level == "high":
                return "I'm not fully sure yet, so I need one confirmation before running it."
            return "Still a bit ambiguous, but this is currently the most likely option."

        if confidence >= 0.74:
            return "이쪽이 가장 잘 맞아 보여."
        if confidence >= 0.48:
            return "완전히 확실하진 않지만, 지금은 이쪽이 제일 유력해 보여."
        if risk_level == "high":
            return "아직 애매해서 실행 전에 한 번만 확인할게."
        if ambiguity_type == "candidate":
            return "아직 애매한 부분은 있지만,"
        return "조금 애매하긴 하지만,"

    def _select_response_mode(
        self,
        *,
        query: str,
        parsed_intent: ParsedIntent,
        verification: VerificationResult,
        execution_result: ExecutionResult,
        citations: list[Citation],
        followup_resolution=None,
        allow_clarification: bool,
    ) -> str:
        lowered = (query or "").lower()
        is_action_like = any(token in lowered for token in ("열어", "요약", "비교", "정리", "open", "summarize", "compare"))
        has_followup = bool(getattr(followup_resolution, "is_followup", False))
        followup_type = str(getattr(followup_resolution, "followup_type", "") or "")
        if is_action_like and parsed_intent.intent in {
            ReasoningIntent.OPEN_FILE,
            ReasoningIntent.LIGHTWEIGHT_ACTION_REQUEST,
            ReasoningIntent.SELECT_PREVIOUS_CANDIDATE,
        }:
            return "task_confirm_execute"
        if execution_result.result_type in {"draft", "comparison"} and is_action_like:
            return "task_result_actions"
        if allow_clarification and verification.ambiguity_level >= 0.72 and self._candidate_gap_is_close(citations):
            return "conversational_clarify"
        if verification.candidate_mode or followup_type == "next_candidate":
            return "conversational_candidate"
        if has_followup:
            return "conversational_soft_confirm"
        return "conversational_direct"

    @staticmethod
    def _candidate_gap_is_close(citations: list[Citation]) -> bool:
        if len(citations) < 2:
            return False
        first = float(citations[0].score)
        second = float(citations[1].score)
        return abs(first - second) <= 0.05

    def _actions_v2(
        self,
        *,
        query: str,
        parsed_intent: ParsedIntent,
        response_language: str,
        citations: list[Citation],
        plan: LocalPlan,
        behavior_policy: BehaviorPolicy | None,
        candidate_mode: bool,
        response_mode: str,
    ) -> list[SuggestedAction]:
        def label(ko: str, en: str) -> str:
            return ko if response_language == "ko" else en

        actions: list[SuggestedAction] = []
        intent = parsed_intent.intent

        if citations and SuggestedActionKind.OPEN_FILE in plan.allowed_actions:
            actions.append(
                SuggestedAction(
                    action_id="open_top_file",
                    kind=SuggestedActionKind.OPEN_FILE,
                    label=label("파일 열기", "Open File"),
                    execution_mode=ActionExecutionMode.SYSTEM,
                    payload={"file_path": citations[0].file_path},
                )
            )

        if citations and SuggestedActionKind.SUMMARIZE_TOP in plan.allowed_actions:
            top_name = Path(citations[0].file_path).name
            prompt = (
                f"{top_name} 핵심을 5줄로 요약해줘."
                if response_language == "ko"
                else f"Summarize the key points of {top_name} in 5 lines."
            )
            actions.append(
                SuggestedAction(
                    action_id="summarize_top",
                    kind=SuggestedActionKind.SUMMARIZE_TOP,
                    label=label("핵심 요약", "Summarize"),
                    execution_mode=ActionExecutionMode.PROMPT_INJECTION,
                    payload={"prompt": prompt, "file_path": citations[0].file_path},
                )
            )

        if citations and SuggestedActionKind.MAKE_SHORTER in plan.allowed_actions:
            prompt = (
                "방금 답변을 3줄로 더 짧게 줄여줘."
                if response_language == "ko"
                else "Make the previous answer shorter in 3 lines."
            )
            actions.append(
                SuggestedAction(
                    action_id="make_shorter",
                    kind=SuggestedActionKind.MAKE_SHORTER,
                    label=label("더 짧게", "Make Shorter"),
                    execution_mode=ActionExecutionMode.PROMPT_INJECTION,
                    payload={"prompt": prompt},
                )
            )

        if len(citations) >= 2 and SuggestedActionKind.COMPARE_TOP in plan.allowed_actions:
            first = Path(citations[0].file_path).name
            second = Path(citations[1].file_path).name
            prompt = (
                f"{first}와 {second}를 비교해서 공통점과 차이점을 표로 정리해줘."
                if response_language == "ko"
                else f"Compare {first} and {second} in a similarities/differences table."
            )
            actions.append(
                SuggestedAction(
                    action_id="compare_top_two",
                    kind=SuggestedActionKind.COMPARE_TOP,
                    label=label("비교 정리", "Compare"),
                    execution_mode=ActionExecutionMode.PROMPT_INJECTION,
                    payload={
                        "prompt": prompt,
                        "file_path": citations[0].file_path,
                        "secondary_file_path": citations[1].file_path,
                    },
                )
            )

        if len(citations) >= 2 and SuggestedActionKind.OPEN_SECOND in plan.allowed_actions:
            actions.append(
                SuggestedAction(
                    action_id="open_second_file",
                    kind=SuggestedActionKind.OPEN_SECOND,
                    label=label("두 번째 파일 열기", "Open Second"),
                    execution_mode=ActionExecutionMode.SYSTEM,
                    payload={"file_path": citations[1].file_path},
                )
            )

        if citations and SuggestedActionKind.SHOW_OTHER_CANDIDATES in plan.allowed_actions:
            prompt = (
                "방금 결과에서 상위 후보 3개를 파일명+차이 한 줄로 보여줘."
                if response_language == "ko"
                else "Show top 3 alternative candidates with one-line differences."
            )
            actions.append(
                SuggestedAction(
                    action_id="show_other_candidates",
                    kind=SuggestedActionKind.SHOW_OTHER_CANDIDATES,
                    label=label("다른 후보 보기", "Show Other Candidates"),
                    execution_mode=ActionExecutionMode.PROMPT_INJECTION,
                    payload={"prompt": prompt},
                )
            )

        if len(citations) >= 2 and SuggestedActionKind.SHOW_DIFF in plan.allowed_actions:
            first = Path(citations[0].file_path).name
            second = Path(citations[1].file_path).name
            prompt = (
                f"{first}와 {second}의 핵심 차이를 bullet 7개로만 보여줘."
                if response_language == "ko"
                else f"Show only the key differences between {first} and {second} in 7 bullets."
            )
            actions.append(
                SuggestedAction(
                    action_id="show_diff",
                    kind=SuggestedActionKind.SHOW_DIFF,
                    label=label("차이 보기", "Show Diff"),
                    execution_mode=ActionExecutionMode.PROMPT_INJECTION,
                    payload={"prompt": prompt},
                )
            )

        if citations and SuggestedActionKind.SHOW_PREVIOUS_CANDIDATE in plan.allowed_actions:
            prompt = (
                "이전 후보를 다시 보여주고 현재 후보와 차이 한 줄만 설명해줘."
                if response_language == "ko"
                else "Show previous candidate and one-line difference from current candidate."
            )
            actions.append(
                SuggestedAction(
                    action_id="show_previous_candidate",
                    kind=SuggestedActionKind.SHOW_PREVIOUS_CANDIDATE,
                    label=label("이전 후보 보기", "Show Previous Candidate"),
                    execution_mode=ActionExecutionMode.PROMPT_INJECTION,
                    payload={"prompt": prompt},
                )
            )

        if SuggestedActionKind.CREATE_DRAFT in plan.allowed_actions:
            prompt = (
                "현재 근거를 바탕으로 실행 가능한 문서 초안을 작성해줘. (제목/핵심 요약/작업 항목)"
                if response_language == "ko"
                else "Create an actionable draft based on current evidence (title/summary/action items)."
            )
            actions.append(
                SuggestedAction(
                    action_id="create_draft",
                    kind=SuggestedActionKind.CREATE_DRAFT,
                    label=label("초안 만들기", "Create Draft"),
                    execution_mode=ActionExecutionMode.PROMPT_INJECTION,
                    payload={"prompt": prompt},
                )
            )

        followup_prompt = (
            "근거를 더 정확히 맞추도록 후속 질문 3개를 제안해줘."
            if response_language == "ko"
            else "Suggest 3 follow-up questions to improve grounding precision."
        )
        if plan.plan_type == "conversation":
            followup_prompt = (
                "사용자가 바로 시작할 수 있도록 자연스러운 다음 질문 3개를 제안해줘. (짧게)"
                if response_language == "ko"
                else "Suggest 3 natural next questions the user can ask to get started. Keep it short."
            )
        if candidate_mode:
            followup_prompt = (
                "지금 질문을 더 정확하게 만들 수 있는 재질문 예시 3개를 만들어줘. (파일명/연도/태그 포함)"
                if response_language == "ko"
                else "Create 3 sharper follow-up queries including file name, year, or tags."
            )
        if (
            SuggestedActionKind.ASK_FOLLOWUP in plan.allowed_actions
            and not (
                plan.plan_type == "conversation"
                and response_mode == "conversational_direct"
                and not candidate_mode
            )
        ):
            actions.append(
                SuggestedAction(
                    action_id="ask_followup",
                    kind=SuggestedActionKind.ASK_FOLLOWUP,
                    label=label("질문 좁히기" if candidate_mode else "다음 질문", "Clarify" if candidate_mode else "Follow-up"),
                    execution_mode=ActionExecutionMode.PROMPT_INJECTION,
                    payload={"prompt": followup_prompt, "source_query": query},
                )
            )

        # Intent-based action pruning: keep only the most useful 1~3 actions.
        if intent == ReasoningIntent.FIND_FILE:
            preferred = [
                SuggestedActionKind.OPEN_FILE,
                SuggestedActionKind.SUMMARIZE_TOP,
                SuggestedActionKind.SHOW_OTHER_CANDIDATES,
            ]
        elif intent == ReasoningIntent.COMPARE_FILES:
            preferred = [
                SuggestedActionKind.COMPARE_TOP,
                SuggestedActionKind.OPEN_FILE,
                SuggestedActionKind.ASK_FOLLOWUP,
            ]
        elif intent in {ReasoningIntent.SUMMARIZE_FILE, ReasoningIntent.EXPLAIN_CONTENT}:
            preferred = [
                SuggestedActionKind.OPEN_FILE,
                SuggestedActionKind.MAKE_SHORTER,
                SuggestedActionKind.ASK_FOLLOWUP,
            ]
        elif intent == ReasoningIntent.DRAFT_EDIT:
            preferred = [
                SuggestedActionKind.SHOW_DIFF,
                SuggestedActionKind.CREATE_DRAFT,
                SuggestedActionKind.ASK_FOLLOWUP,
            ]
        elif response_mode == "conversational_clarify":
            preferred = [
                SuggestedActionKind.OPEN_FILE,
                SuggestedActionKind.OPEN_SECOND,
                SuggestedActionKind.ASK_FOLLOWUP,
            ]
        else:
            preferred = [item.kind for item in actions]

        if behavior_policy and behavior_policy.preferred_action_order:
            priority = {kind: idx for idx, kind in enumerate(behavior_policy.preferred_action_order)}
            actions.sort(key=lambda item: priority.get(item.kind, 100))
        else:
            priority = {kind: idx for idx, kind in enumerate(preferred)}
            actions.sort(key=lambda item: priority.get(item.kind, 100))

        seen: set[str] = set()
        unique: list[SuggestedAction] = []
        for item in actions:
            if item.kind.value in seen:
                continue
            seen.add(item.kind.value)
            unique.append(item)
        return unique[:3]
