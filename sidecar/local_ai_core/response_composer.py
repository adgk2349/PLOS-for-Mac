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

        if response_mode == "task_confirm_execute":
            return "응, 바로 해볼게." if response_language == "ko" else "Sure, I'll do that now."

        if response_mode == "conversational_soft_confirm":
            if response_language == "ko":
                followup_type = str(getattr(followup_resolution, "followup_type", "") or "")
                if followup_type == "refine_filter":
                    return "좋아, 그 조건으로 다시 좁혀볼게."
                if followup_type == "next_candidate":
                    return "좋아, 그럼 다음 후보로 볼게."
                return "응, 방금 맥락 기준으로 이어서 볼게."
            return "Got it. I'll continue from the previous context."

        if response_mode == "conversational_candidate":
            if response_language == "ko":
                if citations:
                    top_name = Path(citations[0].file_path).name
                    return f"찾았어. 지금 기준으론 {top_name}이 제일 유력해 보여."
                return "완전히 확실하진 않지만, 지금은 이쪽이 가장 가까워 보여."
            return "Found one likely candidate. This seems the closest match right now."

        if response_mode == "conversational_clarify":
            return (
                "비슷한 후보가 둘 있어서, 먼저 하나 기준으로 보면 빠를 것 같아."
                if response_language == "ko"
                else "Two close candidates are competing, so a quick choice will speed this up."
            )

        if not citations:
            if response_language == "ko":
                return "좋아요. 요청하신 방향으로 바로 도와드리겠습니다."
            return "Got it. I can help with that."

        top_name = Path(citations[0].file_path).name if citations else ""
        intent = parsed_intent.intent
        if response_language == "ko":
            if intent == ReasoningIntent.FIND_FILE:
                return f"지금 기준이면 {top_name}부터 보는 게 가장 맞아 보여."
            if intent == ReasoningIntent.COMPARE_FILES:
                return "좋아, 바로 비교해볼게."
            if intent == ReasoningIntent.DRAFT_EDIT:
                return "좋아, 바로 수정 가능한 초안으로 정리해봤어."
            if intent == ReasoningIntent.CLASSIFY:
                return "문서 성격 기준으로 분류해봤어."
            return "좋아, 이 맥락 기준으로 바로 정리해볼게."

        if intent == ReasoningIntent.FIND_FILE:
            return f"I found relevant files. The top candidate is {top_name}."
        if intent == ReasoningIntent.COMPARE_FILES:
            return "I gathered comparable evidence and summarized key differences."
        if intent == ReasoningIntent.DRAFT_EDIT:
            return "I produced an editable draft grounded in your local sources."
        if intent == ReasoningIntent.CLASSIFY:
            return "I classified the document and suggested metadata tags."
        return f"I answered using {len(citations)} grounded citations."

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
        if execution_result.result_type == "file_list":
            summary = ResponseComposer._file_list_summary(
                payload=execution_result.structured_payload,
                response_language=response_language,
                candidate_mode=verification.candidate_mode,
                fallback=summary,
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

        if not ungrounded_allowed:
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
        payload: dict,
        response_language: str,
        candidate_mode: bool,
        fallback: str,
    ) -> str:
        items = payload.get("items")
        if not isinstance(items, list) or not items:
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
            if len(names) >= 4:
                break

        if not names:
            return fallback

        if response_language == "ko":
            if candidate_mode:
                return (
                    f"요청하신 내용 기준으로 후보 파일 {len(names)}개를 찾았습니다. "
                    f"우선 {names[0]}부터 확인해보시고, 필요하면 제가 바로 핵심만 정리해드리겠습니다."
                )
            return (
                f"요청하신 내용과 맞는 파일을 찾았습니다. "
                f"가장 관련도가 높은 항목은 {names[0]}이고, 총 {len(names)}개 후보가 있습니다."
            )
        if candidate_mode:
            return (
                f"I found {len(names)} candidate files for your request. "
                f"Start with {names[0]}, then I can summarize or compare the rest."
            )
        return f"I found matching files. The top result is {names[0]}, with {len(names)} strong candidates."

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
        return False

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
            trimmed = ResponseComposer._trim_conversation_summary(compact, query=query)
            if trimmed:
                compact = trimmed
        # Avoid unfinished, clipped endings.
        if compact and not re.search(ResponseComposer._KOREAN_ENDING_REGEX, compact):
            if compact[-1] not in {".", "!", "?"}:
                compact += "."
        return compact

    @staticmethod
    def _trim_conversation_summary(summary: str, *, query: str) -> str:
        if not summary:
            return ""
        text = summary
        # Remove typical leaked thought/log markers.
        text = re.sub(r"(?i)\buser\s*:\s*", "", text)
        text = re.sub(r"(?i)\bassistant\s*:\s*", "", text)
        text = re.sub(r"(?i)\bfollow-?up question\s*:\s*", "", text)
        text = re.sub(r"(?i)\bokay,\s*let'?s see\.?", "", text).strip()
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
        if SuggestedActionKind.ASK_FOLLOWUP in plan.allowed_actions:
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
