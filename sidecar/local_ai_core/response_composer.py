from __future__ import annotations

from pathlib import Path

from .language_utils import insufficient_evidence_message
from .models import (
    ActionExecutionMode,
    ChatIntent,
    Citation,
    SuggestedAction,
    SuggestedActionKind,
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
