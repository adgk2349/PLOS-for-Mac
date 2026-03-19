from __future__ import annotations

from pathlib import Path

from .language_utils import resolve_response_language
from .models import (
    Citation,
    ExecutionResult,
    LocalEngine,
    LocalPlan,
    ReasoningIntent,
    WorkMode,
)
from .models import StartupProfile as StartupProfileType


class LocalExecutor:
    def __init__(self, local_inference):
        self._local_inference = local_inference

    def execute_conversation(
        self,
        *,
        query: str,
        mode: WorkMode,
        startup_profile: StartupProfileType,
        engine: LocalEngine,
        mlx_model_path: str | None,
        llama_model_path: str | None,
        language_preference: str | None,
        session_summary: str | None,
        max_tokens: int = 320,
    ) -> ExecutionResult:
        conversational = self._local_inference.generate_conversational(
            query=query,
            mode=mode,
            profile=startup_profile.value,
            engine=engine,
            mlx_model_path=mlx_model_path,
            llama_model_path=llama_model_path,
            language_preference=language_preference,
            max_tokens=max_tokens,
            session_summary=session_summary,
            allow_static_fallback=False,
        )
        return ExecutionResult(
            result_type="conversation",
            structured_payload={
                "style": "general_chat",
                "ungrounded_allowed": True,
            },
            citations=[],
            tool_logs=[f"conversational_inference:{conversational.engine_used.value}"],
            generated_text=conversational.answer.strip(),
            engine_used=conversational.engine_used,
            used_fallback=conversational.used_fallback,
            runtime_detail=conversational.detail,
        )

    def execute(
        self,
        *,
        query: str,
        mode: WorkMode,
        parsed_intent: ReasoningIntent,
        plan: LocalPlan,
        citations: list[Citation],
        startup_profile: StartupProfileType,
        engine: LocalEngine,
        mlx_model_path: str | None,
        llama_model_path: str | None,
        language_preference: str | None,
        response_length: str = "medium",
    ) -> ExecutionResult:
        selected = self._selected_citations(plan=plan, citations=citations)
        language = resolve_response_language(query, language_preference)
        if not selected:
            if parsed_intent in {
                ReasoningIntent.FOLLOWUP_QUESTION,
                ReasoningIntent.FOLLOWUP_REFINE,
                ReasoningIntent.CONTINUE_PREVIOUS_RESULT,
                ReasoningIntent.SOFT_CONFIRM,
                ReasoningIntent.SELECT_PREVIOUS_CANDIDATE,
                ReasoningIntent.NEXT_CANDIDATE,
                ReasoningIntent.REDUCE_SCOPE,
                ReasoningIntent.LIGHTWEIGHT_ACTION_REQUEST,
                ReasoningIntent.OPEN_FILE,
            }:
                conversational = self._local_inference.generate_conversational(
                    query=query,
                    mode=mode,
                    profile=startup_profile.value,
                    engine=engine,
                    mlx_model_path=mlx_model_path,
                    llama_model_path=llama_model_path,
                    language_preference=language_preference,
                    max_tokens=self._max_tokens_for(response_length="short", mode=mode),
                )
                return ExecutionResult(
                    result_type="answer",
                    structured_payload={
                        "text": conversational.answer,
                        "ungrounded_allowed": True,
                    },
                    citations=[],
                    tool_logs=[f"conversational_inference:{conversational.engine_used.value}", "no_citation_conversation"],
                    generated_text=conversational.answer,
                    engine_used=conversational.engine_used,
                    used_fallback=conversational.used_fallback,
                    runtime_detail=conversational.detail,
                )
            return ExecutionResult(
                result_type="candidate",
                structured_payload={"items": []},
                citations=[],
                tool_logs=["no_selected_citations"],
                generated_text=(
                    "근거가 부족해 후보 형태로 안내합니다."
                    if language == "ko"
                    else "Grounded evidence was insufficient, returning candidate-style output."
                ),
                engine_used=None,
                used_fallback=False,
                runtime_detail=None,
            )

        if plan.plan_type == "file_lookup":
            files = self._file_items(selected)
            text = self._file_lookup_text(files, language)
            return ExecutionResult(
                result_type="file_list",
                structured_payload={"items": files},
                citations=selected,
                tool_logs=["file_discovery"],
                generated_text=text,
                engine_used=None,
                used_fallback=False,
                runtime_detail=None,
            )

        if plan.plan_type == "classification":
            top = selected[0]
            payload = {
                "file_path": top.file_path,
                "category": top.category,
                "tags": top.tags,
                "confidence_hint": round(top.score, 3),
            }
            text = self._classification_text(payload, language)
            return ExecutionResult(
                result_type="classification",
                structured_payload=payload,
                citations=selected,
                tool_logs=["classification_review"],
                generated_text=text,
                engine_used=None,
                used_fallback=False,
                runtime_detail=None,
            )

        if plan.plan_type == "lightweight_action":
            if parsed_intent == ReasoningIntent.OPEN_FILE:
                files = self._file_items(selected[:1])
                text = self._file_lookup_text(files, language)
                return ExecutionResult(
                    result_type="file_list",
                    structured_payload={"items": files},
                    citations=selected[:1],
                    tool_logs=["lightweight_action:open_file_candidate"],
                    generated_text=text,
                    engine_used=None,
                    used_fallback=False,
                    runtime_detail=None,
                )
            prompt = (
                "핵심만 아주 짧게 요약해줘. (3~5줄)"
                if language == "ko"
                else "Summarize only the essentials in 3-5 lines."
            )
            inference = self._local_inference.generate(
                query=prompt,
                mode=mode,
                citations=selected,
                profile=startup_profile.value,
                engine=engine,
                mlx_model_path=mlx_model_path,
                llama_model_path=llama_model_path,
                language_preference=language_preference,
                max_tokens=self._max_tokens_for(response_length="short", mode=mode),
            )
            return ExecutionResult(
                result_type="summary",
                structured_payload={"text": inference.answer},
                citations=selected[:6],
                tool_logs=[f"inference:{inference.engine_used.value}", "lightweight_action:summary"],
                generated_text=inference.answer,
                engine_used=inference.engine_used,
                used_fallback=inference.used_fallback,
                runtime_detail=inference.detail,
            )

        prompt = query
        if plan.plan_type == "comparison" and len(selected) >= 2:
            a = Path(selected[0].file_path).name
            b = Path(selected[1].file_path).name
            prompt = (
                f"{a}와 {b}를 비교해 공통점/차이점/의사결정 근거를 표 형태로 정리해줘."
                if language == "ko"
                else f"Compare {a} and {b} with similarities, differences, and decision rationale in table form."
            )
        elif plan.plan_type == "draft":
            prompt = (
                "아래 근거를 바탕으로 바로 수정 가능한 초안 문서를 작성해줘. 제목, 핵심 요약, 실행 항목 3개를 포함해줘."
                if language == "ko"
                else "Create an editable draft from the evidence with title, summary, and three action items."
            )
        elif plan.plan_type == "summary":
            prompt = (
                "근거 기반으로 핵심만 5~7줄로 요약해줘."
                if language == "ko"
                else "Summarize only the key points in 5-7 lines with grounded evidence."
            )

        inference = self._local_inference.generate(
            query=prompt,
            mode=mode,
            citations=selected,
            profile=startup_profile.value,
            engine=engine,
            mlx_model_path=mlx_model_path,
            llama_model_path=llama_model_path,
            language_preference=language_preference,
            max_tokens=self._max_tokens_for(response_length=response_length, mode=mode),
        )
        result_type = "draft" if plan.plan_type == "draft" else ("comparison" if plan.plan_type == "comparison" else "answer")
        payload = {"text": inference.answer}
        if result_type == "comparison" and len(selected) >= 2:
            payload["compared_files"] = [selected[0].file_path, selected[1].file_path]
        if result_type == "draft":
            payload["editable"] = True

        return ExecutionResult(
            result_type=result_type,
            structured_payload=payload,
            citations=selected,
            tool_logs=[f"inference:{inference.engine_used.value}"],
            generated_text=inference.answer,
            engine_used=inference.engine_used,
            used_fallback=inference.used_fallback,
            runtime_detail=inference.detail,
        )

    @staticmethod
    def _selected_citations(*, plan: LocalPlan, citations: list[Citation]) -> list[Citation]:
        if not plan.selected_chunks and not plan.selected_files:
            return citations[:8]

        selected_chunk_ids = set(plan.selected_chunks)
        selected_doc_ids = set(plan.selected_files)
        output: list[Citation] = []
        for citation in citations:
            if citation.chunk_id in selected_chunk_ids or citation.doc_id in selected_doc_ids:
                output.append(citation)
        if not output:
            return []
        return output[:12]

    @staticmethod
    def _max_tokens_for(*, response_length: str, mode: WorkMode) -> int:
        base = {
            "short": 160,
            "medium": 280,
            "long": 420,
        }.get((response_length or "medium").lower(), 280)
        if mode in {WorkMode.RESEARCH, WorkMode.DEVELOPMENT, WorkMode.PLANNING}:
            base += 40
        if mode == WorkMode.STRICT_SEARCH:
            base = min(base, 220)
        return max(96, min(base, 560))

    @staticmethod
    def _file_items(citations: list[Citation]) -> list[dict]:
        seen: set[str] = set()
        items: list[dict] = []
        for citation in citations:
            if citation.doc_id in seen:
                continue
            seen.add(citation.doc_id)
            items.append(
                {
                    "doc_id": citation.doc_id,
                    "file_path": citation.file_path,
                    "score": round(citation.score, 3),
                    "category": citation.category,
                }
            )
            if len(items) >= 8:
                break
        return items

    @staticmethod
    def _file_lookup_text(items: list[dict], language: str) -> str:
        if not items:
            return "근거가 되는 파일을 찾지 못했습니다." if language == "ko" else "No grounded files were found."
        top = Path(items[0]["file_path"]).name
        if language == "ko":
            return f"관련 파일 {len(items)}개를 찾았습니다. 가장 먼저 확인할 파일은 {top}입니다."
        return f"I found {len(items)} related files. The top match is {top}."

    @staticmethod
    def _classification_text(payload: dict, language: str) -> str:
        category = payload.get("category") or "참고자료"
        tags = payload.get("tags") or []
        tag_text = ", ".join(tags[:5]) if tags else "-"
        if language == "ko":
            return f"문서 성격은 '{category}'로 분류하는 것이 타당합니다. 추천 태그: {tag_text}"
        return f"The document is best classified as '{category}'. Suggested tags: {tag_text}."
