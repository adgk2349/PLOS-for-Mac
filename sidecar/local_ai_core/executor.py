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
                    allow_static_fallback=False,
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

        if plan.plan_type == "summary" and plan.response_strategy == "focused_file_grounded_summary":
            focused_file = self._execute_focused_file_summary(
                mode=mode,
                selected=selected,
                startup_profile=startup_profile,
                engine=engine,
                mlx_model_path=mlx_model_path,
                llama_model_path=llama_model_path,
                language_preference=language_preference,
                response_length=response_length,
                language=language,
            )
            if focused_file is not None:
                return focused_file

        if plan.plan_type == "summary" and plan.response_strategy == "map_reduce_grounded_summary":
            multi_file = self._execute_multi_file_summary(
                query=query,
                mode=mode,
                selected=selected,
                startup_profile=startup_profile,
                engine=engine,
                mlx_model_path=mlx_model_path,
                llama_model_path=llama_model_path,
                language_preference=language_preference,
                response_length=response_length,
                language=language,
            )
            if multi_file is not None:
                return multi_file

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
        max_selected = 12
        if plan.plan_type == "file_lookup":
            max_selected = 180
        if plan.plan_type == "summary" and plan.response_strategy == "focused_file_grounded_summary":
            max_selected = 32
        elif plan.plan_type == "summary" and plan.response_strategy == "map_reduce_grounded_summary":
            max_selected = 24
        if not plan.selected_chunks and not plan.selected_files:
            return citations[: max(8, max_selected)]

        selected_chunk_ids = set(plan.selected_chunks)
        selected_doc_ids = set(plan.selected_files)
        output: list[Citation] = []
        for citation in citations:
            if citation.chunk_id in selected_chunk_ids or citation.doc_id in selected_doc_ids:
                output.append(citation)
        if not output:
            return []
        return output[:max_selected]

    def _execute_multi_file_summary(
        self,
        *,
        query: str,
        mode: WorkMode,
        selected: list[Citation],
        startup_profile: StartupProfileType,
        engine: LocalEngine,
        mlx_model_path: str | None,
        llama_model_path: str | None,
        language_preference: str | None,
        response_length: str,
        language: str,
    ) -> ExecutionResult | None:
        grouped: dict[str, list[Citation]] = {}
        for citation in selected:
            grouped.setdefault(citation.doc_id, []).append(citation)
        if len(grouped) < 2:
            return None

        docs = sorted(
            grouped.items(),
            key=lambda item: max((c.score for c in item[1]), default=0.0),
            reverse=True,
        )
        max_files = {
            "short": 5,
            "medium": 8,
            "long": 10,
        }.get((response_length or "medium").lower(), 8)
        docs = docs[:max_files]
        if len(docs) < 2:
            return None

        map_rows: list[dict] = []
        map_logs: list[str] = []
        map_used_fallback = False
        detail_parts: list[str] = []
        for doc_id, citations in docs:
            ranked = sorted(citations, key=lambda item: item.score, reverse=True)
            file_name = Path(ranked[0].file_path).name
            map_prompt = self._per_file_map_prompt(file_name=file_name, response_language=language)
            map_inference = self._local_inference.generate(
                query=map_prompt,
                mode=mode,
                citations=ranked[:4],
                profile=startup_profile.value,
                engine=engine,
                mlx_model_path=mlx_model_path,
                llama_model_path=llama_model_path,
                language_preference=language_preference,
                max_tokens=self._max_tokens_for(response_length="short", mode=mode),
            )
            map_summary = map_inference.answer.strip()
            if not map_summary:
                map_summary = ranked[0].snippet.strip()
                map_used_fallback = True
            map_rows.append(
                {
                    "doc_id": doc_id,
                    "file_path": ranked[0].file_path,
                    "file_name": file_name,
                    "summary": map_summary,
                    "score": float(ranked[0].score),
                    "modified_at": ranked[0].modified_at,
                    "category": ranked[0].category,
                    "subcategory": ranked[0].subcategory,
                    "tags": ranked[0].tags,
                    "document_type": ranked[0].document_type,
                    "importance": ranked[0].importance,
                }
            )
            map_logs.append(f"map_summary:{map_inference.engine_used.value}:{file_name}")
            if map_inference.used_fallback:
                map_used_fallback = True
            if map_inference.detail:
                detail_parts.append(f"map[{file_name}]={map_inference.detail}")

        reduce_citations: list[Citation] = []
        for idx, row in enumerate(map_rows, start=1):
            reduce_citations.append(
                Citation(
                    doc_id=str(row["doc_id"]),
                    chunk_id=f"{row['doc_id']}:map:{idx}",
                    file_path=str(row["file_path"]),
                    snippet=f"{row['file_name']}: {row['summary']}",
                    score=float(row["score"]),
                    modified_at=row["modified_at"],
                    category=str(row["category"]),
                    subcategory=str(row["subcategory"]),
                    tags=list(row["tags"])[:8],
                    document_type=str(row["document_type"]),
                    importance=float(row["importance"]),
                )
            )

        reduce_prompt = self._reduce_summary_prompt(
            file_count=len(map_rows),
            response_language=language,
        )
        reduce_inference = self._local_inference.generate(
            query=reduce_prompt,
            mode=mode,
            citations=reduce_citations,
            profile=startup_profile.value,
            engine=engine,
            mlx_model_path=mlx_model_path,
            llama_model_path=llama_model_path,
            language_preference=language_preference,
            max_tokens=self._max_tokens_for(response_length=response_length, mode=mode),
        )
        final_text = reduce_inference.answer.strip()
        if not final_text:
            final_text = self._fallback_reduce_summary(map_rows=map_rows, response_language=language)
            map_used_fallback = True
        detail = reduce_inference.detail
        if detail_parts:
            joined = " | ".join(detail_parts[:6])
            detail = f"{detail}; {joined}" if detail else joined

        top_citations: list[Citation] = []
        for _, citations in docs:
            ranked = sorted(citations, key=lambda item: item.score, reverse=True)
            if ranked:
                top_citations.append(ranked[0])

        return ExecutionResult(
            result_type="summary",
            structured_payload={
                "text": final_text,
                "aggregation": "map_reduce",
                "files_considered": len(map_rows),
            },
            citations=top_citations,
            tool_logs=[*map_logs, f"inference:{reduce_inference.engine_used.value}", "summary:map_reduce"],
            generated_text=final_text,
            engine_used=reduce_inference.engine_used,
            used_fallback=bool(reduce_inference.used_fallback or map_used_fallback),
            runtime_detail=detail,
        )

    def _execute_focused_file_summary(
        self,
        *,
        mode: WorkMode,
        selected: list[Citation],
        startup_profile: StartupProfileType,
        engine: LocalEngine,
        mlx_model_path: str | None,
        llama_model_path: str | None,
        language_preference: str | None,
        response_length: str,
        language: str,
    ) -> ExecutionResult | None:
        if not selected:
            return None
        grouped: dict[str, list[Citation]] = {}
        for citation in selected:
            grouped.setdefault(citation.doc_id, []).append(citation)
        if not grouped:
            return None
        best_doc_id, citations = max(grouped.items(), key=lambda item: max((c.score for c in item[1]), default=0.0))
        if not citations:
            return None
        ranked = sorted(citations, key=lambda item: item.score, reverse=True)
        file_name = Path(ranked[0].file_path).name
        summary_length = response_length if (response_length or "").lower() in {"medium", "long"} else "medium"
        prompt = self._focused_file_summary_prompt(file_name=file_name, response_language=language)
        inference = self._local_inference.generate(
            query=prompt,
            mode=mode,
            citations=ranked[:28],
            profile=startup_profile.value,
            engine=engine,
            mlx_model_path=mlx_model_path,
            llama_model_path=llama_model_path,
            language_preference=language_preference,
            max_tokens=self._max_tokens_for(response_length=summary_length, mode=mode),
        )
        text = inference.answer.strip()
        used_fallback = inference.used_fallback
        if not text:
            used_fallback = True
            text = self._fallback_focused_file_summary(citations=ranked, response_language=language)
        return ExecutionResult(
            result_type="summary",
            structured_payload={
                "text": text,
                "aggregation": "focused_file",
                "target_doc_id": best_doc_id,
                "target_file_path": ranked[0].file_path,
            },
            citations=ranked[:8],
            tool_logs=[f"inference:{inference.engine_used.value}", "summary:focused_file"],
            generated_text=text,
            engine_used=inference.engine_used,
            used_fallback=used_fallback,
            runtime_detail=inference.detail,
        )

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
            if len(items) >= 180:
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

    @staticmethod
    def _per_file_map_prompt(*, file_name: str, response_language: str) -> str:
        if response_language == "ko":
            return (
                f"{file_name} 내용만 기준으로 핵심을 2~3문장으로 요약해줘. "
                "중복 문장/복붙 없이 자연스러운 문장으로만 답해줘."
            )
        return (
            f"Summarize only {file_name} in 2-3 sentences. "
            "Use natural wording and avoid repetitive copy-paste phrasing."
        )

    @staticmethod
    def _reduce_summary_prompt(*, file_count: int, response_language: str) -> str:
        if response_language == "ko":
            return (
                f"아래는 파일 {file_count}개의 부분 요약입니다. "
                "중복을 제거해서 전체 흐름 중심으로 5~7줄 핵심 요약을 작성해줘. "
                "파일별 차이가 크면 마지막 줄에 차이점 1줄만 덧붙여줘."
            )
        return (
            f"These are partial summaries from {file_count} files. "
            "Merge them into a concise 5-7 line overall summary without repetition. "
            "If differences matter, add one final line for key differences."
        )

    @staticmethod
    def _fallback_reduce_summary(*, map_rows: list[dict], response_language: str) -> str:
        if not map_rows:
            return "요약할 파일을 찾지 못했습니다." if response_language == "ko" else "No files were available to summarize."
        lines: list[str] = []
        for row in map_rows[:6]:
            name = str(row.get("file_name") or "file")
            summary = str(row.get("summary") or "").strip()
            if not summary:
                continue
            lines.append(f"- {name}: {summary}")
        if not lines:
            return "요약을 생성하지 못했습니다." if response_language == "ko" else "I could not generate a summary."
        if response_language == "ko":
            return "파일별 핵심 요약:\n" + "\n".join(lines)
        return "Per-file key summary:\n" + "\n".join(lines)

    @staticmethod
    def _focused_file_summary_prompt(*, file_name: str, response_language: str) -> str:
        if response_language == "ko":
            return (
                f"{file_name} 전체 내용을 읽고 핵심을 자연스럽게 5~7줄로 요약해줘. "
                "원문 문장을 그대로 반복하지 말고 개념 중심으로 재서술해줘. "
                "답변에 파일명 괄호 표기나 로그 문구는 넣지 마."
            )
        return (
            f"Read the full content of {file_name} and produce a natural 5-7 line summary. "
            "Paraphrase conceptually instead of copying original sentences. "
            "Do not include file-name markers or log-like boilerplate."
        )

    @staticmethod
    def _fallback_focused_file_summary(*, citations: list[Citation], response_language: str) -> str:
        if not citations:
            return "요약할 근거를 찾지 못했습니다." if response_language == "ko" else "No evidence was available for summary."
        lines: list[str] = []
        for item in citations[:6]:
            snippet = str(item.snippet or "").strip()
            if not snippet:
                continue
            lines.append(f"- {snippet}")
        if not lines:
            return "요약을 생성하지 못했습니다." if response_language == "ko" else "I could not generate a summary."
        if response_language == "ko":
            return "핵심 내용:\n" + "\n".join(lines)
        return "Key points:\n" + "\n".join(lines)
