from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
from typing import Any

from ..language_utils import insufficient_evidence_message
from ..models import (
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

from .composer_summary_helpers import ComposerSummaryHelpers
from .composer_trace_helpers import ComposerTraceHelpers


class ResponseComposer(ComposerTraceHelpers, ComposerSummaryHelpers):
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
    _FENCED_CODE_PATTERN = re.compile(r"```[^\n`]*\n[\s\S]*?```")

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
        code = (response_language or "").strip().lower()
        if code == "ko":
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

        if code == "ja":
            if insufficient:
                return "現在の資料では回答の根拠が不足しています。"
            if not citations:
                return "関連する根拠を追加で見つけられませんでした。"
            top_name = Path(citations[0].file_path).name
            if intent == ChatIntent.FILE_SEARCH:
                return f"関連ファイルを{len(citations)}件見つけました。最も関連度が高いのは {top_name} です。"
            if intent == ChatIntent.TASK_REQUEST:
                return f"要求された作業に合う根拠を{len(citations)}件見つけました。すぐ処理を続けられます。"
            if intent == ChatIntent.AMBIGUOUS:
                return f"質問は複数の解釈が可能です。まず関連根拠を{len(citations)}件収集しました。"
            return f"質問に関連する根拠{len(citations)}件を基に整理しました。"

        if code != "en":
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
        code = (response_language or "").strip().lower()

        def label(ko: str, en: str, ja: str) -> str:
            if code == "ko":
                return ko
            if code == "en":
                return en
            if code == "ja":
                return ja
            return ko

        if citations:
            top = citations[0]
            top_name = Path(top.file_path).name
            actions.append(
                SuggestedAction(
                    action_id="open_top_file",
                    kind=SuggestedActionKind.OPEN_FILE,
                    label=label("파일 열기", "Open File", "ファイルを開く"),
                    execution_mode=ActionExecutionMode.SYSTEM,
                    payload={"file_path": top.file_path},
                )
            )

            if code == "ko":
                summarize_prompt = f"{top_name} 핵심만 5줄로 요약해줘. 근거는 로컬 출처로만 써줘."
            elif code == "ja":
                summarize_prompt = f"{top_name} の要点だけを5行で要約してください。根拠はローカル出典のみ使ってください。"
            elif code == "en":
                summarize_prompt = f"Summarize only the key points of {top_name} in 5 lines using local citations only."
            else:
                summarize_prompt = f"{top_name} 핵심만 5줄로 요약해줘. 근거는 로컬 출처로만 써줘."
            actions.append(
                SuggestedAction(
                    action_id="summarize_top",
                    kind=SuggestedActionKind.SUMMARIZE_TOP,
                    label=label("핵심 요약", "Summarize", "要点要約"),
                    execution_mode=ActionExecutionMode.PROMPT_INJECTION,
                    payload={"prompt": summarize_prompt, "file_path": top.file_path},
                )
            )

            if len(citations) >= 2:
                second = citations[1]
                if code == "ko":
                    compare_prompt = (
                        f"{Path(top.file_path).name}와 {Path(second.file_path).name}를 비교해 공통점/차이점을 표로 정리해줘."
                    )
                elif code == "ja":
                    compare_prompt = (
                        f"{Path(top.file_path).name} と {Path(second.file_path).name} を比較し、共通点と相違点を表で整理してください。"
                    )
                elif code == "en":
                    compare_prompt = (
                        f"Compare {Path(top.file_path).name} and {Path(second.file_path).name} with similarities and differences in a table."
                    )
                else:
                    compare_prompt = (
                        f"{Path(top.file_path).name}와 {Path(second.file_path).name}를 비교해 공통점/차이점을 표로 정리해줘."
                    )
                actions.append(
                    SuggestedAction(
                        action_id="compare_top_two",
                        kind=SuggestedActionKind.COMPARE_TOP,
                        label=label("비교 정리", "Compare", "比較整理"),
                        execution_mode=ActionExecutionMode.PROMPT_INJECTION,
                        payload={
                            "prompt": compare_prompt,
                            "file_path": top.file_path,
                            "secondary_file_path": second.file_path,
                        },
                    )
                )

        if insufficient:
            if code == "ko":
                followup_prompt = "질문 범위를 좁혀서 다시 물어볼게. 파일명/연도/태그를 포함해서 질의 예시 3개를 만들어줘."
            elif code == "ja":
                followup_prompt = "質問範囲を絞って再質問します。ファイル名/年/タグを含む質問例を3つ作ってください。"
            elif code == "en":
                followup_prompt = "Help me narrow the question. Create 3 follow-up query examples including file name, year, or tag."
            else:
                followup_prompt = "질문 범위를 좁혀서 다시 물어볼게. 파일명/연도/태그를 포함해서 질의 예시 3개를 만들어줘."
        elif intent == ChatIntent.FILE_SEARCH:
            if code == "ko":
                followup_prompt = "방금 찾은 파일들을 기준으로 핵심만 짧게 요약해줘."
            elif code == "ja":
                followup_prompt = "今見つけたファイルを基に、要点だけを短く要約してください。"
            elif code == "en":
                followup_prompt = "Using the files just found, provide a short key summary."
            else:
                followup_prompt = "방금 찾은 파일들을 기준으로 핵심만 짧게 요약해줘."
        elif intent == ChatIntent.TASK_REQUEST:
            if code == "ko":
                followup_prompt = f"지금 질문({query})을 바로 실행 가능한 단계로 나눠서 정리해줘."
            elif code == "ja":
                followup_prompt = f"現在の質問（{query}）を実行可能な手順に分けて整理してください。"
            elif code == "en":
                followup_prompt = f"Break this request into executable steps: {query}"
            else:
                followup_prompt = f"지금 질문({query})을 바로 실행 가능한 단계로 나눠서 정리해줘."
        else:
            if code == "ko":
                followup_prompt = "근거 중심으로 다음 확인 질문 3개를 추천해줘."
            elif code == "ja":
                followup_prompt = "根拠中心で次の確認質問を3つ提案してください。"
            elif code == "en":
                followup_prompt = "Suggest 3 evidence-oriented follow-up questions."
            else:
                followup_prompt = "근거 중심으로 다음 확인 질문 3개를 추천해줘."

        actions.append(
            SuggestedAction(
                action_id="ask_followup",
                kind=SuggestedActionKind.ASK_FOLLOWUP,
                label=label("다음 질문", "Follow-up", "次の質問"),
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
        prompt_cache_hit: bool = False,
    ) -> ComposedChatResponseV2:
        if plan is None:
            plan = LocalPlan(
                plan_type="conversation",
                selected_files=[],
                selected_chunks=[],
                response_strategy="conversational_assistant",
                allowed_actions=[SuggestedActionKind.ASK_FOLLOWUP],
                external_reasoning_needed=False,
            )
        if verification is None:
            verification = VerificationResult(
                is_valid=False,
                confidence=0.0,
                issues=["verification_missing"],
                ambiguity_level=1.0,
                candidate_mode=True,
            )
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
        if bool(execution_result.structured_payload.get("offer_regenerate")):
            regen_prompt = str(execution_result.structured_payload.get("regenerate_prompt") or query).strip() or str(query or "").strip()
            regen_label = (
                "응답 다시 생성"
                if response_language == "ko"
                else ("再生成" if response_language == "ja" else "Regenerate")
            )
            actions.insert(
                0,
                SuggestedAction(
                    action_id="retry_generation",
                    kind=SuggestedActionKind.ASK_FOLLOWUP,
                    label=regen_label,
                    execution_mode=ActionExecutionMode.PROMPT_INJECTION,
                    payload={"prompt": regen_prompt, "source_query": query, "retry_generation": "1"},
                ),
            )
        if not prefer_action_suggestions:
            actions = []
        if len(actions) > 3:
            actions = actions[:3]
        visible_citations = citations if show_citations else []
        is_complex = (conversation_path in {"system_agent_loop", "deep_research"})
        reasoning_hidden = not (is_complex or self._wants_reasoning_details(query))
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
        trace_events = ResponseComposer._trace_events_from_tool_logs(
            execution_result.tool_logs,
            response_language=response_language,
        )
        metadata: dict[str, Any] = {
            "used_followup_resolution": bool(getattr(followup_resolution, "is_followup", False)),
            "used_clarification": used_clarification,
            "confidence_band": confidence_band,
            "conversation_path": conversation_path,
            "escalated_provider": escalated_provider,
            "reasoning_hidden": reasoning_hidden,
            "direct_first_applied": direct_first_applied,
            "question_count_after_postprocess": max(0, int(question_count_after)),
            "recommendation_shape": recommendation_shape,
        }
        if trace_events:
            metadata["trace_events"] = trace_events
        return ComposedChatResponseV2(
            response_mode=response_mode,
            lead=lead,
            structured_result=structured_result,
            execution_result=execution_result,
            generated_text=execution_result.generated_text,
            citations=visible_citations,
            actions=actions,
            metadata=metadata,
            parsed_intent=parsed_intent,
            plan=plan,
            verification=verification,
            mode=mode,
            used_profile=used_profile,
            is_local=is_local,
            engine_used=engine_used,
            used_fallback=used_fallback,
            runtime_detail=runtime_detail,
            prompt_cache_hit=prompt_cache_hit,
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
        code = (response_language or "").strip().lower()
        if result_type == "runtime_error":
            if code == "ko":
                return "지금은 로컬 실행 환경 점검이 먼저 필요해요."
            if code == "ja":
                return "いまはローカル実行環境の点検が先に必要です。"
            if code != "en":
                return "지금은 로컬 실행 환경 점검이 먼저 필요해요."
            return "The local runtime needs a quick check before I can continue."
        if result_type == "conversation" or ungrounded_allowed:
            return ""

        if response_mode == "conversational_clarify":
            if code == "ko":
                return "비슷한 후보가 둘 있어서, 먼저 하나 기준으로 보면 빠를 것 같아."
            if code == "ja":
                return "近い候補が2つあるので、まず1つを選ぶと早く進められます。"
            if code != "en":
                return "비슷한 후보가 둘 있어서, 먼저 하나 기준으로 보면 빠를 것 같아."
            return "Two close candidates are competing, so a quick choice will speed this up."
        if response_mode == "conversational_candidate" and not citations:
            if code == "ko":
                return "완전히 확실하진 않지만, 지금은 이쪽이 가장 가까워 보여."
            if code == "ja":
                return "まだ曖昧さはありますが、現時点ではこれが最も近い候補です。"
            if code != "en":
                return "완전히 확실하진 않지만, 지금은 이쪽이 가장 가까워 보여."
            return "This is still ambiguous, but this looks like the closest match for now."
        if response_mode in {"task_confirm_execute", "conversational_soft_confirm"}:
            return ""

        if not citations:
            return ""
            
        # Phase 20: RA-RAG Reliability Warning
        if verification.reliability < 0.48:
            if code in {"ko", "ko-kr"}:
                return "⚠️ 일부 검색 결과의 신뢰도가 낮을 수 있습니다. "
            if code in {"ja", "ja-jp"}:
                return "⚠️ 検索結果の信頼性が低い可能性があります。 "
            return "⚠️ Some search results may have low reliability. "

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
        has_verified_external_retrieval = any(
            str(item or "").strip().startswith("retrieved:")
            for item in (execution_result.tool_logs or [])
        )
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
            allow_unverified_urls=has_verified_external_retrieval,
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
