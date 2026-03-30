from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import re
from typing import Any

from ..models import (
    ActionExecutionMode,
    BehaviorPolicy,
    Citation,
    ExecutionResult,
    LocalPlan,
    ParsedIntent,
    ReasoningIntent,
    SuggestedAction,
    SuggestedActionKind,
    VerificationResult,
)


class ComposerTraceHelpers:
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
            "왜", "근거", "판단 과정", "어떻게 판단", "reasoning", "why", "confidence",
            "explain", "상세히", "자세히", "상세하게", "자세하게", "분석", "과정", "설명해줘",
            "어떤 기능", "성능", "수준", "비교", "차이", "어떻게", "심층", "전문적",
        )
        return any(token in lowered for token in cues)

    @staticmethod
    def _trace_events_from_tool_logs(
        tool_logs: list[str],
        *,
        response_language: str,
    ) -> list[dict[str, Any]]:
        if not tool_logs:
            return []
        events: list[dict[str, Any]] = []
        base = datetime.now(timezone.utc)
        for idx, raw in enumerate(tool_logs[:24]):
            item = ComposerTraceHelpers._trace_event_from_log(
                raw,
                at=base + timedelta(milliseconds=idx * 140),
                response_language=response_language,
            )
            if item is None:
                continue
            events.append(item)
        return events

    @staticmethod
    def _trace_event_from_log(
        raw: str,
        *,
        at: datetime,
        response_language: str,
    ) -> dict[str, Any] | None:
        log = str(raw or "").strip()
        if not log:
            return None

        def msg(ko: str, en: str, ja: str) -> str:
            code = (response_language or "").strip().lower()
            if code == "ko":
                return ko
            if code == "en":
                return en
            if code == "ja":
                return ja
            return ko

        def envelope(*, status: str, message: str, source: str, url: str | None = None) -> dict[str, Any] | None:
            clean_message = " ".join(str(message or "").split()).strip()
            if not clean_message or not ComposerTraceHelpers._is_safe_trace_message(clean_message):
                return None
            payload: dict[str, Any] = {
                "status": status,
                "message": clean_message,
                "source": source,
                "at": at.isoformat(),
            }
            clean_url = str(url or "").strip()
            if clean_url:
                payload["url"] = clean_url
            return payload

        if log.startswith("retrieving:"):
            url = log.split(":", 1)[1].strip()
            if response_language == "ja":
                retrieving_message = f"取得中 {url}" if url else "外部エンドポイント取得中"
            else:
                retrieving_message = f"retrieving {url}" if url else "retrieving external endpoint"
            return envelope(
                status="retrieving",
                message=retrieving_message,
                source="external",
                url=url or None,
            )
        if log.startswith("retrieved:"):
            url = log.split(":", 1)[1].strip()
            if response_language == "ja":
                retrieved_message = f"取得完了 {url}" if url else "外部エンドポイント取得完了"
            else:
                retrieved_message = f"retrieved {url}" if url else "retrieved external endpoint"
            return envelope(
                status="retrieved",
                message=retrieved_message,
                source="external",
                url=url or None,
            )
        if log.startswith("web_loop:round="):
            value = log.split("=", 1)[1].strip()
            round_no = value.split("|", 1)[0].strip() if value else "1"
            return envelope(
                status="planning",
                message=msg(
                    f"웹 검색 라운드 {round_no} 계획 수립",
                    f"planning web loop round {round_no}",
                    f"Web検索ラウンド{round_no}を計画中",
                ),
                source="pipeline",
            )
        if log == "web_loop:retrieving":
            return envelope(
                status="retrieving",
                message=msg(
                    "웹 근거를 수집 중입니다",
                    "retrieving web evidence",
                    "Web根拠を取得中です",
                ),
                source="external",
            )
        if log.startswith("web_loop:quality="):
            return envelope(
                status="quality_eval",
                message=msg(
                    "수집 근거 품질을 평가 중입니다",
                    "evaluating evidence quality",
                    "収集した根拠の品質を評価中です",
                ),
                source="pipeline",
            )
        if log == "web_loop:refine_triggered":
            return envelope(
                status="refine_query",
                message=msg(
                    "질의를 정제해 추가 검색을 수행합니다",
                    "refining query for additional search",
                    "クエリを精緻化して追加検索します",
                ),
                source="pipeline",
            )
        if log == "web_loop:finalizing":
            return envelope(
                status="finalizing",
                message=msg(
                    "최종 답변을 정리 중입니다",
                    "finalizing answer",
                    "最終回答を整理中です",
                ),
                source="pipeline",
            )
        if log == "web_loop:converged":
            return envelope(
                status="done",
                message=msg(
                    "웹 근거 모호성이 낮아져 결론을 확정했습니다",
                    "web evidence converged and conclusion is ready",
                    "Web根拠の曖昧さが下がり結論を確定しました",
                ),
                source="pipeline",
            )
        if log == "web_loop:budget_exhausted":
            return envelope(
                status="warning",
                message=msg(
                    "검색 예산에 도달해 현재까지 근거로 답변을 생성합니다",
                    "search budget exhausted; answering with best current evidence",
                    "検索予算に達したため、現時点の最良根拠で回答します",
                ),
                source="pipeline",
            )
        if log.startswith("planning:"):
            value = log.split(":", 1)[1].strip().replace("_", " ")
            return envelope(
                status="planning",
                message=value or msg("계획 수립 중", "planning", "計画中"),
                source="pipeline",
            )
        if log.startswith("done:"):
            value = log.split(":", 1)[1].strip().replace("_", " ")
            if value.startswith("web evidence composed:"):
                count_raw = value.split(":", 1)[1].strip() if ":" in value else "0"
                try:
                    count = max(0, int(count_raw))
                except Exception:
                    count = 0
                return envelope(
                    status="done",
                    message=msg(
                        f"웹 근거 {count}개 정리 완료",
                        f"web evidence composed: {count}",
                        f"Web根拠を{count}件整理しました",
                    ),
                    source="external",
                )
            return envelope(
                status="done",
                message=value or msg("완료", "done", "完了"),
                source="pipeline",
            )
        if log == "web_search:requested":
            return envelope(
                status="retrieving",
                message=msg(
                    "웹 검색 결과 수집 중",
                    "retrieving web search results",
                    "Web検索結果を取得中",
                ),
                source="external",
            )
        if log == "web_search:direct":
            return envelope(
                status="done",
                message=msg(
                    "웹 검색 완료",
                    "web search completed",
                    "Web検索完了",
                ),
                source="external",
            )
        if log.startswith("notice:searxng_json_forbidden:"):
            return envelope(
                status="planning",
                message=msg(
                    "searXNG JSON 경로가 차단되어 HTML 경로로 전환했습니다",
                    "searXNG JSON endpoint blocked; switched to HTML search",
                    "searXNG JSON経路がブロックされたためHTML検索に切り替えました",
                ),
                source="external",
            )
        if log.startswith("notice:searxng_html_non_result_page"):
            return envelope(
                status="warning",
                message=msg(
                    "searXNG HTML 응답이 검색 결과 페이지가 아니어서 건너뛰었습니다",
                    "searXNG HTML response was not a search result page",
                    "searXNG HTML応答が検索結果ページではなかったためスキップしました",
                ),
                source="external",
            )
        if log.startswith("web_discovery:direct_url"):
            return envelope(
                status="planning",
                message=msg(
                    "직접 URL 입력을 감지했습니다",
                    "detected direct URL input",
                    "直接URL入力を検知しました",
                ),
                source="external",
            )
        if log.startswith("web_discovery:count="):
            count_raw = log.split("=", 1)[1].strip() if "=" in log else "0"
            try:
                count = max(0, int(count_raw))
            except Exception:
                count = 0
            return envelope(
                status="done",
                message=msg(
                    f"웹 검색 후보 {count}개 발견",
                    f"discovered {count} web candidates",
                    f"Web候補を{count}件発見",
                ),
                source="external",
            )
        if log.startswith("warning:search_failed:"):
            url = log.split(":", 2)[2].strip() if log.count(":") >= 2 else ""
            return envelope(
                status="warning",
                message=(
                    msg(
                        f"검색 실패 {url}",
                        f"search failed {url}",
                        f"検索失敗 {url}",
                    )
                    if url
                    else msg("검색 실패", "search failed", "検索失敗")
                ),
                source="external",
                url=url or None,
            )
        if log.startswith("warning:fetch_failed:"):
            url = log.split(":", 2)[2].strip() if log.count(":") >= 2 else ""
            return envelope(
                status="warning",
                message=(
                    msg(
                        f"수집 실패 {url}",
                        f"fetch failed {url}",
                        f"取得失敗 {url}",
                    )
                    if url
                    else msg("수집 실패", "fetch failed", "取得失敗")
                ),
                source="external",
                url=url or None,
            )
        if log.startswith("warning:"):
            value = log.split(":", 1)[1].strip()
            if ":" in value:
                reason, url = value.split(":", 1)
                reason_text = reason.replace("_", " ").strip()
                clean_url = url.strip()
                return envelope(
                    status="warning",
                    message=f"{reason_text} {clean_url}".strip(),
                    source="external",
                    url=clean_url or None,
                )
            return envelope(
                status="warning",
                message=value.replace("_", " ").strip() or msg("경고", "warning", "警告"),
                source="external",
            )
        if log.startswith("notice:"):
            value = log.split(":", 1)[1].strip().replace("_", " ")
            return envelope(
                status="planning",
                message=value or msg("알림", "notice", "通知"),
                source="external",
            )
        if log.startswith("web_search:blocked:"):
            reason = log.split(":", 2)[2].strip() if log.count(":") >= 2 else "unknown"
            return envelope(
                status="warning",
                message=msg(
                    f"웹 검색 차단: {reason}",
                    f"web search blocked: {reason}",
                    f"Web検索がブロックされました: {reason}",
                ),
                source="external",
            )
        if log.startswith("web_search:auto_triggered"):
            return envelope(
                status="planning",
                message=msg(
                    "불확실성으로 인해 자동 웹 검색을 시작했습니다",
                    "auto web search triggered by uncertainty",
                    "不確実性のため自動Web検索を開始しました",
                ),
                source="external",
            )
        if log.startswith("web_search:auto_unavailable"):
            return envelope(
                status="warning",
                message=msg(
                    "자동 웹 검색을 사용할 수 없습니다",
                    "auto web search unavailable",
                    "自動Web検索を利用できません",
                ),
                source="external",
            )
        if log.startswith("web_search:auto_suppressed:"):
            reason = log.split(":", 2)[2].strip() if log.count(":") >= 2 else "unknown"
            return envelope(
                status="warning",
                message=msg(
                    f"자동 웹 검색이 정책으로 차단되었습니다: {reason}",
                    f"auto web search suppressed by policy: {reason}",
                    f"自動Web検索がポリシーで抑制されました: {reason}",
                ),
                source="external",
            )
        if log.startswith("external_escalation:"):
            reason = log.split(":", 1)[1].strip().replace("_", " ")
            return envelope(
                status="warning",
                message=msg(
                    f"외부 에스컬레이션 사유: {reason}",
                    f"external escalation reason: {reason}",
                    f"外部エスカレーション理由: {reason}",
                ),
                source="external",
            )

        if log.startswith("router:intent="):
            value = log.split("=", 1)[1].strip().replace("_", " ")
            return envelope(
                status="planning",
                message=msg(
                    f"의도 판별: {value}",
                    f"intent resolved: {value}",
                    f"意図判定: {value}",
                ),
                source="pipeline",
            )
        if log.startswith("router:plan="):
            value = log.split("=", 1)[1].strip().replace("_", " ")
            return envelope(
                status="planning",
                message=msg(
                    f"실행 계획: {value}",
                    f"execution plan: {value}",
                    f"実行計画: {value}",
                ),
                source="pipeline",
            )
        if log.startswith("agent:"):
            value = log.split(":", 1)[1].strip().replace("_", " ")
            return envelope(
                status="planning",
                message=msg(
                    f"에이전트 단계: {value}",
                    f"agent step: {value}",
                    f"エージェント段階: {value}",
                ),
                source="pipeline",
            )
        if log.startswith("focus:"):
            value = log.split(":", 1)[1].strip().replace("_", " ")
            return envelope(
                status="planning",
                message=msg(
                    f"포커스 적용: {value}",
                    f"focus applied: {value}",
                    f"フォーカス適用: {value}",
                ),
                source="retrieval",
            )
        if log.startswith("summary_scope:"):
            value = log.split(":", 1)[1].strip().replace("_", " ")
            return envelope(
                status="planning",
                message=msg(
                    f"요약 범위: {value}",
                    f"summary scope: {value}",
                    f"要約範囲: {value}",
                ),
                source="retrieval",
            )
        if log.startswith("assistive_retrieval:"):
            value = log.split(":", 1)[1].strip().replace("_", " ")
            return envelope(
                status="done",
                message=msg(
                    f"보조 검색: {value}",
                    f"assistive retrieval: {value}",
                    f"補助検索: {value}",
                ),
                source="retrieval",
            )
        if log.startswith("external_escalated"):
            return envelope(
                status="done",
                message=msg(
                    "외부 경로로 에스컬레이션됨",
                    "external escalation completed",
                    "外部経路へエスカレーション完了",
                ),
                source="external",
            )
        if log.startswith("runtime_error:") or log.startswith("fallback:"):
            value = log.replace("_", " ")
            return envelope(
                status="warning",
                message=value,
                source="pipeline",
            )

        normalized = log.replace("_", " ")
        return envelope(
            status="done",
            message=normalized,
            source="pipeline",
        )

    @staticmethod
    def _is_safe_trace_message(message: str) -> bool:
        lowered = str(message or "").strip().lower()
        if not lowered:
            return False
        blocked_tokens = (
            "user:",
            "assistant:",
            "system prompt",
            "instruction",
            "정책문구",
            "사용자 메시지에",
            "바로 반응하세요",
            "final answer",
            "thought:",
            "chain of thought",
            "<thought>",
            "</thought>",
            "<observation>",
            "</observation>",
        )
        return not any(token in lowered for token in blocked_tokens)

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
