from __future__ import annotations

from .models import LocalPlan, ParsedIntent, ReasoningIntent, SuggestedActionKind, WorkMode


class LocalPlanner:
    def build_plan(
        self,
        *,
        parsed_intent: ParsedIntent,
        mode: WorkMode,
        file_doc_ids: list[str],
        chunk_ids: list[str],
        top_score: float,
        prefer_multi_file_summary: bool = False,
        prefer_focused_file_summary: bool = False,
    ) -> LocalPlan:
        intent = parsed_intent.intent

        if intent == ReasoningIntent.GENERAL_CHAT:
            return LocalPlan(
                plan_type="conversation",
                selected_files=[],
                selected_chunks=[],
                response_strategy="conversational_assistant",
                allowed_actions=[SuggestedActionKind.ASK_FOLLOWUP],
                external_reasoning_needed=False,
            )

        if intent == ReasoningIntent.FIND_FILE:
            return LocalPlan(
                plan_type="file_lookup",
                selected_files=file_doc_ids[:140],
                selected_chunks=chunk_ids[:140],
                response_strategy="list_then_offer_actions",
                allowed_actions=[
                    SuggestedActionKind.OPEN_FILE,
                    SuggestedActionKind.SUMMARIZE_TOP,
                    SuggestedActionKind.SHOW_OTHER_CANDIDATES,
                ],
                external_reasoning_needed=False,
            )

        if intent in {ReasoningIntent.FOLLOWUP_REFINE, ReasoningIntent.REDUCE_SCOPE, ReasoningIntent.NEXT_CANDIDATE}:
            return LocalPlan(
                plan_type="file_lookup",
                selected_files=file_doc_ids[:140],
                selected_chunks=chunk_ids[:140],
                response_strategy="candidate_first_then_actions",
                allowed_actions=[
                    SuggestedActionKind.OPEN_FILE,
                    SuggestedActionKind.SUMMARIZE_TOP,
                    SuggestedActionKind.SHOW_OTHER_CANDIDATES,
                    SuggestedActionKind.SHOW_PREVIOUS_CANDIDATE,
                ],
                external_reasoning_needed=False,
            )

        if intent in {ReasoningIntent.SUMMARIZE_FILE}:
            if prefer_focused_file_summary:
                return LocalPlan(
                    plan_type="summary",
                    selected_files=file_doc_ids[:2],
                    selected_chunks=chunk_ids[:30],
                    response_strategy="focused_file_grounded_summary",
                    allowed_actions=[
                        SuggestedActionKind.OPEN_FILE,
                        SuggestedActionKind.MAKE_SHORTER,
                        SuggestedActionKind.ASK_FOLLOWUP,
                    ],
                    external_reasoning_needed=False,
                )
            if prefer_multi_file_summary:
                return LocalPlan(
                    plan_type="summary",
                    selected_files=file_doc_ids[:14],
                    selected_chunks=chunk_ids[:36],
                    response_strategy="map_reduce_grounded_summary",
                    allowed_actions=[
                        SuggestedActionKind.OPEN_FILE,
                        SuggestedActionKind.MAKE_SHORTER,
                        SuggestedActionKind.ASK_FOLLOWUP,
                    ],
                    external_reasoning_needed=False,
                )
            return LocalPlan(
                plan_type="summary",
                selected_files=file_doc_ids[:3],
                selected_chunks=chunk_ids[:8],
                response_strategy="concise_grounded_summary",
                allowed_actions=[
                    SuggestedActionKind.OPEN_FILE,
                    SuggestedActionKind.MAKE_SHORTER,
                    SuggestedActionKind.ASK_FOLLOWUP,
                ],
                external_reasoning_needed=False,
            )

        if intent == ReasoningIntent.COMPARE_FILES:
            return LocalPlan(
                plan_type="comparison",
                selected_files=file_doc_ids[:2],
                selected_chunks=chunk_ids[:10],
                response_strategy="contrast_with_grounding",
                allowed_actions=[
                    SuggestedActionKind.COMPARE_TOP,
                    SuggestedActionKind.OPEN_FILE,
                    SuggestedActionKind.ASK_FOLLOWUP,
                ],
                external_reasoning_needed=(mode == WorkMode.RESEARCH and top_score < 0.35),
            )

        if intent == ReasoningIntent.DRAFT_EDIT:
            return LocalPlan(
                plan_type="draft",
                selected_files=file_doc_ids[:2],
                selected_chunks=chunk_ids[:10],
                response_strategy="generate_editable_draft",
                allowed_actions=[
                    SuggestedActionKind.SHOW_DIFF,
                    SuggestedActionKind.CREATE_DRAFT,
                    SuggestedActionKind.ASK_FOLLOWUP,
                ],
                external_reasoning_needed=(mode in {WorkMode.RESEARCH, WorkMode.PLANNING} and top_score < 0.3),
            )

        if intent == ReasoningIntent.CLASSIFY:
            return LocalPlan(
                plan_type="classification",
                selected_files=file_doc_ids[:4],
                selected_chunks=chunk_ids[:8],
                response_strategy="category_and_tag_recommendation",
                allowed_actions=[
                    SuggestedActionKind.OPEN_FILE,
                    SuggestedActionKind.ASK_FOLLOWUP,
                ],
                external_reasoning_needed=False,
            )

        if intent in {
            ReasoningIntent.FOLLOWUP_QUESTION,
            ReasoningIntent.CONTINUE_PREVIOUS_RESULT,
            ReasoningIntent.SOFT_CONFIRM,
            ReasoningIntent.SELECT_PREVIOUS_CANDIDATE,
            ReasoningIntent.NEXT_CANDIDATE,
        }:
            return LocalPlan(
                plan_type="followup",
                selected_files=file_doc_ids[:3],
                selected_chunks=chunk_ids[:8],
                response_strategy="clarify_and_continue",
                allowed_actions=[
                    SuggestedActionKind.OPEN_FILE,
                    SuggestedActionKind.SUMMARIZE_TOP,
                    SuggestedActionKind.SHOW_OTHER_CANDIDATES,
                ],
                external_reasoning_needed=False,
            )

        if intent in {ReasoningIntent.LIGHTWEIGHT_ACTION_REQUEST, ReasoningIntent.OPEN_FILE}:
            return LocalPlan(
                plan_type="lightweight_action",
                selected_files=file_doc_ids[:2],
                selected_chunks=chunk_ids[:6],
                response_strategy="execute_quick_action",
                allowed_actions=[
                    SuggestedActionKind.OPEN_FILE,
                    SuggestedActionKind.SUMMARIZE_TOP,
                    SuggestedActionKind.ASK_FOLLOWUP,
                ],
                external_reasoning_needed=False,
            )

        return LocalPlan(
            plan_type="explanation",
            selected_files=file_doc_ids[:4],
            selected_chunks=chunk_ids[:9],
            response_strategy="direct_grounded_explanation",
            allowed_actions=[
                SuggestedActionKind.OPEN_FILE,
                SuggestedActionKind.SUMMARIZE_TOP,
                SuggestedActionKind.ASK_FOLLOWUP,
            ],
            external_reasoning_needed=(mode == WorkMode.RESEARCH and top_score < 0.32),
        )
