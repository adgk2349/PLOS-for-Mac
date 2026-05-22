from __future__ import annotations

from typing import Any


class WorkspaceRagFinalizer:
    @staticmethod
    def apply_execution_metadata(
        *,
        execution,
        plan,
        auto_indexed: bool,
        auto_index_triggered: bool,
        metadata_fallback_used: bool,
        week_exact_filter_applied: bool,
        requested_weeks: list[int],
        week_exact_no_match: bool,
        explicit_file_focus_used: bool,
        focused_file_summary_used: bool,
        summary_scope_expanded: bool,
        req_mode,
        work_mode_development,
        review_batch_enabled: bool,
        review_batch_count: int,
        review_batch_partial_failures: int,
    ) -> None:
        execution.tool_logs.append(f"router:plan={plan.plan_type}")
        execution.tool_logs.append("agent:reasoner_primary")
        if auto_indexed:
            execution.tool_logs.append("auto_index:incremental")
            execution.structured_payload["auto_indexed"] = True
        elif auto_index_triggered:
            execution.tool_logs.append("auto_index:started")
            execution.structured_payload["auto_indexing_started"] = True
        if metadata_fallback_used:
            execution.tool_logs.append("fallback:metadata_file_search")
            execution.structured_payload["metadata_fallback_used"] = True
        if week_exact_filter_applied:
            execution.tool_logs.append("focus:week_exact")
            execution.structured_payload["week_exact_filter_used"] = True
            execution.structured_payload["requested_weeks"] = requested_weeks
            if week_exact_no_match:
                execution.structured_payload["week_exact_no_match"] = True
        if explicit_file_focus_used:
            execution.tool_logs.append("focus:explicit_file_name")
            execution.structured_payload["explicit_file_focus_used"] = True
        if focused_file_summary_used:
            execution.tool_logs.append("summary_scope:focused_file")
            execution.structured_payload["focused_file_summary_used"] = True
        if summary_scope_expanded:
            execution.tool_logs.append("summary_scope:multi_file")
            execution.structured_payload["summary_scope_expanded"] = True
        if req_mode == work_mode_development and review_batch_enabled:
            execution.structured_payload["review_batch_enabled"] = True
            execution.structured_payload["review_batch_count"] = review_batch_count
            execution.structured_payload["review_batch_partial_failures"] = review_batch_partial_failures

    @staticmethod
    def apply_development_composed_metadata(
        *,
        composed,
        execution,
        review_batch_enabled: bool,
        review_batch_count: int,
        review_batch_partial_failures: int,
        patch_plan_count: int,
        patch_applied_count: int,
        patch_failed_count: int,
        patch_verification_status: str,
        extract_grounded_line_refs,
    ) -> None:
        grounded_paths = []
        seen_paths: set[str] = set()
        for citation in execution.citations:
            path = str(citation.file_path or "").strip()
            if not path or path in seen_paths:
                continue
            seen_paths.add(path)
            grounded_paths.append(path)
        grounded_line_refs = extract_grounded_line_refs(execution.citations)
        composed.metadata["review_mode"] = True
        composed.metadata["grounded_file_count"] = len(grounded_paths)
        composed.metadata["grounded_line_refs"] = grounded_line_refs
        composed.metadata["review_batch_enabled"] = review_batch_enabled
        composed.metadata["review_batch_count"] = review_batch_count
        composed.metadata["review_batch_partial_failures"] = review_batch_partial_failures
        composed.metadata["patch_plan_count"] = patch_plan_count
        composed.metadata["patch_applied_count"] = patch_applied_count
        composed.metadata["patch_failed_count"] = patch_failed_count
        composed.metadata["verification_status"] = patch_verification_status

