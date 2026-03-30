from __future__ import annotations

from .composer import ResponseComposer


def trace_events_from_logs(tool_logs: list[str], *, response_language: str = "auto") -> list[dict[str, str]]:
    """Convert internal tool logs to safe trace events for UI rendering."""
    return ResponseComposer._trace_events_from_tool_logs(tool_logs or [], response_language=response_language)

