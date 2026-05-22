from __future__ import annotations

from typing import Any, Callable

from ..workspace_rag_components import WorkspaceRagComponents


class WorkspaceRagReranker:
    @staticmethod
    def severity_rank(value: str) -> int:
        token = str(value or "").strip().upper()
        if token == "P0":
            return 0
        if token == "P1":
            return 1
        if token == "P2":
            return 2
        return 3

    def reduce_and_rank_issues(
        self,
        items: list[dict[str, Any]],
        *,
        severity_rank: Callable[[str], int] | None = None,
    ) -> list[dict[str, Any]]:
        rank_fn = severity_rank or self.severity_rank
        return WorkspaceRagComponents.reduce_and_rank_issues(items, severity_rank=rank_fn)

