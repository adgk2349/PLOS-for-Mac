from __future__ import annotations

from ..models import Citation


def compact_citations(citations: list[Citation], *, limit: int = 3) -> list[Citation]:
    """Return top-N citations for concise composition surfaces."""
    if limit <= 0:
        return []
    return list(citations or [])[:limit]

