from __future__ import annotations

from typing import Any


def apply_doc_filters(rows: list[dict[str, Any]], *, search: str | None, filters: Any | None) -> list[dict[str, Any]]:
    result = rows
    if search:
        n = search.strip().lower()
        if n:
            result = [r for r in result if n in r["path"].lower() or n in r["summary"].lower() or n in " ".join(r["tags"]).lower()]
    if not filters:
        return result
    if filters.category:
        result = [r for r in result if r["category"] == filters.category]
    if filters.year is not None:
        result = [r for r in result if r.get("year") == filters.year]
    if filters.project:
        n = filters.project.lower()
        result = [r for r in result if (r.get("project") or "").lower().find(n) >= 0]
    if filters.tags:
        w = {t.lower() for t in filters.tags if t.strip()}
        if w:
            result = [r for r in result if w.intersection({t.lower() for t in r.get("tags", [])})]
    if filters.excluded is not None:
        result = [r for r in result if bool(r.get("excluded")) == bool(filters.excluded)]
    else:
        result = [r for r in result if not bool(r.get("excluded"))]
    return result

