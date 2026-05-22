from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from typing import Any


def parse_json_dict(raw: Any) -> dict[str, Any]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    try:
        val = json.loads(raw)
        return val if isinstance(val, dict) else {}
    except Exception:
        return {}


def parse_json_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, list):
        return [str(i) for i in raw]
    try:
        val = json.loads(raw)
        return [str(i) for i in val] if isinstance(val, list) else []
    except Exception:
        return []


def normalize_metadata(metadata: dict[str, Any] | None) -> dict[str, Any]:
    p = metadata or {}
    tags = [str(t).strip() for t in p.get("tags", []) if str(t).strip()][:8] if isinstance(p.get("tags"), list) else []
    try:
        importance = max(0.0, min(1.0, float(p.get("importance", 0.5))))
    except Exception:
        importance = 0.5
    return {
        "summary": str(p.get("summary") or "")[:260],
        "category": str(p.get("category") or "참고자료"),
        "subcategory": str(p.get("subcategory") or "")[:40],
        "document_type": str(p.get("document_type") or "")[:40],
        "tags": tags,
        "year": p.get("year"),
        "project": str(p.get("project") or "")[:48] or None,
        "importance": importance,
        "excluded": bool(p.get("excluded", False)),
    }


def row_to_raw_dict(row: sqlite3.Row) -> dict[str, Any]:
    return {k: row[k] for k in row.keys()}


def row_to_effective_dict(row: sqlite3.Row) -> dict[str, Any]:
    c = row["user_category"] if row["user_category"] is not None else row["category"]
    sc = row["user_subcategory"] if row["user_subcategory"] is not None else row["subcategory"]
    dt = row["user_document_type"] if row["user_document_type"] is not None else row["document_type"]
    y = row["user_year"] if row["user_year"] is not None else row["year"]
    pj = row["user_project"] if row["user_project"] is not None else row["project"]
    imp = row["user_importance"] if row["user_importance"] is not None else row["importance"]
    ex = row["user_excluded"] if row["user_excluded"] is not None else row["excluded"]
    ts = parse_json_list(row["user_tags"] if row["user_tags"] is not None else row["tags"])
    return {
        "doc_id": row["doc_id"],
        "path": row["path"],
        "file_type": row["file_type"],
        "modified_at": datetime.fromtimestamp(float(row["modified_at"]), tz=timezone.utc),
        "indexed_at": datetime.fromisoformat(row["indexed_at"]),
        "summary": row["summary"] or "",
        "category": c or "참고자료",
        "subcategory": sc or "",
        "document_type": dt or "",
        "tags": ts,
        "year": y,
        "project": pj,
        "importance": float(imp or 0.5),
        "excluded": bool(ex or 0),
    }

