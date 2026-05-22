from __future__ import annotations

from typing import Any


def parse_content_length(raw: str | None) -> int | None:
    if not raw:
        return None
    try:
        parsed = int(raw)
    except Exception:
        return None
    return parsed if parsed > 0 else None


def read_progress_percent(progress: dict[str, Any] | None) -> float | None:
    if not progress:
        return None
    value = progress.get("progress_percent")
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def read_progress_int(progress: dict[str, Any] | None, key: str) -> int | None:
    if not progress:
        return None
    value = progress.get(key)
    if value is None:
        return None
    try:
        return int(value)
    except Exception:
        return None


def progress_ratio(downloaded_bytes: int, total_bytes: int | None) -> float | None:
    if not total_bytes or total_bytes <= 0:
        return None
    ratio = max(0.0, min(1.0, float(downloaded_bytes) / float(total_bytes)))
    return ratio * 100.0

