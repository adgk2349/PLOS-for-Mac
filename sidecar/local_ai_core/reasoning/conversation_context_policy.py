from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class ConversationContextPolicy:
    history_turn_pairs: int
    rolling_summary_chars: int
    inject_digest_into_legacy_prompt: bool


def resolve_conversation_context_policy(
    *,
    startup_profile: Any,
    model_path: str | None,
    digest: dict[str, Any] | None,
) -> ConversationContextPolicy:
    """Choose one context representation for the active model budget.

    Chat-template engines receive digest history as messages. Repeating that
    digest inside the legacy prompt wastes context and makes small models follow
    stale summary instructions. A prompt summary is only retained when there is
    no usable verbatim history.
    """
    profile = str(getattr(startup_profile, "value", startup_profile) or "RECOMMENDED").upper()
    defaults = {
        "FAST": (4, 280),
        "RECOMMENDED": (6, 420),
        "DEEP": (8, 560),
    }
    history_pairs, summary_chars = defaults.get(profile, defaults["RECOMMENDED"])
    if _context_window_hint(model_path) >= 65_536 and profile == "DEEP":
        history_pairs, summary_chars = 10, 700

    override = str(os.getenv("LOCAL_AI_HISTORY_TURNS", "")).strip()
    if override.isdigit():
        history_pairs = max(2, min(int(override), 10))

    payload = digest if isinstance(digest, dict) else {}
    recent = payload.get("recent_turns")
    has_verbatim_history = isinstance(recent, list) and any(
        isinstance(item, dict) and str(item.get("text") or "").strip()
        for item in recent
    )
    return ConversationContextPolicy(
        history_turn_pairs=history_pairs,
        rolling_summary_chars=summary_chars,
        inject_digest_into_legacy_prompt=not has_verbatim_history,
    )


def _context_window_hint(model_path: str | None) -> int:
    path = Path(str(model_path or "")).expanduser()
    config_path = path / "config.json"
    if not config_path.exists():
        return 0
    try:
        payload = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return 0
    if not isinstance(payload, dict):
        return 0
    text_config = payload.get("text_config")
    source = text_config if isinstance(text_config, dict) else payload
    try:
        return max(0, int(source.get("max_position_embeddings") or 0))
    except (AttributeError, TypeError, ValueError):
        return 0
