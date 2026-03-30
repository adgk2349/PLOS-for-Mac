from __future__ import annotations

from pathlib import Path


def test_legacy_helpers_do_not_call_private_local_inference_directly() -> None:
    root = Path(__file__).resolve().parents[1]
    helper_dir = root / "local_ai_core" / "reasoning" / "helpers"
    targets = [
        helper_dir / "core_chat_helpers.py",
        helper_dir / "core_chat_post_helpers.py",
        helper_dir / "web_search_helpers.py",
        helper_dir / "formatting_helpers.py",
    ]
    banned = ("_executor._local_inference.", "execute_conversation(")
    for path in targets:
        text = path.read_text(encoding="utf-8")
        for pattern in banned:
            assert pattern not in text, f"{path} contains banned sync pattern: {pattern}"
