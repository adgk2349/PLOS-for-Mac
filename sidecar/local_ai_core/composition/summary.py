from __future__ import annotations

from .composer import ResponseComposer


def normalize_summary_text(text: str, *, response_language: str = "auto") -> str:
    """Shared summary normalization entrypoint for composition pipeline."""
    return ResponseComposer._naturalize_summary_text(text or "", response_language=response_language)

