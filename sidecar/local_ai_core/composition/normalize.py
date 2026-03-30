from __future__ import annotations

from .composer import ResponseComposer


def strip_instruction_leakage(text: str) -> str:
    """Normalization helper used by composition and memory post-process flows."""
    return ResponseComposer._strip_instruction_leakage(text or "")

