from __future__ import annotations
from dataclasses import dataclass
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ..models import LocalEngine

@dataclass(slots=True)
class InferenceResult:
    answer: str
    engine_used: 'LocalEngine'
    used_fallback: bool = False
    detail: str | None = None

@dataclass(slots=True)
class _ConversationCandidateResult:
    answer: str | None
    rewrite_used: bool = False
    quality_repair_reason: str | None = None
    repair_triggered: bool = False
    repair_success: bool = False
    leak_blocked: bool = False
    direct_first_applied: bool = False
    question_count_after_postprocess: int = 0
    recommendation_shape: str | None = None
