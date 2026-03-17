from __future__ import annotations

from dataclasses import dataclass

from .models import StartupProfile, WorkMode
from .vector_store import VectorHit, VectorStore


@dataclass(slots=True)
class RetrievalPreset:
    top_k: int
    min_score: float
    rerank: bool


_BASE_PRESETS: dict[WorkMode, RetrievalPreset] = {
    WorkMode.GENERAL: RetrievalPreset(top_k=5, min_score=0.0, rerank=False),
    WorkMode.SUMMARY: RetrievalPreset(top_k=6, min_score=0.0, rerank=False),
    WorkMode.RESEARCH: RetrievalPreset(top_k=8, min_score=0.0, rerank=True),
    WorkMode.DEVELOPMENT: RetrievalPreset(top_k=7, min_score=0.0, rerank=True),
    WorkMode.WRITING: RetrievalPreset(top_k=5, min_score=0.0, rerank=False),
    WorkMode.PLANNING: RetrievalPreset(top_k=6, min_score=0.0, rerank=False),
    WorkMode.STRICT_SEARCH: RetrievalPreset(top_k=5, min_score=0.45, rerank=True),
}


def preset_for(mode: WorkMode, startup_profile: StartupProfile, explicit_top_k: int | None = None) -> RetrievalPreset:
    base = _BASE_PRESETS[mode]
    top_k = base.top_k
    min_score = base.min_score

    if startup_profile == StartupProfile.FAST:
        top_k = max(3, top_k - 2)
    elif startup_profile == StartupProfile.DEEP:
        top_k = top_k + 3
        min_score = max(0.05, min_score - 0.02)

    if explicit_top_k:
        top_k = explicit_top_k

    return RetrievalPreset(top_k=top_k, min_score=min_score, rerank=base.rerank)


def retrieve_hits(
    vector_store: VectorStore,
    query_vector: list[float],
    mode: WorkMode,
    startup_profile: StartupProfile,
    explicit_top_k: int | None = None,
) -> tuple[RetrievalPreset, list[VectorHit]]:
    preset = preset_for(mode, startup_profile, explicit_top_k=explicit_top_k)
    hits = vector_store.search(query_vector, limit=preset.top_k)
    filtered = [hit for hit in hits if hit.score >= preset.min_score]

    # For strict mode we intentionally avoid speculative answers when confidence is weak.
    if mode == WorkMode.STRICT_SEARCH and (not filtered or filtered[0].score < 0.6):
        return preset, []

    return preset, filtered
