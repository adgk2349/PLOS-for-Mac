from __future__ import annotations

import re
from dataclasses import dataclass

from .models import ChatFilters, StartupProfile, WorkMode
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

_CATEGORY_HINTS = {
    "학습자료": ("학습", "강의", "study", "course", "노트"),
    "프로젝트문서": ("프로젝트", "기획", "spec", "proposal", "prd"),
    "회의록": ("회의", "미팅", "meeting", "minutes"),
    "아이디어": ("아이디어", "idea", "브레인스토밍"),
    "개인메모": ("메모", "일기", "journal", "private"),
    "참고자료": ("참고", "reference", "paper", "article"),
    "코드관련": ("코드", "api", "swift", "python", "typescript"),
}


def preset_for(
    mode: WorkMode,
    startup_profile: StartupProfile,
    explicit_top_k: int | None = None,
    *,
    query: str | None = None,
) -> RetrievalPreset:
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
    else:
        depth_boost = _query_depth_boost(query or "")
        top_k = min(14, top_k + depth_boost)
        if depth_boost >= 2 and mode != WorkMode.STRICT_SEARCH:
            min_score = max(0.0, min_score - 0.02)

    return RetrievalPreset(top_k=top_k, min_score=min_score, rerank=base.rerank)


def extract_query_hints(query: str) -> ChatFilters:
    text = query.lower()

    category = None
    for target, keywords in _CATEGORY_HINTS.items():
        if any(keyword in text for keyword in keywords):
            category = target
            break

    year = None
    year_match = re.search(r"(19|20)\d{2}", query)
    if year_match:
        year = int(year_match.group(0))

    tags = re.findall(r"#([A-Za-z가-힣0-9_+-]{2,24})", query)
    tags = [tag.strip() for tag in tags if tag.strip()]

    project = None
    project_match = re.search(r"(?:project|프로젝트)\s*[:\-]?\s*([A-Za-z가-힣0-9 _\-]{2,40})", query, re.IGNORECASE)
    if project_match:
        project = project_match.group(1).strip()

    return ChatFilters(category=category, tags=tags, year=year, project=project, excluded=False)


def merge_filters(base: ChatFilters | None, hint: ChatFilters | None) -> ChatFilters | None:
    if base is None and hint is None:
        return None
    base = base or ChatFilters()
    hint = hint or ChatFilters()

    category = base.category or hint.category
    tags = list(dict.fromkeys((base.tags or []) + (hint.tags or [])))
    year = base.year if base.year is not None else hint.year
    project = base.project or hint.project
    excluded = base.excluded if base.excluded is not None else hint.excluded
    return ChatFilters(category=category, tags=tags, year=year, project=project, excluded=excluded)


def retrieve_hits(
    vector_store: VectorStore,
    query_vector: list[float],
    mode: WorkMode,
    startup_profile: StartupProfile,
    explicit_top_k: int | None = None,
    query: str | None = None,
    *,
    allowed_doc_ids: set[str] | None = None,
    filters: ChatFilters | None = None,
    metadata_map: dict[str, dict] | None = None,
) -> tuple[RetrievalPreset, list[VectorHit]]:
    preset = preset_for(mode, startup_profile, explicit_top_k=explicit_top_k, query=query)
    search_limit = max(preset.top_k * 8, 30)
    hits = vector_store.search(query_vector, limit=search_limit)

    if allowed_doc_ids is not None:
        hits = [hit for hit in hits if hit.doc_id in allowed_doc_ids]

    if filters and metadata_map:
        hits = _rerank_with_metadata(hits, filters=filters, metadata_map=metadata_map)

    filtered = [hit for hit in hits if hit.score >= preset.min_score]
    filtered = filtered[: preset.top_k]

    if mode == WorkMode.STRICT_SEARCH and (not filtered or filtered[0].score < 0.6):
        return preset, []

    return preset, filtered


def _rerank_with_metadata(
    hits: list[VectorHit],
    *,
    filters: ChatFilters,
    metadata_map: dict[str, dict],
) -> list[VectorHit]:
    rescored: list[VectorHit] = []
    wanted_tags = {tag.lower() for tag in filters.tags}
    for hit in hits:
        row = metadata_map.get(hit.doc_id) or {}
        bonus = 0.0
        if filters.category and row.get("category") == filters.category:
            bonus += 0.06
        if filters.year is not None and row.get("year") == filters.year:
            bonus += 0.04
        if filters.project and str(row.get("project") or "").lower().find(filters.project.lower()) >= 0:
            bonus += 0.05
        if wanted_tags:
            row_tags = {tag.lower() for tag in row.get("tags", [])}
            overlap = len(wanted_tags.intersection(row_tags))
            bonus += min(overlap, 2) * 0.03

        hit.score = hit.score + bonus
        rescored.append(hit)

    rescored.sort(key=lambda item: item.score, reverse=True)
    return rescored


def _query_depth_boost(query: str) -> int:
    text = (query or "").strip()
    if not text:
        return 0

    lowered = text.lower()
    tokens = re.findall(r"[A-Za-z가-힣0-9_+-]+", text)
    token_count = len(tokens)
    score = 0

    if token_count >= 18:
        score += 1
    if token_count >= 34:
        score += 1

    depth_keywords = (
        "비교",
        "근거",
        "왜",
        "원인",
        "어떻게",
        "단계",
        "설계",
        "tradeoff",
        "compare",
        "analysis",
        "analyze",
        "reason",
        "how",
        "why",
        "step",
        "architecture",
    )
    if any(keyword in lowered for keyword in depth_keywords):
        score += 1

    if text.count("?") >= 2:
        score += 1

    return min(score, 3)
