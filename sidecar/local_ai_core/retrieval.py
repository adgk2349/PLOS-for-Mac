from __future__ import annotations

import re
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

from .models import (
    BehaviorPolicy,
    ChatFilters,
    ChunkCandidate,
    FileCandidate,
    RetrievalBundle,
    StartupProfile,
    WorkMode,
)
from .vector_store import VectorHit, VectorStore


@dataclass(slots=True)
class RetrievalPreset:
    top_k: int
    min_score: float
    rerank: bool


_BASE_PRESETS: dict[WorkMode, RetrievalPreset] = {
    WorkMode.GENERAL: RetrievalPreset(top_k=5, min_score=0.14, rerank=False),
    WorkMode.SUMMARY: RetrievalPreset(top_k=6, min_score=0.14, rerank=False),
    WorkMode.RESEARCH: RetrievalPreset(top_k=8, min_score=0.18, rerank=True),
    WorkMode.DEVELOPMENT: RetrievalPreset(top_k=7, min_score=0.18, rerank=True),
    WorkMode.WRITING: RetrievalPreset(top_k=5, min_score=0.14, rerank=False),
    WorkMode.PLANNING: RetrievalPreset(top_k=6, min_score=0.14, rerank=False),
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


def retrieve_bundle(
    *,
    vector_store: VectorStore,
    query_vector: list[float],
    mode: WorkMode,
    startup_profile: StartupProfile,
    query: str,
    allowed_doc_ids: set[str],
    filters: ChatFilters,
    metadata_map: dict[str, dict],
    behavior_policy: BehaviorPolicy | None,
    explicit_top_k: int | None = None,
) -> tuple[RetrievalPreset, RetrievalBundle]:
    preset = preset_for(mode, startup_profile, explicit_top_k=explicit_top_k, query=query)
    search_limit = max(preset.top_k * 10, 40)
    hits = vector_store.search(query_vector, limit=search_limit)
    hits = [hit for hit in hits if hit.doc_id in allowed_doc_ids]
    if not hits:
        return preset, RetrievalBundle(applied_filters=filters, rerank_features={"query_depth": 0.0})

    rescored = _rerank_v2(
        hits=hits,
        mode=mode,
        filters=filters,
        metadata_map=metadata_map,
        behavior_policy=behavior_policy,
    )
    rescored = [hit for hit in rescored if hit.score >= preset.min_score][: max(preset.top_k, 1)]
    chunk_candidates = [_chunk_candidate_from_hit(hit, metadata_map) for hit in rescored]
    file_candidates = _file_candidates_from_chunks(chunk_candidates)
    bundle = RetrievalBundle(
        file_candidates=file_candidates,
        chunk_candidates=chunk_candidates,
        applied_filters=filters,
        rerank_features={
            "query_depth": float(_query_depth_boost(query)),
            "top_score": float(chunk_candidates[0].score if chunk_candidates else 0.0),
            "file_count": float(len(file_candidates)),
        },
    )
    return preset, bundle


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


def _rerank_v2(
    *,
    hits: list[VectorHit],
    mode: WorkMode,
    filters: ChatFilters,
    metadata_map: dict[str, dict],
    behavior_policy: BehaviorPolicy | None,
) -> list[VectorHit]:
    rescored: list[VectorHit] = []
    wanted_tags = {tag.lower() for tag in filters.tags}
    now_ts = datetime.now(timezone.utc).timestamp()

    for hit in hits:
        row = metadata_map.get(hit.doc_id) or {}
        semantic = hit.score

        metadata_bonus = 0.0
        if filters.category and row.get("category") == filters.category:
            metadata_bonus += 0.06
        if filters.year is not None and row.get("year") == filters.year:
            metadata_bonus += 0.05
        if filters.project and str(row.get("project") or "").lower().find(filters.project.lower()) >= 0:
            metadata_bonus += 0.05
        if wanted_tags:
            overlap = len(wanted_tags.intersection({tag.lower() for tag in row.get("tags", [])}))
            metadata_bonus += min(overlap, 3) * 0.03

        age_days = max(0.0, (now_ts - float(hit.modified_at)) / 86400.0)
        recency_bonus = max(0.0, 0.08 - min(age_days, 90.0) * 0.001)

        mode_bonus = 0.0
        if mode in {WorkMode.RESEARCH, WorkMode.DEVELOPMENT}:
            mode_bonus += min(float(row.get("importance", 0.5)), 1.0) * 0.04
        if mode == WorkMode.STRICT_SEARCH:
            mode_bonus += 0.01

        workspace_weight = _workspace_weight(hit.file_path, behavior_policy)
        workspace_bonus = (workspace_weight - 1.0) * 0.12

        hit.score = semantic + metadata_bonus + recency_bonus + mode_bonus + workspace_bonus
        rescored.append(hit)

    rescored.sort(key=lambda item: item.score, reverse=True)
    return rescored


def _workspace_weight(file_path: str, behavior_policy: BehaviorPolicy | None) -> float:
    if not behavior_policy or not behavior_policy.workspace_weights:
        return 1.0

    resolved = str(Path(file_path).expanduser())
    best = 1.0
    for prefix, raw_weight in behavior_policy.workspace_weights.items():
        if not prefix:
            continue
        if resolved.startswith(str(Path(prefix).expanduser())):
            try:
                weight = float(raw_weight)
            except Exception:
                weight = 1.0
            best = max(best, max(0.5, min(weight, 1.8)))
    return best


def _chunk_candidate_from_hit(hit: VectorHit, metadata_map: dict[str, dict]) -> ChunkCandidate:
    row = metadata_map.get(hit.doc_id) or {}
    snippet = (hit.text[:320] + "...") if len(hit.text) > 320 else hit.text
    return ChunkCandidate(
        doc_id=hit.doc_id,
        chunk_id=hit.chunk_id,
        file_path=hit.file_path,
        snippet=snippet,
        score=hit.score,
        modified_at=datetime.fromtimestamp(hit.modified_at, tz=timezone.utc),
        category=row.get("category", "참고자료"),
        tags=row.get("tags", []),
    )


def _file_candidates_from_chunks(chunks: list[ChunkCandidate]) -> list[FileCandidate]:
    grouped: dict[str, list[ChunkCandidate]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.doc_id, []).append(chunk)

    output: list[FileCandidate] = []
    for doc_id, items in grouped.items():
        items.sort(key=lambda item: item.score, reverse=True)
        top = items[0]
        score = top.score if len(items) == 1 else ((top.score * 0.7) + (items[1].score * 0.3))
        output.append(
            FileCandidate(
                doc_id=doc_id,
                file_path=top.file_path,
                score=score,
                modified_at=top.modified_at,
                category=top.category,
                tags=top.tags,
            )
        )

    output.sort(key=lambda item: item.score, reverse=True)
    return output[:8]


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
