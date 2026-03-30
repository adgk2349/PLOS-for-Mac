from __future__ import annotations

import re
from dataclasses import replace
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path

from ..models import (
    BehaviorPolicy,
    ChatFilters,
    ChunkCandidate,
    FileCandidate,
    RetrievalBundle,
    StartupProfile,
    WorkMode,
    WorkspaceResponse,
)
from ..embedding import RerankerService
from ..vector_store import VectorHit, VectorStore

# ---------------------------------------------------------------------------
# Shared constants
# ---------------------------------------------------------------------------
STRICT_SEARCH_THRESHOLD = 0.7       # Minimum score for STRICT_SEARCH to return results
STRICT_SEARCH_SHORT_CIRCUIT = 0.7   # Used in reasoning_pipeline short-circuit check
LOW_RELEVANCE_THRESHOLD = 0.40      # Fallback short-circuit for other modes


@dataclass(slots=True)
class RetrievalPreset:
    top_k: int
    min_score: float
    rerank: bool
    # Multiplier for search_limit = max(top_k * search_factor, search_floor)
    search_factor: int = 10
    search_floor: int = 40


_BASE_PRESETS: dict[WorkMode, RetrievalPreset] = {
    WorkMode.GENERAL:       RetrievalPreset(top_k=5,  min_score=0.14, rerank=False),
    WorkMode.SUMMARY:       RetrievalPreset(top_k=6,  min_score=0.14, rerank=False),
    WorkMode.RESEARCH:      RetrievalPreset(top_k=8,  min_score=0.18, rerank=True),
    WorkMode.DEVELOPMENT:   RetrievalPreset(top_k=7,  min_score=0.18, rerank=True),
    WorkMode.WRITING:       RetrievalPreset(top_k=5,  min_score=0.14, rerank=False),
    WorkMode.PLANNING:      RetrievalPreset(top_k=6,  min_score=0.14, rerank=False),
    WorkMode.STRICT_SEARCH: RetrievalPreset(top_k=5,  min_score=0.45, rerank=True),
}

_CATEGORY_HINTS = {
    "학습자료":   ("학습", "강의", "study", "course", "노트"),
    "프로젝트문서": ("프로젝트", "기획", "spec", "proposal", "prd"),
    "회의록":    ("회의", "미팅", "meeting", "minutes"),
    "아이디어":   ("아이디어", "idea", "브레인스토밍"),
    "개인메모":   ("메모", "일기", "journal", "private"),
    "참고자료":   ("참고", "reference", "paper", "article"),
    "코드관련":   ("코드", "api", "swift", "python", "typescript"),
}

_CODE_FILE_EXTENSIONS = {
    ".swift", ".m", ".mm", ".h", ".hpp", ".c", ".cc", ".cpp",
    ".py", ".js", ".jsx", ".ts", ".tsx", ".go", ".rs", ".java", ".kt",
    ".rb", ".php", ".cs", ".scala", ".sql", ".sh", ".zsh", ".bash",
    ".yaml", ".yml", ".json", ".toml", ".ini", ".xml",
}
_CODE_QUERY_HINTS = {
    "bug", "fix", "error", "traceback", "exception", "stack",
    "refactor", "review", "code", "function", "class", "method",
    "swift", "python", "typescript", "java", "kotlin", "api", "compile",
    "빌드", "컴파일", "오류", "에러", "버그", "수정", "리팩터", "코드", "함수", "클래스",
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

    return RetrievalPreset(
        top_k=top_k,
        min_score=min_score,
        rerank=base.rerank,
        search_factor=base.search_factor,
        search_floor=base.search_floor,
    )


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
    search_limit = max(preset.top_k * preset.search_factor, preset.search_floor)
    # Level 2: Use hybrid search (Dense + Sparse)
    hits = vector_store.search_hybrid(query_text=query or "", query_vector=query_vector, limit=search_limit)

    if allowed_doc_ids is not None:
        hits = [hit for hit in hits if hit.doc_id in allowed_doc_ids]

    if filters and metadata_map:
        hits = _rerank_with_metadata(hits, filters=filters, metadata_map=metadata_map)

    filtered = [hit for hit in hits if hit.score >= preset.min_score]
    filtered = filtered[: preset.top_k]

    if mode == WorkMode.STRICT_SEARCH and (not filtered or filtered[0].score < STRICT_SEARCH_THRESHOLD):
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
    reranker: RerankerService | None = None,
    response_language: str = "auto",
    explicit_top_k: int | None = None,
    search_limit_override: int | None = None,
) -> tuple[RetrievalPreset, RetrievalBundle]:
    preset = preset_for(mode, startup_profile, explicit_top_k=explicit_top_k, query=query)
    search_limit = search_limit_override or max(preset.top_k * preset.search_factor, preset.search_floor)
    # Level 2: Use hybrid search (Dense + Sparse)
    hits = vector_store.search_hybrid(query_text=query, query_vector=query_vector, limit=search_limit)
    hits = [hit for hit in hits if hit.doc_id in allowed_doc_ids]
    if not hits:
        return preset, RetrievalBundle(applied_filters=filters, rerank_features={"query_depth": 0.0})

    rescored = _rerank_v2(
        hits=hits,
        query=query,
        mode=mode,
        filters=filters,
        metadata_map=metadata_map,
        behavior_policy=behavior_policy,
        reranker=reranker,
        language=response_language,
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
    """Rerank hits using metadata bonuses. Returns new VectorHit instances (no in-place mutation)."""
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

        rescored.append(hit.with_score(hit.score + bonus))

    rescored.sort(key=lambda item: item.score, reverse=True)
    return rescored


def _rerank_v2(
    *,
    hits: list[VectorHit],
    query: str,
    mode: WorkMode,
    filters: ChatFilters,
    metadata_map: dict[str, dict],
    behavior_policy: BehaviorPolicy | None,
    reranker: RerankerService | None = None,
    language: str = "auto",
) -> list[VectorHit]:
    """Full reranking with metadata, recency, mode and workspace bonuses. Immutable score update."""
    rescored: list[VectorHit] = []
    wanted_tags = {tag.lower() for tag in filters.tags}
    query_tokens = set(re.findall(r"[A-Za-z가-힣0-9]{2,}", query.lower()))
    now_ts = datetime.now(timezone.utc).timestamp()

    for hit in hits:
        row = metadata_map.get(hit.doc_id) or {}
        # Hybrid Search: Token Overlap Bonus (BM25-lite)
        hit_tokens = set(re.findall(r"[A-Za-z가-힣0-9]{2,}", hit.text.lower()))
        overlap = len(query_tokens.intersection(hit_tokens))
        token_bonus = min(overlap * 0.05, 0.25)
        
        # LanceDB cosine-distance conversion may yield scores in [-1, 1].
        # We assume hit.score is already normalized relevance in [0, 1].
        score = hit.score + token_bonus
        # Normalize first so metadata bonuses cannot overtake weak semantic hits too aggressively.
        semantic = max(0.0, min((float(hit.score) + 1.0) / 2.0, 1.0))

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
        recency_bonus = max(0.0, 0.06 - min(age_days, 90.0) * 0.0007)

        mode_bonus = 0.0
        if mode in {WorkMode.RESEARCH, WorkMode.DEVELOPMENT}:
            mode_bonus += min(float(row.get("importance", 0.5)), 1.0) * 0.04
        if mode == WorkMode.STRICT_SEARCH:
            mode_bonus += 0.01
        if mode == WorkMode.DEVELOPMENT:
            suffix = Path(str(hit.file_path or "")).suffix.lower()
            row_tags = {str(tag).lower() for tag in (row.get("tags") or []) if str(tag).strip()}
            doc_type = str(row.get("document_type") or "").lower()
            query_is_codeish = bool(query_tokens.intersection(_CODE_QUERY_HINTS))
            if suffix in _CODE_FILE_EXTENSIONS:
                mode_bonus += 0.08
            else:
                mode_bonus -= 0.03 if query_is_codeish else 0.0
            if (
                "code" in doc_type
                or "source" in doc_type
                or "function" in doc_type
                or "api" in doc_type
                or bool(row_tags.intersection({"code", "source", "api", "function", "class"}))
            ):
                mode_bonus += 0.04

        workspace_weight = _workspace_weight(hit.file_path, behavior_policy)
        workspace_bonus = (workspace_weight - 1.0) * 0.08

        bonus_total = metadata_bonus + recency_bonus + mode_bonus + workspace_bonus
        bonus_total = max(-0.12, min(0.24, bonus_total))
        new_score = semantic + bonus_total
        
        # Reliability Calculation (Phase 20)
        reliability = _calculate_reliability(hit.file_path, hit.modified_at, row)
        
        rescored.append(hit.with_score(new_score, reliability=reliability))

    # Neural Reranking: Cross-Encoder Score Override
    if reranker is not None and reranker.is_available and len(rescored) > 0:
        # Only rerank top N vector hits to save latency
        top_n = min(len(rescored), 12)
        subset = rescored[:top_n]
        texts = [h.text for h in subset]
        neural_scores = reranker.rerank(query, texts, language=language)
        
        for i, ns in enumerate(neural_scores):
            # Combine vector/heuristic score with neural score (Weighted Average)
            # ms-marco scores are often logits, so we might need sigmoid or just weight them
            # Here we boost high neural scores
            original = subset[i].score
            combined = (original * 0.4) + (ns * 0.6)
            rescored[i] = subset[i].with_score(combined)

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


def _calculate_reliability(file_path: str, modified_at_ts: float, row: dict) -> float:
    """
    Calculate a reliability score (0.0 to 1.0) based on file metadata.
    Criteria:
    - Extension: Source Code (1.0) > Docs (0.85) > Config (0.7) > Logs/Git (0.3)
    - Recency: Newer files are generally more reliable for active projects.
    - Path: Avoid 'build', 'dist', 'temp' folders.
    """
    path_str = file_path.lower()
    ext = Path(path_str).suffix.lower()
    
    # 1. Extension Baseline
    if ext in {".swift", ".py", ".ts", ".js", ".go", ".c", ".cpp", ".h", ".rs", ".java", ".kt"}:
        rel = 1.0
    elif ext in {".md", ".txt", ".markdown", ".pdf", ".docx"}:
        rel = 0.85
    elif ext in {".json", ".yaml", ".yml", ".xml", ".plist", ".toml"}:
        rel = 0.70
    elif ext in {".log", ".sql", ".csv", ".tsv", ".tmp"}:
        rel = 0.40
    else:
        rel = 0.50

    # 2. Recency Decay (Phase 20 logic)
    # Penalty if older than 180 days (approx 6 months)
    now_ts = datetime.now(timezone.utc).timestamp()
    age_days = (now_ts - modified_at_ts) / 86400.0
    if age_days > 180:
        # Scale down to 0.6 of baseline over a year
        decay = max(0.6, 1.0 - (age_days - 180) / 365.0)
        rel *= decay

    # 3. Path Penalty
    if any(p in path_str for p in ["/build/", "/dist/", "/node_modules/", "/temp/", "/cache/", "/.git/"]):
        rel *= 0.4
    
    # 4. Row Importance (from Phase 18 indexing)
    importance = float(row.get("importance", 0.5))
    # Importance acts as a slight correction (+/- 0.1)
    rel = rel + (importance - 0.5) * 0.2

    return max(0.1, min(1.0, rel))


def _chunk_candidate_from_hit(hit: VectorHit, metadata_map: dict[str, dict]) -> ChunkCandidate:
    row = metadata_map.get(hit.doc_id) or {}
    
    # Phase 16: Contextual RAG Cleanup
    # Strip artificial context prefix for cleaner UI display (snippets)
    clean_text = hit.text
    if clean_text.startswith("[File: "):
        end_idx = clean_text.find("] ")
        if end_idx != -1:
            clean_text = clean_text[end_idx + 2:]
            
    snippet = (clean_text[:320] + "...") if len(clean_text) > 320 else clean_text
    return ChunkCandidate(
        doc_id=hit.doc_id,
        chunk_id=hit.chunk_id,
        file_path=hit.file_path,
        snippet=snippet,
        score=hit.score,
        modified_at=datetime.fromtimestamp(hit.modified_at, tz=timezone.utc),
        category=row.get("category", "참고자료"),
        tags=row.get("tags", []),
        reliability=hit.reliability,
    )


def _file_candidates_from_chunks(chunks: list[ChunkCandidate]) -> list[FileCandidate]:
    grouped: dict[str, list[ChunkCandidate]] = {}
    for chunk in chunks:
        grouped.setdefault(chunk.doc_id, []).append(chunk)

    output: list[FileCandidate] = []
    for doc_id, items in grouped.items():
        items.sort(key=lambda item: item.score, reverse=True)
        top = items[0]
        max_reliability = max(c.reliability for c in items)
        
        if len(items) == 1:
            score = top.score
        elif len(items) == 2:
            score = (top.score * 0.7) + (items[1].score * 0.3)
        else:
            # Weighted average: top chunk 60%, second 25%, rest equally split for 15%
            rest_avg = sum(c.score for c in items[2:]) / len(items[2:])
            score = (top.score * 0.60) + (items[1].score * 0.25) + (rest_avg * 0.15)
        output.append(
            FileCandidate(
                doc_id=doc_id,
                file_path=top.file_path,
                score=score,
                modified_at=top.modified_at,
                category=top.category,
                tags=top.tags,
                reliability=max_reliability,
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
