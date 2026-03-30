from __future__ import annotations
import re
import unicodedata
import time
import os
from pathlib import Path
from typing import Any

def _normalized_match_text(query: str) -> str:
    raw = str(query or "").strip()
    if not raw:
        return ""
    return unicodedata.normalize("NFC", raw).casefold()

def _is_explicit_web_search_request(query: str) -> bool:
    lowered = _normalized_match_text(query)
    if not lowered:
        return False
    if "http://" in lowered or "https://" in lowered:
        return True
    web_targets = ("인터넷", "웹", "web", "online", "링크", "url", "사이트", "site")
    web_actions = ("검색", "search", "찾아", "look up", "크롤", "crawl")
    return any(t in lowered for t in web_targets) and any(a in lowered for a in web_actions)

def _is_followup_web_search_request(*, query: str, last_context: dict | None) -> bool:
    lowered = _normalized_match_text(query)
    if not lowered or not isinstance(last_context, dict):
        return False
    if _is_explicit_web_search_request(lowered):
        return True
    path = str(last_context.get("conversation_path") or "").strip().lower()
    if not path.startswith("external_web_search"):
        return False
    followup_tokens = (
        "더", "자세히", "근거", "출처", "링크", "다시", "계속",
        "검색", "찾아", "search", "look up", "lookup",
        "more", "source",
    )
    return any(token in lowered for token in followup_tokens)

def _should_auto_web_search(*, query: str, parsed_intent, last_context: dict | None) -> bool:
    lowered = _normalized_match_text(query)
    if not lowered:
        return False
    freshness_tokens = ("최신", "최근", "today", "latest", "뉴스", "update", "version")
    uncertain_tokens = ("모르", "확인", "맞아", "검증", "search", "검색")
    if any(token in lowered for token in freshness_tokens):
        return True
    if any(token in lowered for token in uncertain_tokens):
        confidence = float(getattr(parsed_intent, "confidence", 0.0) or 0.0)
        return confidence < 0.58
    return _is_followup_web_search_request(query=query, last_context=last_context)

def _web_search_query_for_turn(*, query: str, last_context: dict | None, is_followup_web_search: bool) -> str:
    cleaned = str(query or "").strip()
    if not cleaned or not is_followup_web_search or not isinstance(last_context, dict):
        return cleaned
    anchor = str(last_context.get("last_user_query") or "").strip()
    if not anchor:
        anchor = str(last_context.get("result_summary") or "").strip()
    if not anchor:
        return cleaned
    return f"{anchor}\n\n후속 질문: {cleaned}"

def _is_freshness_sensitive_query(query: str) -> bool:
    lowered = _normalized_match_text(query)
    if not lowered:
        return False
    if _has_local_file_target_cues(lowered):
        # If it's a file query, we only search if it specifically asks for web/latest
        return any(token in lowered for token in ("최신", "인터넷", "웹", "web", "news", "latest"))
        
    freshness_markers = (
        "최신", "최근", "현재", "지금", "today", "now", "latest",
        "업데이트", "update", "release", "version", "버전",
        "뉴스", "news", "속보", "공식", "official", "공지",
        "api", "pricing", "가격", "주가", "환율", "금리",
        "정책", "법", "규정", "날씨", "weather",
    )
    verify_markers = (
        "맞아", "맞나요", "맞는지", "확실", "확인", "검증", "fact-check", "사실",
    )
    search_markers = (
        "검색", "search", "찾아", "look up", "lookup", "크롤", "crawl",
    )
    if re.search(r"(모르겠|모르면|잘\s*모르).*?(검색|search|찾아|crawl|크롤)", lowered):
        return True
    has_freshness = any(token in lowered for token in freshness_markers)
    has_verification = any(token in lowered for token in verify_markers)
    has_search = any(token in lowered for token in search_markers)
    
    # More natural triggers for general info
    if has_freshness:
        return True
    if (has_verification or has_search) and len(lowered) > 10:
        return True
    return False

def _has_local_file_target_cues(query: str) -> bool:
    lowered = _normalized_match_text(query)
    if not lowered:
        return False
    local_doc_tokens = (
        "파일", "문서", "폴더", "디렉토리", ".txt", ".pdf", ".md", ".docx",
        "file", "document", "folder", "directory", "workspace", "주차",
    )
    if any(token in lowered for token in local_doc_tokens):
        return True
    if re.search(r"\.(txt|pdf|md|docx|py|swift|json|yaml|yml)\b", lowered):
        return True
    return False

def _has_explicit_retrieval_request(query: str, *, target_hint: str | None = None) -> bool:
    lowered = _normalized_match_text(query)
    if not lowered:
        return False
    if _is_explicit_web_search_request(lowered):
        return False
    retrieval_targets = (
        "파일", "문서", "폴더", "디렉토리", "경로", "주차", "태그", "open", "file",
        "document", "folder", "directory", "path", "tag", ".txt", ".pdf", ".md", ".docx",
    )
    retrieval_actions = (
        "찾아", "검색", "보여", "열어", "요약", "정리", "리스트", "목록", "있나", "있어",
        "있나요", "있는지", "존재", "find", "search", "show", "open", "list", "summary",
        "summarize", "exists", "is there",
    )
    has_target = any(token in lowered for token in retrieval_targets)
    if target_hint and str(target_hint).strip() and has_target:
        has_target = True
    has_action = any(token in lowered for token in retrieval_actions)
    if has_target and has_action:
        return True
    scope_all_tokens = ("전체", "전부", "모두", "모든", "all", "every", "entire")
    if has_target and any(token in lowered for token in scope_all_tokens):
        return True
    return False

def _contains_task_cues(lowered: str) -> bool:
    if _has_explicit_retrieval_request(lowered):
        return True
    action_cues = ("비교", "분석", "작성", "초안", "rewrite", "draft", "compare", "analysis")
    return any(token in lowered for token in action_cues)

def _is_greeting_query(query: str) -> bool:
    lowered = _normalized_match_text(query)
    if not lowered:
        return False
    if _contains_task_cues(lowered):
        return False
    tokens = ("안녕", "반가워", "고마워", "감사", "hello", "hi", "hey", "thanks", "thank you", "how are you")
    if not any(token in lowered for token in tokens):
        return False
    token_count = len(re.findall(r"[A-Za-z가-힣0-9_+\-]+", lowered))
    return token_count <= 8

def _is_brief_chat_query(query: str) -> bool:
    raw = (query or "").strip()
    if not raw:
        return False
    lowered = raw.lower()
    cues = ("그렇구나", "알겠", "오케이", "그래", "아하", "맞아", "ㅇㅋ", "ok", "okay", "got it", "makes sense", "cool", "thanks", "thank you")
    if any(cue in lowered for cue in cues):
        return True
    compact = re.sub(r"\s+", "", raw)
    return len(compact) <= 8

def _has_followup_context_signal(query: str) -> bool:
    lowered = _normalized_match_text(query)
    followup_tokens = ("그럼", "그러면", "그리고", "근데", "아까", "방금", "이어서", "그거", "이거", "계속", "다시", "then", "and", "continue", "as above", "follow up")
    return any(token in lowered for token in followup_tokens)

def _has_strong_followup_context_signal(query: str) -> bool:
    lowered = _normalized_match_text(query)
    strong_tokens = ("그거", "이거", "아까", "방금", "이어서", "그 파일", "그 문서", "위에서", "that one", "this one", "previous", "above", "continue")
    if not any(token in lowered for token in strong_tokens):
        return False
    token_count = len(re.findall(r"[A-Za-z가-힣0-9_+\-]+", lowered))
    return token_count <= 18

def _tokenize_query_terms(query: str) -> list[str]:
    raw = str(query or "").strip()
    if not raw:
        return []
    return re.findall(r"[A-Za-z가-힣0-9_]+", raw.lower())

def _normalize_response_length(value: str) -> str:
    lowered = str(value or "").strip().lower()
    if lowered in {"short", "medium", "long"}:
        return lowered
    return "long"

def _response_length_rank(value: str) -> int:
    mapping = {"short": 0, "medium": 1, "long": 2}
    return mapping.get(_normalize_response_length(value), 1)

def _response_length_from_rank(rank: int) -> str:
    if rank <= 0:
        return "short"
    if rank == 1:
        return "medium"
    return "long"

def _system_memory_gb() -> int:
    try:
        import psutil
        return int(psutil.virtual_memory().total / (1024**3))
    except (ImportError, Exception):
        return 16

def _model_size_b(reference: str) -> int | None:
    match = re.search(r"(\d+)b", str(reference).lower())
    if match:
        return int(match.group(1))
    return None

def _extract_excluded_weeks(query: str) -> list[int]:
    match = re.search(r"(\d+)\s*(주차|주)\s*(빼고|제외)", str(query or ""))
    if match:
        return [int(match.group(1))]
    return []

def _has_token_overlap(query_text: str, reference_text: str, *, min_overlap: int = 1) -> bool:
    q_tokens = {t for t in _tokenize_query_terms(query_text) if len(t) >= 2}
    r_tokens = {t for t in _tokenize_query_terms(reference_text) if len(t) >= 2}
    if not q_tokens or not r_tokens:
        return False
    overlap = len(q_tokens.intersection(r_tokens))
    return overlap >= max(1, int(min_overlap))

def _conversation_context_relevance(*, query: str, session_digest: dict[str, Any] | None, last_context: dict[str, Any] | None) -> float:
    query_terms = set(_tokenize_query_terms(query))
    if not query_terms:
        return 0.0
    context_terms: set[str] = set()
    digest = session_digest or {}
    raw_turns = digest.get("recent_turns")
    if isinstance(raw_turns, list):
        for item in raw_turns[-8:]:
            if not isinstance(item, dict):
                continue
            if str(item.get("role") or "").strip().lower() != "user":
                continue
            for token in _tokenize_query_terms(str(item.get("text") or "")):
                context_terms.add(token)
    for item in (digest.get("active_topics") or []):
        for token in _tokenize_query_terms(str(item or "")):
            context_terms.add(token)
    if last_context:
        fields = (
            str(last_context.get("parsed_target") or ""),
            str(last_context.get("result_summary") or ""),
            Path(str(last_context.get("selected_file") or "")).name if last_context.get("selected_file") else "",
        )
        for field in fields:
            for token in _tokenize_query_terms(field):
                context_terms.add(token)
    if not context_terms:
        return 0.0
    inter = len(query_terms.intersection(context_terms))
    union = len(query_terms.union(context_terms))
    return inter / union if union > 0 else 0.0

def _conversation_context_budget_tokens(response_length: str, model_profile: str = "recommended") -> int:
    mapping = {"short": 120, "medium": 220, "long": 440}
    base = mapping.get(str(response_length).lower(), 440)
    profile = str(model_profile or "recommended").lower()
    if profile in {"deep", "advanced"}:
        scaled = int(base * 1.15)
    elif profile == "fast":
        scaled = int(base * 0.85)
    else:
        scaled = base
    
    mem_gb = _system_memory_gb()
    if mem_gb <= 16: cap = 320
    elif mem_gb <= 24: cap = 480
    elif mem_gb <= 32: cap = 640
    elif mem_gb <= 64: cap = 896
    else: cap = 1280
    return max(80, min(scaled, cap))

def _estimate_context_tokens(text: str) -> int:
    val = (text or "").strip()
    return max(1, len(val) // 4) if val else 0

def _looks_like_reasoning_leak(text: str) -> bool:
    if not text or len(text) < 10:
        return False
    # Common system instruction terminology that shouldn't appear in final output
    leak_markers = (
        "반드시 '?'를 붙여주세요",
        "3문장까지 가능합니다",
        "답변이 부족할 경우 추가적인 질문",
        "사용자에게 직접 도움을 주세요",
        "사용자의 말에 바로 반응하세요",
        "사용자 메시지에 명확한 답을 하세요",
        "최종 답변:",
        "답변 형식:",
        "유의 사항:",
    )
    return any(marker in text for marker in leak_markers)

def _extract_path_focus_terms(query: str, topics: list[str]) -> tuple[list[str], bool]:
    lowered = _normalized_match_text(query)
    # Check for strong directory focus signals: "폴더", "디렉토리", "folder", "directory"
    is_strict = any(token in lowered for token in ("폴더", "디렉토리", "folder", "directory"))
    
    terms = []
    # If the user explicitly mentions a topic + folder, it's a strong focus
    for topic in topics:
        if topic in lowered:
            terms.append(topic)
            
    # Also extract any words quoted or capitalized in a path-like way?
    # For now, stick to topics.
    return list(set(terms)), is_strict
