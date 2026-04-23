from __future__ import annotations
import re
import unicodedata
import time
import os
from pathlib import Path
from typing import Any

def _env_flag(name: str, default: str = "0") -> bool:
    raw = str(os.getenv(name, default) or default).strip().lower()
    return raw in {"1", "true", "yes", "on"}

def _normalized_match_text(query: str) -> str:
    raw = str(query or "").strip()
    if not raw:
        return ""
    return unicodedata.normalize("NFC", raw).casefold()


def _normalize_response_style_profile(profile: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(profile, dict):
        return None
    defaults = {
        "verbosity": "medium",
        "tone": "balanced",
        "format": "paragraph",
        "clarification_policy": "ask_when_ambiguous_only",
        "confidence": 0.62,
        "source": "inferred",
    }
    verbosity = str(profile.get("verbosity") or defaults["verbosity"]).strip().lower()
    if verbosity not in {"short", "medium", "long"}:
        verbosity = defaults["verbosity"]
    tone = str(profile.get("tone") or defaults["tone"]).strip().lower()
    if tone not in {"direct", "balanced", "analytic"}:
        tone = defaults["tone"]
    fmt = str(profile.get("format") or defaults["format"]).strip().lower()
    if fmt not in {"paragraph", "bullets"}:
        fmt = defaults["format"]
    source = str(profile.get("source") or defaults["source"]).strip().lower()
    if source not in {"explicit", "inferred"}:
        source = defaults["source"]
    try:
        confidence = float(profile.get("confidence", defaults["confidence"]))
    except Exception:
        confidence = defaults["confidence"]
    clarification_policy = str(
        profile.get("clarification_policy") or defaults["clarification_policy"]
    ).strip().lower()
    if clarification_policy != "ask_when_ambiguous_only":
        clarification_policy = defaults["clarification_policy"]
    return {
        "verbosity": verbosity,
        "tone": tone,
        "format": fmt,
        "clarification_policy": clarification_policy,
        "confidence": max(0.0, min(1.0, confidence)),
        "source": source,
    }


def _merge_response_style_profile(
    *,
    base: dict[str, Any] | None,
    field_updates: dict[str, str] | None,
    source: str,
    confidence: float,
) -> dict[str, Any]:
    merged = dict(_normalize_response_style_profile(base) or {})
    if not merged:
        merged = {
            "verbosity": "medium",
            "tone": "balanced",
            "format": "paragraph",
            "clarification_policy": "ask_when_ambiguous_only",
            "confidence": 0.62,
            "source": "inferred",
        }
    for key in ("verbosity", "tone", "format"):
        value = ""
        if isinstance(field_updates, dict):
            value = str(field_updates.get(key) or "").strip().lower()
        if value:
            merged[key] = value
    merged["source"] = "explicit" if str(source).strip().lower() == "explicit" else "inferred"
    merged["confidence"] = max(0.0, min(1.0, float(confidence)))
    merged["clarification_policy"] = "ask_when_ambiguous_only"
    return _normalize_response_style_profile(merged) or {
        "verbosity": "medium",
        "tone": "balanced",
        "format": "paragraph",
        "clarification_policy": "ask_when_ambiguous_only",
        "confidence": 0.62,
        "source": "inferred",
    }


def _extract_response_style_signal(query: str) -> dict[str, Any]:
    text = _normalized_match_text(query)
    if not text:
        return {"detected": False, "explicit": False, "global_opt_in": False, "field_updates": {}}
    field_updates: dict[str, str] = {}

    short_tokens = ("짧게", "짧은", "짧고", "간단히", "한 줄", "핵심만", "핵심 위주", "brief", "short", "concise")
    long_tokens = ("자세히", "상세히", "길게", "깊게", "상세하게", "in detail", "detailed")
    medium_tokens = ("보통 길이", "중간 길이", "적당히", "medium length")
    if any(token in text for token in short_tokens):
        field_updates["verbosity"] = "short"
    elif any(token in text for token in long_tokens):
        field_updates["verbosity"] = "long"
    elif any(token in text for token in medium_tokens):
        field_updates["verbosity"] = "medium"

    direct_tokens = ("직설", "돌려 말하지", "바로 말", "direct", "straight to")
    analytic_tokens = ("분석적", "근거와 함께", "이유까지", "analytic", "reasoning")
    balanced_tokens = ("균형", "밸런스", "balanced")
    if any(token in text for token in direct_tokens):
        field_updates["tone"] = "direct"
    elif any(token in text for token in analytic_tokens):
        field_updates["tone"] = "analytic"
    elif any(token in text for token in balanced_tokens):
        field_updates["tone"] = "balanced"

    bullet_tokens = ("불릿", "글머리", "목록", "리스트", "bullet", "bullets", "list format")
    paragraph_tokens = ("문단", "서술", "문장형", "paragraph")
    if any(token in text for token in bullet_tokens):
        field_updates["format"] = "bullets"
    elif any(token in text for token in paragraph_tokens):
        field_updates["format"] = "paragraph"

    if not field_updates:
        return {"detected": False, "explicit": False, "global_opt_in": False, "field_updates": {}}

    directive_tokens = (
        "앞으로", "답해", "답변해", "말해", "해줘", "해 줘", "해주", "해주세요",
        "please", "respond", "answer", "use ",
    )
    explicit = any(token in text for token in directive_tokens)
    global_tokens = ("앞으로 항상", "기본적으로", "항상 이렇게", "always", "by default", "default")
    global_opt_in = explicit and any(token in text for token in global_tokens)
    return {
        "detected": True,
        "explicit": bool(explicit),
        "global_opt_in": bool(global_opt_in),
        "field_updates": field_updates,
    }


def _infer_response_style_updates_from_recent_user_turns(
    user_turns: list[str],
    *,
    window: int = 3,
    min_votes: int = 2,
) -> dict[str, str]:
    rows = [str(item or "").strip() for item in user_turns if str(item or "").strip()]
    if not rows:
        return {}
    samples = rows[-max(1, int(window)) :]
    votes: dict[str, dict[str, int]] = {
        "verbosity": {},
        "tone": {},
        "format": {},
    }
    for text in samples:
        signal = _extract_response_style_signal(text)
        if not signal.get("detected"):
            continue
        if bool(signal.get("explicit")):
            continue
        field_updates = signal.get("field_updates")
        if not isinstance(field_updates, dict):
            continue
        for key in ("verbosity", "tone", "format"):
            value = str(field_updates.get(key) or "").strip().lower()
            if not value:
                continue
            votes[key][value] = int(votes[key].get(value, 0)) + 1
    output: dict[str, str] = {}
    vote_threshold = max(1, int(min_votes))
    for key, bucket in votes.items():
        if not bucket:
            continue
        value, count = max(bucket.items(), key=lambda item: int(item[1]))
        if int(count) >= vote_threshold:
            output[key] = str(value)
    return output

def _is_explicit_web_search_request(query: str) -> bool:
    lowered = _normalized_match_text(query)
    if not lowered:
        return False
    if _contains_explicit_local_only_constraint(lowered):
        return False
    if "http://" in lowered or "https://" in lowered:
        return True
    web_targets = ("인터넷", "웹", "web", "online", "링크", "url", "사이트", "site")
    web_actions = ("검색", "search", "찾아", "look up", "크롤", "crawl")
    return any(t in lowered for t in web_targets) and any(a in lowered for a in web_actions)

def _is_explicit_freshness_web_request(query: str) -> bool:
    lowered = _normalized_match_text(query)
    if not lowered:
        return False
    if _contains_explicit_local_only_constraint(lowered):
        return False
    if _is_memory_recall_query(lowered):
        return False

    freshness_tokens = (
        "최신", "최근", "금일", "오늘", "지금", "현재",
        "latest", "recent", "today", "current", "now",
    )
    request_tokens = (
        "알려", "찾아", "확인", "검증", "조회", "정리",
        "tell", "find", "check", "verify", "look up", "show",
    )
    info_targets = (
        "뉴스", "공지", "공식", "업데이트", "버전", "릴리즈", "배포",
        "가격", "주가", "환율", "금리", "정책", "규정", "법", "날씨",
        "news", "announcement", "official", "update", "version", "release",
        "price", "stock", "exchange rate", "interest rate", "policy", "law", "weather",
    )
    personal_tokens = ("내 ", "내가", "my ", "i ", "me ")
    web_tokens = ("웹", "web", "인터넷", "online", "검색", "search")

    has_freshness = any(token in lowered for token in freshness_tokens)
    has_request = any(token in lowered for token in request_tokens)
    has_info_target = any(token in lowered for token in info_targets)
    has_personal = any(token in lowered for token in personal_tokens)
    has_web_token = any(token in lowered for token in web_tokens)

    if not has_freshness:
        return False
    if has_personal and not has_web_token:
        return False
    return has_request or has_info_target or has_web_token

def _is_memory_recall_query(query: str) -> bool:
    lowered = _normalized_match_text(query)
    if not lowered:
        return False
    if _is_explicit_web_search_request(lowered):
        return False
    # Exclude profile-setting utterances such as:
    # "내 닉네임은 파랑고래야. 기억해줘."
    if any(token in lowered for token in ("별명은", "닉네임은", "코드네임은", "코드명은", "nickname is", "nickname:", "codename is", "codename:")) and any(
        cue in lowered for cue in ("기억해줘", "기억해 줘", "remember this", "remember that")
    ):
        return False

    slot_tokens = (
        "이름", "성함", "별명", "닉네임", "코드네임", "코드명", "선호", "취향",
        "name", "nickname", "codename", "preference", "preferences",
    )
    recall_verbs = (
        "기억", "떠올", "상기", "recall", "remember", "remind",
    )
    temporal_recall_tokens = (
        "아까", "방금", "예전", "이전", "전에", "그때", "지난번",
        "earlier", "before", "previous", "last time",
    )
    personal_reference_tokens = ("내 ", "내가", "my ", "i ", "me ")
    question_tokens = (
        "?", "뭐", "무엇", "언제", "어디", "누가", "몇", "맞",
        "what", "when", "where", "who", "which", "how many", "did i", "was i",
    )
    has_slot_token = any(token in lowered for token in slot_tokens)
    has_recall_verb = any(token in lowered for token in recall_verbs)
    has_temporal_recall = any(token in lowered for token in temporal_recall_tokens)
    has_personal_reference = any(token in lowered for token in personal_reference_tokens)
    is_question = any(token in lowered for token in question_tokens)

    explicit_recall_patterns = (
        r"(아까|이전|전에|지난번).*뭐라고",
        r"(아까|이전|전에|지난번).*(말해|얘기|대화|질문|답변)",
        r"(what|which).*(did i|was i|we).*(say|tell|mention|ask)",
        r"(do you remember|can you recall|remember what)",
    )
    is_explicit_recall = any(re.search(pattern, lowered) for pattern in explicit_recall_patterns)

    # High-precision trigger: require either explicit recall phrasing
    # or recall verb + (slot or temporal) in a question form.
    if is_explicit_recall and is_question:
        return True
    if is_question and has_slot_token and has_personal_reference:
        if not any(token in lowered for token in ("정해", "만들어", "지어", "create", "make")):
            return True
    if is_question and has_recall_verb and (has_slot_token or has_temporal_recall):
        return True
    if is_question and has_slot_token and has_temporal_recall:
        return True

    return False

def _is_followup_web_search_request(*, query: str, last_context: dict | None) -> bool:
    lowered = _normalized_match_text(query)
    if not lowered or not isinstance(last_context, dict):
        return False
    if _contains_explicit_local_only_constraint(lowered):
        return False
    if _is_explicit_web_search_request(lowered):
        return True
    if _is_explicit_freshness_web_request(lowered):
        return True
    if _is_memory_recall_query(lowered):
        return False
    path = str(last_context.get("conversation_path") or "").strip().lower()
    if not path.startswith("external_web_search"):
        return False
    # Keep external-web follow-up strict so normal conversation doesn't get trapped
    # in web search mode after a single web turn.
    web_followup_tokens = (
        "검색", "찾아", "search", "look up", "lookup",
        "근거", "출처", "source", "sources", "링크", "link", "links",
        "뉴스", "최신", "업데이트", "recent", "latest", "update",
    )
    return any(token in lowered for token in web_followup_tokens)

def _should_auto_web_search(*, query: str, parsed_intent, last_context: dict | None) -> bool:
    lowered = _normalized_match_text(query)
    if not lowered:
        return False
    if _contains_explicit_local_only_constraint(lowered):
        return False
    if _is_explicit_web_search_request(lowered):
        return True
    if _is_memory_recall_query(lowered):
        return False
    # Auto-web is conservative: only when the user explicitly asks for
    # freshness/current information.
    if not _is_explicit_freshness_web_request(lowered):
        return False
    scores = _intent_routing_scores(query=query, parsed_intent=parsed_intent, last_context=last_context)
    if _should_prefer_local_file_rag(query=query, parsed_intent=parsed_intent, last_context=last_context):
        return False
    threshold = 0.78
    if float(scores.get("local_file_intent", 0.0)) >= 0.56:
        threshold += 0.16
    if float(scores.get("conversational_chat", 0.0)) >= 0.72:
        threshold += 0.10
    if scores["freshness_web_need"] >= threshold:
        return True
    if _is_followup_web_search_request(query=query, last_context=last_context):
        return float(scores.get("freshness_web_need", 0.0)) >= 0.45
    return False

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

def _has_local_file_intent_cues(*, query: str, parsed_intent=None, last_context: dict | None = None) -> bool:
    lowered = _normalized_match_text(query)
    if not lowered:
        return False
    if _has_local_file_target_cues(lowered):
        return True
    if re.search(r"\b(20\d{2}|19\d{2})\b", lowered) or re.search(r"\b\d{1,2}\s*(월|month)\b", lowered):
        return True
    abstract_file_tokens = (
        "그때", "그거", "그 문서", "그 파일", "예전", "지난", "저번", "비슷한",
        "that file", "that doc", "that document", "that time", "around",
    )
    if any(token in lowered for token in abstract_file_tokens):
        return True
    if parsed_intent is not None:
        intent_name = str(getattr(getattr(parsed_intent, "intent", None), "value", getattr(parsed_intent, "intent", "")) or "").lower()
        if intent_name in {
            "find_file",
            "summarize_file",
            "compare_files",
            "open_file",
            "select_previous_candidate",
            "next_candidate",
            "reduce_scope",
        }:
            return True
    if isinstance(last_context, dict):
        if str(last_context.get("selected_file") or "").strip():
            followup_tokens = ("그거", "이거", "아까", "방금", "이어서", "그때", "that", "previous")
            if any(token in lowered for token in followup_tokens):
                return True
        parsed_target = str(last_context.get("parsed_target") or "").strip()
        if parsed_target and _has_token_overlap(lowered, parsed_target, min_overlap=1):
            return True
    return False


def _contains_explicit_local_only_constraint(query: str) -> bool:
    lowered = _normalized_match_text(query)
    if not lowered:
        return False
    local_only_markers = (
        "웹 말고", "웹검색 말고", "웹 검색 말고",
        "인터넷 말고", "인터넷검색 말고", "인터넷 검색 말고", "온라인 말고",
        "web 말고", "search 말고",
        "로컬에서", "내 파일에서", "워크스페이스에서",
        "local only", "no web", "without web",
    )
    return any(marker in lowered for marker in local_only_markers)

def _should_prefer_local_file_rag(*, query: str, parsed_intent=None, last_context: dict | None = None) -> bool:
    lowered = _normalized_match_text(query)
    if not lowered:
        return False
    if _contains_explicit_local_only_constraint(lowered):
        return True
    if _is_explicit_web_search_request(lowered):
        return False
    scores = _intent_routing_scores(query=lowered, parsed_intent=parsed_intent, last_context=last_context)
    local_score = float(scores.get("local_file_intent", 0.0))
    freshness_score = float(scores.get("freshness_web_need", 0.0))
    conversational_score = float(scores.get("conversational_chat", 0.0))
    if freshness_score >= 0.72:
        return False
    if local_score < 0.62:
        return False
    if local_score < (freshness_score + 0.12):
        return False
    # If the utterance is clearly conversational and not strongly file-oriented, avoid forcing RAG.
    if conversational_score >= 0.74 and local_score < 0.78:
        return False
    return True

def _general_file_intent_force_rag_enabled() -> bool:
    return _env_flag("LOCAL_AI_GENERAL_FILE_INTENT_FORCE_RAG", "1")


def _general_implicit_local_rag_enabled() -> bool:
    return _env_flag("LOCAL_AI_GENERAL_IMPLICIT_LOCAL_RAG", "0")


def _general_file_intent_gate_profile() -> str:
    raw = str(os.getenv("LOCAL_AI_GENERAL_FILE_INTENT_GATE_PROFILE", "balanced") or "balanced").strip().lower()
    if raw in {"balanced", "strict", "conservative"}:
        return raw
    return "balanced"


def _allow_local_rag_for_general_hard_gate(
    *,
    query: str,
    parsed_intent=None,
    last_context: dict | None = None,
) -> tuple[bool, str]:
    intent_name = str(
        getattr(getattr(parsed_intent, "intent", None), "value", getattr(parsed_intent, "intent", "")) or ""
    ).lower()
    if intent_name != "general_chat":
        return True, "non_general_intent"

    lowered = _normalized_match_text(query)
    if _contains_explicit_local_only_constraint(lowered):
        return True, "explicit_local_only_constraint"
    if _is_explicit_web_search_request(lowered):
        return False, "explicit_web_request"

    target_hint = str(getattr(parsed_intent, "target", "") or "").strip() if parsed_intent is not None else None
    if _has_explicit_retrieval_request(query=query, target_hint=target_hint):
        return True, "explicit_retrieval_request"

    if isinstance(last_context, dict):
        selected_file = str(last_context.get("selected_file") or "").strip()
        if selected_file and _has_followup_context_signal(query):
            return True, "selected_file_followup"

    if _general_implicit_local_rag_enabled():
        return True, "implicit_local_rag_enabled"
    return False, "hard_gate_non_explicit_retrieval"


def _should_force_local_rag_for_general(
    *,
    query: str,
    parsed_intent=None,
    last_context: dict | None = None,
    routing_scores: dict[str, float] | None = None,
) -> tuple[bool, str]:
    if not _general_file_intent_force_rag_enabled():
        return False, "force_rag_disabled"
    intent_name = str(getattr(getattr(parsed_intent, "intent", None), "value", getattr(parsed_intent, "intent", "")) or "").lower()
    if intent_name != "general_chat":
        return False, "non_general_intent"
    lowered = _normalized_match_text(query)
    if _contains_explicit_local_only_constraint(lowered):
        return True, "explicit_local_only_constraint"

    profile = _general_file_intent_gate_profile()
    scores = dict(routing_scores or _intent_routing_scores(query=query, parsed_intent=parsed_intent, last_context=last_context))
    local_file_intent = float(scores.get("local_file_intent", 0.0))
    local_reference = float(scores.get("local_reference", 0.0))
    context_followup = float(scores.get("context_followup", 0.0))
    freshness_web_need = float(scores.get("freshness_web_need", 0.0))
    confidence = float(getattr(parsed_intent, "confidence", 0.5) or 0.5) if parsed_intent is not None else 0.5
    ambiguity = str(getattr(parsed_intent, "ambiguity", "clear") or "clear").strip().lower()
    ambiguous_query = ambiguity in {"unclear", "ambiguous", "high"} or confidence < 0.62

    local_composite = (local_file_intent * 0.5) + (local_reference * 0.3) + (context_followup * 0.2)

    if profile == "strict":
        local_threshold = 0.46
        dominance_margin = 0.08
        freshness_ceiling = 0.86
    elif profile == "conservative":
        local_threshold = 0.58
        dominance_margin = 0.18
        freshness_ceiling = 0.72
    else:
        local_threshold = 0.50
        dominance_margin = 0.12
        freshness_ceiling = 0.78

    freshness_dominant = freshness_web_need >= max(freshness_ceiling, local_composite + dominance_margin)
    if freshness_dominant and not ambiguous_query:
        return False, "freshness_dominant"

    has_local_signal = (
        local_composite >= local_threshold
        and local_file_intent >= max(0.56, local_threshold - 0.04)
        and (local_reference >= 0.32 or context_followup >= 0.18 or ambiguous_query)
    )
    if has_local_signal:
        return True, f"{profile}_local_signal"
    return False, "insufficient_local_signal"


def _should_allow_auto_web_trigger(
    *,
    query: str,
    parsed_intent=None,
    last_context: dict | None = None,
    routing_scores: dict[str, float] | None = None,
) -> tuple[bool, str]:
    lowered = _normalized_match_text(query)
    if _contains_explicit_local_only_constraint(lowered):
        return False, "explicit_local_only_constraint"
    if _is_memory_recall_query(lowered):
        return False, "memory_recall_prefers_local"

    scores = dict(routing_scores or _intent_routing_scores(query=query, parsed_intent=parsed_intent, last_context=last_context))
    local_file_intent = float(scores.get("local_file_intent", 0.0))
    local_reference = float(scores.get("local_reference", 0.0))
    context_followup = float(scores.get("context_followup", 0.0))
    freshness_web_need = float(scores.get("freshness_web_need", 0.0))
    confidence = float(getattr(parsed_intent, "confidence", 0.5) or 0.5) if parsed_intent is not None else 0.5
    ambiguity = str(getattr(parsed_intent, "ambiguity", "clear") or "clear").strip().lower()

    local_composite = (local_file_intent * 0.5) + (local_reference * 0.3) + (context_followup * 0.2)
    if local_file_intent >= 0.58 and local_composite >= max(0.48, freshness_web_need - 0.04):
        return False, "local_signal_competes_with_freshness"
    if local_reference >= 0.42 and freshness_web_need <= 0.82:
        return False, "context_reference_prefers_local"
    if ambiguity in {"unclear", "ambiguous", "high"} and local_file_intent >= 0.54 and confidence < 0.7:
        return False, "ambiguous_local_file_intent"
    return True, "freshness_dominant"

def _intent_routing_scores(*, query: str, parsed_intent=None, last_context: dict | None = None) -> dict[str, float]:
    lowered = _normalized_match_text(query)
    if not lowered:
        return {
            "local_file_intent": 0.0,
            "freshness_web_need": 0.0,
            "conversational_chat": 0.0,
            "explicit_web_request": 0.0,
            "temporal_reference": 0.0,
            "context_followup": 0.0,
        }

    confidence = float(getattr(parsed_intent, "confidence", 0.5) or 0.5) if parsed_intent is not None else 0.5

    local_score = 0.0
    local_reference_score = 0.0
    if _has_local_file_target_cues(lowered):
        local_score += 0.55
        local_reference_score += 0.62
    if re.search(r"\b(20\d{2}|19\d{2})\b", lowered) or re.search(r"\b\d{1,2}\s*(월|month)\b", lowered):
        local_score += 0.15
    abstract_file_tokens = (
        "그때", "그거", "그 문서", "그 파일", "예전", "지난", "저번", "비슷한",
        "that file", "that doc", "that document", "that time", "around",
    )
    if any(token in lowered for token in abstract_file_tokens):
        local_score += 0.17
        local_reference_score += 0.20
    if parsed_intent is not None:
        intent_name = str(getattr(getattr(parsed_intent, "intent", None), "value", getattr(parsed_intent, "intent", "")) or "").lower()
        if intent_name in {
            "find_file",
            "summarize_file",
            "compare_files",
            "open_file",
            "select_previous_candidate",
            "next_candidate",
            "reduce_scope",
        }:
            local_score += 0.23
            local_reference_score += 0.15
    if isinstance(last_context, dict):
        if str(last_context.get("selected_file") or "").strip():
            followup_tokens = ("그거", "이거", "아까", "방금", "이어서", "그때", "that", "previous")
            if any(token in lowered for token in followup_tokens):
                local_score += 0.18
                local_reference_score += 0.16
        parsed_target = str(last_context.get("parsed_target") or "").strip()
        if parsed_target and _has_token_overlap(lowered, parsed_target, min_overlap=1):
            local_score += 0.15
            local_reference_score += 0.14
    local_reference_score = max(0.0, min(1.0, local_reference_score))
    local_score = max(0.0, min(1.0, local_score))

    freshness_score = 0.0
    temporal_reference_score = 0.0
    explicit_web_request_score = 0.0
    context_followup_score = 0.0
    if _contains_explicit_local_only_constraint(lowered):
        explicit_web_request_score = 0.0
    elif _is_explicit_web_search_request(lowered):
        freshness_score = 1.0
        explicit_web_request_score = 1.0
    else:
        freshness_tokens = ("최신", "최근", "today", "latest", "뉴스", "update", "version", "공식")
        weak_time_tokens = ("현재", "지금", "now")
        freshness_subject_tokens = (
            "api", "release", "pricing", "가격", "주가", "환율", "금리",
            "정책", "법", "규정", "날씨", "weather", "공지",
        )
        uncertain_tokens = ("모르", "확인", "맞아", "검증", "search", "검색")
        if any(token in lowered for token in freshness_tokens):
            freshness_score += 0.78
            temporal_reference_score += 0.74
        if any(token in lowered for token in weak_time_tokens) and any(token in lowered for token in freshness_subject_tokens):
            freshness_score += 0.44
            temporal_reference_score += 0.36
        if any(token in lowered for token in uncertain_tokens):
            freshness_score += max(0.0, (0.62 - confidence) * 0.95)
        if _is_followup_web_search_request(query=lowered, last_context=last_context):
            freshness_score += 0.56
            context_followup_score += 0.68
    if _contains_explicit_local_only_constraint(lowered):
        freshness_score *= 0.25
    temporal_reference_score = max(0.0, min(1.0, temporal_reference_score))
    context_followup_score = max(0.0, min(1.0, context_followup_score))
    freshness_score = max(0.0, min(1.0, freshness_score))

    conversational_score = 0.0
    if _is_greeting_query(lowered):
        conversational_score += 0.80
    if _is_brief_chat_query(lowered):
        conversational_score += 0.45
    if parsed_intent is not None:
        intent_name = str(getattr(getattr(parsed_intent, "intent", None), "value", getattr(parsed_intent, "intent", "")) or "").lower()
        if intent_name == "general_chat":
            conversational_score += 0.18
    if _has_followup_context_signal(lowered) and not _has_local_file_target_cues(lowered):
        conversational_score += 0.08
    conversational_score = max(0.0, min(1.0, conversational_score))

    return {
        "local_file_intent": round(local_score, 4),
        "freshness_web_need": round(freshness_score, 4),
        "conversational_chat": round(conversational_score, 4),
        "explicit_web_request": round(explicit_web_request_score, 4),
        "temporal_reference": round(temporal_reference_score, 4),
        "context_followup": round(context_followup_score, 4),
        "local_reference": round(local_reference_score, 4),
    }

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
    from . import language_profiles
    profile = language_profiles.profile_for_text(query)
    followup_tokens = tuple(set(profile.followup_tokens + _FOLLOWUP_GENERIC_TOKENS))
    if any(token in lowered for token in followup_tokens):
        return True
    return _looks_progressive_followup_query(query)

def _has_strong_followup_context_signal(query: str) -> bool:
    lowered = _normalized_match_text(query)
    from . import language_profiles
    profile = language_profiles.profile_for_text(query)
    strong_tokens = tuple(set(profile.strong_followup_tokens + _STRONG_FOLLOWUP_GENERIC_TOKENS))
    if not any(token in lowered for token in strong_tokens):
        return False
    token_count = len(re.findall(r"[A-Za-z가-힣0-9_+\-]+", lowered))
    return token_count <= 18


def _looks_progressive_followup_query(query: str) -> bool:
    lowered = _normalized_match_text(query)
    if not lowered:
        return False
    from . import language_profiles
    profile = language_profiles.profile_for_text(query)
    switch_cues = tuple(set(profile.topic_switch_tokens + _TOPIC_SWITCH_GENERIC_TOKENS))
    if any(cue in lowered for cue in switch_cues):
        return False
    progressive_tokens = tuple(set(profile.progressive_followup_tokens + _PROGRESSIVE_GENERIC_TOKENS))
    if not any(token in lowered for token in progressive_tokens):
        return False
    token_count = len(re.findall(r"[A-Za-z가-힣0-9_+\-]+", lowered))
    return token_count <= 24


_FOLLOWUP_GENERIC_TOKENS: tuple[str, ...] = (
    "follow up",
    "as above",
    "same context",
)

_STRONG_FOLLOWUP_GENERIC_TOKENS: tuple[str, ...] = (
    "that one",
    "this one",
    "previous",
    "above",
)

_PROGRESSIVE_GENERIC_TOKENS: tuple[str, ...] = (
    "one more",
    "another",
    "next one",
    "harder",
    "more advanced",
    "go deeper",
    "continue with",
    "もう一つ",
    "もう1つ",
    "もっと",
    "次",
)

_TOPIC_SWITCH_GENERIC_TOKENS: tuple[str, ...] = (
    "switch topic",
    "new topic",
    "different topic",
    "unrelated",
)

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
