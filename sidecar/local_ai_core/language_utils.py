from __future__ import annotations

import re


_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
_HIRAGANA_KATAKANA_RE = re.compile(r"[\u3040-\u30ff]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_LATIN_RE = re.compile(r"[A-Za-z]")


def normalize_language_code(value: str | None) -> str | None:
    if value is None:
        return None
    raw = value.strip().lower()
    if not raw:
        return None
    if raw in {"auto", "system", "device", "default"}:
        return None

    alias = {
        "ko": "ko",
        "ko-kr": "ko",
        "korean": "ko",
        "kr": "ko",
        "en": "en",
        "en-us": "en",
        "en-gb": "en",
        "english": "en",
        "ja": "ja",
        "ja-jp": "ja",
        "japanese": "ja",
        "zh": "zh",
        "zh-cn": "zh",
        "zh-tw": "zh",
        "chinese": "zh",
        "es": "es",
        "fr": "fr",
        "de": "de",
        "pt": "pt",
        "it": "it",
        "ru": "ru",
        "ar": "ar",
    }
    if raw in alias:
        return alias[raw]

    if len(raw) >= 2:
        return raw[:2]
    return None


def detect_query_language(query: str) -> str:
    text = (query or "").strip()
    if not text:
        return "en"
    if _HANGUL_RE.search(text):
        return "ko"
    if _HIRAGANA_KATAKANA_RE.search(text):
        return "ja"
    if _CJK_RE.search(text):
        return "zh"
    if _CYRILLIC_RE.search(text):
        return "ru"
    if _ARABIC_RE.search(text):
        return "ar"
    if _LATIN_RE.search(text):
        return "en"
    return "en"


def resolve_response_language(query: str, language_preference: str | None) -> str:
    forced = normalize_language_code(language_preference)
    if forced in {"ko", "en", "ja"}:
        return forced
    if forced:
        return "ko"
    detected = detect_query_language(query)
    if detected in {"ko", "en", "ja"}:
        return detected
    return "ko"


def response_language_instruction(language_code: str) -> str:
    code = normalize_language_code(language_code) or "en"
    names = {
        "ko": "Korean",
        "en": "English",
        "ja": "Japanese",
        "zh": "Chinese",
        "es": "Spanish",
        "fr": "French",
        "de": "German",
        "pt": "Portuguese",
        "it": "Italian",
        "ru": "Russian",
        "ar": "Arabic",
    }
    return f"Respond in {names.get(code, 'English')}."


def insufficient_evidence_message(language_code: str) -> str:
    code = normalize_language_code(language_code) or "en"
    if code == "ko":
        return "근거 부족: 현재 로컬 자료에서 신뢰할 수 있는 근거를 찾지 못했습니다."
    if code == "ja":
        return "根拠不足: 現在のローカル資料から信頼できる根拠を見つけられませんでした。"
    return "Insufficient evidence: no reliable support was found in the current local sources."
