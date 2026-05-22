from __future__ import annotations

import re


_HANGUL_RE = re.compile(r"[\uac00-\ud7a3]")
_HIRAGANA_KATAKANA_RE = re.compile(r"[\u3040-\u30ff]")
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_CYRILLIC_RE = re.compile(r"[\u0400-\u04ff]")
_ARABIC_RE = re.compile(r"[\u0600-\u06ff]")
_LATIN_RE = re.compile(r"[A-Za-z]")
_HAN_RE = re.compile(r"[\u4e00-\u9fff]")


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
    hangul_count = len(_HANGUL_RE.findall(text))
    hira_kata_count = len(_HIRAGANA_KATAKANA_RE.findall(text))
    han_count = len(_HAN_RE.findall(text))
    cyr_count = len(_CYRILLIC_RE.findall(text))
    arabic_count = len(_ARABIC_RE.findall(text))
    latin_count = len(_LATIN_RE.findall(text))

    # Treat Han as Japanese support when Hiragana/Katakana appears together.
    ja_count = hira_kata_count + (han_count if hira_kata_count > 0 else 0)
    zh_count = han_count if hira_kata_count == 0 else 0

    counts = {
        "ko": hangul_count,
        "ja": ja_count,
        "zh": zh_count,
        "ru": cyr_count,
        "ar": arabic_count,
        "en": latin_count,
    }
    lang, score = max(counts.items(), key=lambda item: item[1])
    if score <= 0:
        return "en"
    return lang


def resolve_response_language(query: str, language_preference: str | None) -> str:
    forced = normalize_language_code(language_preference)
    if forced:
        return forced
    return detect_query_language(query)


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
