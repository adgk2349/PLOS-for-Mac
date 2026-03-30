from __future__ import annotations


_LANGUAGE_ALIAS = {
    "ko-kr": "ko",
    "korean": "ko",
    "en-us": "en",
    "english": "en",
    "ja-jp": "ja",
    "japanese": "ja",
    "auto": "auto",
    "ko": "ko",
    "en": "en",
    "ja": "ja",
}


def normalize_language(value: str | None) -> str:
    raw = str(value or "").strip().lower()
    return _LANGUAGE_ALIAS.get(raw, "auto")

