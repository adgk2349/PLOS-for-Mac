from __future__ import annotations
from dataclasses import dataclass, field
from typing import Tuple
import re

@dataclass
class LanguageProfile:
    lang_code: str
    followup_tokens: Tuple[str, ...] = field(default_factory=tuple)
    strong_followup_tokens: Tuple[str, ...] = field(default_factory=tuple)
    topic_switch_tokens: Tuple[str, ...] = field(default_factory=tuple)
    progressive_followup_tokens: Tuple[str, ...] = field(default_factory=tuple)

    @property
    def name(self) -> str:
        return self.lang_code

# Define profiles for supported languages
KOREAN_PROFILE = LanguageProfile(
    lang_code="ko",
    followup_tokens=("그럼", "그래서", "근데", "그리고", "또한", "더", "또", "어떻게", "왜"),
    strong_followup_tokens=("이거", "그거", "저거", "방금", "그거말고", "다시", "한번더"),
    topic_switch_tokens=("다른거", "그거말고", "다른", "주제", "바꿔서"),
    progressive_followup_tokens=("좀더", "계속", "더", "그리고나서", "다음")
)

ENGLISH_PROFILE = LanguageProfile(
    lang_code="en",
    followup_tokens=("then", "so", "but", "and", "also", "more", "how", "why", "what"),
    strong_followup_tokens=("this", "that", "it", "it?", "mention", "repeat", "explain"),
    topic_switch_tokens=("other", "different", "change", "topic", "switch"),
    progressive_followup_tokens=("more", "continue", "next", "then", "after")
)

JAPANESE_PROFILE = LanguageProfile(
    lang_code="ja",
    followup_tokens=("じゃあ", "それでは", "それから", "でも", "また", "もっと", "どうやって", "왜"),
    strong_followup_tokens=("これ", "それ", "あれ", "さっき", "もう一度", "詳しく"),
    topic_switch_tokens=("他", "違う", "変わり", "話題", "変更"),
    progressive_followup_tokens=("もっと", "次", "それから", "続けて")
)

DEFAULT_PROFILE = ENGLISH_PROFILE


def sentence_terminal_chars_for_text(text: str) -> tuple[str, ...]:
    profile = profile_for_text(text)
    if profile.lang_code == "ko":
        return (".", "!", "?", "다", "요")
    if profile.lang_code == "ja":
        return ("。", "！", "？")
    return (".", "!", "?")

def profile_for_text(text: str) -> LanguageProfile:
    """
    Detects language and returns the corresponding LanguageProfile.
    """
    if not text:
        return DEFAULT_PROFILE
    
    # Simple regex based detection
    if re.search(r"[\uac00-\ud7a3]", text):
        return KOREAN_PROFILE
    if re.search(r"[\u3040-\u30ff\u4e00-\u9faf]", text): # Hiragana, Katakana, Kanji
        return JAPANESE_PROFILE
    
    return ENGLISH_PROFILE
