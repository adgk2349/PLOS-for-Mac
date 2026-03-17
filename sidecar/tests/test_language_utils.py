from local_ai_core.language_utils import (
    detect_query_language,
    insufficient_evidence_message,
    resolve_response_language,
    response_language_instruction,
)


def test_detect_query_language_for_korean_and_english():
    assert detect_query_language("이 문서 요약해줘") == "ko"
    assert detect_query_language("Please summarize this document.") == "en"


def test_resolve_response_language_prefers_explicit_setting():
    assert resolve_response_language("Please answer in English", "ko-KR") == "ko"
    assert resolve_response_language("한국어로 답해줘", "auto") == "ko"


def test_language_messages():
    assert "근거 부족" in insufficient_evidence_message("ko")
    assert "Insufficient evidence" in insufficient_evidence_message("en")
    assert response_language_instruction("ko").startswith("Respond in Korean")

