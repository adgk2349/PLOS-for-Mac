from local_ai_core.models import StartupProfile, WorkMode
from local_ai_core.retrieval import extract_query_hints, merge_filters, preset_for


def test_fast_profile_reduces_top_k():
    preset = preset_for(WorkMode.RESEARCH, StartupProfile.FAST)
    assert preset.top_k < 8


def test_deep_profile_increases_top_k_and_lowers_threshold():
    preset = preset_for(WorkMode.GENERAL, StartupProfile.DEEP)
    assert preset.top_k > 5
    assert preset.min_score < 0.15


def test_strict_search_has_high_min_score():
    preset = preset_for(WorkMode.STRICT_SEARCH, StartupProfile.RECOMMENDED)
    assert preset.min_score >= 0.45


def test_query_depth_boost_increases_top_k_for_complex_question():
    simple = preset_for(WorkMode.GENERAL, StartupProfile.RECOMMENDED, query="요약해줘")
    complex_query = (
        "2025 프로젝트 문서들에서 설계 tradeoff와 성능 원인을 비교 분석해줘. "
        "왜 이런 결정을 했는지 근거까지 단계별로 설명해줘?"
    )
    complex_preset = preset_for(WorkMode.GENERAL, StartupProfile.RECOMMENDED, query=complex_query)
    assert complex_preset.top_k > simple.top_k


def test_query_hint_extraction_and_merge():
    hint = extract_query_hints("2025 회의록 #swift project: PLOS")
    assert hint.year == 2025
    assert hint.category == "회의록"
    assert "swift" in [tag.lower() for tag in hint.tags]

    merged = merge_filters(None, hint)
    assert merged is not None
    assert merged.category == "회의록"
