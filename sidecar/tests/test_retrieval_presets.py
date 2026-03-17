from local_ai_core.models import StartupProfile, WorkMode
from local_ai_core.retrieval import preset_for


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
