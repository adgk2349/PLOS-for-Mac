import pytest

from local_ai_core.chat import ChatService, PrivacyError
from local_ai_core.models import PrivacyMode


def test_local_only_blocks_external():
    with pytest.raises(PrivacyError):
        ChatService._enforce_privacy(
            PrivacyMode.LOCAL_ONLY,
            user_confirmed=True,
            hybrid_web_search_enabled=True,
        )


def test_confirm_mode_requires_user_confirmation():
    with pytest.raises(PrivacyError):
        ChatService._enforce_privacy(
            PrivacyMode.CONFIRM_BEFORE_EXTERNAL,
            user_confirmed=False,
            hybrid_web_search_enabled=True,
        )


def test_hybrid_blocks_external_when_web_search_is_disabled():
    with pytest.raises(PrivacyError):
        ChatService._enforce_privacy(
            PrivacyMode.HYBRID,
            user_confirmed=False,
            hybrid_web_search_enabled=False,
        )


def test_hybrid_allows_external_when_web_search_is_enabled():
    ChatService._enforce_privacy(
        PrivacyMode.HYBRID,
        user_confirmed=False,
        hybrid_web_search_enabled=True,
    )
