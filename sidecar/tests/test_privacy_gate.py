import pytest

from local_ai_core.chat import ChatService, PrivacyError
from local_ai_core.models import PrivacyMode


def test_local_only_blocks_external():
    with pytest.raises(PrivacyError):
        ChatService._enforce_privacy(PrivacyMode.LOCAL_ONLY, user_confirmed=True)


def test_confirm_mode_requires_user_confirmation():
    with pytest.raises(PrivacyError):
        ChatService._enforce_privacy(PrivacyMode.CONFIRM_BEFORE_EXTERNAL, user_confirmed=False)


def test_hybrid_allows_external_without_confirmation():
    ChatService._enforce_privacy(PrivacyMode.HYBRID, user_confirmed=False)
