from __future__ import annotations

from .general_chat_recall_memory_mixin import GeneralChatRecallMemoryMixin
from .general_chat_runtime_context_mixin import GeneralChatRuntimeContextMixin
from .general_chat_recall_query_mixin import GeneralChatRecallQueryMixin
from .general_chat_recall_two_pass_mixin import GeneralChatRecallTwoPassMixin
from .general_chat_runtime_context_deterministic_mixin import GeneralChatRuntimeContextDeterministicMixin


class GeneralChatRecallMixin(
    GeneralChatRecallMemoryMixin,
    GeneralChatRuntimeContextMixin,
    GeneralChatRecallQueryMixin,
    GeneralChatRecallTwoPassMixin,
    GeneralChatRuntimeContextDeterministicMixin,
):
    pass
