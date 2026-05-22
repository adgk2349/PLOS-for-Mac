from __future__ import annotations

import logging
from typing import Any

from .base import ReasoningStrategy
from .general_chat_sections import (
    GeneralChatExecutionMixin,
    GeneralChatRecallMixin,
    GeneralChatWebMixin,
    GeneralChatConversationMixin,
    GeneralChatUtilityMixin,
)
from .general_chat_services import (
    ConversationInputBuilder,
    DefaultConversationInputBuilder,
    DefaultMemoryRecallRouter,
    DefaultWebSearchGate,
    MemoryRecallRouter,
    WebSearchGate,
)

logger = logging.getLogger(__name__)

# Global registry for service classes to support metaclass-based class/static method routing
_ALL_SERVICE_CLASSES: list[type] = []


class ServiceClassMeta(type):
    """
    Metaclass that routes class-level attribute accesses (such as classmethods or staticmethods)
    across all general chat service classes to support cross-service static/class references.
    """
    def __getattr__(cls, name: str) -> Any:
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
            
        for service_cls in _ALL_SERVICE_CLASSES:
            if service_cls is cls:
                continue
            for parent in service_cls.__mro__:
                if name in parent.__dict__:
                    return getattr(service_cls, name)
        raise AttributeError(f"class '{cls.__name__}' has no attribute '{name}'")


# Service wrappers that inherit individual mixins and delegate missing attributes to strategy

class GeneralChatExecutionService(GeneralChatExecutionMixin, metaclass=ServiceClassMeta):
    def __init__(self, strategy: GeneralChatStrategy) -> None:
        self.strategy = strategy

    def __getattr__(self, name: str) -> Any:
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        return getattr(self.strategy, name)


class GeneralChatRecallService(GeneralChatRecallMixin, metaclass=ServiceClassMeta):
    def __init__(self, strategy: GeneralChatStrategy) -> None:
        self.strategy = strategy

    def __getattr__(self, name: str) -> Any:
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        return getattr(self.strategy, name)


class GeneralChatWebService(GeneralChatWebMixin, metaclass=ServiceClassMeta):
    def __init__(self, strategy: GeneralChatStrategy) -> None:
        self.strategy = strategy

    def __getattr__(self, name: str) -> Any:
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        return getattr(self.strategy, name)


class GeneralChatConversationService(GeneralChatConversationMixin, metaclass=ServiceClassMeta):
    def __init__(self, strategy: GeneralChatStrategy) -> None:
        self.strategy = strategy

    def __getattr__(self, name: str) -> Any:
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        return getattr(self.strategy, name)


class GeneralChatUtilityService(GeneralChatUtilityMixin, metaclass=ServiceClassMeta):
    def __init__(self, strategy: GeneralChatStrategy) -> None:
        self.strategy = strategy

    def __getattr__(self, name: str) -> Any:
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)
        return getattr(self.strategy, name)


# Register all service classes for class-level lookup
_ALL_SERVICE_CLASSES.extend([
    GeneralChatExecutionService,
    GeneralChatRecallService,
    GeneralChatWebService,
    GeneralChatConversationService,
    GeneralChatUtilityService,
])


class GeneralChatStrategy(ReasoningStrategy):
    """Handles unstructured conversation, conversational memory, and optional fallback to web search."""

    def __init__(
        self,
        *,
        web_search_gate: WebSearchGate | None = None,
        memory_recall_router: MemoryRecallRouter | None = None,
        conversation_input_builder: ConversationInputBuilder | None = None,
    ) -> None:
        # Initialize services as composition attributes
        self.execution_service = GeneralChatExecutionService(self)
        self.recall_service = GeneralChatRecallService(self)
        self.web_service = GeneralChatWebService(self)
        self.conversation_service = GeneralChatConversationService(self)
        self.utility_service = GeneralChatUtilityService(self)

        self.web_search_gate = web_search_gate or DefaultWebSearchGate()
        self.memory_recall_router = memory_recall_router or DefaultMemoryRecallRouter(strategy=self)
        self.conversation_input_builder = conversation_input_builder or DefaultConversationInputBuilder(strategy=self)

    def handles_intent(self, intent: ParsedIntent, followup: FollowUpResolution | None) -> bool:
        return self.execution_service.handles_intent(intent, followup)

    async def execute(
        self,
        *,
        context: ReasoningContext,
        dependencies: dict[str, Any],
    ) -> ComposedChatResponseV2:
        return await self.execution_service.execute(context=context, dependencies=dependencies)

    def __getattr__(self, name: str) -> Any:
        if name.startswith('__') and name.endswith('__'):
            raise AttributeError(name)

        if self.__dict__.get('_in_getattr', False):
            raise AttributeError(name)

        self._in_getattr = True
        try:
            for service in [
                self.execution_service,
                self.recall_service,
                self.web_service,
                self.conversation_service,
                self.utility_service,
            ]:
                if any(name in cls.__dict__ for cls in service.__class__.__mro__):
                    return getattr(service, name)
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{name}'")
        finally:
            self._in_getattr = False

