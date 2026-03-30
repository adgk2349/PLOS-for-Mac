from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from ...models import (
    ComposedChatResponseV2,
    ExecutionResult,
    LocalChatRequestV2,
    ParsedIntent,
    WorkspaceResponse,
)
from ...nlu.followup_resolver import FollowUpResolution


class ReasoningStrategy(ABC):
    """
    Base interface for isolated reasoning strategies based on the parsed intent.
    Strategies are injected into the ReasoningPipeline orchestrator.
    """
    
    @abstractmethod
    def handles_intent(self, intent: ParsedIntent, followup: FollowUpResolution | None) -> bool:
        """
        Returns true if this strategy is capable of handling the parsed intent.
        """
        pass

    @abstractmethod
    async def execute(
        self,
        *,
        request: LocalChatRequestV2,
        workspace: WorkspaceResponse,
        parsed_intent: ParsedIntent,
        followup: FollowUpResolution,
        context: dict[str, Any],
        dependencies: dict[str, Any],
    ) -> ComposedChatResponseV2:
        """
        Execute the strategy's core logic and return the composed response.
        """
        pass
