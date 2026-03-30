from __future__ import annotations

import logging
from typing import Any

from ...models import ReasoningIntent, WorkMode
from ..strategies.general_chat import GeneralChatStrategy
from ..strategies.workspace_rag import WorkspaceRagStrategy

logger = logging.getLogger(__name__)


class StrategyRouter:
    def select(self, *, req, context, strategies: list[Any], force_general_chat: bool):
        if force_general_chat:
            return GeneralChatStrategy()
        if req.mode == WorkMode.DEVELOPMENT and context.parsed_intent.intent == ReasoningIntent.GENERAL_CHAT:
            return WorkspaceRagStrategy()
        for strategy in strategies:
            if strategy.handles_intent(context.parsed_intent, context.followup_resolution):
                return strategy
        logger.warning("[Orchestrator] No matching strategy found. Falling back to General Chat.")
        return GeneralChatStrategy()
