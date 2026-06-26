from __future__ import annotations

import logging
from typing import Any

from ...models import ReasoningIntent, WorkMode
from ..strategies.general_chat import GeneralChatStrategy
from ..strategies.workspace_rag import WorkspaceRagStrategy

logger = logging.getLogger(__name__)


class StrategyRouter:
    _WORKSPACE_RAG_INTENTS = {
        ReasoningIntent.FIND_FILE,
        ReasoningIntent.SUMMARIZE_FILE,
        ReasoningIntent.COMPARE_FILES,
        ReasoningIntent.EXPLAIN_CONTENT,
        ReasoningIntent.DRAFT_EDIT,
        ReasoningIntent.CLASSIFY,
        ReasoningIntent.OPEN_FILE,
        ReasoningIntent.FOLLOWUP_QUESTION,
        ReasoningIntent.FOLLOWUP_REFINE,
        ReasoningIntent.CONTINUE_PREVIOUS_RESULT,
        ReasoningIntent.SOFT_CONFIRM,
        ReasoningIntent.SELECT_PREVIOUS_CANDIDATE,
        ReasoningIntent.NEXT_CANDIDATE,
        ReasoningIntent.REDUCE_SCOPE,
        ReasoningIntent.LIGHTWEIGHT_ACTION_REQUEST,
    }

    def select(self, *, req, context, strategies: list[Any], force_general_chat: bool):
        if force_general_chat:
            return GeneralChatStrategy()
        # In GENERAL mode, prefer the conversation path only for non-explicit
        # conversational intents. Explicit retrieval/task intents should keep
        # their dedicated strategy instead of being swallowed by general chat.
        if (
            req.mode == WorkMode.GENERAL
            and context.parsed_intent.intent not in self._WORKSPACE_RAG_INTENTS
            and context.parsed_intent.intent != ReasoningIntent.SYSTEM_ACTION
        ):
            return GeneralChatStrategy()
        # Keep pure general chat on the conversation-first path except in DEVELOPMENT mode.
        # Routing general chat into WorkspaceRAG in GENERAL mode caused unnecessary fallback hops.
        if context.parsed_intent.intent == ReasoningIntent.GENERAL_CHAT:
            if req.mode == WorkMode.DEVELOPMENT:
                return WorkspaceRagStrategy()
            return GeneralChatStrategy()
        for strategy in strategies:
            if strategy.handles_intent(context.parsed_intent, context.followup_resolution):
                return strategy
        logger.warning("[Orchestrator] No matching strategy found. Falling back to General Chat.")
        return GeneralChatStrategy()
