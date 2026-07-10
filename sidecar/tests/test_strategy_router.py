from __future__ import annotations

from types import SimpleNamespace

from local_ai_core.models import ParsedEntities, ParsedIntent, ParsedTimeFilters, ParsedWorkspaceFilters, ReasoningIntent, WorkMode
from local_ai_core.reasoning.orchestrator.strategy_router import StrategyRouter
from local_ai_core.reasoning.strategies.general_chat import GeneralChatStrategy
from local_ai_core.reasoning.strategies.workspace_rag import WorkspaceRagStrategy


def _parsed_intent(intent: ReasoningIntent) -> ParsedIntent:
    return ParsedIntent(
        intent=intent,
        entities=ParsedEntities(),
        time_filters=ParsedTimeFilters(),
        workspace_filters=ParsedWorkspaceFilters(),
        confidence=0.9,
        operation="chat",
        target=None,
        scope="single",
        ambiguity="clear",
    )


def test_general_mode_keeps_explicit_summary_in_workspace_rag() -> None:
    router = StrategyRouter()
    req = SimpleNamespace(mode=WorkMode.GENERAL)
    context = SimpleNamespace(
        parsed_intent=_parsed_intent(ReasoningIntent.SUMMARIZE_FILE),
        followup_resolution=None,
    )

    selected = router.select(
        req=req,
        context=context,
        strategies=[WorkspaceRagStrategy(), GeneralChatStrategy()],
        force_general_chat=False,
    )

    assert isinstance(selected, WorkspaceRagStrategy)


def test_general_mode_keeps_general_chat_for_conversational_intent() -> None:
    router = StrategyRouter()
    req = SimpleNamespace(mode=WorkMode.GENERAL)
    context = SimpleNamespace(
        parsed_intent=_parsed_intent(ReasoningIntent.GENERAL_CHAT),
        followup_resolution=None,
    )

    selected = router.select(
        req=req,
        context=context,
        strategies=[WorkspaceRagStrategy(), GeneralChatStrategy()],
        force_general_chat=False,
    )

    assert isinstance(selected, GeneralChatStrategy)
