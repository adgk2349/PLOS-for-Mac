import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import MagicMock
from local_ai_core.models import (
    AgentAction,
    LocalChatRequestV2,
    ParsedIntent,
    ReasoningIntent,
    SettingsModel,
    StartupProfile,
    WorkspaceIdentity,
    WorkspaceResponse,
)
from local_ai_core.nlu.followup_resolver import FollowUpResolution
from local_ai_core.reasoning.context import ReasoningContext
from local_ai_core.reasoning.strategies.agentic_loop import AgenticLoopStrategy
from local_ai_core.reasoning.pipeline import ReasoningPipeline

def test_agentic_loop_logic():
    # Setup mocks
    mock_db = MagicMock()
    mock_memory = MagicMock()
    mock_intent = MagicMock()
    mock_executor = MagicMock()
    mock_composer = MagicMock()
    now = datetime.now(timezone.utc)
    mock_db.get_workspace.return_value = WorkspaceResponse(
        included_paths=[],
        excluded_paths=[],
        startup_profile=StartupProfile.RECOMMENDED,
        default_mode="GENERAL",
        updated_at=now,
    )
    mock_db.get_settings.return_value = SettingsModel()
    mock_memory.get_workspace_identity.return_value = WorkspaceIdentity(
        workspace_id="ws-test",
        included_paths_hash="hash",
        version=1,
    )
    
    # Mock executor internal inference
    mock_inference = MagicMock()
    mock_executor._local_inference = mock_inference
    mock_executor.generate_agentic_step = mock_inference.generate_agentic_step
    
    pipeline = ReasoningPipeline(
        db=mock_db,
        vector_store=MagicMock(),
        embedding=MagicMock(),
        providers=MagicMock(),
        local_inference=mock_inference,
        memory=mock_memory,
        indexing=MagicMock()
    )
    # Patch pipeline dependencies used by orchestrator
    pipeline.intent_parser = mock_intent
    pipeline.dependencies["executor"] = mock_executor
    pipeline.dependencies["composer"] = mock_composer
    
    # Mock a system action request
    req = LocalChatRequestV2(query="바탕화면에서 레시피 파일 찾아줘", session_id="test-session")
    mock_intent.parse.return_value = ParsedIntent(intent=ReasoningIntent.SYSTEM_ACTION)
    
    # Mock the iterative steps
    # Step 1: Spotlight search
    action1 = AgentAction(kind="spotlight_search", params={"query": "레시피"}, thought="레시피 파일을 찾아야겠어.")
    # Step 2: Final answer
    action2 = AgentAction(kind="final_answer", params={"answer": "바탕화면에 '김치찌개_레시피.txt'가 있습니다."}, thought="파일을 찾았으니 답변하자.")
    
    mock_inference.generate_agentic_step.side_effect = [action1, action2]
    mock_executor.execute_agent_action.return_value = "/Users/user/Desktop/김치찌개_레시피.txt"
    mock_composer.compose_v2.return_value = MagicMock()

    # Run loop
    result = asyncio.run(pipeline.run(req))
    
    # Assertions
    assert mock_executor.execute_agent_action.called
    assert mock_inference.generate_agentic_step.call_count == 2
    print("Agentic loop verification passed.")

if __name__ == "__main__":
    asyncio.run(test_agentic_loop_logic())


def test_agentic_loop_failure_response_does_not_expose_thought():
    class _StubInference:
        def __init__(self):
            self.calls = 0

        def generate_agentic_step(self, **_kwargs):
            self.calls += 1
            return AgentAction(
                kind="spotlight_search",
                params={"query": "internal"},
                thought="내부 추론: 사용자의 의도를 단계별로 분석 중",
            )

    class _StubExecutor:
        def __init__(self):
            self._local_inference = _StubInference()

        def execute_agent_action(self, _action, _permission_level):
            return "no-op"

    class _CaptureComposer:
        def __init__(self):
            self.last_kwargs = None

        def compose_v2(self, **kwargs):
            self.last_kwargs = kwargs
            return kwargs

    now = datetime.now(timezone.utc)
    context = ReasoningContext(
        req=LocalChatRequestV2(query="내 컴퓨터 전체 검색해줘", mode="GENERAL"),
        workspace=WorkspaceResponse(
            included_paths=[],
            excluded_paths=[],
            startup_profile=StartupProfile.RECOMMENDED,
            default_mode="GENERAL",
            updated_at=now,
        ),
        workspace_identity=WorkspaceIdentity(workspace_id="w", included_paths_hash="h", version=1),
        settings=SettingsModel(),
        session_id="s",
        response_language="ko",
        parsed_intent=ParsedIntent(intent=ReasoningIntent.SYSTEM_ACTION),
        followup_resolution=FollowUpResolution(),
        memory_bundle=SimpleNamespace(),
        behavior_policy={},
        memory_prefs=None,
        last_context={},
        session_digest=None,
        effective_query="내 컴퓨터 전체 검색해줘",
        force_web_search=False,
    )

    strategy = AgenticLoopStrategy()
    executor = _StubExecutor()
    composer = _CaptureComposer()
    asyncio.run(strategy.execute(context=context, dependencies={"executor": executor, "composer": composer}))

    execution = composer.last_kwargs["execution_result"]
    assert "내부 추론" not in execution.generated_text
    assert "생각 중이었어요" not in execution.generated_text
    assert "thought:redacted" in execution.tool_logs
    assert all("내부 추론" not in item for item in execution.tool_logs)
