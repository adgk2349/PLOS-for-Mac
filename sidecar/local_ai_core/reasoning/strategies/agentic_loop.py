import re
from typing import Any

from .base import ReasoningStrategy
from ..context import ReasoningContext
from ...models import (
    ComposedChatResponseV2,
    ExecutionResult,
    LocalPlan,
    ParsedIntent,
    ReasoningIntent,
    VerificationResult,
)
from ...nlu.followup_resolver import FollowUpResolution
from ..executor_contract import bind_async_executor_contract, require_executor_methods


class AgenticLoopStrategy(ReasoningStrategy):
    """
    Handles autonomous, multi-step Reasoning-Action (ReAct) loops for complex tasks.
    """

    def handles_intent(self, intent: ParsedIntent, followup: FollowUpResolution | None) -> bool:
        return intent.intent == ReasoningIntent.SYSTEM_ACTION

    async def execute(
        self,
        *,
        context: ReasoningContext,
        dependencies: dict[str, Any],
    ) -> ComposedChatResponseV2:
        executor = dependencies["executor"]
        composer = dependencies["composer"]

        history: list[dict[str, str]] = []
        if context.session_digest:
            history.append({"role": "system", "content": f"Previous conversation summary: {context.session_digest}"})
            
        current_step = 0
        max_steps = 5
        final_answer = ""
        tool_logs: list[str] = ["agent:mac_system_loop_started"]
        
        while current_step < max_steps:
            current_step += 1
            executor = bind_async_executor_contract(executor)
            require_executor_methods(executor, "generate_agentic_step_async")
            action = await executor.generate_agentic_step_async(
                engine=context.settings.local_engine,
                query=context.req.query,
                history=history,
                profile=context.workspace.startup_profile,
                mlx_model_path=context.settings.mlx_model_path,
                llama_model_path=context.settings.llama_model_path,
                max_tokens=512,
                response_language=context.response_language,
            )
            
            if action.thought:
                history.append({"role": "assistant", "content": f"<thought>{action.thought}</thought>"})
                tool_logs.append("thought:redacted")

            if action.kind == "final_answer":
                final_answer = action.params.get("answer", "No answer provided.")
                break
            
            tool_logs.append(f"action:{action.kind}")
            permission_level = context.settings.system_file_permission
            observation = executor.execute_agent_action(action, permission_level)
            
            history.append({"role": "observation", "content": f"<observation>{observation}</observation>"})
            tool_logs.append(f"observation:{len(observation)} chars")

        if not final_answer.strip():
            final_answer = (
                "연속적인 실행 단계는 완료했지만 최종 답변을 확정하지 못했습니다. 요청 범위를 한 줄로 더 구체화해 주세요."
                if context.response_language == "ko"
                else "I completed the multi-step run but could not finalize an answer. Please narrow the request in one line."
            )

        execution = ExecutionResult(
            result_type="conversation",
            structured_payload={
                "style": "agentic_chat",
                "loop_steps": current_step,
                "history": history,
            },
            citations=[],
            tool_logs=tool_logs,
            generated_text=final_answer,
            engine_used=context.settings.local_engine,
            used_fallback=False,
        )
        
        verification = VerificationResult(is_valid=True, confidence=0.92, issues=[], ambiguity_level=0.1, candidate_mode=False)
        plan = LocalPlan(plan_type="agentic_loop")
        
        return composer.compose_v2(
            query=context.req.query,
            mode=context.req.mode,
            response_language=context.response_language,
            parsed_intent=context.parsed_intent,
            plan=plan,
            execution_result=execution,
            verification=verification,
            behavior_policy=context.behavior_policy,
            response_length=getattr(context.memory_prefs, "response_length", "long") if context.memory_prefs else "long",
            show_citations=False,
            prefer_action_suggestions=True,
            used_profile=context.workspace.startup_profile,
            engine_used=execution.engine_used,
            used_fallback=execution.used_fallback,
            runtime_detail=execution.runtime_detail,
            followup_resolution=None,
            allow_clarification=None,
            conversation_path="system_agent_loop",
            is_local=True,
            prompt_cache_hit=False,
        )
