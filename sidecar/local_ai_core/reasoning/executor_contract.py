from __future__ import annotations

import asyncio
import inspect
from typing import Any


def bind_async_executor_contract(executor: Any) -> Any:
    """
    Ensure executor exposes async entrypoints used by strategies.
    This is a one-time compatibility bridge, not per-call fallback branching.
    """
    if executor is None:
        return executor
    if getattr(executor, "_plos_async_contract_bound", False) is True:
        return executor

    execute_sync = getattr(executor, "execute_conversation", None)
    execute_async = getattr(executor, "execute_conversation_async", None)
    if callable(execute_sync) and (not callable(execute_async) or not inspect.iscoroutinefunction(execute_async)):

        async def _execute_conversation_async(*, timeout_seconds: float | None = None, **kwargs):
            del timeout_seconds
            return await asyncio.to_thread(execute_sync, **kwargs)

        setattr(executor, "execute_conversation_async", _execute_conversation_async)

    step_sync = getattr(executor, "generate_agentic_step", None)
    if not callable(step_sync):
        local_inference = getattr(executor, "_local_inference", None)
        step_sync = getattr(local_inference, "generate_agentic_step", None)
    step_async = getattr(executor, "generate_agentic_step_async", None)
    if callable(step_sync) and (not callable(step_async) or not inspect.iscoroutinefunction(step_async)):

        async def _generate_agentic_step_async(*, timeout_seconds: float | None = None, **kwargs):
            del timeout_seconds
            return await asyncio.to_thread(step_sync, **kwargs)

        setattr(executor, "generate_agentic_step_async", _generate_agentic_step_async)

    setattr(executor, "_plos_async_contract_bound", True)
    return executor


def require_executor_methods(executor: Any, *method_names: str) -> None:
    missing = [name for name in method_names if not callable(getattr(executor, name, None))]
    if missing:
        missing_joined = ", ".join(missing)
        raise AttributeError(f"executor async contract missing methods: {missing_joined}")
