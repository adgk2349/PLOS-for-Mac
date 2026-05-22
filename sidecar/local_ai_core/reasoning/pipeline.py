from __future__ import annotations
from typing import Any
import json
import asyncio
import os
from pathlib import Path
from datetime import datetime, timezone

import logging
import time
import re
import unicodedata
from collections import Counter

from ..models import (
    LocalChatRequestV2,
    ComposedChatResponseV2,
    ExecutionResult,
)
from ..nlu.intent_parser import IntentParser
from ..nlu.followup_resolver import FollowUpResolver
from .context import ReasoningContext, RelevantMemoryBundle
from .strategies.general_chat import GeneralChatStrategy
from .strategies.workspace_rag import WorkspaceRagStrategy
from .strategies.agentic_loop import AgenticLoopStrategy
from .executor_contract import bind_async_executor_contract
from .helpers.retrieval.retrieval_helpers import RetrievalHelpers
from .helpers.system.settings_sys_helpers import SettingsSysHelpers
from .orchestrator import ContextLoader, MemoryCommitter, PipelineCompatDelegates, StrategyRouter
from . import utils

logger = logging.getLogger(__name__)

class ReasoningPipeline(PipelineCompatDelegates):
    """
    The orchestrator that parses user intent, builds the unified ReasoningContext,
    and delegates execution to the appropriate isolated ReasoningStrategy.
    """
    def __init__(self, **dependencies):
        self.dependencies = dependencies
        self.intent_parser = IntentParser()
        self.followup_resolver = FollowUpResolver()
        self._executor = dependencies.get("executor")
        # Backward-compatible provider handle used by helpers/tests.
        self._providers = dependencies.get("provider_router") or dependencies.get("external_providers")
        
        # Registration of strategies in priority order
        self.strategies = [
            AgenticLoopStrategy(),
            WorkspaceRagStrategy(),
            GeneralChatStrategy(),
        ]
        
        # Initialize Helpers and add to dependencies for strategies to consume
        if "helpers" not in self.dependencies:
            self.dependencies["helpers"] = RetrievalHelpers(self.dependencies)
        if "sys_helpers" not in self.dependencies:
            self.dependencies["sys_helpers"] = SettingsSysHelpers(self.dependencies)
        self._context_loader = ContextLoader(
            intent_parser=self.intent_parser,
            followup_resolver=self.followup_resolver,
            digest_to_text=self._session_digest_to_summary_text,
        )
        self._strategy_router = StrategyRouter()
        self._memory_committer = MemoryCommitter()

    def _sync_orchestrator_dependencies(self) -> None:
        """
        Keep dynamically replaceable parser/resolver references in sync.
        This preserves test/runtime overrides like `pipeline.followup_resolver = ...`.
        """
        self._executor = bind_async_executor_contract(self.dependencies.get("executor"))
        if self._executor is not None:
            self.dependencies["executor"] = self._executor
        self.dependencies["intent_parser"] = self.intent_parser
        self.dependencies["followup_resolver"] = self.followup_resolver
        self._context_loader._intent_parser = self.intent_parser
        self._context_loader._followup_resolver = self.followup_resolver

    @staticmethod
    def _chat_log_path() -> Path:
        raw = str(os.getenv("LOCAL_AI_CHAT_LOG_PATH", "") or "").strip()
        if raw:
            return Path(raw).expanduser()
        return Path.cwd() / "tmp" / "chat_stream_live.jsonl"

    def _append_chat_log(self, payload: dict[str, Any]) -> None:
        if str(os.getenv("LOCAL_AI_CHAT_LOG_ENABLED", "1") or "1").strip().lower() in {"0", "false", "no", "off"}:
            return
        try:
            path = self._chat_log_path()
            path.parent.mkdir(parents=True, exist_ok=True)
            row = dict(payload)
            row["ts"] = datetime.now(timezone.utc).isoformat()
            with path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        except Exception:
            return

    def _repair_repetitive_conversation_response(
        self,
        *,
        query: str,
        execution: ExecutionResult,
        response_language: str,
        memory_bundle: RelevantMemoryBundle,
    ) -> ExecutionResult:
        """
        Detects if the assistant is stuck in a loop and attempts a one-time rewrite or fallback.
        """
        text = str(execution.generated_text or "").strip()
        if not text or len(text) < 50:
            return execution

        # Simple trigram-based repetition detection
        sentences = re.split(r"(?<=[.!?])\s+", text)
        if len(sentences) > 4:
            counts = Counter(sentences)
            if any(count > 2 for count in counts.values()):
                logger.warning("[Orchestrator] Repetitive response detected. Attempting repair.")
                # We could run a specialized repair inference here, or just truncate and add a note.
                unique_sentences = []
                for s in sentences:
                    if s not in unique_sentences:
                        unique_sentences.append(s)
                repaired_text = " ".join(unique_sentences)
                return execution.model_copy(update={"generated_text": repaired_text, "runtime_detail": (execution.runtime_detail or "") + "|repetition_repaired=True"})
        
        return execution

    @staticmethod
    def _looks_general_chat_query(query: str) -> bool:
        lowered = utils._normalized_match_text(query)
        if not lowered:
            return True
        local_doc_tokens = ("파일", "문서", "폴더", ".txt", ".pdf", ".md", ".docx", "f#", "p#")
        if any(token in lowered for token in local_doc_tokens):
            return False
        return True

    @staticmethod
    def _session_digest_to_summary_text(
        *,
        digest: dict[str, Any] | None,
        last_context: dict[str, Any] | None,
        max_chars: int = 900,
    ) -> str:
        payload = dict(digest or {})
        lines: list[str] = []

        topics = [str(item).strip() for item in (payload.get("active_topics") or []) if str(item).strip()]
        if topics:
            lines.append("topics: " + ", ".join(topics[:6]))

        facts = [str(item).strip() for item in (payload.get("stable_facts") or []) if str(item).strip()]
        if facts:
            lines.append("facts: " + " | ".join(facts[:4]))

        loops = [str(item).strip() for item in (payload.get("open_loops") or []) if str(item).strip()]
        if loops:
            lines.append("open_loops: " + " | ".join(loops[:3]))

        recent_user_turns: list[str] = []
        for row in payload.get("recent_turns") or []:
            if not isinstance(row, dict):
                continue
            role = str(row.get("role") or "").strip().lower()
            text = " ".join(str(row.get("text") or "").split()).strip()
            if not text:
                continue
            if role == "user":
                recent_user_turns.append(text[:140])
        if recent_user_turns:
            lines.append("recent_user: " + " / ".join(recent_user_turns[-3:]))

        if isinstance(last_context, dict):
            last_query = " ".join(str(last_context.get("last_user_query") or "").split()).strip()
            if last_query:
                lines.append("last_query: " + last_query[:160])
            # Avoid feeding previous assistant phrasing back into next-turn prompt.
            # This reduces style drift, echo loops, and multilingual contamination.

        summary = "\n".join(line for line in lines if line).strip()
        if not summary:
            return ""
        return summary[:max_chars].strip()

    async def _escalate_general_chat(
        self,
        *,
        context: ReasoningContext,
        force_web_search: bool = False,
    ) -> ComposedChatResponseV2:
        """
        Delegates the request to GeneralChatStrategy for handling.
        """
        strategy = GeneralChatStrategy()
        context.force_web_search = force_web_search
        return await strategy.execute(context=context, dependencies=self.dependencies)

    async def run(self, req: LocalChatRequestV2) -> ComposedChatResponseV2:
        start_time = time.time()
        logger.info(f"[Orchestrator] Beginning request routing for query: {req.query}")
        self._sync_orchestrator_dependencies()
        run_ctx = self._context_loader.load(req=req, dependencies=self.dependencies)
        context = run_ctx.context
        try:
            selected_strategy = self._strategy_router.select(
                req=req,
                context=context,
                strategies=self.strategies,
                force_general_chat=run_ctx.force_general_chat,
            )
            logger.info(f"[Orchestrator] Delegating to {selected_strategy.__class__.__name__}")
            composed = await selected_strategy.execute(
                context=context,
                dependencies=self.dependencies
            )

            if composed.execution_result:
                repaired_execution = self._repair_repetitive_conversation_response(
                    query=req.query,
                    execution=composed.execution_result,
                    response_language=run_ctx.response_language,
                    memory_bundle=run_ctx.memory_bundle
                )
                if repaired_execution is not None and repaired_execution is not composed.execution_result:
                    composed.execution_result = repaired_execution
                    composed.generated_text = repaired_execution.generated_text
            
            if composed:
                composed.metadata["web_auto_triggered"] = bool(run_ctx.web_auto_triggered)
            self._memory_committer.commit(
                memory=run_ctx.memory,
                composed=composed,
                req=req,
                context=context,
                session_id=run_ctx.session_id,
                session_digest_text=run_ctx.session_digest_text,
            )
            
            return composed
        except Exception as e:
            logger.exception(f"[Orchestrator] Strategy execution failed: {e}")
            raise

    async def run_stream(self, req: LocalChatRequestV2):
        self._sync_orchestrator_dependencies()
        live_token_streaming = str(os.getenv("LOCAL_AI_STREAM_LIVE_TOKENS", "1") or "1").strip().lower() in {
            "1", "true", "yes", "on"
        }
        conversation_id = str(getattr(req, "conversation_id", "") or getattr(req, "session_id", "") or "").strip()
        session_id = str(getattr(req, "session_id", "") or conversation_id).strip()
        query = str(getattr(req, "query", "") or "")
        self._append_chat_log(
            {
                "event": "request_start",
                "conversation_id": conversation_id,
                "session_id": session_id,
                "query": query,
                "mode": str(getattr(getattr(req, "mode", None), "value", getattr(req, "mode", ""))),
            }
        )
        inference_engine = self.dependencies.get("local_inference")
        token_queue: asyncio.Queue[str] = asyncio.Queue()
        loop = asyncio.get_running_loop()
        queue_wait_timeout = max(0.01, float(os.getenv("LOCAL_AI_STREAM_QUEUE_TIMEOUT_SEC", "0.05")))
        batch_char_limit = max(120, int(os.getenv("LOCAL_AI_STREAM_BATCH_CHARS", "420")))
        stream_prefix_cleaned = False

        def _on_token(piece: str) -> None:
            value = str(piece or "")
            if not value:
                return
            try:
                loop.call_soon_threadsafe(token_queue.put_nowait, value)
            except RuntimeError:
                return

        callback_token = None
        if inference_engine is not None and hasattr(inference_engine, "set_stream_token_callback"):
            try:
                callback_token = inference_engine.set_stream_token_callback(_on_token)
            except (AttributeError, RuntimeError, ValueError):
                callback_token = None

        run_task: asyncio.Task | None = None
        try:
            run_task = asyncio.create_task(self.run(req))
            streamed_any = False
            composed: ComposedChatResponseV2 | None = None
            last_progress_at = asyncio.get_running_loop().time()
            status_interval_sec = max(3.0, float(os.getenv("LOCAL_AI_STREAM_STATUS_INTERVAL_SEC", "8")))

            while True:
                if run_task.done():
                    try:
                        while True:
                            piece = self._drain_token_queue(token_queue, initial_piece=token_queue.get_nowait(), char_limit=batch_char_limit)
                            piece, stream_prefix_cleaned = self._sanitize_stream_piece(
                                piece,
                                prefix_cleaned=stream_prefix_cleaned,
                            )
                            if not piece:
                                continue
                            if live_token_streaming:
                                streamed_any = True
                                self._append_chat_log(
                                    {
                                        "event": "chunk",
                                        "conversation_id": conversation_id,
                                        "session_id": session_id,
                                        "text": piece,
                                    }
                                )
                                yield json.dumps(
                                    {
                                        "type": "chunk",
                                        "text": piece,
                                    },
                                    ensure_ascii=False,
                                ) + "\n"
                    except asyncio.QueueEmpty:
                        pass
                    composed = await run_task
                    break
                try:
                    initial_piece = await asyncio.wait_for(token_queue.get(), timeout=queue_wait_timeout)
                except asyncio.TimeoutError:
                    now = asyncio.get_running_loop().time()
                    status_enabled = str(os.getenv("LOCAL_AI_STREAM_STATUS_ENABLED", "1") or "1").strip().lower() in {
                        "1", "true", "yes", "on"
                    }
                    if not run_task.done() and status_enabled and (now - last_progress_at) >= status_interval_sec:
                        last_progress_at = now
                        status_message = "로컬 모델이 생성 중입니다..."
                        if str(getattr(req, "lang", "") or "").lower().startswith("en"):
                            status_message = "local model is still generating..."
                        yield json.dumps(
                            {
                                "type": "status",
                                "message": status_message,
                            },
                            ensure_ascii=False,
                        ) + "\n"
                        self._append_chat_log(
                            {
                                "event": "status",
                                "conversation_id": conversation_id,
                                "session_id": session_id,
                                "message": status_message,
                            }
                        )
                    continue
                piece = self._drain_token_queue(token_queue, initial_piece=initial_piece, char_limit=batch_char_limit)
                piece, stream_prefix_cleaned = self._sanitize_stream_piece(
                    piece,
                    prefix_cleaned=stream_prefix_cleaned,
                )
                if not piece:
                    continue
                last_progress_at = asyncio.get_running_loop().time()
                if live_token_streaming:
                    streamed_any = True
                    self._append_chat_log(
                        {
                            "event": "chunk",
                            "conversation_id": conversation_id,
                            "session_id": session_id,
                            "text": piece,
                        }
                    )
                    yield json.dumps(
                        {
                            "type": "chunk",
                            "text": piece,
                        },
                        ensure_ascii=False,
                    ) + "\n"

            if composed is None:
                composed = await run_task

            trace_events = []
            if isinstance(composed.metadata, dict):
                candidate_events = composed.metadata.get("trace_events")
                if isinstance(candidate_events, list):
                    trace_events = candidate_events

            for item in trace_events[:24]:
                if not isinstance(item, dict):
                    continue
                message = " ".join(str(item.get("message") or "").split()).strip()
                if not message:
                    continue
                yield json.dumps(
                    {
                        "type": "status",
                        "message": message,
                    },
                    ensure_ascii=False,
                ) + "\n"

            text = str(composed.generated_text or "").strip()
            if text and not streamed_any:
                for chunk_text in self._split_stream_chunks(text):
                    yield json.dumps(
                        {
                            "type": "chunk",
                            "text": chunk_text,
                        },
                        ensure_ascii=False,
                    ) + "\n"

            yield json.dumps(
                {
                    "type": "done",
                    "result": composed.model_dump(mode="json"),
                },
                ensure_ascii=False,
            ) + "\n"
            self._append_chat_log(
                {
                    "event": "done",
                    "conversation_id": conversation_id,
                    "session_id": session_id,
                    "generated_text": str(getattr(composed, "generated_text", "") or ""),
                    "runtime_detail": str(getattr(composed, "runtime_detail", "") or ""),
                }
            )
        except asyncio.CancelledError:
            if run_task is not None and not run_task.done():
                run_task.cancel()
                try:
                    await asyncio.wait_for(run_task, timeout=1.0)
                except BaseException:
                    pass
            raise
        except Exception as exc:
            logger.exception("[Orchestrator] Stream execution failed: %s", exc)
            yield json.dumps(
                {
                    "type": "error",
                    "message": str(exc) or "stream execution failed",
                },
                ensure_ascii=False,
            ) + "\n"
            self._append_chat_log(
                {
                    "event": "error",
                    "conversation_id": conversation_id,
                    "session_id": session_id,
                    "message": str(exc) or "stream execution failed",
                }
            )
        finally:
            if run_task is not None and not run_task.done():
                run_task.cancel()
                try:
                    await asyncio.wait_for(run_task, timeout=1.0)
                except BaseException:
                    pass
            if inference_engine is not None and callback_token is not None and hasattr(inference_engine, "reset_stream_token_callback"):
                try:
                    inference_engine.reset_stream_token_callback(callback_token)
                except (AttributeError, RuntimeError, ValueError):
                    pass

    @staticmethod
    def _drain_token_queue(token_queue: asyncio.Queue[str], *, initial_piece: str, char_limit: int) -> str:
        parts: list[str] = []
        first = str(initial_piece or "")
        if first:
            parts.append(first)
        used = len(first)
        while used < char_limit:
            try:
                nxt = token_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            text = str(nxt or "")
            if not text:
                continue
            parts.append(text)
            used += len(text)
        return "".join(parts)

    @staticmethod
    def _split_stream_chunks(text: str) -> list[str]:
        raw = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        if not raw.strip():
            return []
        sentence_parts = [part.strip() for part in re.split(r"(?<=[.!?。！？])\s+|\n+", raw) if part.strip()]
        if len(sentence_parts) > 1:
            return sentence_parts

        chunk_size = 220
        chunks: list[str] = []
        start = 0
        total = len(raw)
        while start < total:
            end = min(total, start + chunk_size)
            chunks.append(raw[start:end])
            start = end
        return chunks

    @staticmethod
    def _sanitize_stream_piece(piece: str, *, prefix_cleaned: bool) -> tuple[str, bool]:
        text = str(piece or "")
        if not text:
            return "", prefix_cleaned
        text = re.sub(r"(?im)^\s*continuation\s*:\s*", "", text)
        text = re.sub(r"(?im)^:?\s*please\s*provide\s*the\s*text\s*you\s*would\s*like\s*me\s*to\s*continue\.?\s*$", "", text)
        text = re.sub(r"(?i)pleaseprovidethetextyouwouldlikemetocontinue\.?", "", text)
        # Remove leaking control/header tokens everywhere in stream chunks.
        text = re.sub(r"(?is)<\|channel\|?>\s*(?:thought|analysis|final)?\s*", "", text)
        text = re.sub(r"(?is)<\|start_header_id\|>.*?<\|end_header_id\|>\s*", "", text)
        text = re.sub(r"(?is)<\|eot_id\|>", "", text)
        # Strip leading role/answer prefixes only once at stream start.
        cleaned_flag = prefix_cleaned
        if not cleaned_flag:
            text = re.sub(r"(?im)^\s*(?:answer|response|final answer|답변|최종 답변)\s*[:：]\s*", "", text)
            text = re.sub(r"(?im)^\s*(?:assistant|user|you|a|q)\s*[:：]?\s*", "", text)
            cleaned_flag = True
        stripped = text.strip().lower()
        if stripped in {
            "answer",
            "answer:",
            "response",
            "response:",
            "final",
            "final:",
            "final answer",
            "final answer:",
            "thought",
            "analysis",
            "assistant",
            "user",
            ":",
        }:
            return "", cleaned_flag
        if not text.strip():
            return "", cleaned_flag
        return text, cleaned_flag
 
