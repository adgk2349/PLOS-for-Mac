from __future__ import annotations

import asyncio
from datetime import datetime, timezone
import json
import os
from typing import Any

from .storage.db import Database
from .storage.async_adapter import AsyncAdapter
from .embedding import EmbeddingService
from .runtime.external_providers import ProviderRouter
from .language_utils import insufficient_evidence_message, resolve_response_language
from .runtime.local_inference import LocalInferenceEngine
from .memory_service import MemoryService
from .models import (
    ChatFilters,
    Citation,
    ComposedChatResponseV2,
    DeepAnalysisRequest,
    DeepAnalysisResponse,
    ExternalCallEvent,
    LocalChatRequest,
    LocalChatRequestV2,
    LocalChatResponse,
    MemoryEventRequest,
    MemoryEventType,
    PrivacyMode,
    WorkMode,
)
from .reasoning.pipeline import ReasoningPipeline
from .composition.composer import ResponseComposer
from .retrieval import extract_query_hints, merge_filters, retrieve_hits
from .storage.vector_store import VectorStore
from .indexing import IndexingService
from .infrastructure.docker_service import DockerService


class PrivacyError(PermissionError):
    pass


class RoomRoutingService:
    def __init__(self, room_registry: Any | None):
        self._room_registry = room_registry

    def resolve_target(self, *, req: LocalChatRequestV2, default_chat: "ChatService") -> tuple["ChatService", dict[str, Any] | None]:
        if self._room_registry is None:
            return default_chat, {"memory_backend": "global", "room_route_reason": "room_registry_missing"}
        room_id = str(req.conversation_id or req.session_id or "").strip()
        included_paths = list(req.included_paths or [])
        excluded_paths = list(req.excluded_paths or [])
        if not room_id:
            return default_chat, {"memory_backend": "global", "room_route_reason": "room_id_missing"}
        room_chat, room_meta = self._room_registry.resolve_chat_service_for_request(
            room_id=room_id,
            included_paths=included_paths,
            excluded_paths=excluded_paths,
        )
        if room_chat is None:
            fallback_meta = dict(room_meta or {})
            fallback_meta["memory_backend"] = "global"
            return default_chat, fallback_meta
        resolved_meta = dict(room_meta or {})
        resolved_meta["memory_backend"] = "room"
        return room_chat, resolved_meta


class ChatExecutionService:
    def __init__(self) -> None:
        self._route_timeout_seconds = float(os.getenv("LOCAL_AI_ROUTE_TIMEOUT_SECONDS", "240"))

    def execute(self, *, target_chat: "ChatService", req: LocalChatRequestV2):
        if self._route_timeout_seconds <= 0:
            return target_chat._pipeline.run(req)
        return asyncio.wait_for(
            target_chat._pipeline.run(req),
            timeout=max(1.0, self._route_timeout_seconds),
        )

    def execute_stream(self, *, target_chat: "ChatService", req: LocalChatRequestV2):
        return target_chat._pipeline.run_stream(req)


class ChatService:
    def __init__(
        self,
        db: Database,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        provider_router: ProviderRouter,
        local_inference: LocalInferenceEngine,
        memory_service: MemoryService | None = None,
        indexing_service: IndexingService | None = None,
        capability_router=None,
        docker_service: DockerService | None = None,
        room_registry: Any | None = None,
    ):
        self._db = db
        self._vector_store = vector_store
        self._embedding = embedding_service
        self._async_adapter = AsyncAdapter()
        self._providers = provider_router
        self._local_inference = local_inference
        self._room_registry = room_registry
        self._routing = RoomRoutingService(room_registry)
        self._execution = ChatExecutionService()
        self._memory = memory_service or MemoryService(db)
        try:
            self._memory.set_dependencies(vector_store=vector_store, embedding_service=embedding_service)
        except Exception:
            # Memory features must never block chat pipeline boot.
            pass
        self._composer = ResponseComposer()

        from .executor import LocalExecutor
        from .local_planner import LocalPlanner
        from .verifier import ResultVerifier
        from .nlu.clarification_budget import ClarificationBudget

        self._executor = LocalExecutor(local_inference, capability_router)
        self._planner = LocalPlanner()
        self._verifier = ResultVerifier()
        self._clarification_budget = ClarificationBudget()

        self._pipeline = ReasoningPipeline(
            db=db,
            vector_store=vector_store,
            embedding_service=embedding_service,
            local_inference=local_inference,
            composer=self._composer,
            memory=self._memory,
            memory_service=self._memory,
            provider_router=provider_router,
            indexing_service=indexing_service,
            capability_router=capability_router,
            local_planner=self._planner,
            verifier=self._verifier,
            executor=self._executor,
            clarification_budget=self._clarification_budget,
            docker_service=docker_service,
        )

    def _resolve_v2_chat_target(self, req: LocalChatRequestV2) -> tuple["ChatService", dict[str, Any] | None]:
        return self._routing.resolve_target(req=req, default_chat=self)

    def _apply_room_metadata(self, composed: ComposedChatResponseV2, room_meta: dict[str, Any] | None) -> None:
        if not room_meta:
            return
        if not isinstance(composed.metadata, dict):
            composed.metadata = {}
        metadata = composed.metadata
        metadata.update(room_meta)
        if self._room_registry is not None:
            storage_id = str(room_meta.get("room_storage_id") or "").strip()
            if storage_id:
                metadata["room_index_state"] = self._room_registry.room_index_state_by_storage_id(storage_id)
        detail_tokens: list[str] = []
        memory_backend = str(metadata.get("memory_backend") or "").strip()
        if memory_backend:
            detail_tokens.append(f"memory_backend={memory_backend}")
        room_storage_id = str(metadata.get("room_storage_id") or "").strip()
        if room_storage_id:
            detail_tokens.append(f"room_storage_id={room_storage_id}")
        room_scope_hash = str(metadata.get("room_scope_hash") or "").strip()
        if room_scope_hash:
            detail_tokens.append(f"room_scope_hash={room_scope_hash}")
        room_index_state = str(metadata.get("room_index_state") or "").strip()
        if room_index_state:
            detail_tokens.append(f"room_index_state={room_index_state}")
        if detail_tokens:
            detail_suffix = ";".join(detail_tokens)
            current_detail = str(composed.runtime_detail or "").strip()
            composed.runtime_detail = f"{current_detail};{detail_suffix}" if current_detail else detail_suffix
            if composed.execution_result is not None:
                execution_detail = str(composed.execution_result.runtime_detail or "").strip()
                composed.execution_result.runtime_detail = (
                    f"{execution_detail};{detail_suffix}" if execution_detail else detail_suffix
                )

    def local_chat(self, req: LocalChatRequest) -> LocalChatResponse:
        workspace = self._db.get_workspace()
        settings = self._db.get_settings()
        response_language = resolve_response_language(req.query, settings.language)
        query_vector = self._embedding.embed_query(req.query)

        hint_filters = extract_query_hints(req.query)
        merged_filters = merge_filters(req.filters, hint_filters) or ChatFilters()
        if merged_filters.excluded is None:
            merged_filters.excluded = False

        allowed_doc_ids = self._db.find_doc_ids_for_workspace(
            included_paths=workspace.included_paths,
            excluded_paths=workspace.excluded_paths,
            filters=merged_filters,
            search=None,
        )
        if req.filters is None and not allowed_doc_ids:
            merged_filters = ChatFilters(excluded=False)
            allowed_doc_ids = self._db.find_doc_ids_for_workspace(
                included_paths=workspace.included_paths,
                excluded_paths=workspace.excluded_paths,
                filters=merged_filters,
                search=None,
            )
        metadata_map = self._db.get_documents_metadata_map(list(allowed_doc_ids))
        preset, hits = retrieve_hits(
            self._vector_store,
            query_vector,
            mode=req.mode,
            startup_profile=workspace.startup_profile,
            explicit_top_k=req.top_k,
            query=req.query,
            allowed_doc_ids=allowed_doc_ids,
            filters=merged_filters,
            metadata_map=metadata_map,
        )

        if req.mode == WorkMode.STRICT_SEARCH and not hits:
            intent, lead, result_summary, actions = self._composer.compose(
                query=req.query,
                mode=req.mode,
                response_language=response_language,
                citations=[],
                result_summary=insufficient_evidence_message(response_language),
                insufficient=True,
            )
            return LocalChatResponse(
                intent=intent,
                lead=lead,
                result_summary=result_summary,
                citations=[],
                actions=actions,
                reasoning_brief=self._reasoning_brief(
                    mode=req.mode,
                    preset=preset,
                    citations=[],
                    filters=merged_filters,
                    response_language=response_language,
                    strict_insufficient=True,
                ),
                mode=req.mode,
                used_profile=workspace.startup_profile,
                is_local=True,
            )

        citations = [
            Citation(
                doc_id=hit.doc_id,
                chunk_id=hit.chunk_id,
                file_path=hit.file_path,
                snippet=(hit.text[:320] + "...") if len(hit.text) > 320 else hit.text,
                score=hit.score,
                modified_at=datetime.fromtimestamp(hit.modified_at, tz=timezone.utc),
                category=(metadata_map.get(hit.doc_id) or {}).get("category", "참고자료"),
                subcategory=(metadata_map.get(hit.doc_id) or {}).get("subcategory", ""),
                tags=(metadata_map.get(hit.doc_id) or {}).get("tags", []),
                document_type=(metadata_map.get(hit.doc_id) or {}).get("document_type", ""),
                importance=(metadata_map.get(hit.doc_id) or {}).get("importance", 0.5),
            )
            for hit in hits
        ]

        inference = self._local_inference.generate(
            query=req.query,
            mode=req.mode,
            citations=citations,
            profile=workspace.startup_profile.value,
            engine=settings.local_engine,
            mlx_model_path=settings.mlx_model_path,
            llama_model_path=settings.llama_model_path,
            language_preference=settings.language,
        )
        intent, lead, result_summary, actions = self._composer.compose(
            query=req.query,
            mode=req.mode,
            response_language=response_language,
            citations=citations,
            result_summary=inference.answer,
            insufficient=False,
        )

        return LocalChatResponse(
            intent=intent,
            lead=lead,
            result_summary=result_summary,
            citations=citations,
            actions=actions,
            reasoning_brief=self._reasoning_brief(
                mode=req.mode,
                preset=preset,
                citations=citations,
                filters=merged_filters,
                response_language=response_language,
                strict_insufficient=False,
            ),
            mode=req.mode,
            used_profile=workspace.startup_profile,
            is_local=True,
            engine_used=inference.engine_used,
            used_fallback=inference.used_fallback,
            runtime_detail=inference.detail,
        )

    async def local_chat_v2(self, req: LocalChatRequestV2) -> ComposedChatResponseV2:
        target_chat, room_meta = self._resolve_v2_chat_target(req)
        execution = getattr(self, "_execution", None)
        if execution is not None:
            composed = await execution.execute(target_chat=target_chat, req=req)
        else:
            composed = await (target_chat._pipeline.run(req))
        self._apply_room_metadata(composed, room_meta)
        return composed

    def local_chat_v2_stream(self, req: LocalChatRequestV2):
        target_chat, room_meta = self._resolve_v2_chat_target(req)
        execution = getattr(self, "_execution", None)
        if execution is not None:
            source = execution.execute_stream(target_chat=target_chat, req=req)
        else:
            source = target_chat._pipeline.run_stream(req)

        async def _wrapped():
            if room_meta and str(room_meta.get("room_index_state") or "") == "indexing":
                yield json.dumps(
                    {
                        "type": "status",
                        "message": "room indexing in progress; grounded retrieval will improve when indexing completes",
                    },
                    ensure_ascii=False,
                ) + "\n"
            route_reason = str((room_meta or {}).get("room_route_reason") or "").strip()
            has_room_scope = bool(str((room_meta or {}).get("room_storage_id") or "").strip())
            if route_reason and has_room_scope:
                yield json.dumps(
                    {
                        "type": "status",
                        "message": f"room routing: {route_reason}",
                    },
                    ensure_ascii=False,
                ) + "\n"
            async for line in source:
                if not room_meta:
                    yield line
                    continue
                try:
                    payload = json.loads(str(line or "").strip())
                except Exception:
                    yield line
                    continue
                if payload.get("type") == "done" and isinstance(payload.get("result"), dict):
                    result = payload["result"]
                    metadata = result.get("metadata")
                    if not isinstance(metadata, dict):
                        metadata = {}
                    metadata.update(room_meta)
                    if self._room_registry is not None:
                        storage_id = str(room_meta.get("room_storage_id") or "").strip()
                        if storage_id:
                            metadata["room_index_state"] = self._room_registry.room_index_state_by_storage_id(storage_id)
                    result["metadata"] = metadata
                    payload["result"] = result
                yield json.dumps(payload, ensure_ascii=False) + "\n"

        return _wrapped()

    async def deep_analysis(self, req: DeepAnalysisRequest) -> DeepAnalysisResponse:
        settings = await self._async_adapter.run(self._db.get_settings)
        self._enforce_privacy(
            settings.privacy_mode,
            req.user_confirmed,
            settings.hybrid_web_search_enabled,
        )

        provider_result = await self._providers.analyze(
            provider=req.provider,
            query=req.query,
            mode=req.mode,
            citations=req.selected_citations,
            language_preference=settings.language,
            allow_web_search=(
                settings.privacy_mode != PrivacyMode.LOCAL_ONLY
                and bool(settings.hybrid_web_search_enabled)
            ),
        )
        timestamp = await self._async_adapter.run(
            self._db.record_external_call,
            provider=req.provider,
            sent_chars=provider_result.sent_chars,
            approved_by_user=req.user_confirmed,
        )

        event = ExternalCallEvent(
            provider=req.provider,
            sent_chars=provider_result.sent_chars,
            approved_by_user=req.user_confirmed,
            timestamp=timestamp,
        )
        workspace_identity = await self._async_adapter.run(self._memory.get_workspace_identity)
        related_doc_ids = list(dict.fromkeys([item.doc_id for item in req.selected_citations if item.doc_id]))
        await self._async_adapter.run(
            self._memory.writeMemoryEvent,
            event=MemoryEventRequest(
                event_type=MemoryEventType.EXTERNAL_ANALYSIS,
                session_id=None,
                workspace_id=workspace_identity.workspace_id,
                summary=req.query[:220],
                related_file_ids=related_doc_ids[:8],
                metadata_json={
                    "provider": req.provider,
                    "mode": req.mode.value,
                    "approved_by_user": bool(req.user_confirmed),
                    "sent_chars": provider_result.sent_chars,
                    "event_timestamp": timestamp.isoformat(),
                },
                importance=0.7,
            ),
        )

        return DeepAnalysisResponse(
            answer=provider_result.answer,
            provider=req.provider,
            event=event,
            is_local=False,
        )

    @staticmethod
    def _enforce_privacy(mode: PrivacyMode, user_confirmed: bool, hybrid_web_search_enabled: bool = False) -> None:
        if mode == PrivacyMode.LOCAL_ONLY:
            raise PrivacyError("External calls are disabled in LOCAL_ONLY mode")
        if mode == PrivacyMode.HYBRID and not hybrid_web_search_enabled:
            raise PrivacyError("External calls are disabled while hybrid web search is turned off")
        if mode == PrivacyMode.CONFIRM_BEFORE_EXTERNAL and not user_confirmed:
            raise PrivacyError("User confirmation is required before external calls")

    @staticmethod
    def _reasoning_brief(
        *,
        mode: WorkMode,
        preset,
        citations: list[Citation],
        filters: ChatFilters | None,
        response_language: str,
        strict_insufficient: bool,
    ) -> str:
        top_score = f"{citations[0].score:.3f}" if citations else "-"
        filter_parts: list[str] = []
        if filters:
            if filters.category:
                filter_parts.append(filters.category)
            if filters.year is not None:
                filter_parts.append(str(filters.year))
            if filters.project:
                filter_parts.append(filters.project)
            if filters.tags:
                filter_parts.append(f"tags:{len(filters.tags)}")
        filter_text = ", ".join(filter_parts) if filter_parts else "-"

        if response_language == "ko":
            if strict_insufficient:
                return (
                    f"판단 로그: 모드={mode.value}, 컨텍스트 top_k={preset.top_k}, "
                    "엄격 검색 임계치 미달로 근거 부족 응답을 반환합니다."
                )
            return (
                f"판단 로그: 모드={mode.value}, 컨텍스트 top_k={preset.top_k}, "
                f"채택 근거={len(citations)}개, 최고 점수={top_score}, 필터={filter_text}."
            )

        if strict_insufficient:
            return (
                f"Reasoning log: mode={mode.value}, context top_k={preset.top_k}, "
                "strict threshold not met, returning insufficient-evidence response."
            )
        return (
            f"Reasoning log: mode={mode.value}, context top_k={preset.top_k}, "
            f"accepted citations={len(citations)}, top score={top_score}, filters={filter_text}."
        )
