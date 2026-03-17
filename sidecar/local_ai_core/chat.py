from __future__ import annotations

from datetime import datetime, timezone

from .db import Database
from .embedding import EmbeddingService
from .external_providers import ProviderRouter
from .language_utils import insufficient_evidence_message, resolve_response_language
from .local_inference import LocalInferenceEngine
from .models import (
    ChatFilters,
    Citation,
    DeepAnalysisRequest,
    DeepAnalysisResponse,
    ExternalCallEvent,
    LocalChatRequest,
    LocalChatResponse,
    PrivacyMode,
    WorkMode,
)
from .response_composer import ResponseComposer
from .retrieval import extract_query_hints, merge_filters, retrieve_hits
from .vector_store import VectorStore


class PrivacyError(PermissionError):
    pass


class ChatService:
    def __init__(
        self,
        db: Database,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        provider_router: ProviderRouter,
        local_inference: LocalInferenceEngine,
    ):
        self._db = db
        self._vector_store = vector_store
        self._embedding = embedding_service
        self._providers = provider_router
        self._local_inference = local_inference
        self._composer = ResponseComposer()

    def local_chat(self, req: LocalChatRequest) -> LocalChatResponse:
        workspace = self._db.get_workspace()
        settings = self._db.get_settings()
        response_language = resolve_response_language(req.query, settings.language)
        query_vector = self._embedding.embed_query(req.query)

        hint_filters = extract_query_hints(req.query)
        merged_filters = merge_filters(req.filters, hint_filters) or ChatFilters()
        if merged_filters.excluded is None:
            merged_filters.excluded = False

        allowed_doc_ids = self._db.find_doc_ids(filters=merged_filters, search=None)
        if req.filters is None and not allowed_doc_ids:
            merged_filters = ChatFilters(excluded=False)
            allowed_doc_ids = self._db.find_doc_ids(filters=merged_filters, search=None)
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

    async def deep_analysis(self, req: DeepAnalysisRequest) -> DeepAnalysisResponse:
        settings = self._db.get_settings()
        self._enforce_privacy(settings.privacy_mode, req.user_confirmed)

        provider_result = await self._providers.analyze(
            provider=req.provider,
            query=req.query,
            mode=req.mode,
            citations=req.selected_citations,
            language_preference=settings.language,
        )
        timestamp = self._db.record_external_call(
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

        return DeepAnalysisResponse(
            answer=provider_result.answer,
            provider=req.provider,
            event=event,
            is_local=False,
        )

    @staticmethod
    def _enforce_privacy(mode: PrivacyMode, user_confirmed: bool) -> None:
        if mode == PrivacyMode.LOCAL_ONLY:
            raise PrivacyError("External calls are disabled in LOCAL_ONLY mode")
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
