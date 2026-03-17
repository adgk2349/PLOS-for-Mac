from __future__ import annotations

from datetime import datetime, timezone

from .db import Database
from .embedding import EmbeddingService
from .external_providers import ProviderRouter
from .local_inference import LocalInferenceEngine
from .models import (
    Citation,
    DeepAnalysisRequest,
    DeepAnalysisResponse,
    ExternalCallEvent,
    LocalChatRequest,
    LocalChatResponse,
    PrivacyMode,
    WorkMode,
)
from .retrieval import retrieve_hits
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

    def local_chat(self, req: LocalChatRequest) -> LocalChatResponse:
        workspace = self._db.get_workspace()
        query_vector = self._embedding.embed_query(req.query)
        preset, hits = retrieve_hits(
            self._vector_store,
            query_vector,
            mode=req.mode,
            startup_profile=workspace.startup_profile,
            explicit_top_k=req.top_k,
        )

        if req.mode == WorkMode.STRICT_SEARCH and not hits:
            return LocalChatResponse(
                answer="근거 부족: 현재 로컬 자료에서 신뢰할 수 있는 근거를 찾지 못했습니다.",
                citations=[],
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
            )
            for hit in hits
        ]

        answer = self._local_inference.generate(
            query=req.query,
            mode=req.mode,
            citations=citations,
            profile=workspace.startup_profile.value,
        )

        return LocalChatResponse(
            answer=answer,
            citations=citations,
            mode=req.mode,
            used_profile=workspace.startup_profile,
            is_local=True,
        )

    async def deep_analysis(self, req: DeepAnalysisRequest) -> DeepAnalysisResponse:
        settings = self._db.get_settings()
        self._enforce_privacy(settings.privacy_mode, req.user_confirmed)

        provider_result = await self._providers.analyze(
            provider=req.provider,
            query=req.query,
            mode=req.mode,
            citations=req.selected_citations,
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
