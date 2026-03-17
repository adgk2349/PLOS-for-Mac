from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, Request, status

from .auth import SessionAuth
from .chat import ChatService, PrivacyError
from .config import settings
from .db import Database
from .embedding import EmbeddingService
from .external_providers import ProviderRouter
from .indexing import IndexingService
from .local_inference import LocalInferenceEngine
from .models import (
    DeepAnalysisRequest,
    DeepAnalysisResponse,
    FailureItem,
    FailureListResponse,
    IndexJobRequest,
    IndexJobStatus,
    LocalChatRequest,
    LocalChatResponse,
    SettingsModel,
    WorkspaceResponse,
    WorkspaceUpdateRequest,
)
from .vector_store import VectorStore


class AppState:
    def __init__(self):
        self.db = Database(settings.sqlite_path)
        self.embedding = EmbeddingService(dim=settings.embedding_dim)
        self.vector_store = VectorStore(settings.lancedb_path, dim=settings.embedding_dim)
        self.indexing = IndexingService(self.db, self.vector_store, self.embedding)
        self.providers = ProviderRouter()
        self.local_inference = LocalInferenceEngine()
        self.chat = ChatService(
            self.db,
            self.vector_store,
            self.embedding,
            self.providers,
            self.local_inference,
        )
        self.auth = SessionAuth(settings.session_token)


app_state = AppState()


def _auth_dependency(request: Request) -> None:
    app_state.auth.verify_request(request)


@asynccontextmanager
async def lifespan(_: FastAPI):
    app_state.indexing.start_watcher()
    yield
    app_state.indexing.stop_watcher()


def create_app() -> FastAPI:
    app = FastAPI(title="Local AI Core Sidecar", version="0.1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/workspaces", response_model=WorkspaceResponse, dependencies=[Depends(_auth_dependency)])
    def update_workspace(payload: WorkspaceUpdateRequest) -> WorkspaceResponse:
        return app_state.db.update_workspace(payload)

    @app.post("/v1/index/jobs", dependencies=[Depends(_auth_dependency)])
    def start_index_job(payload: IndexJobRequest) -> dict[str, str]:
        workspace = app_state.db.get_workspace()
        job = app_state.indexing.start_job(payload.scope, workspace)
        return {"job_id": job.job_id}

    @app.get("/v1/index/jobs/{job_id}", response_model=IndexJobStatus, dependencies=[Depends(_auth_dependency)])
    def get_index_job(job_id: str) -> IndexJobStatus:
        job = app_state.indexing.get_job(job_id)
        if not job:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Job not found")
        return job

    @app.get("/v1/index/failures", response_model=FailureListResponse, dependencies=[Depends(_auth_dependency)])
    def get_failures() -> FailureListResponse:
        failures = [FailureItem(**item) for item in app_state.indexing.list_failures()]
        return FailureListResponse(failures=failures)

    @app.post("/v1/chat/local", response_model=LocalChatResponse, dependencies=[Depends(_auth_dependency)])
    def local_chat(payload: LocalChatRequest) -> LocalChatResponse:
        return app_state.chat.local_chat(payload)

    @app.post("/v1/chat/deep-analysis", response_model=DeepAnalysisResponse, dependencies=[Depends(_auth_dependency)])
    async def deep_analysis(payload: DeepAnalysisRequest) -> DeepAnalysisResponse:
        try:
            return await app_state.chat.deep_analysis(payload)
        except PrivacyError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.get("/v1/settings", response_model=SettingsModel, dependencies=[Depends(_auth_dependency)])
    def get_settings() -> SettingsModel:
        return app_state.db.get_settings()

    @app.put("/v1/settings", response_model=SettingsModel, dependencies=[Depends(_auth_dependency)])
    def update_settings(payload: SettingsModel) -> SettingsModel:
        return app_state.db.update_settings(payload)

    @app.get("/v1/status", dependencies=[Depends(_auth_dependency)])
    def get_status() -> dict:
        snapshot = app_state.db.get_status_snapshot()
        snapshot["privacy_mode"] = app_state.db.get_settings().privacy_mode
        return snapshot

    return app


app = create_app()
