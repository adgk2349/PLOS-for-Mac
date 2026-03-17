from __future__ import annotations

from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status

from .auth import SessionAuth
from .chat import ChatService, PrivacyError
from .classification import DocumentClassifier
from .config import settings
from .db import Database
from .embedding import EmbeddingService
from .external_providers import ProviderRouter
from .indexing import IndexingService
from .local_inference import LocalInferenceEngine
from .model_manager import ModelManager
from .models import (
    ChatFilters,
    DeepAnalysisRequest,
    DeepAnalysisResponse,
    ModelCatalogActivateRequest,
    ModelCatalogActivateResponse,
    ModelCatalogDeleteResponse,
    ModelCatalogInstallRequest,
    ModelCatalogInstallResponse,
    ModelCatalogResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentMetadataUpdate,
    FailureItem,
    FailureListResponse,
    IndexJobRequest,
    IndexJobStatus,
    LocalEngine,
    LocalChatRequest,
    LocalChatResponse,
    ModelDownloadRequest,
    ModelDownloadResponse,
    ModelListResponse,
    RuntimePrepareRequest,
    RuntimePrepareResponse,
    SettingsModel,
    StartupProfile,
    WorkspaceResponse,
    WorkspaceUpdateRequest,
)
from .vector_store import VectorStore


def _profile_to_startup(profile: str):
    mapping = {
        "fast": StartupProfile.FAST,
        "balanced": StartupProfile.RECOMMENDED,
        "advanced": StartupProfile.DEEP,
    }
    return mapping.get(profile, StartupProfile.RECOMMENDED)


class AppState:
    def __init__(self):
        self.db = Database(settings.sqlite_path)
        self.embedding = EmbeddingService(dim=settings.embedding_dim)
        self.vector_store = VectorStore(settings.lancedb_path, dim=settings.embedding_dim)
        self.providers = ProviderRouter()
        self.local_inference = LocalInferenceEngine()
        self.classifier = DocumentClassifier(self.embedding, self.local_inference)
        self.indexing = IndexingService(self.db, self.vector_store, self.embedding, self.classifier)
        self.model_manager = ModelManager(settings.data_dir)
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
    app = FastAPI(title="Local AI Core Sidecar", version="0.2.0.1", lifespan=lifespan)

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

    @app.get("/v1/docs", response_model=DocumentListResponse, dependencies=[Depends(_auth_dependency)])
    def list_docs(
        search: str | None = None,
        category: str | None = None,
        tags: str | None = None,
        year: int | None = None,
        project: str | None = None,
        excluded: bool | None = False,
        limit: int = 100,
        offset: int = 0,
    ) -> DocumentListResponse:
        tag_list = [item.strip() for item in (tags or "").split(",") if item.strip()]
        effective_limit = max(1, min(limit, 300))
        effective_offset = max(0, offset)
        filters = ChatFilters(
            category=category,
            tags=tag_list,
            year=year,
            project=project,
            excluded=excluded,
        )
        docs, total = app_state.db.list_documents(
            search=search,
            filters=filters,
            limit=effective_limit,
            offset=effective_offset,
        )
        return DocumentListResponse(
            documents=docs,
            total=total,
            offset=effective_offset,
            limit=effective_limit,
        )

    @app.put("/v1/docs/{doc_id}/metadata", response_model=DocumentMetadata, dependencies=[Depends(_auth_dependency)])
    def update_doc_metadata(doc_id: str, payload: DocumentMetadataUpdate) -> DocumentMetadata:
        try:
            return app_state.db.update_document_metadata(doc_id, payload)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    @app.post("/v1/docs/{doc_id}/reclassify", response_model=DocumentMetadata, dependencies=[Depends(_auth_dependency)])
    def reclassify_doc(doc_id: str) -> DocumentMetadata:
        try:
            return app_state.indexing.reclassify_document(doc_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/v1/models/download", response_model=ModelDownloadResponse, dependencies=[Depends(_auth_dependency)])
    def download_model(payload: ModelDownloadRequest) -> ModelDownloadResponse:
        try:
            return app_state.model_manager.download_model(
                url=payload.url,
                engine=payload.engine,
                filename=payload.filename,
            )
        except httpx.HTTPError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=f"download failed: {exc}") from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.get("/v1/models", response_model=ModelListResponse, dependencies=[Depends(_auth_dependency)])
    def list_models() -> ModelListResponse:
        return ModelListResponse(models=app_state.model_manager.list_models())

    @app.get("/v1/models/catalog", response_model=ModelCatalogResponse, dependencies=[Depends(_auth_dependency)])
    def get_model_catalog() -> ModelCatalogResponse:
        settings_model = app_state.db.get_settings()
        return app_state.model_manager.catalog_with_status(settings_model)

    @app.post("/v1/models/catalog/install", response_model=ModelCatalogInstallResponse, dependencies=[Depends(_auth_dependency)])
    def install_catalog_model(payload: ModelCatalogInstallRequest) -> ModelCatalogInstallResponse:
        try:
            return app_state.model_manager.install_catalog_model(payload.model_id)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except Exception as exc:
            raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc

    @app.post("/v1/models/catalog/activate", response_model=ModelCatalogActivateResponse, dependencies=[Depends(_auth_dependency)])
    def activate_catalog_model(payload: ModelCatalogActivateRequest) -> ModelCatalogActivateResponse:
        try:
            activated = app_state.model_manager.activate_catalog_model(payload.model_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        settings_model = app_state.db.get_settings()
        settings_model.local_engine = activated.engine
        if activated.engine == LocalEngine.MLX:
            settings_model.mlx_model_path = activated.model_path
        else:
            settings_model.llama_model_path = activated.model_path
        settings_model.model_profile = activated.profile
        settings_model.startup_profile = _profile_to_startup(activated.profile)
        app_state.db.update_settings(settings_model)
        return activated

    @app.delete("/v1/models/catalog/{model_id}", response_model=ModelCatalogDeleteResponse, dependencies=[Depends(_auth_dependency)])
    def delete_catalog_model(model_id: str) -> ModelCatalogDeleteResponse:
        try:
            return app_state.model_manager.delete_catalog_model(model_id)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/v1/models/runtime/prepare", response_model=RuntimePrepareResponse, dependencies=[Depends(_auth_dependency)])
    def prepare_runtime(payload: RuntimePrepareRequest) -> RuntimePrepareResponse:
        settings_model = app_state.db.get_settings()
        return app_state.local_inference.prepare_runtime(
            engine=payload.engine,
            profile=settings_model.model_profile,
            mlx_model_path=payload.model_path if payload.engine == LocalEngine.MLX else settings_model.mlx_model_path,
            llama_model_path=payload.model_path if payload.engine == LocalEngine.LLAMA_CPP else settings_model.llama_model_path,
        )

    return app


app = create_app()
