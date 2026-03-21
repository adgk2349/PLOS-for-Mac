from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import os

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
from .memory_service import MemoryService
from .model_manager import ModelManager
from .models import (
    ChatFilters,
    ComposedChatResponseV2,
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
    MemoryClearRequest,
    MemoryClearResponse,
    MemoryEventRequest,
    MemoryEventResponse,
    MemoryPinRequest,
    MemoryPinResponse,
    LocalChatRequest,
    LocalChatResponse,
    LocalChatRequestV2,
    ModelDownloadRequest,
    ModelDownloadResponse,
    ModelListResponse,
    RuntimePrepareRequest,
    RuntimePrepareResponse,
    SessionMemoryResponse,
    SettingsModel,
    StartupProfile,
    UserPreferencesResponse,
    WorkspaceMemoryResponse,
    EpisodicMemoryResponse,
    PinnedMemoryResponse,
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
        self.memory = MemoryService(self.db)
        self.model_manager = ModelManager(settings.data_dir)
        self.chat = ChatService(
            self.db,
            self.vector_store,
            self.embedding,
            self.providers,
            self.local_inference,
            self.memory,
            self.indexing,
        )
        self.auth = SessionAuth(settings.session_token)


app_state = AppState()


def _auth_dependency(request: Request) -> None:
    app_state.auth.verify_request(request)


def _expected_parent_pid() -> int | None:
    raw = (os.getenv("LOCAL_AI_PARENT_PID") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    if value <= 1:
        return None
    return value


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return False
    return True


async def _watch_parent_lifecycle(parent_pid: int) -> None:
    while True:
        await asyncio.sleep(1.5)
        if os.getppid() == parent_pid:
            continue
        if _pid_alive(parent_pid):
            continue
        os._exit(0)


@asynccontextmanager
async def lifespan(_: FastAPI):
    parent_monitor_task: asyncio.Task | None = None
    expected_parent = _expected_parent_pid()
    if expected_parent is not None:
        parent_monitor_task = asyncio.create_task(_watch_parent_lifecycle(expected_parent))
    app_state.indexing.start_watcher()
    try:
        yield
    finally:
        if parent_monitor_task is not None:
            parent_monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await parent_monitor_task
        app_state.indexing.stop_watcher()




def _safe_provider_error_message(exc: httpx.HTTPStatusError) -> str:
    status_code = exc.response.status_code if exc.response is not None else None
    provider_detail = ""
    if exc.response is not None:
        try:
            payload = exc.response.json()
            if isinstance(payload, dict):
                raw_error = payload.get("error")
                if isinstance(raw_error, dict):
                    provider_detail = str(raw_error.get("message") or "").strip()
                elif isinstance(raw_error, str):
                    provider_detail = raw_error.strip()
                elif payload.get("detail"):
                    provider_detail = str(payload.get("detail")).strip()
        except Exception:
            provider_detail = ""

    if status_code == 429:
        if provider_detail:
            return f"외부 제공자 호출 한도를 초과했습니다(429). {provider_detail}"
        return "외부 제공자 호출 한도를 초과했습니다(429). 잠시 후 다시 시도하거나 다른 제공자를 선택해 주세요."
    if status_code in {401, 403}:
        if provider_detail:
            return f"외부 제공자 인증에 실패했습니다({status_code}). {provider_detail}"
        return f"외부 제공자 인증에 실패했습니다({status_code}). API 키를 확인해 주세요."

    if provider_detail:
        return f"외부 제공자 호출 실패({status_code or 'unknown'}): {provider_detail}"
    return f"외부 제공자 호출 실패({status_code or 'unknown'})"


def create_app() -> FastAPI:
    app = FastAPI(title="Local AI Core Sidecar", version="0.3.0", lifespan=lifespan)

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

    @app.post("/v2/chat/local", response_model=ComposedChatResponseV2, dependencies=[Depends(_auth_dependency)])
    def local_chat_v2(payload: LocalChatRequestV2) -> ComposedChatResponseV2:
        return app_state.chat.local_chat_v2(payload)

    @app.post("/v1/chat/deep-analysis", response_model=DeepAnalysisResponse, dependencies=[Depends(_auth_dependency)])
    async def deep_analysis(payload: DeepAnalysisRequest) -> DeepAnalysisResponse:
        try:
            return await app_state.chat.deep_analysis(payload)
        except PrivacyError as exc:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except httpx.HTTPStatusError as exc:
            status_code = exc.response.status_code if exc.response is not None else status.HTTP_502_BAD_GATEWAY
            if status_code < 400 or status_code >= 600:
                status_code = status.HTTP_502_BAD_GATEWAY
            raise HTTPException(status_code=status_code, detail=_safe_provider_error_message(exc)) from exc
        except httpx.HTTPError as exc:
            raise HTTPException(
                status_code=status.HTTP_502_BAD_GATEWAY,
                detail=f"외부 제공자 네트워크 오류: {exc}",
            ) from exc

    @app.get("/v1/settings", response_model=SettingsModel, dependencies=[Depends(_auth_dependency)])
    def get_settings() -> SettingsModel:
        return app_state.db.get_settings()

    @app.put("/v1/settings", response_model=SettingsModel, dependencies=[Depends(_auth_dependency)])
    def update_settings(payload: SettingsModel) -> SettingsModel:
        return app_state.db.update_settings(payload)

    @app.get("/v1/memory/session/relevant", response_model=SessionMemoryResponse, dependencies=[Depends(_auth_dependency)])
    def get_relevant_session_memory(session_id: str) -> SessionMemoryResponse:
        items = app_state.memory.get_relevant_session_memory(session_id)
        return SessionMemoryResponse(items=items)

    @app.get("/v1/memory/workspace/relevant", response_model=WorkspaceMemoryResponse, dependencies=[Depends(_auth_dependency)])
    def get_relevant_workspace_memory(workspace_id: str, intent: str | None = None) -> WorkspaceMemoryResponse:
        items = app_state.memory.get_relevant_workspace_memory(workspace_id, intent)
        return WorkspaceMemoryResponse(items=items)

    @app.get("/v1/memory/preferences", response_model=UserPreferencesResponse, dependencies=[Depends(_auth_dependency)])
    def get_user_preferences() -> UserPreferencesResponse:
        items = app_state.memory.get_user_preferences()
        return UserPreferencesResponse(items=items)

    @app.get("/v1/memory/episodic/relevant", response_model=EpisodicMemoryResponse, dependencies=[Depends(_auth_dependency)])
    def get_relevant_episodic_memory(
        workspace_id: str | None = None,
        intent: str | None = None,
        related_file_ids: str | None = None,
    ) -> EpisodicMemoryResponse:
        related = [item.strip() for item in (related_file_ids or "").split(",") if item.strip()]
        items = app_state.memory.get_relevant_episodic_memory(workspace_id, intent, related)
        return EpisodicMemoryResponse(items=items)

    @app.post("/v1/memory/events", response_model=MemoryEventResponse, dependencies=[Depends(_auth_dependency)])
    def write_memory_event(payload: MemoryEventRequest) -> MemoryEventResponse:
        return app_state.memory.writeMemoryEvent(payload)

    @app.post("/v1/memory/clear", response_model=MemoryClearResponse, dependencies=[Depends(_auth_dependency)])
    def clear_memory(payload: MemoryClearRequest) -> MemoryClearResponse:
        return app_state.memory.clearMemory(
            scope=payload.scope,
            workspace_id=payload.workspace_id,
            session_id=payload.session_id,
        )

    @app.post("/v1/memory/pin", response_model=MemoryPinResponse, dependencies=[Depends(_auth_dependency)])
    def pin_memory(payload: MemoryPinRequest) -> MemoryPinResponse:
        item = app_state.memory.pinMemory(
            memory_id=payload.memory_id,
            scope=payload.scope,
            workspace_id=payload.workspace_id,
            title=payload.title,
            content=payload.content,
        )
        return MemoryPinResponse(item=item)

    @app.delete("/v1/memory/pin/{memory_id}", response_model=dict[str, bool], dependencies=[Depends(_auth_dependency)])
    def unpin_memory(memory_id: str) -> dict[str, bool]:
        return {"removed": app_state.memory.unpinMemory(memory_id)}

    @app.get("/v1/memory/pins", response_model=PinnedMemoryResponse, dependencies=[Depends(_auth_dependency)])
    def list_pins(scope: str | None = None, workspace_id: str | None = None) -> PinnedMemoryResponse:
        items = app_state.memory.listPinnedMemory(scope=scope, workspace_id=workspace_id)
        return PinnedMemoryResponse(items=items)

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
        filters = ChatFilters(
            category=category,
            tags=tag_list,
            year=year,
            project=project,
            excluded=excluded,
        )
        workspace = app_state.db.get_workspace()
        allowed_doc_ids = app_state.db.find_doc_ids_for_workspace(
            included_paths=workspace.included_paths,
            excluded_paths=workspace.excluded_paths,
            filters=filters,
            search=search,
        )
        docs, total = app_state.db.list_documents(
            search=search,
            filters=filters,
            allowed_doc_ids=allowed_doc_ids,
            limit=max(1, min(limit, 300)),
            offset=max(0, offset),
        )
        return DocumentListResponse(documents=docs, total=total, offset=offset, limit=limit)

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
