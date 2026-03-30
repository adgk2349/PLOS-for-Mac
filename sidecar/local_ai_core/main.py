from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager, suppress
import logging
import os
from typing import Any

import httpx
from fastapi import Depends, FastAPI, HTTPException, Request, status
from fastapi.responses import JSONResponse

from .chat import PrivacyError
from .plugins import ExtensionKernel, FinetuneJobService, PluginRegistryService
from .models import (
    ChatFilters,
    ComposedChatResponseV2,
    DeepAnalysisRequest,
    DeepAnalysisResponse,
    DownloadProgressResponse,
    ModelCatalogActivateRequest,
    ModelCatalogActivateResponse,
    ModelCatalogDeleteResponse,
    ModelCatalogInstallRequest,
    ModelCatalogInstallResponse,
    ModelCatalogResponse,
    DocumentListResponse,
    DocumentMetadata,
    DocumentMetadataUpdate,
    ExtensionCapabilitiesResponse,
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
    PluginEnableResponse,
    PluginRegisterRequest,
    PluginRegistryResponse,
    RuntimePrepareRequest,
    RuntimePrepareResponse,
    SessionMemoryResponse,
    SettingsModel,
    StartupProfile,
    UserPreferencesResponse,
    WorkspaceMemoryResponse,
    EpisodicMemoryResponse,
    FinetuneJobStatusResponse,
    FinetuneJobSubmitRequest,
    FinetuneJobSubmitResponse,
    FinetuneModelPublishRequest,
    FinetuneModelPublishResponse,
    PinnedMemoryResponse,
    WorkspaceResponse,
    WorkspaceUpdateRequest,
)
from .platform_services import PlatformServices
from .config import settings
from .container import ServiceContainer

logger = logging.getLogger(__name__)


def _profile_to_startup(profile: str):
    mapping = {
        "fast": StartupProfile.FAST,
        "balanced": StartupProfile.RECOMMENDED,
        "advanced": StartupProfile.DEEP,
    }
    return mapping.get(profile, StartupProfile.RECOMMENDED)


class AppState:
    def __init__(self, *, platform_services: PlatformServices | None = None, container: ServiceContainer | None = None):
        self.container = container or ServiceContainer()
        self.docker = self.container.docker()
        self.db = self.container.db()
        self.embedding = self.container.embedding()
        self.vector_store = self.container.vector_store()
        self.providers = self.container.providers_router()
        self.local_inference = self.container.local_inference()
        self.classifier = self.container.classifier()
        self.extensions = self.container.extensions()
        self.indexing = self.container.indexing()
        self.memory = self.container.memory()
        self.model_manager = self.container.model_manager()
        self.model_manager.set_runtime_controller(self.local_inference)
        self.plugins = self.container.plugins()
        self.finetune_jobs = self.container.finetune_jobs()
        self.platform = platform_services or self.container.platform()
        self.auth = self.platform.auth_provider
        self.room_storage = self.container.room_storage()
        self.chat = self.container.chat()
        self.chat_facade = self.container.chat_facade()
        self.workspace_service = self.container.workspace_service()
        self.memory_facade = self.container.memory_facade()
        self.inference_service = self.container.inference_service()
        self.system = self.container.system()
        self._initialized = False
        self._init_task = None

    async def wait_until_ready(self):
        """Wait for the background initialization task to complete."""
        if self._initialized:
            return
        if self._init_task is None:
            self._init_task = asyncio.create_task(self.initialize())
        await self._init_task

    async def initialize(self):
        """Asynchronous initialization of heavy components."""
        if self._initialized:
            return
        
        try:
            logger.info("Starting AppState asynchronous initialization...")
            await self.system.initialize()
            
            self._initialized = True
            logger.info("AppState initialization completed successfully.")
        except Exception as e:
            logger.exception(f"CRITICAL: AppState initialization failed: {e}")
            # Re-raise so wait_until_ready caller knows it's broken
            raise


app_state = AppState()


def set_app_state(state: AppState) -> None:
    global app_state
    app_state = state


def _auth_dependency(request: Request) -> None:
    app_state.auth.verify_request(request)


async def _require_ready() -> None:
    try:
        await app_state.wait_until_ready()
    except Exception as exc:
        logger.exception("AppState readiness check failed: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Sidecar initialization failed. Check runtime logs and restart.",
        ) from exc


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
        logger.info("Parent process died. Cleaning up Docker services before exit...")
        # Clean up Docker synchronously during force-exit scenario
        app_state.docker.stop(shutdown_desktop=False, remove_stack=False)
        os._exit(0)


@asynccontextmanager
async def lifespan(_: FastAPI):
    parent_monitor_task: asyncio.Task | None = None
    expected_parent = _expected_parent_pid()
    if expected_parent is not None:
        parent_monitor_task = asyncio.create_task(_watch_parent_lifecycle(expected_parent))
    
    # Perform deferred initialization in the background to allow the server to start immediately.
    # This ensures /health becomes responsive as soon as uvicorn starts.
    app_state._init_task = asyncio.create_task(app_state.initialize())
    init_task = app_state._init_task
    
    try:
        yield
    finally:
        # Ensure init_task is finished before shutdown if it's still running.
        if not init_task.done():
            init_task.cancel()
            with suppress(asyncio.CancelledError):
                await init_task
        if parent_monitor_task is not None:
            parent_monitor_task.cancel()
            with suppress(asyncio.CancelledError):
                await parent_monitor_task
        
        # Stop indexing watcher and Docker services
        app_state.indexing.stop_watcher()
        app_state.room_storage.close_all()
        if await app_state.system.shutdown():
            logger.info("Docker services stopped gracefully.")
        else:
            logger.warning("Docker services failed to stop gracefully.")




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


def create_app(*, state: AppState | None = None) -> FastAPI:
    if state is not None:
        set_app_state(state)
    app = FastAPI(title="Local AI Core Sidecar", version="0.3.0", lifespan=lifespan)

    @app.middleware("http")
    async def readiness_guard(request: Request, call_next):
        exempt_paths = {"/health", "/docs", "/redoc", "/openapi.json"}
        if request.url.path not in exempt_paths:
            try:
                await _require_ready()
            except HTTPException as exc:
                return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
        return await call_next(request)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/workspaces", response_model=WorkspaceResponse, dependencies=[Depends(_auth_dependency)])
    async def update_workspace(payload: WorkspaceUpdateRequest) -> WorkspaceResponse:
        await app_state.wait_until_ready()
        return await app_state.workspace_service.update_workspace(payload)

    @app.post("/v1/index/jobs", dependencies=[Depends(_auth_dependency)])
    async def start_index_job(payload: IndexJobRequest) -> dict[str, str]:
        await app_state.wait_until_ready()
        workspace = await app_state.workspace_service.get_workspace()
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

    @app.get("/v1/rooms/{room_id}/storage/status", dependencies=[Depends(_auth_dependency)])
    async def get_room_storage_status(room_id: str) -> dict[str, Any]:
        await app_state.wait_until_ready()
        safe_room_id = str(room_id or "").strip()
        if not safe_room_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="room_id is required")
        return app_state.room_storage.room_storage_status(room_id=safe_room_id)

    @app.post("/v1/rooms/{room_id}/storage/reindex", dependencies=[Depends(_auth_dependency)])
    async def reindex_room_storage(room_id: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        await app_state.wait_until_ready()
        safe_room_id = str(room_id or "").strip()
        if not safe_room_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="room_id is required")
        body = payload or {}
        raw_included = body.get("included_paths")
        raw_excluded = body.get("excluded_paths")
        included = [str(item).strip() for item in raw_included] if isinstance(raw_included, list) else None
        excluded = [str(item).strip() for item in raw_excluded] if isinstance(raw_excluded, list) else None
        result = await app_state.room_storage.reindex_room_async(
            room_id=safe_room_id,
            scope=str(body.get("scope") or "full"),
            included_paths=included,
            excluded_paths=excluded,
        )
        if not bool(result.get("ok", False)):
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(result.get("error") or "room storage not found"))
        return result

    @app.delete("/v1/rooms/{room_id}/storage", dependencies=[Depends(_auth_dependency)])
    async def delete_room_storage(room_id: str) -> dict[str, Any]:
        await app_state.wait_until_ready()
        safe_room_id = str(room_id or "").strip()
        if not safe_room_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="room_id is required")
        return await app_state.room_storage.delete_room_storage_async(room_id=safe_room_id)

    def _resolve_room_memory_or_404(room_id: str, room_scope_hash: str | None = None):
        safe_room_id = str(room_id or "").strip()
        if not safe_room_id:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="room_id is required")
        resolved = app_state.room_storage.resolve_last_memory_service(
            room_id=safe_room_id,
            scope_hash=room_scope_hash,
        )
        if resolved is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="room storage memory not found")
        return resolved

    @app.get("/v1/rooms/{room_id}/memory/session/relevant", response_model=SessionMemoryResponse, dependencies=[Depends(_auth_dependency)])
    async def room_memory_session_relevant(
        room_id: str,
        session_id: str | None = None,
        room_scope_hash: str | None = None,
    ) -> SessionMemoryResponse:
        await app_state.wait_until_ready()
        memory, handle = _resolve_room_memory_or_404(room_id, room_scope_hash)
        target_session = str(session_id or handle.key.room_id or "").strip()
        if not target_session:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="session_id is required")
        items = await app_state.memory_facade.get_relevant_session(memory=memory, session_id=target_session)
        return SessionMemoryResponse(items=items)

    @app.get("/v1/rooms/{room_id}/memory/workspace/relevant", response_model=WorkspaceMemoryResponse, dependencies=[Depends(_auth_dependency)])
    async def room_memory_workspace_relevant(
        room_id: str,
        workspace_id: str | None = None,
        intent: str | None = None,
        room_scope_hash: str | None = None,
    ) -> WorkspaceMemoryResponse:
        await app_state.wait_until_ready()
        memory, _handle = _resolve_room_memory_or_404(room_id, room_scope_hash)
        target_workspace = str(workspace_id or memory.get_workspace_identity().workspace_id or "").strip()
        if not target_workspace:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="workspace_id is required")
        items = await app_state.memory_facade.get_relevant_workspace(
            memory=memory,
            workspace_id=target_workspace,
            intent=intent,
        )
        return WorkspaceMemoryResponse(items=items)

    @app.get("/v1/rooms/{room_id}/memory/episodic/relevant", response_model=EpisodicMemoryResponse, dependencies=[Depends(_auth_dependency)])
    async def room_memory_episodic_relevant(
        room_id: str,
        workspace_id: str | None = None,
        intent: str | None = None,
        related_file_ids: str | None = None,
        room_scope_hash: str | None = None,
    ) -> EpisodicMemoryResponse:
        await app_state.wait_until_ready()
        memory, _handle = _resolve_room_memory_or_404(room_id, room_scope_hash)
        related = [item.strip() for item in (related_file_ids or "").split(",") if item.strip()]
        target_workspace = str(workspace_id or memory.get_workspace_identity().workspace_id or "").strip() or None
        items = await app_state.memory_facade.get_relevant_episodic(
            memory=memory,
            workspace_id=target_workspace,
            intent=intent,
            related_files=related,
        )
        return EpisodicMemoryResponse(items=items)

    @app.post("/v1/rooms/{room_id}/memory/events", response_model=MemoryEventResponse, dependencies=[Depends(_auth_dependency)])
    async def room_memory_write_event(
        room_id: str,
        payload: MemoryEventRequest,
        room_scope_hash: str | None = None,
    ) -> MemoryEventResponse:
        await app_state.wait_until_ready()
        memory, handle = _resolve_room_memory_or_404(room_id, room_scope_hash)
        event = payload
        if not str(event.workspace_id or "").strip():
            event = payload.model_copy(update={"workspace_id": memory.get_workspace_identity().workspace_id})
        return await app_state.memory_facade.room_write_event(
            room_id=handle.key.room_id,
            room_scope_hash=room_scope_hash,
            memory=memory,
            payload=event,
        )

    @app.post("/v1/rooms/{room_id}/memory/clear", response_model=MemoryClearResponse, dependencies=[Depends(_auth_dependency)])
    async def room_memory_clear(
        room_id: str,
        payload: MemoryClearRequest,
        room_scope_hash: str | None = None,
    ) -> MemoryClearResponse:
        await app_state.wait_until_ready()
        memory, handle = _resolve_room_memory_or_404(room_id, room_scope_hash)
        return await app_state.memory_facade.room_clear(
            room_id=handle.key.room_id,
            room_scope_hash=room_scope_hash,
            memory=memory,
            payload=payload,
        )

    @app.get("/v1/rooms/{room_id}/memory/pins", response_model=PinnedMemoryResponse, dependencies=[Depends(_auth_dependency)])
    async def room_memory_list_pins(
        room_id: str,
        scope: str | None = None,
        workspace_id: str | None = None,
        room_scope_hash: str | None = None,
    ) -> PinnedMemoryResponse:
        await app_state.wait_until_ready()
        memory, _handle = _resolve_room_memory_or_404(room_id, room_scope_hash)
        items = memory.list_pinned_memory(scope=scope, workspace_id=workspace_id)
        return PinnedMemoryResponse(items=items)

    @app.post("/v1/rooms/{room_id}/memory/pin", response_model=MemoryPinResponse, dependencies=[Depends(_auth_dependency)])
    async def room_memory_pin(
        room_id: str,
        payload: MemoryPinRequest,
        room_scope_hash: str | None = None,
    ) -> MemoryPinResponse:
        await app_state.wait_until_ready()
        memory, handle = _resolve_room_memory_or_404(room_id, room_scope_hash)
        item = await app_state.memory_facade.room_pin(
            room_id=handle.key.room_id,
            room_scope_hash=room_scope_hash,
            memory=memory,
            payload=payload,
        )
        return MemoryPinResponse(item=item)

    @app.delete("/v1/rooms/{room_id}/memory/pin/{memory_id}", response_model=dict[str, bool], dependencies=[Depends(_auth_dependency)])
    async def room_memory_unpin(room_id: str, memory_id: str, room_scope_hash: str | None = None) -> dict[str, bool]:
        await app_state.wait_until_ready()
        memory, handle = _resolve_room_memory_or_404(room_id, room_scope_hash)
        removed = await app_state.memory_facade.room_unpin(
            room_id=handle.key.room_id,
            room_scope_hash=room_scope_hash,
            memory=memory,
            memory_id=memory_id,
        )
        return {"removed": bool(removed)}

    @app.get("/v1/extensions/capabilities", response_model=ExtensionCapabilitiesResponse, dependencies=[Depends(_auth_dependency)])
    def get_extension_capabilities() -> ExtensionCapabilitiesResponse:
        return app_state.extensions.capabilities_snapshot()

    @app.get("/v1/extensions/plugins", response_model=PluginRegistryResponse, dependencies=[Depends(_auth_dependency)])
    async def list_extension_plugins() -> PluginRegistryResponse:
        await app_state.wait_until_ready()
        return await app_state.workspace_service.run_read(app_state.plugins.list_plugins)

    @app.post("/v1/extensions/plugins/register", response_model=PluginRegistryResponse, dependencies=[Depends(_auth_dependency)])
    async def register_extension_plugin(payload: PluginRegisterRequest) -> PluginRegistryResponse:
        await app_state.wait_until_ready()
        try:
            await app_state.workspace_service.run_write(app_state.plugins.register_plugin, payload)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        return await app_state.workspace_service.run_read(app_state.plugins.list_plugins)

    @app.post("/v1/extensions/plugins/{plugin_id}/enable", response_model=PluginEnableResponse, dependencies=[Depends(_auth_dependency)])
    async def enable_extension_plugin(plugin_id: str) -> PluginEnableResponse:
        await app_state.wait_until_ready()
        try:
            plugin = await app_state.workspace_service.run_write(app_state.plugins.enable_plugin, plugin_id)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        capabilities = await app_state.workspace_service.run_read(app_state.plugins.current_capability_states)
        return PluginEnableResponse(plugin=plugin, capabilities=capabilities)

    @app.post("/v1/extensions/plugins/{plugin_id}/disable", response_model=PluginEnableResponse, dependencies=[Depends(_auth_dependency)])
    async def disable_extension_plugin(plugin_id: str) -> PluginEnableResponse:
        await app_state.wait_until_ready()
        try:
            plugin = await app_state.workspace_service.run_write(app_state.plugins.disable_plugin, plugin_id)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        capabilities = await app_state.workspace_service.run_read(app_state.plugins.current_capability_states)
        return PluginEnableResponse(plugin=plugin, capabilities=capabilities)

    @app.delete("/v1/extensions/plugins/{plugin_id}", response_model=dict[str, bool], dependencies=[Depends(_auth_dependency)])
    async def delete_extension_plugin(plugin_id: str) -> dict[str, bool]:
        await app_state.wait_until_ready()
        try:
            removed = await app_state.workspace_service.run_write(app_state.plugins.delete_plugin, plugin_id)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        return {"removed": bool(removed)}

    @app.post("/v1/extensions/finetune/jobs/submit", response_model=FinetuneJobSubmitResponse, dependencies=[Depends(_auth_dependency)])
    async def submit_finetune_job(payload: FinetuneJobSubmitRequest) -> FinetuneJobSubmitResponse:
        await app_state.wait_until_ready()
        return app_state.finetune_jobs.submit_job(payload)

    @app.get("/v1/extensions/finetune/jobs/{job_id}/status", response_model=FinetuneJobStatusResponse, dependencies=[Depends(_auth_dependency)])
    async def get_finetune_job_status(job_id: str) -> FinetuneJobStatusResponse:
        await app_state.wait_until_ready()
        try:
            return app_state.finetune_jobs.get_job_status(job_id=job_id)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    @app.post("/v1/extensions/finetune/jobs/publish", response_model=FinetuneModelPublishResponse, dependencies=[Depends(_auth_dependency)])
    async def publish_finetune_model(payload: FinetuneModelPublishRequest) -> FinetuneModelPublishResponse:
        await app_state.wait_until_ready()
        try:
            return app_state.finetune_jobs.publish_model(payload)
        except KeyError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc

    @app.post("/v1/chat/local", response_model=LocalChatResponse, dependencies=[Depends(_auth_dependency)])
    async def local_chat(payload: LocalChatRequest) -> LocalChatResponse:
        await app_state.wait_until_ready()
        return app_state.chat_facade.local_chat(payload)

    @app.post("/v2/chat/local", response_model=ComposedChatResponseV2, dependencies=[Depends(_auth_dependency)])
    async def local_chat_v2(payload: LocalChatRequestV2) -> ComposedChatResponseV2:
        await app_state.wait_until_ready()
        try:
            return await app_state.chat_facade.local_chat_v2(payload)
        except asyncio.TimeoutError as exc:
            raise HTTPException(
                status_code=status.HTTP_504_GATEWAY_TIMEOUT,
                detail="local inference timeout",
            ) from exc

    @app.post("/v2/chat/local/stream", dependencies=[Depends(_auth_dependency)])
    async def local_chat_v2_stream(payload: LocalChatRequestV2):
        await app_state.wait_until_ready()
        from fastapi.responses import StreamingResponse
        return StreamingResponse(
            app_state.chat_facade.local_chat_v2_stream(payload),
            media_type="text/event-stream"
        )

    @app.post("/v1/chat/deep-analysis", response_model=DeepAnalysisResponse, dependencies=[Depends(_auth_dependency)])
    async def deep_analysis(payload: DeepAnalysisRequest) -> DeepAnalysisResponse:
        try:
            return await app_state.chat_facade.deep_analysis(payload)
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
    async def get_settings() -> SettingsModel:
        await app_state.wait_until_ready()
        return await app_state.workspace_service.get_settings()

    @app.put("/v1/settings", response_model=SettingsModel, dependencies=[Depends(_auth_dependency)])
    async def update_settings(payload: SettingsModel) -> SettingsModel:
        await app_state.wait_until_ready()
        return await app_state.workspace_service.update_settings(payload)

    @app.get("/v1/memory/session/relevant", response_model=SessionMemoryResponse, dependencies=[Depends(_auth_dependency)])
    async def get_relevant_session_memory(session_id: str) -> SessionMemoryResponse:
        await app_state.wait_until_ready()
        items = await app_state.memory_facade.get_relevant_session(memory=app_state.memory, session_id=session_id)
        return SessionMemoryResponse(items=items)

    @app.get("/v1/memory/workspace/relevant", response_model=WorkspaceMemoryResponse, dependencies=[Depends(_auth_dependency)])
    async def get_relevant_workspace_memory(workspace_id: str, intent: str | None = None) -> WorkspaceMemoryResponse:
        await app_state.wait_until_ready()
        items = await app_state.memory_facade.get_relevant_workspace(
            memory=app_state.memory,
            workspace_id=workspace_id,
            intent=intent,
        )
        return WorkspaceMemoryResponse(items=items)

    @app.get("/v1/memory/preferences", response_model=UserPreferencesResponse, dependencies=[Depends(_auth_dependency)])
    async def get_user_preferences() -> UserPreferencesResponse:
        await app_state.wait_until_ready()
        items = await app_state.memory_facade.list_preferences(memory=app_state.memory)
        return UserPreferencesResponse(items=items)

    @app.get("/v1/memory/episodic/relevant", response_model=EpisodicMemoryResponse, dependencies=[Depends(_auth_dependency)])
    async def get_relevant_episodic_memory(
        workspace_id: str | None = None,
        intent: str | None = None,
        related_file_ids: str | None = None,
    ) -> EpisodicMemoryResponse:
        await app_state.wait_until_ready()
        related = [item.strip() for item in (related_file_ids or "").split(",") if item.strip()]
        items = await app_state.memory_facade.get_relevant_episodic(
            memory=app_state.memory,
            workspace_id=workspace_id,
            intent=intent,
            related_files=related,
        )
        return EpisodicMemoryResponse(items=items)

    @app.post("/v1/memory/events", response_model=MemoryEventResponse, dependencies=[Depends(_auth_dependency)])
    async def write_memory_event(payload: MemoryEventRequest) -> MemoryEventResponse:
        await app_state.wait_until_ready()
        return await app_state.memory_facade.write_event(memory=app_state.memory, payload=payload)

    @app.post("/v1/memory/clear", response_model=MemoryClearResponse, dependencies=[Depends(_auth_dependency)])
    async def clear_memory(payload: MemoryClearRequest) -> MemoryClearResponse:
        await app_state.wait_until_ready()
        return await app_state.memory_facade.clear(memory=app_state.memory, payload=payload)

    @app.post("/v1/memory/pin", response_model=MemoryPinResponse, dependencies=[Depends(_auth_dependency)])
    async def pin_memory(payload: MemoryPinRequest) -> MemoryPinResponse:
        await app_state.wait_until_ready()
        item = await app_state.memory_facade.pin(memory=app_state.memory, payload=payload)
        return MemoryPinResponse(item=item)

    @app.delete("/v1/memory/pin/{memory_id}", response_model=dict[str, bool], dependencies=[Depends(_auth_dependency)])
    async def unpin_memory(memory_id: str) -> dict[str, bool]:
        await app_state.wait_until_ready()
        removed = await app_state.memory_facade.unpin(memory=app_state.memory, memory_id=memory_id)
        return {"removed": bool(removed)}

    @app.get("/v1/memory/pins", response_model=PinnedMemoryResponse, dependencies=[Depends(_auth_dependency)])
    async def list_pins(scope: str | None = None, workspace_id: str | None = None) -> PinnedMemoryResponse:
        await app_state.wait_until_ready()
        items = await app_state.memory_facade.list_pins(
            memory=app_state.memory,
            scope=scope,
            workspace_id=workspace_id,
        )
        return PinnedMemoryResponse(items=items)

    @app.get("/v1/status", dependencies=[Depends(_auth_dependency)])
    async def get_status() -> dict:
        await app_state.wait_until_ready()
        snapshot = await app_state.workspace_service.get_status_snapshot()
        snapshot["privacy_mode"] = (await app_state.workspace_service.get_settings()).privacy_mode
        return snapshot

    @app.get("/v1/docs", response_model=DocumentListResponse, dependencies=[Depends(_auth_dependency)])
    async def list_docs(
        search: str | None = None,
        category: str | None = None,
        tags: str | None = None,
        year: int | None = None,
        project: str | None = None,
        excluded: bool | None = False,
        limit: int = 100,
        offset: int = 0,
    ) -> DocumentListResponse:
        await app_state.wait_until_ready()
        tag_list = [item.strip() for item in (tags or "").split(",") if item.strip()]
        filters = ChatFilters(
            category=category,
            tags=tag_list,
            year=year,
            project=project,
            excluded=excluded,
        )
        workspace = await app_state.workspace_service.get_workspace()
        allowed_doc_ids = await app_state.workspace_service.find_doc_ids_for_workspace(
            included_paths=workspace.included_paths,
            excluded_paths=workspace.excluded_paths,
            filters=filters,
            search=search,
        )
        docs, total = await app_state.workspace_service.list_documents(
            search=search,
            filters=filters,
            allowed_doc_ids=allowed_doc_ids,
            limit=max(1, min(limit, 300)),
            offset=max(0, offset),
        )
        return DocumentListResponse(documents=docs, total=total, offset=offset, limit=limit)

    @app.put("/v1/docs/{doc_id}/metadata", response_model=DocumentMetadata, dependencies=[Depends(_auth_dependency)])
    async def update_doc_metadata(doc_id: str, payload: DocumentMetadataUpdate) -> DocumentMetadata:
        await app_state.wait_until_ready()
        try:
            return await app_state.workspace_service.update_document_metadata(doc_id, payload)
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

    @app.get("/v1/models/download/progress", response_model=DownloadProgressResponse, dependencies=[Depends(_auth_dependency)])
    def model_download_progress() -> DownloadProgressResponse:
        return DownloadProgressResponse(items=app_state.model_manager.get_download_progress())

    @app.get("/v1/models", response_model=ModelListResponse, dependencies=[Depends(_auth_dependency)])
    def list_models() -> ModelListResponse:
        health = app_state.inference_service.health()
        try:
            models = app_state.model_manager.list_models(runtime_health=health)
        except TypeError:
            # Backward compatibility for test doubles or older adapters without runtime_health.
            models = app_state.model_manager.list_models()
        return ModelListResponse(models=models)

    @app.get("/v1/models/catalog", response_model=ModelCatalogResponse, dependencies=[Depends(_auth_dependency)])
    async def get_model_catalog() -> ModelCatalogResponse:
        await app_state.wait_until_ready()
        settings_model = await app_state.workspace_service.get_settings()
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
    async def activate_catalog_model(payload: ModelCatalogActivateRequest) -> ModelCatalogActivateResponse:
        await app_state.wait_until_ready()
        try:
            activated = app_state.model_manager.activate_catalog_model(payload.model_id)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

        settings_model = await app_state.workspace_service.get_settings()
        settings_model.local_engine = activated.engine
        if activated.engine == LocalEngine.MLX:
            settings_model.mlx_model_path = activated.model_path
            settings_model.llama_model_path = None
        else:
            settings_model.mlx_model_path = None
            settings_model.llama_model_path = activated.model_path
        settings_model.model_profile = activated.profile
        settings_model.startup_profile = _profile_to_startup(activated.profile)
        await app_state.workspace_service.update_settings(settings_model)
        return activated

    @app.delete("/v1/models/catalog/{model_id}", response_model=ModelCatalogDeleteResponse, dependencies=[Depends(_auth_dependency)])
    def delete_catalog_model(model_id: str) -> ModelCatalogDeleteResponse:
        try:
            return app_state.model_manager.delete_catalog_model(model_id)
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    @app.post("/v1/models/runtime/prepare", response_model=RuntimePrepareResponse, dependencies=[Depends(_auth_dependency)])
    async def prepare_runtime(payload: RuntimePrepareRequest) -> RuntimePrepareResponse:
        await app_state.wait_until_ready()
        settings_model = await app_state.workspace_service.get_settings()
        return app_state.local_inference.prepare_runtime(
            engine=payload.engine,
            profile=settings_model.model_profile,
            mlx_model_path=payload.model_path if payload.engine == LocalEngine.MLX else settings_model.mlx_model_path,
            llama_model_path=payload.model_path if payload.engine == LocalEngine.LLAMA_CPP else settings_model.llama_model_path,
        )

    return app


app = create_app()
