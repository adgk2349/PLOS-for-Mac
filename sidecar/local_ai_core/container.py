from __future__ import annotations

import logging
import os
from pathlib import Path

try:
    from dependency_injector import containers, providers  # type: ignore

    _HAS_DI = True
except Exception:  # pragma: no cover - optional runtime dependency
    containers = None
    providers = None
    _HAS_DI = False

from .chat import ChatService
from .config import settings
from .embedding import EmbeddingService
from .indexing import DocumentClassifier, IndexingService
from .infrastructure.docker_service import DockerService
from .memory_service import MemoryService
from .platform_services import load_platform_services
from .plugins import ExtensionKernel, FinetuneJobService, PluginRegistryService
from .room_storage import RoomStorageRegistry
from .runtime.external_providers import ProviderRouter
from .runtime.local_inference import LocalInferenceEngine
from .runtime.model_manager import ModelManager
from .services import ChatFacade, InferenceService, MemoryFacade, SystemOrchestrator, WorkspaceService
from .storage.db import Database
from .storage.vector_store import VectorStore

logger = logging.getLogger(__name__)


def _resolve_searxng_dir() -> str:
    base_sidecar_dir = os.path.dirname(os.path.abspath(__file__))
    searxng_dir = os.path.join(os.path.dirname(base_sidecar_dir), "searxng")
    os.makedirs(searxng_dir, exist_ok=True)
    return searxng_dir


def _resolve_models_dir() -> Path | None:
    models_dir_env = str(os.environ.get("LOCAL_AI_MODELS_DIR", "") or "").strip()
    return Path(models_dir_env).expanduser().resolve() if models_dir_env else None


if _HAS_DI:
    class ServiceContainer(containers.DeclarativeContainer):  # type: ignore[misc]
        config = providers.Object(settings)

        docker = providers.Singleton(DockerService, _resolve_searxng_dir())
        db = providers.Singleton(Database, sqlite_path=config.provided.sqlite_path, skip_init=True, docker=docker)
        embedding = providers.Singleton(EmbeddingService, dim=config.provided.embedding_dim)
        vector_store = providers.Singleton(VectorStore, config.provided.lancedb_path, dim=config.provided.embedding_dim)
        providers_router = providers.Singleton(ProviderRouter)
        local_inference = providers.Singleton(LocalInferenceEngine)
        classifier = providers.Singleton(DocumentClassifier, embedding, local_inference)
        extensions = providers.Singleton(ExtensionKernel, db)
        indexing = providers.Singleton(
            IndexingService,
            db,
            vector_store,
            embedding,
            classifier,
            capability_router=extensions.provided.router,
            settings_loader=db.provided.get_settings,
        )
        memory = providers.Singleton(MemoryService, db)
        model_manager = providers.Singleton(ModelManager, config.provided.data_dir, models_dir=providers.Callable(_resolve_models_dir))
        plugins = providers.Singleton(PluginRegistryService, db=db, extension_kernel=extensions)
        finetune_jobs = providers.Singleton(FinetuneJobService, extension_kernel=extensions)
        platform = providers.Singleton(load_platform_services, session_token=config.provided.session_token)
        room_storage = providers.Singleton(
            RoomStorageRegistry,
            base_data_dir=providers.Callable(lambda s: s.data_dir / "rooms", config),
            embedding_service=embedding,
            provider_router=providers_router,
            local_inference=local_inference,
            capability_router=extensions.provided.router,
            docker_service=docker,
            settings_loader=db.provided.get_settings,
            workspace_loader=db.provided.get_workspace,
            embedding_dim=config.provided.embedding_dim,
        )
        chat = providers.Singleton(
            ChatService,
            db,
            vector_store,
            embedding,
            providers_router,
            local_inference,
            memory,
            indexing,
            capability_router=extensions.provided.router,
            docker_service=docker,
            room_registry=room_storage,
        )
        chat_facade = providers.Singleton(ChatFacade, chat)
        workspace_service = providers.Singleton(WorkspaceService, db=db, indexing=indexing)
        memory_facade = providers.Singleton(MemoryFacade, workspace_service=workspace_service, room_storage=room_storage)
        inference_service = providers.Singleton(InferenceService, local_inference=local_inference, model_manager=model_manager)
        system = providers.Singleton(SystemOrchestrator, db=db, indexing=indexing, docker=docker)
else:
    class ServiceContainer:
        """Fallback container for environments where dependency_injector is unavailable."""

        def __init__(self):
            logger.warning("dependency_injector is unavailable; using fallback service container.")
            self._singletons: dict[str, object] = {}

        def _get(self, key: str, factory):
            if key not in self._singletons:
                self._singletons[key] = factory()
            return self._singletons[key]

        def docker(self):
            return self._get("docker", lambda: DockerService(_resolve_searxng_dir()))

        def db(self):
            return self._get("db", lambda: Database(settings.sqlite_path, skip_init=True, docker=self.docker()))

        def embedding(self):
            return self._get("embedding", lambda: EmbeddingService(dim=settings.embedding_dim))

        def vector_store(self):
            return self._get("vector_store", lambda: VectorStore(settings.lancedb_path, dim=settings.embedding_dim))

        def providers_router(self):
            return self._get("providers_router", ProviderRouter)

        def local_inference(self):
            return self._get("local_inference", LocalInferenceEngine)

        def classifier(self):
            return self._get("classifier", lambda: DocumentClassifier(self.embedding(), self.local_inference()))

        def extensions(self):
            return self._get("extensions", lambda: ExtensionKernel(self.db()))

        def indexing(self):
            return self._get(
                "indexing",
                lambda: IndexingService(
                    self.db(),
                    self.vector_store(),
                    self.embedding(),
                    self.classifier(),
                    capability_router=self.extensions().router,
                    settings_loader=self.db().get_settings,
                ),
            )

        def memory(self):
            return self._get("memory", lambda: MemoryService(self.db()))

        def model_manager(self):
            return self._get("model_manager", lambda: ModelManager(settings.data_dir, models_dir=_resolve_models_dir()))

        def plugins(self):
            return self._get("plugins", lambda: PluginRegistryService(db=self.db(), extension_kernel=self.extensions()))

        def finetune_jobs(self):
            return self._get("finetune_jobs", lambda: FinetuneJobService(extension_kernel=self.extensions()))

        def platform(self):
            return self._get("platform", lambda: load_platform_services(settings.session_token))

        def room_storage(self):
            return self._get(
                "room_storage",
                lambda: RoomStorageRegistry(
                    base_data_dir=settings.data_dir / "rooms",
                    embedding_service=self.embedding(),
                    provider_router=self.providers_router(),
                    local_inference=self.local_inference(),
                    capability_router=self.extensions().router,
                    docker_service=self.docker(),
                    settings_loader=self.db().get_settings,
                    workspace_loader=self.db().get_workspace,
                    embedding_dim=settings.embedding_dim,
                ),
            )

        def chat(self):
            return self._get(
                "chat",
                lambda: ChatService(
                    self.db(),
                    self.vector_store(),
                    self.embedding(),
                    self.providers_router(),
                    self.local_inference(),
                    self.memory(),
                    self.indexing(),
                    capability_router=self.extensions().router,
                    docker_service=self.docker(),
                    room_registry=self.room_storage(),
                ),
            )

        def chat_facade(self):
            return self._get("chat_facade", lambda: ChatFacade(self.chat()))

        def workspace_service(self):
            return self._get("workspace_service", lambda: WorkspaceService(db=self.db(), indexing=self.indexing()))

        def memory_facade(self):
            return self._get(
                "memory_facade",
                lambda: MemoryFacade(workspace_service=self.workspace_service(), room_storage=self.room_storage()),
            )

        def inference_service(self):
            return self._get(
                "inference_service",
                lambda: InferenceService(local_inference=self.local_inference(), model_manager=self.model_manager()),
            )

        def system(self):
            return self._get(
                "system",
                lambda: SystemOrchestrator(db=self.db(), indexing=self.indexing(), docker=self.docker()),
            )
