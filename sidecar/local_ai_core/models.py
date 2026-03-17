from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class PrivacyMode(str, Enum):
    LOCAL_ONLY = "LOCAL_ONLY"
    HYBRID = "HYBRID"
    CONFIRM_BEFORE_EXTERNAL = "CONFIRM_BEFORE_EXTERNAL"


class WorkMode(str, Enum):
    GENERAL = "GENERAL"
    SUMMARY = "SUMMARY"
    RESEARCH = "RESEARCH"
    DEVELOPMENT = "DEVELOPMENT"
    WRITING = "WRITING"
    PLANNING = "PLANNING"
    STRICT_SEARCH = "STRICT_SEARCH"


class StartupProfile(str, Enum):
    FAST = "FAST"
    RECOMMENDED = "RECOMMENDED"
    DEEP = "DEEP"


class LocalEngine(str, Enum):
    MLX = "mlx"
    LLAMA_CPP = "llama_cpp"


class ActionPermissionMode(str, Enum):
    ASK_PER_ACTION = "ASK_PER_ACTION"
    ASK_EVERY_TIME = "ASK_EVERY_TIME"


class ChatIntent(str, Enum):
    FILE_SEARCH = "FILE_SEARCH"
    DOCUMENT_QA = "DOCUMENT_QA"
    TASK_REQUEST = "TASK_REQUEST"
    AMBIGUOUS = "AMBIGUOUS"


class SuggestedActionKind(str, Enum):
    OPEN_FILE = "OPEN_FILE"
    SUMMARIZE_TOP = "SUMMARIZE_TOP"
    COMPARE_TOP = "COMPARE_TOP"
    ASK_FOLLOWUP = "ASK_FOLLOWUP"


class ActionExecutionMode(str, Enum):
    PROMPT_INJECTION = "PROMPT_INJECTION"
    SYSTEM = "SYSTEM"


class WorkspaceUpdateRequest(BaseModel):
    included_paths: list[str] = Field(default_factory=list)
    excluded_paths: list[str] = Field(default_factory=list)
    startup_profile: StartupProfile = StartupProfile.RECOMMENDED
    default_mode: WorkMode = WorkMode.GENERAL


class WorkspaceResponse(WorkspaceUpdateRequest):
    updated_at: datetime


class IndexJobRequest(BaseModel):
    scope: Literal["incremental", "full"]


class IndexJobStatus(BaseModel):
    job_id: str
    scope: Literal["incremental", "full"]
    status: Literal["queued", "running", "completed", "failed"]
    progress: float = 0.0
    processed_files: int = 0
    failed_files: int = 0
    stage: Literal["queued", "scan", "parse", "classify", "embed", "store", "done", "failed"] = "queued"
    error: str | None = None


class FailureItem(BaseModel):
    path: str
    reason: str
    last_attempt_at: datetime


class FailureListResponse(BaseModel):
    failures: list[FailureItem]


class ChatFilters(BaseModel):
    category: str | None = None
    tags: list[str] = Field(default_factory=list)
    year: int | None = None
    project: str | None = None
    excluded: bool | None = None


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    file_path: str
    snippet: str
    score: float
    modified_at: datetime
    category: str = "참고자료"
    subcategory: str = ""
    tags: list[str] = Field(default_factory=list)
    document_type: str = ""
    importance: float = 0.5


class LocalChatRequest(BaseModel):
    query: str
    mode: WorkMode = WorkMode.GENERAL
    conversation_id: str | None = None
    top_k: int | None = None
    filters: ChatFilters | None = None


class SuggestedAction(BaseModel):
    action_id: str
    kind: SuggestedActionKind
    label: str
    execution_mode: ActionExecutionMode
    payload: dict[str, str] = Field(default_factory=dict)


class LocalChatResponse(BaseModel):
    intent: ChatIntent
    lead: str
    result_summary: str
    citations: list[Citation] = Field(default_factory=list)
    actions: list[SuggestedAction] = Field(default_factory=list)
    reasoning_brief: str | None = None
    mode: WorkMode
    used_profile: StartupProfile
    is_local: bool = True
    engine_used: LocalEngine | None = None
    used_fallback: bool = False
    runtime_detail: str | None = None


class DeepAnalysisRequest(BaseModel):
    query: str
    mode: WorkMode = WorkMode.GENERAL
    provider: Literal["openai", "anthropic"]
    selected_citations: list[Citation] = Field(default_factory=list)
    user_confirmed: bool = False


class ExternalCallEvent(BaseModel):
    provider: str
    sent_chars: int
    approved_by_user: bool
    timestamp: datetime


class DeepAnalysisResponse(BaseModel):
    answer: str
    provider: str
    event: ExternalCallEvent
    is_local: bool = False


class SettingsModel(BaseModel):
    privacy_mode: PrivacyMode = PrivacyMode.HYBRID
    startup_profile: StartupProfile = StartupProfile.RECOMMENDED
    model_profile: str = "recommended"
    local_engine: LocalEngine = LocalEngine.MLX
    mlx_model_path: str | None = None
    llama_model_path: str | None = None
    reindex_policy: Literal["filewatch_incremental", "manual_only", "scheduled_full"] = "filewatch_incremental"
    language: str = "auto"
    action_permission_mode: ActionPermissionMode = ActionPermissionMode.ASK_PER_ACTION


class DocumentMetadata(BaseModel):
    doc_id: str
    path: str
    file_type: str
    modified_at: datetime
    indexed_at: datetime
    summary: str = ""
    category: str = "참고자료"
    subcategory: str = ""
    document_type: str = ""
    tags: list[str] = Field(default_factory=list)
    year: int | None = None
    project: str | None = None
    importance: float = 0.5
    excluded: bool = False


class DocumentMetadataUpdate(BaseModel):
    category: str | None = None
    subcategory: str | None = None
    document_type: str | None = None
    tags: list[str] | None = None
    year: int | None = None
    project: str | None = None
    importance: float | None = None
    excluded: bool | None = None


class DocumentListResponse(BaseModel):
    documents: list[DocumentMetadata] = Field(default_factory=list)
    total: int = 0
    offset: int = 0
    limit: int = 100


class ClassificationResult(BaseModel):
    summary: str = ""
    category: str = "참고자료"
    subcategory: str = ""
    document_type: str = ""
    tags: list[str] = Field(default_factory=list)
    year: int | None = None
    project: str | None = None
    importance: float = 0.5


class ModelDownloadRequest(BaseModel):
    url: str
    engine: LocalEngine = LocalEngine.MLX
    filename: str | None = None


class ModelDownloadResponse(BaseModel):
    file_name: str
    saved_path: str
    engine: LocalEngine
    bytes_written: int


class ModelListItem(BaseModel):
    file_name: str
    path: str
    engine: LocalEngine
    size_bytes: int
    modified_at: datetime


class ModelListResponse(BaseModel):
    models: list[ModelListItem] = Field(default_factory=list)


class RuntimePrepareRequest(BaseModel):
    engine: LocalEngine = LocalEngine.MLX
    model_path: str | None = None


class RuntimePrepareResponse(BaseModel):
    engine: LocalEngine
    ready: bool
    package_available: bool
    model_path: str | None = None
    model_exists: bool = False
    accelerator: str = ""
    detail: str = ""


class DistributionType(str, Enum):
    HUGGINGFACE_REPO = "huggingface_repo"
    HUGGINGFACE_FILE = "huggingface_file"


class ModelInstallStatus(str, Enum):
    NOT_INSTALLED = "not_installed"
    DOWNLOADING = "downloading"
    INSTALLED = "installed"
    ACTIVE = "active"
    FAILED = "failed"


class ModelSupportFlags(BaseModel):
    chat: bool = True
    rag: bool = True
    tool_use: bool = False
    vision: bool = False


class ModelCatalogItem(BaseModel):
    id: str
    name: str
    profile: Literal["fast", "balanced", "advanced"]
    engine: LocalEngine
    distribution_type: DistributionType
    repo_id: str
    filename: str | None = None
    download_label: str
    description: str
    size_gb: float
    recommended_for: list[str] = Field(default_factory=list)
    recommended_memory_gb: int = 8
    tags: list[str] = Field(default_factory=list)
    supports: ModelSupportFlags = Field(default_factory=ModelSupportFlags)
    default: bool = False


class ModelCatalogManifest(BaseModel):
    version: int = 1
    default_profile: Literal["fast", "balanced", "advanced"] = "balanced"
    models: list[ModelCatalogItem] = Field(default_factory=list)


class ModelCatalogItemState(ModelCatalogItem):
    status: ModelInstallStatus = ModelInstallStatus.NOT_INSTALLED
    installed_path: str | None = None
    active: bool = False
    failure_reason: str | None = None


class ModelCatalogResponse(BaseModel):
    version: int = 1
    default_profile: Literal["fast", "balanced", "advanced"] = "balanced"
    models: list[ModelCatalogItemState] = Field(default_factory=list)


class ModelCatalogInstallRequest(BaseModel):
    model_id: str


class ModelCatalogInstallResponse(BaseModel):
    model_id: str
    status: ModelInstallStatus
    engine: LocalEngine
    saved_path: str | None = None
    detail: str = ""


class ModelCatalogActivateRequest(BaseModel):
    model_id: str


class ModelCatalogActivateResponse(BaseModel):
    model_id: str
    engine: LocalEngine
    model_path: str
    profile: Literal["fast", "balanced", "advanced"]


class ModelCatalogDeleteResponse(BaseModel):
    model_id: str
    removed: bool = False
