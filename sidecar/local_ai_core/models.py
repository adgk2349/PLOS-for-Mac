from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

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


class WorkspaceMemoryMode(str, Enum):
    NORMAL = "normal"
    DISABLED = "disabled"
    PINNED_ONLY = "pinned_only"


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
    SHOW_DIFF = "SHOW_DIFF"
    CREATE_DRAFT = "CREATE_DRAFT"
    SHOW_OTHER_CANDIDATES = "SHOW_OTHER_CANDIDATES"
    MAKE_SHORTER = "MAKE_SHORTER"
    OPEN_SECOND = "OPEN_SECOND"
    SHOW_PREVIOUS_CANDIDATE = "SHOW_PREVIOUS_CANDIDATE"


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


class ResponseLength(str, Enum):
    SHORT = "short"
    MEDIUM = "medium"
    LONG = "long"


class BehaviorPolicy(BaseModel):
    workspace_weights: dict[str, float] = Field(default_factory=dict)
    preferred_mode: WorkMode | None = None
    preferred_action_order: list[SuggestedActionKind] = Field(default_factory=list)
    preferred_response_length: ResponseLength = ResponseLength.MEDIUM


class BehaviorOverrides(BaseModel):
    workspace_weights: dict[str, float] | None = None
    preferred_mode: WorkMode | None = None
    preferred_action_order: list[SuggestedActionKind] | None = None
    preferred_response_length: ResponseLength | None = None


class LocalChatRequestV2(LocalChatRequest):
    behavior_overrides: BehaviorOverrides | None = None
    session_id: str | None = None


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


class ReasoningIntent(str, Enum):
    GENERAL_CHAT = "general_chat"
    FIND_FILE = "find_file"
    SUMMARIZE_FILE = "summarize_file"
    COMPARE_FILES = "compare_files"
    EXPLAIN_CONTENT = "explain_content"
    DRAFT_EDIT = "draft_edit"
    CLASSIFY = "classify"
    FOLLOWUP_QUESTION = "followup_question"
    FOLLOWUP_REFINE = "followup_refine"
    CONTINUE_PREVIOUS_RESULT = "continue_previous_result"
    SOFT_CONFIRM = "soft_confirm"
    SELECT_PREVIOUS_CANDIDATE = "select_previous_candidate"
    NEXT_CANDIDATE = "next_candidate"
    REDUCE_SCOPE = "reduce_scope"
    LIGHTWEIGHT_ACTION_REQUEST = "lightweight_action_request"
    OPEN_FILE = "open_file"


class ParsedEntities(BaseModel):
    file_names: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    topics: list[str] = Field(default_factory=list)
    projects: list[str] = Field(default_factory=list)


class ParsedTimeFilters(BaseModel):
    year: int | None = None
    year_from: int | None = None
    year_to: int | None = None
    relative_days: int | None = None


class ParsedWorkspaceFilters(BaseModel):
    included_paths: list[str] = Field(default_factory=list)
    excluded_paths: list[str] = Field(default_factory=list)


class ParsedIntent(BaseModel):
    intent: ReasoningIntent
    entities: ParsedEntities = Field(default_factory=ParsedEntities)
    time_filters: ParsedTimeFilters = Field(default_factory=ParsedTimeFilters)
    workspace_filters: ParsedWorkspaceFilters = Field(default_factory=ParsedWorkspaceFilters)
    confidence: float = 0.5


class FileCandidate(BaseModel):
    doc_id: str
    file_path: str
    score: float
    modified_at: datetime
    category: str = "참고자료"
    tags: list[str] = Field(default_factory=list)


class ChunkCandidate(BaseModel):
    doc_id: str
    chunk_id: str
    file_path: str
    snippet: str
    score: float
    modified_at: datetime
    category: str = "참고자료"
    tags: list[str] = Field(default_factory=list)


class RetrievalBundle(BaseModel):
    file_candidates: list[FileCandidate] = Field(default_factory=list)
    chunk_candidates: list[ChunkCandidate] = Field(default_factory=list)
    applied_filters: ChatFilters = Field(default_factory=ChatFilters)
    rerank_features: dict[str, float] = Field(default_factory=dict)


class LocalPlan(BaseModel):
    plan_type: str
    selected_files: list[str] = Field(default_factory=list)
    selected_chunks: list[str] = Field(default_factory=list)
    response_strategy: str = "direct_answer"
    allowed_actions: list[SuggestedActionKind] = Field(default_factory=list)
    external_reasoning_needed: bool = False


class ExecutionResult(BaseModel):
    result_type: str
    structured_payload: dict[str, Any] = Field(default_factory=dict)
    citations: list[Citation] = Field(default_factory=list)
    tool_logs: list[str] = Field(default_factory=list)
    generated_text: str = ""
    engine_used: LocalEngine | None = None
    used_fallback: bool = False
    runtime_detail: str | None = None


class VerificationResult(BaseModel):
    is_valid: bool = True
    confidence: float = 0.7
    issues: list[str] = Field(default_factory=list)
    ambiguity_level: float = 0.0
    candidate_mode: bool = False


class StructuredResult(BaseModel):
    result_type: str
    summary: str
    details: list[str] = Field(default_factory=list)
    data: dict[str, Any] = Field(default_factory=dict)


class ComposedChatResponseV2(BaseModel):
    response_mode: str = "conversational_direct"
    lead: str
    structured_result: StructuredResult
    citations: list[Citation] = Field(default_factory=list)
    actions: list[SuggestedAction] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    parsed_intent: ParsedIntent
    plan: LocalPlan
    verification: VerificationResult
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
    adaptive_personalization_enabled: bool = True
    session_memory_enabled: bool = True
    workspace_memory_enabled: bool = True
    local_memory_only: bool = True
    workspace_memory_mode: WorkspaceMemoryMode = WorkspaceMemoryMode.NORMAL


class WorkspaceIdentity(BaseModel):
    workspace_id: str
    included_paths_hash: str
    version: int = 1


class SessionMemoryItem(BaseModel):
    id: str
    session_id: str
    key: str
    value_json: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    updated_at: datetime
    expires_at: datetime | None = None


class WorkspaceMemoryItem(BaseModel):
    id: str
    workspace_id: str
    memory_type: str
    key: str
    value_json: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.62
    source: Literal["explicit", "inferred"] = "inferred"
    created_at: datetime
    updated_at: datetime


class UserPreferenceItem(BaseModel):
    id: str
    key: str
    value_json: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.62
    source: Literal["explicit", "inferred"] = "inferred"
    created_at: datetime
    updated_at: datetime


class EpisodicMemoryEvent(BaseModel):
    id: str
    workspace_id: str | None = None
    event_type: str
    summary: str
    related_file_ids: list[str] = Field(default_factory=list)
    related_action_ids: list[str] = Field(default_factory=list)
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    importance: float = 0.5
    created_at: datetime


class PinnedMemoryItem(BaseModel):
    id: str
    scope: Literal["global", "workspace"] = "global"
    workspace_id: str | None = None
    title: str
    content: str
    created_at: datetime
    updated_at: datetime


class RelevantMemoryBundle(BaseModel):
    workspace_identity: WorkspaceIdentity | None = None
    session_items: list[SessionMemoryItem] = Field(default_factory=list)
    workspace_items: list[WorkspaceMemoryItem] = Field(default_factory=list)
    preference_items: list[UserPreferenceItem] = Field(default_factory=list)
    episodic_items: list[EpisodicMemoryEvent] = Field(default_factory=list)
    pinned_items: list[PinnedMemoryItem] = Field(default_factory=list)


class MemoryEventType(str, Enum):
    QUERY = "query"
    FILE_DISCOVERY = "file_discovery"
    COMPARISON = "comparison"
    SUMMARY_CREATED = "summary_created"
    DRAFT_CREATED = "draft_created"
    EXTERNAL_ANALYSIS = "external_analysis"
    MANUAL_OVERRIDE = "manual_override"
    ACTION_EXECUTED = "action_executed"


class MemoryEventRequest(BaseModel):
    event_type: MemoryEventType
    session_id: str | None = None
    workspace_id: str | None = None
    summary: str
    related_file_ids: list[str] = Field(default_factory=list)
    related_action_ids: list[str] = Field(default_factory=list)
    metadata_json: dict[str, Any] = Field(default_factory=dict)
    importance: float = 0.5


class MemoryEventResponse(BaseModel):
    event_id: str
    accepted: bool = True


class MemoryClearScope(str, Enum):
    ALL = "all"
    WORKSPACE = "workspace"
    SESSION = "session"
    INFERRED_ONLY = "inferred_only"
    EPISODIC = "episodic"


class MemoryClearRequest(BaseModel):
    scope: MemoryClearScope
    workspace_id: str | None = None
    session_id: str | None = None


class MemoryClearResponse(BaseModel):
    cleared_rows: int = 0
    scope: MemoryClearScope


class MemoryPinRequest(BaseModel):
    memory_id: str | None = None
    scope: Literal["global", "workspace"] = "global"
    workspace_id: str | None = None
    title: str | None = None
    content: str | None = None


class MemoryPinResponse(BaseModel):
    item: PinnedMemoryItem


class SessionMemoryResponse(BaseModel):
    items: list[SessionMemoryItem] = Field(default_factory=list)


class WorkspaceMemoryResponse(BaseModel):
    items: list[WorkspaceMemoryItem] = Field(default_factory=list)


class UserPreferencesResponse(BaseModel):
    items: list[UserPreferenceItem] = Field(default_factory=list)


class EpisodicMemoryResponse(BaseModel):
    items: list[EpisodicMemoryEvent] = Field(default_factory=list)


class PinnedMemoryResponse(BaseModel):
    items: list[PinnedMemoryItem] = Field(default_factory=list)


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
