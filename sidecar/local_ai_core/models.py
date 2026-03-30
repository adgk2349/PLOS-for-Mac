from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator
from .settings.normalize import normalize_language


class PrivacyMode(str, Enum):
    LOCAL_ONLY = "LOCAL_ONLY"
    HYBRID = "HYBRID"
    CONFIRM_BEFORE_EXTERNAL = "CONFIRM_BEFORE_EXTERNAL"
    EXTERNAL_ALLOWED = "EXTERNAL_ALLOWED"


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


class SystemFilePermission(str, Enum):
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    FULL_ACCESS = "full_access"


class WorkspaceMemoryMode(str, Enum):
    NORMAL = "normal"
    DISABLED = "disabled"
    PINNED_ONLY = "pinned_only"


class ExtensionCapability(str, Enum):
    RETRIEVER_SEARCH = "retriever.search"
    RERANKER_RANK = "reranker.rank"
    SUMMARIZER_GENERATE = "summarizer.generate"
    RETRIEVAL_QUERY_TRANSFORM = "retrieval.query_transform"
    RETRIEVAL_POST_FILTER = "retrieval.post_filter"
    CHUNKING_STRATEGY = "chunking.strategy"
    EMBEDDING_PROVIDER = "embedding.provider"
    INDEXING_PREPROCESS = "indexing.preprocess"
    FINETUNE_JOB_SUBMIT = "finetune.job_submit"
    FINETUNE_JOB_STATUS = "finetune.job_status"
    FINETUNE_MODEL_PUBLISH = "finetune.model_publish"


class PluginBuildTarget(str, Enum):
    COMMUNITY = "community"
    ENTERPRISE = "enterprise"
    BOTH = "both"


class PluginCapabilitySource(str, Enum):
    BUILT_IN = "built_in"
    DISABLED = "disabled"


class PluginErrorCode(str, Enum):
    PLUGIN_TIMEOUT = "PLUGIN_TIMEOUT"
    PLUGIN_UNAVAILABLE = "PLUGIN_UNAVAILABLE"
    PLUGIN_VALIDATION_ERROR = "PLUGIN_VALIDATION_ERROR"
    PLUGIN_PERMISSION_DENIED = "PLUGIN_PERMISSION_DENIED"


class PluginPrivacyMode(str, Enum):
    LOCAL_ONLY = "LOCAL_ONLY"
    HYBRID = "HYBRID"
    EXTERNAL_ALLOWED = "EXTERNAL_ALLOWED"


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
    reliability: float = 1.0


class AgentAction(BaseModel):
    kind: Literal["spotlight_search", "get_metadata", "execute_command", "final_answer"]
    params: dict[str, Any] = Field(default_factory=dict)
    thought: str = ""


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
    preferred_response_length: ResponseLength = ResponseLength.LONG


class BehaviorOverrides(BaseModel):
    workspace_weights: dict[str, float] | None = None
    preferred_mode: WorkMode | None = None
    preferred_action_order: list[SuggestedActionKind] | None = None
    preferred_response_length: ResponseLength | None = None


class LocalChatRequestV2(LocalChatRequest):
    behavior_overrides: BehaviorOverrides | None = None
    session_id: str | None = None
    included_paths: list[str] | None = None
    excluded_paths: list[str] | None = None
    development_action: Literal["review", "fix"] | None = None
    fix_mode: Literal["plan_only", "apply_patch"] | None = None


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
    SYSTEM_ACTION = "system_action"


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
    operation: Literal["chat", "find", "summarize", "open"] = "chat"
    target: str | None = None
    scope: Literal["single", "top_n", "all"] = "single"
    ambiguity: Literal["clear", "unclear"] = "clear"


class FileCandidate(BaseModel):
    doc_id: str
    file_path: str
    score: float
    modified_at: datetime
    category: str = "참고자료"
    tags: list[str] = Field(default_factory=list)
    reliability: float = 1.0


class ChunkCandidate(BaseModel):
    doc_id: str
    chunk_id: str
    file_path: str
    snippet: str
    score: float
    modified_at: datetime
    category: str = "참고자료"
    tags: list[str] = Field(default_factory=list)
    reliability: float = 1.0


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
    result_type: Literal[
        "answer",
        "summary",
        "conversation",
        "comparison",
        "classification",
        "file_list",
        "insufficient",
        "candidate",
        "agent_action",
    ]
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
    reliability: float = 1.0
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
    execution_result: ExecutionResult | None = None
    generated_text: str = ""
    citations: list[Citation] = Field(default_factory=list)
    actions: list[SuggestedAction] = Field(default_factory=list)
    prompt_cache_hit: bool = False
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
    hybrid_web_search_enabled: bool = False
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
    system_file_permission: SystemFilePermission = SystemFilePermission.READ_ONLY
    searxng_url: str | None = None
    auto_start_searxng: bool = False

    @field_validator("language", mode="before")
    @classmethod
    def _normalize_language(cls, value: str | None) -> str:
        return normalize_language(value)


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


class WebMemorySource(BaseModel):
    title: str = ""
    url: str
    snippet: str = ""


class WebMemoryEntry(BaseModel):
    entry_id: str
    query: str
    answer_summary: str
    sources: list[WebMemorySource] = Field(default_factory=list)
    source_count: int = 0
    confidence: float = 0.0
    created_at: str
    conversation_path: str = ""
    vector_memory_id: str | None = None


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
    semantic_memories: list[dict[str, Any]] = Field(default_factory=list)


class MemoryEventType(str, Enum):
    QUERY = "query"
    ANSWER = "answer"
    SUMMARY = "summary"
    FILE_LIST = "file_list"
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
    total_bytes: int | None = None
    progress_percent: float | None = None
    download_id: str | None = None


class DownloadProgressKind(str, Enum):
    CATALOG = "catalog"
    DIRECT = "direct"


class DownloadProgressStatus(str, Enum):
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class DownloadProgressItem(BaseModel):
    download_id: str
    kind: DownloadProgressKind
    status: DownloadProgressStatus = DownloadProgressStatus.RUNNING
    model_id: str | None = None
    engine: LocalEngine | None = None
    file_name: str | None = None
    downloaded_bytes: int = 0
    total_bytes: int | None = None
    progress_percent: float | None = None
    detail: str | None = None
    error: str | None = None
    updated_at: datetime


class DownloadProgressResponse(BaseModel):
    items: list[DownloadProgressItem] = Field(default_factory=list)


class ModelListItem(BaseModel):
    file_name: str
    path: str
    engine: LocalEngine
    size_bytes: int
    modified_at: datetime
    loaded: bool = False
    resident_engine: LocalEngine | None = None
    last_used_at: datetime | None = None


class ModelListResponse(BaseModel):
    models: list[ModelListItem] = Field(default_factory=list)


class ModelResidencyPolicy(BaseModel):
    allow_dual_resident: bool = False
    max_resident_models: int = 1
    memory_guard_threshold: float = 0.90


class RoomRoutingResult(BaseModel):
    backend: Literal["room", "global"] = "global"
    room_storage_id: str | None = None
    room_scope_hash: str | None = None
    room_route_reason: str | None = None


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


class PluginManifestV1(BaseModel):
    plugin_id: str
    version: str
    api_version: Literal["v1"] = "v1"
    capabilities: list[ExtensionCapability] = Field(default_factory=list)
    privacy_mode: PluginPrivacyMode
    permissions: list[str] = Field(default_factory=list)
    entrypoint: str
    signature: str | None = None
    build_target: PluginBuildTarget = PluginBuildTarget.BOTH

    @field_validator("plugin_id")
    @classmethod
    def _validate_plugin_id(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("plugin_id must not be empty")
        return text

    @field_validator("capabilities")
    @classmethod
    def _validate_capabilities(cls, values: list[ExtensionCapability]) -> list[ExtensionCapability]:
        deduped = list(dict.fromkeys(values))
        if len(deduped) != len(values):
            raise ValueError("capabilities must not contain duplicates")
        return deduped

    @field_validator("entrypoint")
    @classmethod
    def _validate_entrypoint(cls, value: str) -> str:
        text = value.strip()
        if not text:
            raise ValueError("entrypoint must not be empty")
        return text


class CapabilityRequestMeta(BaseModel):
    request_id: str
    timeout_ms: int = 5000
    workspace_id: str | None = None
    session_id: str | None = None
    trace_id: str | None = None


class CapabilityRequestEnvelope(BaseModel):
    capability: ExtensionCapability
    meta: CapabilityRequestMeta
    payload: dict[str, Any] = Field(default_factory=dict)


class CapabilityResponseEnvelope(BaseModel):
    capability: ExtensionCapability
    source: PluginCapabilitySource
    payload: dict[str, Any] = Field(default_factory=dict)
    plugin_id: str | None = None
    error_code: PluginErrorCode | None = None
    error_message: str | None = None
    blocked_reason: str | None = None
    trace_id: str | None = None


class PluginCapabilityEnvelope(BaseModel):
    capability: ExtensionCapability
    plugin_id: str | None = None
    source: PluginCapabilitySource
    effective_privacy_mode: PluginPrivacyMode | None = None
    blocked_reason: str | None = None
    error_code: PluginErrorCode | None = None
    error_message: str | None = None
    trace_id: str | None = None


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
    repo_url: str | None = None
    filename: str | None = None
    allow_patterns: list[str] = Field(default_factory=list)
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
    progress_percent: float | None = None
    downloaded_bytes: int | None = None
    total_bytes: int | None = None


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


class ExtensionCapabilityState(BaseModel):
    capability: ExtensionCapability
    source: PluginCapabilitySource
    plugin_enabled: bool = False
    plugin_id: str | None = None
    error_code: PluginErrorCode | None = None
    plugin_privacy_mode: PluginPrivacyMode | None = None
    effective_privacy_mode: PluginPrivacyMode | None = None
    blocked_reason: str | None = None


class ExtensionCapabilitiesResponse(BaseModel):
    version: int = 1
    capabilities: list[ExtensionCapabilityState] = Field(default_factory=list)


class PluginRegistryEntry(BaseModel):
    plugin_id: str
    manifest: PluginManifestV1
    enabled: bool = False
    state: str = "disabled"
    updated_at: datetime
    validation_error: str | None = None
    is_builtin: bool = False


class PluginRegistryResponse(BaseModel):
    entries: list[PluginRegistryEntry] = Field(default_factory=list)


class PluginRegisterRequest(BaseModel):
    manifest: PluginManifestV1
    enabled: bool = False


class PluginEnableResponse(BaseModel):
    plugin: PluginRegistryEntry
    capabilities: list[ExtensionCapabilityState] = Field(default_factory=list)


class FinetuneJobState(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class FinetuneJobSubmitRequest(BaseModel):
    plugin_id: str
    job_name: str
    dataset_uri: str
    base_model: str
    params: dict[str, Any] = Field(default_factory=dict)


class FinetuneJobSubmitResponse(BaseModel):
    job_id: str
    plugin_id: str
    state: FinetuneJobState = FinetuneJobState.QUEUED
    created_at: datetime


class FinetuneJobStatusResponse(BaseModel):
    job_id: str
    plugin_id: str
    state: FinetuneJobState
    detail: str = ""
    updated_at: datetime
    metrics: dict[str, Any] = Field(default_factory=dict)


class FinetuneModelPublishRequest(BaseModel):
    job_id: str
    target_model_id: str
    artifact_uri: str


class FinetuneModelPublishResponse(BaseModel):
    ok: bool = True
    job_id: str
    plugin_id: str
    target_model_id: str
    published_at: datetime
