import Foundation

// MARK: - API Models
enum APIBaseModelsSplitMarker {}

enum JSONValue: Codable, Hashable {
    case string(String)
    case number(Double)
    case bool(Bool)
    case object([String: JSONValue])
    case array([JSONValue])
    case null

    init(from decoder: Decoder) throws {
        let container = try decoder.singleValueContainer()
        if container.decodeNil() {
            self = .null
        } else if let value = try? container.decode(Bool.self) {
            self = .bool(value)
        } else if let value = try? container.decode(Double.self) {
            self = .number(value)
        } else if let value = try? container.decode(String.self) {
            self = .string(value)
        } else if let value = try? container.decode([String: JSONValue].self) {
            self = .object(value)
        } else if let value = try? container.decode([JSONValue].self) {
            self = .array(value)
        } else {
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Unsupported JSON value")
        }
    }

    func encode(to encoder: Encoder) throws {
        var container = encoder.singleValueContainer()
        switch self {
        case let .string(value):
            try container.encode(value)
        case let .number(value):
            try container.encode(value)
        case let .bool(value):
            try container.encode(value)
        case let .object(value):
            try container.encode(value)
        case let .array(value):
            try container.encode(value)
        case .null:
            try container.encodeNil()
        }
    }
}

extension JSONValue {
    var boolValue: Bool? {
        if case let .bool(value) = self {
            return value
        }
        return nil
    }

    var numberValue: Double? {
        if case let .number(value) = self {
            return value
        }
        return nil
    }

    var stringValue: String? {
        if case let .string(value) = self {
            return value
        }
        return nil
    }

    var objectValue: [String: JSONValue]? {
        if case let .object(value) = self {
            return value
        }
        return nil
    }

    var arrayValue: [JSONValue]? {
        if case let .array(value) = self {
            return value
        }
        return nil
    }

    var boolCoercedValue: Bool? {
        if let value = boolValue {
            return value
        }
        if let value = numberValue {
            return value != 0
        }
        guard let raw = stringValue?.trimmingCharacters(in: .whitespacesAndNewlines).lowercased(), !raw.isEmpty else {
            return nil
        }
        switch raw {
        case "true", "1", "yes", "y", "on":
            return true
        case "false", "0", "no", "n", "off":
            return false
        default:
            return nil
        }
    }
}

enum PrivacyMode: String, CaseIterable, Codable, Identifiable {
    case localOnly = "LOCAL_ONLY"
    case hybrid = "HYBRID"
    case externalAllowed = "EXTERNAL_ALLOWED"
    case confirmBeforeExternal = "CONFIRM_BEFORE_EXTERNAL"

    var id: String { rawValue }
    static var allCases: [PrivacyMode] { [.localOnly, .hybrid, .externalAllowed] }

    func title(language: AppLanguage) -> String {
        switch self {
        case .localOnly:
            return L10n.tr("privacy_mode.local_only", language: language, fallbackKo: "완전 로컬", fallbackEn: "Local only", fallbackJa: "ローカルのみ")
        case .hybrid:
            return L10n.tr("privacy_mode.hybrid", language: language, fallbackKo: "하이브리드", fallbackEn: "Hybrid", fallbackJa: "ハイブリッド")
        case .externalAllowed:
            return L10n.tr("privacy_mode.external_allowed", language: language, fallbackKo: "외부 호출 허용", fallbackEn: "External allowed", fallbackJa: "外部呼び出し許可")
        case .confirmBeforeExternal:
            return L10n.tr("privacy_mode.confirm_before_external", language: language, fallbackKo: "확인 후 외부 호출", fallbackEn: "Confirm before external", fallbackJa: "外部呼び出し前に確認")
        }
    }

    var title: String { title(language: L10n.loadSelection()) }
}

enum WorkMode: String, CaseIterable, Codable, Identifiable {
    case general = "GENERAL"
    case summary = "SUMMARY"
    case research = "RESEARCH"
    case development = "DEVELOPMENT"
    case writing = "WRITING"
    case planning = "PLANNING"
    case strictSearch = "STRICT_SEARCH"

    var id: String { rawValue }

    func title(language: AppLanguage) -> String {
        switch self {
        case .general:
            return L10n.tr("work_mode.general", language: language, fallbackKo: "일반", fallbackEn: "General", fallbackJa: "一般")
        case .summary:
            return L10n.tr("work_mode.summary", language: language, fallbackKo: "요약", fallbackEn: "Summary", fallbackJa: "要約")
        case .research:
            return L10n.tr("work_mode.research", language: language, fallbackKo: "연구", fallbackEn: "Research", fallbackJa: "リサーチ")
        case .development:
            return L10n.tr("work_mode.development", language: language, fallbackKo: "개발", fallbackEn: "Development", fallbackJa: "開発")
        case .writing:
            return L10n.tr("work_mode.writing", language: language, fallbackKo: "글쓰기", fallbackEn: "Writing", fallbackJa: "執筆")
        case .planning:
            return L10n.tr("work_mode.planning", language: language, fallbackKo: "기획", fallbackEn: "Planning", fallbackJa: "計画")
        case .strictSearch:
            return L10n.tr("work_mode.strict_search", language: language, fallbackKo: "엄격 검색", fallbackEn: "Strict search", fallbackJa: "厳格検索")
        }
    }

    var title: String { title(language: L10n.loadSelection()) }
}

enum StartupProfile: String, CaseIterable, Codable, Identifiable {
    case fast = "FAST"
    case recommended = "RECOMMENDED"
    case deep = "DEEP"

    var id: String { rawValue }

    func title(language: AppLanguage) -> String {
        switch self {
        case .fast:
            return L10n.tr("startup_profile.fast", language: language, fallbackKo: "빠르게 시작", fallbackEn: "Fast start", fallbackJa: "高速開始")
        case .recommended:
            return L10n.tr("startup_profile.recommended", language: language, fallbackKo: "추천 설정", fallbackEn: "Recommended", fallbackJa: "おすすめ設定")
        case .deep:
            return L10n.tr("startup_profile.deep", language: language, fallbackKo: "깊은 분석", fallbackEn: "Deep analysis", fallbackJa: "深い分析")
        }
    }

    var title: String { title(language: L10n.loadSelection()) }
}

enum LocalEngine: String, CaseIterable, Codable, Identifiable {
    case mlx = "mlx"
    case llamaCPP = "llama_cpp"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .mlx:
            return "MLX"
        case .llamaCPP:
            return "llama.cpp"
        }
    }
}

enum ActionPermissionMode: String, CaseIterable, Codable, Identifiable {
    case askPerAction = "ASK_PER_ACTION"
    case askEveryTime = "ASK_EVERY_TIME"

    var id: String { rawValue }

    func title(language: AppLanguage) -> String {
        switch self {
        case .askPerAction:
            return L10n.tr("action_permission.ask_per_action", language: language, fallbackKo: "행동별 1회 승인", fallbackEn: "Approve per action", fallbackJa: "操作ごとに承認")
        case .askEveryTime:
            return L10n.tr("action_permission.ask_every_time", language: language, fallbackKo: "매번 승인", fallbackEn: "Approve every time", fallbackJa: "毎回承認")
        }
    }

    var title: String { title(language: L10n.loadSelection()) }
}

enum SystemFilePermission: String, CaseIterable, Codable, Identifiable {
    case readOnly = "read_only"
    case readWrite = "read_write"
    case fullAccess = "full_access"

    var id: String { rawValue }

    func title(language: AppLanguage) -> String {
        switch self {
        case .readOnly:
            return L10n.tr("system_file_permission.read_only", language: language, fallbackKo: "읽기 전용", fallbackEn: "Read only", fallbackJa: "読み取り専用")
        case .readWrite:
            return L10n.tr("system_file_permission.read_write", language: language, fallbackKo: "읽기/쓰기", fallbackEn: "Read/write", fallbackJa: "読み書き")
        case .fullAccess:
            return L10n.tr("system_file_permission.full_access", language: language, fallbackKo: "전체 접근", fallbackEn: "Full access", fallbackJa: "フルアクセス")
        }
    }

    var title: String { title(language: L10n.loadSelection()) }
}

enum WorkspaceMemoryMode: String, CaseIterable, Codable, Identifiable {
    case normal = "normal"
    case disabled = "disabled"
    case pinnedOnly = "pinned_only"

    var id: String { rawValue }

    func title(language: AppLanguage) -> String {
        switch self {
        case .normal:
            return L10n.tr("workspace_memory_mode.normal", language: language, fallbackKo: "일반", fallbackEn: "Normal", fallbackJa: "通常")
        case .disabled:
            return L10n.tr("workspace_memory_mode.disabled", language: language, fallbackKo: "비활성화", fallbackEn: "Disabled", fallbackJa: "無効")
        case .pinnedOnly:
            return L10n.tr("workspace_memory_mode.pinned_only", language: language, fallbackKo: "고정 메모리만 사용", fallbackEn: "Pinned only", fallbackJa: "固定メモリのみ")
        }
    }

    var title: String { title(language: L10n.loadSelection()) }
}

enum ChatIntent: String, Codable {
    case fileSearch = "FILE_SEARCH"
    case documentQA = "DOCUMENT_QA"
    case taskRequest = "TASK_REQUEST"
    case ambiguous = "AMBIGUOUS"
}

enum SuggestedActionKind: String, Codable {
    case openFile = "OPEN_FILE"
    case summarizeTop = "SUMMARIZE_TOP"
    case compareTop = "COMPARE_TOP"
    case askFollowup = "ASK_FOLLOWUP"
    case showDiff = "SHOW_DIFF"
    case createDraft = "CREATE_DRAFT"
    case showOtherCandidates = "SHOW_OTHER_CANDIDATES"
    case makeShorter = "MAKE_SHORTER"
    case openSecond = "OPEN_SECOND"
    case showPreviousCandidate = "SHOW_PREVIOUS_CANDIDATE"
}

enum ActionExecutionMode: String, Codable {
    case promptInjection = "PROMPT_INJECTION"
    case system = "SYSTEM"
}

enum MemoryEventType: String, CaseIterable, Codable, Identifiable {
    case query = "query"
    case fileDiscovery = "file_discovery"
    case comparison = "comparison"
    case summaryCreated = "summary_created"
    case draftCreated = "draft_created"
    case externalAnalysis = "external_analysis"
    case manualOverride = "manual_override"
    case actionExecuted = "action_executed"

    var id: String { rawValue }
}

enum MemoryClearScope: String, CaseIterable, Codable, Identifiable {
    case all = "all"
    case workspace = "workspace"
    case session = "session"
    case inferredOnly = "inferred_only"
    case episodic = "episodic"

    var id: String { rawValue }
}

enum ModelInstallStatus: String, Codable {
    case notInstalled = "not_installed"
    case downloading = "downloading"
    case installed = "installed"
    case active = "active"
    case failed = "failed"

    func title(language: AppLanguage) -> String {
        switch self {
        case .notInstalled:
            return L10n.tr("model_install_status.not_installed", language: language, fallbackKo: "미설치", fallbackEn: "Not installed", fallbackJa: "未インストール")
        case .downloading:
            return L10n.tr("model_install_status.downloading", language: language, fallbackKo: "다운로드 중", fallbackEn: "Downloading", fallbackJa: "ダウンロード中")
        case .installed:
            return L10n.tr("model_install_status.installed", language: language, fallbackKo: "설치됨", fallbackEn: "Installed", fallbackJa: "インストール済み")
        case .active:
            return L10n.tr("model_install_status.active", language: language, fallbackKo: "사용 중", fallbackEn: "Active", fallbackJa: "使用中")
        case .failed:
            return L10n.tr("model_install_status.failed", language: language, fallbackKo: "실패", fallbackEn: "Failed", fallbackJa: "失敗")
        }
    }

    var title: String { title(language: L10n.loadSelection()) }
}

struct WorkspaceUpdateRequest: Codable {
    var included_paths: [String]
    var excluded_paths: [String]
    var startup_profile: StartupProfile
    var default_mode: WorkMode
}

struct WorkspaceResponse: Codable {
    var included_paths: [String]
    var excluded_paths: [String]
    var startup_profile: StartupProfile
    var default_mode: WorkMode
    var updated_at: Date
}

struct IndexJobRequest: Codable {
    var scope: String
}

struct IndexJobStartResponse: Codable {
    var job_id: String
}

struct IndexJobStatus: Codable {
    var job_id: String
    var scope: String
    var status: String
    var progress: Double
    var processed_files: Int
    var failed_files: Int
    var stage: String
    var error: String?
}

struct RoomStorageVariantStatus: Codable {
    var room_storage_id: String
    var scope_hash: String
    var data_dir: String
    var indexed_docs: Int
    var chunk_count: Int
    var session_memory_count: Int
    var workspace_memory_count: Int
    var bytes_used: Int
    var room_index_state: String
    var index_progress: Double?
    var index_stage: String?
    var processed_files: Int?
    var failed_files: Int?
    var job_id: String?
    var job_status: String?
}

struct RoomStorageStatusResponse: Codable {
    var room_id: String
    var room_key: String
    var variant_count: Int
    var total_bytes: Int
    var variants: [RoomStorageVariantStatus]
}

struct RoomStorageReindexRequest: Codable {
    var scope: String
    var included_paths: [String]?
    var excluded_paths: [String]?
}

struct RoomStorageReindexResponse: Codable {
    var ok: Bool
    var room_storage_id: String
    var room_scope_hash: String
    var room_index_state: String
    var job_id: String?
    var index_progress: Double?
    var index_stage: String?
    var processed_files: Int?
    var failed_files: Int?
    var job_status: String?
    var started: Bool
}

struct Citation: Codable, Identifiable, Hashable {
    var doc_id: String
    var chunk_id: String
    var file_path: String
    var snippet: String
    var score: Double
    var modified_at: Date
    var category: String = "참고자료"
    var subcategory: String = ""
    var tags: [String] = []
    var document_type: String = ""
    var importance: Double = 0.5
    var reliability: Double = 1.0

    var id: String { chunk_id }
}

struct ChatFilters: Codable, Hashable {
    var category: String?
    var tags: [String]
    var year: Int?
    var project: String?
    var excluded: Bool?
}

struct LocalChatRequest: Codable {
    var query: String
    var mode: WorkMode
    var conversation_id: String?
    var top_k: Int?
    var filters: ChatFilters?
}

enum ResponseLength: String, Codable {
    case short = "short"
    case medium = "medium"
    case long = "long"
}

struct BehaviorOverrides: Codable {
    var workspace_weights: [String: Double]?
    var preferred_mode: WorkMode?
    var preferred_action_order: [SuggestedActionKind]?
    var preferred_response_length: ResponseLength?
}

struct LocalChatRequestV2: Codable {
    var query: String
    var mode: WorkMode
    var conversation_id: String?
    var session_id: String?
    var top_k: Int?
    var filters: ChatFilters?
    var included_paths: [String]?
    var excluded_paths: [String]?
    var behavior_overrides: BehaviorOverrides?
    var development_action: String? = nil
    var fix_mode: String? = nil
}

struct SuggestedAction: Codable, Identifiable, Hashable {
    var action_id: String
    var kind: SuggestedActionKind
    var label: String
    var execution_mode: ActionExecutionMode
    var payload: [String: String]

    var id: String { action_id }
}

struct LocalChatResponse: Codable {
    var intent: ChatIntent
    var lead: String
    var result_summary: String
    var citations: [Citation]
    var actions: [SuggestedAction]
    var reasoning_brief: String?
    var mode: WorkMode
    var used_profile: StartupProfile
    var is_local: Bool
    var engine_used: LocalEngine?
    var used_fallback: Bool?
    var runtime_detail: String?
}

enum ReasoningIntent: String, Codable {
    case generalChat = "general_chat"
    case findFile = "find_file"
    case summarizeFile = "summarize_file"
    case compareFiles = "compare_files"
    case explainContent = "explain_content"
    case draftEdit = "draft_edit"
    case classify = "classify"
    case followupQuestion = "followup_question"
    case followupRefine = "followup_refine"
    case continuePreviousResult = "continue_previous_result"
    case softConfirm = "soft_confirm"
    case selectPreviousCandidate = "select_previous_candidate"
    case nextCandidate = "next_candidate"
    case reduceScope = "reduce_scope"
    case lightweightActionRequest = "lightweight_action_request"
    case openFile = "open_file"
}

struct ParsedEntities: Codable {
    var file_names: [String]
    var tags: [String]
    var topics: [String]
    var projects: [String]
}

struct ParsedTimeFilters: Codable {
    var year: Int?
    var year_from: Int?
    var year_to: Int?
    var relative_days: Int?
}

struct ParsedWorkspaceFilters: Codable {
    var included_paths: [String]
    var excluded_paths: [String]
}

struct ParsedIntent: Codable {
    var intent: ReasoningIntent
    var entities: ParsedEntities
    var time_filters: ParsedTimeFilters
    var workspace_filters: ParsedWorkspaceFilters
    var confidence: Double
    var operation: String?
    var target: String?
    var scope: String?
    var ambiguity: String?
}

struct LocalPlan: Codable {
    var plan_type: String
    var selected_files: [String]
    var selected_chunks: [String]
    var response_strategy: String
    var allowed_actions: [SuggestedActionKind]
    var external_reasoning_needed: Bool
}

struct VerificationResult: Codable {
    var is_valid: Bool
    var confidence: Double
    var issues: [String]
    var ambiguity_level: Double
    var candidate_mode: Bool
    var reliability: Double = 1.0
}

struct StructuredResult: Codable {
    var result_type: String
    var summary: String
    var details: [String]
    var data: [String: JSONValue]
}

struct ChatStreamEvent: Codable {
    var type: String
    var message: String?
    var text: String?
    var result: ComposedChatResponseV2?
}

struct ComposedChatResponseV2: Codable {
    var response_mode: String?
    var lead: String
    var structured_result: StructuredResult
    var generated_text: String?
    var citations: [Citation]
    var actions: [SuggestedAction]
    var prompt_cache_hit: Bool?
    var metadata: [String: JSONValue]?
    var parsed_intent: ParsedIntent
    var plan: LocalPlan
    var verification: VerificationResult
    var mode: WorkMode
    var used_profile: StartupProfile
    var is_local: Bool
    var engine_used: LocalEngine?
    var used_fallback: Bool?
    var runtime_detail: String?
}

struct DeepAnalysisRequest: Codable {
    var query: String
    var mode: WorkMode
    var provider: String
    var selected_citations: [Citation]
    var user_confirmed: Bool
}

struct ExternalCallEvent: Codable {
    var provider: String
    var sent_chars: Int
    var approved_by_user: Bool
    var timestamp: Date
}

struct DeepAnalysisResponse: Codable {
    var answer: String
    var provider: String
    var event: ExternalCallEvent
    var is_local: Bool
}

struct SettingsModel: Codable {
    var privacy_mode: PrivacyMode = .hybrid
    var hybrid_web_search_enabled: Bool = false
    var system_file_permission: SystemFilePermission = .readOnly
    var startup_profile: StartupProfile = .recommended
    var model_profile: String = "balanced"
    var local_engine: LocalEngine?
    var mlx_model_path: String?
    var llama_model_path: String?
    var reindex_policy: String = "filewatch_incremental"
    var language: String = "auto"
    var action_permission_mode: ActionPermissionMode?
    var adaptive_personalization_enabled: Bool = true
    var session_memory_enabled: Bool = true
    var workspace_memory_enabled: Bool = true
    var local_memory_only: Bool = true
    var workspace_memory_mode: WorkspaceMemoryMode = .normal
    var searxng_url: String?
    var auto_start_searxng: Bool = false
}

struct WorkspaceIdentity: Codable {
    var workspace_id: String
    var included_paths_hash: String
    var version: Int
}

struct SessionMemoryItem: Codable, Identifiable {
    var id: String
    var session_id: String
    var key: String
    var value_json: [String: JSONValue]
    var created_at: Date
    var updated_at: Date
    var expires_at: Date?
}

struct WorkspaceMemoryItem: Codable, Identifiable {
    var id: String
    var workspace_id: String
    var memory_type: String
    var key: String
    var value_json: [String: JSONValue]
    var confidence: Double
    var source: String
    var created_at: Date
    var updated_at: Date
}

struct UserPreferenceItem: Codable, Identifiable {
    var id: String
    var key: String
    var value_json: [String: JSONValue]
    var confidence: Double
    var source: String
    var created_at: Date
    var updated_at: Date
}

struct EpisodicMemoryEvent: Codable, Identifiable {
    var id: String
    var workspace_id: String?
    var event_type: String
    var summary: String
    var related_file_ids: [String]
    var related_action_ids: [String]
    var metadata_json: [String: JSONValue]
    var importance: Double
    var created_at: Date
}

struct PinnedMemoryItem: Codable, Identifiable {
    var id: String
    var scope: String
    var workspace_id: String?
    var title: String
    var content: String
    var created_at: Date
    var updated_at: Date
}

struct SessionMemoryResponse: Codable {
    var items: [SessionMemoryItem]
}

struct WorkspaceMemoryResponse: Codable {
    var items: [WorkspaceMemoryItem]
}

struct UserPreferencesResponse: Codable {
    var items: [UserPreferenceItem]
}

struct EpisodicMemoryResponse: Codable {
    var items: [EpisodicMemoryEvent]
}

struct PinnedMemoryResponse: Codable {
    var items: [PinnedMemoryItem]
}

struct MemoryEventRequest: Codable {
    var event_type: MemoryEventType
    var session_id: String?
    var workspace_id: String?
    var summary: String
    var related_file_ids: [String]
    var related_action_ids: [String]
    var metadata_json: [String: JSONValue]
    var importance: Double
}

struct MemoryEventResponse: Codable {
    var event_id: String
    var accepted: Bool
}

struct MemoryClearRequest: Codable {
    var scope: MemoryClearScope
    var workspace_id: String?
    var session_id: String?
}

struct MemoryClearResponse: Codable {
    var cleared_rows: Int
    var scope: MemoryClearScope
}

struct MemoryPinRequest: Codable {
    var memory_id: String?
    var scope: String
    var workspace_id: String?
    var title: String?
    var content: String?
}

struct MemoryPinResponse: Codable {
    var item: PinnedMemoryItem
}

struct FailureItem: Codable, Identifiable {
    var path: String
    var reason: String
    var last_attempt_at: Date

    var id: String { path }
}

struct FailureListResponse: Codable {
    var failures: [FailureItem]
}

struct StatusSnapshot: Codable {
    struct LatestExternalCall: Codable {
        var provider: String
        var timestamp: String
    }

    var indexed_docs: Int
    var last_indexed_at: String?
    var latest_external_call: LatestExternalCall?
    var included_paths: [String]
    var privacy_mode: PrivacyMode
}

struct DocumentMetadata: Codable, Identifiable, Hashable {
    var doc_id: String
    var path: String
    var file_type: String
    var modified_at: Date
    var indexed_at: Date
    var summary: String
    var category: String
    var subcategory: String
    var document_type: String
    var tags: [String]
    var year: Int?
    var project: String?
    var importance: Double
    var excluded: Bool

    var id: String { doc_id }
}

struct DocumentListResponse: Codable {
    var documents: [DocumentMetadata]
    var total: Int
    var offset: Int
    var limit: Int
}

struct DocumentMetadataUpdateRequest: Codable {
    var category: String?
    var subcategory: String?
    var document_type: String?
    var tags: [String]?
    var year: Int?
    var project: String?
    var importance: Double?
    var excluded: Bool?
}

struct ModelDownloadRequest: Codable {
    var url: String
    var engine: LocalEngine
    var filename: String?
}

struct ModelDownloadResponse: Codable {
    var file_name: String
    var saved_path: String
    var engine: LocalEngine
    var bytes_written: Int
    var total_bytes: Int?
    var progress_percent: Double?
    var download_id: String?
}

enum DownloadProgressKind: String, Codable {
    case catalog
    case direct
}

enum DownloadProgressStatus: String, Codable {
    case running
    case completed
    case failed
}

struct DownloadProgressItem: Codable, Hashable, Identifiable {
    var download_id: String
    var kind: DownloadProgressKind
    var status: DownloadProgressStatus
    var model_id: String?
    var engine: LocalEngine?
    var file_name: String?
    var downloaded_bytes: Int
    var total_bytes: Int?
    var progress_percent: Double?
    var detail: String?
    var error: String?
    var updated_at: Date

    var id: String { download_id }
}

struct DownloadProgressResponse: Codable {
    var items: [DownloadProgressItem]
}

struct ModelListItem: Codable, Identifiable, Hashable {
    var file_name: String
    var path: String
    var engine: LocalEngine
    var size_bytes: Int
    var modified_at: Date

    var id: String { path }
}

struct ModelListResponse: Codable {
    var models: [ModelListItem]
}

struct RuntimePrepareRequest: Codable {
    var engine: LocalEngine
    var model_path: String?
}

struct RuntimePrepareResponse: Codable {
    var engine: LocalEngine
    var ready: Bool
    var package_available: Bool
    var model_path: String?
    var model_exists: Bool
    var accelerator: String
    var detail: String
}

struct ModelSupportFlags: Codable, Hashable {
    var chat: Bool
    var rag: Bool
    var tool_use: Bool
    var vision: Bool
}

struct ModelCatalogItem: Codable, Identifiable, Hashable {
    var id: String
    var name: String
    var profile: String
    var engine: LocalEngine
    var distribution_type: String
    var repo_id: String
    var filename: String?
    var download_label: String
    var description: String
    var size_gb: Double
    var recommended_for: [String]
    var recommended_memory_gb: Int
    var tags: [String]
    var supports: ModelSupportFlags
    var `default`: Bool
    var status: ModelInstallStatus
    var installed_path: String?
    var active: Bool
    var failure_reason: String?
    var progress_percent: Double?
    var downloaded_bytes: Int?
    var total_bytes: Int?

    var profileTitle: String {
        switch profile {
        case "fast":
            return "빠르게 시작"
        case "balanced":
            return "추천 설정"
        case "advanced":
            return "깊은 분석"
        default:
            return profile
        }
    }
}

struct ModelCatalogResponse: Codable {
    var version: Int
    var default_profile: String
    var models: [ModelCatalogItem]
}

struct ModelCatalogInstallRequest: Codable {
    var model_id: String
}

struct ModelCatalogInstallResponse: Codable {
    var model_id: String
    var status: ModelInstallStatus
    var engine: LocalEngine
    var saved_path: String?
    var detail: String
}

struct ModelCatalogActivateRequest: Codable {
    var model_id: String
}

struct ModelCatalogActivateResponse: Codable {
    var model_id: String
    var engine: LocalEngine
    var model_path: String
    var profile: String
}

struct ModelCatalogDeleteResponse: Codable {
    var model_id: String
    var removed: Bool
}
