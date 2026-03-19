import Foundation

// MARK: - API Models

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
}

enum PrivacyMode: String, CaseIterable, Codable, Identifiable {
    case localOnly = "LOCAL_ONLY"
    case hybrid = "HYBRID"
    case confirmBeforeExternal = "CONFIRM_BEFORE_EXTERNAL"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .localOnly:
            return "완전 로컬"
        case .hybrid:
            return "하이브리드"
        case .confirmBeforeExternal:
            return "확인 후 외부 호출"
        }
    }
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

    var title: String {
        switch self {
        case .general:
            return "일반"
        case .summary:
            return "요약"
        case .research:
            return "연구"
        case .development:
            return "개발"
        case .writing:
            return "글쓰기"
        case .planning:
            return "기획"
        case .strictSearch:
            return "엄격 검색"
        }
    }
}

enum StartupProfile: String, CaseIterable, Codable, Identifiable {
    case fast = "FAST"
    case recommended = "RECOMMENDED"
    case deep = "DEEP"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .fast:
            return "빠르게 시작"
        case .recommended:
            return "추천 설정"
        case .deep:
            return "깊은 분석"
        }
    }
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

    var title: String {
        switch self {
        case .askPerAction:
            return "행동별 1회 승인"
        case .askEveryTime:
            return "매번 승인"
        }
    }
}

enum WorkspaceMemoryMode: String, CaseIterable, Codable, Identifiable {
    case normal = "normal"
    case disabled = "disabled"
    case pinnedOnly = "pinned_only"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .normal:
            return "일반"
        case .disabled:
            return "비활성화"
        case .pinnedOnly:
            return "Pinned만 사용"
        }
    }
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

    var title: String {
        switch self {
        case .notInstalled:
            return "미설치"
        case .downloading:
            return "다운로드 중"
        case .installed:
            return "설치됨"
        case .active:
            return "사용 중"
        case .failed:
            return "실패"
        }
    }
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
    var behavior_overrides: BehaviorOverrides?
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
}

struct StructuredResult: Codable {
    var result_type: String
    var summary: String
    var details: [String]
    var data: [String: JSONValue]
}

struct ComposedChatResponseV2: Codable {
    var response_mode: String?
    var lead: String
    var structured_result: StructuredResult
    var citations: [Citation]
    var actions: [SuggestedAction]
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

struct ChatMessage: Identifiable, Codable {
    enum Source: String, Codable {
        case user
        case local
        case external
    }

    let id: UUID
    let source: Source
    let text: String?
    let intent: ChatIntent?
    let lead: String?
    let resultSummary: String?
    let structuredResult: StructuredResult?
    let responseMetadata: [String: JSONValue]?
    let parsedIntent: ParsedIntent?
    let plan: LocalPlan?
    let verification: VerificationResult?
    let reasoningBrief: String?
    let actions: [SuggestedAction]
    let timestamp: Date

    init(id: UUID = UUID(), source: Source, text: String, timestamp: Date) {
        self.id = id
        self.source = source
        self.text = text
        intent = nil
        lead = nil
        resultSummary = nil
        structuredResult = nil
        responseMetadata = nil
        parsedIntent = nil
        plan = nil
        verification = nil
        reasoningBrief = nil
        actions = []
        self.timestamp = timestamp
    }

    init(id: UUID = UUID(), local response: LocalChatResponse, timestamp: Date) {
        self.id = id
        source = .local
        text = nil
        intent = response.intent
        lead = response.lead
        resultSummary = response.result_summary
        structuredResult = nil
        responseMetadata = nil
        parsedIntent = nil
        plan = nil
        verification = nil
        reasoningBrief = response.reasoning_brief
        actions = response.actions
        self.timestamp = timestamp
    }

    init(id: UUID = UUID(), localV2 response: ComposedChatResponseV2, timestamp: Date) {
        self.id = id
        source = .local
        text = nil
        intent = nil
        lead = response.lead
        resultSummary = response.structured_result.summary
        structuredResult = response.structured_result
        responseMetadata = response.metadata
        parsedIntent = response.parsed_intent
        plan = response.plan
        verification = response.verification
        reasoningBrief = nil
        actions = response.actions
        self.timestamp = timestamp
    }
}

struct ChatRoom: Codable, Identifiable {
    var id: String
    var title: String
    var messages: [ChatMessage]
    var citations: [Citation]
    var latestQueryForDeepAnalysis: String?
    var updatedAt: Date

    static func makeDefault() -> ChatRoom {
        ChatRoom(
            id: UUID().uuidString,
            title: "새 채팅",
            messages: [],
            citations: [],
            latestQueryForDeepAnalysis: nil,
            updatedAt: Date()
        )
    }
}
