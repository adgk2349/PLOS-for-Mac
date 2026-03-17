import Foundation

// MARK: - API Models

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
}

enum ActionExecutionMode: String, Codable {
    case promptInjection = "PROMPT_INJECTION"
    case system = "SYSTEM"
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
    var privacy_mode: PrivacyMode
    var startup_profile: StartupProfile
    var model_profile: String
    var local_engine: LocalEngine?
    var mlx_model_path: String?
    var llama_model_path: String?
    var reindex_policy: String
    var language: String
    var action_permission_mode: ActionPermissionMode?
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

struct ChatMessage: Identifiable {
    enum Source {
        case user
        case local
        case external
    }

    let id = UUID()
    let source: Source
    let text: String?
    let intent: ChatIntent?
    let lead: String?
    let resultSummary: String?
    let reasoningBrief: String?
    let actions: [SuggestedAction]
    let timestamp: Date

    init(source: Source, text: String, timestamp: Date) {
        self.source = source
        self.text = text
        intent = nil
        lead = nil
        resultSummary = nil
        reasoningBrief = nil
        actions = []
        self.timestamp = timestamp
    }

    init(local response: LocalChatResponse, timestamp: Date) {
        source = .local
        text = nil
        intent = response.intent
        lead = response.lead
        resultSummary = response.result_summary
        reasoningBrief = response.reasoning_brief
        actions = response.actions
        self.timestamp = timestamp
    }
}
