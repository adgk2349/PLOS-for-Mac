import Foundation

enum PrivacyMode: String, CaseIterable, Codable, Identifiable {
    case localOnly = "LOCAL_ONLY"
    case hybrid = "HYBRID"
    case confirmBeforeExternal = "CONFIRM_BEFORE_EXTERNAL"

    var id: String { rawValue }

    var title: String {
        switch self {
        case .localOnly: return "완전 로컬"
        case .hybrid: return "하이브리드"
        case .confirmBeforeExternal: return "확인 후 외부 호출"
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
        case .general: return "일반"
        case .summary: return "요약"
        case .research: return "연구"
        case .development: return "개발"
        case .writing: return "글쓰기"
        case .planning: return "기획"
        case .strictSearch: return "엄격 검색"
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
        case .fast: return "빠르게 시작"
        case .recommended: return "추천 설정"
        case .deep: return "깊은 분석"
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

    var id: String { chunk_id }
}

struct LocalChatRequest: Codable {
    var query: String
    var mode: WorkMode
    var conversation_id: String?
    var top_k: Int?
}

struct LocalChatResponse: Codable {
    var answer: String
    var citations: [Citation]
    var mode: WorkMode
    var used_profile: StartupProfile
    var is_local: Bool
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
    var reindex_policy: String
    var language: String
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

struct ChatMessage: Identifiable {
    enum Source {
        case user
        case local
        case external
    }

    let id = UUID()
    let source: Source
    let text: String
    let timestamp: Date
}
