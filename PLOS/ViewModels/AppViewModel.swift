import AppKit
import Combine
import CryptoKit
import Foundation

final class BookmarkStore {
    private let defaultsKey = "local_ai_bookmarks"
    private let defaults = UserDefaults.standard

    struct BookmarkEntry: Codable {
        let path: String
        let bookmarkData: Data
    }

    func loadURLs() -> [URL] {
        guard
            let data = defaults.data(forKey: defaultsKey),
            let entries = try? JSONDecoder().decode([BookmarkEntry].self, from: data)
        else {
            return []
        }

        var urls: [URL] = []
        for entry in entries {
            var stale = false
            guard let resolved = try? URL(
                resolvingBookmarkData: entry.bookmarkData,
                options: [.withSecurityScope],
                relativeTo: nil,
                bookmarkDataIsStale: &stale
            ) else {
                continue
            }
            _ = resolved.startAccessingSecurityScopedResource()
            urls.append(resolved)
        }
        return urls
    }

    func save(urls: [URL]) {
        var entries: [BookmarkEntry] = []
        for url in urls {
            guard let data = try? url.bookmarkData(options: [.withSecurityScope], includingResourceValuesForKeys: nil, relativeTo: nil) else {
                continue
            }
            entries.append(BookmarkEntry(path: url.path, bookmarkData: data))
        }

        guard let encoded = try? JSONEncoder().encode(entries) else {
            return
        }
        defaults.set(encoded, forKey: defaultsKey)
    }
}

// MARK: - View Model

enum OnboardingStep: Int, CaseIterable {
    case welcome
    case dataSelection
    case startProfile
    case privacyInfo
    case indexing
    case ready
}

enum ChatResponseRoute: String, CaseIterable, Identifiable {
    case hybrid
    case localOnly
    case apiOnly

    var id: String { rawValue }

    var title: String {
        switch self {
        case .hybrid:
            return "하이브리드"
        case .localOnly:
            return "로컬"
        case .apiOnly:
            return "항상 API 호출"
        }
    }
}

enum QuickInferencePreset: String, CaseIterable, Identifiable {
    case fast
    case quality
    case highQuality

    var id: String { rawValue }

    var title: String {
        switch self {
        case .fast:
            return "빠른 추론"
        case .quality:
            return "품질"
        case .highQuality:
            return "고품질 추론"
        }
    }

    var detail: String {
        switch self {
        case .fast:
            return "가볍고 빠른 응답"
        case .quality:
            return "속도와 정확도 균형"
        case .highQuality:
            return "응답 품질 우선"
        }
    }

    var startupProfile: StartupProfile {
        switch self {
        case .fast:
            return .fast
        case .quality:
            return .recommended
        case .highQuality:
            return .deep
        }
    }
}

@MainActor
final class AppViewModel: ObservableObject {
    static let fixedCategories = ["학습자료", "프로젝트문서", "회의록", "아이디어", "개인메모", "참고자료", "코드관련"]

    @Published var onboardingStep: OnboardingStep = .welcome
    @Published var hasFinishedOnboarding = false

    @Published var includedFolderURLs: [URL] = []
    @Published var excludedPaths: [String] = []
    @Published var startupProfile: StartupProfile = .recommended
    @Published var privacyMode: PrivacyMode = .hybrid
    @Published var defaultWorkMode: WorkMode = .general

    @Published var selectedMode: WorkMode = .general
    @Published var selectedProvider = "openai"
    @Published var inputQuery = ""
    @Published var openAIAPIKey = ""
    @Published var anthropicAPIKey = ""
    @Published var localEngine: LocalEngine = .mlx
    @Published var mlxModelPath = ""
    @Published var llamaModelPath = ""
    @Published var modelDownloadURL = ""
    @Published var modelDownloadFilename = ""
    @Published var modelDownloadEngine: LocalEngine = .llamaCPP
    @Published var availableModels: [ModelListItem] = []
    @Published var catalogModels: [ModelCatalogItem] = []
    @Published var catalogDefaultProfile = "balanced"
    @Published var isCatalogBusy = false
    @Published var catalogInstallingModelID: String?
    @Published var showAdvancedModelDetails = false
    @Published var isDownloadingModel = false
    @Published var localRuntimeDetail = ""
    @Published var chatFilterCategory = ""
    @Published var chatFilterTags = ""
    @Published var chatFilterYear = ""
    @Published var chatFilterProject = ""
    @Published var isCitationDrawerVisible = false
    @Published var actionPermissionMode: ActionPermissionMode = .askPerAction
    @Published var pendingSystemAction: SuggestedAction?
    @Published var pendingExternalDirectQuery: String?
    @Published var highlightedCitationPath: String?
    @Published var adaptivePersonalizationEnabled = true
    @Published var sessionMemoryEnabled = true
    @Published var workspaceMemoryEnabled = true
    @Published var localMemoryOnly = true
    @Published var workspaceMemoryMode: WorkspaceMemoryMode = .normal
    @Published var sessionMemoryItems: [SessionMemoryItem] = []
    @Published var workspaceMemoryItems: [WorkspaceMemoryItem] = []
    @Published var preferenceMemoryItems: [UserPreferenceItem] = []
    @Published var episodicMemoryItems: [EpisodicMemoryEvent] = []
    @Published var pinnedMemoryItems: [PinnedMemoryItem] = []
    @Published var memorySearchText = ""

    @Published var indexProgress: Double = 0
    @Published var indexStageText = "준비 중"

    @Published var chatRooms: [ChatRoom] = []
    @Published var selectedChatRoomID: String = ""
    @Published var chatMessages: [ChatMessage] = []
    @Published var citations: [Citation] = []

    @Published var statusSnapshot: StatusSnapshot?
    @Published var failureItems: [FailureItem] = []
    @Published var documents: [DocumentMetadata] = []
    @Published var documentsTotal = 0
    @Published var documentSearchText = ""
    @Published var documentFilterCategory = ""
    @Published var documentFilterTag = ""
    @Published var documentFilterYear = ""
    @Published var documentFilterProject = ""
    @Published var showExcludedDocuments = false

    @Published var lastError: String?
    @Published var isBusy = false
    @Published var needsExternalConfirmation = false
    @Published var chatResponseRoute: ChatResponseRoute = .hybrid
    @Published var quickInferencePreset: QuickInferencePreset = .quality

    // MARK: - UserDefaults Keys
    enum UDKey {
        static let onboardingFinished   = "local_ai_onboarding_finished"
        static let approvedActions      = "local_ai_approved_system_actions"
        static let chatRooms            = "local_ai_chat_rooms_v1"
        static let activeChatRoom       = "local_ai_active_chat_room_id_v1"
        static let chatResponseRoute    = "local_ai_chat_response_route_v1"
        static let quickInferencePreset = "local_ai_quick_inference_preset_v1"
        static let localEngine          = "local_ai_local_engine_v1"
        static let mlxModelPath         = "local_ai_mlx_model_path_v1"
        static let llamaModelPath       = "local_ai_llama_model_path_v1"
    }

    let sidecar = SidecarProcessManager()
    let bookmarkStore = BookmarkStore()
    let chatRoomService = ChatRoomService()
    let chatFlowService = ChatFlowService()
    let workspaceSyncService = WorkspaceSyncService()
    let memoryServiceAdapter = MemoryServiceAdapter()
    let modelRuntimeService = ModelRuntimeService()
    let appPreferencesStore = AppPreferencesStore()
    var approvedSystemActionKinds: Set<String> = []
    var isRecoveringSession = false
    var lastPostChatStateRefreshAt = Date.distantPast
    let postChatStateRefreshInterval: TimeInterval = 8

    /// Returns the current active conversation ID without causing side-effects.
    /// Call `ensureActiveConversation()` first when a valid room is required.
    var activeConversationID: String {
        if !selectedChatRoomID.isEmpty {
            return selectedChatRoomID
        }
        return chatRooms.first(where: { !$0.isArchived })?.id ?? chatRooms.first?.id ?? ""
    }

    /// Ensures at least one chat room exists. If none, creates a default room.
    /// Must be called before any operation that requires a valid conversation ID.
    func updateDocumentMetadata(
        docID: String,
        category: String?,
        subcategory: String?,
        documentType: String?,
        tags: [String]?,
        year: Int?,
        project: String?,
        importance: Double?,
        excluded: Bool?
    ) async {
        do {
            _ = try await performWithSidecarRetry { client in
                try await client.updateDocumentMetadata(
                    docID: docID,
                    payload: DocumentMetadataUpdateRequest(
                        category: category,
                        subcategory: subcategory,
                        document_type: documentType,
                        tags: tags,
                        year: year,
                        project: project,
                        importance: importance,
                        excluded: excluded
                    )
                )
            }
            do {
                try await writeMemoryEvent(
                    eventType: .manualOverride,
                    summary: "document metadata override",
                    relatedFileIDs: [docID],
                    relatedActionIDs: [],
                    metadata: [
                        "doc_id": .string(docID),
                        "action": .string("metadata_update"),
                    ],
                    importance: 0.78
                )
            } catch {
                if !isEndpointNotFound(error) {
                    lastError = "메모리 이벤트 기록 실패: \(error.localizedDescription)"
                }
            }
            await refreshDocuments()
        } catch {
            if isEndpointNotFound(error) {
                lastError = "문서 메타 수정 API가 없습니다. sidecar 업데이트 후 다시 시도해 주세요."
            } else {
                handleViewModelError(error)
            }
        }
    }

    func reclassifyDocument(docID: String) async {
        do {
            _ = try await performWithSidecarRetry { client in
                try await client.reclassifyDocument(docID: docID)
            }
            await refreshDocuments()
        } catch {
            if isEndpointNotFound(error) {
                lastError = "문서 재분류 API가 없습니다. sidecar 업데이트 후 다시 시도해 주세요."
            } else {
                handleViewModelError(error)
            }
        }
    }

}
