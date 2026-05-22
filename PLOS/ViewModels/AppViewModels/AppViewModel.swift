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

    func title(language: AppLanguage) -> String {
        switch self {
        case .hybrid:
            return L10n.tr("chat_response_route.hybrid", language: language, fallbackKo: "하이브리드", fallbackEn: "Hybrid", fallbackJa: "ハイブリッド")
        case .localOnly:
            return L10n.tr("chat_response_route.local_only", language: language, fallbackKo: "로컬", fallbackEn: "Local", fallbackJa: "ローカル")
        case .apiOnly:
            return L10n.tr("chat_response_route.api_only", language: language, fallbackKo: "항상 API 호출", fallbackEn: "Always API", fallbackJa: "常にAPI呼び出し")
        }
    }

    var title: String { title(language: L10n.loadSelection()) }
}

enum QuickInferencePreset: String, CaseIterable, Identifiable {
    case fast
    case quality
    case highQuality

    var id: String { rawValue }

    func title(language: AppLanguage) -> String {
        switch self {
        case .fast:
            return L10n.tr("quick_inference_preset.fast.title", language: language, fallbackKo: "빠른 추론", fallbackEn: "Fast", fallbackJa: "高速推論")
        case .quality:
            return L10n.tr("quick_inference_preset.quality.title", language: language, fallbackKo: "균형", fallbackEn: "Balanced", fallbackJa: "バランス")
        case .highQuality:
            return L10n.tr("quick_inference_preset.high_quality.title", language: language, fallbackKo: "고품질 추론", fallbackEn: "High quality", fallbackJa: "高品質推論")
        }
    }

    func detail(language: AppLanguage) -> String {
        switch self {
        case .fast:
            return L10n.tr("quick_inference_preset.fast.detail", language: language, fallbackKo: "가볍고 빠른 응답", fallbackEn: "Lightweight and quick", fallbackJa: "軽量で高速な応答")
        case .quality:
            return L10n.tr("quick_inference_preset.quality.detail", language: language, fallbackKo: "속도와 정확도 균형", fallbackEn: "Balanced speed and quality", fallbackJa: "速度と精度のバランス")
        case .highQuality:
            return L10n.tr("quick_inference_preset.high_quality.detail", language: language, fallbackKo: "응답 품질 우선", fallbackEn: "Prioritize quality", fallbackJa: "品質優先")
        }
    }

    var title: String { title(language: L10n.loadSelection()) }
    var detail: String { detail(language: L10n.loadSelection()) }

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

struct LiveThinkingTraceEvent: Identifiable, Equatable {
    let id = UUID()
    let status: String
    let message: String
    let source: String
    let url: String?
    let at: String?
}

struct ComposerAttachment: Identifiable, Hashable {
    let id: String
    let kind: ChatAttachmentKind
    let filePath: String
    let fileName: String
    let mimeType: String?
}

enum MainPanelSelection: String, Codable {
    case chat
    case plugin
}

@MainActor
final class AppViewModel: ObservableObject {
    static let fixedCategories = ["학습자료", "프로젝트문서", "회의록", "아이디어", "개인메모", "참고자료", "코드관련"]

    @Published var onboardingStep: OnboardingStep = .welcome
    @Published var hasFinishedOnboarding = false

    @Published var includedFolderURLs: [URL] = []
    @Published var excludedPaths: [String] = []
    @Published var startupProfile: StartupProfile = .recommended
    @Published var appLanguage: AppLanguage = .auto {
        didSet {
            L10n.saveSelection(appLanguage)
        }
    }
    @Published var privacyMode: PrivacyMode = .hybrid
    @Published var hybridWebSearchEnabled = false
    @Published var systemFilePermission: SystemFilePermission = .readOnly
    @Published var defaultWorkMode: WorkMode = .general

    @Published var selectedMode: WorkMode = .general
    @Published var selectedProvider = "openai"
    @Published var inputQuery = ""
    @Published var composerAttachments: [ComposerAttachment] = []
    @Published var openAIAPIKey = ""
    @Published var anthropicAPIKey = ""
    @Published var localEngine: LocalEngine = .mlx
    @Published var mlxModelPath = ""
    @Published var llamaModelPath = ""
    @Published var modelDownloadURL = ""
    @Published var modelDownloadFilename = ""
    @Published var modelDownloadEngine: LocalEngine = .llamaCPP
    @Published var modelsStorageDirectoryPath = ""
    @Published var runtimeStorageDirectoryPath = ""
    @Published var effectiveModelsStorageDirectoryPath = ""
    @Published var effectiveRuntimeStorageDirectoryPath = ""
    @Published var storageDirectoryWarning = ""
    @Published var availableModels: [ModelListItem] = []
    @Published var catalogModels: [ModelCatalogItem] = []
    @Published var catalogDefaultProfile = "balanced"
    @Published var isCatalogBusy = false
    @Published var catalogInstallingModelID: String?
    @Published var catalogInstallProgress: [String: Double] = [:]
    @Published var modelDownloadProgressPercent: Double?
    @Published var showAdvancedModelDetails = false
    @Published var sidecarVisionEnabled = true
    @Published var sidecarVisionCaptionModel = "microsoft/git-base-coco"
    @Published var sidecarVisionClassifyModel = "google/vit-base-patch16-224"
    @Published var sidecarMlxKVQEnabled = false
    @Published var sidecarMlxKVQMode: SidecarMlxKVQMode = .turbo3
    @Published var sidecarMlxKVQBits = 3
    @Published var sidecarConversationTurboEnabled = false
    @Published var sidecarInferenceTimeoutDisabled = false
    @Published var sidecarMainResponseTimeoutSeconds = 240
    @Published var sidecarAuxiliaryTimeoutSeconds = 12
    @Published var showThinkingProcessInChat = true
    @Published var pluginUIToggleStates: [String: Bool] = [:]
    @Published var selectedMainPanel: MainPanelSelection = .chat
    @Published var selectedPluginPanelID = ""
    @Published var selectedPluginPanelViewID = ""
    @Published var activePluginPanel: PluginPanelOpenResponse?
    @Published var pluginPanelPrompt = ""
    @Published var pluginPanelNegativePrompt = ""
    @Published var pluginPanelWidth = 1024
    @Published var pluginPanelHeight = 1024
    @Published var pluginPanelSteps = 6
    @Published var pluginPanelSeedText = ""
    @Published var pluginPanelBatch = 1
    @Published var pluginPanelModelID = "sdxl-lightning"
    @Published var pluginPanelRepoID = "ByteDance/SDXL-Lightning"
    @Published var pluginPanelFilename = ""
    @Published var pluginPanelDownloadMessage = ""
    @Published var pluginPanelInstalledModels: [String] = []
    @Published var pluginPanelActiveModelID = ""
    @Published var pluginPanelImages: [String] = []
    @Published var pluginPanelIsBusy = false
    @Published var pluginPanelLastJobID: String?
    @Published var extensionCapabilities: [ExtensionCapabilityState] = []
    @Published var pluginEntries: [PluginRegistryEntry] = []
    @Published var isPluginBusy = false
    @Published var pluginDraftID = ""
    @Published var pluginDraftVersion = "0.1.0"
    @Published var pluginDraftEntrypoint = ""
    @Published var pluginDraftPermissions = ""
    @Published var pluginDraftSignature = ""
    @Published var pluginDraftBuildTarget: PluginBuildTarget = .community
    @Published var pluginDraftPrivacyMode: PluginPrivacyMode = .localOnly
    @Published var pluginDraftEnabled = false
    @Published var pluginDraftCapabilities: Set<ExtensionCapability> = Set(ExtensionCapability.allCases)
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
    @Published var searxngURL: String = "http://localhost:8080"
    @Published var autoStartSearXNG = false

    @Published var indexProgress: Double = 0
    @Published var indexStageText = "준비 중"

    @Published var chatRooms: [ChatRoom] = []
    @Published var selectedChatRoomID: String = ""
    @Published var chatMessages: [ChatMessage] = []
    @Published var citations: [Citation] = []
    @Published var roomStorageStatusByRoomID: [String: RoomStorageStatusResponse] = [:]
    @Published var roomIndexStateByRoomID: [String: String] = [:]
    @Published var roomIndexProgressByRoomID: [String: Double] = [:]

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
    @Published var isModelRuntimeBusy = false
    @Published var isGeneratingChatResponse = false
    @Published var activeGeneratingMessageID: UUID?
    @Published var liveThinkingTraceEvents: [LiveThinkingTraceEvent] = []
    @Published var lastSettingsSavedAt: Date = .distantPast
    @Published var needsExternalConfirmation = false
    @Published var chatResponseRoute: ChatResponseRoute = .hybrid
    @Published var quickInferencePreset: QuickInferencePreset = .quality
    @Published var roleplayModeEnabled = false
    @Published var sidecarRecoveryState: String = "idle"
    @Published var sidecarRecoveryAttempt: Int = 0
    @Published var nativeCrashDetected = false
    @Published var isSidecarReadyForChat = false

    // MARK: - UserDefaults Keys
    enum UDKey {
        static let onboardingFinished   = "local_ai_onboarding_finished"
        static let approvedActions      = "local_ai_approved_system_actions"
        static let chatRooms            = "local_ai_chat_rooms_v1"
        static let activeChatRoom       = "local_ai_active_chat_room_id_v1"
        static let chatResponseRoute    = "local_ai_chat_response_route_v1"
        static let quickInferencePreset = "local_ai_quick_inference_preset_v1"
        static let roleplayModeEnabled  = "local_ai_roleplay_mode_enabled_v1"
        static let localEngine          = "local_ai_local_engine_v1"
        static let mlxModelPath         = "local_ai_mlx_model_path_v1"
        static let llamaModelPath       = "local_ai_llama_model_path_v1"
        static let modelsStorageDir     = "local_ai_models_storage_dir_v1"
        static let runtimeStorageDir    = "local_ai_runtime_storage_dir_v1"
        static let sidecarVisionEnabled = "local_ai_sidecar_vision_enabled_v1"
        static let sidecarVisionCaptionModel = "local_ai_sidecar_vision_caption_model_v1"
        static let sidecarVisionClassifyModel = "local_ai_sidecar_vision_classify_model_v1"
        static let sidecarMlxKVQEnabled = "local_ai_sidecar_mlx_kvq_enabled_v1"
        static let sidecarMlxKVQMode = "local_ai_sidecar_mlx_kvq_mode_v1"
        static let sidecarMlxKVQBits = "local_ai_sidecar_mlx_kvq_bits_v1"
        static let sidecarConversationTurboEnabled = "local_ai_sidecar_conversation_turbo_enabled_v1"
        static let sidecarInferenceTimeoutDisabled = "local_ai_sidecar_inference_timeout_disabled_v1"
        static let sidecarMainResponseTimeoutSeconds = "local_ai_sidecar_main_response_timeout_seconds_v1"
        static let sidecarAuxiliaryTimeoutSeconds = "local_ai_sidecar_auxiliary_timeout_seconds_v1"
        static let showThinkingProcessInChat = "local_ai_show_thinking_process_in_chat_v1"
        static let appLanguage          = L10n.userDefaultsKey
        static let searxngURL          = "local_ai_searxng_url_v1"
        static let autoStartSearXNG     = "local_ai_auto_start_searxng_v1"
    }

    let sidecar = SidecarProcessManager()
    let bookmarkStore = BookmarkStore()
    let chatRoomService = ChatRoomService()
    let chatFlowService = ChatFlowService()
    let workspaceSyncService = WorkspaceSyncService()
    let memoryServiceAdapter = MemoryServiceAdapter()
    let modelRuntimeService = ModelRuntimeService()
    let extensionServiceAdapter = ExtensionServiceAdapter()
    let appPreferencesStore = AppPreferencesStore()
    var approvedSystemActionKinds: Set<String> = []
    var isRecoveringSession = false
    var lastPostChatStateRefreshAt = Date.distantPast
    let postChatStateRefreshInterval: TimeInterval = 8
    var activeLocalChatTask: Task<Void, Never>?
    var isStreamingRoomStateDirty = false
    var streamingDirtyRoomID: String?
    var roomIndexPollingTasks: [String: Task<Void, Never>] = [:]
    var pluginPanelStatusStreamTask: Task<Void, Never>?
    var sidecarRecoveryAttemptTimestamps: [Date] = []
    var roleplayPersonaHintByRoomID: [String: String] = [:]

    func clearResolvedErrorIfNeeded() {
        guard let current = lastError?.trimmingCharacters(in: .whitespacesAndNewlines), !current.isEmpty else {
            return
        }
        let lowered = current.lowercased()
        let recoverableTokens = [
            "http",
            "connection",
            "timeout",
            "sidecar",
            "runtime",
            "세션 재연결 실패",
            "상태 동기화 실패",
            "메모리 이벤트 기록 실패",
            "failed",
            "error",
            "실패",
            "오류",
        ]
        guard recoverableTokens.contains(where: { lowered.contains($0.lowercased()) }) else {
            return
        }
        lastError = nil
    }

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
