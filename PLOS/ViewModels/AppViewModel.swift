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

    private let sidecar = SidecarProcessManager()
    private let bookmarkStore = BookmarkStore()
    private let onboardingDefaultsKey = "local_ai_onboarding_finished"
    private let approvedSystemActionsDefaultsKey = "local_ai_approved_system_actions"
    private let chatRoomsDefaultsKey = "local_ai_chat_rooms_v1"
    private let activeChatRoomDefaultsKey = "local_ai_active_chat_room_id_v1"
    private let chatResponseRouteDefaultsKey = "local_ai_chat_response_route_v1"
    private let quickInferencePresetDefaultsKey = "local_ai_quick_inference_preset_v1"
    private let localEngineDefaultsKey = "local_ai_local_engine_v1"
    private let mlxModelPathDefaultsKey = "local_ai_mlx_model_path_v1"
    private let llamaModelPathDefaultsKey = "local_ai_llama_model_path_v1"
    private var approvedSystemActionKinds: Set<String> = []
    private var isRecoveringSession = false
    private var lastPostChatStateRefreshAt = Date.distantPast
    private let postChatStateRefreshInterval: TimeInterval = 8

    private var activeConversationID: String {
        if !selectedChatRoomID.isEmpty {
            return selectedChatRoomID
        }
        if let first = chatRooms.first {
            return first.id
        }
        let created = ChatRoom.makeDefault()
        chatRooms = [created]
        selectedChatRoomID = created.id
        chatMessages = []
        citations = []
        persistChatRooms()
        return created.id
    }

    var currentChatRoomTitle: String {
        chatRooms.first(where: { $0.id == activeConversationID })?.title ?? "새 채팅"
    }

    private var activeRoomIndex: Int? {
        chatRooms.firstIndex(where: { $0.id == activeConversationID })
    }

    private var activeLatestQueryForDeepAnalysis: String? {
        guard let idx = activeRoomIndex else { return nil }
        return chatRooms[idx].latestQueryForDeepAnalysis
    }

    var currentPrivacyBadge: String {
        switch privacyMode {
        case .localOnly:
            return "Local Only"
        case .hybrid:
            return "Hybrid"
        case .confirmBeforeExternal:
            return "Confirm"
        }
    }

    private var activeModelPath: String {
        switch localEngine {
        case .mlx:
            return mlxModelPath
        case .llamaCPP:
            return llamaModelPath
        }
    }

    var installedModelsSorted: [ModelListItem] {
        var deduped: [String: ModelListItem] = [:]
        for model in availableModels {
            guard Self.isDisplayableModelArtifact(model) else { continue }
            let selectionPath = normalizedSelectionPath(for: model)
            guard !selectionPath.isEmpty else { continue }

            let existing = deduped[selectionPath]
            let chosenDate = max(existing?.modified_at ?? .distantPast, model.modified_at)
            let chosenSize = max(existing?.size_bytes ?? 0, model.size_bytes)
            let displayName = URL(fileURLWithPath: selectionPath).lastPathComponent
            deduped[selectionPath] = ModelListItem(
                file_name: displayName,
                path: selectionPath,
                engine: model.engine,
                size_bytes: chosenSize,
                modified_at: chosenDate
            )
        }

        var output = Array(deduped.values).sorted { $0.modified_at > $1.modified_at }
        let currentPath = activeModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if !currentPath.isEmpty, !output.contains(where: { $0.engine == localEngine && Self.samePath($0.path, currentPath) }) {
            output.insert(
                ModelListItem(
                    file_name: URL(fileURLWithPath: currentPath).lastPathComponent,
                    path: URL(fileURLWithPath: currentPath).standardizedFileURL.path,
                    engine: localEngine,
                    size_bytes: 0,
                    modified_at: .distantPast
                ),
                at: 0
            )
        }
        return output
    }

    var activeModelDisplayName: String {
        let currentPath = activeModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if currentPath.isEmpty {
            return "모델 선택"
        }
        if let matched = installedModelsSorted.first(where: { $0.engine == localEngine && Self.samePath($0.path, currentPath) }) {
            return matched.file_name
        }
        return URL(fileURLWithPath: currentPath).lastPathComponent
    }

    func isInstalledModelActive(_ model: ModelListItem) -> Bool {
        model.engine == localEngine && Self.samePath(model.path, activeModelPath)
    }

    func bootstrap() async {
        hasFinishedOnboarding = UserDefaults.standard.bool(forKey: onboardingDefaultsKey)
        includedFolderURLs = bookmarkStore.loadURLs()
        loadApprovedSystemActionKinds()
        loadChatRooms()
        loadChatResponseRoute()
        loadLocalModelPreferenceSnapshot()
        loadSecretAPIKeys()
        syncQuickInferencePresetFromProfile()

        do {
            try await sidecar.start()
            if hasFinishedOnboarding {
                try await syncWorkspaceAndSettings()
                try await refreshRemoteState()
            }
        } catch {
            handleViewModelError(error)
        }
    }

    func shutdown() {
        sidecar.stop()
    }

    func createChatRoom() {
        let room = ChatRoom.makeDefault()
        chatRooms.insert(room, at: 0)
        selectedChatRoomID = room.id
        chatMessages = []
        citations = []
        highlightedCitationPath = nil
        persistChatRooms()
        Task {
            do {
                try await refreshMemoryState()
            } catch {
                if !isEndpointNotFound(error) {
                    handleViewModelError(error)
                }
            }
        }
    }

    func selectChatRoom(_ roomID: String) {
        guard let room = chatRooms.first(where: { $0.id == roomID }) else {
            return
        }
        selectedChatRoomID = room.id
        chatMessages = room.messages
        citations = room.citations
        highlightedCitationPath = nil
        UserDefaults.standard.set(room.id, forKey: activeChatRoomDefaultsKey)
        Task {
            do {
                try await refreshMemoryState()
            } catch {
                if !isEndpointNotFound(error) {
                    handleViewModelError(error)
                }
            }
        }
    }

    func deleteChatRoom(_ roomID: String) {
        guard chatRooms.count > 1 else {
            return
        }
        chatRooms.removeAll { $0.id == roomID }
        if selectedChatRoomID == roomID, let next = chatRooms.first {
            selectedChatRoomID = next.id
            chatMessages = next.messages
            citations = next.citations
        }
        persistChatRooms()
        Task {
            do {
                try await refreshMemoryState()
            } catch {
                if !isEndpointNotFound(error) {
                    handleViewModelError(error)
                }
            }
        }
    }

    func addFolder() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = true
        panel.prompt = "선택"

        if panel.runModal() == .OK {
            let existing = Set(includedFolderURLs.map(\.path))
            let newURLs = panel.urls.filter { !existing.contains($0.path) }
            includedFolderURLs.append(contentsOf: newURLs)
            persistBookmarks()
        }
    }

    func removeFolder(_ path: String) {
        includedFolderURLs.removeAll { $0.path == path }
        persistBookmarks()
    }

    func persistBookmarks() {
        bookmarkStore.save(urls: includedFolderURLs)
    }

    func startOnboardingIndexingFlow() async {
        guard !includedFolderURLs.isEmpty else {
            lastError = "최소 1개 이상의 폴더를 선택해 주세요."
            return
        }

        onboardingStep = .indexing
        isBusy = true
        defer { isBusy = false }

        do {
            try await syncWorkspaceAndSettings()
            try await runIndexing(scope: "full")
            onboardingStep = .ready
            try await refreshRemoteState()
        } catch {
            handleViewModelError(error)
        }
    }

    func finalizeOnboarding() {
        hasFinishedOnboarding = true
        UserDefaults.standard.set(true, forKey: onboardingDefaultsKey)
    }

    func askLocal() async {
        let query = inputQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        await askLocal(query: query, appendUserMessage: true)
        inputQuery = ""
    }

    func deepAnalyzeTapped() {
        if privacyMode == .localOnly {
            lastError = "완전 로컬 모드에서는 외부 호출이 비활성화됩니다."
            return
        }
        if privacyMode == .confirmBeforeExternal {
            needsExternalConfirmation = true
            return
        }
        Task {
            await performDeepAnalysis(userConfirmed: true)
        }
    }

    func confirmExternalCall() {
        if let pending = pendingExternalDirectQuery {
            pendingExternalDirectQuery = nil
            Task { await performDeepAnalysis(userConfirmed: true, queryOverride: pending) }
        } else {
            Task { await performDeepAnalysis(userConfirmed: true) }
        }
    }

    func cancelExternalCall() {
        pendingExternalDirectQuery = nil
    }

    func setChatResponseRoute(_ route: ChatResponseRoute) {
        chatResponseRoute = route
        switch route {
        case .localOnly:
            privacyMode = .localOnly
        case .hybrid:
            if privacyMode == .localOnly {
                privacyMode = .hybrid
            }
        case .apiOnly:
            if privacyMode == .localOnly {
                privacyMode = .hybrid
            }
        }
        UserDefaults.standard.set(route.rawValue, forKey: chatResponseRouteDefaultsKey)
        Task { await saveSettingsAndWorkspace() }
    }

    func applyQuickInferencePreset(_ preset: QuickInferencePreset) {
        quickInferencePreset = preset
        startupProfile = preset.startupProfile
        persistLocalModelPreferenceSnapshot()
        Task { await saveSettingsAndWorkspace() }
    }

    var currentChatTranscript: String {
        chatMessages
            .map { message in
                switch message.source {
                case .user:
                    return "You: \(message.text ?? "")"
                case .local:
                    let lead = message.lead ?? ""
                    let summary = message.resultSummary ?? ""
                    return "Local AI: \(lead)\n\(summary)".trimmingCharacters(in: .whitespacesAndNewlines)
                case .external:
                    return "External: \(message.text ?? "")"
                }
            }
            .joined(separator: "\n\n")
    }

    func copyCurrentChatTranscriptToClipboard() {
        let transcript = currentChatTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !transcript.isEmpty else {
            lastError = "복사할 대화 내용이 없습니다."
            return
        }
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(transcript, forType: .string)
    }

    func attachFileIntoComposer() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.prompt = "첨부"
        guard panel.runModal() == .OK, let url = panel.url else {
            return
        }
        let token = "첨부 파일: \(url.path)"
        if inputQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            inputQuery = token
        } else {
            inputQuery += "\n\(token)"
        }
    }

    func startSystemDictation() {
        let selector = NSSelectorFromString("startDictation:")
        if !NSApp.sendAction(selector, to: nil, from: nil) {
            lastError = "시스템 받아쓰기를 시작하지 못했습니다."
        }
    }

    func executeAction(_ action: SuggestedAction) async {
        if let path = action.payload["file_path"], !path.isEmpty {
            highlightedCitationPath = path
        }
        switch action.execution_mode {
        case .promptInjection:
            await executePromptInjectionAction(action)
        case .system:
            await executeSystemAction(action)
        }
    }

    func confirmPendingSystemAction() {
        guard let action = pendingSystemAction else {
            return
        }
        pendingSystemAction = nil
        if actionPermissionMode == .askPerAction {
            approvedSystemActionKinds.insert(action.kind.rawValue)
            persistApprovedSystemActionKinds()
        }
        performSystemAction(action)
    }

    func cancelPendingSystemAction() {
        pendingSystemAction = nil
    }

    func performDeepAnalysis(userConfirmed: Bool, queryOverride: String? = nil) async {
        let targetQuery = queryOverride?.trimmingCharacters(in: .whitespacesAndNewlines)
        let query = (targetQuery?.isEmpty == false ? targetQuery : activeLatestQueryForDeepAnalysis)
        guard let query else {
            lastError = "먼저 로컬 질문을 실행해 주세요."
            return
        }

        isBusy = true
        defer { isBusy = false }

        do {
            let response = try await performWithSidecarRetry { client in
                try await client.deepAnalysis(
                    DeepAnalysisRequest(
                        query: query,
                        mode: selectedMode,
                        provider: selectedProvider,
                        selected_citations: citations,
                        user_confirmed: userConfirmed
                    )
                )
            }
            appendChatMessage(ChatMessage(source: .external, text: response.answer, timestamp: Date()))
            try await refreshRemoteState()
        } catch {
            handleViewModelError(error)
        }
    }

    func triggerFullReindex() async {
        isBusy = true
        defer { isBusy = false }

        do {
            try await syncWorkspaceAndSettings()
            try await runIndexing(scope: "full")
            try await refreshRemoteState()
        } catch {
            handleViewModelError(error)
        }
    }

    func saveSettingsAndWorkspace() async {
        isBusy = true
        defer { isBusy = false }

        do {
            let secretChanged = persistSecretAPIKeys()
            persistLocalModelPreferenceSnapshot()
            if secretChanged {
                sidecar.stop()
                _ = try await ensureSidecarClient()
            }
            try await syncWorkspaceAndSettings()
            do {
                try await writeMemoryEvent(
                    eventType: .manualOverride,
                    summary: "settings updated",
                    relatedFileIDs: [],
                    relatedActionIDs: [],
                    metadata: [
                        "default_mode": .string(defaultWorkMode.rawValue),
                        "privacy_rule": .string(privacyMode.rawValue),
                        "workspace_memory_mode": .string(workspaceMemoryMode.rawValue),
                    ],
                    importance: 0.9
                )
            } catch {
                if !isEndpointNotFound(error) {
                    lastError = "메모리 이벤트 기록 실패: \(error.localizedDescription)"
                }
            }
            try await refreshRemoteState()
        } catch {
            handleViewModelError(error)
        }
    }

    func refreshRemoteState() async throws {
        let settings = try await performWithSidecarRetry { try await $0.getSettings() }
        privacyMode = settings.privacy_mode
        startupProfile = settings.startup_profile
        syncQuickInferencePresetFromProfile()
        localEngine = settings.local_engine ?? .mlx
        mlxModelPath = settings.mlx_model_path ?? ""
        llamaModelPath = settings.llama_model_path ?? ""
        persistLocalModelPreferenceSnapshot()
        actionPermissionMode = settings.action_permission_mode ?? .askPerAction
        adaptivePersonalizationEnabled = settings.adaptive_personalization_enabled
        sessionMemoryEnabled = settings.session_memory_enabled
        workspaceMemoryEnabled = settings.workspace_memory_enabled
        localMemoryOnly = settings.local_memory_only
        workspaceMemoryMode = settings.workspace_memory_mode

        let status = try await performWithSidecarRetry { try await $0.getStatus() }
        statusSnapshot = status

        let failures = try await performWithSidecarRetry { try await $0.getFailures() }
        failureItems = failures.failures

        do {
            availableModels = try await performWithSidecarRetry { try await $0.listModels().models }
        } catch {
            if !isEndpointNotFound(error) {
                throw error
            }
            availableModels = []
        }

        do {
            let catalog = try await performWithSidecarRetry { try await $0.getModelCatalog() }
            catalogDefaultProfile = catalog.default_profile
            catalogModels = catalog.models
        } catch {
            if !isEndpointNotFound(error) {
                throw error
            }
            catalogModels = []
        }

        do {
            let docs = try await performWithSidecarRetry { client in
                try await client.listDocuments(
                    search: documentSearchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : documentSearchText,
                    category: documentFilterCategory.isEmpty ? nil : documentFilterCategory,
                    tags: parseTagText(documentFilterTag),
                    year: Int(documentFilterYear),
                    project: documentFilterProject.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : documentFilterProject,
                    excluded: showExcludedDocuments ? true : false
                )
            }
            documents = docs.documents
            documentsTotal = docs.total
        } catch {
            if isEndpointNotFound(error) {
                documents = []
                documentsTotal = 0
            } else {
                throw error
            }
        }

        do {
            try await refreshMemoryState()
        } catch {
            if !isEndpointNotFound(error) {
                throw error
            }
            sessionMemoryItems = []
            workspaceMemoryItems = []
            preferenceMemoryItems = []
            episodicMemoryItems = []
            pinnedMemoryItems = []
        }

    }

    func refreshDocuments() async {
        do {
            let docs = try await performWithSidecarRetry { client in
                try await client.listDocuments(
                    search: documentSearchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : documentSearchText,
                    category: documentFilterCategory.isEmpty ? nil : documentFilterCategory,
                    tags: parseTagText(documentFilterTag),
                    year: Int(documentFilterYear),
                    project: documentFilterProject.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : documentFilterProject,
                    excluded: showExcludedDocuments ? true : false
                )
            }
            documents = docs.documents
            documentsTotal = docs.total
        } catch {
            if isEndpointNotFound(error) {
                documents = []
                documentsTotal = 0
                lastError = "문서 메타 API를 찾지 못했습니다. sidecar를 재시작해 최신 버전을 적용해 주세요."
            } else {
                handleViewModelError(error)
            }
        }
    }

    func refreshMemoryState() async throws {
        let workspaceID = currentWorkspaceID()
        let session = try await performWithSidecarRetry { try await $0.getRelevantSessionMemory(sessionID: activeConversationID) }
        let workspace = try await performWithSidecarRetry { try await $0.getRelevantWorkspaceMemory(workspaceID: workspaceID, intent: selectedMode.rawValue.lowercased()) }
        let prefs = try await performWithSidecarRetry { try await $0.getMemoryPreferences() }
        let episodic = try await performWithSidecarRetry {
            try await $0.getRelevantEpisodicMemory(
                workspaceID: workspaceID,
                intent: selectedMode.rawValue.lowercased(),
                relatedFileIDs: citations.map(\.doc_id)
            )
        }
        let pins = try await performWithSidecarRetry { try await $0.listPins(scope: nil, workspaceID: workspaceID) }

        sessionMemoryItems = session.items
        workspaceMemoryItems = workspace.items
        preferenceMemoryItems = prefs.items
        episodicMemoryItems = episodic.items
        pinnedMemoryItems = pins.items
    }

    func clearMemory(scope: MemoryClearScope) async {
        do {
            let workspaceID = currentWorkspaceID()
            let workspaceScoped = scope == .workspace || scope == .episodic || scope == .inferredOnly
            _ = try await performWithSidecarRetry { client in
                try await client.clearMemory(
                    MemoryClearRequest(
                        scope: scope,
                        workspace_id: workspaceScoped ? workspaceID : nil,
                        session_id: scope == .session ? activeConversationID : nil
                    )
                )
            }
            try await refreshMemoryState()
        } catch {
            handleViewModelError(error)
        }
    }

    func pinMemory(memoryID: String?, title: String, content: String, workspaceScoped: Bool) async {
        do {
            let request = MemoryPinRequest(
                memory_id: memoryID,
                scope: workspaceScoped ? "workspace" : "global",
                workspace_id: workspaceScoped ? currentWorkspaceID() : nil,
                title: title,
                content: content
            )
            _ = try await performWithSidecarRetry { client in
                try await client.pinMemory(request)
            }
            try await refreshMemoryState()
        } catch {
            handleViewModelError(error)
        }
    }

    func unpinMemory(memoryID: String) async {
        do {
            _ = try await performWithSidecarRetry { client in
                try await client.unpinMemory(memoryID: memoryID)
            }
            try await refreshMemoryState()
        } catch {
            handleViewModelError(error)
        }
    }

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

    private func syncWorkspaceAndSettings() async throws {
        persistBookmarks()

        _ = try await performWithSidecarRetry { client in
            try await client.updateWorkspace(
                WorkspaceUpdateRequest(
                    included_paths: includedFolderURLs.map(\.path),
                    excluded_paths: excludedPaths,
                    startup_profile: startupProfile,
                    default_mode: defaultWorkMode
                )
            )
        }

        _ = try await performWithSidecarRetry { client in
            try await client.updateSettings(
                SettingsModel(
                    privacy_mode: privacyMode,
                    startup_profile: startupProfile,
                    model_profile: profileKey(from: startupProfile),
                    local_engine: localEngine,
                    mlx_model_path: mlxModelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : mlxModelPath,
                    llama_model_path: llamaModelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : llamaModelPath,
                    reindex_policy: "filewatch_incremental",
                    language: "auto",
                    action_permission_mode: actionPermissionMode,
                    adaptive_personalization_enabled: adaptivePersonalizationEnabled,
                    session_memory_enabled: sessionMemoryEnabled,
                    workspace_memory_enabled: workspaceMemoryEnabled,
                    local_memory_only: localMemoryOnly,
                    workspace_memory_mode: workspaceMemoryMode
                )
            )
        }

        do {
            _ = try await performWithSidecarRetry { client in
                try await prepareSelectedRuntime(using: client)
            }
        } catch {
            if !isEndpointNotFound(error) {
                localRuntimeDetail = "엔진 준비 경고: \(error.localizedDescription)"
            }
        }
    }

    private func runIndexing(scope: String) async throws {
        let start = try await performWithSidecarRetry { client in
            try await client.startIndexJob(scope: scope)
        }

        while true {
            let status = try await performWithSidecarRetry { client in
                try await client.getIndexJob(jobID: start.job_id)
            }
            indexProgress = status.progress
            indexStageText = stageLabel(status.stage)

            if status.status == "failed" {
                throw APIError(message: status.error ?? "Indexing failed")
            }
            if status.status == "completed" {
                break
            }
            try await Task.sleep(nanoseconds: 300_000_000)
        }
    }

    private func ensureSidecarClient() async throws -> SidecarAPIClient {
        if let client = sidecar.apiClient {
            do {
                try await client.health()
                _ = try await client.getSettings()
                return client
            } catch {
                sidecar.stop()
            }
        }
        try await sidecar.start()
        guard let client = sidecar.apiClient else {
            throw APIError(message: "Sidecar client unavailable: sidecar 시작 후에도 API 클라이언트가 생성되지 않았습니다.")
        }
        return client
    }

    private func stageLabel(_ stage: String) -> String {
        switch stage {
        case "scan":
            return "문서 분석 중"
        case "parse":
            return "텍스트 파싱 중"
        case "classify":
            return "문서 의미 분류 중"
        case "embed":
            return "검색 정확도 준비 중"
        case "store":
            return "작업 환경 최적화 중"
        case "done":
            return "완료"
        default:
            return "준비 중"
        }
    }

    private func parseTagText(_ text: String) -> [String] {
        text
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private func profileKey(from startup: StartupProfile) -> String {
        switch startup {
        case .fast:
            return "fast"
        case .recommended:
            return "balanced"
        case .deep:
            return "advanced"
        }
    }

    func chooseModelFile(for engine: LocalEngine) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.prompt = "선택"

        if panel.runModal() == .OK, let url = panel.url {
            switch engine {
            case .mlx:
                mlxModelPath = url.path
            case .llamaCPP:
                llamaModelPath = url.path
            }
            persistLocalModelPreferenceSnapshot()
        }
    }

    func applyDownloadedModel(_ model: ModelListItem) {
        let selectedPath = normalizedSelectionPath(for: model)
        guard !selectedPath.isEmpty else {
            return
        }
        localEngine = model.engine
        switch model.engine {
        case .mlx:
            mlxModelPath = selectedPath
        case .llamaCPP:
            llamaModelPath = selectedPath
        }
        persistLocalModelPreferenceSnapshot()
    }

    func selectInstalledModel(_ model: ModelListItem) async {
        let previousEngine = localEngine
        let previousMLXPath = mlxModelPath
        let previousLlamaPath = llamaModelPath

        applyDownloadedModel(model)
        isBusy = true
        defer { isBusy = false }

        do {
            try await syncWorkspaceAndSettings()
            try await refreshRemoteState()
        } catch {
            localEngine = previousEngine
            mlxModelPath = previousMLXPath
            llamaModelPath = previousLlamaPath
            persistLocalModelPreferenceSnapshot()
            handleViewModelError(error)
        }
    }

    func prepareRuntimeNow() async {
        isBusy = true
        defer { isBusy = false }

        do {
            _ = try await performWithSidecarRetry { client in
                try await prepareSelectedRuntime(using: client)
            }
        } catch {
            handleViewModelError(error)
        }
    }

    func downloadModel() async {
        let url = modelDownloadURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !url.isEmpty else {
            lastError = "모델 다운로드 URL을 입력해 주세요."
            return
        }

        isDownloadingModel = true
        defer { isDownloadingModel = false }

        do {
            let response = try await performWithSidecarRetry { client in
                try await client.downloadModel(
                    ModelDownloadRequest(
                        url: url,
                        engine: modelDownloadEngine,
                        filename: modelDownloadFilename.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : modelDownloadFilename
                    )
                )
            }
            switch response.engine {
            case .mlx:
                mlxModelPath = response.saved_path
            case .llamaCPP:
                llamaModelPath = response.saved_path
            }
            availableModels = try await performWithSidecarRetry { try await $0.listModels().models }
            modelDownloadURL = ""
            modelDownloadFilename = ""
        } catch {
            handleViewModelError(error)
        }
    }

    func installCatalogModel(_ modelID: String) async {
        isCatalogBusy = true
        defer { isCatalogBusy = false }

        do {
            _ = try await performWithSidecarRetry { client in
                try await client.installCatalogModel(modelID: modelID)
            }
            let catalog = try await performWithSidecarRetry { try await $0.getModelCatalog() }
            catalogDefaultProfile = catalog.default_profile
            catalogModels = catalog.models
            availableModels = try await performWithSidecarRetry { try await $0.listModels().models }
        } catch {
            handleViewModelError(error)
        }
    }

    func activateCatalogModel(_ modelID: String) async {
        isCatalogBusy = true
        defer { isCatalogBusy = false }

        do {
            let activated = try await performWithSidecarRetry { client in
                try await client.activateCatalogModel(modelID: modelID)
            }
            localEngine = activated.engine
            switch activated.engine {
            case .mlx:
                mlxModelPath = activated.model_path
                startupProfile = activated.profile == "fast" ? .fast : (activated.profile == "advanced" ? .deep : .recommended)
            case .llamaCPP:
                llamaModelPath = activated.model_path
                startupProfile = activated.profile == "fast" ? .fast : (activated.profile == "advanced" ? .deep : .recommended)
            }
            persistLocalModelPreferenceSnapshot()
            try await syncWorkspaceAndSettings()
            let catalog = try await performWithSidecarRetry { try await $0.getModelCatalog() }
            catalogDefaultProfile = catalog.default_profile
            catalogModels = catalog.models
            try await refreshRemoteState()
        } catch {
            handleViewModelError(error)
        }
    }

    func deleteCatalogModel(_ modelID: String) async {
        isCatalogBusy = true
        defer { isCatalogBusy = false }

        do {
            _ = try await performWithSidecarRetry { client in
                try await client.deleteCatalogModel(modelID: modelID)
            }
            let catalog = try await performWithSidecarRetry { try await $0.getModelCatalog() }
            catalogDefaultProfile = catalog.default_profile
            catalogModels = catalog.models
            availableModels = try await performWithSidecarRetry { try await $0.listModels().models }
        } catch {
            handleViewModelError(error)
        }
    }

    private func currentChatFilters() -> ChatFilters? {
        let category = chatFilterCategory.trimmingCharacters(in: .whitespacesAndNewlines)
        let project = chatFilterProject.trimmingCharacters(in: .whitespacesAndNewlines)
        let tags = parseTagText(chatFilterTags)
        let year = Int(chatFilterYear)

        if category.isEmpty, tags.isEmpty, year == nil, project.isEmpty {
            return nil
        }
        return ChatFilters(
            category: category.isEmpty ? nil : category,
            tags: tags,
            year: year,
            project: project.isEmpty ? nil : project,
            excluded: false
        )
    }

    private func prepareSelectedRuntime(using client: SidecarAPIClient) async throws -> RuntimePrepareResponse? {
        let modelPath: String?
        switch localEngine {
        case .mlx:
            let trimmed = mlxModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
            modelPath = trimmed.isEmpty ? nil : trimmed
        case .llamaCPP:
            let trimmed = llamaModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
            modelPath = trimmed.isEmpty ? nil : trimmed
        }

        do {
            let runtime = try await client.prepareRuntime(
                RuntimePrepareRequest(
                    engine: localEngine,
                    model_path: modelPath
                )
            )

            localRuntimeDetail = "\(runtime.engine.title): \(runtime.detail) (\(runtime.accelerator))"
            guard runtime.ready else {
                throw APIError(message: runtime.detail)
            }
            return runtime
        } catch {
            if isEndpointNotFound(error) {
                return nil
            }
            throw error
        }
    }

    private func handleViewModelError(_ error: Error) {
        if error is CancellationError {
            return
        }
        if isInvalidSessionTokenError(error) {
            if !isRecoveringSession {
                isRecoveringSession = true
                lastError = nil
                Task {
                    await recoverSessionFromInvalidToken()
                }
            }
            return
        }
        lastError = error.localizedDescription
    }

    private func recoverSessionFromInvalidToken() async {
        defer { isRecoveringSession = false }
        do {
            sidecar.stop()
            try await sidecar.start()
            try await refreshRemoteState()
            lastError = nil
        } catch {
            lastError = "세션 재연결 실패: \(error.localizedDescription)"
        }
    }

    private func isEndpointNotFound(_ error: Error) -> Bool {
        guard let apiError = error as? APIError else {
            return false
        }
        return apiError.message.contains("HTTP 404")
    }

    private func isInvalidSessionTokenError(_ error: Error) -> Bool {
        guard let apiError = error as? APIError else {
            return false
        }
        let lower = apiError.message.lowercased()
        return lower.contains("http 401") && lower.contains("invalid session token")
    }

    private func performWithSidecarRetry<T>(_ operation: (SidecarAPIClient) async throws -> T) async throws -> T {
        let client = try await ensureSidecarClient()
        do {
            return try await operation(client)
        } catch {
            guard isInvalidSessionTokenError(error) else {
                throw error
            }
            sidecar.stop()
            try await sidecar.start()
            guard let refreshed = sidecar.apiClient else {
                throw APIError(message: "세션 토큰을 갱신했지만 sidecar client를 다시 가져오지 못했습니다.")
            }
            return try await operation(refreshed)
        }
    }

    private func askLocal(query: String, appendUserMessage: Bool) async {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        isBusy = true
        defer { isBusy = false }

        if appendUserMessage {
            appendChatMessage(ChatMessage(source: .user, text: trimmed, timestamp: Date()))
        }
        citations = []
        highlightedCitationPath = nil
        syncActiveRoom(citations: citations)

        if chatResponseRoute == .apiOnly {
            await askExternalDirect(query: trimmed)
            return
        }

        do {
            var handled = false
            try await performWithSidecarRetry { client in
                _ = try await prepareSelectedRuntime(using: client)
                do {
                    let responseV2 = try await client.localChatV2(
                        LocalChatRequestV2(
                            query: trimmed,
                            mode: selectedMode,
                            conversation_id: activeConversationID,
                            session_id: activeConversationID,
                            top_k: nil,
                            filters: currentChatFilters(),
                            behavior_overrides: nil
                        )
                    )
                    citations = responseV2.citations
                    if let runtimeDetail = responseV2.runtime_detail, !runtimeDetail.isEmpty {
                        localRuntimeDetail = runtimeDetail
                    }
                    appendChatMessage(ChatMessage(localV2: responseV2, timestamp: Date()))
                    syncActiveRoom(citations: citations, latestQueryForDeepAnalysis: trimmed)
                    handled = true
                } catch {
                    #if DEBUG
                        let response = try await client.localChat(
                            LocalChatRequest(
                                query: trimmed,
                                mode: selectedMode,
                                conversation_id: activeConversationID,
                                top_k: nil,
                                filters: currentChatFilters()
                            )
                        )
                        citations = response.citations
                        if let runtimeDetail = response.runtime_detail, !runtimeDetail.isEmpty {
                            localRuntimeDetail = runtimeDetail
                        }
                        appendChatMessage(ChatMessage(local: response, timestamp: Date()))
                        syncActiveRoom(citations: citations, latestQueryForDeepAnalysis: trimmed)
                        handled = true
                    #else
                        throw error
                    #endif
                }
                return ()
            }
            if !handled {
                throw APIError(message: "로컬 채팅 응답을 처리하지 못했습니다.")
            }
            await refreshPostChatStateIfNeeded()
        } catch {
            handleViewModelError(error)
        }
    }

    private func refreshPostChatStateIfNeeded(force: Bool = false) async {
        let now = Date()
        if !force, now.timeIntervalSince(lastPostChatStateRefreshAt) < postChatStateRefreshInterval {
            return
        }
        do {
            let status = try await performWithSidecarRetry { try await $0.getStatus() }
            statusSnapshot = status
            let failures = try await performWithSidecarRetry { try await $0.getFailures() }
            failureItems = failures.failures
            lastPostChatStateRefreshAt = now
        } catch {
            if !isEndpointNotFound(error), lastError == nil {
                lastError = "상태 동기화 실패: \(error.localizedDescription)"
            }
        }
    }

    private func askExternalDirect(query: String) async {
        if privacyMode == .localOnly {
            lastError = "현재 응답 경로는 항상 API 호출이지만, 프라이버시 모드가 로컬 전용입니다."
            return
        }
        if privacyMode == .confirmBeforeExternal {
            pendingExternalDirectQuery = query
            needsExternalConfirmation = true
            return
        }
        await performDeepAnalysis(userConfirmed: true, queryOverride: query)
    }

    private func executePromptInjectionAction(_ action: SuggestedAction) async {
        guard let prompt = action.payload["prompt"]?.trimmingCharacters(in: .whitespacesAndNewlines), !prompt.isEmpty else {
            lastError = "액션 프롬프트가 비어 있어 실행할 수 없습니다."
            return
        }
        await recordActionMemoryEvent(action, summary: prompt)
        inputQuery = prompt
        await askLocal(query: prompt, appendUserMessage: true)
        inputQuery = ""
    }

    private func executeSystemAction(_ action: SuggestedAction) async {
        switch actionPermissionMode {
        case .askEveryTime:
            pendingSystemAction = action
        case .askPerAction:
            if approvedSystemActionKinds.contains(action.kind.rawValue) {
                performSystemAction(action)
            } else {
                pendingSystemAction = action
            }
        }
    }

    private func performSystemAction(_ action: SuggestedAction) {
        switch action.kind {
        case .openFile, .openSecond:
            guard let filePath = action.payload["file_path"], !filePath.isEmpty else {
                lastError = "열 파일 경로가 없습니다."
                return
            }
            let opened = NSWorkspace.shared.open(URL(fileURLWithPath: filePath))
            if !opened {
                lastError = "파일을 열지 못했습니다: \(filePath)"
            } else {
                Task {
                    await recordActionMemoryEvent(action, summary: "open file: \(filePath)")
                }
            }
        case .summarizeTop, .compareTop, .askFollowup, .showDiff, .createDraft, .showOtherCandidates, .makeShorter, .showPreviousCandidate:
            // These action kinds are handled via prompt injection path.
            break
        }
    }

    private func recordActionMemoryEvent(_ action: SuggestedAction, summary: String) async {
        var metadata: [String: JSONValue] = [
            "action_kind": .string(action.kind.rawValue),
            "execution_mode": .string(action.execution_mode.rawValue),
        ]
        for (key, value) in action.payload {
            metadata[key] = .string(value)
        }
        do {
            try await writeMemoryEvent(
                eventType: .actionExecuted,
                summary: summary,
                relatedFileIDs: action.payload["file_path"].map { [$0] } ?? [],
                relatedActionIDs: [action.action_id],
                metadata: metadata,
                importance: 0.56
            )
        } catch {
            handleViewModelError(error)
        }
    }

    @discardableResult
    private func writeMemoryEvent(
        eventType: MemoryEventType,
        summary: String,
        relatedFileIDs: [String],
        relatedActionIDs: [String],
        metadata: [String: JSONValue],
        importance: Double
    ) async throws -> MemoryEventResponse {
        try await performWithSidecarRetry { client in
            try await client.writeMemoryEvent(
                MemoryEventRequest(
                    event_type: eventType,
                    session_id: activeConversationID,
                    workspace_id: currentWorkspaceID(),
                    summary: summary,
                    related_file_ids: relatedFileIDs,
                    related_action_ids: relatedActionIDs,
                    metadata_json: metadata,
                    importance: importance
                )
            )
        }
    }

    private func currentWorkspaceID() -> String {
        let included = includedFolderURLs
            .map(\.path)
            .map { NSString(string: $0).expandingTildeInPath }
            .sorted()
        let excluded = excludedPaths
            .map { NSString(string: $0).expandingTildeInPath }
            .sorted()
        let combined = included.map { "+:\($0)" } + excluded.map { "-:\($0)" }
        let joined = combined.joined(separator: "\n")
        let digest = Insecure.SHA1.hash(data: Data(joined.utf8))
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return String(hex.prefix(16))
    }

    private func loadApprovedSystemActionKinds() {
        let stored = UserDefaults.standard.stringArray(forKey: approvedSystemActionsDefaultsKey) ?? []
        approvedSystemActionKinds = Set(stored)
    }

    private func loadChatResponseRoute() {
        let raw = UserDefaults.standard.string(forKey: chatResponseRouteDefaultsKey) ?? ChatResponseRoute.hybrid.rawValue
        chatResponseRoute = ChatResponseRoute(rawValue: raw) ?? .hybrid
        switch chatResponseRoute {
        case .localOnly:
            privacyMode = .localOnly
        case .hybrid, .apiOnly:
            if privacyMode == .localOnly {
                privacyMode = .hybrid
            }
        }
    }

    private func loadLocalModelPreferenceSnapshot() {
        let defaults = UserDefaults.standard
        if let rawPreset = defaults.string(forKey: quickInferencePresetDefaultsKey),
           let preset = QuickInferencePreset(rawValue: rawPreset)
        {
            quickInferencePreset = preset
            startupProfile = preset.startupProfile
        }
        if let rawEngine = defaults.string(forKey: localEngineDefaultsKey),
           let engine = LocalEngine(rawValue: rawEngine)
        {
            localEngine = engine
        }
        if let savedMLXPath = defaults.string(forKey: mlxModelPathDefaultsKey) {
            mlxModelPath = savedMLXPath
        }
        if let savedLlamaPath = defaults.string(forKey: llamaModelPathDefaultsKey) {
            llamaModelPath = savedLlamaPath
        }
    }

    private func persistLocalModelPreferenceSnapshot() {
        let defaults = UserDefaults.standard
        defaults.set(quickInferencePreset.rawValue, forKey: quickInferencePresetDefaultsKey)
        defaults.set(localEngine.rawValue, forKey: localEngineDefaultsKey)
        defaults.set(mlxModelPath, forKey: mlxModelPathDefaultsKey)
        defaults.set(llamaModelPath, forKey: llamaModelPathDefaultsKey)
    }

    private func loadSecretAPIKeys() {
        openAIAPIKey = AppSecretStore.read("openai_api_key") ?? ""
        anthropicAPIKey = AppSecretStore.read("anthropic_api_key") ?? ""
    }

    @discardableResult
    private func persistSecretAPIKeys() -> Bool {
        var changed = false

        let newOpenAI = openAIAPIKey.trimmingCharacters(in: .whitespacesAndNewlines)
        let oldOpenAI = AppSecretStore.read("openai_api_key") ?? ""
        if newOpenAI != oldOpenAI {
            changed = true
            if newOpenAI.isEmpty {
                AppSecretStore.delete("openai_api_key")
            } else {
                _ = AppSecretStore.save(newOpenAI, for: "openai_api_key")
            }
        }

        let newAnthropic = anthropicAPIKey.trimmingCharacters(in: .whitespacesAndNewlines)
        let oldAnthropic = AppSecretStore.read("anthropic_api_key") ?? ""
        if newAnthropic != oldAnthropic {
            changed = true
            if newAnthropic.isEmpty {
                AppSecretStore.delete("anthropic_api_key")
            } else {
                _ = AppSecretStore.save(newAnthropic, for: "anthropic_api_key")
            }
        }
        return changed
    }

    private func syncQuickInferencePresetFromProfile() {
        switch startupProfile {
        case .fast:
            quickInferencePreset = .fast
        case .recommended:
            quickInferencePreset = .quality
        case .deep:
            quickInferencePreset = .highQuality
        }
    }

    private func normalizedSelectionPath(for model: ModelListItem) -> String {
        let rawPath = model.path.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !rawPath.isEmpty else { return "" }
        let url = URL(fileURLWithPath: rawPath).standardizedFileURL

        if model.engine == .mlx {
            var isDirectory = ObjCBool(false)
            if FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), !isDirectory.boolValue {
                let parent = url.deletingLastPathComponent()
                if parent.lastPathComponent.lowercased() == "mlx" {
                    return ""
                }
                return parent.path
            }
        }

        return url.path
    }

    private static func isDisplayableModelArtifact(_ model: ModelListItem) -> Bool {
        let name = model.file_name.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if name.isEmpty || name.hasPrefix(".") {
            return false
        }
        if name == ".gitignore" || name.hasSuffix(".metadata") || name == "catalog_state.json" {
            return false
        }
        if model.engine == .llamaCPP {
            return name.hasSuffix(".gguf")
        }
        return true
    }

    private static func samePath(_ lhs: String, _ rhs: String) -> Bool {
        URL(fileURLWithPath: lhs).standardizedFileURL.path == URL(fileURLWithPath: rhs).standardizedFileURL.path
    }

    private func persistApprovedSystemActionKinds() {
        UserDefaults.standard.set(Array(approvedSystemActionKinds).sorted(), forKey: approvedSystemActionsDefaultsKey)
    }

    private func loadChatRooms() {
        if
            let data = UserDefaults.standard.data(forKey: chatRoomsDefaultsKey),
            let decoded = try? JSONDecoder().decode([ChatRoom].self, from: data),
            !decoded.isEmpty
        {
            let migrated = decoded.map { room -> ChatRoom in
                var updated = room
                let normalized = room.title.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                if normalized == "plos chat" || normalized == "chatgpt" {
                    updated.title = "새 채팅"
                }
                return updated
            }
            chatRooms = migrated.sorted { $0.updatedAt > $1.updatedAt }
        } else {
            chatRooms = [ChatRoom.makeDefault()]
        }

        let preferredRoomID = UserDefaults.standard.string(forKey: activeChatRoomDefaultsKey)
        if let preferredRoomID, chatRooms.contains(where: { $0.id == preferredRoomID }) {
            selectedChatRoomID = preferredRoomID
        } else if let first = chatRooms.first {
            selectedChatRoomID = first.id
        }

        if let room = chatRooms.first(where: { $0.id == selectedChatRoomID }) {
            chatMessages = room.messages
            citations = room.citations
        } else {
            chatMessages = []
            citations = []
        }
    }

    private func persistChatRooms() {
        guard let encoded = try? JSONEncoder().encode(chatRooms) else {
            return
        }
        UserDefaults.standard.set(encoded, forKey: chatRoomsDefaultsKey)
        let activeID = selectedChatRoomID.isEmpty ? chatRooms.first?.id : selectedChatRoomID
        if let activeID {
            UserDefaults.standard.set(activeID, forKey: activeChatRoomDefaultsKey)
        }
    }

    private func appendChatMessage(_ message: ChatMessage) {
        chatMessages.append(message)
        syncActiveRoom(messages: chatMessages)
    }

    private func syncActiveRoom(
        messages: [ChatMessage]? = nil,
        citations: [Citation]? = nil,
        latestQueryForDeepAnalysis: String? = nil
    ) {
        guard let index = activeRoomIndex else {
            return
        }
        if let messages {
            chatRooms[index].messages = messages
        }
        if let citations {
            chatRooms[index].citations = citations
        }
        if let latestQueryForDeepAnalysis {
            chatRooms[index].latestQueryForDeepAnalysis = latestQueryForDeepAnalysis
        }
        chatRooms[index].updatedAt = Date()
        if chatRooms[index].title == "새 채팅" {
            if let firstUser = chatRooms[index].messages.first(where: { $0.source == .user })?.text {
                let trimmed = firstUser.trimmingCharacters(in: .whitespacesAndNewlines)
                if !trimmed.isEmpty {
                    chatRooms[index].title = String(trimmed.prefix(26))
                }
            }
        }
        // Keep newest rooms at top like GPT-style history list.
        let activeID = chatRooms[index].id
        chatRooms.sort { $0.updatedAt > $1.updatedAt }
        if selectedChatRoomID != activeID {
            selectedChatRoomID = activeID
        }
        persistChatRooms()
    }
}
