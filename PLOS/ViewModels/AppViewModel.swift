import AppKit
import Combine
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
            urls.append(resolved)
        }
        return urls
    }

    func startAccessing(urls: [URL]) {
        for url in urls {
            _ = url.startAccessingSecurityScopedResource()
        }
    }

    func stopAccessing(urls: [URL]) {
        for url in urls {
            url.stopAccessingSecurityScopedResource()
        }
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

enum ProcessingRoute {
    case local
    case external
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
    @Published var currentRoute: ProcessingRoute = .local
    @Published var chatFilterCategory = ""
    @Published var chatFilterTags = ""
    @Published var chatFilterYear = ""
    @Published var chatFilterProject = ""
    @Published var isCitationDrawerVisible = false
    @Published var actionPermissionMode: ActionPermissionMode = .askPerAction
    @Published var pendingSystemAction: SuggestedAction?
    @Published var highlightedCitationPath: String?

    @Published var indexProgress: Double = 0
    @Published var indexStageText = "준비 중"

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

    private let sidecar = SidecarProcessManager()
    private let bookmarkStore = BookmarkStore()
    private let onboardingDefaultsKey = "local_ai_onboarding_finished"
    private let approvedSystemActionsDefaultsKey = "local_ai_approved_system_actions"
    private var approvedSystemActionKinds: Set<String> = []
    private var latestQueryForDeepAnalysis: String?

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

    func bootstrap() async {
        hasFinishedOnboarding = UserDefaults.standard.bool(forKey: onboardingDefaultsKey)
        includedFolderURLs = bookmarkStore.loadURLs()
        bookmarkStore.startAccessing(urls: includedFolderURLs)
        loadApprovedSystemActionKinds()

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
        bookmarkStore.stopAccessing(urls: includedFolderURLs)
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
            bookmarkStore.startAccessing(urls: newURLs)
            includedFolderURLs.append(contentsOf: newURLs)
            persistBookmarks()
        }
    }

    func removeFolder(_ path: String) {
        let removed = includedFolderURLs.filter { $0.path == path }
        bookmarkStore.stopAccessing(urls: removed)
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

    func performDeepAnalysis(userConfirmed: Bool) async {
        guard let query = latestQueryForDeepAnalysis else {
            lastError = "먼저 로컬 질문을 실행해 주세요."
            return
        }

        isBusy = true
        defer { isBusy = false }

        do {
            let client = try await ensureSidecarClient()
            let response = try await client.deepAnalysis(
                DeepAnalysisRequest(
                    query: query,
                    mode: selectedMode,
                    provider: selectedProvider,
                    selected_citations: citations,
                    user_confirmed: userConfirmed
                )
            )
            chatMessages.append(ChatMessage(source: .external, text: response.answer, timestamp: Date()))
            currentRoute = .external
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
            try await syncWorkspaceAndSettings()
            try await refreshRemoteState()
        } catch {
            handleViewModelError(error)
        }
    }

    func refreshRemoteState() async throws {
        let client = try await ensureSidecarClient()

        let settings = try await client.getSettings()
        privacyMode = settings.privacy_mode
        startupProfile = settings.startup_profile
        localEngine = settings.local_engine ?? .mlx
        mlxModelPath = settings.mlx_model_path ?? ""
        llamaModelPath = settings.llama_model_path ?? ""
        actionPermissionMode = settings.action_permission_mode ?? .askPerAction

        let status = try await client.getStatus()
        statusSnapshot = status

        let failures = try await client.getFailures()
        failureItems = failures.failures

        do {
            availableModels = try await client.listModels().models
        } catch {
            if !isEndpointNotFound(error) {
                throw error
            }
            availableModels = []
        }

        do {
            let catalog = try await client.getModelCatalog()
            catalogDefaultProfile = catalog.default_profile
            catalogModels = catalog.models
        } catch {
            if !isEndpointNotFound(error) {
                throw error
            }
            catalogModels = []
        }

        do {
            let docs = try await client.listDocuments(
                search: documentSearchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : documentSearchText,
                category: documentFilterCategory.isEmpty ? nil : documentFilterCategory,
                tags: parseTagText(documentFilterTag),
                year: Int(documentFilterYear),
                project: documentFilterProject.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : documentFilterProject,
                excluded: showExcludedDocuments ? true : false
            )
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
    }

    func refreshDocuments() async {
        do {
            let client = try await ensureSidecarClient()
            let docs = try await client.listDocuments(
                search: documentSearchText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : documentSearchText,
                category: documentFilterCategory.isEmpty ? nil : documentFilterCategory,
                tags: parseTagText(documentFilterTag),
                year: Int(documentFilterYear),
                project: documentFilterProject.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : documentFilterProject,
                excluded: showExcludedDocuments ? true : false
            )
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
            let client = try await ensureSidecarClient()
            _ = try await client.updateDocumentMetadata(
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
            let client = try await ensureSidecarClient()
            _ = try await client.reclassifyDocument(docID: docID)
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
        let client = try await ensureSidecarClient()

        persistBookmarks()

        _ = try await client.updateWorkspace(
            WorkspaceUpdateRequest(
                included_paths: includedFolderURLs.map(\.path),
                excluded_paths: excludedPaths,
                startup_profile: startupProfile,
                default_mode: defaultWorkMode
            )
        )

        _ = try await client.updateSettings(
            SettingsModel(
                privacy_mode: privacyMode,
                startup_profile: startupProfile,
                model_profile: profileKey(from: startupProfile),
                local_engine: localEngine,
                mlx_model_path: mlxModelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : mlxModelPath,
                llama_model_path: llamaModelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : llamaModelPath,
                reindex_policy: "filewatch_incremental",
                language: "auto",
                action_permission_mode: actionPermissionMode
                )
            )

        do {
            _ = try await prepareSelectedRuntime(using: client)
        } catch {
            if !isEndpointNotFound(error) {
                localRuntimeDetail = "엔진 준비 경고: \(error.localizedDescription)"
            }
        }
    }

    private func runIndexing(scope: String) async throws {
        let client = try await ensureSidecarClient()

        let start = try await client.startIndexJob(scope: scope)

        while true {
            let status = try await client.getIndexJob(jobID: start.job_id)
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
        }
    }

    func applyDownloadedModel(_ model: ModelListItem) {
        switch model.engine {
        case .mlx:
            mlxModelPath = model.path
        case .llamaCPP:
            llamaModelPath = model.path
        }
    }

    func prepareRuntimeNow() async {
        isBusy = true
        defer { isBusy = false }

        do {
            let client = try await ensureSidecarClient()
            _ = try await prepareSelectedRuntime(using: client)
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
            let client = try await ensureSidecarClient()
            let response = try await client.downloadModel(
                ModelDownloadRequest(
                    url: url,
                    engine: modelDownloadEngine,
                    filename: modelDownloadFilename.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : modelDownloadFilename
                )
            )
            switch response.engine {
            case .mlx:
                mlxModelPath = response.saved_path
            case .llamaCPP:
                llamaModelPath = response.saved_path
            }
            availableModels = try await client.listModels().models
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
            let client = try await ensureSidecarClient()
            _ = try await client.installCatalogModel(modelID: modelID)
            let catalog = try await client.getModelCatalog()
            catalogDefaultProfile = catalog.default_profile
            catalogModels = catalog.models
            availableModels = try await client.listModels().models
        } catch {
            handleViewModelError(error)
        }
    }

    func activateCatalogModel(_ modelID: String) async {
        isCatalogBusy = true
        defer { isCatalogBusy = false }

        do {
            let client = try await ensureSidecarClient()
            let activated = try await client.activateCatalogModel(modelID: modelID)
            localEngine = activated.engine
            switch activated.engine {
            case .mlx:
                mlxModelPath = activated.model_path
                startupProfile = activated.profile == "fast" ? .fast : (activated.profile == "advanced" ? .deep : .recommended)
            case .llamaCPP:
                llamaModelPath = activated.model_path
                startupProfile = activated.profile == "fast" ? .fast : (activated.profile == "advanced" ? .deep : .recommended)
            }
            try await syncWorkspaceAndSettings()
            let catalog = try await client.getModelCatalog()
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
            let client = try await ensureSidecarClient()
            _ = try await client.deleteCatalogModel(modelID: modelID)
            let catalog = try await client.getModelCatalog()
            catalogDefaultProfile = catalog.default_profile
            catalogModels = catalog.models
            availableModels = try await client.listModels().models
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
        lastError = error.localizedDescription
    }

    private func isEndpointNotFound(_ error: Error) -> Bool {
        guard let apiError = error as? APIError else {
            return false
        }
        return apiError.message.contains("HTTP 404")
    }

    private func askLocal(query: String, appendUserMessage: Bool) async {
        let trimmed = query.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return }

        isBusy = true
        defer { isBusy = false }

        if appendUserMessage {
            chatMessages.append(ChatMessage(source: .user, text: trimmed, timestamp: Date()))
        }
        citations = []
        highlightedCitationPath = nil

        do {
            let client = try await ensureSidecarClient()
            _ = try await prepareSelectedRuntime(using: client)
            let response = try await client.localChat(
                LocalChatRequest(
                    query: trimmed,
                    mode: selectedMode,
                    conversation_id: nil,
                    top_k: nil,
                    filters: currentChatFilters()
                )
            )
            citations = response.citations
            if let runtimeDetail = response.runtime_detail, !runtimeDetail.isEmpty {
                localRuntimeDetail = runtimeDetail
            }
            chatMessages.append(ChatMessage(local: response, timestamp: Date()))
            currentRoute = .local
            latestQueryForDeepAnalysis = trimmed
            try await refreshRemoteState()
        } catch {
            handleViewModelError(error)
        }
    }

    private func executePromptInjectionAction(_ action: SuggestedAction) async {
        guard let prompt = action.payload["prompt"]?.trimmingCharacters(in: .whitespacesAndNewlines), !prompt.isEmpty else {
            lastError = "액션 프롬프트가 비어 있어 실행할 수 없습니다."
            return
        }
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
        case .openFile:
            guard let filePath = action.payload["file_path"], !filePath.isEmpty else {
                lastError = "열 파일 경로가 없습니다."
                return
            }
            let resolvedPath = URL(fileURLWithPath: filePath).standardizedFileURL.path
            guard isAllowedOpenFilePath(resolvedPath) else {
                lastError = "허용되지 않은 경로라 파일을 열 수 없습니다: \(resolvedPath)"
                return
            }
            let opened = NSWorkspace.shared.open(URL(fileURLWithPath: resolvedPath))
            if !opened {
                lastError = "파일을 열지 못했습니다: \(resolvedPath)"
            }
        case .summarizeTop, .compareTop, .askFollowup:
            // These action kinds are handled via prompt injection path.
            break
        }
    }

    private func loadApprovedSystemActionKinds() {
        let stored = UserDefaults.standard.stringArray(forKey: approvedSystemActionsDefaultsKey) ?? []
        approvedSystemActionKinds = Set(stored)
    }

    private func persistApprovedSystemActionKinds() {
        UserDefaults.standard.set(Array(approvedSystemActionKinds).sorted(), forKey: approvedSystemActionsDefaultsKey)
    }

    private func isAllowedOpenFilePath(_ candidatePath: String) -> Bool {
        for root in includedFolderURLs {
            let rootPath = root.standardizedFileURL.path
            if candidatePath == rootPath {
                return true
            }
            if candidatePath.hasPrefix(rootPath.hasSuffix("/") ? rootPath : "\(rootPath)/") {
                return true
            }
        }
        return false
    }
}
