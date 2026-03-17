import AppKit
import Foundation

enum OnboardingStep: Int, CaseIterable {
    case welcome
    case dataSelection
    case startProfile
    case privacyInfo
    case indexing
    case ready

    var title: String {
        switch self {
        case .welcome: return "Welcome"
        case .dataSelection: return "자료 선택"
        case .startProfile: return "시작 방식"
        case .privacyInfo: return "프라이버시"
        case .indexing: return "인덱싱"
        case .ready: return "첫 질문"
        }
    }
}

@MainActor
final class AppViewModel: ObservableObject {
    @Published var onboardingStep: OnboardingStep = .welcome
    @Published var includedFolderURLs: [URL] = []
    @Published var excludedPaths: [String] = []
    @Published var startupProfile: StartupProfile = .recommended
    @Published var privacyMode: PrivacyMode = .hybrid
    @Published var defaultWorkMode: WorkMode = .general

    @Published var indexProgress: Double = 0
    @Published var indexStageText: String = "대기 중"
    @Published var hasFinishedOnboarding: Bool = false

    @Published var inputQuery: String = ""
    @Published var selectedMode: WorkMode = .general
    @Published var chatMessages: [ChatMessage] = []
    @Published var citations: [Citation] = []
    @Published var selectedProvider: String = "openai"

    @Published var statusSnapshot: StatusSnapshot?
    @Published var failureItems: [FailureItem] = []
    @Published var isBusy: Bool = false
    @Published var lastError: String?
    @Published var needsExternalConfirmation = false

    private let sidecar = SidecarProcessManager()
    private let bookmarkStore = BookmarkStore()
    private let onboardingDefaultsKey = "local_ai_onboarding_finished"
    private var latestQueryForDeepAnalysis: String?

    var currentPrivacyBadge: String {
        switch privacyMode {
        case .localOnly: return "Local Only"
        case .hybrid: return "Hybrid"
        case .confirmBeforeExternal: return "Confirm"
        }
    }

    func bootstrap() async {
        hasFinishedOnboarding = UserDefaults.standard.bool(forKey: onboardingDefaultsKey)
        includedFolderURLs = bookmarkStore.loadURLs()

        do {
            try await sidecar.start()
            if hasFinishedOnboarding {
                try await refreshRemoteState()
            }
        } catch {
            lastError = error.localizedDescription
        }
    }

    func shutdown() {
        sidecar.stop()
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
            bookmarkStore.save(urls: includedFolderURLs)
        }
    }

    func removeFolder(_ path: String) {
        includedFolderURLs.removeAll { $0.path == path }
        bookmarkStore.save(urls: includedFolderURLs)
    }

    func goToNextOnboardingStep() {
        guard let next = OnboardingStep(rawValue: onboardingStep.rawValue + 1) else { return }
        onboardingStep = next
    }

    func goToPreviousOnboardingStep() {
        guard let prev = OnboardingStep(rawValue: onboardingStep.rawValue - 1) else { return }
        onboardingStep = prev
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
            lastError = error.localizedDescription
        }
    }

    func finalizeOnboarding() {
        hasFinishedOnboarding = true
        UserDefaults.standard.set(true, forKey: onboardingDefaultsKey)
    }

    func askLocal() async {
        let query = inputQuery.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !query.isEmpty else { return }
        guard let client = sidecar.apiClient else {
            lastError = "Sidecar가 준비되지 않았습니다."
            return
        }

        isBusy = true
        defer { isBusy = false }

        chatMessages.append(ChatMessage(source: .user, text: query, timestamp: Date()))
        inputQuery = ""

        do {
            let response = try await client.localChat(
                LocalChatRequest(query: query, mode: selectedMode, conversation_id: nil, top_k: nil)
            )
            latestQueryForDeepAnalysis = query
            citations = response.citations
            chatMessages.append(ChatMessage(source: .local, text: response.answer, timestamp: Date()))
            try await refreshRemoteState()
        } catch {
            lastError = error.localizedDescription
        }
    }

    func deepAnalyzeTapped() {
        if privacyMode == .confirmBeforeExternal {
            needsExternalConfirmation = true
        } else {
            Task {
                await performDeepAnalysis(userConfirmed: privacyMode != .localOnly)
            }
        }
    }

    func performDeepAnalysis(userConfirmed: Bool) async {
        guard let query = latestQueryForDeepAnalysis else {
            lastError = "먼저 로컬 질문을 수행해 주세요."
            return
        }
        guard let client = sidecar.apiClient else {
            lastError = "Sidecar가 준비되지 않았습니다."
            return
        }

        isBusy = true
        defer { isBusy = false }

        do {
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
            try await refreshRemoteState()
        } catch {
            lastError = error.localizedDescription
        }
    }

    func updateSettings() async {
        guard let client = sidecar.apiClient else { return }

        isBusy = true
        defer { isBusy = false }

        do {
            let payload = SettingsModel(
                privacy_mode: privacyMode,
                startup_profile: startupProfile,
                model_profile: startupProfile.rawValue.lowercased(),
                reindex_policy: "filewatch_incremental",
                language: "ko-KR"
            )
            _ = try await client.updateSettings(payload)
            try await refreshRemoteState()
        } catch {
            lastError = error.localizedDescription
        }
    }

    func triggerFullReindex() async {
        isBusy = true
        defer { isBusy = false }

        do {
            try await runIndexing(scope: "full")
            try await refreshRemoteState()
        } catch {
            lastError = error.localizedDescription
        }
    }

    func refreshRemoteState() async throws {
        guard let client = sidecar.apiClient else { return }

        let remoteSettings = try await client.getSettings()
        privacyMode = remoteSettings.privacy_mode
        startupProfile = remoteSettings.startup_profile

        let status = try await client.getStatus()
        statusSnapshot = status

        let failures = try await client.getFailures()
        failureItems = failures.failures
    }

    private func syncWorkspaceAndSettings() async throws {
        guard let client = sidecar.apiClient else {
            throw APIError(message: "Sidecar client unavailable")
        }

        bookmarkStore.save(urls: includedFolderURLs)

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
                model_profile: startupProfile.rawValue.lowercased(),
                reindex_policy: "filewatch_incremental",
                language: "ko-KR"
            )
        )
    }

    private func runIndexing(scope: String) async throws {
        guard let client = sidecar.apiClient else {
            throw APIError(message: "Sidecar client unavailable")
        }

        let start = try await client.startIndexJob(scope: scope)
        var terminalStates: Set<String> = ["completed", "failed"]

        while true {
            let status = try await client.getIndexJob(jobID: start.job_id)
            indexProgress = status.progress
            indexStageText = stageLabel(stage: status.stage)

            if terminalStates.contains(status.status) {
                if status.status == "failed" {
                    throw APIError(message: status.error ?? "Indexing failed")
                }
                break
            }
            try await Task.sleep(nanoseconds: 300_000_000)
        }
    }

    private func stageLabel(stage: String) -> String {
        switch stage {
        case "scan": return "문서 분석 중"
        case "parse": return "텍스트 파싱 중"
        case "embed": return "검색 정확도 준비 중"
        case "store": return "작업 환경 최적화 중"
        case "done": return "완료"
        default: return "준비 중"
        }
    }
}
