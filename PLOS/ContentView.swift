import AppKit
import Combine
import Foundation
import SwiftUI

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

// MARK: - Infra

struct APIError: Error, LocalizedError {
    let message: String

    var errorDescription: String? {
        message
    }
}

final class SidecarAPIClient {
    private let baseURL: URL
    private let sessionToken: String
    private let urlSession: URLSession
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder

    init(baseURL: URL, sessionToken: String, urlSession: URLSession = .shared) {
        self.baseURL = baseURL
        self.sessionToken = sessionToken
        self.urlSession = urlSession

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            let raw = try container.decode(String.self)
            if let date = SidecarAPIClient.iso8601WithFractional.date(from: raw) ?? SidecarAPIClient.iso8601.date(from: raw) {
                return date
            }
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Invalid date: \(raw)")
        }
        self.decoder = decoder

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        self.encoder = encoder
    }

    private static let iso8601: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()

    private static let iso8601WithFractional: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    func health() async throws {
        _ = try await request(path: "/health", method: "GET") as [String: String]
    }

    func updateWorkspace(_ payload: WorkspaceUpdateRequest) async throws -> WorkspaceResponse {
        try await request(path: "/v1/workspaces", method: "POST", body: payload)
    }

    func startIndexJob(scope: String) async throws -> IndexJobStartResponse {
        try await request(path: "/v1/index/jobs", method: "POST", body: IndexJobRequest(scope: scope))
    }

    func getIndexJob(jobID: String) async throws -> IndexJobStatus {
        try await request(path: "/v1/index/jobs/\(jobID)", method: "GET")
    }

    func getFailures() async throws -> FailureListResponse {
        try await request(path: "/v1/index/failures", method: "GET")
    }

    func localChat(_ payload: LocalChatRequest) async throws -> LocalChatResponse {
        try await request(path: "/v1/chat/local", method: "POST", body: payload)
    }

    func deepAnalysis(_ payload: DeepAnalysisRequest) async throws -> DeepAnalysisResponse {
        try await request(path: "/v1/chat/deep-analysis", method: "POST", body: payload)
    }

    func getSettings() async throws -> SettingsModel {
        try await request(path: "/v1/settings", method: "GET")
    }

    func updateSettings(_ payload: SettingsModel) async throws -> SettingsModel {
        try await request(path: "/v1/settings", method: "PUT", body: payload)
    }

    func getStatus() async throws -> StatusSnapshot {
        try await request(path: "/v1/status", method: "GET")
    }

    private func request<T: Decodable>(path: String, method: String) async throws -> T {
        try await request(path: path, method: method, body: Optional<String>.none as String?)
    }

    private func request<T: Decodable, U: Encodable>(path: String, method: String, body: U?) async throws -> T {
        guard let url = URL(string: path, relativeTo: baseURL) else {
            throw APIError(message: "Invalid URL path: \(path)")
        }

        var req = URLRequest(url: url)
        req.httpMethod = method
        req.setValue(sessionToken, forHTTPHeaderField: "x-session-token")

        if let body {
            req.httpBody = try encoder.encode(body)
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }

        let (data, response) = try await urlSession.data(for: req)
        guard let http = response as? HTTPURLResponse else {
            throw APIError(message: "Invalid response")
        }

        guard (200 ..< 300).contains(http.statusCode) else {
            let text = String(data: data, encoding: .utf8) ?? "unknown error"
            throw APIError(message: "HTTP \(http.statusCode): \(text)")
        }

        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            let payload = String(data: data, encoding: .utf8) ?? ""
            throw APIError(message: "Decode failed: \(error.localizedDescription). Payload: \(payload)")
        }
    }
}

@MainActor
final class SidecarProcessManager: ObservableObject {
    @Published private(set) var isRunning = false

    private(set) var sessionToken = UUID().uuidString
    private var process: Process?
    private(set) var apiClient: SidecarAPIClient?
    private var startupLogs = ""

    private let host = "127.0.0.1"
    private let port = 8777

    func start() async throws {
        if isRunning, apiClient != nil {
            return
        }

        startupLogs = ""
        let sidecarDirectory = try resolveSidecarDirectory()
        let python = try ensureSidecarEnvironment(sidecarDirectory: sidecarDirectory)
        let dataDirectory = sidecarDirectory.appendingPathComponent("data")
        try FileManager.default.createDirectory(at: dataDirectory, withIntermediateDirectories: true)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            python,
            "-m", "uvicorn",
            "local_ai_core.main:app",
            "--host", host,
            "--port", String(port)
        ]
        process.currentDirectoryURL = sidecarDirectory

        var env = ProcessInfo.processInfo.environment
        env["LOCAL_AI_SESSION_TOKEN"] = sessionToken
        env["LOCAL_AI_DATA_DIR"] = dataDirectory.path
        env["PYTHONUNBUFFERED"] = "1"
        process.environment = env

        let pipe = Pipe()
        attachLogCapture(to: pipe)
        process.standardOutput = pipe
        process.standardError = pipe

        do {
            try process.run()
        } catch {
            throw APIError(message: "Sidecar 실행 실패: \(error.localizedDescription)")
        }

        self.process = process

        let client = SidecarAPIClient(
            baseURL: URL(string: "http://\(host):\(port)")!,
            sessionToken: sessionToken
        )
        self.apiClient = client

        try await waitUntilHealthy(client: client)
        isRunning = true
    }

    func stop() {
        process?.terminate()
        process = nil
        apiClient = nil
        isRunning = false
    }

    private func waitUntilHealthy(client: SidecarAPIClient) async throws {
        for _ in 0 ..< 80 {
            if let process, !process.isRunning {
                throw APIError(message: "Sidecar가 시작 직후 종료되었습니다.\n\(startupLogSummary())")
            }
            do {
                try await client.health()
                return
            } catch {
                try await Task.sleep(nanoseconds: 250_000_000)
            }
        }
        throw APIError(message: "Sidecar가 정상 상태가 되지 않았습니다. (health timeout)\n\(startupLogSummary())")
    }

    private func attachLogCapture(to pipe: Pipe) {
        pipe.fileHandleForReading.readabilityHandler = { [weak self] handle in
            let data = handle.availableData
            guard !data.isEmpty else { return }
            guard let text = String(data: data, encoding: .utf8) else { return }
            Task { @MainActor in
                guard let self else { return }
                self.startupLogs.append(text)
                if self.startupLogs.count > 8000 {
                    self.startupLogs.removeFirst(self.startupLogs.count - 8000)
                }
            }
        }
    }

    private func startupLogSummary() -> String {
        let trimmed = startupLogs.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return "sidecar 로그가 비어 있습니다."
        }
        return "최근 sidecar 로그:\n\(trimmed.suffix(1200))"
    }

    private func ensureSidecarEnvironment(sidecarDirectory: URL) throws -> String {
        let fm = FileManager.default
        let venvPython = sidecarDirectory.appendingPathComponent(".venv/bin/python3")

        if !fm.fileExists(atPath: venvPython.path) {
            try runCommand(
                executable: "/usr/bin/env",
                arguments: ["python3", "-m", "venv", ".venv"],
                cwd: sidecarDirectory,
                step: "Python 가상환경 생성"
            )
        }

        if !hasRequiredModules(python: venvPython.path, cwd: sidecarDirectory) {
            try runCommand(
                executable: venvPython.path,
                arguments: ["-m", "pip", "install", "-e", "."],
                cwd: sidecarDirectory,
                step: "sidecar 의존성 설치"
            )
        }

        return venvPython.path
    }

    private func hasRequiredModules(python: String, cwd: URL) -> Bool {
        do {
            try runCommand(
                executable: python,
                arguments: ["-c", "import uvicorn, fastapi, httpx, pydantic, lancedb"],
                cwd: cwd,
                step: "sidecar 모듈 점검"
            )
            return true
        } catch {
            return false
        }
    }

    private func runCommand(
        executable: String,
        arguments: [String],
        cwd: URL,
        step: String
    ) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.currentDirectoryURL = cwd

        let out = Pipe()
        let err = Pipe()
        process.standardOutput = out
        process.standardError = err

        do {
            try process.run()
        } catch {
            throw APIError(message: "\(step) 실행 실패: \(error.localizedDescription)")
        }

        process.waitUntilExit()
        let stdout = String(data: out.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
        let stderr = String(data: err.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""

        guard process.terminationStatus == 0 else {
            let log = (stderr.isEmpty ? stdout : stderr).trimmingCharacters(in: .whitespacesAndNewlines)
            throw APIError(message: "\(step) 실패 (exit \(process.terminationStatus))\n\(log)")
        }
    }

    private func resolveSidecarDirectory() throws -> URL {
        let fm = FileManager.default
        let defaults = UserDefaults.standard
        let sidecarDefaultsKey = "local_ai_sidecar_dir"

        func validatedSidecarURL(_ candidate: URL) -> URL? {
            let standardized = candidate.standardizedFileURL
            let mainPy = standardized.appendingPathComponent("local_ai_core/main.py")
            if fm.fileExists(atPath: mainPy.path) {
                defaults.set(standardized.path, forKey: sidecarDefaultsKey)
                return standardized
            }
            return nil
        }

        if let envPath = ProcessInfo.processInfo.environment["LOCAL_AI_SIDECAR_DIR"], !envPath.isEmpty {
            if let found = validatedSidecarURL(URL(fileURLWithPath: envPath)) {
                return found
            }
        }

        if let savedPath = defaults.string(forKey: sidecarDefaultsKey), !savedPath.isEmpty {
            if let found = validatedSidecarURL(URL(fileURLWithPath: savedPath)) {
                return found
            }
        }

        let cwd = URL(fileURLWithPath: fm.currentDirectoryPath)
        var candidates: [URL] = [cwd]

        // Source-path hint works reliably for local Xcode/Debug builds.
        let sourceRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        candidates.append(sourceRoot)

        // App bundle-relative candidates for packaged builds.
        var bundleCursor = Bundle.main.bundleURL
        for _ in 0 ..< 6 {
            candidates.append(bundleCursor)
            bundleCursor = bundleCursor.deletingLastPathComponent()
        }

        var cursor = cwd
        for _ in 0 ..< 6 {
            cursor = cursor.deletingLastPathComponent()
            candidates.append(cursor)
        }

        // Common local development locations.
        let home = fm.homeDirectoryForCurrentUser
        candidates.append(home.appendingPathComponent("Desktop/Development/PLOS"))
        candidates.append(home.appendingPathComponent("Development/PLOS"))
        candidates.append(home.appendingPathComponent("Documents/PLOS"))

        for root in candidates {
            if let found = validatedSidecarURL(root) {
                return found
            }
            if let found = validatedSidecarURL(root.appendingPathComponent("sidecar")) {
                return found
            }
        }

        throw APIError(message: "sidecar 디렉터리를 자동으로 찾지 못했습니다. sidecar 폴더가 프로젝트 내에 있는지 확인해 주세요.")
    }
}

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

@MainActor
final class AppViewModel: ObservableObject {
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

    @Published var indexProgress: Double = 0
    @Published var indexStageText = "준비 중"

    @Published var chatMessages: [ChatMessage] = []
    @Published var citations: [Citation] = []

    @Published var statusSnapshot: StatusSnapshot?
    @Published var failureItems: [FailureItem] = []

    @Published var lastError: String?
    @Published var isBusy = false
    @Published var needsExternalConfirmation = false

    private let sidecar = SidecarProcessManager()
    private let bookmarkStore = BookmarkStore()
    private let onboardingDefaultsKey = "local_ai_onboarding_finished"
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

        do {
            try await sidecar.start()
            if hasFinishedOnboarding {
                try await syncWorkspaceAndSettings()
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
            let response = try await client.localChat(LocalChatRequest(query: query, mode: selectedMode, conversation_id: nil, top_k: nil))
            citations = response.citations
            chatMessages.append(ChatMessage(source: .local, text: response.answer, timestamp: Date()))
            latestQueryForDeepAnalysis = query
            try await refreshRemoteState()
        } catch {
            lastError = error.localizedDescription
        }
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

    func performDeepAnalysis(userConfirmed: Bool) async {
        guard let query = latestQueryForDeepAnalysis else {
            lastError = "먼저 로컬 질문을 실행해 주세요."
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

    func triggerFullReindex() async {
        isBusy = true
        defer { isBusy = false }

        do {
            try await syncWorkspaceAndSettings()
            try await runIndexing(scope: "full")
            try await refreshRemoteState()
        } catch {
            lastError = error.localizedDescription
        }
    }

    func saveSettingsAndWorkspace() async {
        isBusy = true
        defer { isBusy = false }

        do {
            try await syncWorkspaceAndSettings()
            try await refreshRemoteState()
        } catch {
            lastError = error.localizedDescription
        }
    }

    func refreshRemoteState() async throws {
        guard let client = sidecar.apiClient else {
            return
        }

        let settings = try await client.getSettings()
        privacyMode = settings.privacy_mode
        startupProfile = settings.startup_profile

        let status = try await client.getStatus()
        statusSnapshot = status

        let failures = try await client.getFailures()
        failureItems = failures.failures
    }

    private func syncWorkspaceAndSettings() async throws {
        guard let client = sidecar.apiClient else {
            throw APIError(message: "Sidecar client unavailable")
        }

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

    private func stageLabel(_ stage: String) -> String {
        switch stage {
        case "scan":
            return "문서 분석 중"
        case "parse":
            return "텍스트 파싱 중"
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
}

// MARK: - Views

struct ContentView: View {
    @ObservedObject var viewModel: AppViewModel

    var body: some View {
        VStack(spacing: 0) {
            headerBar

            Divider()

            if viewModel.hasFinishedOnboarding {
                MainWorkspaceView(viewModel: viewModel)
            } else {
                OnboardingView(viewModel: viewModel)
            }

            if let error = viewModel.lastError {
                Divider()
                Text(error)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
            }
        }
    }

    private var headerBar: some View {
        HStack {
            Label("Local AI Core", systemImage: "brain")
                .font(.headline)

            Spacer()

            Text("Privacy: \(viewModel.currentPrivacyBadge)")
                .font(.caption)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(.thinMaterial)
                .clipShape(Capsule())

            if let snapshot = viewModel.statusSnapshot, snapshot.latest_external_call != nil {
                Image(systemName: "network")
                    .foregroundStyle(.orange)
                    .help("최근 외부 호출 있음")
            } else {
                Image(systemName: "desktopcomputer")
                    .foregroundStyle(.green)
                    .help("로컬 처리")
            }
        }
        .padding(12)
    }
}

struct OnboardingView: View {
    @ObservedObject var viewModel: AppViewModel

    var body: some View {
        GeometryReader { proxy in
            ZStack {
                LinearGradient(
                    colors: [
                        Color(red: 0.09, green: 0.22, blue: 0.16),
                        Color(red: 0.10, green: 0.37, blue: 0.24),
                        Color(red: 0.85, green: 0.95, blue: 0.90)
                    ],
                    startPoint: .topLeading,
                    endPoint: .bottomTrailing
                )
                .ignoresSafeArea()

                HStack(spacing: 22) {
                    installerSidebar
                        .frame(width: min(290, max(240, proxy.size.width * 0.3)))

                    installerContent
                }
                .padding(24)
            }
        }
    }

    private var installerSidebar: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 8) {
                Image(systemName: "checkmark.shield.fill")
                    .foregroundStyle(Color.green)
                Text("Installer")
                    .font(.headline.weight(.semibold))
            }

            Text("Local AI Core for Mac")
                .font(.title3.weight(.bold))

            ForEach(OnboardingStep.allCases, id: \.rawValue) { step in
                stepRow(step)
            }

            Spacer()

            Text("로컬 우선 · 선택형 외부 호출")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .padding(20)
        .frame(maxHeight: .infinity, alignment: .top)
        .background(.ultraThinMaterial, in: RoundedRectangle(cornerRadius: 18))
        .overlay(
            RoundedRectangle(cornerRadius: 18)
                .stroke(Color.white.opacity(0.35), lineWidth: 1)
        )
    }

    private var installerContent: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                Text("Step \(viewModel.onboardingStep.rawValue + 1) of \(OnboardingStep.allCases.count)")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)

                Spacer()

                Text(stepBadgeText)
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(Color.green.opacity(0.15))
                    .clipShape(Capsule())
            }

            Text(stepTitle)
                .font(.title.weight(.bold))

            Text(stepDescription)
                .foregroundStyle(.secondary)

            Group {
                switch viewModel.onboardingStep {
                case .welcome:
                    welcomeStep
                case .dataSelection:
                    dataSelectionStep
                case .startProfile:
                    startupProfileStep
                case .privacyInfo:
                    privacyStep
                case .indexing:
                    indexingStep
                case .ready:
                    readyStep
                }
            }
            .frame(maxHeight: .infinity, alignment: .top)
        }
        .padding(28)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .background(.regularMaterial, in: RoundedRectangle(cornerRadius: 20))
        .overlay(
            RoundedRectangle(cornerRadius: 20)
                .stroke(Color.white.opacity(0.45), lineWidth: 1)
        )
    }

    private var stepBadgeText: String {
        switch viewModel.onboardingStep {
        case .welcome:
            return "WELCOME"
        case .dataSelection:
            return "DATA"
        case .startProfile:
            return "PROFILE"
        case .privacyInfo:
            return "PRIVACY"
        case .indexing:
            return "INDEX"
        case .ready:
            return "READY"
        }
    }

    private var stepTitle: String {
        switch viewModel.onboardingStep {
        case .welcome:
            return "당신의 Mac에서 시작되는 개인 AI"
        case .dataSelection:
            return "어떤 자료를 참고할까요?"
        case .startProfile:
            return "어떤 방식으로 시작할까요?"
        case .privacyInfo:
            return "기본적으로 로컬에서 처리합니다"
        case .indexing:
            return "로컬 인덱싱 준비"
        case .ready:
            return "준비가 끝났습니다"
        }
    }

    private var stepDescription: String {
        switch viewModel.onboardingStep {
        case .welcome:
            return "선택한 자료만 정리하고, 필요한 경우에만 외부 AI를 호출합니다."
        case .dataSelection:
            return "인덱싱할 자료 범위를 명확히 선택해 데이터 통제권을 유지합니다."
        case .startProfile:
            return "속도/품질 성향을 시작 프로필로 지정합니다."
        case .privacyInfo:
            return "외부 호출 정책을 먼저 고정하면 이후 동작이 예측 가능해집니다."
        case .indexing:
            return "로컬 검색과 응답 품질을 위해 데이터 준비를 수행합니다."
        case .ready:
            return "이제 설치가 완료되었습니다. 첫 질문으로 바로 시작할 수 있습니다."
        }
    }

    private func stepRow(_ step: OnboardingStep) -> some View {
        let isCurrent = step == viewModel.onboardingStep
        let isDone = step.rawValue < viewModel.onboardingStep.rawValue

        return HStack(spacing: 10) {
            ZStack {
                Circle()
                    .fill(isDone || isCurrent ? Color.green : Color.gray.opacity(0.25))
                    .frame(width: 18, height: 18)
                if isDone {
                    Image(systemName: "checkmark")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(.white)
                } else {
                    Circle()
                        .stroke(Color.white.opacity(0.8), lineWidth: 1)
                        .frame(width: 8, height: 8)
                }
            }

            Text(stepName(step))
                .font(.subheadline.weight(isCurrent ? .semibold : .regular))
                .foregroundStyle(isCurrent ? .primary : .secondary)
        }
    }

    private func stepName(_ step: OnboardingStep) -> String {
        switch step {
        case .welcome:
            return "환영"
        case .dataSelection:
            return "자료 선택"
        case .startProfile:
            return "시작 방식"
        case .privacyInfo:
            return "프라이버시"
        case .indexing:
            return "인덱싱"
        case .ready:
            return "완료"
        }
    }

    private var welcomeStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("웹 검색 중심 도구가 아니라, 내 Mac 안의 문서/노트/프로젝트 맥락을 복원하는 작업 코어입니다.")
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(14)
                .background(Color.green.opacity(0.12), in: RoundedRectangle(cornerRadius: 12))

            Button("시작하기") {
                viewModel.onboardingStep = .dataSelection
            }
            .buttonStyle(.borderedProminent)
        }
    }

    private var dataSelectionStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("선택한 자료만 로컬 인덱싱됩니다. 나중에 언제든 변경 가능합니다.")
                .foregroundStyle(.secondary)

            HStack {
                Button("Documents") {
                    appendDefaultDirectory("Documents")
                }
                Button("Desktop") {
                    appendDefaultDirectory("Desktop")
                }
                Button("Downloads") {
                    appendDefaultDirectory("Downloads")
                }
                Button("특정 폴더 추가") {
                    viewModel.addFolder()
                }
            }
            .buttonStyle(.bordered)

            List {
                ForEach(viewModel.includedFolderURLs, id: \.path) { url in
                    HStack {
                        Text(url.path)
                            .lineLimit(1)
                        Spacer()
                        Button("제거") {
                            viewModel.removeFolder(url.path)
                        }
                    }
                }
            }
            .frame(minHeight: 220)

            HStack {
                Button("뒤로") {
                    viewModel.onboardingStep = .welcome
                }
                Spacer()
                Button("다음") {
                    viewModel.onboardingStep = .startProfile
                }
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.includedFolderURLs.isEmpty)
            }
        }
    }

    private var startupProfileStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                profileCard(.fast, subtitle: "빠른 설치, 경량 응답")
                profileCard(.recommended, subtitle: "속도·품질 균형")
                profileCard(.deep, subtitle: "느리지만 더 깊은 분석")
            }
            .frame(maxWidth: .infinity)

            Text("나중에 설정에서 변경할 수 있습니다.")
                .foregroundStyle(.secondary)

            HStack {
                Button("뒤로") {
                    viewModel.onboardingStep = .dataSelection
                }
                Spacer()
                Button("다음") {
                    viewModel.onboardingStep = .privacyInfo
                }
                .buttonStyle(.borderedProminent)
            }
        }
    }

    private var privacyStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("외부 AI 사용 여부는 설정에서 언제든 바꿀 수 있습니다.")

            Picker("프라이버시 모드", selection: $viewModel.privacyMode) {
                ForEach(PrivacyMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            HStack {
                Button("뒤로") {
                    viewModel.onboardingStep = .startProfile
                }
                Spacer()
                Button("계속") {
                    viewModel.onboardingStep = .indexing
                }
                .buttonStyle(.borderedProminent)
            }
        }
    }

    private var indexingStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(viewModel.indexStageText)
                .font(.headline)

            ProgressView(value: viewModel.indexProgress)
                .tint(.green)

            Text("단순 로딩이 아닌 준비 상태를 단계적으로 표시합니다.")
                .foregroundStyle(.secondary)

            HStack {
                Button("뒤로") {
                    viewModel.onboardingStep = .privacyInfo
                }
                Spacer()
                Button("인덱싱 시작") {
                    Task {
                        await viewModel.startOnboardingIndexingFlow()
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.isBusy)
            }
        }
    }

    private var readyStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("이제 당신의 자료를 기반으로 질문할 수 있습니다.")
                .font(.headline)

            suggestionButton("이 프로젝트의 핵심 목표가 뭐였지?")
            suggestionButton("이 폴더 문서들 핵심만 요약해줘")
            suggestionButton("지난번 메모 기준으로 다음 할 일 정리해줘")

            Picker("작업 모드", selection: $viewModel.selectedMode) {
                ForEach(WorkMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            HStack {
                Spacer()
                Button("작업 시작") {
                    viewModel.finalizeOnboarding()
                }
                .buttonStyle(.borderedProminent)
            }
        }
    }

    private func suggestionButton(_ text: String) -> some View {
        Button(text) {
            viewModel.inputQuery = text
        }
        .buttonStyle(.bordered)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func profileCard(_ profile: StartupProfile, subtitle: String) -> some View {
        let isSelected = viewModel.startupProfile == profile
        return Button {
            viewModel.startupProfile = profile
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                Text(profile.title)
                    .font(.headline)
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(isSelected ? Color.green.opacity(0.18) : Color.gray.opacity(0.10))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(isSelected ? Color.green : Color.clear, lineWidth: 1.5)
            )
        }
        .buttonStyle(.plain)
    }

    private func appendDefaultDirectory(_ name: String) {
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return
        }
        if !viewModel.includedFolderURLs.contains(where: { $0.path == url.path }) {
            viewModel.includedFolderURLs.append(url)
            viewModel.persistBookmarks()
        }
    }
}

struct MainWorkspaceView: View {
    @ObservedObject var viewModel: AppViewModel

    var body: some View {
        TabView {
            ChatPanelView(viewModel: viewModel)
                .tabItem {
                    Label("질의응답", systemImage: "message")
                }

            StatusPanelView(viewModel: viewModel)
                .tabItem {
                    Label("상태", systemImage: "gauge")
                }

            SettingsPanelView(viewModel: viewModel)
                .tabItem {
                    Label("설정", systemImage: "gearshape")
                }
        }
    }
}

struct ChatPanelView: View {
    @ObservedObject var viewModel: AppViewModel

    var body: some View {
        VStack(spacing: 12) {
            controls
            messageList
            citationList
            composer
        }
        .padding(16)
        .confirmationDialog("외부 AI 호출 전 확인", isPresented: $viewModel.needsExternalConfirmation) {
            Button("승인하고 실행") {
                Task {
                    await viewModel.performDeepAnalysis(userConfirmed: true)
                }
            }
            Button("취소", role: .cancel) {}
        } message: {
            Text("선택된 자료 일부가 외부 제공자에 전달될 수 있습니다.")
        }
    }

    private var controls: some View {
        HStack {
            Picker("작업 모드", selection: $viewModel.selectedMode) {
                ForEach(WorkMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .frame(maxWidth: 420)

            Picker("제공자", selection: $viewModel.selectedProvider) {
                Text("OpenAI").tag("openai")
                Text("Anthropic").tag("anthropic")
            }
            .frame(width: 180)

            Spacer()

            Button("더 깊게 분석") {
                viewModel.deepAnalyzeTapped()
            }
            .disabled(viewModel.chatMessages.isEmpty || viewModel.isBusy)
        }
    }

    private var messageList: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 8) {
                ForEach(viewModel.chatMessages) { message in
                    HStack(alignment: .top) {
                        Text(label(for: message.source))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .frame(width: 54, alignment: .leading)

                        Text(message.text)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(10)
                            .background(background(for: message.source))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                }
            }
        }
        .frame(maxHeight: 300)
    }

    private var citationList: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("출처")
                .font(.headline)

            if viewModel.citations.isEmpty {
                Text("출처가 아직 없습니다.")
                    .foregroundStyle(.secondary)
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 8) {
                        ForEach(viewModel.citations) { citation in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(citation.file_path)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                                Text(citation.snippet)
                                    .font(.subheadline)
                                    .lineLimit(3)
                                Text(String(format: "score %.3f", citation.score))
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(10)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.gray.opacity(0.10))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                    }
                }
                .frame(maxHeight: 180)
            }
        }
    }

    private var composer: some View {
        HStack {
            TextField("질문을 입력하세요", text: $viewModel.inputQuery, axis: .vertical)
                .lineLimit(1 ... 5)
                .textFieldStyle(.roundedBorder)

            Button("전송") {
                Task {
                    await viewModel.askLocal()
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(viewModel.inputQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || viewModel.isBusy)
        }
    }

    private func label(for source: ChatMessage.Source) -> String {
        switch source {
        case .user:
            return "USER"
        case .local:
            return "LOCAL"
        case .external:
            return "EXT"
        }
    }

    private func background(for source: ChatMessage.Source) -> Color {
        switch source {
        case .user:
            return Color.blue.opacity(0.14)
        case .local:
            return Color.green.opacity(0.12)
        case .external:
            return Color.orange.opacity(0.16)
        }
    }
}

struct StatusPanelView: View {
    @ObservedObject var viewModel: AppViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("상태 패널")
                    .font(.title2.weight(.bold))

                if let snapshot = viewModel.statusSnapshot {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("인덱싱 문서 수: \(snapshot.indexed_docs)")
                        Text("마지막 인덱싱: \(snapshot.last_indexed_at ?? "-")")
                        Text("현재 프라이버시 모드: \(snapshot.privacy_mode.title)")
                        Text("최근 외부 호출: \(snapshot.latest_external_call?.provider ?? "없음")")
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .background(Color.gray.opacity(0.12))
                    .clipShape(RoundedRectangle(cornerRadius: 8))
                } else {
                    Text("상태 정보를 불러오는 중입니다.")
                        .foregroundStyle(.secondary)
                }

                VStack(alignment: .leading, spacing: 6) {
                    Text("인덱싱 대상 폴더")
                        .font(.headline)
                    ForEach(viewModel.includedFolderURLs, id: \.path) { url in
                        Text(url.path)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("실패 파일 목록")
                        .font(.headline)
                    if viewModel.failureItems.isEmpty {
                        Text("실패한 파일이 없습니다.")
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(viewModel.failureItems) { item in
                            VStack(alignment: .leading, spacing: 3) {
                                Text(item.path)
                                    .lineLimit(1)
                                Text(item.reason)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(8)
                            .background(Color.red.opacity(0.08))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                    }
                }
            }
            .padding(16)
        }
        .task {
            do {
                try await viewModel.refreshRemoteState()
            } catch {
                viewModel.lastError = error.localizedDescription
            }
        }
    }
}

struct SettingsPanelView: View {
    @ObservedObject var viewModel: AppViewModel
    @State private var newExcludePath = ""

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("설정")
                    .font(.title2.weight(.bold))

                VStack(alignment: .leading, spacing: 8) {
                    Text("프라이버시")
                        .font(.headline)
                    Picker("프라이버시 모드", selection: $viewModel.privacyMode) {
                        ForEach(PrivacyMode.allCases) { mode in
                            Text(mode.title).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("시작 방식")
                        .font(.headline)
                    Picker("시작 방식", selection: $viewModel.startupProfile) {
                        ForEach(StartupProfile.allCases) { profile in
                            Text(profile.title).tag(profile)
                        }
                    }
                    .pickerStyle(.radioGroup)
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("기본 작업 모드")
                        .font(.headline)
                    Picker("기본 작업 모드", selection: $viewModel.defaultWorkMode) {
                        ForEach(WorkMode.allCases) { mode in
                            Text(mode.title).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("인덱싱 대상 폴더")
                        .font(.headline)

                    HStack {
                        Button("폴더 추가") {
                            viewModel.addFolder()
                        }
                        Button("전체 재인덱싱") {
                            Task {
                                await viewModel.triggerFullReindex()
                            }
                        }
                        .disabled(viewModel.isBusy)
                    }

                    ForEach(viewModel.includedFolderURLs, id: \.path) { url in
                        HStack {
                            Text(url.path)
                                .lineLimit(1)
                            Spacer()
                            Button("삭제") {
                                viewModel.removeFolder(url.path)
                            }
                        }
                    }
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("제외 폴더")
                        .font(.headline)

                    HStack {
                        TextField("제외 경로", text: $newExcludePath)
                        Button("추가") {
                            let trimmed = newExcludePath.trimmingCharacters(in: .whitespacesAndNewlines)
                            guard !trimmed.isEmpty else { return }
                            if !viewModel.excludedPaths.contains(trimmed) {
                                viewModel.excludedPaths.append(trimmed)
                            }
                            newExcludePath = ""
                        }
                    }

                    ForEach(viewModel.excludedPaths, id: \.self) { path in
                        HStack {
                            Text(path)
                                .lineLimit(1)
                            Spacer()
                            Button("제거") {
                                viewModel.excludedPaths.removeAll { $0 == path }
                            }
                        }
                    }
                }

                HStack {
                    Spacer()
                    Button("설정 저장") {
                        Task {
                            await viewModel.saveSettingsAndWorkspace()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(viewModel.isBusy)
                }
            }
            .padding(16)
        }
    }
}
