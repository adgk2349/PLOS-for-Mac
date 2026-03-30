import Combine
import Darwin
import Foundation

struct SidecarPythonRuntimeConfig {
    let pythonExecutable: String
    let pythonPath: String?
}

struct SidecarStorageResolution {
    let requestedModelsDirectory: URL
    let effectiveModelsDirectory: URL
    let requestedRuntimeDirectory: URL
    let effectiveRuntimeDirectory: URL
    let modelFallbackReason: String?
    let runtimeFallbackReason: String?
}

@MainActor
final class SidecarProcessManager: ObservableObject {
    private struct LaunchArtifacts {
        let process: Process
        let logURL: URL
        let logHandle: FileHandle
        let baseURL: URL
    }

    @Published private(set) var isRunning = false
    @Published private(set) var storageResolution: SidecarStorageResolution?

    private(set) var sessionToken = UUID().uuidString
    private(set) var apiClient: SidecarAPIClient?

    private var process: Process?
    private var sidecarLogURL: URL?
    private var sidecarLogHandle: FileHandle?
    private var isStarting = false
    private var preferredModelsDirectory: URL?
    private var preferredRuntimeDirectory: URL?

    private let host = "127.0.0.1"
    private var preferredPort = 8777

    func configureStorageDirectories(modelsDirectory: URL?, runtimeDirectory: URL?) {
        preferredModelsDirectory = modelsDirectory?.standardizedFileURL
        preferredRuntimeDirectory = runtimeDirectory?.standardizedFileURL
    }

    func start() async throws {
        if isRunning, let client = apiClient {
            do {
                try await client.health()
                _ = try await client.getSettings()
                return
            } catch {
                stop()
                sessionToken = UUID().uuidString
            }
        }

        if isStarting {
            for _ in 0 ..< 120 {
                if isRunning, apiClient != nil {
                    return
                }
                try await Task.sleep(nanoseconds: 100_000_000)
            }
        }

        isStarting = true
        defer { isStarting = false }

        sessionToken = UUID().uuidString

        let sidecarDirectory = try resolveSidecarDirectory()
        let requestedRuntimeDirectory = (preferredRuntimeDirectory ?? defaultRuntimeDirectory()).standardizedFileURL
        let runtimeDirectory = try SidecarBootstrapService.prepareRuntimeDirectory(
            preferredDirectory: requestedRuntimeDirectory
        )
        let runtimeFallbackReason: String? = runtimeDirectory.standardizedFileURL.path == requestedRuntimeDirectory.path
            ? nil
            : "요청 경로를 사용할 수 없어 \(runtimeDirectory.path)로 폴백"
        migrateRuntimeVenvIfNeeded(
            targetRuntimeDirectory: runtimeDirectory,
            legacyRuntimeDirectories: legacyRuntimeDirectories(excluding: runtimeDirectory)
        )
        let runtimeConfig = try SidecarBootstrapService.ensureSidecarEnvironment(
            sidecarDirectory: sidecarDirectory,
            runtimeDirectory: runtimeDirectory
        )

        let dataDirectory = try resolveDataDirectory(
            sidecarDirectory: sidecarDirectory,
            runtimeDirectory: runtimeDirectory
        )
        let requestedModelsDirectory = (preferredModelsDirectory ?? defaultModelsDirectory()).standardizedFileURL
        let (modelsDirectory, modelFallbackReason) = try resolveModelsDirectory(
            requestedDirectory: requestedModelsDirectory,
            runtimeDirectory: runtimeDirectory,
            sidecarDirectory: sidecarDirectory
        )
        migrateModelsIfNeeded(
            targetModelsDirectory: modelsDirectory,
            legacyModelRoots: legacyModelDirectories(
                sidecarDirectory: sidecarDirectory,
                runtimeDirectory: runtimeDirectory,
                dataDirectory: dataDirectory,
                excluding: modelsDirectory
            )
        )
        storageResolution = SidecarStorageResolution(
            requestedModelsDirectory: requestedModelsDirectory,
            effectiveModelsDirectory: modelsDirectory,
            requestedRuntimeDirectory: requestedRuntimeDirectory,
            effectiveRuntimeDirectory: runtimeDirectory,
            modelFallbackReason: modelFallbackReason,
            runtimeFallbackReason: runtimeFallbackReason
        )

        let openAIKey = SidecarSecretStore.read("openai_api_key")
        let anthropicKey = SidecarSecretStore.read("anthropic_api_key")

        let ports = SidecarPortService.portCandidates(preferred: preferredPort)
        SidecarPortService.terminateStaleRuntimeSidecars(runtimeDirectory: runtimeDirectory, ports: ports)

        var lastError: Error?

        for port in ports {
            if SidecarPortService.isPortListening(port) {
                continue
            }

            var launched: LaunchArtifacts?
            do {
                launched = try launchSidecarProcess(
                    runtimeConfig: runtimeConfig,
                    runtimeDirectory: runtimeDirectory,
                    dataDirectory: dataDirectory,
                    modelsDirectory: modelsDirectory,
                    host: host,
                    port: port,
                    sessionToken: sessionToken,
                    openAIKey: openAIKey,
                    anthropicKey: anthropicKey
                )

                guard let launched else {
                    throw APIError(message: "Sidecar 런치 아티팩트 생성 실패")
                }

                process = launched.process
                sidecarLogURL = launched.logURL
                sidecarLogHandle = launched.logHandle

                let client = SidecarAPIClient(
                    baseURL: launched.baseURL,
                    sessionToken: sessionToken
                )
                apiClient = client

                try await waitUntilHealthy(client: client)

                preferredPort = port
                isRunning = true
                return
            } catch {
                lastError = error
                if let launched {
                    launched.process.terminate()
                    try? launched.logHandle.close()
                }
                process = nil
                sidecarLogURL = nil
                sidecarLogHandle = nil
                apiClient = nil
            }
        }

        throw lastError ?? APIError(message: "Sidecar 시작 실패: 사용 가능한 포트를 찾지 못했습니다.")
    }

    func stop() {
        if let proc = process, proc.isRunning {
            proc.terminate()
            for _ in 0 ..< 20 where proc.isRunning {
                usleep(50_000)
            }
            if proc.isRunning {
                proc.interrupt()
                for _ in 0 ..< 10 where proc.isRunning {
                    usleep(50_000)
                }
            }
            if proc.isRunning {
                _ = kill(proc.processIdentifier, SIGKILL)
            }
        }

        process = nil
        try? sidecarLogHandle?.close()
        sidecarLogHandle = nil
        apiClient = nil
        isRunning = false
        isStarting = false
    }

    private func waitUntilHealthy(client: SidecarAPIClient) async throws {
        for _ in 0 ..< 80 {
            if let process, !process.isRunning {
                throw APIError(message: "Sidecar가 시작 직후 종료되었습니다.\n\(sidecarLogSummary())")
            }
            do {
                try await client.health()
                return
            } catch {
                if Self.isInvalidSessionTokenError(error) {
                    throw APIError(message: "해당 포트에 다른 sidecar 세션이 이미 실행 중입니다. 포트를 자동 전환해 다시 시도해 주세요.\n\(sidecarLogSummary())")
                }
                try await Task.sleep(nanoseconds: 200_000_000)
            }
        }
        throw APIError(message: "Sidecar가 정상 상태가 되지 않았습니다. (health timeout)\n\(sidecarLogSummary())")
    }

    private func launchSidecarProcess(
        runtimeConfig: SidecarPythonRuntimeConfig,
        runtimeDirectory: URL,
        dataDirectory: URL,
        modelsDirectory: URL,
        host: String,
        port: Int,
        sessionToken: String,
        openAIKey: String?,
        anthropicKey: String?
    ) throws -> LaunchArtifacts {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            runtimeConfig.pythonExecutable,
            "-m", "uvicorn",
            "local_ai_core.main:app",
            "--host", host,
            "--port", String(port)
        ]
        process.currentDirectoryURL = runtimeDirectory

        var env = ProcessInfo.processInfo.environment
        env["LOCAL_AI_SESSION_TOKEN"] = sessionToken
        env["LOCAL_AI_DATA_DIR"] = dataDirectory.path
        env["LOCAL_AI_STRICT_DATA_DIR"] = "1"
        env["LOCAL_AI_MODELS_DIR"] = modelsDirectory.path
        env["LOCAL_AI_PARENT_PID"] = String(ProcessInfo.processInfo.processIdentifier)
        env["PYTHONUNBUFFERED"] = "1"
        env["PATH"] = SidecarEnvironmentService.normalizedRuntimePath(existing: env["PATH"])
        env["LOCAL_AI_POPPLER_PATH"] = SidecarEnvironmentService.detectPopplerDirectory() ?? "/opt/homebrew/bin"
        env["LOCAL_AI_TESSERACT_CMD"] = SidecarEnvironmentService.detectTesseractExecutable() ?? "/opt/homebrew/bin/tesseract"
        if let tessdataPrefix = SidecarEnvironmentService.detectTessdataPrefix() {
            env["TESSDATA_PREFIX"] = tessdataPrefix
        }
        if let openAIKey, !openAIKey.isEmpty {
            env["OPENAI_API_KEY"] = openAIKey
        }
        if let anthropicKey, !anthropicKey.isEmpty {
            env["ANTHROPIC_API_KEY"] = anthropicKey
        }
        if let runtimePythonPath = runtimeConfig.pythonPath {
            if let existing = env["PYTHONPATH"], !existing.isEmpty {
                env["PYTHONPATH"] = "\(runtimePythonPath):\(existing)"
            } else {
                env["PYTHONPATH"] = runtimePythonPath
            }
        }
        process.environment = env

        let (logURL, logHandle) = try SidecarLaunchService.makeLogFile(prefix: "sidecar-runtime")
        process.standardOutput = logHandle
        process.standardError = logHandle

        do {
            try process.run()
        } catch {
            try? logHandle.close()
            throw APIError(message: "Sidecar 실행 실패(포트 \(port)): \(error.localizedDescription)")
        }

        return LaunchArtifacts(
            process: process,
            logURL: logURL,
            logHandle: logHandle,
            baseURL: URL(string: "http://\(host):\(port)")!
        )
    }

    private func sidecarLogSummary() -> String {
        guard let url = sidecarLogURL else {
            return "sidecar 로그 파일을 찾지 못했습니다."
        }
        guard let data = try? Data(contentsOf: url), !data.isEmpty else {
            return "sidecar 로그가 비어 있습니다."
        }
        let text = String(decoding: data, as: UTF8.self)
        return "최근 sidecar 로그:\n\(text.suffix(1400))"
    }

    private nonisolated static func isInvalidSessionTokenError(_ error: Error) -> Bool {
        guard let apiError = error as? APIError else {
            return false
        }
        let lower = apiError.message.lowercased()
        return lower.contains("http 401") && lower.contains("invalid session token")
    }

    private func resolveDataDirectory(sidecarDirectory: URL, runtimeDirectory: URL) throws -> URL {
        let fm = FileManager.default
        let preferred = sidecarDirectory.appendingPathComponent("data", isDirectory: true)
        let fallback = runtimeDirectory.appendingPathComponent("data", isDirectory: true)
        for candidate in [preferred, fallback] {
            do {
                try fm.createDirectory(at: candidate, withIntermediateDirectories: true)
                let probe = candidate.appendingPathComponent(".write-probe-\(UUID().uuidString)")
                try Data("ok".utf8).write(to: probe, options: .atomic)
                try? fm.removeItem(at: probe)
                return candidate
            } catch {
                continue
            }
        }
        throw APIError(message: "sidecar 데이터 디렉터리를 생성할 수 없습니다. (\(preferred.path), \(fallback.path))")
    }

    private func defaultModelsDirectory() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Documents/PLOS/LocalAI/models", isDirectory: true)
            .standardizedFileURL
    }

    private func defaultRuntimeDirectory() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Documents/PLOS/LocalAI/runtime", isDirectory: true)
            .standardizedFileURL
    }

    private func appSupportRuntimeDirectory() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/LocalAICore/SidecarRuntime", isDirectory: true)
            .standardizedFileURL
    }

    private func appSupportModelsDirectory() -> URL {
        appSupportRuntimeDirectory().appendingPathComponent("data/models", isDirectory: true).standardizedFileURL
    }

    private func resolveModelsDirectory(
        requestedDirectory: URL,
        runtimeDirectory: URL,
        sidecarDirectory: URL
    ) throws -> (URL, String?) {
        let fallbackCandidates: [URL] = [
            appSupportModelsDirectory(),
            runtimeDirectory.appendingPathComponent("data/models", isDirectory: true),
            sidecarDirectory.appendingPathComponent("data/models", isDirectory: true),
        ]
        if canCreateAndWriteDirectory(requestedDirectory) {
            return (requestedDirectory.standardizedFileURL, nil)
        }
        for fallback in fallbackCandidates {
            if canCreateAndWriteDirectory(fallback) {
                return (fallback.standardizedFileURL, "요청 경로 접근 불가")
            }
        }
        throw APIError(message: "모델 저장 경로를 생성할 수 없습니다. 요청 경로: \(requestedDirectory.path)")
    }

    private func canCreateAndWriteDirectory(_ directory: URL) -> Bool {
        let fm = FileManager.default
        do {
            try fm.createDirectory(at: directory, withIntermediateDirectories: true)
            let probe = directory.appendingPathComponent(".write-probe-\(UUID().uuidString)")
            try Data("ok".utf8).write(to: probe, options: .atomic)
            try? fm.removeItem(at: probe)
            return true
        } catch {
            return false
        }
    }

    private func legacyRuntimeDirectories(excluding target: URL) -> [URL] {
        let fm = FileManager.default
        let normalizedTarget = target.standardizedFileURL.path
        var candidates: [URL] = [
            appSupportRuntimeDirectory(),
            fm.temporaryDirectory
                .appendingPathComponent("LocalAICore/SidecarRuntime", isDirectory: true)
                .standardizedFileURL,
        ]
        if let preferred = preferredRuntimeDirectory {
            candidates.insert(preferred.standardizedFileURL, at: 0)
        }
        var seen = Set<String>()
        return candidates.filter { candidate in
            let path = candidate.standardizedFileURL.path
            if path == normalizedTarget {
                return false
            }
            return seen.insert(path).inserted
        }
    }

    private func legacyModelDirectories(
        sidecarDirectory: URL,
        runtimeDirectory: URL,
        dataDirectory: URL,
        excluding target: URL
    ) -> [URL] {
        var candidates: [URL] = [
            dataDirectory.appendingPathComponent("models", isDirectory: true),
            sidecarDirectory.appendingPathComponent("data/models", isDirectory: true),
            runtimeDirectory.appendingPathComponent("data/models", isDirectory: true),
            appSupportModelsDirectory(),
        ]
        if let preferred = preferredModelsDirectory {
            candidates.insert(preferred.standardizedFileURL, at: 0)
        }
        let normalizedTarget = target.standardizedFileURL.path
        var seen = Set<String>()
        return candidates.filter { candidate in
            let path = candidate.standardizedFileURL.path
            if path == normalizedTarget {
                return false
            }
            return seen.insert(path).inserted
        }
    }

    private func migrateRuntimeVenvIfNeeded(targetRuntimeDirectory: URL, legacyRuntimeDirectories: [URL]) {
        let fm = FileManager.default
        let targetVenv = targetRuntimeDirectory.appendingPathComponent(".venv", isDirectory: true)
        guard !fm.fileExists(atPath: targetVenv.path) else {
            return
        }
        for sourceRuntime in legacyRuntimeDirectories {
            let sourceVenv = sourceRuntime.appendingPathComponent(".venv", isDirectory: true)
            guard fm.fileExists(atPath: sourceVenv.path) else { continue }
            do {
                try moveOrCopyDirectory(source: sourceVenv, destination: targetVenv)
                break
            } catch {
                continue
            }
        }
    }

    private func migrateModelsIfNeeded(targetModelsDirectory: URL, legacyModelRoots: [URL]) {
        let fm = FileManager.default
        guard canCreateAndWriteDirectory(targetModelsDirectory) else {
            return
        }
        for sourceRoot in legacyModelRoots {
            guard fm.fileExists(atPath: sourceRoot.path) else { continue }
            for engineFolder in ["mlx", "llama_cpp"] {
                let sourceEngineDir = sourceRoot.appendingPathComponent(engineFolder, isDirectory: true)
                guard fm.fileExists(atPath: sourceEngineDir.path) else { continue }
                let targetEngineDir = targetModelsDirectory.appendingPathComponent(engineFolder, isDirectory: true)
                if directoryHasItems(targetEngineDir) {
                    continue
                }
                do {
                    try moveOrCopyDirectory(source: sourceEngineDir, destination: targetEngineDir)
                } catch {
                    continue
                }
            }
        }
    }

    private func directoryHasItems(_ directory: URL) -> Bool {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(atPath: directory.path) else {
            return false
        }
        return !items.isEmpty
    }

    private func moveOrCopyDirectory(source: URL, destination: URL) throws {
        let fm = FileManager.default
        if fm.fileExists(atPath: destination.path) {
            return
        }
        do {
            try fm.moveItem(at: source, to: destination)
            return
        } catch {
            try SidecarBootstrapService.copyDirectory(source: source, destination: destination)
            guard fm.fileExists(atPath: destination.path) else {
                throw APIError(message: "디렉터리 복사 실패: \(source.path) -> \(destination.path)")
            }
            try? fm.removeItem(at: source)
        }
    }

    private func resolveSidecarDirectory() throws -> URL {
        let fm = FileManager.default
        let defaults = UserDefaults.standard
        let sidecarDefaultsKey = "local_ai_sidecar_dir"

        func sidecarCandidates(from base: URL) -> [URL] {
            let root = base.standardizedFileURL
            return [
                root,
                root.appendingPathComponent("sidecar", isDirectory: true),
                root.appendingPathComponent("staged-sidecar", isDirectory: true),
                root.appendingPathComponent("Resources/sidecar", isDirectory: true),
                root.appendingPathComponent("Resources/staged-sidecar", isDirectory: true),
                root.appendingPathComponent("Contents/Resources/sidecar", isDirectory: true),
                root.appendingPathComponent("Contents/Resources/staged-sidecar", isDirectory: true),
            ]
        }

        func validatedSidecarURL(_ candidate: URL) -> URL? {
            for dir in sidecarCandidates(from: candidate) {
                let mainPy = dir.appendingPathComponent("local_ai_core/main.py")
                let pyproject = dir.appendingPathComponent("pyproject.toml")
                if fm.fileExists(atPath: mainPy.path), fm.fileExists(atPath: pyproject.path) {
                    defaults.set(dir.path, forKey: sidecarDefaultsKey)
                    return dir
                }
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

        let sourceRoot = URL(fileURLWithPath: #filePath)
            .deletingLastPathComponent()
            .deletingLastPathComponent()
        candidates.append(sourceRoot)

        if let resourceURL = Bundle.main.resourceURL {
            candidates.append(resourceURL)
        }
        if let executableURL = Bundle.main.executableURL {
            candidates.append(executableURL.deletingLastPathComponent())
        }

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

        let home = fm.homeDirectoryForCurrentUser
        candidates.append(home.appendingPathComponent("Desktop/Development/PLOS"))
        candidates.append(home.appendingPathComponent("Desktop/Development/PLOS-for-Mac-push"))
        candidates.append(home.appendingPathComponent("Development/PLOS"))
        candidates.append(home.appendingPathComponent("Development/PLOS-for-Mac-push"))
        candidates.append(home.appendingPathComponent("Documents/PLOS"))

        let devRoots = [
            home.appendingPathComponent("Desktop/Development", isDirectory: true),
            home.appendingPathComponent("Development", isDirectory: true),
        ]
        for devRoot in devRoots where fm.fileExists(atPath: devRoot.path) {
            if let children = try? fm.contentsOfDirectory(
                at: devRoot,
                includingPropertiesForKeys: [.isDirectoryKey],
                options: [.skipsHiddenFiles]
            ) {
                for child in children where child.lastPathComponent.lowercased().contains("plos") {
                    candidates.append(child)
                }
            }
        }

        var seen = Set<String>()
        for root in candidates {
            let standardizedPath = root.standardizedFileURL.path
            if !seen.insert(standardizedPath).inserted {
                continue
            }
            if let found = validatedSidecarURL(root) {
                return found
            }
        }

        throw APIError(
            message: "sidecar 디렉터리를 자동으로 찾지 못했습니다. `LOCAL_AI_SIDECAR_DIR`를 sidecar 루트로 지정하거나 프로젝트의 `sidecar/local_ai_core/main.py` 존재 여부를 확인해 주세요."
        )
    }
}
