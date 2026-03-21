import Combine
import Darwin
import Foundation

struct SidecarPythonRuntimeConfig {
    let pythonExecutable: String
    let pythonPath: String?
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

    private(set) var sessionToken = UUID().uuidString
    private(set) var apiClient: SidecarAPIClient?

    private var process: Process?
    private var sidecarLogURL: URL?
    private var sidecarLogHandle: FileHandle?
    private var isStarting = false

    private let host = "127.0.0.1"
    private var preferredPort = 8777

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
        let runtimeDirectory = try SidecarBootstrapService.prepareRuntimeDirectory()
        let runtimeConfig = try SidecarBootstrapService.ensureSidecarEnvironment(
            sidecarDirectory: sidecarDirectory,
            runtimeDirectory: runtimeDirectory
        )

        let dataDirectory = runtimeDirectory.appendingPathComponent("data")
        try FileManager.default.createDirectory(at: dataDirectory, withIntermediateDirectories: true)

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
                _ = try await client.getSettings()
                return
            } catch {
                if Self.isInvalidSessionTokenError(error) {
                    throw APIError(message: "해당 포트에 다른 sidecar 세션이 이미 실행 중입니다. 포트를 자동 전환해 다시 시도해 주세요.\n\(sidecarLogSummary())")
                }
                try await Task.sleep(nanoseconds: 250_000_000)
            }
        }
        throw APIError(message: "Sidecar가 정상 상태가 되지 않았습니다. (health timeout)\n\(sidecarLogSummary())")
    }

    private func launchSidecarProcess(
        runtimeConfig: SidecarPythonRuntimeConfig,
        runtimeDirectory: URL,
        dataDirectory: URL,
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
