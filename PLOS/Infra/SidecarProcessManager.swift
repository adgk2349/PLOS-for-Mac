import Combine
import CryptoKit
import Darwin
import Foundation
import Security

enum AppSecretStore {
    private static let service = "com.redbridge.plos"

    static func read(_ key: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
            kSecReturnData as String: true,
            kSecMatchLimit as String: kSecMatchLimitOne,
        ]
        var item: CFTypeRef?
        let status = SecItemCopyMatching(query as CFDictionary, &item)
        guard status == errSecSuccess, let data = item as? Data else {
            return nil
        }
        return String(data: data, encoding: .utf8)
    }

    @discardableResult
    static func save(_ value: String, for key: String) -> Bool {
        let data = Data(value.utf8)
        let baseQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
        ]
        let updateStatus = SecItemUpdate(baseQuery as CFDictionary, [kSecValueData as String: data] as CFDictionary)
        if updateStatus == errSecSuccess {
            return true
        }
        let addQuery: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
            kSecValueData as String: data,
        ]
        let addStatus = SecItemAdd(addQuery as CFDictionary, nil)
        return addStatus == errSecSuccess
    }

    static func delete(_ key: String) {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword,
            kSecAttrService as String: service,
            kSecAttrAccount as String: key,
        ]
        SecItemDelete(query as CFDictionary)
    }
}

@MainActor
final class SidecarProcessManager: ObservableObject {
    private struct PythonRuntimeConfig {
        let pythonExecutable: String
        let pythonPath: String?
    }

    private struct LaunchArtifacts {
        let process: Process
        let logURL: URL
        let logHandle: FileHandle
        let baseURL: URL
    }

    @Published private(set) var isRunning = false

    private(set) var sessionToken = UUID().uuidString
    private var process: Process?
    private(set) var apiClient: SidecarAPIClient?
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

        // Always mint a fresh token when creating a new sidecar session.
        sessionToken = UUID().uuidString

        let sidecarDirectory = try resolveSidecarDirectory()
        let runtimeDirectory = try Self.prepareRuntimeDirectory()
        let runtimeConfig = try await Task.detached(priority: .userInitiated) {
            try Self.ensureSidecarEnvironment(sidecarDirectory: sidecarDirectory, runtimeDirectory: runtimeDirectory)
        }.value
        let dataDirectory = runtimeDirectory.appendingPathComponent("data")
        try FileManager.default.createDirectory(at: dataDirectory, withIntermediateDirectories: true)
        let openAIKey = AppSecretStore.read("openai_api_key")
        let anthropicKey = AppSecretStore.read("anthropic_api_key")
        let ports = Self.portCandidates(preferred: preferredPort)
        var lastError: Error?

        for port in ports {
            if Self.isPortListening(port) {
                continue
            }
            var launched: LaunchArtifacts?
            do {
                launched = try Self.launchSidecarProcess(
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

                self.process = launched.process
                self.sidecarLogURL = launched.logURL
                self.sidecarLogHandle = launched.logHandle

                let client = SidecarAPIClient(
                    baseURL: launched.baseURL,
                    sessionToken: sessionToken
                )
                self.apiClient = client

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
                self.process = nil
                self.sidecarLogURL = nil
                self.sidecarLogHandle = nil
                self.apiClient = nil
            }
        }

        throw lastError ?? APIError(message: "Sidecar 시작 실패: 사용 가능한 포트를 찾지 못했습니다.")
    }

    func stop() {
        process?.terminate()
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

    private nonisolated static func portCandidates(preferred: Int) -> [Int] {
        var ports = [preferred]
        let fallback = [8777, 8787, 8797, 8807, 8817, 8827]
        for port in fallback where !ports.contains(port) {
            ports.append(port)
        }
        for _ in 0 ..< 12 {
            if let dynamic = findAvailablePort(), !ports.contains(dynamic) {
                ports.append(dynamic)
            }
        }
        for port in stride(from: 18080, through: 18140, by: 2) where !ports.contains(port) {
            ports.append(port)
        }
        return ports
    }

    private nonisolated static func findAvailablePort() -> Int? {
        let socketFD = socket(AF_INET, SOCK_STREAM, 0)
        guard socketFD >= 0 else {
            return nil
        }
        defer { _ = close(socketFD) }

        var addr = sockaddr_in()
        addr.sin_len = UInt8(MemoryLayout<sockaddr_in>.size)
        addr.sin_family = sa_family_t(AF_INET)
        addr.sin_port = in_port_t(0).bigEndian
        addr.sin_addr = in_addr(s_addr: inet_addr("127.0.0.1"))

        let bindResult = withUnsafePointer(to: &addr) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                bind(socketFD, $0, socklen_t(MemoryLayout<sockaddr_in>.size))
            }
        }
        guard bindResult == 0 else {
            return nil
        }

        var assigned = sockaddr_in()
        var len = socklen_t(MemoryLayout<sockaddr_in>.size)
        let nameResult = withUnsafeMutablePointer(to: &assigned) {
            $0.withMemoryRebound(to: sockaddr.self, capacity: 1) {
                getsockname(socketFD, $0, &len)
            }
        }
        guard nameResult == 0 else {
            return nil
        }
        return Int(UInt16(bigEndian: assigned.sin_port))
    }

    private nonisolated static func isPortListening(_ port: Int) -> Bool {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = ["lsof", "-nP", "-iTCP:\(port)", "-sTCP:LISTEN", "-t"]
        let output = Pipe()
        process.standardOutput = output
        process.standardError = Pipe()
        do {
            try process.run()
            process.waitUntilExit()
            if process.terminationStatus != 0 {
                return false
            }
            let raw = String(data: output.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8) ?? ""
            return !raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        } catch {
            return false
        }
    }

    private nonisolated static func isInvalidSessionTokenError(_ error: Error) -> Bool {
        guard let apiError = error as? APIError else {
            return false
        }
        let lower = apiError.message.lowercased()
        return lower.contains("http 401") && lower.contains("invalid session token")
    }

    private nonisolated static func launchSidecarProcess(
        runtimeConfig: PythonRuntimeConfig,
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
        env["PYTHONUNBUFFERED"] = "1"
        env["PATH"] = Self.normalizedRuntimePath(existing: env["PATH"])
        env["LOCAL_AI_POPPLER_PATH"] = Self.detectPopplerDirectory() ?? "/opt/homebrew/bin"
        env["LOCAL_AI_TESSERACT_CMD"] = Self.detectTesseractExecutable() ?? "/opt/homebrew/bin/tesseract"
        if let tessdataPrefix = Self.detectTessdataPrefix() {
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

        let (logURL, logHandle) = try makeLogFile(prefix: "sidecar-runtime")
        process.standardOutput = logHandle
        process.standardError = logHandle

        do {
            try process.run()
        } catch {
            try? logHandle.close()
            throw APIError(message: "Sidecar 실행 실패(포트 \(port)): \(error.localizedDescription)")
        }

        let baseURL = URL(string: "http://\(host):\(port)")!
        return LaunchArtifacts(
            process: process,
            logURL: logURL,
            logHandle: logHandle,
            baseURL: baseURL
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

    private nonisolated static func ensureSidecarEnvironment(sidecarDirectory: URL, runtimeDirectory: URL) throws -> PythonRuntimeConfig {
        let fm = FileManager.default
        let runtimeVenvDirectory = runtimeDirectory.appendingPathComponent(".venv")
        let runtimeVenvPython = runtimeVenvDirectory.appendingPathComponent("bin/python3")
        let runtimeSitePackages = runtimeDirectory.appendingPathComponent("site-packages")
        let stagedSourceDirectory = runtimeDirectory.appendingPathComponent("staged-sidecar-\(UUID().uuidString)", isDirectory: true)
        let probeDataDirectory = runtimeDirectory.appendingPathComponent("probe-data", isDirectory: true)
        let runtimePackagePythonPath = runtimeDirectory.path
        let runtimeTargetPythonPath = "\(runtimeSitePackages.path):\(runtimePackagePythonPath)"
        let lockFD = try acquireRuntimeBootstrapLock(runtimeDirectory: runtimeDirectory)

        defer { releaseRuntimeBootstrapLock(lockFD) }

        try stageSidecarSource(from: sidecarDirectory, to: stagedSourceDirectory)
        defer { try? fm.removeItem(at: stagedSourceDirectory) }
        try syncRuntimePackage(stagedSourceDirectory: stagedSourceDirectory, runtimeDirectory: runtimeDirectory)
        try fm.createDirectory(at: probeDataDirectory, withIntermediateDirectories: true)

        let pythonCandidates = resolveSystemPythonExecutables()
        guard !pythonCandidates.isEmpty else {
            throw APIError(message: "호환 가능한 python3 실행 파일(3.11~3.13)을 찾지 못했습니다. Homebrew로 python@3.11 또는 python@3.12 설치 후 다시 시도해 주세요.")
        }

        var bootstrapErrors: [String] = []
        if hasRequiredModules(
            python: runtimeVenvPython.path,
            pythonPath: runtimePackagePythonPath,
            probeDataDir: probeDataDirectory.path
        ) {
            return PythonRuntimeConfig(pythonExecutable: runtimeVenvPython.path, pythonPath: runtimePackagePythonPath)
        }

        for systemPython in pythonCandidates {
            let candidateVenvDirectory = runtimeDirectory.appendingPathComponent(".venv-build-\(UUID().uuidString)")
            let candidateVenvPython = candidateVenvDirectory.appendingPathComponent("bin/python3")
            defer {
                if fm.fileExists(atPath: candidateVenvDirectory.path) {
                    try? fm.removeItem(at: candidateVenvDirectory)
                }
            }

            do {
                if fm.fileExists(atPath: candidateVenvDirectory.path) {
                    try? fm.removeItem(at: candidateVenvDirectory)
                }

                try runCommand(
                    executable: systemPython,
                    arguments: ["-m", "venv", candidateVenvDirectory.path],
                    cwd: nil,
                    step: "Python 가상환경 생성"
                )
                try runCommand(
                    executable: candidateVenvPython.path,
                    arguments: ["-m", "ensurepip", "--upgrade"],
                    cwd: nil,
                    step: "pip 초기화"
                )
                try installSidecarDependencies(
                    withPython: candidateVenvPython.path,
                    targetPath: nil,
                    runtimeDirectory: runtimeDirectory
                )

                var candidateReady = hasRequiredModules(
                    python: candidateVenvPython.path,
                    pythonPath: runtimePackagePythonPath,
                    probeDataDir: probeDataDirectory.path
                )
                if !candidateReady {
                    try installSidecarDependencies(
                        withPython: candidateVenvPython.path,
                        targetPath: nil,
                        runtimeDirectory: runtimeDirectory,
                        force: true
                    )
                    candidateReady = hasRequiredModules(
                        python: candidateVenvPython.path,
                        pythonPath: runtimePackagePythonPath,
                        probeDataDir: probeDataDirectory.path
                    )
                }

                if candidateReady {
                    try activateRuntimeVenv(candidateVenvDirectory: candidateVenvDirectory, runtimeVenvDirectory: runtimeVenvDirectory)
                    guard hasRequiredModules(
                        python: runtimeVenvPython.path,
                        pythonPath: runtimePackagePythonPath,
                        probeDataDir: probeDataDirectory.path
                    ) else {
                        bootstrapErrors.append("\(systemPython): 활성화 후 모듈 점검 실패")
                        continue
                    }
                    return PythonRuntimeConfig(pythonExecutable: runtimeVenvPython.path, pythonPath: runtimePackagePythonPath)
                }
                bootstrapErrors.append("\(systemPython): 설치 후 모듈 점검 실패")
            } catch {
                bootstrapErrors.append("\(systemPython) [venv]: \(error.localizedDescription)")
            }

            do {
                if fm.fileExists(atPath: runtimeSitePackages.path) {
                    try? fm.removeItem(at: runtimeSitePackages)
                }
                try fm.createDirectory(at: runtimeSitePackages, withIntermediateDirectories: true)

                try installSidecarDependencies(
                    withPython: systemPython,
                    targetPath: runtimeSitePackages.path,
                    runtimeDirectory: runtimeDirectory
                )

                var targetReady = hasRequiredModules(
                    python: systemPython,
                    pythonPath: runtimeTargetPythonPath,
                    probeDataDir: probeDataDirectory.path
                )
                if !targetReady {
                    try installSidecarDependencies(
                        withPython: systemPython,
                        targetPath: runtimeSitePackages.path,
                        runtimeDirectory: runtimeDirectory,
                        force: true
                    )
                    targetReady = hasRequiredModules(
                        python: systemPython,
                        pythonPath: runtimeTargetPythonPath,
                        probeDataDir: probeDataDirectory.path
                    )
                }

                if targetReady {
                    return PythonRuntimeConfig(
                        pythonExecutable: systemPython,
                        pythonPath: runtimeTargetPythonPath
                    )
                }
                bootstrapErrors.append("\(systemPython) [target]: 설치 후 모듈 점검 실패")
            } catch {
                bootstrapErrors.append("\(systemPython) [target]: \(error.localizedDescription)")
            }
        }

        throw APIError(
            message: "sidecar 의존성 설치 실패: Python 환경 구성이 완료되지 않았습니다.\n\(bootstrapErrors.joined(separator: "\n"))"
        )
    }

    private nonisolated static func stageSidecarSource(from sidecarDirectory: URL, to stagedSourceDirectory: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: stagedSourceDirectory, withIntermediateDirectories: true)

        let packageSource = sidecarDirectory.appendingPathComponent("local_ai_core", isDirectory: true)
        let pyprojectSource = sidecarDirectory.appendingPathComponent("pyproject.toml")

        let packageTarget = stagedSourceDirectory.appendingPathComponent("local_ai_core", isDirectory: true)
        let pyprojectTarget = stagedSourceDirectory.appendingPathComponent("pyproject.toml")

        guard fm.fileExists(atPath: packageSource.path), fm.fileExists(atPath: pyprojectSource.path) else {
            throw APIError(message: "sidecar 소스 파일을 찾지 못했습니다. local_ai_core 또는 pyproject.toml이 없습니다.")
        }

        try copyDirectory(source: packageSource, destination: packageTarget)
        let pyprojectData = try Data(contentsOf: pyprojectSource)
        try pyprojectData.write(to: pyprojectTarget, options: .atomic)
    }

    private nonisolated static func syncRuntimePackage(stagedSourceDirectory: URL, runtimeDirectory: URL) throws {
        let fm = FileManager.default
        let stagedPackage = stagedSourceDirectory.appendingPathComponent("local_ai_core", isDirectory: true)
        let runtimePackage = runtimeDirectory.appendingPathComponent("local_ai_core", isDirectory: true)
        if fm.fileExists(atPath: runtimePackage.path) {
            try? fm.removeItem(at: runtimePackage)
        }
        try copyDirectory(source: stagedPackage, destination: runtimePackage)
    }

    private nonisolated static func installSidecarDependencies(
        withPython pythonExecutable: String,
        targetPath: String?,
        runtimeDirectory: URL,
        force: Bool = false
    ) throws {
        let stampURL = dependencyStampURL(runtimeDirectory: runtimeDirectory, pythonExecutable: pythonExecutable, targetPath: targetPath)
        let fingerprint = dependencyFingerprint(pythonExecutable: pythonExecutable, targetPath: targetPath)
        if !force, let existing = try? String(contentsOf: stampURL, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines), existing == fingerprint {
            return
        }

        var arguments = ["-m", "pip", "install", "--upgrade"]
        if let targetPath, !targetPath.isEmpty {
            arguments.append(contentsOf: ["--target", targetPath])
        }
        arguments.append(contentsOf: sidecarPipDependencies())
        try runCommand(
            executable: pythonExecutable,
            arguments: arguments,
            cwd: nil,
            step: "sidecar 의존성 설치"
        )
        try fingerprint.write(to: stampURL, atomically: true, encoding: .utf8)
    }

    private nonisolated static func dependencyStampURL(
        runtimeDirectory: URL,
        pythonExecutable: String,
        targetPath: String?
    ) -> URL {
        let stampDirectory = runtimeDirectory.appendingPathComponent(".deps-stamps", isDirectory: true)
        try? FileManager.default.createDirectory(at: stampDirectory, withIntermediateDirectories: true)
        let scope = (targetPath == nil || targetPath?.isEmpty == true) ? "venv" : "target"
        let key = "\(scope)|\(pythonVersionLabel(pythonExecutable))|\(targetPath ?? "")"
        let digest = SHA256.hash(data: Data(key.utf8)).compactMap { String(format: "%02x", $0) }.joined()
        return stampDirectory.appendingPathComponent("\(digest).stamp")
    }

    private nonisolated static func dependencyFingerprint(pythonExecutable: String, targetPath: String?) -> String {
        let lines = [
            "python=\(pythonVersionLabel(pythonExecutable))",
            "target=\(targetPath ?? "")",
            "deps=\(sidecarPipDependencies().joined(separator: "|"))",
        ]
        return lines.joined(separator: "\n")
    }

    private nonisolated static func pythonVersionLabel(_ executable: String) -> String {
        guard let version = pythonVersionTuple(executable) else {
            return executable
        }
        return "\(version.0).\(version.1)"
    }

    private nonisolated static func sidecarPipDependencies() -> [String] {
        [
            "fastapi>=0.116.0",
            "uvicorn[standard]>=0.35.0",
            "pydantic>=2.8.0",
            "pydantic-settings>=2.3.0",
            "httpx>=0.28.0",
            "lancedb>=0.22.0",
            "numpy>=2.2.0",
            "pypdf>=5.3.0",
            "pytesseract>=0.3.13",
            "Pillow>=11.1.0",
            "pdf2image>=1.17.0",
            "pypdfium2>=4.30.0",
            "rapidocr_onnxruntime>=1.4.4",
            "cryptography>=3.1",
            "huggingface-hub>=0.30.0"
        ]
    }

    private nonisolated static func copyDirectory(source: URL, destination: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: destination, withIntermediateDirectories: true)

        let entries = try fm.contentsOfDirectory(
            at: source,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )

        for entry in entries {
            let name = entry.lastPathComponent
            if name == "__pycache__" || name.hasSuffix(".pyc") {
                continue
            }

            let target = destination.appendingPathComponent(name, isDirectory: false)
            let values = try entry.resourceValues(forKeys: [.isDirectoryKey])
            if values.isDirectory == true {
                try copyDirectory(source: entry, destination: target)
            } else {
                let data = try Data(contentsOf: entry)
                try data.write(to: target, options: .atomic)
            }
        }
    }

    private nonisolated static func acquireRuntimeBootstrapLock(runtimeDirectory: URL) throws -> Int32 {
        let fm = FileManager.default
        let lockURL = runtimeDirectory.appendingPathComponent(".bootstrap.lock")
        if !fm.fileExists(atPath: lockURL.path) {
            fm.createFile(atPath: lockURL.path, contents: nil)
        }

        let fd = lockURL.path.withCString { path in
            open(path, O_CREAT | O_RDWR, mode_t(S_IRUSR | S_IWUSR))
        }
        guard fd >= 0 else {
            let err = String(cString: strerror(errno))
            throw APIError(message: "sidecar bootstrap lock 파일 오픈 실패: \(err)")
        }
        guard flock(fd, LOCK_EX) == 0 else {
            let err = String(cString: strerror(errno))
            close(fd)
            throw APIError(message: "sidecar bootstrap lock 획득 실패: \(err)")
        }
        return fd
    }

    private nonisolated static func releaseRuntimeBootstrapLock(_ fd: Int32) {
        guard fd >= 0 else { return }
        _ = flock(fd, LOCK_UN)
        _ = close(fd)
    }

    private nonisolated static func activateRuntimeVenv(candidateVenvDirectory: URL, runtimeVenvDirectory: URL) throws {
        let fm = FileManager.default
        let backupVenvDirectory = runtimeVenvDirectory.deletingLastPathComponent()
            .appendingPathComponent(".venv-backup-\(UUID().uuidString)")

        var movedOld = false
        if fm.fileExists(atPath: runtimeVenvDirectory.path) {
            try fm.moveItem(at: runtimeVenvDirectory, to: backupVenvDirectory)
            movedOld = true
        }

        do {
            try fm.moveItem(at: candidateVenvDirectory, to: runtimeVenvDirectory)
            if movedOld, fm.fileExists(atPath: backupVenvDirectory.path) {
                try? fm.removeItem(at: backupVenvDirectory)
            }
        } catch {
            if fm.fileExists(atPath: runtimeVenvDirectory.path) {
                try? fm.removeItem(at: runtimeVenvDirectory)
            }
            if movedOld, fm.fileExists(atPath: backupVenvDirectory.path) {
                try? fm.moveItem(at: backupVenvDirectory, to: runtimeVenvDirectory)
            }
            throw error
        }
    }

    private nonisolated static func prepareRuntimeDirectory() throws -> URL {
        let fm = FileManager.default
        var candidates: [URL] = []

        if let appSupportBase = fm.urls(for: .applicationSupportDirectory, in: .userDomainMask).first {
            candidates.append(
                appSupportBase
                    .appendingPathComponent("LocalAICore", isDirectory: true)
                    .appendingPathComponent("SidecarRuntime", isDirectory: true)
            )
        }

        candidates.append(
            fm.homeDirectoryForCurrentUser
                .appendingPathComponent("Library/Application Support", isDirectory: true)
                .appendingPathComponent("LocalAICore", isDirectory: true)
                .appendingPathComponent("SidecarRuntime", isDirectory: true)
        )

        candidates.append(
            fm.temporaryDirectory
                .appendingPathComponent("LocalAICore", isDirectory: true)
                .appendingPathComponent("SidecarRuntime", isDirectory: true)
        )

        var errors: [String] = []
        for dir in candidates {
            do {
                try fm.createDirectory(at: dir, withIntermediateDirectories: true)
                let probe = dir.appendingPathComponent(".write-test-\(UUID().uuidString)")
                try Data("ok".utf8).write(to: probe, options: .atomic)
                try? fm.removeItem(at: probe)
                return dir
            } catch {
                errors.append("\(dir.path): \(error.localizedDescription)")
            }
        }

        throw APIError(message: "sidecar runtime 디렉터리를 생성하지 못했습니다.\n\(errors.joined(separator: "\n"))")
    }

    private nonisolated static func hasRequiredModules(
        python: String,
        pythonPath: String?,
        probeDataDir: String
    ) -> Bool {
        do {
            var overrides: [String: String] = [
                "LOCAL_AI_DATA_DIR": probeDataDir,
                "LOCAL_AI_SESSION_TOKEN": "probe-session-token",
            ]
            if let pythonPath, !pythonPath.isEmpty {
                overrides["PYTHONPATH"] = pythonPath
            }
            try runCommand(
                executable: python,
                arguments: [
                    "-c",
                    "import importlib.util as u, inspect; import uvicorn, fastapi, httpx, pydantic, lancedb, local_ai_core.main as m; assert u.find_spec('pypdf') or u.find_spec('PyPDF2') or u.find_spec('pypdfium2'); assert u.find_spec('cryptography'); assert u.find_spec('huggingface_hub'); assert u.find_spec('rapidocr_onnxruntime'); src=inspect.getsource(m.create_app); assert '/v1/docs' in src and '/v1/models/download' in src and '/v1/models/catalog' in src"
                ],
                cwd: nil,
                step: "sidecar 모듈 점검",
                envOverrides: overrides
            )
            return true
        } catch {
            return false
        }
    }

    private nonisolated static func resolveSystemPythonExecutables() -> [String] {
        let fm = FileManager.default
        let candidates = [
            "/opt/homebrew/bin/python3.11",
            "/opt/homebrew/bin/python3.12",
            "/opt/homebrew/bin/python3.13",
            "/usr/local/bin/python3.11",
            "/usr/local/bin/python3.12",
            "/usr/local/bin/python3.13",
            "/opt/homebrew/bin/python3",
            "/usr/local/bin/python3",
            "/usr/bin/python3"
        ]

        var valid: [String] = []
        for candidate in candidates where fm.isExecutableFile(atPath: candidate) {
            if isRunnablePythonExecutable(candidate) {
                guard let (_, minor) = pythonVersionTuple(candidate), (11 ... 13).contains(minor) else {
                    continue
                }
                valid.append(candidate)
            }
        }
        return valid
    }

    private nonisolated static func isRunnablePythonExecutable(_ path: String) -> Bool {
        let fm = FileManager.default
        guard fm.fileExists(atPath: path), fm.isExecutableFile(atPath: path) else {
            return false
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = ["--version"]
        process.standardOutput = Pipe()
        process.standardError = Pipe()

        do {
            try process.run()
        } catch {
            return false
        }
        process.waitUntilExit()
        return process.terminationStatus == 0
    }

    private nonisolated static func pythonVersionTuple(_ path: String) -> (Int, Int)? {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: path)
        process.arguments = [
            "-c",
            "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")"
        ]
        let output = Pipe()
        process.standardOutput = output
        process.standardError = Pipe()

        do {
            try process.run()
        } catch {
            return nil
        }
        process.waitUntilExit()
        guard process.terminationStatus == 0 else { return nil }
        guard let raw = String(data: output.fileHandleForReading.readDataToEndOfFile(), encoding: .utf8)?
            .trimmingCharacters(in: .whitespacesAndNewlines)
        else {
            return nil
        }
        let parts = raw.split(separator: ".")
        guard parts.count == 2, let major = Int(parts[0]), let minor = Int(parts[1]) else {
            return nil
        }
        return (major, minor)
    }

    private nonisolated static func runCommand(
        executable: String,
        arguments: [String],
        cwd: URL?,
        step: String,
        envOverrides: [String: String]? = nil
    ) throws {
        let fm = FileManager.default
        let stdoutURL = fm.temporaryDirectory.appendingPathComponent("local-ai-core-\(UUID().uuidString).stdout.log")
        let stderrURL = fm.temporaryDirectory.appendingPathComponent("local-ai-core-\(UUID().uuidString).stderr.log")
        fm.createFile(atPath: stdoutURL.path, contents: nil)
        fm.createFile(atPath: stderrURL.path, contents: nil)

        let outHandle = try FileHandle(forWritingTo: stdoutURL)
        let errHandle = try FileHandle(forWritingTo: stderrURL)

        defer {
            try? outHandle.close()
            try? errHandle.close()
            try? fm.removeItem(at: stdoutURL)
            try? fm.removeItem(at: stderrURL)
        }

        let process = Process()
        process.executableURL = URL(fileURLWithPath: executable)
        process.arguments = arguments
        process.currentDirectoryURL = cwd
        var env = ProcessInfo.processInfo.environment
        env.removeValue(forKey: "PYTHONHOME")
        env.removeValue(forKey: "VIRTUAL_ENV")
        env["PATH"] = normalizedRuntimePath(existing: env["PATH"])
        if let envOverrides {
            for (k, v) in envOverrides {
                env[k] = v
            }
        }
        process.environment = env

        process.standardOutput = outHandle
        process.standardError = errHandle

        do {
            try process.run()
        } catch {
            throw APIError(
                message: "\(step) 실행 실패: \(error.localizedDescription)\nexec=\(executable)\nargs=\(arguments.joined(separator: " "))\ncwd=\(cwd?.path ?? "(nil)")"
            )
        }

        process.waitUntilExit()

        try? outHandle.synchronize()
        try? errHandle.synchronize()
        let stdout = (try? String(contentsOf: stdoutURL, encoding: .utf8)) ?? ""
        let stderr = (try? String(contentsOf: stderrURL, encoding: .utf8)) ?? ""

        guard process.terminationStatus == 0 else {
            let log = (stderr.isEmpty ? stdout : stderr).trimmingCharacters(in: .whitespacesAndNewlines)
            throw APIError(message: "\(step) 실패 (exit \(process.terminationStatus))\n\(log)")
        }
    }

    private nonisolated static func normalizedRuntimePath(existing: String?) -> String {
        var segments = (existing ?? "")
            .split(separator: ":")
            .map(String.init)
            .filter { segment in
                !segment.contains("/LocalAICore/SidecarRuntime/.venv/bin")
            }

        for required in ["/opt/homebrew/bin", "/opt/homebrew/sbin", "/usr/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"] {
            if !segments.contains(required) {
                segments.append(required)
            }
        }
        return segments.joined(separator: ":")
    }

    private nonisolated static func detectPopplerDirectory() -> String? {
        let fm = FileManager.default
        let candidates = ["/opt/homebrew/bin", "/usr/local/bin"]
        for dir in candidates {
            let pdftoppm = "\(dir)/pdftoppm"
            let pdfinfo = "\(dir)/pdfinfo"
            if fm.isExecutableFile(atPath: pdftoppm), fm.isExecutableFile(atPath: pdfinfo) {
                return dir
            }
            if fm.fileExists(atPath: pdftoppm), fm.fileExists(atPath: pdfinfo) {
                return dir
            }
        }
        for dir in candidates where fm.fileExists(atPath: dir) {
            return dir
        }
        return nil
    }

    private nonisolated static func detectTesseractExecutable() -> String? {
        let fm = FileManager.default
        let candidates = ["/opt/homebrew/bin/tesseract", "/usr/local/bin/tesseract"]
        for path in candidates where fm.isExecutableFile(atPath: path) {
            return path
        }
        for path in candidates where fm.fileExists(atPath: path) {
            return path
        }
        return nil
    }

    private nonisolated static func detectTessdataPrefix() -> String? {
        let fm = FileManager.default
        let candidates = [
            "/opt/homebrew/share",
            "/usr/local/share",
            "/opt/homebrew/share/tessdata",
            "/usr/local/share/tessdata",
        ]
        for path in candidates where fm.fileExists(atPath: path) {
            if path.hasSuffix("/tessdata") {
                return String(path.dropLast("/tessdata".count))
            }
            return path
        }
        return nil
    }

    private nonisolated static func makeLogFile(prefix: String) throws -> (URL, FileHandle) {
        let fm = FileManager.default
        let url = fm.temporaryDirectory.appendingPathComponent("\(prefix)-\(UUID().uuidString).log")
        fm.createFile(atPath: url.path, contents: nil)
        return (url, try FileHandle(forWritingTo: url))
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

        // Source-path hint works reliably for local Xcode/Debug builds.
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
                for child in children {
                    let name = child.lastPathComponent.lowercased()
                    if name.contains("plos") {
                        candidates.append(child)
                    }
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
