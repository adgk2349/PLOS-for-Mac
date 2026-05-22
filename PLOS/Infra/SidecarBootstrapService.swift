import CryptoKit
import Darwin
import Foundation

enum SidecarBootstrapService {
    nonisolated static func ensureSidecarEnvironment(sidecarDirectory: URL, runtimeDirectory: URL) throws -> SidecarPythonRuntimeConfig {
        let fm = FileManager.default
        let includeMLX = shouldInstallMLXSupport()
        let runtimeVenvDirectory = runtimeDirectory.appendingPathComponent(".venv")
        let runtimeVenvPython = runtimeVenvDirectory.appendingPathComponent("bin/python3")
        let stagedSourceDirectory = runtimeDirectory.appendingPathComponent("staged-sidecar-\(UUID().uuidString)", isDirectory: true)
        let probeDataDirectory = runtimeDirectory.appendingPathComponent("probe-data", isDirectory: true)
        let runtimePackagePythonPath = runtimeDirectory.path
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
            probeDataDir: probeDataDirectory.path,
            includeMLX: includeMLX
        ) {
            return SidecarPythonRuntimeConfig(pythonExecutable: runtimeVenvPython.path, pythonPath: runtimePackagePythonPath)
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
                    runtimeDirectory: runtimeDirectory,
                    includeMLX: includeMLX
                )

                var candidateReady = hasRequiredModules(
                    python: candidateVenvPython.path,
                    pythonPath: runtimePackagePythonPath,
                    probeDataDir: probeDataDirectory.path,
                    includeMLX: includeMLX
                )
                if !candidateReady {
                    try installSidecarDependencies(
                        withPython: candidateVenvPython.path,
                        targetPath: nil,
                        runtimeDirectory: runtimeDirectory,
                        includeMLX: includeMLX,
                        force: true
                    )
                    candidateReady = hasRequiredModules(
                        python: candidateVenvPython.path,
                        pythonPath: runtimePackagePythonPath,
                        probeDataDir: probeDataDirectory.path,
                        includeMLX: includeMLX
                    )
                }

                if candidateReady {
                    try activateRuntimeVenv(candidateVenvDirectory: candidateVenvDirectory, runtimeVenvDirectory: runtimeVenvDirectory)
                    guard hasRequiredModules(
                        python: runtimeVenvPython.path,
                        pythonPath: runtimePackagePythonPath,
                        probeDataDir: probeDataDirectory.path,
                        includeMLX: includeMLX
                    ) else {
                        bootstrapErrors.append("\(systemPython): 활성화 후 모듈 점검 실패")
                        continue
                    }
                    return SidecarPythonRuntimeConfig(pythonExecutable: runtimeVenvPython.path, pythonPath: runtimePackagePythonPath)
                }
                bootstrapErrors.append("\(systemPython): 설치 후 모듈 점검 실패")
            } catch {
                bootstrapErrors.append("\(systemPython) [venv]: \(error.localizedDescription)")
            }
        }

        throw APIError(
            message: "sidecar 의존성 설치 실패: Python 환경 구성이 완료되지 않았습니다.\n\(bootstrapErrors.joined(separator: "\n"))"
        )
    }

    nonisolated static func stageSidecarSource(from sidecarDirectory: URL, to stagedSourceDirectory: URL) throws {
        let fm = FileManager.default
        try fm.createDirectory(at: stagedSourceDirectory, withIntermediateDirectories: true)

        let packageSource = sidecarDirectory.appendingPathComponent("local_ai_core", isDirectory: true)
        let pyprojectSource = sidecarDirectory.appendingPathComponent("pyproject.toml")
        let searxngSource = sidecarDirectory.appendingPathComponent("searxng", isDirectory: true)

        let packageTarget = stagedSourceDirectory.appendingPathComponent("local_ai_core", isDirectory: true)
        let pyprojectTarget = stagedSourceDirectory.appendingPathComponent("pyproject.toml")
        let searxngTarget = stagedSourceDirectory.appendingPathComponent("searxng", isDirectory: true)

        guard fm.fileExists(atPath: packageSource.path), fm.fileExists(atPath: pyprojectSource.path) else {
            throw APIError(message: "sidecar 소스 파일을 찾지 못했습니다. local_ai_core 또는 pyproject.toml이 없습니다.")
        }

        try copyDirectory(source: packageSource, destination: packageTarget)
        let pyprojectData = try Data(contentsOf: pyprojectSource)
        try pyprojectData.write(to: pyprojectTarget, options: .atomic)
        if fm.fileExists(atPath: searxngSource.path) {
            try copyDirectory(source: searxngSource, destination: searxngTarget)
        }
    }

    nonisolated static func syncRuntimePackage(stagedSourceDirectory: URL, runtimeDirectory: URL) throws {
        let fm = FileManager.default
        let stagedPackage = stagedSourceDirectory.appendingPathComponent("local_ai_core", isDirectory: true)
        let runtimePackage = runtimeDirectory.appendingPathComponent("local_ai_core", isDirectory: true)
        if fm.fileExists(atPath: runtimePackage.path) {
            try? fm.removeItem(at: runtimePackage)
        }
        try copyDirectory(source: stagedPackage, destination: runtimePackage)

        let stagedSearxng = stagedSourceDirectory.appendingPathComponent("searxng", isDirectory: true)
        let runtimeSearxng = runtimeDirectory.appendingPathComponent("searxng", isDirectory: true)
        if fm.fileExists(atPath: stagedSearxng.path) {
            if fm.fileExists(atPath: runtimeSearxng.path) {
                try? fm.removeItem(at: runtimeSearxng)
            }
            try copyDirectory(source: stagedSearxng, destination: runtimeSearxng)
        }
    }

    nonisolated static func installSidecarDependencies(
        withPython pythonExecutable: String,
        targetPath: String?,
        runtimeDirectory: URL,
        includeMLX: Bool,
        force: Bool = false
    ) throws {
        let stampURL = dependencyStampURL(runtimeDirectory: runtimeDirectory, pythonExecutable: pythonExecutable, targetPath: targetPath)
        let fingerprint = dependencyFingerprint(pythonExecutable: pythonExecutable, targetPath: targetPath, includeMLX: includeMLX)
        if !force, let existing = try? String(contentsOf: stampURL, encoding: .utf8).trimmingCharacters(in: .whitespacesAndNewlines), existing == fingerprint {
            return
        }

        let dependencies = sidecarPipDependencies(includeMLX: includeMLX)
        let coreDependencies = dependencies.filter { !$0.hasPrefix("mlx-lm") }

        var arguments = ["-m", "pip", "install", "--upgrade"]
        if let targetPath, !targetPath.isEmpty {
            arguments.append(contentsOf: ["--target", targetPath])
        }
        arguments.append(contentsOf: coreDependencies)
        try runCommand(
            executable: pythonExecutable,
            arguments: arguments,
            cwd: nil,
            step: "sidecar 의존성 설치"
        )

        if includeMLX, let mlxDependency = dependencies.first(where: { $0.hasPrefix("mlx-lm") }) {
            var optionalArguments = ["-m", "pip", "install", "--upgrade"]
            if let targetPath, !targetPath.isEmpty {
                optionalArguments.append(contentsOf: ["--target", targetPath])
            }
            optionalArguments.append(mlxDependency)
            try runCommand(
                executable: pythonExecutable,
                arguments: optionalArguments,
                cwd: nil,
                step: "sidecar 선택 의존성 설치(mlx)"
            )
        }
        try fingerprint.write(to: stampURL, atomically: true, encoding: .utf8)
    }

    nonisolated static func dependencyStampURL(
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

    nonisolated static func dependencyFingerprint(pythonExecutable: String, targetPath: String?, includeMLX: Bool) -> String {
        let lines = [
            "python=\(pythonVersionLabel(pythonExecutable))",
            "target=\(targetPath ?? "")",
            "deps=\(sidecarPipDependencies(includeMLX: includeMLX).joined(separator: "|"))",
        ]
        return lines.joined(separator: "\n")
    }

    nonisolated static func pythonVersionLabel(_ executable: String) -> String {
        guard let version = pythonVersionTuple(executable) else {
            return executable
        }
        return "\(version.0).\(version.1)"
    }

    nonisolated static func sidecarPipDependencies(includeMLX: Bool) -> [String] {
        var deps = [
            "fastapi>=0.116.0",
            "uvicorn[standard]>=0.35.0",
            "pydantic>=2.8.0",
            "pydantic-settings>=2.3.0",
            "httpx>=0.28.0",
            "sentry-sdk[fastapi]>=2.20.0",
            "lancedb>=0.22.0",
            "numpy>=2.2.0",
            "pypdf>=5.3.0",
            "pytesseract>=0.3.13",
            "Pillow>=11.1.0",
            "pdf2image>=1.17.0",
            "pypdfium2>=4.30.0",
            "rapidocr_onnxruntime>=1.4.4",
            "cryptography>=3.1",
            // Keep hf-hub below 2.x and verify runtime API compatibility in probe.
            "huggingface-hub>=0.34.0,<2.0.0",
            "sentence-transformers>=3.0.0",
            "llama-cpp-python>=0.3.9"
        ]
        if includeMLX {
            deps.append("mlx-lm>=0.31.2")
        }
        return deps
    }

    nonisolated static func shouldInstallMLXSupport() -> Bool {
        let overrideRaw = (ProcessInfo.processInfo.environment["LOCAL_AI_FORCE_MLX_LM"] ?? "")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        if ["1", "true", "yes", "on"].contains(overrideRaw) {
            return true
        }
        if ["0", "false", "no", "off"].contains(overrideRaw) {
            return false
        }
        #if arch(arm64)
        return true
        #else
        return false
        #endif
    }

    nonisolated static func copyDirectory(source: URL, destination: URL) throws {
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

    nonisolated static func acquireRuntimeBootstrapLock(runtimeDirectory: URL) throws -> Int32 {
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

    nonisolated static func releaseRuntimeBootstrapLock(_ fd: Int32) {
        guard fd >= 0 else { return }
        _ = flock(fd, LOCK_UN)
        _ = close(fd)
    }

    nonisolated static func activateRuntimeVenv(candidateVenvDirectory: URL, runtimeVenvDirectory: URL) throws {
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

    nonisolated static func prepareRuntimeDirectory(preferredDirectory: URL? = nil) throws -> URL {
        let fm = FileManager.default
        var candidates: [URL] = []

        if let preferredDirectory {
            candidates = [preferredDirectory.standardizedFileURL]
        } else {
            throw APIError(message: "런타임 디렉터리가 지정되지 않았습니다. 단일 런타임 폴더 경로를 설정해 주세요.")
        }

        var deduped: [URL] = []
        var seen = Set<String>()
        for candidate in candidates {
            let path = candidate.standardizedFileURL.path
            if seen.insert(path).inserted {
                deduped.append(candidate.standardizedFileURL)
            }
        }

        var errors: [String] = []
        for dir in deduped {
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

    nonisolated static func hasRequiredModules(
        python: String,
        pythonPath: String?,
        probeDataDir: String,
        includeMLX: Bool
    ) -> Bool {
        do {
            var overrides: [String: String] = [
                "LOCAL_AI_DATA_DIR": probeDataDir,
                "LOCAL_AI_SESSION_TOKEN": "probe-session-token",
            ]
            if let pythonPath, !pythonPath.isEmpty {
                overrides["PYTHONPATH"] = pythonPath
            }
            let mlxProbe = includeMLX ? "; u.find_spec('mlx_lm')" : ""
            try runCommand(
                executable: python,
                arguments: [
                    "-c",
                    "import importlib.util as u, inspect; import uvicorn, fastapi, httpx, pydantic, lancedb, local_ai_core.main as m; assert u.find_spec('pypdf') or u.find_spec('PyPDF2') or u.find_spec('pypdfium2'); assert u.find_spec('cryptography'); assert u.find_spec('huggingface_hub'); from huggingface_hub.utils import get_session; assert callable(get_session); assert u.find_spec('rapidocr_onnxruntime'); assert u.find_spec('sentence_transformers'); assert u.find_spec('llama_cpp')\(mlxProbe); src=inspect.getsource(m.create_app); assert '/v1/docs' in src and '/v1/models/download' in src and '/v1/models/catalog' in src"
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

    nonisolated static func resolveSystemPythonExecutables() -> [String] {
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

    nonisolated static func isRunnablePythonExecutable(_ path: String) -> Bool {
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

    nonisolated static func pythonVersionTuple(_ path: String) -> (Int, Int)? {
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

    nonisolated static func runCommand(
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
        env["PATH"] = SidecarEnvironmentService.normalizedRuntimePath(existing: env["PATH"])
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


}
