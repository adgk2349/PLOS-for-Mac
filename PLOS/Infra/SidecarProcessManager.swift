import Combine
import Darwin
import Foundation

enum SidecarMlxKVQMode: String, CaseIterable, Codable, Identifiable {
    case off
    case turbo3
    case turbo4

    var id: String { rawValue }
}

struct SidecarPythonRuntimeConfig: Sendable {
    let pythonExecutable: String
    let pythonPath: String?
}

struct SidecarStorageResolution: Sendable {
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

    private struct HardwareProfile {
        let memoryGB: Int
        let cpuCores: Int
        let appleSilicon: Bool
    }

    private struct RuntimeTuning {
        let llamaNCtx: Int
        let llamaNBatch: Int
        let llamaNUbatch: Int
        let inferenceTimeoutSeconds: Int
        let routeTimeoutSeconds: Int
        let inferenceMaxWorkers: Int
        let inferenceQueueTimeoutSeconds: Double
        let inferenceQueueRetries: Int
        let inferenceQueueTotalWaitCapSeconds: Double
        let generationRetryMaxAttempts: Int
        let generationRetryTotalBudgetMs: Int
        let generationRetryTokenBackoffSteps: String
        let conversationMaxTokensScale: Double
        let streamQueueMaxItems: Int
        let streamBatchChars: Int
    }

    private struct BootstrapPreparation: Sendable {
        let runtimeConfig: SidecarPythonRuntimeConfig
        let runtimeDirectory: URL
        let dataDirectory: URL
        let modelsDirectory: URL
        let modelFallbackReason: String?
        let runtimeFallbackReason: String?
    }

    @Published private(set) var isRunning = false
    @Published private(set) var storageResolution: SidecarStorageResolution?
    @Published private(set) var lastTerminationContext: String?

    private(set) var sessionToken = UUID().uuidString
    private(set) var apiClient: SidecarAPIClient?

    private var process: Process?
    private var sidecarLogURL: URL?
    private var sidecarLogHandle: FileHandle?
    private var isStarting = false
    private var isStopping = false
    private var preferredModelsDirectory: URL?
    private var preferredRuntimeDirectory: URL?
    private var visionEnabled: Bool = true
    private var visionCaptionModel: String = "microsoft/git-base-coco"
    private var visionClassifyModel: String = "google/vit-base-patch16-224"
    private var mlxKVQEnabled: Bool = false
    private var mlxKVQMode: SidecarMlxKVQMode = .off
    private var mlxKVQBits: Int = 3
    private var conversationTurboEnabled: Bool = false
    private var inferenceTimeoutDisabled: Bool = false
    private var mainResponseTimeoutSeconds: Int = 180
    private var auxiliaryTimeoutSeconds: Int = 10

    private let host = "127.0.0.1"
    private var preferredPort = 8777

    private nonisolated static func detectHardwareProfile(env: [String: String]) -> HardwareProfile {
        let allowExternalOverrideRaw = (env["LOCAL_AI_ALLOW_EXTERNAL_MEMORY_OVERRIDE"] ?? "0")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let allowExternalOverride = ["1", "true", "yes", "on"].contains(allowExternalOverrideRaw)
        let overrideMemory = allowExternalOverride
            ? (env["LOCAL_AI_SYSTEM_MEMORY_GB_OVERRIDE"] ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            : ""
        let memoryGB: Int = {
            if let parsed = Int(overrideMemory), parsed > 0 {
                return parsed
            }
            let physicalMemory = ProcessInfo.processInfo.physicalMemory
            let gb = Int(physicalMemory / 1_073_741_824)
            return max(1, gb)
        }()
        let cpuCores = max(1, ProcessInfo.processInfo.activeProcessorCount)
        return HardwareProfile(
            memoryGB: memoryGB,
            cpuCores: cpuCores,
            appleSilicon: isAppleSilicon()
        )
    }

    private nonisolated static func isAppleSilicon() -> Bool {
        var value: Int32 = 0
        var size = MemoryLayout<Int32>.size
        let result = sysctlbyname("hw.optional.arm64", &value, &size, nil, 0)
        return result == 0 && value == 1
    }

    private nonisolated static func runtimeTuning(for profile: HardwareProfile, turboEnabled: Bool) -> RuntimeTuning {
        var nCtx = 4096
        var nBatch = 512
        var nUBatch = 256
        var inferenceTimeout = turboEnabled ? 240 : 160
        var routeTimeout = turboEnabled ? 320 : 240
        var inferenceWorkers = 2
        var inferenceQueueTimeout: Double = 10
        var inferenceQueueRetries = 2
        var inferenceQueueTotalWaitCap: Double = 30
        var generationRetryMaxAttempts = 3
        var generationRetryTotalBudgetMs = 45_000
        var generationRetryTokenBackoffSteps = "1.0,0.65,0.45"
        var conversationMaxTokensScale: Double = 1.0
        var streamQueueMaxItems = 512
        var streamBatchChars = 180

        switch profile.memoryGB {
        case ...8:
            nCtx = 2048
            nBatch = 192
            nUBatch = 96
            inferenceTimeout = turboEnabled ? 180 : 120
            routeTimeout = turboEnabled ? 240 : 170
            inferenceWorkers = 1
            inferenceQueueTimeout = 6
            inferenceQueueRetries = 1
            inferenceQueueTotalWaitCap = 10
            generationRetryMaxAttempts = 2
            generationRetryTotalBudgetMs = 18_000
            generationRetryTokenBackoffSteps = "0.85,0.55"
            conversationMaxTokensScale = 0.58
            streamQueueMaxItems = 256
            streamBatchChars = 140
        case ...16:
            nCtx = 4096
            nBatch = 192
            nUBatch = 96
            inferenceTimeout = turboEnabled ? 220 : 150
            routeTimeout = turboEnabled ? 300 : 210
            inferenceWorkers = 1
            inferenceQueueTimeout = 7
            inferenceQueueRetries = 1
            inferenceQueueTotalWaitCap = 14
            generationRetryMaxAttempts = 2
            generationRetryTotalBudgetMs = 22_000
            generationRetryTokenBackoffSteps = "0.90,0.60"
            conversationMaxTokensScale = 0.70
            streamQueueMaxItems = 320
            streamBatchChars = 150
        case ...24:
            nCtx = 4096
            nBatch = 384
            nUBatch = 192
            inferenceTimeout = turboEnabled ? 200 : 120
            routeTimeout = turboEnabled ? 260 : 180
            inferenceWorkers = 2
            inferenceQueueTimeout = 8
            inferenceQueueRetries = 2
            inferenceQueueTotalWaitCap = 22
            generationRetryMaxAttempts = 3
            generationRetryTotalBudgetMs = 30_000
            generationRetryTokenBackoffSteps = "1.0,0.65,0.45"
            conversationMaxTokensScale = 0.82
            streamQueueMaxItems = 384
            streamBatchChars = 170
        case ...32:
            nCtx = 4096
            nBatch = 512
            nUBatch = 256
            inferenceTimeout = turboEnabled ? 180 : 100
            routeTimeout = turboEnabled ? 240 : 160
            inferenceWorkers = 2
            conversationMaxTokensScale = 0.92
        case ...64:
            nCtx = 6144
            nBatch = 768
            nUBatch = 384
            inferenceTimeout = turboEnabled ? 160 : 90
            routeTimeout = turboEnabled ? 220 : 150
            inferenceWorkers = 2
        default:
            nCtx = 8192
            nBatch = 1024
            nUBatch = 512
            inferenceTimeout = turboEnabled ? 140 : 75
            routeTimeout = turboEnabled ? 200 : 130
            inferenceWorkers = 3
        }

        if profile.cpuCores <= 6 {
            nBatch = max(128, Int(Double(nBatch) * 0.70))
            nUBatch = max(64, Int(Double(nUBatch) * 0.70))
        } else if profile.cpuCores <= 8 {
            nBatch = max(128, Int(Double(nBatch) * 0.85))
            nUBatch = max(64, Int(Double(nUBatch) * 0.85))
        }

        if !profile.appleSilicon {
            nBatch = max(96, Int(Double(nBatch) * 0.75))
            nUBatch = max(48, Int(Double(nUBatch) * 0.75))
            nCtx = max(2048, Int(Double(nCtx) * 0.75))
            inferenceTimeout = Int(Double(inferenceTimeout) * 1.15)
            routeTimeout = Int(Double(routeTimeout) * 1.15)
            conversationMaxTokensScale = min(conversationMaxTokensScale, 0.85)
            inferenceWorkers = min(inferenceWorkers, 2)
        }

        nBatch = max(32, min(4096, nBatch))
        nUBatch = max(16, min(nBatch, nUBatch))
        nCtx = max(512, min(32768, nCtx))
        inferenceTimeout = max(45, inferenceTimeout)
        routeTimeout = max(inferenceTimeout, routeTimeout)
        inferenceWorkers = max(1, min(4, inferenceWorkers))
        inferenceQueueTimeout = max(1, min(30, inferenceQueueTimeout))
        inferenceQueueRetries = max(0, min(5, inferenceQueueRetries))
        inferenceQueueTotalWaitCap = max(inferenceQueueTimeout, min(120, inferenceQueueTotalWaitCap))
        generationRetryMaxAttempts = max(1, min(6, generationRetryMaxAttempts))
        generationRetryTotalBudgetMs = max(8_000, min(180_000, generationRetryTotalBudgetMs))
        conversationMaxTokensScale = max(0.35, min(1.0, conversationMaxTokensScale))
        streamQueueMaxItems = max(96, min(1024, streamQueueMaxItems))
        streamBatchChars = max(120, min(900, streamBatchChars))

        return RuntimeTuning(
            llamaNCtx: nCtx,
            llamaNBatch: nBatch,
            llamaNUbatch: nUBatch,
            inferenceTimeoutSeconds: inferenceTimeout,
            routeTimeoutSeconds: routeTimeout,
            inferenceMaxWorkers: inferenceWorkers,
            inferenceQueueTimeoutSeconds: inferenceQueueTimeout,
            inferenceQueueRetries: inferenceQueueRetries,
            inferenceQueueTotalWaitCapSeconds: inferenceQueueTotalWaitCap,
            generationRetryMaxAttempts: generationRetryMaxAttempts,
            generationRetryTotalBudgetMs: generationRetryTotalBudgetMs,
            generationRetryTokenBackoffSteps: generationRetryTokenBackoffSteps,
            conversationMaxTokensScale: conversationMaxTokensScale,
            streamQueueMaxItems: streamQueueMaxItems,
            streamBatchChars: streamBatchChars
        )
    }

    func configureStorageDirectories(modelsDirectory: URL?, runtimeDirectory: URL?) {
        preferredModelsDirectory = modelsDirectory?.standardizedFileURL
        preferredRuntimeDirectory = runtimeDirectory?.standardizedFileURL
    }

    func configureVisionRuntime(enabled: Bool, captionModel: String, classifyModel: String) {
        visionEnabled = enabled
        visionCaptionModel = captionModel.trimmingCharacters(in: .whitespacesAndNewlines)
        visionClassifyModel = classifyModel.trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func configureMlxKVQRuntime(enabled: Bool, mode: SidecarMlxKVQMode, bits: Int) {
        mlxKVQEnabled = enabled
        mlxKVQMode = mode
        mlxKVQBits = min(8, max(2, bits))
    }

    func configureConversationTurboRuntime(enabled: Bool) {
        conversationTurboEnabled = enabled
    }

    func configureInferenceTimeoutRuntime(
        disabled: Bool,
        mainResponseTimeoutSeconds: Int? = nil,
        auxiliaryTimeoutSeconds: Int? = nil
    ) {
        inferenceTimeoutDisabled = disabled
        if let mainResponseTimeoutSeconds {
            self.mainResponseTimeoutSeconds = min(3600, max(30, mainResponseTimeoutSeconds))
        }
        if let auxiliaryTimeoutSeconds {
            self.auxiliaryTimeoutSeconds = min(120, max(4, auxiliaryTimeoutSeconds))
        }
    }

    func start() async throws {
        if isRunning, let client = apiClient {
            do {
                _ = try await client.health()
                _ = try await client.getSettings()
                return
            } catch {
                await stop()
                sessionToken = UUID().uuidString
            }
        }

        if isStarting {
            while isStarting {
                if isRunning, apiClient != nil {
                    return
                }
                try await Task.sleep(nanoseconds: 100_000_000)
            }
            if isRunning, apiClient != nil {
                return
            }
        }

        isStarting = true
        defer { isStarting = false }
        lastTerminationContext = nil

        sessionToken = UUID().uuidString

        let sidecarDirectory = try resolveSidecarDirectory()
        let preferredRuntimeDirectory = self.preferredRuntimeDirectory
        let preferredModelsDirectory = self.preferredModelsDirectory
        let requestedRuntimeDirectory = (preferredRuntimeDirectory ?? Self.defaultRuntimeDirectory()).standardizedFileURL
        let requestedModelsDirectory = (preferredModelsDirectory ?? Self.defaultModelsDirectory()).standardizedFileURL

        let preparation = try await Task.detached(priority: .userInitiated) {
            try Self.prepareBootstrapArtifacts(
                sidecarDirectory: sidecarDirectory,
                requestedRuntimeDirectory: requestedRuntimeDirectory,
                requestedModelsDirectory: requestedModelsDirectory,
                preferredRuntimeDirectory: preferredRuntimeDirectory,
                preferredModelsDirectory: preferredModelsDirectory
            )
        }.value

        let runtimeConfig = preparation.runtimeConfig
        let runtimeDirectory = preparation.runtimeDirectory
        let dataDirectory = preparation.dataDirectory
        let modelsDirectory = preparation.modelsDirectory

        storageResolution = SidecarStorageResolution(
            requestedModelsDirectory: requestedModelsDirectory,
            effectiveModelsDirectory: modelsDirectory,
            requestedRuntimeDirectory: requestedRuntimeDirectory,
            effectiveRuntimeDirectory: runtimeDirectory,
            modelFallbackReason: preparation.modelFallbackReason,
            runtimeFallbackReason: preparation.runtimeFallbackReason
        )

        let openAIKey = SidecarSecretStore.read("openai_api_key")
        let anthropicKey = SidecarSecretStore.read("anthropic_api_key")

        let ports = SidecarPortService.portCandidates(preferred: preferredPort)
        SidecarPortService.terminateStaleRuntimeSidecars(runtimeDirectory: runtimeDirectory, ports: ports)

        var lastError: Error?

        for port in ports {
            if SidecarPortService.isPortListening(port) {
                let existingClient = SidecarAPIClient(
                    baseURL: URL(string: "http://\(host):\(port)")!,
                    sessionToken: sessionToken
                )
                do {
                    try await waitUntilHealthy(client: existingClient)
                    _ = try await existingClient.getSettings()
                    apiClient = existingClient
                    preferredPort = port
                    isRunning = true
                    return
                } catch {
                    // Port is occupied by another process/session; try next candidate.
                }
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
                    await terminateLaunchedProcess(launched.process)
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

    private func waitForExit(_ proc: Process, polls: Int, intervalNs: UInt64) async -> Bool {
        for _ in 0 ..< max(1, polls) {
            if !proc.isRunning {
                return true
            }
            try? await Task.sleep(nanoseconds: intervalNs)
        }
        return !proc.isRunning
    }

    private func terminateLaunchedProcess(_ proc: Process) async {
        guard proc.isRunning else { return }
        proc.terminate()
        _ = await waitForExit(proc, polls: 20, intervalNs: 50_000_000)
        if proc.isRunning {
            proc.interrupt()
            _ = await waitForExit(proc, polls: 10, intervalNs: 50_000_000)
        }
        if proc.isRunning {
            _ = kill(proc.processIdentifier, SIGKILL)
            _ = await waitForExit(proc, polls: 10, intervalNs: 20_000_000)
        }
    }

    func stop() async {
        isStopping = true
        if let proc = process, proc.isRunning {
            proc.terminate()
            _ = await waitForExit(proc, polls: 20, intervalNs: 50_000_000)
            if proc.isRunning {
                proc.interrupt()
                _ = await waitForExit(proc, polls: 10, intervalNs: 50_000_000)
            }
            if proc.isRunning {
                _ = kill(proc.processIdentifier, SIGKILL)
                _ = await waitForExit(proc, polls: 10, intervalNs: 20_000_000)
            }
        }

        process = nil
        try? sidecarLogHandle?.close()
        sidecarLogHandle = nil
        apiClient = nil
        isRunning = false
        isStarting = false
        isStopping = false
    }

    private func waitUntilHealthy(client: SidecarAPIClient) async throws {
        for _ in 0 ..< 80 {
            if let process, !process.isRunning {
                throw APIError(message: "Sidecar가 시작 직후 종료되었습니다.\n\(sidecarLogSummary())")
            }
            do {
                _ = try await client.health()
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
        let hardwareProfile = Self.detectHardwareProfile(env: env)
        let tuning = Self.runtimeTuning(for: hardwareProfile, turboEnabled: conversationTurboEnabled)
        env["LOCAL_AI_SESSION_TOKEN"] = sessionToken
        env["LOCAL_AI_DATA_DIR"] = dataDirectory.path
        env["LOCAL_AI_STRICT_DATA_DIR"] = "1"
        env["LOCAL_AI_MODELS_DIR"] = modelsDirectory.path
        env["LOCAL_AI_PARENT_PID"] = String(ProcessInfo.processInfo.processIdentifier)
        env["PYTHONUNBUFFERED"] = "1"
        env["PATH"] = SidecarEnvironmentService.normalizedRuntimePath(existing: env["PATH"])
        if env["LOCAL_AI_LLAMA_OPT_ENABLED"] == nil { env["LOCAL_AI_LLAMA_OPT_ENABLED"] = "1" }
        if env["LOCAL_AI_LLAMA_FLASH_ATTN"] == nil { env["LOCAL_AI_LLAMA_FLASH_ATTN"] = "1" }
        if env["LOCAL_AI_LLAMA_OFFLOAD_KQV"] == nil { env["LOCAL_AI_LLAMA_OFFLOAD_KQV"] = "1" }
        if env["LOCAL_AI_LLAMA_TYPE_K"] == nil { env["LOCAL_AI_LLAMA_TYPE_K"] = "Q4_0" }
        if env["LOCAL_AI_LLAMA_TYPE_V"] == nil { env["LOCAL_AI_LLAMA_TYPE_V"] = "Q4_0" }
        if env["LOCAL_AI_SYSTEM_MEMORY_GB_OVERRIDE"] == nil {
            env["LOCAL_AI_SYSTEM_MEMORY_GB_OVERRIDE"] = String(hardwareProfile.memoryGB)
        }
        if env["LOCAL_AI_LLAMA_N_CTX"] == nil { env["LOCAL_AI_LLAMA_N_CTX"] = String(tuning.llamaNCtx) }
        if env["LOCAL_AI_LLAMA_N_BATCH"] == nil { env["LOCAL_AI_LLAMA_N_BATCH"] = String(tuning.llamaNBatch) }
        if env["LOCAL_AI_LLAMA_N_UBATCH"] == nil { env["LOCAL_AI_LLAMA_N_UBATCH"] = String(tuning.llamaNUbatch) }
        if env["LOCAL_AI_INFERENCE_MAX_WORKERS"] == nil { env["LOCAL_AI_INFERENCE_MAX_WORKERS"] = String(tuning.inferenceMaxWorkers) }
        if env["LOCAL_AI_INFERENCE_QUEUE_TIMEOUT_SECONDS"] == nil { env["LOCAL_AI_INFERENCE_QUEUE_TIMEOUT_SECONDS"] = String(format: "%.1f", tuning.inferenceQueueTimeoutSeconds) }
        if env["LOCAL_AI_INFERENCE_QUEUE_RETRIES"] == nil { env["LOCAL_AI_INFERENCE_QUEUE_RETRIES"] = String(tuning.inferenceQueueRetries) }
        if env["LOCAL_AI_INFERENCE_QUEUE_TOTAL_WAIT_CAP_SECONDS"] == nil { env["LOCAL_AI_INFERENCE_QUEUE_TOTAL_WAIT_CAP_SECONDS"] = String(format: "%.1f", tuning.inferenceQueueTotalWaitCapSeconds) }
        if env["GEN_RETRY_MAX_ATTEMPTS"] == nil { env["GEN_RETRY_MAX_ATTEMPTS"] = String(tuning.generationRetryMaxAttempts) }
        if env["GEN_RETRY_TOKEN_BACKOFF_STEPS"] == nil { env["GEN_RETRY_TOKEN_BACKOFF_STEPS"] = tuning.generationRetryTokenBackoffSteps }
        if env["LOCAL_AI_CONVERSATION_MAX_TOKENS_SCALE"] == nil { env["LOCAL_AI_CONVERSATION_MAX_TOKENS_SCALE"] = String(format: "%.2f", tuning.conversationMaxTokensScale) }
        if env["LOCAL_AI_FORCE_MAX_CONVERSATION_TOKENS"] == nil { env["LOCAL_AI_FORCE_MAX_CONVERSATION_TOKENS"] = "1" }
        if env["LOCAL_AI_STREAM_QUEUE_MAX_ITEMS"] == nil { env["LOCAL_AI_STREAM_QUEUE_MAX_ITEMS"] = String(tuning.streamQueueMaxItems) }
        if env["LOCAL_AI_STREAM_BATCH_CHARS"] == nil { env["LOCAL_AI_STREAM_BATCH_CHARS"] = String(tuning.streamBatchChars) }
        if env["GEN_RETRY_NO_FALLBACK"] == nil { env["GEN_RETRY_NO_FALLBACK"] = "0" }
        env["LOCAL_AI_POPPLER_PATH"] = SidecarEnvironmentService.detectPopplerDirectory() ?? "/opt/homebrew/bin"
        env["LOCAL_AI_TESSERACT_CMD"] = SidecarEnvironmentService.detectTesseractExecutable() ?? "/opt/homebrew/bin/tesseract"
        env["LOCAL_AI_VISION_ENABLED"] = visionEnabled ? "1" : "0"
        if !visionCaptionModel.isEmpty {
            env["LOCAL_AI_VISION_CAPTION_MODEL"] = visionCaptionModel
        }
        if !visionClassifyModel.isEmpty {
            env["LOCAL_AI_VISION_CLASSIFY_MODEL"] = visionClassifyModel
        }
        env["LOCAL_AI_MLX_KVQ_ENABLED"] = mlxKVQEnabled ? "1" : "0"
        env["LOCAL_AI_MLX_KVQ_MODE"] = mlxKVQMode.rawValue
        env["LOCAL_AI_MLX_KVQ_BITS"] = String(mlxKVQBits)
        if env["LOCAL_AI_MLX_ISOLATE_PROCESS"] == nil { env["LOCAL_AI_MLX_ISOLATE_PROCESS"] = "1" }
        if env["LOCAL_AI_MLX_ISOLATED_TIMEOUT_SECONDS"] == nil { env["LOCAL_AI_MLX_ISOLATED_TIMEOUT_SECONDS"] = "240" }
        if env["LOCAL_AI_STREAM_TOKEN_DELAY_MS"] == nil { env["LOCAL_AI_STREAM_TOKEN_DELAY_MS"] = "15" }
        if env["LOCAL_AI_FALLBACK_ENGINE_SWITCH"] == nil { env["LOCAL_AI_FALLBACK_ENGINE_SWITCH"] = "0" }
        if env["LOCAL_AI_MLX_RETRY_ONLY"] == nil { env["LOCAL_AI_MLX_RETRY_ONLY"] = "1" }
        env["LOCAL_AI_CONVERSATION_TURBO"] = conversationTurboEnabled ? "1" : "0"
        let resolvedMainTimeout = min(3600, max(30, mainResponseTimeoutSeconds))
        let resolvedAuxTimeout = min(120, max(4, auxiliaryTimeoutSeconds))
        let forceMaxConversationTokensRaw = (env["LOCAL_AI_FORCE_MAX_CONVERSATION_TOKENS"] ?? "0")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let forceMaxConversationTokens = ["1", "true", "yes", "on"].contains(forceMaxConversationTokensRaw)
        let effectiveMainTimeout = forceMaxConversationTokens
            ? max(resolvedMainTimeout, 420)
            : resolvedMainTimeout
        let tunedRetryBudgetMs = tuning.generationRetryTotalBudgetMs
        let timeoutAlignedRetryBudgetMs = min(180_000, max(8_000, effectiveMainTimeout * 1_000))
        let effectiveRetryBudgetMs = max(tunedRetryBudgetMs, timeoutAlignedRetryBudgetMs)
        env["LOCAL_AI_MAIN_RESPONSE_TIMEOUT_SECONDS"] = inferenceTimeoutDisabled ? "0" : String(effectiveMainTimeout)
        env["LOCAL_AI_AUX_TIMEOUT_SECONDS"] = String(resolvedAuxTimeout)
        env["LOCAL_AI_CLARIFICATION_TIMEOUT_SECONDS"] = String(resolvedAuxTimeout)
        env["LOCAL_AI_RAG_HYPOTHESIS_TIMEOUT_SECONDS"] = String(max(8, min(40, resolvedAuxTimeout + 6)))
        if inferenceTimeoutDisabled {
            env["LOCAL_AI_INFERENCE_TIMEOUT_SECONDS"] = String(resolvedAuxTimeout)
            env["LOCAL_AI_ROUTE_TIMEOUT_SECONDS"] = "0"
            env["GEN_RETRY_TOTAL_BUDGET_MS"] = String(effectiveRetryBudgetMs)
        } else {
            let effectiveInferenceTimeout = max(
                tuning.inferenceTimeoutSeconds,
                min(1800, effectiveMainTimeout)
            )
            let effectiveRouteTimeout = max(
                max(tuning.routeTimeoutSeconds, effectiveInferenceTimeout + 30),
                min(3600, effectiveMainTimeout + 60)
            )
            if env["LOCAL_AI_INFERENCE_TIMEOUT_SECONDS"] == nil {
                env["LOCAL_AI_INFERENCE_TIMEOUT_SECONDS"] = String(effectiveInferenceTimeout)
            }
            if env["LOCAL_AI_ROUTE_TIMEOUT_SECONDS"] == nil {
                env["LOCAL_AI_ROUTE_TIMEOUT_SECONDS"] = String(effectiveRouteTimeout)
            }
            env["GEN_RETRY_TOTAL_BUDGET_MS"] = String(effectiveRetryBudgetMs)
        }
        let disableInferenceLimitsRaw = (env["LOCAL_AI_DISABLE_INFERENCE_LIMITS"] ?? "0")
            .trimmingCharacters(in: .whitespacesAndNewlines)
            .lowercased()
        let disableInferenceLimits = ["1", "true", "yes", "on"].contains(disableInferenceLimitsRaw)
        if disableInferenceLimits {
            // Fail-open mode: prefer waiting for real model output over timeout fallback text.
            env["LOCAL_AI_MAIN_RESPONSE_TIMEOUT_SECONDS"] = "0"
            env["LOCAL_AI_INFERENCE_TIMEOUT_SECONDS"] = "0"
            env["LOCAL_AI_ROUTE_TIMEOUT_SECONDS"] = "0"
            env["LOCAL_AI_INFERENCE_QUEUE_TIMEOUT_SECONDS"] = "60"
            env["LOCAL_AI_INFERENCE_QUEUE_RETRIES"] = "5"
            env["LOCAL_AI_INFERENCE_QUEUE_TOTAL_WAIT_CAP_SECONDS"] = "120"
            env["GEN_RETRY_TOTAL_BUDGET_MS"] = "180000"
            env["GEN_RETRY_UNTIL_CANCEL"] = "1"
            env["GEN_RETRY_UNTIL_CANCEL_MAX_ATTEMPTS"] = "64"
            env["GEN_RETRY_UNTIL_CANCEL_MAX_SECONDS"] = "1800"
            env["GEN_RETRY_LAST_RESORT_CLARIFY"] = "0"
        }
        if env["LOCAL_AI_PLUGIN_RUNTIME_MODE"] == nil {
            env["LOCAL_AI_PLUGIN_RUNTIME_MODE"] = "disabled"
        }
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
        process.terminationHandler = { [weak self] terminated in
            guard let self else { return }
            Task { @MainActor in
                self.handleProcessTermination(terminated)
            }
        }

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
        let crashInfo: String = {
            let value = (lastTerminationContext ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            guard !value.isEmpty else { return "" }
            return "최근 sidecar 종료 정보:\n\(value)\n"
        }()
        guard let url = sidecarLogURL else {
            return crashInfo.isEmpty ? "sidecar 로그 파일을 찾지 못했습니다." : "\(crashInfo)sidecar 로그 파일을 찾지 못했습니다."
        }
        guard let data = try? Data(contentsOf: url), !data.isEmpty else {
            return crashInfo.isEmpty ? "sidecar 로그가 비어 있습니다." : "\(crashInfo)sidecar 로그가 비어 있습니다."
        }
        let text = String(decoding: data, as: UTF8.self)
        return "\(crashInfo)최근 sidecar 로그:\n\(text.suffix(1400))"
    }

    private func handleProcessTermination(_ terminated: Process) {
        guard !isStopping else { return }
        guard process === terminated else { return }
        let reasonText: String
        switch terminated.terminationReason {
        case .exit:
            reasonText = "exit"
        case .uncaughtSignal:
            reasonText = "uncaught_signal"
        @unknown default:
            reasonText = "unknown"
        }
        let logTail = readSidecarLogTail(maxCharacters: 900)
        if let logTail, !logTail.isEmpty {
            lastTerminationContext = "reason=\(reasonText); status=\(terminated.terminationStatus)\nlog_tail:\n\(logTail)"
        } else {
            lastTerminationContext = "reason=\(reasonText); status=\(terminated.terminationStatus)"
        }
        process = nil
        try? sidecarLogHandle?.close()
        sidecarLogHandle = nil
        apiClient = nil
        isRunning = false
        isStarting = false
    }

    private func readSidecarLogTail(maxCharacters: Int) -> String? {
        guard let url = sidecarLogURL else { return nil }
        guard let data = try? Data(contentsOf: url), !data.isEmpty else { return nil }
        let text = String(decoding: data, as: UTF8.self)
        let tail = String(text.suffix(max(200, maxCharacters))).trimmingCharacters(in: .whitespacesAndNewlines)
        return tail.isEmpty ? nil : tail
    }

    private nonisolated static func isInvalidSessionTokenError(_ error: Error) -> Bool {
        guard let apiError = error as? APIError else {
            return false
        }
        let lower = apiError.message.lowercased()
        return lower.contains("http 401") && lower.contains("invalid session token")
    }

    private nonisolated static func prepareBootstrapArtifacts(
        sidecarDirectory: URL,
        requestedRuntimeDirectory: URL,
        requestedModelsDirectory: URL,
        preferredRuntimeDirectory: URL?,
        preferredModelsDirectory: URL?
    ) throws -> BootstrapPreparation {
        let runtimeDirectory = try SidecarBootstrapService.prepareRuntimeDirectory(
            preferredDirectory: requestedRuntimeDirectory
        )
        let runtimeFallbackReason: String? = runtimeDirectory.standardizedFileURL.path == requestedRuntimeDirectory.path
            ? nil
            : "요청 경로를 사용할 수 없어 \(runtimeDirectory.path)로 폴백"

        migrateRuntimeVenvIfNeeded(
            targetRuntimeDirectory: runtimeDirectory,
            legacyRuntimeDirectories: legacyRuntimeDirectories(
                excluding: runtimeDirectory,
                preferredRuntimeDirectory: preferredRuntimeDirectory
            )
        )

        let runtimeConfig = try SidecarBootstrapService.ensureSidecarEnvironment(
            sidecarDirectory: sidecarDirectory,
            runtimeDirectory: runtimeDirectory
        )
        let dataDirectory = try resolveDataDirectory(
            sidecarDirectory: sidecarDirectory,
            runtimeDirectory: runtimeDirectory
        )
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
                excluding: modelsDirectory,
                preferredModelsDirectory: preferredModelsDirectory
            )
        )

        return BootstrapPreparation(
            runtimeConfig: runtimeConfig,
            runtimeDirectory: runtimeDirectory,
            dataDirectory: dataDirectory,
            modelsDirectory: modelsDirectory,
            modelFallbackReason: modelFallbackReason,
            runtimeFallbackReason: runtimeFallbackReason
        )
    }

    private nonisolated static func resolveDataDirectory(sidecarDirectory: URL, runtimeDirectory: URL) throws -> URL {
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

    private nonisolated static func defaultModelsDirectory() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("PLOS/LocalAI/models", isDirectory: true)
            .standardizedFileURL
    }

    private nonisolated static func defaultRuntimeDirectory() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("PLOS/LocalAI/runtime", isDirectory: true)
            .standardizedFileURL
    }

    private nonisolated static func appSupportRuntimeDirectory() -> URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Library/Application Support/LocalAICore/SidecarRuntime", isDirectory: true)
            .standardizedFileURL
    }

    private nonisolated static func appSupportModelsDirectory() -> URL {
        appSupportRuntimeDirectory().appendingPathComponent("data/models", isDirectory: true).standardizedFileURL
    }

    private nonisolated static func resolveModelsDirectory(
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

    private nonisolated static func canCreateAndWriteDirectory(_ directory: URL) -> Bool {
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

    private nonisolated static func legacyRuntimeDirectories(
        excluding target: URL,
        preferredRuntimeDirectory: URL?
    ) -> [URL] {
        let fm = FileManager.default
        let oldDocumentsRuntime = fm.homeDirectoryForCurrentUser
            .appendingPathComponent("Documents/PLOS/LocalAI/runtime", isDirectory: true)
            .standardizedFileURL
        let normalizedTarget = target.standardizedFileURL.path
        var candidates: [URL] = [
            oldDocumentsRuntime,
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

    private nonisolated static func legacyModelDirectories(
        sidecarDirectory: URL,
        runtimeDirectory: URL,
        dataDirectory: URL,
        excluding target: URL,
        preferredModelsDirectory: URL?
    ) -> [URL] {
        let oldDocumentsModels = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Documents/PLOS/LocalAI/models", isDirectory: true)
            .standardizedFileURL
        var candidates: [URL] = [
            oldDocumentsModels,
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

    private nonisolated static func migrateRuntimeVenvIfNeeded(targetRuntimeDirectory: URL, legacyRuntimeDirectories: [URL]) {
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

    private nonisolated static func migrateModelsIfNeeded(targetModelsDirectory: URL, legacyModelRoots: [URL]) {
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
                do {
                    try mergeDirectoryContents(source: sourceEngineDir, destination: targetEngineDir)
                } catch {
                    continue
                }
            }
        }
    }

    private nonisolated static func directoryHasItems(_ directory: URL) -> Bool {
        let fm = FileManager.default
        guard let items = try? fm.contentsOfDirectory(atPath: directory.path) else {
            return false
        }
        return !items.isEmpty
    }

    private nonisolated static func mergeDirectoryContents(source: URL, destination: URL) throws {
        let fm = FileManager.default
        if !fm.fileExists(atPath: source.path) {
            return
        }
        if !fm.fileExists(atPath: destination.path) {
            try moveOrCopyDirectory(source: source, destination: destination)
            return
        }

        let entries = try fm.contentsOfDirectory(
            at: source,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        )
        for entry in entries {
            let values = try entry.resourceValues(forKeys: [.isDirectoryKey])
            let isDirectory = values.isDirectory == true
            let target = destination.appendingPathComponent(entry.lastPathComponent, isDirectory: isDirectory)
            if isDirectory {
                if !fm.fileExists(atPath: target.path) {
                    try fm.createDirectory(at: target, withIntermediateDirectories: true)
                }
                try mergeDirectoryContents(source: entry, destination: target)
                continue
            }
            if fm.fileExists(atPath: target.path) {
                continue
            }
            if !fm.fileExists(atPath: destination.path) {
                try fm.createDirectory(at: destination, withIntermediateDirectories: true)
            }
            try fm.copyItem(at: entry, to: target)
        }
    }

    private nonisolated static func moveOrCopyDirectory(source: URL, destination: URL) throws {
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
