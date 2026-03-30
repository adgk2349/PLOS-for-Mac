import AppKit
import Combine
import CryptoKit
import Foundation

@MainActor
extension AppViewModel {
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
            persistAppLanguagePreference()
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
                        "hybrid_web_search_enabled": .bool(hybridWebSearchEnabled),
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
            lastSettingsSavedAt = Date()
        } catch {
            handleViewModelError(error)
        }
    }


    func refreshRemoteState() async throws {
        let coreSnapshot = try await performWithSidecarRetry { client in
            try await workspaceSyncService.fetchCoreSnapshot(client: client)
        }
        let settings = coreSnapshot.settings

        privacyMode = settings.privacy_mode
        appLanguage = L10n.selectionFromSettings(settings.language)
        persistAppLanguagePreference()
        hybridWebSearchEnabled = settings.hybrid_web_search_enabled
        systemFilePermission = settings.system_file_permission
        startupProfile = settings.startup_profile
        syncQuickInferencePresetFromProfile()
        localEngine = settings.local_engine ?? .mlx
        mlxModelPath = settings.mlx_model_path ?? ""
        llamaModelPath = settings.llama_model_path ?? ""
        sanitizeLocalModelSelection()
        persistLocalModelPreferenceSnapshot()
        actionPermissionMode = settings.action_permission_mode ?? .askPerAction
        adaptivePersonalizationEnabled = settings.adaptive_personalization_enabled
        sessionMemoryEnabled = settings.session_memory_enabled
        workspaceMemoryEnabled = settings.workspace_memory_enabled
        localMemoryOnly = settings.local_memory_only
        workspaceMemoryMode = settings.workspace_memory_mode
        searxngURL = settings.searxng_url ?? "http://localhost:8080"
        autoStartSearXNG = settings.auto_start_searxng
        persistSearXNGPreference()

        statusSnapshot = coreSnapshot.status
        failureItems = coreSnapshot.failures

        do {
            availableModels = try await performWithSidecarRetry { client in
                try await modelRuntimeService.fetchInstalledModels(client: client)
            }
        } catch {
            if !isEndpointNotFound(error) {
                throw error
            }
            availableModels = []
        }

        do {
            let catalog = try await performWithSidecarRetry { client in
                try await modelRuntimeService.fetchCatalog(client: client)
            }
            catalogDefaultProfile = catalog.default_profile
            catalogModels = catalog.models
        } catch {
            if !isEndpointNotFound(error) {
                throw error
            }
            catalogModels = []
        }

        do {
            try await refreshExtensionState()
        } catch {
            if !isEndpointNotFound(error) {
                throw error
            }
            extensionCapabilities = []
            pluginEntries = []
        }

        do {
            let docs = try await performWithSidecarRetry { client in
                try await workspaceSyncService.fetchDocuments(
                    client: client,
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
                try await workspaceSyncService.fetchDocuments(
                    client: client,
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


    func syncWorkspaceAndSettings() async throws {
        persistBookmarks()

        _ = try await performWithSidecarRetry { client in
            try await workspaceSyncService.syncWorkspaceAndSettings(
                client: client,
                workspace: WorkspaceUpdateRequest(
                    included_paths: includedFolderURLs.map(\.path),
                    excluded_paths: excludedPaths,
                    startup_profile: startupProfile,
                    default_mode: defaultWorkMode
                ),
                settings: SettingsModel(
                    privacy_mode: privacyMode,
                    hybrid_web_search_enabled: hybridWebSearchEnabled,
                    system_file_permission: systemFilePermission,
                    startup_profile: startupProfile,
                    model_profile: profileKey(from: startupProfile),
                    local_engine: localEngine,
                    mlx_model_path: mlxModelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : mlxModelPath,
                    llama_model_path: llamaModelPath.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : llamaModelPath,
                    reindex_policy: "filewatch_incremental",
                    language: L10n.sidecarLanguageCode(for: appLanguage),
                    action_permission_mode: actionPermissionMode,
                    adaptive_personalization_enabled: adaptivePersonalizationEnabled,
                    session_memory_enabled: sessionMemoryEnabled,
                    workspace_memory_enabled: workspaceMemoryEnabled,
                    local_memory_only: localMemoryOnly,
                    workspace_memory_mode: workspaceMemoryMode,
                    searxng_url: searxngURL,
                    auto_start_searxng: autoStartSearXNG
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


    func runIndexing(scope: String) async throws {
        let start = try await performWithSidecarRetry { client in
            try await workspaceSyncService.startIndexJob(client: client, scope: scope)
        }

        while true {
            let status = try await performWithSidecarRetry { client in
                try await workspaceSyncService.getIndexJob(client: client, jobID: start.job_id)
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


    func ensureSidecarClient() async throws -> SidecarAPIClient {
        if let client = sidecar.apiClient {
            do {
                try await client.health()
                _ = try await client.getSettings()
                syncStorageDirectoryResolutionFromSidecar()
                return client
            } catch {
                sidecar.stop()
            }
        }
        try await sidecar.start()
        syncStorageDirectoryResolutionFromSidecar()
        guard let client = sidecar.apiClient else {
            throw APIError(message: "Sidecar client unavailable: sidecar 시작 후에도 API 클라이언트가 생성되지 않았습니다.")
        }
        return client
    }


    func stageLabel(_ stage: String) -> String {
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


    func currentChatFilters() -> ChatFilters? {
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


    func prepareSelectedRuntime(using client: SidecarAPIClient) async throws -> RuntimePrepareResponse? {
        let modelPath = modelRuntimeService.resolveModelPath(
            engine: localEngine,
            mlxModelPath: mlxModelPath,
            llamaModelPath: llamaModelPath,
            modelsRootPath: effectiveModelsStorageDirectoryPath.isEmpty ? modelsStorageDirectoryPath : effectiveModelsStorageDirectoryPath
        )

        do {
            let runtime = try await modelRuntimeService.prepareRuntime(
                client: client,
                engine: localEngine,
                modelPath: modelPath
            )

            var shouldPersist = false
            if runtime.engine != localEngine {
                localEngine = runtime.engine
                shouldPersist = true
            }
            if let resolvedPath = runtime.model_path, !resolvedPath.isEmpty {
                switch runtime.engine {
                case .mlx:
                    if mlxModelPath != resolvedPath {
                        mlxModelPath = resolvedPath
                        shouldPersist = true
                    }
                case .llamaCPP:
                    if llamaModelPath != resolvedPath {
                        llamaModelPath = resolvedPath
                        shouldPersist = true
                    }
                }
            }
            if shouldPersist {
                persistLocalModelPreferenceSnapshot()
            }

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


    func handleViewModelError(_ error: Error) {
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


    func recoverSessionFromInvalidToken() async {
        defer { isRecoveringSession = false }
        do {
            sidecar.stop()
            try await sidecar.start()
            syncStorageDirectoryResolutionFromSidecar()
            try await refreshRemoteState()
            lastError = nil
        } catch {
            lastError = "세션 재연결 실패: \(error.localizedDescription)"
        }
    }


    func isEndpointNotFound(_ error: Error) -> Bool {
        guard let apiError = error as? APIError else {
            return false
        }
        return apiError.message.contains("HTTP 404")
    }


    func isInvalidSessionTokenError(_ error: Error) -> Bool {
        guard let apiError = error as? APIError else {
            return false
        }
        let lower = apiError.message.lowercased()
        return lower.contains("http 401") && lower.contains("invalid session token")
    }


    func performWithSidecarRetry<T>(_ operation: (SidecarAPIClient) async throws -> T) async throws -> T {
        let client = try await ensureSidecarClient()
        do {
            let value = try await operation(client)
            clearResolvedErrorIfNeeded()
            return value
        } catch {
            guard isInvalidSessionTokenError(error) else {
                throw error
            }
            sidecar.stop()
            try await sidecar.start()
            syncStorageDirectoryResolutionFromSidecar()
            guard let refreshed = sidecar.apiClient else {
                throw APIError(message: "세션 토큰을 갱신했지만 sidecar client를 다시 가져오지 못했습니다.")
            }
            let value = try await operation(refreshed)
            clearResolvedErrorIfNeeded()
            return value
        }
    }
}
