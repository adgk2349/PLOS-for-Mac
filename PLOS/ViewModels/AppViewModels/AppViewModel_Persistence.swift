import AppKit
import Combine
import CryptoKit
import Foundation

@MainActor
extension AppViewModel {
    private var defaultModelsStorageDirectoryURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Documents/PLOS/LocalAI/models", isDirectory: true)
            .standardizedFileURL
    }

    private var defaultRuntimeStorageDirectoryURL: URL {
        FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent("Documents/PLOS/LocalAI/runtime", isDirectory: true)
            .standardizedFileURL
    }

    private func normalizedStoragePath(_ raw: String, fallback: URL) -> String {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else {
            return fallback.path
        }
        return URL(fileURLWithPath: trimmed).standardizedFileURL.path
    }

    private func storageURL(from raw: String) -> URL? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        return URL(fileURLWithPath: trimmed).standardizedFileURL
    }

    func loadStorageDirectoryPreferences() {
        let defaults = UserDefaults.standard
        let savedModels = defaults.string(forKey: UDKey.modelsStorageDir)
        let savedRuntime = defaults.string(forKey: UDKey.runtimeStorageDir)
        modelsStorageDirectoryPath = normalizedStoragePath(savedModels ?? "", fallback: defaultModelsStorageDirectoryURL)
        runtimeStorageDirectoryPath = normalizedStoragePath(savedRuntime ?? "", fallback: defaultRuntimeStorageDirectoryURL)
        sidecar.configureStorageDirectories(
            modelsDirectory: storageURL(from: modelsStorageDirectoryPath),
            runtimeDirectory: storageURL(from: runtimeStorageDirectoryPath)
        )
    }

    func persistStorageDirectoryPreferences() {
        let defaults = UserDefaults.standard
        let normalizedModels = normalizedStoragePath(modelsStorageDirectoryPath, fallback: defaultModelsStorageDirectoryURL)
        let normalizedRuntime = normalizedStoragePath(runtimeStorageDirectoryPath, fallback: defaultRuntimeStorageDirectoryURL)
        modelsStorageDirectoryPath = normalizedModels
        runtimeStorageDirectoryPath = normalizedRuntime
        defaults.set(normalizedModels, forKey: UDKey.modelsStorageDir)
        defaults.set(normalizedRuntime, forKey: UDKey.runtimeStorageDir)
    }

    func syncStorageDirectoryResolutionFromSidecar() {
        guard let resolution = sidecar.storageResolution else {
            if effectiveModelsStorageDirectoryPath.isEmpty {
                effectiveModelsStorageDirectoryPath = normalizedStoragePath(modelsStorageDirectoryPath, fallback: defaultModelsStorageDirectoryURL)
            }
            if effectiveRuntimeStorageDirectoryPath.isEmpty {
                effectiveRuntimeStorageDirectoryPath = normalizedStoragePath(runtimeStorageDirectoryPath, fallback: defaultRuntimeStorageDirectoryURL)
            }
            return
        }
        modelsStorageDirectoryPath = resolution.requestedModelsDirectory.path
        runtimeStorageDirectoryPath = resolution.requestedRuntimeDirectory.path
        effectiveModelsStorageDirectoryPath = resolution.effectiveModelsDirectory.path
        effectiveRuntimeStorageDirectoryPath = resolution.effectiveRuntimeDirectory.path
        var warningParts: [String] = []
        if let modelReason = resolution.modelFallbackReason, !modelReason.isEmpty {
            warningParts.append("모델 저장 경로 폴백: \(modelReason)")
        }
        if let runtimeReason = resolution.runtimeFallbackReason, !runtimeReason.isEmpty {
            warningParts.append("런타임 경로 폴백: \(runtimeReason)")
        }
        storageDirectoryWarning = warningParts.joined(separator: " | ")
    }

    func applyStorageDirectoryPreferences(restartSidecar: Bool = true) async {
        modelsStorageDirectoryPath = normalizedStoragePath(modelsStorageDirectoryPath, fallback: defaultModelsStorageDirectoryURL)
        runtimeStorageDirectoryPath = normalizedStoragePath(runtimeStorageDirectoryPath, fallback: defaultRuntimeStorageDirectoryURL)
        persistStorageDirectoryPreferences()
        sidecar.configureStorageDirectories(
            modelsDirectory: storageURL(from: modelsStorageDirectoryPath),
            runtimeDirectory: storageURL(from: runtimeStorageDirectoryPath)
        )
        guard restartSidecar else {
            return
        }
        do {
            isSidecarReadyForChat = false
            await sidecar.stop()
            try await sidecar.start()
            isSidecarReadyForChat = sidecar.apiClient != nil
            syncStorageDirectoryResolutionFromSidecar()
            try await refreshRemoteState()
            lastSettingsSavedAt = Date()
            lastError = nil
        } catch {
            isSidecarReadyForChat = false
            handleViewModelError(error)
        }
    }

    func resetModelsStorageDirectoryToDefault() async {
        modelsStorageDirectoryPath = defaultModelsStorageDirectoryURL.path
        await applyStorageDirectoryPreferences()
    }

    func resetRuntimeStorageDirectoryToDefault() async {
        runtimeStorageDirectoryPath = defaultRuntimeStorageDirectoryURL.path
        await applyStorageDirectoryPreferences()
    }

    func chooseModelsStorageDirectory() async {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "선택"
        if panel.runModal() == .OK, let url = panel.url {
            modelsStorageDirectoryPath = url.standardizedFileURL.path
            await applyStorageDirectoryPreferences()
        }
    }

    func chooseRuntimeStorageDirectory() async {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "선택"
        if panel.runModal() == .OK, let url = panel.url {
            runtimeStorageDirectoryPath = url.standardizedFileURL.path
            await applyStorageDirectoryPreferences()
        }
    }

    func loadAppLanguagePreference() {
        appLanguage = L10n.loadSelection()
    }

    func persistAppLanguagePreference() {
        L10n.saveSelection(appLanguage)
    }

    func loadSearXNGPreference() {
        searxngURL = UserDefaults.standard.string(forKey: UDKey.searxngURL) ?? "http://localhost:8080"
        autoStartSearXNG = UserDefaults.standard.bool(forKey: UDKey.autoStartSearXNG)
    }

    func persistSearXNGPreference() {
        UserDefaults.standard.set(searxngURL, forKey: UDKey.searxngURL)
        UserDefaults.standard.set(autoStartSearXNG, forKey: UDKey.autoStartSearXNG)
    }

    func loadSidecarVisionPreferences() {
        let defaults = UserDefaults.standard
        if defaults.object(forKey: UDKey.sidecarVisionEnabled) != nil {
            sidecarVisionEnabled = defaults.bool(forKey: UDKey.sidecarVisionEnabled)
        } else {
            sidecarVisionEnabled = true
        }
        sidecarVisionCaptionModel = defaults.string(forKey: UDKey.sidecarVisionCaptionModel) ?? "microsoft/git-base-coco"
        sidecarVisionClassifyModel = defaults.string(forKey: UDKey.sidecarVisionClassifyModel) ?? "google/vit-base-patch16-224"
        if defaults.object(forKey: UDKey.sidecarMlxKVQEnabled) != nil {
            sidecarMlxKVQEnabled = defaults.bool(forKey: UDKey.sidecarMlxKVQEnabled)
        } else {
            sidecarMlxKVQEnabled = false
        }
        if let rawMode = defaults.string(forKey: UDKey.sidecarMlxKVQMode),
           let parsed = SidecarMlxKVQMode(rawValue: rawMode)
        {
            sidecarMlxKVQMode = parsed
        } else {
            sidecarMlxKVQMode = .turbo3
        }
        if defaults.object(forKey: UDKey.sidecarMlxKVQBits) != nil {
            sidecarMlxKVQBits = min(8, max(2, defaults.integer(forKey: UDKey.sidecarMlxKVQBits)))
        } else {
            sidecarMlxKVQBits = 3
        }
        if defaults.object(forKey: UDKey.sidecarConversationTurboEnabled) != nil {
            sidecarConversationTurboEnabled = defaults.bool(forKey: UDKey.sidecarConversationTurboEnabled)
        } else {
            sidecarConversationTurboEnabled = false
        }
        if defaults.object(forKey: UDKey.sidecarInferenceTimeoutDisabled) != nil {
            sidecarInferenceTimeoutDisabled = defaults.bool(forKey: UDKey.sidecarInferenceTimeoutDisabled)
        } else {
            sidecarInferenceTimeoutDisabled = false
        }
        if defaults.object(forKey: UDKey.sidecarMainResponseTimeoutSeconds) != nil {
            sidecarMainResponseTimeoutSeconds = min(3600, max(30, defaults.integer(forKey: UDKey.sidecarMainResponseTimeoutSeconds)))
        } else {
            sidecarMainResponseTimeoutSeconds = 240
        }
        if defaults.object(forKey: UDKey.sidecarAuxiliaryTimeoutSeconds) != nil {
            sidecarAuxiliaryTimeoutSeconds = min(120, max(4, defaults.integer(forKey: UDKey.sidecarAuxiliaryTimeoutSeconds)))
        } else {
            sidecarAuxiliaryTimeoutSeconds = 12
        }
        if defaults.object(forKey: UDKey.showThinkingProcessInChat) != nil {
            showThinkingProcessInChat = defaults.bool(forKey: UDKey.showThinkingProcessInChat)
        } else {
            showThinkingProcessInChat = true
        }
        applySidecarVisionRuntimeConfiguration()
    }

    func persistSidecarVisionPreferences() {
        let defaults = UserDefaults.standard
        defaults.set(sidecarVisionEnabled, forKey: UDKey.sidecarVisionEnabled)
        defaults.set(sidecarVisionCaptionModel, forKey: UDKey.sidecarVisionCaptionModel)
        defaults.set(sidecarVisionClassifyModel, forKey: UDKey.sidecarVisionClassifyModel)
        defaults.set(sidecarMlxKVQEnabled, forKey: UDKey.sidecarMlxKVQEnabled)
        defaults.set(sidecarMlxKVQMode.rawValue, forKey: UDKey.sidecarMlxKVQMode)
        defaults.set(min(8, max(2, sidecarMlxKVQBits)), forKey: UDKey.sidecarMlxKVQBits)
        defaults.set(sidecarConversationTurboEnabled, forKey: UDKey.sidecarConversationTurboEnabled)
        defaults.set(sidecarInferenceTimeoutDisabled, forKey: UDKey.sidecarInferenceTimeoutDisabled)
        defaults.set(
            min(3600, max(30, sidecarMainResponseTimeoutSeconds)),
            forKey: UDKey.sidecarMainResponseTimeoutSeconds
        )
        defaults.set(
            min(120, max(4, sidecarAuxiliaryTimeoutSeconds)),
            forKey: UDKey.sidecarAuxiliaryTimeoutSeconds
        )
        defaults.set(showThinkingProcessInChat, forKey: UDKey.showThinkingProcessInChat)
    }

    func applySidecarVisionRuntimeConfiguration() {
        sidecar.configureVisionRuntime(
            enabled: sidecarVisionEnabled,
            captionModel: sidecarVisionCaptionModel,
            classifyModel: sidecarVisionClassifyModel
        )
        sidecar.configureMlxKVQRuntime(
            enabled: sidecarMlxKVQEnabled,
            mode: sidecarMlxKVQMode,
            bits: sidecarMlxKVQBits
        )
        sidecar.configureConversationTurboRuntime(enabled: sidecarConversationTurboEnabled)
        sidecar.configureInferenceTimeoutRuntime(
            disabled: sidecarInferenceTimeoutDisabled,
            mainResponseTimeoutSeconds: sidecarMainResponseTimeoutSeconds,
            auxiliaryTimeoutSeconds: sidecarAuxiliaryTimeoutSeconds
        )
    }

    func sidecarVisionSettingsChangedFromPersisted() -> Bool {
        let defaults = UserDefaults.standard
        let persistedEnabled: Bool
        if defaults.object(forKey: UDKey.sidecarVisionEnabled) != nil {
            persistedEnabled = defaults.bool(forKey: UDKey.sidecarVisionEnabled)
        } else {
            persistedEnabled = true
        }
        let persistedCaption = defaults.string(forKey: UDKey.sidecarVisionCaptionModel) ?? "microsoft/git-base-coco"
        let persistedClassify = defaults.string(forKey: UDKey.sidecarVisionClassifyModel) ?? "google/vit-base-patch16-224"
        let persistedKVQEnabled: Bool
        if defaults.object(forKey: UDKey.sidecarMlxKVQEnabled) != nil {
            persistedKVQEnabled = defaults.bool(forKey: UDKey.sidecarMlxKVQEnabled)
        } else {
            persistedKVQEnabled = false
        }
        let persistedKVQMode = SidecarMlxKVQMode(rawValue: defaults.string(forKey: UDKey.sidecarMlxKVQMode) ?? "") ?? .turbo3
        let persistedKVQBits: Int
        if defaults.object(forKey: UDKey.sidecarMlxKVQBits) != nil {
            persistedKVQBits = min(8, max(2, defaults.integer(forKey: UDKey.sidecarMlxKVQBits)))
        } else {
            persistedKVQBits = 3
        }
        let persistedConversationTurboEnabled: Bool
        if defaults.object(forKey: UDKey.sidecarConversationTurboEnabled) != nil {
            persistedConversationTurboEnabled = defaults.bool(forKey: UDKey.sidecarConversationTurboEnabled)
        } else {
            persistedConversationTurboEnabled = false
        }
        let persistedInferenceTimeoutDisabled: Bool
        if defaults.object(forKey: UDKey.sidecarInferenceTimeoutDisabled) != nil {
            persistedInferenceTimeoutDisabled = defaults.bool(forKey: UDKey.sidecarInferenceTimeoutDisabled)
        } else {
            persistedInferenceTimeoutDisabled = false
        }
        let persistedMainResponseTimeoutSeconds: Int
        if defaults.object(forKey: UDKey.sidecarMainResponseTimeoutSeconds) != nil {
            persistedMainResponseTimeoutSeconds = min(3600, max(30, defaults.integer(forKey: UDKey.sidecarMainResponseTimeoutSeconds)))
        } else {
            persistedMainResponseTimeoutSeconds = 240
        }
        let persistedAuxiliaryTimeoutSeconds: Int
        if defaults.object(forKey: UDKey.sidecarAuxiliaryTimeoutSeconds) != nil {
            persistedAuxiliaryTimeoutSeconds = min(120, max(4, defaults.integer(forKey: UDKey.sidecarAuxiliaryTimeoutSeconds)))
        } else {
            persistedAuxiliaryTimeoutSeconds = 12
        }
        return persistedEnabled != sidecarVisionEnabled ||
            persistedCaption != sidecarVisionCaptionModel ||
            persistedClassify != sidecarVisionClassifyModel ||
            persistedKVQEnabled != sidecarMlxKVQEnabled ||
            persistedKVQMode != sidecarMlxKVQMode ||
            persistedKVQBits != min(8, max(2, sidecarMlxKVQBits)) ||
            persistedConversationTurboEnabled != sidecarConversationTurboEnabled ||
            persistedInferenceTimeoutDisabled != sidecarInferenceTimeoutDisabled ||
            persistedMainResponseTimeoutSeconds != min(3600, max(30, sidecarMainResponseTimeoutSeconds)) ||
            persistedAuxiliaryTimeoutSeconds != min(120, max(4, sidecarAuxiliaryTimeoutSeconds))
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


    func parseTagText(_ text: String) -> [String] {
        text
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }


    func profileKey(from startup: StartupProfile) -> String {
        switch startup {
        case .fast:
            return "fast"
        case .recommended:
            return "balanced"
        case .deep:
            return "advanced"
        }
    }


    func loadApprovedSystemActionKinds() {
        approvedSystemActionKinds = appPreferencesStore.loadApprovedActionKinds(key: UDKey.approvedActions)
    }


    func loadChatResponseRoute() {
        chatResponseRoute = appPreferencesStore.loadChatResponseRoute(
            key: UDKey.chatResponseRoute,
            fallback: .hybrid
        )
        switch chatResponseRoute {
        case .localOnly:
            privacyMode = .localOnly
        case .hybrid, .apiOnly:
            if privacyMode == .localOnly {
                privacyMode = .hybrid
            }
        }
    }

    func loadRoleplayPreference() {
        let defaults = UserDefaults.standard
        if defaults.object(forKey: UDKey.roleplayModeEnabled) != nil {
            roleplayModeEnabled = defaults.bool(forKey: UDKey.roleplayModeEnabled)
        } else {
            roleplayModeEnabled = false
        }
    }

    func persistRoleplayPreference() {
        UserDefaults.standard.set(roleplayModeEnabled, forKey: UDKey.roleplayModeEnabled)
    }


    func loadLocalModelPreferenceSnapshot() {
        let snapshot = appPreferencesStore.loadLocalModelSnapshot(
            presetKey: UDKey.quickInferencePreset,
            engineKey: UDKey.localEngine,
            mlxPathKey: UDKey.mlxModelPath,
            llamaPathKey: UDKey.llamaModelPath
        )
        if let preset = snapshot.preset {
            quickInferencePreset = preset
            startupProfile = preset.startupProfile
        }
        if let engine = snapshot.engine {
            localEngine = engine
        }
        if let savedMLXPath = snapshot.mlxModelPath {
            mlxModelPath = savedMLXPath
        }
        if let savedLlamaPath = snapshot.llamaModelPath {
            llamaModelPath = savedLlamaPath
        }
        sanitizeLocalModelSelection()
    }


    func persistLocalModelPreferenceSnapshot() {
        sanitizeLocalModelSelection()
        appPreferencesStore.persistLocalModelSnapshot(
            preset: quickInferencePreset,
            engine: localEngine,
            mlxModelPath: mlxModelPath,
            llamaModelPath: llamaModelPath,
            presetKey: UDKey.quickInferencePreset,
            engineKey: UDKey.localEngine,
            mlxPathKey: UDKey.mlxModelPath,
            llamaPathKey: UDKey.llamaModelPath
        )
    }

    func sanitizeLocalModelSelection() {
        let mlx = mlxModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        let llama = llamaModelPath.trimmingCharacters(in: .whitespacesAndNewlines)

        switch localEngine {
        case .mlx:
            if !mlx.isEmpty {
                llamaModelPath = ""
            } else if !llama.isEmpty {
                localEngine = .llamaCPP
                mlxModelPath = ""
            }
        case .llamaCPP:
            if !llama.isEmpty {
                mlxModelPath = ""
            } else if !mlx.isEmpty {
                localEngine = .mlx
                llamaModelPath = ""
            }
        }
    }


    func loadSecretAPIKeys() {
        let snapshot = appPreferencesStore.loadSecrets()
        openAIAPIKey = snapshot.openAI
        anthropicAPIKey = snapshot.anthropic
    }


    @discardableResult
    func persistSecretAPIKeys() -> Bool {
        appPreferencesStore.persistSecrets(
            openAI: openAIAPIKey,
            anthropic: anthropicAPIKey
        )
    }


    func syncQuickInferencePresetFromProfile() {
        switch startupProfile {
        case .fast:
            quickInferencePreset = .fast
        case .recommended:
            quickInferencePreset = .quality
        case .deep:
            quickInferencePreset = .highQuality
        }
    }


    func normalizedSelectionPath(for model: ModelListItem) -> String {
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

    static func isDisplayableModelArtifact(_ model: ModelListItem) -> Bool {
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

    static func samePath(_ lhs: String, _ rhs: String) -> Bool {
        URL(fileURLWithPath: lhs).standardizedFileURL.path == URL(fileURLWithPath: rhs).standardizedFileURL.path
    }


    func persistApprovedSystemActionKinds() {
        appPreferencesStore.persistApprovedActionKinds(approvedSystemActionKinds, key: UDKey.approvedActions)
    }
}
