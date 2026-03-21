import AppKit
import Combine
import CryptoKit
import Foundation

@MainActor
extension AppViewModel {
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
    }


    func persistLocalModelPreferenceSnapshot() {
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
