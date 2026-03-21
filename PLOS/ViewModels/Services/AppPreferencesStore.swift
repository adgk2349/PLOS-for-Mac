import Foundation

struct LocalModelPreferenceSnapshot {
    let preset: QuickInferencePreset?
    let engine: LocalEngine?
    let mlxModelPath: String?
    let llamaModelPath: String?
}

struct APISecretSnapshot {
    let openAI: String
    let anthropic: String
}

final class AppPreferencesStore {
    private let defaults: UserDefaults

    init(defaults: UserDefaults = .standard) {
        self.defaults = defaults
    }

    func loadApprovedActionKinds(key: String) -> Set<String> {
        Set(defaults.stringArray(forKey: key) ?? [])
    }

    func persistApprovedActionKinds(_ kinds: Set<String>, key: String) {
        defaults.set(Array(kinds).sorted(), forKey: key)
    }

    func loadChatResponseRoute(key: String, fallback: ChatResponseRoute) -> ChatResponseRoute {
        let raw = defaults.string(forKey: key) ?? fallback.rawValue
        return ChatResponseRoute(rawValue: raw) ?? fallback
    }

    func persistChatResponseRoute(_ route: ChatResponseRoute, key: String) {
        defaults.set(route.rawValue, forKey: key)
    }

    func loadLocalModelSnapshot(
        presetKey: String,
        engineKey: String,
        mlxPathKey: String,
        llamaPathKey: String
    ) -> LocalModelPreferenceSnapshot {
        let preset = defaults.string(forKey: presetKey).flatMap(QuickInferencePreset.init(rawValue:))
        let engine = defaults.string(forKey: engineKey).flatMap(LocalEngine.init(rawValue:))
        let mlxModelPath = defaults.string(forKey: mlxPathKey)
        let llamaModelPath = defaults.string(forKey: llamaPathKey)
        return LocalModelPreferenceSnapshot(
            preset: preset,
            engine: engine,
            mlxModelPath: mlxModelPath,
            llamaModelPath: llamaModelPath
        )
    }

    func persistLocalModelSnapshot(
        preset: QuickInferencePreset,
        engine: LocalEngine,
        mlxModelPath: String,
        llamaModelPath: String,
        presetKey: String,
        engineKey: String,
        mlxPathKey: String,
        llamaPathKey: String
    ) {
        defaults.set(preset.rawValue, forKey: presetKey)
        defaults.set(engine.rawValue, forKey: engineKey)
        defaults.set(mlxModelPath, forKey: mlxPathKey)
        defaults.set(llamaModelPath, forKey: llamaPathKey)
    }

    func loadSecrets() -> APISecretSnapshot {
        APISecretSnapshot(
            openAI: AppSecretStore.read("openai_api_key") ?? "",
            anthropic: AppSecretStore.read("anthropic_api_key") ?? ""
        )
    }

    func persistSecrets(openAI: String, anthropic: String) -> Bool {
        var changed = false

        let newOpenAI = openAI.trimmingCharacters(in: .whitespacesAndNewlines)
        let oldOpenAI = AppSecretStore.read("openai_api_key") ?? ""
        if newOpenAI != oldOpenAI {
            changed = true
            if newOpenAI.isEmpty {
                AppSecretStore.delete("openai_api_key")
            } else {
                _ = AppSecretStore.save(newOpenAI, for: "openai_api_key")
            }
        }

        let newAnthropic = anthropic.trimmingCharacters(in: .whitespacesAndNewlines)
        let oldAnthropic = AppSecretStore.read("anthropic_api_key") ?? ""
        if newAnthropic != oldAnthropic {
            changed = true
            if newAnthropic.isEmpty {
                AppSecretStore.delete("anthropic_api_key")
            } else {
                _ = AppSecretStore.save(newAnthropic, for: "anthropic_api_key")
            }
        }

        return changed
    }
}
