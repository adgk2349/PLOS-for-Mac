import AppKit
import Foundation
import SwiftUI
import UniformTypeIdentifiers

@MainActor
extension AppViewModel {
    func refreshExtensionState() async throws {
        let snapshot = try await performWithSidecarRetry { client in
            async let capabilities = extensionServiceAdapter.fetchCapabilities(client: client)
            async let registry = extensionServiceAdapter.fetchPluginRegistry(client: client)
            return try await (capabilities, registry)
        }
        extensionCapabilities = snapshot.0.capabilities.sorted { $0.capability.rawValue < $1.capability.rawValue }
        pluginEntries = sortPluginEntries(snapshot.1.entries)
    }

    func refreshExtensionsNow() async {
        isPluginBusy = true
        defer { isPluginBusy = false }
        do {
            try await refreshExtensionState()
        } catch {
            if isEndpointNotFound(error) {
                extensionCapabilities = []
                pluginEntries = []
                return
            }
            handleViewModelError(error)
        }
    }

    func registerPluginFromDraft() async {
        let pluginID = pluginDraftID.trimmingCharacters(in: .whitespacesAndNewlines)
        let version = pluginDraftVersion.trimmingCharacters(in: .whitespacesAndNewlines)
        let entrypoint = pluginDraftEntrypoint.trimmingCharacters(in: .whitespacesAndNewlines)
        let signature = pluginDraftSignature.trimmingCharacters(in: .whitespacesAndNewlines)
        let capabilities = Array(pluginDraftCapabilities).sorted { $0.rawValue < $1.rawValue }
        let permissions = parsePluginPermissions(pluginDraftPermissions)

        guard !pluginID.isEmpty else {
            lastError = "plugin_id를 입력해 주세요."
            return
        }
        guard !version.isEmpty else {
            lastError = "버전을 입력해 주세요."
            return
        }
        guard !entrypoint.isEmpty else {
            lastError = "entrypoint를 입력해 주세요."
            return
        }
        guard !capabilities.isEmpty else {
            lastError = "최소 1개 capability를 선택해 주세요."
            return
        }

        isPluginBusy = true
        defer { isPluginBusy = false }

        do {
            let request = PluginRegisterRequest(
                manifest: PluginManifestV1(
                    plugin_id: pluginID,
                    version: version,
                    api_version: "v1",
                    capabilities: capabilities,
                    privacy_mode: pluginDraftPrivacyMode,
                    permissions: permissions,
                    entrypoint: entrypoint,
                    signature: signature.isEmpty ? nil : signature,
                    build_target: pluginDraftBuildTarget
                ),
                enabled: pluginDraftEnabled
            )

            let snapshot = try await performWithSidecarRetry { client in
                _ = try await extensionServiceAdapter.registerPlugin(client: client, request: request)
                async let capabilitiesResponse = extensionServiceAdapter.fetchCapabilities(client: client)
                async let registryResponse = extensionServiceAdapter.fetchPluginRegistry(client: client)
                return try await (capabilitiesResponse, registryResponse)
            }
            extensionCapabilities = snapshot.0.capabilities.sorted { $0.capability.rawValue < $1.capability.rawValue }
            pluginEntries = sortPluginEntries(snapshot.1.entries)
            resetPluginDraft()
        } catch {
            handleViewModelError(error)
        }
    }

    func registerPluginFromManifestFile() async {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        if #available(macOS 12.0, *) {
            panel.allowedContentTypes = [.json]
        } else {
            panel.allowedFileTypes = ["json"]
        }
        panel.prompt = "추가"

        guard panel.runModal() == .OK, let url = panel.url else {
            return
        }

        await registerPluginFromManifestURL(url)
    }

    func registerPluginFromDroppedItemProviders(_ providers: [NSItemProvider]) async {
        guard !providers.isEmpty else {
            return
        }
        guard let url = await resolveDroppedManifestURL(from: providers) else {
            lastError = "드롭한 파일에서 플러그인 manifest(.json)를 찾지 못했습니다."
            return
        }
        await registerPluginFromManifestURL(url)
    }

    private func registerPluginFromManifestURL(_ url: URL) async {
        let accessed = url.startAccessingSecurityScopedResource()
        defer {
            if accessed {
                url.stopAccessingSecurityScopedResource()
            }
        }

        isPluginBusy = true
        defer { isPluginBusy = false }

        do {
            let data = try Data(contentsOf: url)
            let request = try decodePluginRegisterRequest(from: data)
            let snapshot = try await performWithSidecarRetry { client in
                _ = try await extensionServiceAdapter.registerPlugin(client: client, request: request)
                async let capabilitiesResponse = extensionServiceAdapter.fetchCapabilities(client: client)
                async let registryResponse = extensionServiceAdapter.fetchPluginRegistry(client: client)
                return try await (capabilitiesResponse, registryResponse)
            }
            extensionCapabilities = snapshot.0.capabilities.sorted { $0.capability.rawValue < $1.capability.rawValue }
            pluginEntries = sortPluginEntries(snapshot.1.entries)
        } catch {
            handleViewModelError(error)
        }
    }

    private func resolveDroppedManifestURL(from providers: [NSItemProvider]) async -> URL? {
        for provider in providers where provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
            guard let url = await loadURL(from: provider, typeIdentifier: UTType.fileURL.identifier) else {
                continue
            }
            if url.pathExtension.lowercased() == "json" {
                return url
            }
        }
        return nil
    }

    private func loadURL(from provider: NSItemProvider, typeIdentifier: String) async -> URL? {
        await withCheckedContinuation { continuation in
            provider.loadItem(forTypeIdentifier: typeIdentifier, options: nil) { item, _ in
                if let url = item as? URL {
                    continuation.resume(returning: url)
                    return
                }
                if let nsURL = item as? NSURL, let url = nsURL as URL? {
                    continuation.resume(returning: url)
                    return
                }
                if let data = item as? Data, let url = URL(dataRepresentation: data, relativeTo: nil) {
                    continuation.resume(returning: url)
                    return
                }
                if let text = item as? String, let url = URL(string: text) {
                    continuation.resume(returning: url)
                    return
                }
                continuation.resume(returning: nil)
            }
        }
    }

    func enablePlugin(_ pluginID: String) async {
        isPluginBusy = true
        defer { isPluginBusy = false }

        do {
            let response = try await performWithSidecarRetry { client in
                try await extensionServiceAdapter.enablePlugin(client: client, pluginID: pluginID)
            }
            upsertPluginEntry(response.plugin)
            extensionCapabilities = response.capabilities.sorted { $0.capability.rawValue < $1.capability.rawValue }
        } catch {
            handleViewModelError(error)
        }
    }

    func disablePlugin(_ pluginID: String) async {
        isPluginBusy = true
        defer { isPluginBusy = false }

        do {
            let response = try await performWithSidecarRetry { client in
                try await extensionServiceAdapter.disablePlugin(client: client, pluginID: pluginID)
            }
            upsertPluginEntry(response.plugin)
            extensionCapabilities = response.capabilities.sorted { $0.capability.rawValue < $1.capability.rawValue }
        } catch {
            handleViewModelError(error)
        }
    }

    func deletePlugin(_ pluginID: String) async {
        isPluginBusy = true
        defer { isPluginBusy = false }

        do {
            _ = try await performWithSidecarRetry { client in
                try await extensionServiceAdapter.deletePlugin(client: client, pluginID: pluginID)
            }
            pluginEntries.removeAll { $0.plugin_id == pluginID }
            do {
                try await refreshExtensionState()
            } catch {
                if !isEndpointNotFound(error) {
                    throw error
                }
                extensionCapabilities = []
            }
        } catch {
            handleViewModelError(error)
        }
    }

    func pluginDraftCapabilityBinding(_ capability: ExtensionCapability) -> Binding<Bool> {
        Binding(
            get: { self.pluginDraftCapabilities.contains(capability) },
            set: { isSelected in
                if isSelected {
                    self.pluginDraftCapabilities.insert(capability)
                } else {
                    self.pluginDraftCapabilities.remove(capability)
                }
            }
        )
    }

    func resetPluginDraft() {
        pluginDraftID = ""
        pluginDraftVersion = "0.1.0"
        pluginDraftEntrypoint = ""
        pluginDraftPermissions = ""
        pluginDraftSignature = ""
        pluginDraftBuildTarget = .community
        pluginDraftPrivacyMode = .localOnly
        pluginDraftEnabled = false
        pluginDraftCapabilities = Set(ExtensionCapability.allCases)
    }

    private func sortPluginEntries(_ entries: [PluginRegistryEntry]) -> [PluginRegistryEntry] {
        entries.sorted { lhs, rhs in
            let lhsBuiltIn = isBuiltInPluginEntry(lhs)
            let rhsBuiltIn = isBuiltInPluginEntry(rhs)
            if lhsBuiltIn != rhsBuiltIn {
                return lhsBuiltIn && !rhsBuiltIn
            }
            if lhs.enabled != rhs.enabled {
                return lhs.enabled && !rhs.enabled
            }
            if lhs.updated_at != rhs.updated_at {
                return lhs.updated_at > rhs.updated_at
            }
            return lhs.plugin_id < rhs.plugin_id
        }
    }

    private func upsertPluginEntry(_ entry: PluginRegistryEntry) {
        if let index = pluginEntries.firstIndex(where: { $0.plugin_id == entry.plugin_id }) {
            pluginEntries[index] = entry
        } else {
            pluginEntries.append(entry)
        }
        pluginEntries = sortPluginEntries(pluginEntries)
    }

    func isBuiltInPluginEntry(_ entry: PluginRegistryEntry) -> Bool {
        if entry.is_builtin == true {
            return true
        }
        return entry.plugin_id == "builtin.core" || entry.state == "built_in"
    }

    private func parsePluginPermissions(_ text: String) -> [String] {
        text
            .split(separator: ",")
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
    }

    private func decodePluginRegisterRequest(from data: Data) throws -> PluginRegisterRequest {
        let decoder = JSONDecoder()
        if let wrapped = try? decoder.decode(PluginRegisterRequest.self, from: data) {
            return wrapped
        }
        if var object = (try? JSONSerialization.jsonObject(with: data)) as? [String: Any] {
            if var nested = object["manifest"] as? [String: Any] {
                if nested["privacy_mode"] == nil {
                    nested["privacy_mode"] = PluginPrivacyMode.localOnly.rawValue
                }
                object["manifest"] = nested
                let patched = try JSONSerialization.data(withJSONObject: object, options: [])
                if let wrapped = try? decoder.decode(PluginRegisterRequest.self, from: patched) {
                    return wrapped
                }
            } else if object["plugin_id"] != nil {
                if object["privacy_mode"] == nil {
                    object["privacy_mode"] = PluginPrivacyMode.localOnly.rawValue
                }
                let patched = try JSONSerialization.data(withJSONObject: object, options: [])
                if let manifest = try? decoder.decode(PluginManifestV1.self, from: patched) {
                    return PluginRegisterRequest(manifest: manifest, enabled: false)
                }
            }
        }
        let manifest = try decoder.decode(PluginManifestV1.self, from: data)
        return PluginRegisterRequest(manifest: manifest, enabled: false)
    }
}
