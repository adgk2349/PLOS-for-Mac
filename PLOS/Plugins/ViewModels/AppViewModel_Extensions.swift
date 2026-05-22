import AppKit
import Foundation
import SwiftUI
import UniformTypeIdentifiers

private struct PluginImportDescriptor {
    let manifestURL: URL
    let pluginDirectory: URL
    let shouldPrepareRuntime: Bool
}

private enum PluginImportBootstrapper {
    static func prepare(
        request: PluginRegisterRequest,
        pluginDirectory: URL,
        shouldPrepareRuntime: Bool
    ) throws -> PluginRegisterRequest {
        guard shouldPrepareRuntime else { return request }

        var patched = request
        let entrypoint = patched.manifest.entrypoint.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !entrypoint.isEmpty, !entrypoint.hasPrefix("builtin://") else { return patched }

        let fm = FileManager.default
        let requirementsURL = pluginDirectory.appendingPathComponent("requirements.txt")
        let venvDirectory = pluginDirectory.appendingPathComponent(".venv", isDirectory: true)
        let venvPythonURL = venvDirectory.appendingPathComponent("bin/python3")
        if fm.fileExists(atPath: requirementsURL.path) {
            try ensureVirtualEnvIfNeeded(
                pluginDirectory: pluginDirectory,
                requirementsURL: requirementsURL,
                venvDirectory: venvDirectory,
                venvPythonURL: venvPythonURL
            )
        }

        let launcherURL = pluginDirectory.appendingPathComponent(".plos_plugin_entrypoint.sh")
        let launcherScript = makeLauncherScript(
            pluginDirectory: pluginDirectory,
            entrypoint: entrypoint,
            preferredPython: fm.isExecutableFile(atPath: venvPythonURL.path) ? venvPythonURL.path : nil
        )
        try launcherScript.write(to: launcherURL, atomically: true, encoding: .utf8)
        try fm.setAttributes([.posixPermissions: 0o755], ofItemAtPath: launcherURL.path)

        patched.manifest.entrypoint = launcherURL.path
        return patched
    }

    private static func ensureVirtualEnvIfNeeded(
        pluginDirectory: URL,
        requirementsURL: URL,
        venvDirectory: URL,
        venvPythonURL: URL
    ) throws {
        let fm = FileManager.default
        let stampURL = pluginDirectory.appendingPathComponent(".plos_plugin_requirements.stamp")
        let requirementsMTime = (try? fm.attributesOfItem(atPath: requirementsURL.path)[.modificationDate] as? Date) ?? .distantPast
        let stampMTime = (try? fm.attributesOfItem(atPath: stampURL.path)[.modificationDate] as? Date) ?? .distantPast
        let needsInstall = !fm.isExecutableFile(atPath: venvPythonURL.path) || stampMTime < requirementsMTime
        guard needsInstall else { return }

        if !fm.isExecutableFile(atPath: venvPythonURL.path) {
            let pythonCandidates = SidecarBootstrapService.resolveSystemPythonExecutables()
            guard let systemPython = pythonCandidates.first else {
                throw APIError(message: "플러그인 Python 환경 생성을 위한 python3(3.11~3.13)를 찾지 못했습니다.")
            }
            try SidecarBootstrapService.runCommand(
                executable: systemPython,
                arguments: ["-m", "venv", venvDirectory.path],
                cwd: pluginDirectory,
                step: "플러그인 가상환경 생성"
            )
        }

        try SidecarBootstrapService.runCommand(
            executable: venvPythonURL.path,
            arguments: ["-m", "pip", "install", "--upgrade", "pip"],
            cwd: pluginDirectory,
            step: "플러그인 pip 업그레이드"
        )
        try SidecarBootstrapService.runCommand(
            executable: venvPythonURL.path,
            arguments: ["-m", "pip", "install", "-r", requirementsURL.path],
            cwd: pluginDirectory,
            step: "플러그인 의존성 설치"
        )
        try "ok".write(to: stampURL, atomically: true, encoding: .utf8)
    }

    private static func makeLauncherScript(
        pluginDirectory: URL,
        entrypoint: String,
        preferredPython: String?
    ) -> String {
        let directory = shellDoubleQuoted(pluginDirectory.path)
        if let (module, extraArgs) = parsePythonModuleEntrypoint(entrypoint) {
            let pythonExec = shellDoubleQuoted(preferredPython ?? "python3")
            let moduleArg = shellDoubleQuoted(module)
            let trailing = extraArgs.map { shellDoubleQuoted($0) }.joined(separator: " ")
            let suffix = trailing.isEmpty ? "" : " \(trailing)"
            return """
#!/usr/bin/env bash
set -euo pipefail
cd \(directory)
exec \(pythonExec) -m \(moduleArg)\(suffix) "$@"
"""
        }

        let forwarded = "\(entrypoint) \"$@\""
        return """
#!/usr/bin/env bash
set -euo pipefail
cd \(directory)
exec /bin/bash -lc \(shellSingleQuoted(forwarded)) -- "$@"
"""
    }

    private static func parsePythonModuleEntrypoint(_ entrypoint: String) -> (module: String, extraArgs: [String])? {
        let tokens = entrypoint
            .split(whereSeparator: \.isWhitespace)
            .map(String.init)
        guard tokens.count >= 3 else { return nil }
        guard tokens[0].lowercased().contains("python") else { return nil }
        guard tokens[1] == "-m" else { return nil }
        let module = tokens[2]
        guard module.range(of: #"^[A-Za-z_][A-Za-z0-9_\.]*$"#, options: .regularExpression) != nil else {
            return nil
        }
        return (module, Array(tokens.dropFirst(3)))
    }

    private static func shellSingleQuoted(_ value: String) -> String {
        "'" + value.replacingOccurrences(of: "'", with: "'\"'\"'") + "'"
    }

    private static func shellDoubleQuoted(_ value: String) -> String {
        "\"\(value.replacingOccurrences(of: "\\", with: "\\\\").replacingOccurrences(of: "\"", with: "\\\""))\""
    }
}

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
        reconcileSelectedPluginPanel()
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
                reconcileSelectedPluginPanel()
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
            reconcileSelectedPluginPanel()
            resetPluginDraft()
        } catch {
            handleViewModelError(error)
        }
    }

    func registerPluginFromManifestFile() async {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = false
        if #available(macOS 12.0, *) {
            panel.allowedContentTypes = [.json, .folder]
        } else {
            panel.allowedFileTypes = ["json"]
        }
        panel.prompt = "추가"

        guard panel.runModal() == .OK, let url = panel.url else {
            return
        }

        await registerPluginFromImportURL(url)
    }

    func registerPluginFromDroppedItemProviders(_ providers: [NSItemProvider]) async {
        guard !providers.isEmpty else {
            return
        }
        guard let url = await resolveDroppedPluginURL(from: providers) else {
            lastError = "드롭한 항목에서 플러그인 폴더 또는 manifest(.json)를 찾지 못했습니다."
            return
        }
        await registerPluginFromImportURL(url)
    }

    private func registerPluginFromImportURL(_ url: URL) async {
        let descriptor: PluginImportDescriptor
        do {
            descriptor = try resolvePluginImportDescriptor(url)
        } catch {
            handleViewModelError(error)
            return
        }

        let rootAccessed = url.startAccessingSecurityScopedResource()
        let manifestAccessed = descriptor.manifestURL.path == url.path ? false : descriptor.manifestURL.startAccessingSecurityScopedResource()
        defer {
            if rootAccessed {
                url.stopAccessingSecurityScopedResource()
            }
            if manifestAccessed {
                descriptor.manifestURL.stopAccessingSecurityScopedResource()
            }
        }

        isPluginBusy = true
        defer { isPluginBusy = false }

        do {
            let data = try Data(contentsOf: descriptor.manifestURL)
            let decoded = try decodePluginRegisterRequest(from: data)
            let request = try PluginImportBootstrapper.prepare(
                request: decoded,
                pluginDirectory: descriptor.pluginDirectory,
                shouldPrepareRuntime: descriptor.shouldPrepareRuntime
            )
            let snapshot = try await performWithSidecarRetry { client in
                _ = try await extensionServiceAdapter.registerPlugin(client: client, request: request)
                async let capabilitiesResponse = extensionServiceAdapter.fetchCapabilities(client: client)
                async let registryResponse = extensionServiceAdapter.fetchPluginRegistry(client: client)
                return try await (capabilitiesResponse, registryResponse)
            }
            extensionCapabilities = snapshot.0.capabilities.sorted { $0.capability.rawValue < $1.capability.rawValue }
            pluginEntries = sortPluginEntries(snapshot.1.entries)
            reconcileSelectedPluginPanel()
        } catch {
            handleViewModelError(error)
        }
    }

    private func resolveDroppedPluginURL(from providers: [NSItemProvider]) async -> URL? {
        let fm = FileManager.default
        for provider in providers where provider.hasItemConformingToTypeIdentifier(UTType.fileURL.identifier) {
            guard let url = await loadURL(from: provider, typeIdentifier: UTType.fileURL.identifier) else {
                continue
            }
            var isDirectory: ObjCBool = false
            if fm.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue {
                return url
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
            reconcileSelectedPluginPanel()
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
            reconcileSelectedPluginPanel()
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
            reconcileSelectedPluginPanel()
        } catch {
            handleViewModelError(error)
        }
    }

    struct ComposerPluginToggleItem: Identifiable, Hashable {
        let id: String
        let pluginID: String
        let toggleID: String
        let title: String
        let help: String?
        let defaultValue: Bool
        let pluginEnabled: Bool
    }

    var composerPluginToggles: [ComposerPluginToggleItem] {
        let language = appLanguage
        var items: [ComposerPluginToggleItem] = []

        for entry in pluginEntries {
            guard let ui = entry.manifest.ui else { continue }
            for toggle in ui.toggles where toggle.location == .composerModelControls {
                let storageKey = pluginToggleStorageKey(pluginID: entry.plugin_id, toggleID: toggle.id)
                items.append(
                    ComposerPluginToggleItem(
                        id: storageKey,
                        pluginID: entry.plugin_id,
                        toggleID: toggle.id,
                        title: toggle.title(language: language),
                        help: toggle.help(language: language),
                        defaultValue: toggle.default_enabled,
                        pluginEnabled: entry.enabled
                    )
                )
            }
        }

        return items.sorted { lhs, rhs in
            if lhs.pluginEnabled != rhs.pluginEnabled {
                return lhs.pluginEnabled && !rhs.pluginEnabled
            }
            if lhs.pluginID != rhs.pluginID {
                return lhs.pluginID < rhs.pluginID
            }
            return lhs.toggleID < rhs.toggleID
        }
    }

    func pluginToggleBinding(pluginID: String, toggleID: String, defaultValue: Bool) -> Binding<Bool> {
        let key = pluginToggleStorageKey(pluginID: pluginID, toggleID: toggleID)
        return Binding(
            get: {
                if let cached = self.pluginUIToggleStates[key] {
                    return cached
                }
                let defaults = UserDefaults.standard
                let value: Bool
                if defaults.object(forKey: key) == nil {
                    value = defaultValue
                } else {
                    value = defaults.bool(forKey: key)
                }
                self.pluginUIToggleStates[key] = value
                return value
            },
            set: { isOn in
                self.pluginUIToggleStates[key] = isOn
                UserDefaults.standard.set(isOn, forKey: key)
            }
        )
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

    private func pluginToggleStorageKey(pluginID: String, toggleID: String) -> String {
        let normalizedPluginID = pluginID.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedToggleID = toggleID.trimmingCharacters(in: .whitespacesAndNewlines)
        return "local_ai_plugin_ui_toggle_v1.\(normalizedPluginID).\(normalizedToggleID)"
    }

    private func reconcileSelectedPluginPanel() {
        guard selectedMainPanel == .plugin else { return }
        let pluginID = selectedPluginPanelID.trimmingCharacters(in: .whitespacesAndNewlines)
        let panelID = selectedPluginPanelViewID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !pluginID.isEmpty else {
            switchToChatPanel()
            return
        }
        guard let entry = pluginEntries.first(where: { $0.plugin_id == pluginID }) else {
            switchToChatPanel()
            return
        }
        guard !panelID.isEmpty else { return }
        let views = entry.manifest.ui?.views ?? []
        let panelExists = views.contains(where: { $0.location == .mainPanel && $0.id == panelID })
        if !panelExists {
            switchToChatPanel()
        }
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

    private func resolvePluginImportDescriptor(_ url: URL) throws -> PluginImportDescriptor {
        let fm = FileManager.default
        var isDirectory: ObjCBool = false
        guard fm.fileExists(atPath: url.path, isDirectory: &isDirectory) else {
            throw APIError(message: "선택한 경로를 찾을 수 없습니다: \(url.path)")
        }

        if isDirectory.boolValue {
            let manifestURL = url.appendingPathComponent("plugin.json")
            guard fm.fileExists(atPath: manifestURL.path) else {
                throw APIError(message: "플러그인 폴더에 plugin.json이 없습니다: \(manifestURL.path)")
            }
            return PluginImportDescriptor(manifestURL: manifestURL, pluginDirectory: url, shouldPrepareRuntime: true)
        }

        guard url.pathExtension.lowercased() == "json" else {
            throw APIError(message: "지원하지 않는 파일 형식입니다. plugin.json 또는 플러그인 폴더를 선택해 주세요.")
        }
        let pluginDirectory = url.deletingLastPathComponent()
        let shouldPrepareRuntime = false
        return PluginImportDescriptor(
            manifestURL: url,
            pluginDirectory: pluginDirectory,
            shouldPrepareRuntime: shouldPrepareRuntime
        )
    }
}
