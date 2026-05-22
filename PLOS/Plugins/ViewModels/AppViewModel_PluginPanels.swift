import Foundation

@MainActor
extension AppViewModel {
    struct SidebarPluginPanelItem: Identifiable, Hashable {
        let id: String
        let pluginID: String
        let panelID: String
        let title: String
        let subtitle: String?
        let pluginEnabled: Bool
        let viewType: PluginUIViewType
    }

    var sidebarPluginPanels: [SidebarPluginPanelItem] {
        let language = appLanguage
        var items: [SidebarPluginPanelItem] = []

        for entry in pluginEntries where !isBuiltInPluginEntry(entry) {
            let views = entry.manifest.ui?.views ?? []
            for view in views where view.location == .mainPanel {
                items.append(
                    SidebarPluginPanelItem(
                        id: "\(entry.plugin_id)::\(view.id)",
                        pluginID: entry.plugin_id,
                        panelID: view.id,
                        title: view.title(language: language),
                        subtitle: view.help(language: language),
                        pluginEnabled: entry.enabled,
                        viewType: view.view_type
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
            return lhs.panelID < rhs.panelID
        }
    }

    var selectedPluginPanelCompositeID: String {
        let pluginID = selectedPluginPanelID.trimmingCharacters(in: .whitespacesAndNewlines)
        let panelID = selectedPluginPanelViewID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !pluginID.isEmpty else { return "" }
        guard !panelID.isEmpty else { return pluginID }
        return "\(pluginID)::\(panelID)"
    }

    var selectedPluginPanelSelection: (pluginID: String, panelID: String?)? {
        let pluginID = selectedPluginPanelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !pluginID.isEmpty else { return nil }
        let panelID = selectedPluginPanelViewID.trimmingCharacters(in: .whitespacesAndNewlines)
        return (pluginID: pluginID, panelID: panelID.isEmpty ? nil : panelID)
    }

    func selectPluginPanel(pluginID: String, panelID: String? = nil) {
        selectedPluginPanelID = pluginID
        selectedPluginPanelViewID = panelID ?? ""
        selectedMainPanel = .plugin
        Task {
            await openPluginPanel(pluginID: pluginID, panelID: panelID)
        }
    }

    func switchToChatPanel() {
        pluginPanelStatusStreamTask?.cancel()
        pluginPanelStatusStreamTask = nil
        selectedMainPanel = .chat
    }

    func openPluginPanel(pluginID: String, panelID: String? = nil) async {
        pluginPanelStatusStreamTask?.cancel()
        pluginPanelStatusStreamTask = nil
        pluginPanelIsBusy = true
        defer { pluginPanelIsBusy = false }

        do {
            let response = try await performWithSidecarRetry { client in
                try await extensionServiceAdapter.openPanel(
                    client: client,
                    request: PluginPanelOpenRequest(plugin_id: pluginID, panel_id: panelID)
                )
            }
            selectedPluginPanelID = response.plugin_id
            selectedPluginPanelViewID = response.panel_id
            activePluginPanel = response
            applyPluginPanelDefaults(response.defaults)
            pluginPanelImages = []
            pluginPanelDownloadMessage = ""
            pluginPanelLastJobID = nil
        } catch {
            handleViewModelError(error)
            selectedMainPanel = .chat
        }
    }

    func submitActivePluginPanelAction() async {
        let seed = Int(pluginPanelSeedText.trimmingCharacters(in: .whitespacesAndNewlines))
        let payload: [String: JSONValue] = [
            "prompt": .string(pluginPanelPrompt),
            "negative_prompt": .string(pluginPanelNegativePrompt),
            "width": .number(Double(pluginPanelWidth)),
            "height": .number(Double(pluginPanelHeight)),
            "steps": .number(Double(pluginPanelSteps)),
            "batch": .number(Double(pluginPanelBatch)),
            "model_id": .string(pluginPanelModelID),
            "seed": seed.map { .number(Double($0)) } ?? .null,
        ]
        await submitPluginPanelAction(action: "generate", payload: payload)
    }

    func downloadPluginPanelModel() async {
        let modelID = pluginPanelModelID.trimmingCharacters(in: .whitespacesAndNewlines)
        let repoID = pluginPanelRepoID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !modelID.isEmpty else {
            lastError = "model_id를 입력해 주세요."
            return
        }
        var payload: [String: JSONValue] = [
            "model_id": .string(modelID),
            "repo_id": .string(repoID),
        ]
        let filename = pluginPanelFilename.trimmingCharacters(in: .whitespacesAndNewlines)
        if !filename.isEmpty {
            payload["filename"] = .string(filename)
        }
        await submitPluginPanelAction(action: "download_model", payload: payload)
    }

    func refreshPluginPanelModels() async {
        await submitPluginPanelAction(action: "list_models", payload: [:])
    }

    func setActivePluginPanelModel() async {
        let modelID = pluginPanelModelID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !modelID.isEmpty else {
            lastError = "model_id를 입력해 주세요."
            return
        }
        await submitPluginPanelAction(
            action: "set_active_model",
            payload: ["model_id": .string(modelID)]
        )
    }

    private func submitPluginPanelAction(action: String, payload: [String: JSONValue]) async {
        guard let panel = activePluginPanel else { return }
        pluginPanelIsBusy = true
        defer { pluginPanelIsBusy = false }

        do {
            let actionResponse = try await performWithSidecarRetry { client in
                try await extensionServiceAdapter.submitPanelAction(
                    client: client,
                    request: PluginPanelActionRequest(
                        plugin_id: panel.plugin_id,
                        panel_id: panel.panel_id,
                        action: action,
                        payload: payload
                    )
                )
            }
            pluginPanelLastJobID = actionResponse.job_id
            applyPluginPanelResult(actionResponse.result)
            if let error = actionResponse.error, !error.isEmpty {
                lastError = error
            }

            let initialStatus = actionResponse.status.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
            if !isTerminalPluginPanelStatus(initialStatus) {
                try await streamPluginPanelStatusUpdates(
                    pluginID: panel.plugin_id,
                    panelID: panel.panel_id,
                    jobID: actionResponse.job_id
                )
            }
        } catch {
            handleViewModelError(error)
        }
    }

    private func streamPluginPanelStatusUpdates(pluginID: String, panelID: String, jobID: String) async throws {
        pluginPanelStatusStreamTask?.cancel()

        let stream = try await performWithSidecarRetry { client in
            try await extensionServiceAdapter.streamPanelStatus(client: client, jobID: jobID)
        }

        let task = Task { @MainActor [weak self] in
            guard let self else { return }
            do {
                for try await status in stream {
                    if Task.isCancelled {
                        return
                    }
                    if self.pluginPanelLastJobID != jobID {
                        return
                    }
                    let statusPluginID = status.plugin_id.trimmingCharacters(in: .whitespacesAndNewlines)
                    let statusPanelID = status.panel_id.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard statusPluginID == pluginID, statusPanelID == panelID else {
                        continue
                    }
                    self.applyPluginPanelResult(status.result)
                    if let error = status.error, !error.isEmpty {
                        self.lastError = error
                    }
                    let normalized = status.status.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
                    if self.isTerminalPluginPanelStatus(normalized) {
                        return
                    }
                }
            } catch {
                if !Task.isCancelled {
                    self.handleViewModelError(error)
                }
            }
        }

        pluginPanelStatusStreamTask = task
        await task.value
        if pluginPanelLastJobID == jobID {
            pluginPanelStatusStreamTask = nil
        }
    }

    private func isTerminalPluginPanelStatus(_ status: String) -> Bool {
        status != "queued" && status != "running"
    }

    private func applyPluginPanelDefaults(_ defaults: [String: JSONValue]) {
        if let width = defaults["width"]?.numberValue {
            pluginPanelWidth = max(256, Int(width))
        }
        if let height = defaults["height"]?.numberValue {
            pluginPanelHeight = max(256, Int(height))
        }
        if let steps = defaults["steps"]?.numberValue {
            pluginPanelSteps = max(1, Int(steps))
        }
        if let batch = defaults["batch"]?.numberValue {
            pluginPanelBatch = max(1, Int(batch))
        }
        if let modelID = defaults["model_id"]?.stringValue, !modelID.isEmpty {
            pluginPanelModelID = modelID
        } else if let modelID = defaults["default_model_id"]?.stringValue, !modelID.isEmpty {
            pluginPanelModelID = modelID
        }
        if let activeModelID = defaults["active_model_id"]?.stringValue {
            pluginPanelActiveModelID = activeModelID
        }
        if let installed = defaults["installed_models"]?.arrayValue {
            pluginPanelInstalledModels = installed.compactMap { $0.stringValue }
        }
    }

    private func applyPluginPanelResult(_ result: [String: JSONValue]) {
        if let images = result["images"]?.arrayValue {
            let uris = images.compactMap { item -> String? in
                guard let object = item.objectValue else { return nil }
                return object["uri"]?.stringValue
            }
            if !uris.isEmpty {
                pluginPanelImages = uris
            }
        }
        if let message = result["message"]?.stringValue {
            pluginPanelDownloadMessage = message
        }
        if let modelID = result["active_model_id"]?.stringValue {
            pluginPanelActiveModelID = modelID
        }
        if let models = result["installed_models"]?.arrayValue {
            pluginPanelInstalledModels = models.compactMap { $0.stringValue }
        }
    }
}
