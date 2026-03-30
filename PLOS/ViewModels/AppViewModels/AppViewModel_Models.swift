import AppKit
import Combine
import CryptoKit
import Foundation

@MainActor
extension AppViewModel {
    func isInstalledModelActive(_ model: ModelListItem) -> Bool {
        model.engine == localEngine && Self.samePath(model.path, activeModelPath)
    }


    func chooseModelFile(for engine: LocalEngine) {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = false
        panel.canChooseFiles = true
        panel.allowsMultipleSelection = false
        panel.prompt = "선택"

        if panel.runModal() == .OK, let url = panel.url {
            switch engine {
            case .mlx:
                mlxModelPath = url.path
            case .llamaCPP:
                llamaModelPath = url.path
            }
            persistLocalModelPreferenceSnapshot()
        }
    }


    func applyDownloadedModel(_ model: ModelListItem) {
        let selectedPath = normalizedSelectionPath(for: model)
        guard !selectedPath.isEmpty else {
            return
        }
        localEngine = model.engine
        switch model.engine {
        case .mlx:
            mlxModelPath = selectedPath
            llamaModelPath = ""
        case .llamaCPP:
            llamaModelPath = selectedPath
            mlxModelPath = ""
        }
        persistLocalModelPreferenceSnapshot()
    }


    func selectInstalledModel(_ model: ModelListItem) async {
        let previousEngine = localEngine
        let previousMLXPath = mlxModelPath
        let previousLlamaPath = llamaModelPath

        applyDownloadedModel(model)
        isBusy = true
        isModelRuntimeBusy = true
        defer {
            isModelRuntimeBusy = false
            isBusy = false
        }

        do {
            try await syncWorkspaceAndSettings()
            try await refreshRemoteState()
        } catch {
            localEngine = previousEngine
            mlxModelPath = previousMLXPath
            llamaModelPath = previousLlamaPath
            persistLocalModelPreferenceSnapshot()
            handleViewModelError(error)
        }
    }


    func prepareRuntimeNow() async {
        isBusy = true
        isModelRuntimeBusy = true
        defer {
            isModelRuntimeBusy = false
            isBusy = false
        }

        do {
            _ = try await performWithSidecarRetry { client in
                try await prepareSelectedRuntime(using: client)
            }
        } catch {
            handleViewModelError(error)
        }
    }


    func downloadModel() async {
        let url = modelDownloadURL.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !url.isEmpty else {
            lastError = "모델 다운로드 URL을 입력해 주세요."
            return
        }

        isDownloadingModel = true
        isModelRuntimeBusy = true
        modelDownloadProgressPercent = nil
        defer { isDownloadingModel = false }
        defer { isModelRuntimeBusy = false }
        defer { modelDownloadProgressPercent = nil }

        let progressTask = Task { @MainActor [weak self] in
            guard let self else { return }
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 650_000_000)
                if Task.isCancelled { break }
                do {
                    let snapshot = try await self.performWithSidecarRetry { client in
                        try await self.modelRuntimeService.fetchDownloadProgress(client: client)
                    }
                    self.applyLiveDownloadProgress(snapshot.items)
                } catch {
                    // Keep main download flow alive even when progress polling fails transiently.
                }
            }
        }
        defer { progressTask.cancel() }

        do {
            let response = try await performWithSidecarRetry { client in
                try await modelRuntimeService.downloadModel(
                    client: client,
                    request: ModelDownloadRequest(
                        url: url,
                        engine: modelDownloadEngine,
                        filename: modelDownloadFilename.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? nil : modelDownloadFilename
                    )
                )
            }
            if let progress = response.progress_percent {
                modelDownloadProgressPercent = min(max(progress / 100.0, 0.0), 1.0)
            } else {
                modelDownloadProgressPercent = nil
            }
            switch response.engine {
            case .mlx:
                mlxModelPath = response.saved_path
            case .llamaCPP:
                llamaModelPath = response.saved_path
            }
            availableModels = try await performWithSidecarRetry { client in
                try await modelRuntimeService.fetchInstalledModels(client: client)
            }
            modelDownloadURL = ""
            modelDownloadFilename = ""
        } catch {
            handleViewModelError(error)
        }
    }


    func installCatalogModel(_ modelID: String) async {
        isCatalogBusy = true
        catalogInstallingModelID = modelID
        catalogInstallProgress.removeValue(forKey: modelID)
        updateCatalogModelStatus(modelID: modelID, status: .downloading)
        if let index = catalogModels.firstIndex(where: { $0.id == modelID }) {
            catalogModels[index].progress_percent = nil
            catalogModels[index].downloaded_bytes = nil
            catalogModels[index].total_bytes = nil
            catalogModels[index].failure_reason = nil
        }
        defer { isCatalogBusy = false }
        defer {
            if catalogInstallingModelID == modelID {
                catalogInstallingModelID = nil
            }
        }

        let progressTask = Task { @MainActor [weak self] in
            guard let self else { return }
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 650_000_000)
                if Task.isCancelled {
                    break
                }

                do {
                    let progress = try await self.performWithSidecarRetry { client in
                        try await self.modelRuntimeService.fetchDownloadProgress(client: client)
                    }
                    self.applyLiveDownloadProgress(progress.items)
                } catch {
                    // Keep install flow running even if progress polling fails transiently.
                }
            }
        }
        defer {
            progressTask.cancel()
        }

        do {
            _ = try await performWithSidecarRetry { client in
                try await modelRuntimeService.installCatalogModel(client: client, modelID: modelID)
            }
            catalogInstallProgress[modelID] = 1.0
            let catalog = try await performWithSidecarRetry { client in
                try await modelRuntimeService.fetchCatalog(client: client)
            }
            catalogDefaultProfile = catalog.default_profile
            catalogModels = catalog.models
            availableModels = try await performWithSidecarRetry { client in
                try await modelRuntimeService.fetchInstalledModels(client: client)
            }
            Task { @MainActor [weak self] in
                try? await Task.sleep(nanoseconds: 1_000_000_000)
                self?.catalogInstallProgress.removeValue(forKey: modelID)
            }
        } catch {
            catalogInstallProgress.removeValue(forKey: modelID)
            updateCatalogModelStatus(modelID: modelID, status: .failed)
            handleViewModelError(error)
        }
    }


    func updateCatalogModelStatus(modelID: String, status: ModelInstallStatus) {
        guard let index = catalogModels.firstIndex(where: { $0.id == modelID }) else {
            return
        }
        catalogModels[index].status = status
        if status != .failed {
            catalogModels[index].failure_reason = nil
        }
    }

    func applyLiveDownloadProgress(_ items: [DownloadProgressItem]) {
        let catalogItems = items.filter { $0.kind == .catalog }
        for item in catalogItems {
            guard let modelID = item.model_id else { continue }
            if let percent = item.progress_percent {
                catalogInstallProgress[modelID] = min(max(percent / 100.0, 0.0), 1.0)
            } else {
                catalogInstallProgress.removeValue(forKey: modelID)
            }
            guard let index = catalogModels.firstIndex(where: { $0.id == modelID }) else { continue }
            catalogModels[index].progress_percent = item.progress_percent
            catalogModels[index].downloaded_bytes = item.downloaded_bytes
            catalogModels[index].total_bytes = item.total_bytes
            switch item.status {
            case .running:
                catalogModels[index].status = .downloading
            case .failed:
                catalogModels[index].status = .failed
                if let error = item.error, !error.isEmpty {
                    catalogModels[index].failure_reason = error
                }
            case .completed:
                if catalogModels[index].status == .downloading {
                    catalogModels[index].status = .installed
                }
            }
        }

        if isDownloadingModel,
           let latestDirect = items
            .filter({ $0.kind == .direct })
            .sorted(by: { $0.updated_at > $1.updated_at })
            .first
        {
            if let percent = latestDirect.progress_percent {
                modelDownloadProgressPercent = min(max(percent / 100.0, 0.0), 1.0)
                localRuntimeDetail = String(format: "직접 다운로드 진행률 %.1f%%", percent)
            } else if latestDirect.status == .running {
                modelDownloadProgressPercent = nil
                localRuntimeDetail = "직접 다운로드 진행률: 크기 정보 없음"
            }
        }
    }


    func activateCatalogModel(_ modelID: String) async {
        isCatalogBusy = true
        defer { isCatalogBusy = false }

        do {
            let activated = try await performWithSidecarRetry { client in
                try await modelRuntimeService.activateCatalogModel(client: client, modelID: modelID)
            }
            localEngine = activated.engine
            switch activated.engine {
            case .mlx:
                mlxModelPath = activated.model_path
                llamaModelPath = ""
                startupProfile = activated.profile == "fast" ? .fast : (activated.profile == "advanced" ? .deep : .recommended)
            case .llamaCPP:
                llamaModelPath = activated.model_path
                mlxModelPath = ""
                startupProfile = activated.profile == "fast" ? .fast : (activated.profile == "advanced" ? .deep : .recommended)
            }
            persistLocalModelPreferenceSnapshot()
            try await syncWorkspaceAndSettings()
            let catalog = try await performWithSidecarRetry { client in
                try await modelRuntimeService.fetchCatalog(client: client)
            }
            catalogDefaultProfile = catalog.default_profile
            catalogModels = catalog.models
            try await refreshRemoteState()
        } catch {
            handleViewModelError(error)
        }
    }


    func deleteCatalogModel(_ modelID: String) async {
        isCatalogBusy = true
        defer { isCatalogBusy = false }

        do {
            _ = try await performWithSidecarRetry { client in
                try await modelRuntimeService.deleteCatalogModel(client: client, modelID: modelID)
            }
            let catalog = try await performWithSidecarRetry { client in
                try await modelRuntimeService.fetchCatalog(client: client)
            }
            catalogDefaultProfile = catalog.default_profile
            catalogModels = catalog.models
            availableModels = try await performWithSidecarRetry { client in
                try await modelRuntimeService.fetchInstalledModels(client: client)
            }
        } catch {
            handleViewModelError(error)
        }
    }
}
