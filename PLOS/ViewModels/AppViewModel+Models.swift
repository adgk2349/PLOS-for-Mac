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
        case .llamaCPP:
            llamaModelPath = selectedPath
        }
        persistLocalModelPreferenceSnapshot()
    }


    func selectInstalledModel(_ model: ModelListItem) async {
        let previousEngine = localEngine
        let previousMLXPath = mlxModelPath
        let previousLlamaPath = llamaModelPath

        applyDownloadedModel(model)
        isBusy = true
        defer { isBusy = false }

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
        defer { isBusy = false }

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
        defer { isDownloadingModel = false }

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
        updateCatalogModelStatus(modelID: modelID, status: .downloading)
        defer { isCatalogBusy = false }
        defer {
            if catalogInstallingModelID == modelID {
                catalogInstallingModelID = nil
            }
        }

        do {
            _ = try await performWithSidecarRetry { client in
                try await modelRuntimeService.installCatalogModel(client: client, modelID: modelID)
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
                startupProfile = activated.profile == "fast" ? .fast : (activated.profile == "advanced" ? .deep : .recommended)
            case .llamaCPP:
                llamaModelPath = activated.model_path
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
