import Foundation

final class ModelRuntimeService {
    func resolveModelPath(
        engine: LocalEngine,
        mlxModelPath: String,
        llamaModelPath: String,
        modelsRootPath: String?
    ) -> String? {
        switch engine {
        case .mlx:
            let trimmed = mlxModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return nil }
            if looksLikeHFRepoID(trimmed) {
                return trimmed
            }
            let explicit = URL(fileURLWithPath: trimmed).standardizedFileURL.path
            if FileManager.default.fileExists(atPath: explicit) {
                return explicit
            }
            if let remapped = remapToModelsRoot(
                explicitPath: explicit,
                engineFolder: "mlx",
                modelsRootPath: modelsRootPath,
                preferredExtension: nil
            ) {
                return remapped
            }
            return nil
        case .llamaCPP:
            let trimmed = llamaModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
            guard !trimmed.isEmpty else { return nil }
            let explicit = URL(fileURLWithPath: trimmed).standardizedFileURL.path
            if FileManager.default.fileExists(atPath: explicit) {
                return explicit
            }
            if let remapped = remapToModelsRoot(
                explicitPath: explicit,
                engineFolder: "llama_cpp",
                modelsRootPath: modelsRootPath,
                preferredExtension: "gguf"
            ) {
                return remapped
            }
            return nil
        }
    }

    private func looksLikeHFRepoID(_ value: String) -> Bool {
        let trimmed = value.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty { return false }
        if trimmed.hasPrefix("/") || trimmed.hasPrefix("~") {
            return false
        }
        if trimmed.contains(".gguf") {
            return false
        }
        if trimmed.contains(" ") {
            return false
        }
        let slashCount = trimmed.filter { $0 == "/" }.count
        return slashCount == 1
    }

    private func remapToModelsRoot(
        explicitPath: String,
        engineFolder: String,
        modelsRootPath: String?,
        preferredExtension: String?
    ) -> String? {
        let trimmedRoot = (modelsRootPath ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmedRoot.isEmpty else { return nil }
        let root = URL(fileURLWithPath: trimmedRoot).standardizedFileURL
        let engineRoot = root.appendingPathComponent(engineFolder, isDirectory: true)
        guard FileManager.default.fileExists(atPath: engineRoot.path) else { return nil }

        let explicitURL = URL(fileURLWithPath: explicitPath).standardizedFileURL
        let leafName = explicitURL.lastPathComponent
        if !leafName.isEmpty {
            let direct = engineRoot.appendingPathComponent(leafName)
            if FileManager.default.fileExists(atPath: direct.path) {
                return direct.path
            }
        }

        let targetExt = (preferredExtension ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard let enumerator = FileManager.default.enumerator(
            at: engineRoot,
            includingPropertiesForKeys: [.isRegularFileKey, .isDirectoryKey, .contentModificationDateKey],
            options: [.skipsHiddenFiles, .skipsPackageDescendants]
        ) else {
            return nil
        }

        var latestFile: URL?
        var latestDate = Date.distantPast
        for case let fileURL as URL in enumerator {
            if !targetExt.isEmpty, fileURL.pathExtension.lowercased() != targetExt {
                continue
            }
            let values = try? fileURL.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey, .contentModificationDateKey])
            if values?.isRegularFile != true {
                continue
            }
            let modified = values?.contentModificationDate ?? Date.distantPast
            if modified > latestDate {
                latestDate = modified
                latestFile = fileURL
            }
        }
        return latestFile?.path
    }

    func prepareRuntime(
        client: SidecarAPIClient,
        engine: LocalEngine,
        modelPath: String?
    ) async throws -> RuntimePrepareResponse {
        try await client.prepareRuntime(
            RuntimePrepareRequest(
                engine: engine,
                model_path: modelPath
            )
        )
    }

    func downloadModel(client: SidecarAPIClient, request: ModelDownloadRequest) async throws -> ModelDownloadResponse {
        try await client.downloadModel(request)
    }

    func installCatalogModel(client: SidecarAPIClient, modelID: String) async throws -> ModelCatalogInstallResponse {
        try await client.installCatalogModel(modelID: modelID)
    }

    func activateCatalogModel(client: SidecarAPIClient, modelID: String) async throws -> ModelCatalogActivateResponse {
        try await client.activateCatalogModel(modelID: modelID)
    }

    func deleteCatalogModel(client: SidecarAPIClient, modelID: String) async throws -> ModelCatalogDeleteResponse {
        try await client.deleteCatalogModel(modelID: modelID)
    }

    func fetchCatalog(client: SidecarAPIClient) async throws -> ModelCatalogResponse {
        try await client.getModelCatalog()
    }

    func fetchInstalledModels(client: SidecarAPIClient) async throws -> [ModelListItem] {
        try await client.listModels().models
    }

    func fetchDownloadProgress(client: SidecarAPIClient) async throws -> DownloadProgressResponse {
        try await client.getModelDownloadProgress()
    }
}
