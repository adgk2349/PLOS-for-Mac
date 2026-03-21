import Foundation

final class ModelRuntimeService {
    func resolveModelPath(engine: LocalEngine, mlxModelPath: String, llamaModelPath: String) -> String? {
        switch engine {
        case .mlx:
            let trimmed = mlxModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        case .llamaCPP:
            let trimmed = llamaModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
            return trimmed.isEmpty ? nil : trimmed
        }
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
}
