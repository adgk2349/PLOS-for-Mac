import Foundation

// MARK: - Infra

struct APIError: Error, LocalizedError {
    let message: String

    var errorDescription: String? {
        message
    }
}

final class SidecarAPIClient {
    private let baseURL: URL
    private let sessionToken: String
    private let urlSession: URLSession
    private let decoder: JSONDecoder
    private let encoder: JSONEncoder

    init(baseURL: URL, sessionToken: String, urlSession: URLSession = .shared) {
        self.baseURL = baseURL
        self.sessionToken = sessionToken
        self.urlSession = urlSession

        let decoder = JSONDecoder()
        decoder.dateDecodingStrategy = .custom { decoder in
            let container = try decoder.singleValueContainer()
            let raw = try container.decode(String.self)
            if let date = SidecarAPIClient.iso8601WithFractional.date(from: raw) ?? SidecarAPIClient.iso8601.date(from: raw) {
                return date
            }
            throw DecodingError.dataCorruptedError(in: container, debugDescription: "Invalid date: \(raw)")
        }
        self.decoder = decoder

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        self.encoder = encoder
    }

    private static let iso8601: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime]
        return formatter
    }()

    private static let iso8601WithFractional: ISO8601DateFormatter = {
        let formatter = ISO8601DateFormatter()
        formatter.formatOptions = [.withInternetDateTime, .withFractionalSeconds]
        return formatter
    }()

    func health() async throws {
        _ = try await request(path: "/health", method: "GET") as [String: String]
    }

    func updateWorkspace(_ payload: WorkspaceUpdateRequest) async throws -> WorkspaceResponse {
        try await request(path: "/v1/workspaces", method: "POST", body: payload)
    }

    func startIndexJob(scope: String) async throws -> IndexJobStartResponse {
        try await request(path: "/v1/index/jobs", method: "POST", body: IndexJobRequest(scope: scope))
    }

    func getIndexJob(jobID: String) async throws -> IndexJobStatus {
        try await request(path: "/v1/index/jobs/\(jobID)", method: "GET")
    }

    func getFailures() async throws -> FailureListResponse {
        try await request(path: "/v1/index/failures", method: "GET")
    }

    func localChat(_ payload: LocalChatRequest) async throws -> LocalChatResponse {
        try await request(path: "/v1/chat/local", method: "POST", body: payload)
    }

    func deepAnalysis(_ payload: DeepAnalysisRequest) async throws -> DeepAnalysisResponse {
        try await request(path: "/v1/chat/deep-analysis", method: "POST", body: payload)
    }

    func getSettings() async throws -> SettingsModel {
        try await request(path: "/v1/settings", method: "GET")
    }

    func updateSettings(_ payload: SettingsModel) async throws -> SettingsModel {
        try await request(path: "/v1/settings", method: "PUT", body: payload)
    }

    func getStatus() async throws -> StatusSnapshot {
        try await request(path: "/v1/status", method: "GET")
    }

    func listDocuments(
        search: String?,
        category: String?,
        tags: [String],
        year: Int?,
        project: String?,
        excluded: Bool?
    ) async throws -> DocumentListResponse {
        guard var components = URLComponents(url: baseURL.appendingPathComponent("/v1/docs"), resolvingAgainstBaseURL: true) else {
            throw APIError(message: "Invalid docs URL")
        }
        var queryItems: [URLQueryItem] = []
        if let search, !search.isEmpty {
            queryItems.append(URLQueryItem(name: "search", value: search))
        }
        if let category, !category.isEmpty {
            queryItems.append(URLQueryItem(name: "category", value: category))
        }
        if !tags.isEmpty {
            queryItems.append(URLQueryItem(name: "tags", value: tags.joined(separator: ",")))
        }
        if let year {
            queryItems.append(URLQueryItem(name: "year", value: "\(year)"))
        }
        if let project, !project.isEmpty {
            queryItems.append(URLQueryItem(name: "project", value: project))
        }
        if let excluded {
            queryItems.append(URLQueryItem(name: "excluded", value: excluded ? "true" : "false"))
        }
        components.queryItems = queryItems.isEmpty ? nil : queryItems
        guard let url = components.url else {
            throw APIError(message: "Invalid docs query URL")
        }
        return try await request(url: url, method: "GET", body: Optional<String>.none as String?)
    }

    func updateDocumentMetadata(docID: String, payload: DocumentMetadataUpdateRequest) async throws -> DocumentMetadata {
        try await request(path: "/v1/docs/\(docID)/metadata", method: "PUT", body: payload)
    }

    func reclassifyDocument(docID: String) async throws -> DocumentMetadata {
        try await request(path: "/v1/docs/\(docID)/reclassify", method: "POST")
    }

    func downloadModel(_ payload: ModelDownloadRequest) async throws -> ModelDownloadResponse {
        try await request(path: "/v1/models/download", method: "POST", body: payload)
    }

    func listModels() async throws -> ModelListResponse {
        try await request(path: "/v1/models", method: "GET")
    }

    func getModelCatalog() async throws -> ModelCatalogResponse {
        try await request(path: "/v1/models/catalog", method: "GET")
    }

    func installCatalogModel(modelID: String) async throws -> ModelCatalogInstallResponse {
        try await request(
            path: "/v1/models/catalog/install",
            method: "POST",
            body: ModelCatalogInstallRequest(model_id: modelID),
            timeout: 3600
        )
    }

    func activateCatalogModel(modelID: String) async throws -> ModelCatalogActivateResponse {
        try await request(
            path: "/v1/models/catalog/activate",
            method: "POST",
            body: ModelCatalogActivateRequest(model_id: modelID)
        )
    }

    func deleteCatalogModel(modelID: String) async throws -> ModelCatalogDeleteResponse {
        try await request(path: "/v1/models/catalog/\(modelID)", method: "DELETE")
    }

    func prepareRuntime(_ payload: RuntimePrepareRequest) async throws -> RuntimePrepareResponse {
        try await request(path: "/v1/models/runtime/prepare", method: "POST", body: payload, timeout: 1800)
    }

    private func request<T: Decodable>(path: String, method: String, timeout: TimeInterval? = nil) async throws -> T {
        try await request(path: path, method: method, body: Optional<String>.none as String?, timeout: timeout)
    }

    private func request<T: Decodable, U: Encodable>(
        path: String,
        method: String,
        body: U?,
        timeout: TimeInterval? = nil
    ) async throws -> T {
        guard let url = URL(string: path, relativeTo: baseURL) else {
            throw APIError(message: "Invalid URL path: \(path)")
        }
        return try await request(url: url, method: method, body: body, timeout: timeout)
    }

    private func request<T: Decodable, U: Encodable>(
        url: URL,
        method: String,
        body: U?,
        timeout: TimeInterval? = nil
    ) async throws -> T {
        var req = URLRequest(url: url)
        req.httpMethod = method
        req.setValue(sessionToken, forHTTPHeaderField: "x-session-token")
        if let timeout {
            req.timeoutInterval = timeout
        }

        if let body {
            req.httpBody = try encoder.encode(body)
            req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }

        let (data, response) = try await urlSession.data(for: req)
        guard let http = response as? HTTPURLResponse else {
            throw APIError(message: "Invalid response")
        }

        guard (200 ..< 300).contains(http.statusCode) else {
            let text = String(data: data, encoding: .utf8) ?? "unknown error"
            throw APIError(message: "HTTP \(http.statusCode): \(text)")
        }

        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            let payload = String(data: data, encoding: .utf8) ?? ""
            throw APIError(message: "Decode failed: \(error.localizedDescription). Payload: \(payload)")
        }
    }
}
