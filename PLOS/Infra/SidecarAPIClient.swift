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

    func getRoomStorageStatus(roomID: String) async throws -> RoomStorageStatusResponse {
        try await request(path: "/v1/rooms/\(roomID)/storage/status", method: "GET")
    }

    func reindexRoomStorage(
        roomID: String,
        scope: String = "full",
        includedPaths: [String]? = nil,
        excludedPaths: [String]? = nil
    ) async throws -> RoomStorageReindexResponse {
        let payload = RoomStorageReindexRequest(
            scope: scope,
            included_paths: includedPaths,
            excluded_paths: excludedPaths
        )
        return try await request(path: "/v1/rooms/\(roomID)/storage/reindex", method: "POST", body: payload)
    }

    func deleteRoomStorage(roomID: String) async throws -> Bool {
        let response: [String: JSONValue] = try await request(path: "/v1/rooms/\(roomID)/storage", method: "DELETE")
        return response["removed"]?.boolCoercedValue ?? false
    }

    func getExtensionCapabilities() async throws -> ExtensionCapabilitiesResponse {
        try await request(path: "/v1/extensions/capabilities", method: "GET")
    }

    func listExtensionPlugins() async throws -> PluginRegistryResponse {
        try await request(path: "/v1/extensions/plugins", method: "GET")
    }

    func registerExtensionPlugin(_ payload: PluginRegisterRequest) async throws -> PluginRegistryResponse {
        try await request(path: "/v1/extensions/plugins/register", method: "POST", body: payload)
    }

    func enableExtensionPlugin(pluginID: String) async throws -> PluginEnableResponse {
        try await request(path: "/v1/extensions/plugins/\(pluginID)/enable", method: "POST")
    }

    func disableExtensionPlugin(pluginID: String) async throws -> PluginEnableResponse {
        try await request(path: "/v1/extensions/plugins/\(pluginID)/disable", method: "POST")
    }

    func deleteExtensionPlugin(pluginID: String) async throws -> Bool {
        let response: [String: JSONValue] = try await request(path: "/v1/extensions/plugins/\(pluginID)", method: "DELETE")
        return response["removed"]?.boolCoercedValue ?? false
    }

    func localChat(_ payload: LocalChatRequest) async throws -> LocalChatResponse {
        try await request(path: "/v1/chat/local", method: "POST", body: payload)
    }

    func localChatV2(_ payload: LocalChatRequestV2) async throws -> ComposedChatResponseV2 {
        try await request(path: "/v2/chat/local", method: "POST", body: payload)
    }

    func localChatV2Stream(_ payload: LocalChatRequestV2) async throws -> AsyncThrowingStream<ChatStreamEvent, Error> {
        guard let url = URL(string: "/v2/chat/local/stream", relativeTo: baseURL) else {
            throw APIError(message: "Invalid URL path: /v2/chat/local/stream")
        }
        var req = URLRequest(url: url)
        req.httpMethod = "POST"
        req.setValue(sessionToken, forHTTPHeaderField: "x-session-token")
        
        req.httpBody = try encoder.encode(payload)
        req.setValue("application/json", forHTTPHeaderField: "Content-Type")
        req.timeoutInterval = 300 // long timeout for SSE
        
        let (bytes, response) = try await urlSession.bytes(for: req)
        guard let http = response as? HTTPURLResponse else {
            throw APIError(message: "Invalid response")
        }
        guard (200 ..< 300).contains(http.statusCode) else {
            throw APIError(message: "HTTP \(http.statusCode) stream error")
        }
        
        return AsyncThrowingStream { continuation in
            Task {
                do {
                    for try await line in bytes.lines {
                        let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
                        guard !trimmed.isEmpty else { continue }
                        guard let data = trimmed.data(using: .utf8) else { continue }
                        let event = try self.decoder.decode(ChatStreamEvent.self, from: data)
                        continuation.yield(event)
                    }
                    continuation.finish()
                } catch {
                    continuation.finish(throwing: error)
                }
            }
        }
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

    func getRelevantSessionMemory(sessionID: String) async throws -> SessionMemoryResponse {
        guard var components = URLComponents(url: baseURL.appendingPathComponent("/v1/memory/session/relevant"), resolvingAgainstBaseURL: true) else {
            throw APIError(message: "Invalid memory session URL")
        }
        components.queryItems = [URLQueryItem(name: "session_id", value: sessionID)]
        guard let url = components.url else {
            throw APIError(message: "Invalid memory session query URL")
        }
        return try await request(url: url, method: "GET", body: Optional<String>.none as String?)
    }

    func getRoomRelevantSessionMemory(
        roomID: String,
        sessionID: String?,
        roomScopeHash: String? = nil
    ) async throws -> SessionMemoryResponse {
        guard var components = URLComponents(url: baseURL.appendingPathComponent("/v1/rooms/\(roomID)/memory/session/relevant"), resolvingAgainstBaseURL: true) else {
            throw APIError(message: "Invalid room memory session URL")
        }
        var items: [URLQueryItem] = []
        if let sessionID, !sessionID.isEmpty {
            items.append(URLQueryItem(name: "session_id", value: sessionID))
        }
        if let roomScopeHash, !roomScopeHash.isEmpty {
            items.append(URLQueryItem(name: "room_scope_hash", value: roomScopeHash))
        }
        components.queryItems = items.isEmpty ? nil : items
        guard let url = components.url else {
            throw APIError(message: "Invalid room memory session query URL")
        }
        return try await request(url: url, method: "GET", body: Optional<String>.none as String?)
    }

    func getRelevantWorkspaceMemory(workspaceID: String, intent: String?) async throws -> WorkspaceMemoryResponse {
        guard var components = URLComponents(url: baseURL.appendingPathComponent("/v1/memory/workspace/relevant"), resolvingAgainstBaseURL: true) else {
            throw APIError(message: "Invalid memory workspace URL")
        }
        var items = [URLQueryItem(name: "workspace_id", value: workspaceID)]
        if let intent, !intent.isEmpty {
            items.append(URLQueryItem(name: "intent", value: intent))
        }
        components.queryItems = items
        guard let url = components.url else {
            throw APIError(message: "Invalid memory workspace query URL")
        }
        return try await request(url: url, method: "GET", body: Optional<String>.none as String?)
    }

    func getRoomRelevantWorkspaceMemory(
        roomID: String,
        workspaceID: String?,
        intent: String?,
        roomScopeHash: String? = nil
    ) async throws -> WorkspaceMemoryResponse {
        guard var components = URLComponents(url: baseURL.appendingPathComponent("/v1/rooms/\(roomID)/memory/workspace/relevant"), resolvingAgainstBaseURL: true) else {
            throw APIError(message: "Invalid room memory workspace URL")
        }
        var items = [URLQueryItem]()
        if let workspaceID, !workspaceID.isEmpty {
            items.append(URLQueryItem(name: "workspace_id", value: workspaceID))
        }
        if let intent, !intent.isEmpty {
            items.append(URLQueryItem(name: "intent", value: intent))
        }
        if let roomScopeHash, !roomScopeHash.isEmpty {
            items.append(URLQueryItem(name: "room_scope_hash", value: roomScopeHash))
        }
        components.queryItems = items.isEmpty ? nil : items
        guard let url = components.url else {
            throw APIError(message: "Invalid room memory workspace query URL")
        }
        return try await request(url: url, method: "GET", body: Optional<String>.none as String?)
    }

    func getMemoryPreferences() async throws -> UserPreferencesResponse {
        try await request(path: "/v1/memory/preferences", method: "GET")
    }

    func getRelevantEpisodicMemory(
        workspaceID: String?,
        intent: String?,
        relatedFileIDs: [String]
    ) async throws -> EpisodicMemoryResponse {
        guard var components = URLComponents(url: baseURL.appendingPathComponent("/v1/memory/episodic/relevant"), resolvingAgainstBaseURL: true) else {
            throw APIError(message: "Invalid memory episodic URL")
        }
        var items: [URLQueryItem] = []
        if let workspaceID, !workspaceID.isEmpty {
            items.append(URLQueryItem(name: "workspace_id", value: workspaceID))
        }
        if let intent, !intent.isEmpty {
            items.append(URLQueryItem(name: "intent", value: intent))
        }
        if !relatedFileIDs.isEmpty {
            items.append(URLQueryItem(name: "related_file_ids", value: relatedFileIDs.joined(separator: ",")))
        }
        components.queryItems = items.isEmpty ? nil : items
        guard let url = components.url else {
            throw APIError(message: "Invalid memory episodic query URL")
        }
        return try await request(url: url, method: "GET", body: Optional<String>.none as String?)
    }

    func getRoomRelevantEpisodicMemory(
        roomID: String,
        workspaceID: String?,
        intent: String?,
        relatedFileIDs: [String],
        roomScopeHash: String? = nil
    ) async throws -> EpisodicMemoryResponse {
        guard var components = URLComponents(url: baseURL.appendingPathComponent("/v1/rooms/\(roomID)/memory/episodic/relevant"), resolvingAgainstBaseURL: true) else {
            throw APIError(message: "Invalid room memory episodic URL")
        }
        var items: [URLQueryItem] = []
        if let workspaceID, !workspaceID.isEmpty {
            items.append(URLQueryItem(name: "workspace_id", value: workspaceID))
        }
        if let intent, !intent.isEmpty {
            items.append(URLQueryItem(name: "intent", value: intent))
        }
        if !relatedFileIDs.isEmpty {
            items.append(URLQueryItem(name: "related_file_ids", value: relatedFileIDs.joined(separator: ",")))
        }
        if let roomScopeHash, !roomScopeHash.isEmpty {
            items.append(URLQueryItem(name: "room_scope_hash", value: roomScopeHash))
        }
        components.queryItems = items.isEmpty ? nil : items
        guard let url = components.url else {
            throw APIError(message: "Invalid room memory episodic query URL")
        }
        return try await request(url: url, method: "GET", body: Optional<String>.none as String?)
    }

    func writeMemoryEvent(_ payload: MemoryEventRequest) async throws -> MemoryEventResponse {
        try await request(path: "/v1/memory/events", method: "POST", body: payload)
    }

    func writeRoomMemoryEvent(
        roomID: String,
        payload: MemoryEventRequest,
        roomScopeHash: String? = nil
    ) async throws -> MemoryEventResponse {
        guard var components = URLComponents(url: baseURL.appendingPathComponent("/v1/rooms/\(roomID)/memory/events"), resolvingAgainstBaseURL: true) else {
            throw APIError(message: "Invalid room memory events URL")
        }
        if let roomScopeHash, !roomScopeHash.isEmpty {
            components.queryItems = [URLQueryItem(name: "room_scope_hash", value: roomScopeHash)]
        }
        guard let url = components.url else {
            throw APIError(message: "Invalid room memory events query URL")
        }
        return try await request(url: url, method: "POST", body: payload)
    }

    func clearMemory(_ payload: MemoryClearRequest) async throws -> MemoryClearResponse {
        try await request(path: "/v1/memory/clear", method: "POST", body: payload)
    }

    func clearRoomMemory(
        roomID: String,
        _ payload: MemoryClearRequest,
        roomScopeHash: String? = nil
    ) async throws -> MemoryClearResponse {
        guard var components = URLComponents(url: baseURL.appendingPathComponent("/v1/rooms/\(roomID)/memory/clear"), resolvingAgainstBaseURL: true) else {
            throw APIError(message: "Invalid room memory clear URL")
        }
        if let roomScopeHash, !roomScopeHash.isEmpty {
            components.queryItems = [URLQueryItem(name: "room_scope_hash", value: roomScopeHash)]
        }
        guard let url = components.url else {
            throw APIError(message: "Invalid room memory clear query URL")
        }
        return try await request(url: url, method: "POST", body: payload)
    }

    func pinMemory(_ payload: MemoryPinRequest) async throws -> MemoryPinResponse {
        try await request(path: "/v1/memory/pin", method: "POST", body: payload)
    }

    func pinRoomMemory(
        roomID: String,
        _ payload: MemoryPinRequest,
        roomScopeHash: String? = nil
    ) async throws -> MemoryPinResponse {
        guard var components = URLComponents(url: baseURL.appendingPathComponent("/v1/rooms/\(roomID)/memory/pin"), resolvingAgainstBaseURL: true) else {
            throw APIError(message: "Invalid room memory pin URL")
        }
        if let roomScopeHash, !roomScopeHash.isEmpty {
            components.queryItems = [URLQueryItem(name: "room_scope_hash", value: roomScopeHash)]
        }
        guard let url = components.url else {
            throw APIError(message: "Invalid room memory pin query URL")
        }
        return try await request(url: url, method: "POST", body: payload)
    }

    func unpinMemory(memoryID: String) async throws -> Bool {
        let response: [String: JSONValue] = try await request(path: "/v1/memory/pin/\(memoryID)", method: "DELETE")
        return response["removed"]?.boolCoercedValue ?? false
    }

    func unpinRoomMemory(roomID: String, memoryID: String, roomScopeHash: String? = nil) async throws -> Bool {
        guard var components = URLComponents(url: baseURL.appendingPathComponent("/v1/rooms/\(roomID)/memory/pin/\(memoryID)"), resolvingAgainstBaseURL: true) else {
            throw APIError(message: "Invalid room memory unpin URL")
        }
        if let roomScopeHash, !roomScopeHash.isEmpty {
            components.queryItems = [URLQueryItem(name: "room_scope_hash", value: roomScopeHash)]
        }
        guard let url = components.url else {
            throw APIError(message: "Invalid room memory unpin query URL")
        }
        let response: [String: JSONValue] = try await request(url: url, method: "DELETE", body: Optional<String>.none as String?)
        return response["removed"]?.boolCoercedValue ?? false
    }

    func listPins(scope: String?, workspaceID: String?) async throws -> PinnedMemoryResponse {
        guard var components = URLComponents(url: baseURL.appendingPathComponent("/v1/memory/pins"), resolvingAgainstBaseURL: true) else {
            throw APIError(message: "Invalid memory pins URL")
        }
        var items: [URLQueryItem] = []
        if let scope, !scope.isEmpty {
            items.append(URLQueryItem(name: "scope", value: scope))
        }
        if let workspaceID, !workspaceID.isEmpty {
            items.append(URLQueryItem(name: "workspace_id", value: workspaceID))
        }
        components.queryItems = items.isEmpty ? nil : items
        guard let url = components.url else {
            throw APIError(message: "Invalid memory pins query URL")
        }
        return try await request(url: url, method: "GET", body: Optional<String>.none as String?)
    }

    func listRoomPins(
        roomID: String,
        scope: String?,
        workspaceID: String?,
        roomScopeHash: String? = nil
    ) async throws -> PinnedMemoryResponse {
        guard var components = URLComponents(url: baseURL.appendingPathComponent("/v1/rooms/\(roomID)/memory/pins"), resolvingAgainstBaseURL: true) else {
            throw APIError(message: "Invalid room memory pins URL")
        }
        var items: [URLQueryItem] = []
        if let scope, !scope.isEmpty {
            items.append(URLQueryItem(name: "scope", value: scope))
        }
        if let workspaceID, !workspaceID.isEmpty {
            items.append(URLQueryItem(name: "workspace_id", value: workspaceID))
        }
        if let roomScopeHash, !roomScopeHash.isEmpty {
            items.append(URLQueryItem(name: "room_scope_hash", value: roomScopeHash))
        }
        components.queryItems = items.isEmpty ? nil : items
        guard let url = components.url else {
            throw APIError(message: "Invalid room memory pins query URL")
        }
        return try await request(url: url, method: "GET", body: Optional<String>.none as String?)
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

    func getModelDownloadProgress() async throws -> DownloadProgressResponse {
        try await request(path: "/v1/models/download/progress", method: "GET")
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
