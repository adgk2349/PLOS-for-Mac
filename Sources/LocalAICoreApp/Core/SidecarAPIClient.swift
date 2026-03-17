import Foundation

struct APIError: Error, LocalizedError {
    let message: String

    var errorDescription: String? { message }
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
        decoder.dateDecodingStrategy = .iso8601
        self.decoder = decoder

        let encoder = JSONEncoder()
        encoder.dateEncodingStrategy = .iso8601
        self.encoder = encoder
    }

    func health() async throws {
        _ = try await request(path: "/health", method: "GET", body: Optional<String>.none as String?) as [String: String]
    }

    func updateWorkspace(_ payload: WorkspaceUpdateRequest) async throws -> WorkspaceResponse {
        try await request(path: "/v1/workspaces", method: "POST", body: payload)
    }

    func startIndexJob(scope: String) async throws -> IndexJobStartResponse {
        try await request(path: "/v1/index/jobs", method: "POST", body: IndexJobRequest(scope: scope))
    }

    func getIndexJob(jobID: String) async throws -> IndexJobStatus {
        try await request(path: "/v1/index/jobs/\(jobID)", method: "GET", body: Optional<String>.none as String?)
    }

    func getFailures() async throws -> FailureListResponse {
        try await request(path: "/v1/index/failures", method: "GET", body: Optional<String>.none as String?)
    }

    func localChat(_ payload: LocalChatRequest) async throws -> LocalChatResponse {
        try await request(path: "/v1/chat/local", method: "POST", body: payload)
    }

    func deepAnalysis(_ payload: DeepAnalysisRequest) async throws -> DeepAnalysisResponse {
        try await request(path: "/v1/chat/deep-analysis", method: "POST", body: payload)
    }

    func getSettings() async throws -> SettingsModel {
        try await request(path: "/v1/settings", method: "GET", body: Optional<String>.none as String?)
    }

    func updateSettings(_ payload: SettingsModel) async throws -> SettingsModel {
        try await request(path: "/v1/settings", method: "PUT", body: payload)
    }

    func getStatus() async throws -> StatusSnapshot {
        try await request(path: "/v1/status", method: "GET", body: Optional<String>.none as String?)
    }

    private func request<T: Decodable, U: Encodable>(path: String, method: String, body: U?) async throws -> T {
        guard let url = URL(string: path, relativeTo: baseURL) else {
            throw APIError(message: "Invalid URL path: \(path)")
        }
        var request = URLRequest(url: url)
        request.httpMethod = method
        request.setValue(sessionToken, forHTTPHeaderField: "x-session-token")

        if let body {
            request.httpBody = try encoder.encode(body)
            request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        }

        let (data, response) = try await urlSession.data(for: request)
        guard let http = response as? HTTPURLResponse else {
            throw APIError(message: "Invalid response")
        }

        guard (200..<300).contains(http.statusCode) else {
            let text = String(data: data, encoding: .utf8) ?? "unknown error"
            throw APIError(message: "HTTP \(http.statusCode): \(text)")
        }

        do {
            return try decoder.decode(T.self, from: data)
        } catch {
            let text = String(data: data, encoding: .utf8) ?? ""
            throw APIError(message: "Decode failed: \(error.localizedDescription). Payload: \(text)")
        }
    }
}
