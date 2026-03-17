import Foundation

@MainActor
final class SidecarProcessManager: ObservableObject {
    @Published private(set) var isRunning = false

    private(set) var sessionToken = UUID().uuidString
    private var process: Process?
    private(set) var apiClient: SidecarAPIClient?

    private let host = "127.0.0.1"
    private let port = 8777

    func start() async throws {
        if isRunning, let client = apiClient {
            _ = client
            return
        }

        let workspaceRoot = URL(fileURLWithPath: FileManager.default.currentDirectoryPath)
        let sidecarDirectory = workspaceRoot.appendingPathComponent("sidecar")
        let dataDirectory = sidecarDirectory.appendingPathComponent("data")
        try FileManager.default.createDirectory(at: dataDirectory, withIntermediateDirectories: true)

        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/env")
        process.arguments = [
            "python3",
            "-m",
            "uvicorn",
            "local_ai_core.main:app",
            "--host",
            host,
            "--port",
            String(port)
        ]
        process.currentDirectoryURL = sidecarDirectory

        var env = ProcessInfo.processInfo.environment
        env["LOCAL_AI_SESSION_TOKEN"] = sessionToken
        env["LOCAL_AI_DATA_DIR"] = dataDirectory.path
        process.environment = env

        let stdout = Pipe()
        process.standardOutput = stdout
        process.standardError = stdout

        try process.run()
        self.process = process

        let client = SidecarAPIClient(
            baseURL: URL(string: "http://\(host):\(port)")!,
            sessionToken: sessionToken
        )
        self.apiClient = client

        try await waitUntilHealthy(client: client)
        isRunning = true
    }

    func stop() {
        process?.terminate()
        process = nil
        isRunning = false
    }

    private func waitUntilHealthy(client: SidecarAPIClient) async throws {
        for _ in 0..<40 {
            do {
                try await client.health()
                return
            } catch {
                try await Task.sleep(nanoseconds: 250_000_000)
            }
        }
        throw APIError(message: "Sidecar did not become healthy")
    }
}
