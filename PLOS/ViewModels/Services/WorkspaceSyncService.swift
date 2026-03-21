import Foundation

struct WorkspaceCoreSnapshot {
    let settings: SettingsModel
    let status: StatusSnapshot
    let failures: [FailureItem]
}

final class WorkspaceSyncService {
    func fetchCoreSnapshot(client: SidecarAPIClient) async throws -> WorkspaceCoreSnapshot {
        async let settingsTask = client.getSettings()
        async let statusTask = client.getStatus()
        async let failuresTask = client.getFailures()

        let (settings, status, failures) = try await (settingsTask, statusTask, failuresTask)
        return WorkspaceCoreSnapshot(settings: settings, status: status, failures: failures.failures)
    }

    func fetchDocuments(
        client: SidecarAPIClient,
        search: String?,
        category: String?,
        tags: [String],
        year: Int?,
        project: String?,
        excluded: Bool?
    ) async throws -> DocumentListResponse {
        try await client.listDocuments(
            search: search,
            category: category,
            tags: tags,
            year: year,
            project: project,
            excluded: excluded
        )
    }

    func syncWorkspaceAndSettings(
        client: SidecarAPIClient,
        workspace: WorkspaceUpdateRequest,
        settings: SettingsModel
    ) async throws {
        _ = try await client.updateWorkspace(workspace)
        _ = try await client.updateSettings(settings)
    }

    func startIndexJob(client: SidecarAPIClient, scope: String) async throws -> IndexJobStartResponse {
        try await client.startIndexJob(scope: scope)
    }

    func getIndexJob(client: SidecarAPIClient, jobID: String) async throws -> IndexJobStatus {
        try await client.getIndexJob(jobID: jobID)
    }
}
