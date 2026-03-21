import Foundation

struct MemoryStateSnapshot {
    let session: [SessionMemoryItem]
    let workspace: [WorkspaceMemoryItem]
    let preferences: [UserPreferenceItem]
    let episodic: [EpisodicMemoryEvent]
    let pinned: [PinnedMemoryItem]
}

final class MemoryServiceAdapter {
    func fetchMemoryState(
        client: SidecarAPIClient,
        sessionID: String,
        workspaceID: String,
        intent: String,
        relatedFileIDs: [String]
    ) async throws -> MemoryStateSnapshot {
        let session = try await client.getRelevantSessionMemory(sessionID: sessionID)
        let workspace = try await client.getRelevantWorkspaceMemory(workspaceID: workspaceID, intent: intent)
        let prefs = try await client.getMemoryPreferences()
        let episodic = try await client.getRelevantEpisodicMemory(
            workspaceID: workspaceID,
            intent: intent,
            relatedFileIDs: relatedFileIDs
        )
        let pins = try await client.listPins(scope: nil, workspaceID: workspaceID)

        return MemoryStateSnapshot(
            session: session.items,
            workspace: workspace.items,
            preferences: prefs.items,
            episodic: episodic.items,
            pinned: pins.items
        )
    }

    func clearMemory(client: SidecarAPIClient, request: MemoryClearRequest) async throws -> MemoryClearResponse {
        try await client.clearMemory(request)
    }

    func pinMemory(client: SidecarAPIClient, request: MemoryPinRequest) async throws -> MemoryPinResponse {
        try await client.pinMemory(request)
    }

    func unpinMemory(client: SidecarAPIClient, memoryID: String) async throws -> Bool {
        try await client.unpinMemory(memoryID: memoryID)
    }
}
