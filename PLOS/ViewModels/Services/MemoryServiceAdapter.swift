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
        roomID: String?,
        roomScopeHash: String?,
        sessionID: String,
        workspaceID: String,
        intent: String,
        relatedFileIDs: [String]
    ) async throws -> MemoryStateSnapshot {
        let safeRoomID = (roomID ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        var session: SessionMemoryResponse
        var workspace: WorkspaceMemoryResponse
        var episodic: EpisodicMemoryResponse
        var pins: PinnedMemoryResponse
        if !safeRoomID.isEmpty {
            session = try await client.getRoomRelevantSessionMemory(
                roomID: safeRoomID,
                sessionID: sessionID,
                roomScopeHash: roomScopeHash
            )
            workspace = try await client.getRoomRelevantWorkspaceMemory(
                roomID: safeRoomID,
                workspaceID: workspaceID,
                intent: intent,
                roomScopeHash: roomScopeHash
            )
            episodic = try await client.getRoomRelevantEpisodicMemory(
                roomID: safeRoomID,
                workspaceID: workspaceID,
                intent: intent,
                relatedFileIDs: relatedFileIDs,
                roomScopeHash: roomScopeHash
            )
            pins = try await client.listRoomPins(
                roomID: safeRoomID,
                scope: nil,
                workspaceID: workspaceID,
                roomScopeHash: roomScopeHash
            )
        } else {
            session = try await client.getRelevantSessionMemory(sessionID: sessionID)
            workspace = try await client.getRelevantWorkspaceMemory(workspaceID: workspaceID, intent: intent)
            episodic = try await client.getRelevantEpisodicMemory(
                workspaceID: workspaceID,
                intent: intent,
                relatedFileIDs: relatedFileIDs
            )
            pins = try await client.listPins(scope: nil, workspaceID: workspaceID)
        }
        let prefs = try await client.getMemoryPreferences()

        return MemoryStateSnapshot(
            session: session.items,
            workspace: workspace.items,
            preferences: prefs.items,
            episodic: episodic.items,
            pinned: pins.items
        )
    }

    func clearMemory(
        client: SidecarAPIClient,
        roomID: String?,
        roomScopeHash: String?,
        request: MemoryClearRequest
    ) async throws -> MemoryClearResponse {
        let safeRoomID = (roomID ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        if !safeRoomID.isEmpty {
            return try await client.clearRoomMemory(
                roomID: safeRoomID,
                request,
                roomScopeHash: roomScopeHash
            )
        }
        return try await client.clearMemory(request)
    }

    func pinMemory(
        client: SidecarAPIClient,
        roomID: String?,
        roomScopeHash: String?,
        request: MemoryPinRequest
    ) async throws -> MemoryPinResponse {
        let safeRoomID = (roomID ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        if !safeRoomID.isEmpty {
            return try await client.pinRoomMemory(
                roomID: safeRoomID,
                request,
                roomScopeHash: roomScopeHash
            )
        }
        return try await client.pinMemory(request)
    }

    func unpinMemory(
        client: SidecarAPIClient,
        roomID: String?,
        roomScopeHash: String?,
        memoryID: String
    ) async throws -> Bool {
        let safeRoomID = (roomID ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        if !safeRoomID.isEmpty {
            return try await client.unpinRoomMemory(
                roomID: safeRoomID,
                memoryID: memoryID,
                roomScopeHash: roomScopeHash
            )
        }
        return try await client.unpinMemory(memoryID: memoryID)
    }
}
