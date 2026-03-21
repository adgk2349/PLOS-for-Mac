import AppKit
import Combine
import CryptoKit
import Foundation

@MainActor
extension AppViewModel {
    func refreshMemoryState() async throws {
        let workspaceID = currentWorkspaceID()
        let snapshot = try await performWithSidecarRetry { client in
            try await memoryServiceAdapter.fetchMemoryState(
                client: client,
                sessionID: activeConversationID,
                workspaceID: workspaceID,
                intent: selectedMode.rawValue.lowercased(),
                relatedFileIDs: citations.map(\.doc_id)
            )
        }

        sessionMemoryItems = snapshot.session
        workspaceMemoryItems = snapshot.workspace
        preferenceMemoryItems = snapshot.preferences
        episodicMemoryItems = snapshot.episodic
        pinnedMemoryItems = snapshot.pinned
    }


    func clearMemory(scope: MemoryClearScope) async {
        do {
            let workspaceID = currentWorkspaceID()
            let workspaceScoped = scope == .workspace || scope == .episodic || scope == .inferredOnly
            _ = try await performWithSidecarRetry { client in
                try await memoryServiceAdapter.clearMemory(
                    client: client,
                    request: MemoryClearRequest(
                        scope: scope,
                        workspace_id: workspaceScoped ? workspaceID : nil,
                        session_id: scope == .session ? activeConversationID : nil
                    )
                )
            }
            try await refreshMemoryState()
        } catch {
            handleViewModelError(error)
        }
    }


    func pinMemory(memoryID: String?, title: String, content: String, workspaceScoped: Bool) async {
        do {
            let request = MemoryPinRequest(
                memory_id: memoryID,
                scope: workspaceScoped ? "workspace" : "global",
                workspace_id: workspaceScoped ? currentWorkspaceID() : nil,
                title: title,
                content: content
            )
            _ = try await performWithSidecarRetry { client in
                try await memoryServiceAdapter.pinMemory(client: client, request: request)
            }
            try await refreshMemoryState()
        } catch {
            handleViewModelError(error)
        }
    }


    func unpinMemory(memoryID: String) async {
        do {
            _ = try await performWithSidecarRetry { client in
                try await memoryServiceAdapter.unpinMemory(client: client, memoryID: memoryID)
            }
            try await refreshMemoryState()
        } catch {
            handleViewModelError(error)
        }
    }
}
