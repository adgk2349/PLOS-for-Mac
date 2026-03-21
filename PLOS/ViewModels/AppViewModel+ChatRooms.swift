import AppKit
import Combine
import CryptoKit
import Foundation

@MainActor
extension AppViewModel {
    @discardableResult
    func ensureActiveConversation() -> String {
        if
            !selectedChatRoomID.isEmpty,
            chatRooms.contains(where: { $0.id == selectedChatRoomID })
        {
            return selectedChatRoomID
        }
        if let first = chatRooms.first(where: { !$0.isArchived }) ?? chatRooms.first {
            selectedChatRoomID = first.id
            return first.id
        }
        let created = chatRoomService.makeDefaultRoom()
        chatRooms = [created]
        selectedChatRoomID = created.id
        chatMessages = []
        citations = []
        persistChatRooms()
        return created.id
    }

    var currentChatRoomTitle: String {
        chatRooms.first(where: { $0.id == activeConversationID })?.title ?? "새 채팅"
    }

    var inboxChatRooms: [ChatRoom] {
        chatRooms
            .filter { !$0.isArchived }
            .sorted { $0.updatedAt > $1.updatedAt }
    }

    var archivedChatRooms: [ChatRoom] {
        chatRooms
            .filter(\.isArchived)
            .sorted { lhs, rhs in
                let l = lhs.archivedAt ?? lhs.updatedAt
                let r = rhs.archivedAt ?? rhs.updatedAt
                return l > r
            }
    }

    var activeRoomIndex: Int? {
        chatRooms.firstIndex(where: { $0.id == activeConversationID })
    }

    var activeLatestQueryForDeepAnalysis: String? {
        guard let idx = activeRoomIndex else { return nil }
        return chatRooms[idx].latestQueryForDeepAnalysis
    }

    var currentPrivacyBadge: String {
        switch privacyMode {
        case .localOnly:
            return "Local Only"
        case .hybrid:
            return "Hybrid"
        case .confirmBeforeExternal:
            return "Confirm"
        }
    }

    var activeModelPath: String {
        switch localEngine {
        case .mlx:
            return mlxModelPath
        case .llamaCPP:
            return llamaModelPath
        }
    }

    var installedModelsSorted: [ModelListItem] {
        var deduped: [String: ModelListItem] = [:]
        for model in availableModels {
            guard Self.isDisplayableModelArtifact(model) else { continue }
            let selectionPath = normalizedSelectionPath(for: model)
            guard !selectionPath.isEmpty else { continue }

            let existing = deduped[selectionPath]
            let chosenDate = max(existing?.modified_at ?? .distantPast, model.modified_at)
            let chosenSize = max(existing?.size_bytes ?? 0, model.size_bytes)
            let displayName = URL(fileURLWithPath: selectionPath).lastPathComponent
            deduped[selectionPath] = ModelListItem(
                file_name: displayName,
                path: selectionPath,
                engine: model.engine,
                size_bytes: chosenSize,
                modified_at: chosenDate
            )
        }

        var output = Array(deduped.values).sorted { $0.modified_at > $1.modified_at }
        let currentPath = activeModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if !currentPath.isEmpty, !output.contains(where: { $0.engine == localEngine && Self.samePath($0.path, currentPath) }) {
            output.insert(
                ModelListItem(
                    file_name: URL(fileURLWithPath: currentPath).lastPathComponent,
                    path: URL(fileURLWithPath: currentPath).standardizedFileURL.path,
                    engine: localEngine,
                    size_bytes: 0,
                    modified_at: .distantPast
                ),
                at: 0
            )
        }
        return output
    }

    var activeModelDisplayName: String {
        let currentPath = activeModelPath.trimmingCharacters(in: .whitespacesAndNewlines)
        if currentPath.isEmpty {
            return "모델 선택"
        }
        if let matched = installedModelsSorted.first(where: { $0.engine == localEngine && Self.samePath($0.path, currentPath) }) {
            return matched.file_name
        }
        return URL(fileURLWithPath: currentPath).lastPathComponent
    }


    func createChatRoom() {
        let room = chatRoomService.makeDefaultRoom()
        chatRooms.insert(room, at: 0)
        selectedChatRoomID = room.id
        chatMessages = []
        citations = []
        highlightedCitationPath = nil
        persistChatRooms()
        Task {
            do {
                try await refreshMemoryState()
            } catch {
                if !isEndpointNotFound(error) {
                    handleViewModelError(error)
                }
            }
        }
    }


    func selectChatRoom(_ roomID: String) {
        guard let room = chatRooms.first(where: { $0.id == roomID }) else {
            return
        }
        selectedChatRoomID = room.id
        chatMessages = room.messages
        citations = room.citations
        highlightedCitationPath = nil
        UserDefaults.standard.set(room.id, forKey: UDKey.activeChatRoom)
        Task {
            do {
                try await refreshMemoryState()
            } catch {
                if !isEndpointNotFound(error) {
                    handleViewModelError(error)
                }
            }
        }
    }


    func deleteChatRoom(_ roomID: String) {
        chatRooms.removeAll { $0.id == roomID }
        if chatRooms.isEmpty {
            let created = chatRoomService.makeDefaultRoom()
            chatRooms = [created]
        }
        if selectedChatRoomID == roomID {
            let next = chatRooms.first(where: { !$0.isArchived }) ?? chatRooms.first
            selectedChatRoomID = next?.id ?? ""
            chatMessages = next?.messages ?? []
            citations = next?.citations ?? []
        }
        persistChatRooms()
        Task {
            do {
                try await refreshMemoryState()
            } catch {
                if !isEndpointNotFound(error) {
                    handleViewModelError(error)
                }
            }
        }
    }


    func archiveChatRoom(_ roomID: String) {
        guard let index = chatRooms.firstIndex(where: { $0.id == roomID }) else {
            return
        }
        guard !chatRooms[index].isArchived else {
            return
        }
        chatRooms[index].archivedAt = Date()
        chatRooms[index].updatedAt = Date()

        if selectedChatRoomID == roomID {
            if let next = chatRooms.first(where: { !$0.isArchived && $0.id != roomID }) {
                selectedChatRoomID = next.id
                chatMessages = next.messages
                citations = next.citations
            } else {
                let created = chatRoomService.makeDefaultRoom()
                chatRooms.insert(created, at: 0)
                selectedChatRoomID = created.id
                chatMessages = []
                citations = []
            }
        }
        persistChatRooms()
    }


    func unarchiveChatRoom(_ roomID: String, selectAfterRestore: Bool = false) {
        guard let index = chatRooms.firstIndex(where: { $0.id == roomID }) else {
            return
        }
        chatRooms[index].archivedAt = nil
        chatRooms[index].updatedAt = Date()
        chatRooms.sort { $0.updatedAt > $1.updatedAt }
        if selectAfterRestore {
            if let restored = chatRooms.first(where: { $0.id == roomID }) {
                selectedChatRoomID = restored.id
                chatMessages = restored.messages
                citations = restored.citations
            }
        }
        persistChatRooms()
    }


    func loadChatRooms() {
        let snapshot = chatRoomService.loadChatRooms(
            defaults: .standard,
            roomsKey: UDKey.chatRooms,
            activeRoomKey: UDKey.activeChatRoom
        )
        chatRooms = snapshot.rooms
        selectedChatRoomID = snapshot.selectedRoomID
        chatMessages = snapshot.messages
        citations = snapshot.citations
    }


    func selectFirstInboxRoomIfNeeded() {
        guard let selected = chatRooms.first(where: { $0.id == selectedChatRoomID }) else {
            if let firstInbox = chatRooms.first(where: { !$0.isArchived }) {
                selectChatRoom(firstInbox.id)
            }
            return
        }
        if selected.isArchived, let firstInbox = chatRooms.first(where: { !$0.isArchived }) {
            selectChatRoom(firstInbox.id)
        }
    }


    func persistChatRooms() {
        chatRoomService.persistChatRooms(
            chatRooms,
            selectedRoomID: selectedChatRoomID,
            defaults: .standard,
            roomsKey: UDKey.chatRooms,
            activeRoomKey: UDKey.activeChatRoom
        )
    }


    func appendChatMessage(_ message: ChatMessage) {
        chatMessages.append(message)
        syncActiveRoom(messages: chatMessages)
    }


    func syncActiveRoom(
        messages: [ChatMessage]? = nil,
        citations: [Citation]? = nil,
        latestQueryForDeepAnalysis: String? = nil
    ) {
        guard let index = activeRoomIndex else {
            return
        }
        if let messages {
            chatRooms[index].messages = messages
        }
        if let citations {
            chatRooms[index].citations = citations
        }
        if let latestQueryForDeepAnalysis {
            chatRooms[index].latestQueryForDeepAnalysis = latestQueryForDeepAnalysis
        }
        chatRooms[index].updatedAt = Date()
        if chatRooms[index].title == "새 채팅" {
            if let firstUser = chatRooms[index].messages.first(where: { $0.source == .user })?.text {
                let generated = summarizeChatRoomTitle(from: firstUser)
                if generated != "새 채팅" {
                    chatRooms[index].title = generated
                }
            }
        }
        // Keep newest rooms at top like GPT-style history list.
        let activeID = chatRooms[index].id
        chatRooms.sort { $0.updatedAt > $1.updatedAt }
        if selectedChatRoomID != activeID {
            selectedChatRoomID = activeID
        }
        persistChatRooms()
    }


    func summarizeChatRoomTitle(from firstUserText: String) -> String {
        chatRoomService.summarizeChatRoomTitle(from: firstUserText)
    }
}
