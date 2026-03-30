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
            chatMessages = first.messages
            citations = first.citations
            resetTransientGenerationUIState()
            return first.id
        }
        let created = chatRoomService.makeDefaultRoom()
        chatRooms = [created]
        selectedChatRoomID = created.id
        chatMessages = []
        citations = []
        resetTransientGenerationUIState()
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
        case .externalAllowed:
            return "External"
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
        resetTransientGenerationUIState()
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
        resetTransientGenerationUIState()
        highlightedCitationPath = nil
        UserDefaults.standard.set(room.id, forKey: UDKey.activeChatRoom)
        Task {
            do {
                try await refreshMemoryState()
                await refreshRoomStorageStatus(for: room.id)
            } catch {
                if !isEndpointNotFound(error) {
                    handleViewModelError(error)
                }
            }
        }
    }


    func deleteChatRoom(_ roomID: String) {
        chatRooms.removeAll { $0.id == roomID }
        roomStorageStatusByRoomID.removeValue(forKey: roomID)
        roomIndexStateByRoomID.removeValue(forKey: roomID)
        roomIndexProgressByRoomID.removeValue(forKey: roomID)
        stopRoomIndexPolling(for: roomID)
        if chatRooms.isEmpty {
            let created = chatRoomService.makeDefaultRoom()
            chatRooms = [created]
        }
        if selectedChatRoomID == roomID {
            let next = chatRooms.first(where: { !$0.isArchived }) ?? chatRooms.first
            selectedChatRoomID = next?.id ?? ""
            chatMessages = next?.messages ?? []
            citations = next?.citations ?? []
            resetTransientGenerationUIState()
        }
        persistChatRooms()
        Task {
            await deleteRoomStorageIfExists(roomID)
        }
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
            resetTransientGenerationUIState()
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
                resetTransientGenerationUIState()
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
        resetTransientGenerationUIState()
        Task {
            await refreshRoomStorageStatus(for: activeConversationID)
        }
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

    @discardableResult
    func mutateChatMessage(id: UUID, mutate: (inout ChatMessage) -> Void) -> Bool {
        guard let index = chatMessages.firstIndex(where: { $0.id == id }) else {
            return false
        }
        mutate(&chatMessages[index])
        syncActiveRoom(messages: chatMessages)
        return true
    }

    @discardableResult
    func upsertStreamingLocalMessage(messageID: UUID?, delta: String, timestamp: Date = Date()) -> UUID {
        if let existingID = messageID, mutateChatMessage(id: existingID, mutate: { message in
            let original = message.text ?? ""
            message.text = (original + delta).precomposedStringWithCanonicalMapping
            message.isStreaming = true
        }) {
            return existingID
        }
        var message = ChatMessage(source: .local, text: delta, timestamp: timestamp)
        message.isStreaming = true
        appendChatMessage(message)
        return message.id
    }

    @discardableResult
    func finalizeStreamingLocalMessage(
        messageID: UUID?,
        response: ComposedChatResponseV2?,
        finalText: String?
    ) -> UUID? {
        guard let messageID else { return nil }
        let updated = mutateChatMessage(id: messageID) { message in
            if let response {
                var merged = ChatMessage(id: message.id, localV2: response, timestamp: message.timestamp)
                if let finalText, !finalText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    merged.text = finalText.precomposedStringWithCanonicalMapping
                    merged.lead = nil
                    merged.resultSummary = nil
                }
                merged.isStreaming = false
                message = merged
            } else if let finalText {
                message.text = finalText.precomposedStringWithCanonicalMapping
                message.isStreaming = false
            } else {
                message.isStreaming = false
            }
        }
        return updated ? messageID : nil
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
        if chatRoomService.shouldAutoRetitle(chatRooms[index].title) {
            let generated = summarizeChatRoomTitle(from: chatRooms[index].messages)
            if generated != "새 채팅" {
                chatRooms[index].title = generated
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

    func workspaceScopeForRoom(_ roomID: String?) -> (includedPaths: [String], excludedPaths: [String]) {
        guard
            let roomID,
            let room = chatRooms.first(where: { $0.id == roomID })
        else {
            return (includedFolderURLs.map(\.path), excludedPaths)
        }
        let included = room.includedFolderURLs?.map(\.path) ?? includedFolderURLs.map(\.path)
        let excluded = room.excludedPaths ?? excludedPaths
        return (included, excluded)
    }

    func requestScopeForRoom(_ roomID: String?) -> (includedPaths: [String], excludedPaths: [String]) {
        let scope = workspaceScopeForRoom(roomID)
        guard
            scope.includedPaths.isEmpty,
            let roomID,
            let index = chatRooms.firstIndex(where: { $0.id == roomID })
        else {
            return scope
        }
        let fallbackIncluded = chatRooms[index].lastResolvedIncludedPaths ?? []
        let fallbackExcluded = chatRooms[index].lastResolvedExcludedPaths ?? []
        if fallbackIncluded.isEmpty {
            return scope
        }
        return (fallbackIncluded, fallbackExcluded)
    }

    func requestScopeHashForRoom(_ roomID: String?) -> String? {
        guard
            let roomID,
            let room = chatRooms.first(where: { $0.id == roomID })
        else {
            return nil
        }
        let hash = room.lastResolvedRoomScopeHash?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
        return hash.isEmpty ? nil : hash
    }

    func rememberRequestScopeForRoom(_ roomID: String?, includedPaths: [String], excludedPaths: [String]) {
        guard
            let roomID,
            let index = chatRooms.firstIndex(where: { $0.id == roomID })
        else {
            return
        }
        let normalizedIncluded = includedPaths.map { URL(fileURLWithPath: $0).standardizedFileURL.path }
        let normalizedExcluded = excludedPaths.map { URL(fileURLWithPath: $0).standardizedFileURL.path }
        if chatRooms[index].lastResolvedIncludedPaths == normalizedIncluded &&
            chatRooms[index].lastResolvedExcludedPaths == normalizedExcluded {
            return
        }
        chatRooms[index].lastResolvedIncludedPaths = normalizedIncluded
        chatRooms[index].lastResolvedExcludedPaths = normalizedExcluded
        chatRooms[index].updatedAt = Date()
        persistChatRooms()
    }

    func rememberRoomRoutingMetadata(_ roomID: String?, metadata: [String: JSONValue]?) {
        guard
            let roomID,
            let index = chatRooms.firstIndex(where: { $0.id == roomID }),
            let metadata
        else {
            return
        }
        let scopeHash = metadata["room_scope_hash"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines)
        let storageID = metadata["room_storage_id"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedScope = (scopeHash ?? "").isEmpty ? nil : scopeHash
        let normalizedStorage = (storageID ?? "").isEmpty ? nil : storageID
        if chatRooms[index].lastResolvedRoomScopeHash == normalizedScope &&
            chatRooms[index].lastResolvedRoomStorageID == normalizedStorage {
            return
        }
        chatRooms[index].lastResolvedRoomScopeHash = normalizedScope
        chatRooms[index].lastResolvedRoomStorageID = normalizedStorage
        chatRooms[index].updatedAt = Date()
        persistChatRooms()
    }

    func roomUsesWorkspaceOverride(_ roomID: String) -> Bool {
        guard let room = chatRooms.first(where: { $0.id == roomID }) else {
            return false
        }
        return room.includedFolderURLs != nil || room.excludedPaths != nil
    }

    func addIncludedFolderToRoom(_ roomID: String) {
        guard let index = chatRooms.firstIndex(where: { $0.id == roomID }) else {
            return
        }
        let panel = NSOpenPanel()
        panel.canChooseFiles = false
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        panel.prompt = "추가"
        guard panel.runModal() == .OK else {
            return
        }
        let selectedURLs = panel.urls.map { $0.standardizedFileURL }
        guard !selectedURLs.isEmpty else {
            return
        }
        var existing = chatRooms[index].includedFolderURLs ?? []
        var existingPaths = Set(existing.map(\.path))
        for url in selectedURLs where !existingPaths.contains(url.path) {
            existing.append(url)
            existingPaths.insert(url.path)
        }
        chatRooms[index].includedFolderURLs = existing
        chatRooms[index].updatedAt = Date()
        persistChatRooms()
        Task {
            await triggerRoomStorageReindex(roomID)
        }
    }

    func addExcludedPathToRoom(_ roomID: String) {
        guard let index = chatRooms.firstIndex(where: { $0.id == roomID }) else {
            return
        }
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = true
        panel.allowsMultipleSelection = true
        panel.prompt = "제외 추가"
        guard panel.runModal() == .OK else {
            return
        }

        let selectedPaths = panel.urls.map { $0.standardizedFileURL.path }
        guard !selectedPaths.isEmpty else {
            return
        }

        var existing = chatRooms[index].excludedPaths ?? []
        for path in selectedPaths where !existing.contains(where: { Self.samePath($0, path) }) {
            existing.append(path)
        }

        chatRooms[index].excludedPaths = existing.isEmpty ? nil : existing
        chatRooms[index].updatedAt = Date()
        persistChatRooms()
        Task {
            await triggerRoomStorageReindex(roomID)
        }
    }

    func removeExcludedPathFromRoom(_ roomID: String, path: String) {
        guard let index = chatRooms.firstIndex(where: { $0.id == roomID }) else {
            return
        }
        var excluded = chatRooms[index].excludedPaths ?? []
        let previousCount = excluded.count
        excluded.removeAll { Self.samePath($0, path) }
        guard excluded.count != previousCount else {
            return
        }
        chatRooms[index].excludedPaths = excluded.isEmpty ? nil : excluded
        chatRooms[index].updatedAt = Date()
        persistChatRooms()
        Task {
            await triggerRoomStorageReindex(roomID)
        }
    }

    func clearExcludedPathsForRoom(_ roomID: String) {
        guard let index = chatRooms.firstIndex(where: { $0.id == roomID }) else {
            return
        }
        guard chatRooms[index].excludedPaths != nil else {
            return
        }
        chatRooms[index].excludedPaths = nil
        chatRooms[index].updatedAt = Date()
        persistChatRooms()
        Task {
            await triggerRoomStorageReindex(roomID)
        }
    }

    func clearRoomWorkspaceOverride(_ roomID: String) {
        guard let index = chatRooms.firstIndex(where: { $0.id == roomID }) else {
            return
        }
        chatRooms[index].includedFolderURLs = nil
        chatRooms[index].excludedPaths = nil
        chatRooms[index].updatedAt = Date()
        persistChatRooms()
        Task {
            await triggerRoomStorageReindex(roomID)
        }
    }

    func refreshRoomStorageStatus(for roomID: String, managePolling: Bool = true) async {
        let safeRoomID = roomID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !safeRoomID.isEmpty else { return }
        do {
            let status = try await performWithSidecarRetry { client in
                try await client.getRoomStorageStatus(roomID: safeRoomID)
            }
            roomStorageStatusByRoomID[safeRoomID] = status
            let firstVariant = status.variants.first
            let roomState = firstVariant?.room_index_state ?? "idle"
            roomIndexStateByRoomID[safeRoomID] = roomState
            if let progress = firstVariant?.index_progress {
                roomIndexProgressByRoomID[safeRoomID] = min(max(progress, 0.0), 1.0)
            } else if roomState == "ready" {
                roomIndexProgressByRoomID[safeRoomID] = 1.0
            } else if roomState == "idle" {
                roomIndexProgressByRoomID.removeValue(forKey: safeRoomID)
            }
            if managePolling {
                syncRoomIndexPolling(for: safeRoomID, roomState: roomState)
            }
        } catch {
            if !isEndpointNotFound(error) {
                handleViewModelError(error)
            }
        }
    }

    func triggerRoomStorageReindex(_ roomID: String) async {
        let safeRoomID = roomID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !safeRoomID.isEmpty else { return }
        let scope = workspaceScopeForRoom(safeRoomID)
        let included = scope.includedPaths
        guard !included.isEmpty else { return }
        let excluded = scope.excludedPaths
        do {
            _ = try await performWithSidecarRetry { client in
                try await client.reindexRoomStorage(
                    roomID: safeRoomID,
                    scope: "full",
                    includedPaths: included,
                    excludedPaths: excluded
                )
            }
            await refreshRoomStorageStatus(for: safeRoomID)
        } catch {
            if !isEndpointNotFound(error) {
                handleViewModelError(error)
            }
        }
    }

    func deleteRoomStorageIfExists(_ roomID: String) async {
        let safeRoomID = roomID.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !safeRoomID.isEmpty else { return }
        do {
            _ = try await performWithSidecarRetry { client in
                try await client.deleteRoomStorage(roomID: safeRoomID)
            }
        } catch {
            if !isEndpointNotFound(error) {
                handleViewModelError(error)
            }
        }
    }


    func summarizeChatRoomTitle(from firstUserText: String) -> String {
        chatRoomService.summarizeChatRoomTitle(from: firstUserText)
    }

    func summarizeChatRoomTitle(from messages: [ChatMessage]) -> String {
        chatRoomService.summarizeChatRoomTitle(from: messages)
    }

    private func resetTransientGenerationUIState() {
        activeGeneratingMessageID = nil
        liveThinkingTraceEvents = []
        isGeneratingChatResponse = false
    }

    private func syncRoomIndexPolling(for roomID: String, roomState: String) {
        if roomState == "indexing" {
            startRoomIndexPolling(for: roomID)
        } else {
            stopRoomIndexPolling(for: roomID)
        }
    }

    private func startRoomIndexPolling(for roomID: String) {
        if roomIndexPollingTasks[roomID] != nil {
            return
        }
        roomIndexPollingTasks[roomID] = Task { [weak self] in
            guard let self else { return }
            while !Task.isCancelled {
                try? await Task.sleep(nanoseconds: 1_200_000_000)
                if Task.isCancelled { break }
                await self.refreshRoomStorageStatus(for: roomID, managePolling: false)
                let state = self.roomIndexStateByRoomID[roomID] ?? "idle"
                if state != "indexing" {
                    break
                }
            }
            _ = await MainActor.run { [weak self] in
                self?.roomIndexPollingTasks.removeValue(forKey: roomID)
            }
        }
    }

    private func stopRoomIndexPolling(for roomID: String) {
        roomIndexPollingTasks[roomID]?.cancel()
        roomIndexPollingTasks.removeValue(forKey: roomID)
    }
}
