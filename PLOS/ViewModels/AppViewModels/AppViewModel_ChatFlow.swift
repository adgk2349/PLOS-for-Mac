import AppKit
import Combine
import CryptoKit
import Foundation

@MainActor
extension AppViewModel {
    func askLocal() async {
        let query = chatFlowService.normalizeQuery(inputQuery)
        guard !query.isEmpty else { return }
        inputQuery = ""
        await askLocal(query: query, appendUserMessage: true)
    }


    func deepAnalyzeTapped() {
        if privacyMode == .localOnly {
            lastError = "완전 로컬 모드에서는 외부 호출이 비활성화됩니다."
            return
        }
        if privacyMode == .hybrid, !hybridWebSearchEnabled {
            lastError = "하이브리드 모드에서 웹검색(인터넷 경로)이 꺼져 있어 외부 호출을 사용할 수 없습니다."
            return
        }
        if privacyMode == .confirmBeforeExternal {
            needsExternalConfirmation = true
            return
        }
        Task {
            await performDeepAnalysis(userConfirmed: true)
        }
    }


    func confirmExternalCall() {
        if let pending = pendingExternalDirectQuery {
            pendingExternalDirectQuery = nil
            Task { await performDeepAnalysis(userConfirmed: true, queryOverride: pending) }
        } else {
            Task { await performDeepAnalysis(userConfirmed: true) }
        }
    }


    func cancelExternalCall() {
        pendingExternalDirectQuery = nil
    }


    func setChatResponseRoute(_ route: ChatResponseRoute) {
        chatResponseRoute = route
        switch route {
        case .localOnly:
            privacyMode = .localOnly
        case .hybrid:
            if privacyMode == .localOnly {
                privacyMode = .hybrid
            }
        case .apiOnly:
            if privacyMode == .localOnly {
                privacyMode = .hybrid
            }
        }
        appPreferencesStore.persistChatResponseRoute(route, key: UDKey.chatResponseRoute)
        Task { await saveSettingsAndWorkspace() }
    }


    func applyQuickInferencePreset(_ preset: QuickInferencePreset) {
        quickInferencePreset = preset
        startupProfile = preset.startupProfile
        persistLocalModelPreferenceSnapshot()
        Task { await saveSettingsAndWorkspace() }
    }

    var currentChatTranscript: String {
        chatMessages
            .map { message in
                switch message.source {
                case .user:
                    return "You: \((message.text ?? "").precomposedStringWithCanonicalMapping)"
                case .local:
                    let lead = (message.lead ?? "").precomposedStringWithCanonicalMapping
                    let summary = (message.resultSummary ?? "").precomposedStringWithCanonicalMapping
                    return "Local AI: \(lead)\n\(summary)".trimmingCharacters(in: .whitespacesAndNewlines)
                case .external:
                    return "External: \((message.text ?? "").precomposedStringWithCanonicalMapping)"
                }
            }
            .joined(separator: "\n\n")
    }


    func copyCurrentChatTranscriptToClipboard() {
        let transcript = currentChatTranscript.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !transcript.isEmpty else {
            lastError = "복사할 대화 내용이 없습니다."
            return
        }
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(transcript, forType: .string)
    }


    func attachFileIntoComposer() {
        let panel = NSOpenPanel()
        panel.canChooseFiles = true
        panel.canChooseDirectories = false
        panel.allowsMultipleSelection = false
        panel.prompt = "첨부"
        guard panel.runModal() == .OK, let url = panel.url else {
            return
        }
        let token = "첨부 파일: \(url.path)"
        if inputQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            inputQuery = token
        } else {
            inputQuery += "\n\(token)"
        }
    }


    func startSystemDictation() {
        let selector = NSSelectorFromString("startDictation:")
        if !NSApp.sendAction(selector, to: nil, from: nil) {
            lastError = "시스템 받아쓰기를 시작하지 못했습니다."
        }
    }


    func executeAction(_ action: SuggestedAction) async {
        if let path = action.payload["file_path"], !path.isEmpty {
            highlightedCitationPath = path
        }
        switch action.execution_mode {
        case .promptInjection:
            await executePromptInjectionAction(action)
        case .system:
            await executeSystemAction(action)
        }
    }


    func confirmPendingSystemAction() {
        guard let action = pendingSystemAction else {
            return
        }
        pendingSystemAction = nil
        if actionPermissionMode == .askPerAction {
            approvedSystemActionKinds.insert(action.kind.rawValue)
            persistApprovedSystemActionKinds()
        }
        performSystemAction(action)
    }


    func cancelPendingSystemAction() {
        pendingSystemAction = nil
    }


    func performDeepAnalysis(userConfirmed: Bool, queryOverride: String? = nil) async {
        let targetQuery = queryOverride?.precomposedStringWithCanonicalMapping.trimmingCharacters(in: .whitespacesAndNewlines)
        let query = (targetQuery?.isEmpty == false ? targetQuery : activeLatestQueryForDeepAnalysis)
        guard let query else {
            lastError = "먼저 로컬 질문을 실행해 주세요."
            return
        }

        isBusy = true
        isGeneratingChatResponse = true
        activeGeneratingMessageID = nil
        liveThinkingTraceEvents = []
        defer {
            activeGeneratingMessageID = nil
            liveThinkingTraceEvents = []
            isGeneratingChatResponse = false
            isBusy = false
        }

        do {
            let response = try await performWithSidecarRetry { client in
                try await client.deepAnalysis(
                    DeepAnalysisRequest(
                        query: query,
                        mode: selectedMode,
                        provider: selectedProvider,
                        selected_citations: citations,
                        user_confirmed: userConfirmed
                    )
                )
            }
            appendChatMessage(ChatMessage(source: .external, text: response.answer, timestamp: Date()))
            try await refreshRemoteState()
        } catch {
            handleViewModelError(error)
        }
    }


    func askLocal(query: String, appendUserMessage: Bool) async {
        let trimmed = chatFlowService.normalizeQuery(query)
        guard !trimmed.isEmpty else { return }

        isBusy = true
        isGeneratingChatResponse = true
        activeGeneratingMessageID = nil
        liveThinkingTraceEvents = []
        defer {
            activeGeneratingMessageID = nil
            liveThinkingTraceEvents = []
            isGeneratingChatResponse = false
            isBusy = false
        }

        if appendUserMessage {
            appendChatMessage(ChatMessage(source: .user, text: trimmed, timestamp: Date()))
        }
        citations = []
        highlightedCitationPath = nil
        syncActiveRoom(citations: citations)

        if chatResponseRoute == .apiOnly {
            await askExternalDirect(query: trimmed)
            return
        }

        do {
            var handled = false
            try await performWithSidecarRetry { client in
                _ = try await prepareSelectedRuntime(using: client)
                let workspaceScope = requestScopeForRoom(activeConversationID)
                rememberRequestScopeForRoom(
                    activeConversationID,
                    includedPaths: workspaceScope.includedPaths,
                    excludedPaths: workspaceScope.excludedPaths
                )
                let requestV2 = LocalChatRequestV2(
                    query: trimmed,
                    mode: selectedMode,
                    conversation_id: activeConversationID,
                    session_id: activeConversationID,
                    top_k: nil,
                    filters: currentChatFilters(),
                    included_paths: workspaceScope.includedPaths,
                    excluded_paths: workspaceScope.excludedPaths,
                    behavior_overrides: nil
                )
                do {
                    let stream = try await client.localChatV2Stream(requestV2)
                    var streamedTextBuffer = ""
                    var didReceiveDone = false
                    var streamingMessageID: UUID?

                    for try await event in stream {
                        if event.type == "status" {
                            appendLiveThinkingTraceStatus(event.message)
                            continue
                        } else if event.type == "chunk" {
                            guard let chunk = event.text, !chunk.isEmpty else { continue }
                            let mergedChunk = mergedStreamingChunk(buffer: streamedTextBuffer, incomingChunk: chunk)
                            streamedTextBuffer += mergedChunk
                            streamingMessageID = upsertStreamingLocalMessage(
                                messageID: streamingMessageID,
                                delta: mergedChunk
                            )
                            activeGeneratingMessageID = streamingMessageID
                        } else if event.type == "done", let result = event.result {
                            didReceiveDone = true
                            citations = result.citations
                            rememberRoomRoutingMetadata(activeConversationID, metadata: result.metadata)
                            let bufferedRaw = streamedTextBuffer
                            let generatedText = (result.generated_text ?? "").precomposedStringWithCanonicalMapping
                            let streamSelection = selectFinalStreamText(
                                bufferedRaw: bufferedRaw,
                                generatedText: generatedText
                            )
                            let finalText = streamSelection.text
                            localRuntimeDetail = composeStreamRuntimeDetail(
                                baseDetail: result.runtime_detail,
                                finalTextSource: streamSelection.source,
                                bufferedCount: bufferedRaw.count,
                                generatedCount: generatedText.count
                            )
                            if let finalizedID = finalizeStreamingLocalMessage(
                                messageID: streamingMessageID,
                                response: result,
                                finalText: finalText
                            ) {
                                activeGeneratingMessageID = finalizedID
                            } else {
                                var finalMsg = ChatMessage(localV2: result, timestamp: Date())
                                if !finalText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                                    finalMsg.text = finalText.precomposedStringWithCanonicalMapping
                                    finalMsg.lead = nil
                                    finalMsg.resultSummary = nil
                                }
                                finalMsg.isStreaming = false
                                appendChatMessage(finalMsg)
                                activeGeneratingMessageID = finalMsg.id
                            }
                        } else if event.type == "error" {
                            let reason = (event.message ?? "알 수 없음").trimmingCharacters(in: .whitespacesAndNewlines)
                            appendLiveThinkingTraceStatus("오류: \(reason)")
                            let bufferedRaw = streamedTextBuffer
                            let buffered = bufferedRaw.trimmingCharacters(in: .whitespacesAndNewlines)
                            let composed = buffered.isEmpty
                                ? "[오류 발생: \(reason)]"
                                : "\(bufferedRaw)\n\n[오류 발생: \(reason)]"
                            if let finalizedID = finalizeStreamingLocalMessage(
                                messageID: streamingMessageID,
                                response: nil,
                                finalText: composed
                            ) {
                                activeGeneratingMessageID = finalizedID
                            } else {
                                let errorMsg = ChatMessage(source: .local, text: composed, timestamp: Date())
                                appendChatMessage(errorMsg)
                                activeGeneratingMessageID = errorMsg.id
                            }
                        }
                    }

                    if !didReceiveDone {
                        let bufferedRaw = streamedTextBuffer
                        let buffered = bufferedRaw.trimmingCharacters(in: .whitespacesAndNewlines)
                        if !buffered.isEmpty {
                            if let finalizedID = finalizeStreamingLocalMessage(
                                messageID: streamingMessageID,
                                response: nil,
                                finalText: bufferedRaw
                            ) {
                                activeGeneratingMessageID = finalizedID
                            } else {
                                let partialMsg = ChatMessage(source: .local, text: bufferedRaw, timestamp: Date())
                                appendChatMessage(partialMsg)
                                activeGeneratingMessageID = partialMsg.id
                            }
                        }
                    }

                    syncActiveRoom(citations: citations, latestQueryForDeepAnalysis: trimmed)
                    handled = true
                } catch {
                    #if DEBUG
                        let responseV2 = try await client.localChatV2(requestV2)
                        citations = responseV2.citations
                        rememberRoomRoutingMetadata(activeConversationID, metadata: responseV2.metadata)
                        if let runtimeDetail = responseV2.runtime_detail, !runtimeDetail.isEmpty {
                            localRuntimeDetail = runtimeDetail
                        }
                        appendChatMessage(ChatMessage(localV2: responseV2, timestamp: Date()))
                        syncActiveRoom(citations: citations, latestQueryForDeepAnalysis: trimmed)
                        handled = true
                    #else
                        throw error
                    #endif
                }
                return ()
            }
            if !handled {
                throw APIError(message: "로컬 채팅 응답을 처리하지 못했습니다.")
            }
            await refreshPostChatStateIfNeeded()
        } catch {
            handleViewModelError(error)
        }
    }

    private func selectFinalStreamText(bufferedRaw: String, generatedText: String) -> (text: String, source: String) {
        let bufferedTrimmed = bufferedRaw.trimmingCharacters(in: .whitespacesAndNewlines)
        let generatedTrimmed = generatedText.trimmingCharacters(in: .whitespacesAndNewlines)
        if bufferedTrimmed.isEmpty {
            return (generatedText, "generated_empty_buffer")
        }
        if generatedTrimmed.isEmpty {
            return (bufferedRaw, "buffered_generated_empty")
        }
        if generatedText.count > bufferedRaw.count, generatedText.hasPrefix(bufferedRaw) {
            return (generatedText, "generated_superstring")
        }
        let normalizedBuffered = bufferedTrimmed.replacingOccurrences(of: "\\s+", with: "", options: .regularExpression)
        let normalizedGenerated = generatedTrimmed.replacingOccurrences(of: "\\s+", with: "", options: .regularExpression)
        if !normalizedGenerated.isEmpty,
           normalizedGenerated.count >= normalizedBuffered.count,
           normalizedGenerated.hasPrefix(normalizedBuffered)
        {
            return (generatedText, "generated_superstring_ws")
        }
        return (bufferedRaw, "buffered")
    }

    private func mergedStreamingChunk(buffer: String, incomingChunk: String) -> String {
        guard !buffer.isEmpty else { return incomingChunk }
        guard let first = incomingChunk.first else { return incomingChunk }
        if isWhitespaceLike(first) || isPunctuationLike(first) {
            return incomingChunk
        }
        guard let last = buffer.last else { return incomingChunk }
        if isWhitespaceLike(last) || isPunctuationLike(last) {
            return incomingChunk
        }
        return " " + incomingChunk
    }

    private func isWhitespaceLike(_ ch: Character) -> Bool {
        ch.unicodeScalars.allSatisfy { CharacterSet.whitespacesAndNewlines.contains($0) }
    }

    private func isPunctuationLike(_ ch: Character) -> Bool {
        ch.unicodeScalars.allSatisfy { CharacterSet.punctuationCharacters.contains($0) }
    }

    private func composeStreamRuntimeDetail(
        baseDetail: String?,
        finalTextSource: String,
        bufferedCount: Int,
        generatedCount: Int
    ) -> String {
        let base = (baseDetail ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        let streamTrace = "stream_final=\(finalTextSource);buffered_chars=\(bufferedCount);generated_chars=\(generatedCount)"
        if base.isEmpty {
            return streamTrace
        }
        if base.contains("stream_final=") {
            return base
        }
        return "\(base);\(streamTrace)"
    }

    private func appendLiveThinkingTraceStatus(_ rawStatus: String?) {
        let message = (rawStatus ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !message.isEmpty else { return }
        let lowered = message.lowercased()
        if lowered == "room routing: room_scope_missing" ||
            lowered == "room routing: room_registry_missing" ||
            lowered == "room routing: room_id_missing"
        {
            return
        }
        let status: String
        if ["warning", "blocked", "error", "실패", "차단", "오류"].contains(where: { lowered.contains($0) }) {
            status = "warning"
        } else if ["retrieved", "done", "finished", "완료", "검색 완료"].contains(where: { lowered.contains($0) }) {
            status = "retrieved"
        } else if ["retrieving", "search", "fetch", "검색", "수집", "탐색", "web"].contains(where: { lowered.contains($0) }) {
            status = "retrieving"
        } else {
            status = "done"
        }
        let source = ["web", "search", "retriev", "검색", "수집"].contains(where: { lowered.contains($0) })
            ? "retrieval"
            : "pipeline"
        if let last = liveThinkingTraceEvents.last,
           last.status == status,
           last.message.caseInsensitiveCompare(message) == .orderedSame {
            return
        }
        liveThinkingTraceEvents.append(
            LiveThinkingTraceEvent(
                status: status,
                message: message,
                source: source,
                url: nil,
                at: ISO8601DateFormatter().string(from: Date())
            )
        )
        if liveThinkingTraceEvents.count > 24 {
            liveThinkingTraceEvents.removeFirst(liveThinkingTraceEvents.count - 24)
        }
    }

    func refreshPostChatStateIfNeeded(force: Bool = false) async {
        let now = Date()
        if !force, now.timeIntervalSince(lastPostChatStateRefreshAt) < postChatStateRefreshInterval {
            return
        }
        do {
            let status = try await performWithSidecarRetry { try await $0.getStatus() }
            statusSnapshot = status
            let failures = try await performWithSidecarRetry { try await $0.getFailures() }
            failureItems = failures.failures
            lastPostChatStateRefreshAt = now
        } catch {
            if !isEndpointNotFound(error), lastError == nil {
                lastError = "상태 동기화 실패: \(error.localizedDescription)"
            }
        }
    }


    func askExternalDirect(query: String) async {
        if privacyMode == .localOnly {
            lastError = "현재 응답 경로는 항상 API 호출이지만, 프라이버시 모드가 로컬 전용입니다."
            return
        }
        if privacyMode == .hybrid, !hybridWebSearchEnabled {
            lastError = "하이브리드 모드에서 웹검색(인터넷 경로)이 꺼져 있어 API 호출을 사용할 수 없습니다."
            return
        }
        if privacyMode == .confirmBeforeExternal {
            pendingExternalDirectQuery = query
            needsExternalConfirmation = true
            return
        }
        await performDeepAnalysis(userConfirmed: true, queryOverride: query)
    }


    func executePromptInjectionAction(_ action: SuggestedAction) async {
        guard let prompt = action.payload["prompt"]?.trimmingCharacters(in: .whitespacesAndNewlines), !prompt.isEmpty else {
            lastError = "액션 프롬프트가 비어 있어 실행할 수 없습니다."
            return
        }
        await recordActionMemoryEvent(action, summary: prompt)
        inputQuery = prompt
        await askLocal(query: prompt, appendUserMessage: true)
        inputQuery = ""
    }


    func executeSystemAction(_ action: SuggestedAction) async {
        switch actionPermissionMode {
        case .askEveryTime:
            pendingSystemAction = action
        case .askPerAction:
            if approvedSystemActionKinds.contains(action.kind.rawValue) {
                performSystemAction(action)
            } else {
                pendingSystemAction = action
            }
        }
    }


    func performSystemAction(_ action: SuggestedAction) {
        switch action.kind {
        case .openFile, .openSecond:
            guard let filePath = action.payload["file_path"], !filePath.isEmpty else {
                lastError = "열 파일 경로가 없습니다."
                return
            }
            let opened = NSWorkspace.shared.open(URL(fileURLWithPath: filePath))
            if !opened {
                lastError = "파일을 열지 못했습니다: \(filePath)"
            } else {
                Task {
                    await recordActionMemoryEvent(action, summary: "open file: \(filePath)")
                }
            }
        case .summarizeTop, .compareTop, .askFollowup, .showDiff, .createDraft, .showOtherCandidates, .makeShorter, .showPreviousCandidate:
            // These action kinds are handled via prompt injection path.
            break
        }
    }


    func recordActionMemoryEvent(_ action: SuggestedAction, summary: String) async {
        var metadata: [String: JSONValue] = [
            "action_kind": .string(action.kind.rawValue),
            "execution_mode": .string(action.execution_mode.rawValue),
        ]
        for (key, value) in action.payload {
            metadata[key] = .string(value)
        }
        do {
            try await writeMemoryEvent(
                eventType: .actionExecuted,
                summary: summary,
                relatedFileIDs: action.payload["file_path"].map { [$0] } ?? [],
                relatedActionIDs: [action.action_id],
                metadata: metadata,
                importance: 0.56
            )
        } catch {
            handleViewModelError(error)
        }
    }


    @discardableResult
    func writeMemoryEvent(
        eventType: MemoryEventType,
        summary: String,
        relatedFileIDs: [String],
        relatedActionIDs: [String],
        metadata: [String: JSONValue],
        importance: Double
    ) async throws -> MemoryEventResponse {
        try await performWithSidecarRetry { client in
            let payload = MemoryEventRequest(
                event_type: eventType,
                session_id: activeConversationID,
                workspace_id: currentWorkspaceID(),
                summary: summary,
                related_file_ids: relatedFileIDs,
                related_action_ids: relatedActionIDs,
                metadata_json: metadata,
                importance: importance
            )
            let safeRoomID = activeConversationID.trimmingCharacters(in: .whitespacesAndNewlines)
            if !safeRoomID.isEmpty {
                return try await client.writeRoomMemoryEvent(
                    roomID: safeRoomID,
                    payload: payload,
                    roomScopeHash: requestScopeHashForRoom(activeConversationID)
                )
            }
            return try await client.writeMemoryEvent(payload)
        }
    }


    func currentWorkspaceID() -> String {
        let scope = requestScopeForRoom(activeConversationID)
        return chatFlowService.workspaceID(
            includedFolderURLs: scope.includedPaths.map { URL(fileURLWithPath: $0) },
            excludedPaths: scope.excludedPaths
        )
    }
}
