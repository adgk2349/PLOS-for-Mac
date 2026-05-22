import AppKit
import Combine
import CryptoKit
import Foundation
import UniformTypeIdentifiers

@MainActor
extension AppViewModel {
    func submitOrStopLocalChatFromComposer() {
        if isGeneratingChatResponse {
            stopActiveLocalChatGeneration()
            return
        }
        guard isChatComposerEnabled else {
            lastError = "로컬 서버 준비 중입니다. 연결 완료 후 다시 시도해 주세요."
            return
        }
        guard !isBusy else { return }
        activeLocalChatTask?.cancel()
        activeLocalChatTask = Task { [weak self] in
            guard let self else { return }
            await self.askLocal()
            await MainActor.run {
                self.activeLocalChatTask = nil
            }
        }
    }

    func stopActiveLocalChatGeneration() {
        guard activeLocalChatTask != nil || activeGeneratingMessageID != nil || isGeneratingChatResponse else {
            return
        }
        activeLocalChatTask?.cancel()
        activeLocalChatTask = nil
        if let activeID = activeGeneratingMessageID {
            _ = finalizeStreamingLocalMessage(messageID: activeID, response: nil, finalText: nil)
        }
        flushStreamingRoomState(updateTimestamp: true, reorder: true, persist: true)
        activeGeneratingMessageID = nil
        liveThinkingTraceEvents = []
        isGeneratingChatResponse = false
        isBusy = false
    }

    func askLocal() async {
        let query = chatFlowService.normalizeQuery(inputQuery)
        let attachments = composerAttachments
        guard !query.isEmpty || !attachments.isEmpty else { return }
        inputQuery = ""
        composerAttachments = []
        await askLocal(query: query, attachments: attachments, appendUserMessage: true)
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

    func setRoleplayMode(_ enabled: Bool) {
        roleplayModeEnabled = enabled
        persistRoleplayPreference()
        if !enabled {
            roleplayPersonaHintByRoomID.removeValue(forKey: activeConversationID)
        }
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
        panel.allowsMultipleSelection = true
        panel.prompt = "첨부"
        guard panel.runModal() == .OK else {
            return
        }
        var existingPaths = Set(composerAttachments.map(\.filePath))
        var updates = composerAttachments
        for url in panel.urls {
            let filePath = url.standardizedFileURL.path
            guard !existingPaths.contains(filePath) else { continue }
            existingPaths.insert(filePath)
            let kind = attachmentKind(for: url)
            let mimeType = mimeTypeForAttachment(url)
            updates.append(
                ComposerAttachment(
                    id: UUID().uuidString,
                    kind: kind,
                    filePath: filePath,
                    fileName: url.lastPathComponent,
                    mimeType: mimeType
                )
            )
        }
        composerAttachments = updates
    }

    func removeComposerAttachment(_ id: String) {
        composerAttachments.removeAll { $0.id == id }
    }

    var canSubmitChatInput: Bool {
        !inputQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || !composerAttachments.isEmpty
    }

    var isChatComposerEnabled: Bool {
        isSidecarReadyForChat && sidecarRecoveryState != "recovering"
    }

    private func extractRoleplayPersonaHint(from query: String) -> String? {
        let normalized = query.precomposedStringWithCanonicalMapping.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !normalized.isEmpty else { return nil }
        let patterns = [
            #"(?:넌|너는|당신은|넌 이제|너는 이제)\s*(?:지금부터\s*)?(.{1,40}?)(?:이야|야|로\s*행동해|처럼\s*말해|역할극|코스프레)"#,
            #"(?:you are|act as|roleplay as)\s+(.{1,40}?)(?:[.!?]|$)"#,
            #"(?:\bI want you to be\b)\s+(.{1,40}?)(?:[.!?]|$)"#
        ]
        for pattern in patterns {
            guard let regex = try? NSRegularExpression(pattern: pattern, options: [.caseInsensitive]) else {
                continue
            }
            let range = NSRange(normalized.startIndex..<normalized.endIndex, in: normalized)
            guard let match = regex.firstMatch(in: normalized, options: [], range: range), match.numberOfRanges > 1 else {
                continue
            }
            guard let personaRange = Range(match.range(at: 1), in: normalized) else {
                continue
            }
            let persona = normalized[personaRange]
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .trimmingCharacters(in: CharacterSet(charactersIn: "\"'“”‘’.,!?"))
            if !persona.isEmpty {
                return String(persona.prefix(40))
            }
        }
        return nil
    }

    private func attachmentKind(for url: URL) -> ChatAttachmentKind {
        if let type = UTType(filenameExtension: url.pathExtension.lowercased()) {
            if type.conforms(to: .image) {
                return .image
            }
            if type.conforms(to: .audio) {
                return .audio
            }
        }
        return .file
    }

    private func mimeTypeForAttachment(_ url: URL) -> String? {
        guard let type = UTType(filenameExtension: url.pathExtension.lowercased()) else {
            return nil
        }
        return type.preferredMIMEType
    }

    private func buildChatAttachments(_ attachments: [ComposerAttachment]) -> [ChatAttachmentV2] {
        attachments.map { item in
            ChatAttachmentV2(
                id: item.id,
                kind: item.kind,
                file_path: item.filePath,
                file_name: item.fileName,
                mime_type: item.mimeType
            )
        }
    }

    private func userDisplayText(query: String, attachments: [ComposerAttachment]) -> String {
        let base = query.trimmingCharacters(in: .whitespacesAndNewlines)
        if attachments.isEmpty {
            return base
        }
        let lines = attachments.map { item in
            let label: String
            switch item.kind {
            case .file:
                label = "file"
            case .image:
                label = "image"
            case .audio:
                label = "audio"
            }
            return "첨부[\(label)]: \(item.fileName)"
        }
        if base.isEmpty {
            return lines.joined(separator: "\n")
        }
        return ([base] + lines).joined(separator: "\n")
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
            if error is CancellationError {
                lastError = nil
                return
            }
            handleViewModelError(error)
        }
    }


    func askLocal(query: String, appendUserMessage: Bool) async {
        await askLocal(query: query, attachments: [], appendUserMessage: appendUserMessage)
    }

    func askLocal(query: String, attachments: [ComposerAttachment], appendUserMessage: Bool) async {
        guard isChatComposerEnabled else {
            lastError = "로컬 서버 준비 중입니다. 연결 완료 후 다시 시도해 주세요."
            return
        }
        let trimmed = chatFlowService.normalizeQuery(query)
        guard !trimmed.isEmpty || !attachments.isEmpty else { return }

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
            let displayText = userDisplayText(query: trimmed, attachments: attachments)
            appendChatMessage(ChatMessage(source: .user, text: displayText, timestamp: Date()))
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
                let conversationID = activeConversationID
                let workspaceScope = requestScopeForRoom(conversationID)
                rememberRequestScopeForRoom(
                    conversationID,
                    includedPaths: workspaceScope.includedPaths,
                    excludedPaths: workspaceScope.excludedPaths
                )
                var roleplayPersona: String?
                if roleplayModeEnabled {
                    if let extracted = extractRoleplayPersonaHint(from: trimmed) {
                        roleplayPersonaHintByRoomID[conversationID] = extracted
                    }
                    let savedPersona = roleplayPersonaHintByRoomID[conversationID]?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
                    roleplayPersona = savedPersona.isEmpty ? nil : String(savedPersona.prefix(40))
                }
                let requestV2 = LocalChatRequestV2(
                    query: trimmed,
                    mode: selectedMode,
                    conversation_id: conversationID,
                    session_id: conversationID,
                    top_k: nil,
                    filters: currentChatFilters(),
                    included_paths: workspaceScope.includedPaths,
                    excluded_paths: workspaceScope.excludedPaths,
                    behavior_overrides: nil,
                    attachments: attachments.isEmpty ? nil : buildChatAttachments(attachments),
                    roleplay_mode: roleplayModeEnabled ? true : nil,
                    roleplay_persona: roleplayPersona
                )
                do {
                    let stream = try await client.localChatV2Stream(requestV2)
                    var streamedTextBuffer = ""
                    var didReceiveDone = false
                    var streamingMessageID: UUID?
                    var streamSanitizer = StreamReasoningSanitizer()
                    var capturedReasoningNotes: [String] = []
                    let reasoningAnchorID = UUID()
                    activeGeneratingMessageID = reasoningAnchorID

                    for try await event in stream {
                        if event.type == "status" {
                            appendLiveThinkingTraceStatus(event.message)
                            continue
                        } else if event.type == "chunk" {
                            guard let chunk = event.text, !chunk.isEmpty else { continue }
                            let routed = routeStreamChunk(chunk, sanitizer: &streamSanitizer)
                            if !routed.reasoningNotes.isEmpty {
                                capturedReasoningNotes.append(contentsOf: routed.reasoningNotes)
                                for note in routed.reasoningNotes {
                                    appendLiveThinkingTraceStatus(note)
                                }
                            }
                            if !routed.answerText.isEmpty {
                                streamedTextBuffer += routed.answerText
                                streamingMessageID = upsertStreamingLocalMessage(
                                    messageID: streamingMessageID,
                                    delta: routed.answerText
                                )
                                activeGeneratingMessageID = streamingMessageID
                            }
                        } else if event.type == "done", let result = event.result {
                            didReceiveDone = true
                            let generatedSplit = splitReasoningAndAnswer(from: result.generated_text ?? "")
                            let bufferedSplit = splitReasoningAndAnswer(from: streamedTextBuffer)
                            let mergedNotes = dedupeReasoningNotes(capturedReasoningNotes + generatedSplit.reasoningNotes + bufferedSplit.reasoningNotes)
                            var finalizedResult = result
                            if !mergedNotes.isEmpty {
                                finalizedResult.metadata = mergeReasoningTraceEvents(
                                    metadata: finalizedResult.metadata,
                                    reasoningNotes: mergedNotes
                                )
                            }
                            var metadata = finalizedResult.metadata ?? [:]
                            metadata["engine_recovery_attempt"] = .number(Double(sidecarRecoveryAttempt))
                            metadata["native_crash_detected"] = .bool(nativeCrashDetected)
                            finalizedResult.metadata = metadata

                            citations = finalizedResult.citations
                            rememberRoomRoutingMetadata(activeConversationID, metadata: finalizedResult.metadata)
                            let bufferedRaw = sanitizeFinalGeneratedText(bufferedSplit.answerText)
                            let generatedText = sanitizeFinalGeneratedText(generatedSplit.answerText)
                            let directGeneratedText = sanitizeFinalGeneratedText(result.generated_text ?? "")
                            let streamSelection = selectFinalStreamText(
                                bufferedRaw: bufferedRaw,
                                generatedText: generatedText
                            )
                            var finalText = preferredDisplayText(
                                primary: streamSelection.text,
                                fallbackLead: finalizedResult.lead,
                                fallbackSummary: finalizedResult.structured_result.summary
                            )
                            if !isMeaningfulAssistantText(finalText ?? "") {
                                if isMeaningfulAssistantText(directGeneratedText) {
                                    finalText = directGeneratedText
                                } else if isMeaningfulAssistantText(generatedText) {
                                    finalText = generatedText
                                } else if isMeaningfulAssistantText(bufferedRaw) {
                                    finalText = bufferedRaw
                                } else {
                                    finalText = emptyAssistantResponseFallbackText()
                                }
                            }
                            localRuntimeDetail = composeStreamRuntimeDetail(
                                baseDetail: finalizedResult.runtime_detail,
                                finalTextSource: streamSelection.source,
                                bufferedCount: bufferedRaw.count,
                                generatedCount: generatedText.count
                            )
                            if let finalizedID = finalizeStreamingLocalMessage(
                                messageID: streamingMessageID,
                                response: finalizedResult,
                                finalText: finalText
                            ) {
                                activeGeneratingMessageID = finalizedID
                            } else {
                                var finalMsg = ChatMessage(localV2: finalizedResult, timestamp: Date())
                                if let finalText, !finalText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
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
                            let bufferedSplit = splitReasoningAndAnswer(from: streamedTextBuffer)
                            let bufferedRaw = sanitizeFinalGeneratedText(bufferedSplit.answerText)
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
                        let bufferedSplit = splitReasoningAndAnswer(from: streamedTextBuffer)
                        let bufferedRaw = sanitizeFinalGeneratedText(bufferedSplit.answerText)
                        let buffered = bufferedRaw.trimmingCharacters(in: .whitespacesAndNewlines)
                        let fallbackText = buffered.isEmpty ? emptyAssistantResponseFallbackText() : bufferedRaw
                        if let finalizedID = finalizeStreamingLocalMessage(
                            messageID: streamingMessageID,
                            response: nil,
                            finalText: fallbackText
                        ) {
                            activeGeneratingMessageID = finalizedID
                        } else {
                            let partialMsg = ChatMessage(source: .local, text: fallbackText, timestamp: Date())
                            appendChatMessage(partialMsg)
                            activeGeneratingMessageID = partialMsg.id
                        }
                    }

                    syncActiveRoom(citations: citations, latestQueryForDeepAnalysis: trimmed)
                    handled = true
                } catch {
                    #if DEBUG
                        var responseV2 = try await client.localChatV2(requestV2)
                        var metadata = responseV2.metadata ?? [:]
                        metadata["engine_recovery_attempt"] = .number(Double(sidecarRecoveryAttempt))
                        metadata["native_crash_detected"] = .bool(nativeCrashDetected)
                        responseV2.metadata = metadata
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
        } catch is CancellationError {
            flushStreamingRoomState(updateTimestamp: true, reorder: true, persist: true)
            lastError = nil
        } catch {
            handleViewModelError(error)
        }
    }

    private func selectFinalStreamText(bufferedRaw: String, generatedText: String) -> (text: String, source: String) {
        let bufferedTrimmed = bufferedRaw.trimmingCharacters(in: .whitespacesAndNewlines)
        let generatedTrimmed = generatedText.trimmingCharacters(in: .whitespacesAndNewlines)
        if generatedTrimmed.isEmpty {
            return (bufferedRaw, "buffered_generated_empty")
        }
        if bufferedTrimmed.isEmpty {
            return (generatedText, "generated_empty_buffer")
        }
        let bufferedIncomplete = looksIncompleteAssistantText(bufferedTrimmed)
        let generatedIncomplete = looksIncompleteAssistantText(generatedTrimmed)
        if bufferedIncomplete && !generatedIncomplete {
            return (generatedText, "generated_preferred_buffer_incomplete")
        }
        if generatedIncomplete && !bufferedIncomplete {
            return (bufferedRaw, "buffered_preferred_generated_incomplete")
        }
        // When finalized payload is unexpectedly short, keep streamed buffer to avoid end-of-stream overwrite regressions.
        if shouldPreferBufferedStreamText(bufferedTrimmed: bufferedTrimmed, generatedTrimmed: generatedTrimmed) {
            return (bufferedRaw, "buffered_preferred_short_final")
        }
        return (generatedText, "generated_preferred")
    }

    private func shouldPreferBufferedStreamText(bufferedTrimmed: String, generatedTrimmed: String) -> Bool {
        let bufferedHasLeakMarkers = containsInternalLeakMarkers(bufferedTrimmed)
        let generatedHasLeakMarkers = containsInternalLeakMarkers(generatedTrimmed)
        if bufferedHasLeakMarkers && !generatedHasLeakMarkers {
            return false
        }
        if generatedHasLeakMarkers && !bufferedHasLeakMarkers {
            return true
        }
        if generatedTrimmed.count < 64 && bufferedTrimmed.count >= max(180, generatedTrimmed.count * 3) {
            return true
        }
        if bufferedTrimmed.contains("```"), !generatedTrimmed.contains("```"), bufferedTrimmed.count > generatedTrimmed.count + 80 {
            return true
        }
        let bufferedSentenceCount = bufferedTrimmed.split(whereSeparator: { ".!?。！？\n".contains($0) }).count
        let generatedSentenceCount = generatedTrimmed.split(whereSeparator: { ".!?。！？\n".contains($0) }).count
        if generatedSentenceCount <= 1 && bufferedSentenceCount >= 3 && bufferedTrimmed.count > generatedTrimmed.count + 120 {
            return true
        }
        return false
    }

    private func looksIncompleteAssistantText(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        if trimmed.range(of: #"(?im)(?:^|\n)\s*\d{1,2}[.)]\s*$"#, options: .regularExpression) != nil {
            return true
        }
        if trimmed.contains("```"), trimmed.components(separatedBy: "```").count % 2 == 0 {
            return true
        }
        if trimmed.range(of: #"[:;,(\[{`-]\s*$"#, options: .regularExpression) != nil {
            return true
        }
        if trimmed.count >= 120,
           trimmed.range(of: #"[.!?。！？]\s*$"#, options: .regularExpression) == nil
        {
            return true
        }
        return false
    }

    private func containsInternalLeakMarkers(_ text: String) -> Bool {
        let lowered = text.lowercased()
        if lowered.contains("\"hypotheses\"") || lowered.contains("input message:") {
            return true
        }
        if lowered.contains("&lt;channel&gt;") || lowered.contains("&lt;/channel&gt;") {
            return true
        }
        if lowered.contains("&lt;analysis&gt;") || lowered.contains("&lt;/analysis&gt;") {
            return true
        }
        if lowered.contains("&lt;final&gt;") || lowered.contains("&lt;/final&gt;") {
            return true
        }
        if lowered.contains("<tool_code>") || lowered.contains("</tool_code>") {
            return true
        }
        if lowered.contains("<searching>") || lowered.contains("<tool_running>") {
            return true
        }
        if lowered.contains("<action>") || lowered.contains("<final_answer>") {
            return true
        }
        if lowered.contains("<channel>") || lowered.contains("</channel>") {
            return true
        }
        if lowered.contains("<analysis>") || lowered.contains("</analysis>") {
            return true
        }
        if lowered.contains("<final>") || lowered.contains("</final>") {
            return true
        }
        if lowered.contains("return strict json only") {
            return true
        }
        if lowered.range(of: #"^\s*(?:&lt;/?[a-z_]+&gt;|</?[a-z_]+>)\.?\s*$"#, options: .regularExpression) != nil {
            return true
        }
        if lowered.contains("answer:") {
            let hasFileHint = lowered.contains(".txt)") || lowered.contains(".md)") || lowered.contains(".pdf)")
            if hasFileHint {
                return true
            }
        }
        return false
    }

    private func preferredDisplayText(primary: String, fallbackLead: String, fallbackSummary: String) -> String? {
        let sanitizedPrimary = sanitizeFinalGeneratedText(primary)
        if isMeaningfulAssistantText(sanitizedPrimary) {
            return sanitizedPrimary
        }
        let summary = sanitizeFinalGeneratedText(fallbackSummary)
        if isMeaningfulAssistantText(summary) {
            return nil
        }
        let lead = sanitizeFinalGeneratedText(fallbackLead)
        if isMeaningfulAssistantText(lead) {
            return nil
        }
        return sanitizedPrimary.isEmpty ? nil : sanitizedPrimary
    }

    private func isMeaningfulAssistantText(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return false }
        if containsInternalLeakMarkers(trimmed) {
            return false
        }
        let hasReadableLetter = trimmed.range(
            of: #"[A-Za-z0-9가-힣ぁ-ゖァ-ヺ一-龥]"#,
            options: .regularExpression
        ) != nil
        if !hasReadableLetter {
            return false
        }
        let compact = trimmed.lowercased().replacingOccurrences(of: "\\s+", with: "", options: .regularExpression)
        let placeholders: Set<String> = [
            "(이미지없음)",
            "[이미지없음]",
            "이미지없음",
            "(noimage)",
            "[noimage]",
            "noimage",
            "imagenotavailable",
            "(imagenotavailable)",
            "[imagenotavailable]",
            "imageunavailable",
            "(imageunavailable)",
            "[imageunavailable]",
        ]
        if placeholders.contains(compact) {
            return false
        }
        return true
    }

    private func emptyAssistantResponseFallbackText() -> String {
        L10n.tr(
            "chat.empty_response_fallback",
            language: appLanguage,
            fallbackKo: "응답이 중간에 비어 종료되었습니다. 한 번 더 시도해 주세요.",
            fallbackEn: "The response ended with empty output. Please try once more.",
            fallbackJa: "応答が途中で空のまま終了しました。もう一度お試しください。"
        )
    }

    private struct StreamReasoningSanitizer {
        var isInsideReasoningTag = false
        var isInsideToolCodeTag = false
        var pendingTagFragment = ""
        var answerStarted = false
    }

    private struct StreamChunkRoute {
        var answerText: String
        var reasoningNotes: [String]
    }

    private func routeStreamChunk(_ incoming: String, sanitizer: inout StreamReasoningSanitizer) -> StreamChunkRoute {
        let cleaned = sanitizeStreamChunk(incoming, sanitizer: &sanitizer)
        if cleaned.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return StreamChunkRoute(answerText: "", reasoningNotes: [])
        }
        if sanitizer.answerStarted {
            return StreamChunkRoute(answerText: cleaned, reasoningNotes: [])
        }
        let split = splitReasoningAndAnswer(from: cleaned)
        let hasAnswer = !split.answerText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        if hasAnswer {
            sanitizer.answerStarted = true
        }
        return StreamChunkRoute(answerText: split.answerText, reasoningNotes: split.reasoningNotes)
    }

    private func sanitizeStreamChunk(_ incoming: String, sanitizer: inout StreamReasoningSanitizer) -> String {
        StreamTagParser.sanitizeStreamChunk(
            incoming,
            isInsideReasoningTag: &sanitizer.isInsideReasoningTag,
            isInsideToolCodeTag: &sanitizer.isInsideToolCodeTag,
            pendingTagFragment: &sanitizer.pendingTagFragment
        )
    }

    private func splitReasoningAndAnswer(from raw: String) -> (reasoningNotes: [String], answerText: String) {
        StreamTagParser.splitReasoningAndAnswer(from: raw)
    }

    private func dedupeReasoningNotes(_ notes: [String]) -> [String] {
        StreamTagParser.dedupeReasoningNotes(notes)
    }

    private func mergeReasoningTraceEvents(
        metadata: [String: JSONValue]?,
        reasoningNotes: [String]
    ) -> [String: JSONValue]? {
        let notes = dedupeReasoningNotes(reasoningNotes)
        guard !notes.isEmpty else { return metadata }

        var merged = metadata ?? [:]
        var events: [JSONValue] = []
        if let existing = merged["trace_events"]?.arrayValue {
            events = existing
        }
        let now = ISO8601DateFormatter().string(from: Date())
        for note in notes.prefix(12) {
            events.append(
                .object([
                    "status": .string("done"),
                    "message": .string(note),
                    "source": .string("model_reasoning"),
                    "at": .string(now),
                ])
            )
        }
        merged["trace_events"] = .array(events)
        return merged
    }

    private func sanitizeFinalGeneratedText(_ raw: String) -> String {
        var value = raw.precomposedStringWithCanonicalMapping
        value = value.replacingOccurrences(
            of: #"(?is)<tool_code\b[^>]*>\s*([\s\S]*?)\s*</tool_code>"#,
            with: "\n```\n$1\n```\n",
            options: .regularExpression
        )
        value = value.replacingOccurrences(
            of: #"(?is)<(final_answer|assistant_response)\b[^>]*>(.*?)</\1>"#,
            with: "$2",
            options: .regularExpression
        )
        let patterns = [
            #"(?is)<(thought|think|thinking|analysis|reasoning|cot|searching|tool_running)\b[^>]*>.*?</\1>"#,
            #"(?is)</?(thought|think|thinking|analysis|reasoning|cot|searching|tool_running)\b[^>]*>"#,
            #"(?is)</?(tool_code|tool_result|action|final_answer|assistant_response|tool_[a-z0-9_]+)\b[^>]*>"#,
            #"(?is)</?(channel|analysis|final|assistant|system|user|message|observation|plan|reflection)\b[^>]*>"#,
            #"(?is)&lt;/?(?:thought|think|thinking|analysis|reasoning|cot|searching|tool_running|tool_code|tool_result|action|final_answer|assistant_response|channel|final|assistant|system|user|message|observation|plan|reflection)\b[^&]*&gt;"#,
            #"(?im)^\s*(?:thinking|thinking process|reasoning|reasoning process|chain of thought|internal monologue)\s*[:：].*$"#,
            #"(?im)^\s*(?:let(?:'|’)s think(?: step by step)?|we need to think)\s*[:：]?\s*$"#,
            #"(?is)```(?:json)?\s*\{[\s\S]{0,5000}?"hypotheses"\s*:\s*\[[\s\S]*?\}\s*```"#,
            #"(?im)^\s*\{[^\n]*"hypotheses"\s*:\s*\[[^\n]*\}\s*$"#,
            #"(?is)\bInput message:\s*.+?\n\s*Answer:\s*"#,
            #"(?im)^\s*[-•*]?\s*\([^)\n]+\.(?:txt|md|markdown|pdf|docx|py|swift|json|ya?ml)\)\s*.*\bAnswer\s*:\s*"#,
            #"(?im)^\s*json\s*$"#,
            #"(?im)^\s*<\s*/?\s*(?:channel|analysis|final|assistant|system|user|message|observation|plan|reflection)\s*>\s*[.:：-]?\s*$"#,
            #"(?im)^\s*&lt;\s*/?\s*(?:channel|analysis|final|assistant|system|user|message|observation|plan|reflection)\s*&gt;\s*[.:：-]?\s*$"#,
            #"(?im)^\s*(?:<\s*/?\s*[a-z_]+\s*>|&lt;\s*/?\s*[a-z_]+\s*&gt;)\s*[.:：-]?\s*$"#,
            #"(?im)^\s*[.:：-]+\s*$"#,
        ]
        for pattern in patterns {
            value = value.replacingOccurrences(of: pattern, with: "", options: .regularExpression)
        }
        value = value.replacingOccurrences(
            of: #"(?im)^\s*amente(?:[.!?~,:;\-\s]+)"#,
            with: "",
            options: .regularExpression
        )
        value = value.replacingOccurrences(of: #"\n{3,}"#, with: "\n\n", options: .regularExpression)
        return value.trimmingCharacters(in: .whitespacesAndNewlines)
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
        guard showThinkingProcessInChat else { return }
        let message = (rawStatus ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        guard !message.isEmpty else { return }
        if isGenericThinkingPlaceholder(message) {
            return
        }
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

    private func isGenericThinkingPlaceholder(_ message: String) -> Bool {
        let normalized = message
            .precomposedStringWithCanonicalMapping
            .lowercased()
            .replacingOccurrences(of: "…", with: "...")
            .replacingOccurrences(of: #"\s+"#, with: "", options: .regularExpression)
        let patterns = [
            #"^(?:생각중|thinking|思考中)[.!?…。！？]*$"#,
            #"^(?:생각중)\.\.\.$"#,
            #"^(?:thinking)\.\.\.$"#,
            #"^(?:思考中)\.\.\.$"#,
        ]
        return patterns.contains { pattern in
            normalized.range(of: pattern, options: .regularExpression) != nil
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
