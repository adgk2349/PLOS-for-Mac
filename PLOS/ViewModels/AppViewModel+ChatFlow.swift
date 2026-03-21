import AppKit
import Combine
import CryptoKit
import Foundation

@MainActor
extension AppViewModel {
    func askLocal() async {
        let query = chatFlowService.normalizeQuery(inputQuery)
        await askLocal(query: query, appendUserMessage: true)
        inputQuery = ""
    }


    func deepAnalyzeTapped() {
        if privacyMode == .localOnly {
            lastError = "완전 로컬 모드에서는 외부 호출이 비활성화됩니다."
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
        defer { isBusy = false }

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
        defer { isBusy = false }

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
                do {
                    let responseV2 = try await client.localChatV2(
                        LocalChatRequestV2(
                            query: trimmed,
                            mode: selectedMode,
                            conversation_id: activeConversationID,
                            session_id: activeConversationID,
                            top_k: nil,
                            filters: currentChatFilters(),
                            behavior_overrides: nil
                        )
                    )
                    citations = responseV2.citations
                    if let runtimeDetail = responseV2.runtime_detail, !runtimeDetail.isEmpty {
                        localRuntimeDetail = runtimeDetail
                    }
                    appendChatMessage(ChatMessage(localV2: responseV2, timestamp: Date()))
                    syncActiveRoom(citations: citations, latestQueryForDeepAnalysis: trimmed)
                    handled = true
                } catch {
                    #if DEBUG
                        let response = try await client.localChat(
                            LocalChatRequest(
                                query: trimmed,
                                mode: selectedMode,
                                conversation_id: activeConversationID,
                                top_k: nil,
                                filters: currentChatFilters()
                            )
                        )
                        citations = response.citations
                        if let runtimeDetail = response.runtime_detail, !runtimeDetail.isEmpty {
                            localRuntimeDetail = runtimeDetail
                        }
                        appendChatMessage(ChatMessage(local: response, timestamp: Date()))
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
            try await client.writeMemoryEvent(
                MemoryEventRequest(
                    event_type: eventType,
                    session_id: activeConversationID,
                    workspace_id: currentWorkspaceID(),
                    summary: summary,
                    related_file_ids: relatedFileIDs,
                    related_action_ids: relatedActionIDs,
                    metadata_json: metadata,
                    importance: importance
                )
            )
        }
    }


    func currentWorkspaceID() -> String {
        chatFlowService.workspaceID(
            includedFolderURLs: includedFolderURLs,
            excludedPaths: excludedPaths
        )
    }
}
