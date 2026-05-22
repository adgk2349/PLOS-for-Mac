import AppKit
import Foundation
import SwiftUI
#if canImport(MarkdownUI)
import MarkdownUI
#endif

enum MainWorkspaceChatPanelSplitMarker {}

struct ChatPanelView: View {
    @ObservedObject var viewModel: AppViewModel
    @Binding var isSidebarCollapsed: Bool
    @State private var showPresetMenu = false
    @State private var expandedReasoningMessageIDs: Set<UUID> = []
    @State private var expandedLiveReasoningMessageIDs: Set<UUID> = []
    @State private var revealedCharactersByKey: [String: Int] = [:]
    @State private var animatingRevealKeys: Set<String> = []
    @State private var scrollViewportHeight: CGFloat = 0
    @State private var bottomAnchorMinY: CGFloat = CGFloat.greatestFiniteMagnitude
    @State private var showScrollToBottomButton = false
    @State private var hasAutoScrolledOnEntry = false
    @State private var hoveredAssistantCopyMessageIDs: Set<UUID> = []
    @State private var hoveredCodeBlockIDs: Set<String> = []
    @Environment(\.colorScheme) private var colorScheme
    private let chatBottomAnchorID = "chat-bottom-anchor"
    private let bottomGapThreshold: CGFloat = 72
    private var language: AppLanguage { viewModel.appLanguage }

    private func t(_ ko: String, _ en: String, _ ja: String) -> String {
        L10n.text(ko, en, ja, language: language)
    }

    private static let messageTimestampFormatter: DateFormatter = {
        let formatter = DateFormatter()
        formatter.locale = .current
        formatter.timeZone = .current
        formatter.dateFormat = "M/d HH:mm"
        return formatter
    }()

    var body: some View {
        ZStack(alignment: .top) {
            conversationColumn

            header
                .padding(.top, 12)
                .padding(.horizontal, 10)
        }
        .padding(10)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .confirmationDialog(t("외부 AI 호출 전 확인", "Confirm external AI call", "外部AI呼び出しの確認"), isPresented: $viewModel.needsExternalConfirmation) {
            Button(t("승인하고 실행", "Approve and run", "承認して実行")) {
                viewModel.confirmExternalCall()
            }
            Button(t("취소", "Cancel", "キャンセル"), role: .cancel) {
                viewModel.cancelExternalCall()
            }
        } message: {
            Text(t("선택된 자료 일부가 외부 제공자에 전달될 수 있습니다.", "Some selected material may be sent to an external provider.", "選択した資料の一部が外部プロバイダに送信される可能性があります。"))
        }
        .confirmationDialog(
            t("시스템 액션 승인", "Approve system action", "システム操作の承認"),
            isPresented: Binding(
                get: { viewModel.pendingSystemAction != nil },
                set: { shown in
                    if !shown {
                        viewModel.cancelPendingSystemAction()
                    }
                }
            )
        ) {
            Button(t("허용하고 실행", "Allow and run", "許可して実行")) {
                viewModel.confirmPendingSystemAction()
            }
            Button(t("취소", "Cancel", "キャンセル"), role: .cancel) {
                viewModel.cancelPendingSystemAction()
            }
        } message: {
            if let path = viewModel.pendingSystemAction?.payload["file_path"], !path.isEmpty {
                Text(t("로컬 파일을 엽니다: \(path)", "Open local file: \(path)", "ローカルファイルを開きます: \(path)"))
            } else {
                Text(t("시스템 액션 실행 권한이 필요합니다.", "Permission is required to run this system action.", "このシステム操作の実行には権限が必要です。"))
            }
        }
        .animation(.none, value: colorScheme)
    }

    private var header: some View {
        let topBarCircleTint: Color = colorScheme == .dark
            ? Color.white.opacity(0.08)
            : Color.black.opacity(0.035)
        let presetTint: Color = colorScheme == .dark
            ? Color.white.opacity(0.08)
            : Color.black.opacity(0.03)
        let presetStroke: Color = colorScheme == .dark
            ? Color.white.opacity(0.22)
            : Color.black.opacity(0.16)
        return HStack(spacing: 10) {
            if isSidebarCollapsed {
                Button {
                    withAnimation(.easeInOut(duration: 0.18)) {
                        isSidebarCollapsed = false
                    }
                } label: {
                    Image(systemName: "sidebar.right")
                        .font(.system(size: 13, weight: .semibold))
                        .frame(width: 18, height: 18)
                        .frame(width: 40, height: 40)
                        .plosGlassCircle(tint: topBarCircleTint)
                }
                .buttonStyle(.plain)
            }

            Button {
                showPresetMenu.toggle()
            } label: {
                HStack(spacing: 8) {
                    Text("PLOS")
                        .font(.headline.weight(.semibold))
                    Text(viewModel.quickInferencePreset.title(language: language))
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Image(systemName: "chevron.down")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .background {
                    Capsule(style: .continuous)
                        .fill(.clear)
                        .glassEffect(
                            .regular.tint(presetTint),
                            in: Capsule(style: .continuous)
                        )
                }
                .overlay {
                    Capsule(style: .continuous)
                        .stroke(presetStroke, lineWidth: 1)
                }
                .clipShape(Capsule(style: .continuous))
            }
            .buttonStyle(.plain)
            .popover(isPresented: $showPresetMenu, arrowEdge: .bottom) {
                VStack(alignment: .leading, spacing: 8) {
                    ForEach(Array(QuickInferencePreset.allCases), id: \.rawValue) { preset in
                        Button {
                            viewModel.applyQuickInferencePreset(preset)
                            showPresetMenu = false
                        } label: {
                            HStack {
                                VStack(alignment: .leading, spacing: 2) {
                                    Text(preset.title(language: language))
                                        .font(.subheadline.weight(.semibold))
                                    Text(preset.detail(language: language))
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer(minLength: 12)
                                if viewModel.quickInferencePreset == preset {
                                    Image(systemName: "checkmark")
                                        .font(.caption.weight(.semibold))
                                }
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 8)
                            .plosGlassInputFrame(radius: 10)
                        }
                        .buttonStyle(.plain)
                    }
                    Divider()
                        .padding(.vertical, 2)
                    Toggle(
                        t("역할극 모드", "Roleplay mode", "ロールプレイモード"),
                        isOn: Binding(
                            get: { viewModel.roleplayModeEnabled },
                            set: { viewModel.setRoleplayMode($0) }
                        )
                    )
                    .toggleStyle(.switch)
                    .font(.subheadline.weight(.semibold))
                    .padding(.horizontal, 6)
                    Text(
                        t(
                            "캐릭터/말투 유지 우선. 사실 검색 라우팅은 줄어듭니다.",
                            "Prioritizes character/tone consistency and reduces factual web-search routing.",
                            "キャラクター/口調の維持を優先し、事実検索ルーティングを抑えます。"
                        )
                    )
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 6)
                }
                .padding(10)
                .frame(width: 320)
                .background(PLOSGlassBackground())
            }

            Spacer(minLength: 12)

            HStack(spacing: 10) {
                ShareLink(item: viewModel.currentChatTranscript.isEmpty ? "PLOS" : viewModel.currentChatTranscript) {
                    Image(systemName: "square.and.arrow.up")
                        .font(.system(size: 13, weight: .semibold))
                        .frame(width: 18, height: 18)
                        .frame(width: 40, height: 40)
                        .plosGlassCircle(tint: topBarCircleTint)
                }
                .buttonStyle(.plain)
                .focusable(false)
                .contentShape(Circle())

                Button {
                    viewModel.copyCurrentChatTranscriptToClipboard()
                } label: {
                    Image(systemName: "doc.on.doc")
                        .font(.system(size: 13, weight: .semibold))
                        .frame(width: 18, height: 18)
                        .frame(width: 40, height: 40)
                        .plosGlassCircle(tint: topBarCircleTint)
                }
                .buttonStyle(.plain)
                .focusable(false)
                .contentShape(Circle())
            }
            .padding(.trailing, 2)
        }
        .frame(maxWidth: 920)
        .frame(maxWidth: .infinity)
    }

    private var conversationColumn: some View {
        ScrollViewReader { proxy in
            ZStack(alignment: .bottomTrailing) {
                ScrollView {
                    LazyVStack(spacing: 12) {
                        ForEach(viewModel.chatMessages) { message in
                            messageRow(message)
                        }
                        if viewModel.isGeneratingChatResponse {
                            generatingIndicatorRow
                        }
                        Color.clear
                            .frame(height: 1)
                            .id(chatBottomAnchorID)
                            .background(
                                GeometryReader { geo in
                                    Color.clear.preference(
                                        key: ChatBottomAnchorMinYPreferenceKey.self,
                                        value: geo.frame(in: .named("chat-scroll")).minY
                                    )
                                }
                            )
                    }
                    .frame(maxWidth: 920)
                    .frame(maxWidth: .infinity)
                    .padding(.top, 72)
                    .padding(.horizontal, 6)
                    .padding(.bottom, 170)
                }
                .coordinateSpace(name: "chat-scroll")
                .background(
                    GeometryReader { geo in
                        Color.clear.preference(
                            key: ChatViewportHeightPreferenceKey.self,
                            value: geo.size.height
                        )
                    }
                )
                .scrollIndicators(.hidden)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .onAppear {
                    scheduleInitialScroll(with: proxy)
                }
                .onChange(of: viewModel.chatMessages.count) { _, _ in
                    guard let last = viewModel.chatMessages.last else { return }
                    primeRevealState(for: last)
                    if isNearBottom {
                        scrollToBottom(with: proxy, animated: true)
                    }
                }
                .onChange(of: latestAssistantRevealSignature) { _, _ in
                    guard let last = viewModel.chatMessages.last(where: { $0.source == .local || $0.source == .external }) else {
                        return
                    }
                    guard !last.isStreaming else { return }
                    synchronizeRevealState(for: last)
                }
                .onChange(of: viewModel.isGeneratingChatResponse) { _, isGenerating in
                    if !isGenerating {
                        expandedLiveReasoningMessageIDs.removeAll()
                    }
                }
                .onChange(of: viewModel.selectedChatRoomID) { _, _ in
                    revealedCharactersByKey.removeAll()
                    animatingRevealKeys.removeAll()
                    hasAutoScrolledOnEntry = false
                    expandedLiveReasoningMessageIDs.removeAll()
                    DispatchQueue.main.asyncAfter(deadline: .now() + 0.08) {
                        scrollToBottom(with: proxy, animated: true)
                        hasAutoScrolledOnEntry = true
                    }
                }
                .onPreferenceChange(ChatViewportHeightPreferenceKey.self) { height in
                    scrollViewportHeight = height
                    refreshScrollButtonVisibility()
                }
                .onPreferenceChange(ChatBottomAnchorMinYPreferenceKey.self) { minY in
                    bottomAnchorMinY = minY
                    refreshScrollButtonVisibility()
                }

                if showScrollToBottomButton {
                    Button {
                        scrollToBottom(with: proxy, animated: true)
                    } label: {
                        Image(systemName: "arrow.down")
                            .font(.system(size: 13, weight: .semibold))
                            .frame(width: 16, height: 16)
                            .frame(width: 36, height: 36)
                            .plosGlassCircle()
                    }
                    .buttonStyle(.plain)
                    .contentShape(Circle())
                    .padding(.trailing, 18)
                    .padding(.bottom, 96)
                    .transition(.move(edge: .bottom).combined(with: .opacity))
                }

                composer
                    .padding(.horizontal, 6)
                    .padding(.bottom, 8)
            }
        }
    }

    private var isNearBottom: Bool {
        guard scrollViewportHeight > 0, bottomAnchorMinY.isFinite else { return true }
        return bottomAnchorMinY <= (scrollViewportHeight + bottomGapThreshold)
    }

    private func refreshScrollButtonVisibility() {
        showScrollToBottomButton = !isNearBottom
    }

    private func scheduleInitialScroll(with proxy: ScrollViewProxy) {
        guard !hasAutoScrolledOnEntry else { return }
        hasAutoScrolledOnEntry = true
        DispatchQueue.main.asyncAfter(deadline: .now() + 0.12) {
            scrollToBottom(with: proxy, animated: true)
        }
    }

    private func scrollToBottom(with proxy: ScrollViewProxy, animated: Bool) {
        let action = {
            proxy.scrollTo(chatBottomAnchorID, anchor: .bottom)
        }
        if animated {
            withAnimation(.easeOut(duration: 0.26)) {
                action()
            }
        } else {
            action()
        }
        showScrollToBottomButton = false
    }

    private func messageRow(_ message: ChatMessage) -> some View {
        HStack {
            VStack(alignment: message.source == .user ? .trailing : .leading, spacing: 4) {
                messageContainer(message)
                messageTimestampRow(for: message)
            }
        }
        .frame(maxWidth: .infinity, alignment: message.source == .user ? .trailing : .leading)
        .onAppear {
            primeRevealState(for: message)
        }
    }

    @ViewBuilder
    private func messageContainer(_ message: ChatMessage) -> some View {
        switch message.source {
        case .user:
            let raw = nfc(message.text ?? "")
            let compact = isCompactUserBubble(raw)

            if compact {
                VStack(alignment: .trailing, spacing: 0) {
                    markdownRichContent(raw, font: .body, keyPrefix: "\(message.id.uuidString)-user")
                        .multilineTextAlignment(.trailing)
                }
                .padding(12)
                .fixedSize(horizontal: true, vertical: false)
                .background {
                    Capsule(style: .continuous)
                        .fill(.clear)
                        .glassEffect(
                            .regular.tint(PLOSGlassTheme.userBubbleTint(for: colorScheme)),
                            in: Capsule(style: .continuous)
                        )
                }
                .overlay {
                    Capsule(style: .continuous)
                        .stroke(PLOSGlassTheme.userBubbleStroke(for: colorScheme), lineWidth: 1)
                }
                .clipShape(Capsule(style: .continuous))
            } else {
                VStack(alignment: .trailing, spacing: 0) {
                    markdownRichContent(raw, font: .body, keyPrefix: "\(message.id.uuidString)-user")
                        .multilineTextAlignment(.trailing)
                }
                .padding(12)
                .frame(maxWidth: 540, alignment: .trailing)
                .background {
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .fill(.clear)
                        .glassEffect(
                            .regular.tint(PLOSGlassTheme.userBubbleTint(for: colorScheme)),
                            in: RoundedRectangle(cornerRadius: 18, style: .continuous)
                        )
                }
                .overlay {
                    RoundedRectangle(cornerRadius: 18, style: .continuous)
                        .stroke(PLOSGlassTheme.userBubbleStroke(for: colorScheme), lineWidth: 1)
                }
                .clipShape(RoundedRectangle(cornerRadius: 18, style: .continuous))
            }

        case .external:
            VStack(alignment: .leading, spacing: 8) {
                HStack(spacing: 8) {
                    Text(t("외부 응답", "External response", "外部応答"))
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Spacer(minLength: 0)
                    assistantMessageCopyButton(for: message)
                }
                markdownRichContent(
                    animatedText(for: message, field: "text", raw: nfc(message.text ?? "")),
                    font: .body,
                    keyPrefix: "\(message.id.uuidString)-external-text"
                )
            }
            .padding(12)
            .frame(maxWidth: 840, alignment: .leading)
            .plosGlassPanel()

        case .local:
            VStack(alignment: .leading, spacing: 8) {
                localHeader(message)
                localBody(message)
            }
            .padding(.horizontal, 2)
            .padding(.vertical, 4)
            .frame(maxWidth: 840, alignment: .leading)
        }
    }

    private func isCompactUserBubble(_ text: String) -> Bool {
        let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
        return !trimmed.contains("\n") && trimmed.count <= 56
    }

    private func isReasoningInProgress(for message: ChatMessage) -> Bool {
        viewModel.isGeneratingChatResponse && message.id == viewModel.activeGeneratingMessageID
    }

    private func toggleReasoningExpansion(for message: ChatMessage) {
        if expandedReasoningMessageIDs.contains(message.id) {
            expandedReasoningMessageIDs.remove(message.id)
        } else {
            expandedReasoningMessageIDs.insert(message.id)
        }
    }

    private func isLiveReasoningExpanded(messageID: UUID?) -> Bool {
        guard let messageID else { return false }
        return expandedLiveReasoningMessageIDs.contains(messageID)
    }

    private func toggleLiveReasoningExpansion(messageID: UUID?) {
        guard let messageID else { return }
        if expandedLiveReasoningMessageIDs.contains(messageID) {
            expandedLiveReasoningMessageIDs.remove(messageID)
        } else {
            expandedLiveReasoningMessageIDs.insert(messageID)
        }
    }

    private func reasoningDetailText(for message: ChatMessage) -> String {
        var sections: [String] = []

        if let brief = message.reasoningBrief,
           !brief.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            sections.append("\(t("요약 판단", "Summary reasoning", "要約判断"))\n\(brief)")
        }

        if let parsed = message.parsedIntent {
            var rows: [String] = []
            rows.append("\(t("의도", "Intent", "意図")): \(parsed.intent.rawValue)")
            rows.append(String(format: t("의도 신뢰도: %.2f", "Intent confidence: %.2f", "意図信頼度: %.2f"), parsed.confidence))
            if let operation = parsed.operation, !operation.isEmpty {
                rows.append("\(t("작업 연산", "Operation", "操作")): \(operation)")
            }
            if let scope = parsed.scope, !scope.isEmpty {
                rows.append("\(t("요청 범위", "Scope", "範囲")): \(scope)")
            }
            if let target = parsed.target, !target.isEmpty {
                rows.append("\(t("요청 대상", "Target", "対象")): \(target)")
            }
            if let ambiguity = parsed.ambiguity, !ambiguity.isEmpty {
                rows.append("\(t("모호성 판단", "Ambiguity", "曖昧性")): \(ambiguity)")
            }

            if !parsed.entities.file_names.isEmpty {
                rows.append("\(t("파일 엔티티", "File entities", "ファイルエンティティ")): \(parsed.entities.file_names.joined(separator: ", "))")
            }
            if !parsed.entities.tags.isEmpty {
                rows.append("\(t("태그 엔티티", "Tag entities", "タグエンティティ")): \(parsed.entities.tags.joined(separator: ", "))")
            }
            if !parsed.entities.topics.isEmpty {
                rows.append("\(t("주제 엔티티", "Topic entities", "トピックエンティティ")): \(parsed.entities.topics.joined(separator: ", "))")
            }
            if !parsed.entities.projects.isEmpty {
                rows.append("\(t("프로젝트 엔티티", "Project entities", "プロジェクトエンティティ")): \(parsed.entities.projects.joined(separator: ", "))")
            }
            if let year = parsed.time_filters.year {
                rows.append("\(t("연도 필터", "Year filter", "年フィルタ")): \(year)")
            }
            if let from = parsed.time_filters.year_from, let to = parsed.time_filters.year_to {
                rows.append("\(t("기간 필터", "Range filter", "期間フィルタ")): \(from)-\(to)")
            }
            if let days = parsed.time_filters.relative_days {
                rows.append(t("상대 기간: 최근 \(days)일", "Relative window: last \(days) days", "相対期間: 直近\(days)日"))
            }
            if !parsed.workspace_filters.included_paths.isEmpty {
                rows.append("\(t("포함 경로 수", "Included paths", "含まれるパス数")): \(parsed.workspace_filters.included_paths.count)")
            }
            if !parsed.workspace_filters.excluded_paths.isEmpty {
                rows.append("\(t("제외 경로 수", "Excluded paths", "除外パス数")): \(parsed.workspace_filters.excluded_paths.count)")
            }

            sections.append(rows.joined(separator: "\n"))
        }

        if let plan = message.plan {
            var rows: [String] = []
            rows.append("\(t("계획 유형", "Plan type", "計画タイプ")): \(plan.plan_type)")
            rows.append("\(t("응답 전략", "Response strategy", "応答戦略")): \(plan.response_strategy)")
            rows.append("\(t("선택 파일 수", "Selected files", "選択ファイル数")): \(plan.selected_files.count)")
            rows.append("\(t("선택 청크 수", "Selected chunks", "選択チャンク数")): \(plan.selected_chunks.count)")
            if !plan.allowed_actions.isEmpty {
                rows.append("\(t("허용 액션", "Allowed actions", "許可アクション")): \(plan.allowed_actions.map(\.rawValue).joined(separator: ", "))")
            }
            rows.append("\(t("외부 추론 필요", "External reasoning needed", "外部推論が必要")): \(plan.external_reasoning_needed ? t("예", "Yes", "はい") : t("아니오", "No", "いいえ"))")
            sections.append(rows.joined(separator: "\n"))
        }

        if let verify = message.verification {
            var rows: [String] = []
            rows.append("\(t("검증 유효성", "Verification", "検証")): \(verify.is_valid ? t("유효", "Valid", "有効") : t("재검토", "Review", "再確認"))")
            rows.append(String(format: t("검증 신뢰도: %.2f", "Verification confidence: %.2f", "検証信頼度: %.2f"), verify.confidence))
            rows.append(String(format: t("모호성: %.2f", "Ambiguity: %.2f", "曖昧性: %.2f"), verify.ambiguity_level))
            rows.append("\(t("후보 모드", "Candidate mode", "候補モード")): \(verify.candidate_mode ? t("예", "Yes", "はい") : t("아니오", "No", "いいえ"))")
            if !verify.issues.isEmpty {
                rows.append("\(t("이슈", "Issues", "課題")): \(verify.issues.joined(separator: ", "))")
            }
            sections.append(rows.joined(separator: "\n"))
        }

        if sections.isEmpty {
            return t("추론 상세 정보가 없습니다.", "No reasoning details.", "推論詳細はありません。")
        }

        return sections.joined(separator: "\n\n")
    }

    @ViewBuilder
    private func localHeader(_ message: ChatMessage) -> some View {
        HStack(spacing: 8) {
            Text(t("PLOS", "PLOS", "PLOS"))
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)

            if message.responseMetadata?["conversation_path"] == .string("external_escalated") {
                Text(t("외부 에스컬레이션", "External escalated", "外部エスカレーション"))
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.orange.opacity(0.24), in: Rectangle())
            }

            Spacer(minLength: 0)
            assistantMessageCopyButton(for: message)
        }
    }

    @ViewBuilder
    private func localBody(_ message: ChatMessage) -> some View {
        let traceEvents = thinkingTraceEvents(for: message)

        if let text = message.text?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty {
            let content = animatedText(for: message, field: "text", raw: nfc(text))
            markdownRichContent(
                content,
                font: .body,
                keyPrefix: "\(message.id.uuidString)-local-text"
            )
        }
        if let lead = message.lead, !lead.isEmpty {
            let content = animatedText(for: message, field: "lead", raw: nfc(lead))
            markdownRichContent(
                content,
                font: .body.weight(.semibold),
                keyPrefix: "\(message.id.uuidString)-lead"
            )
        }
        if let summary = message.resultSummary, !summary.isEmpty {
            let content = animatedText(for: message, field: "summary", raw: nfc(summary))
            markdownRichContent(
                content,
                font: .body,
                keyPrefix: "\(message.id.uuidString)-summary"
            )
        }
        let artifacts = message.artifacts ?? []
        if !artifacts.isEmpty {
            artifactSection(artifacts)
        }

        let smallCitations = citationsForMessage(message)
        if !smallCitations.isEmpty {
            miniCitationStrip(smallCitations)
        }

        if viewModel.showThinkingProcessInChat, hasReasoningSignal(message) {
            if isReasoningInProgress(for: message) {
                let isExpanded = isLiveReasoningExpanded(messageID: message.id)
                VStack(alignment: .leading, spacing: 3) {
                    Button {
                        toggleLiveReasoningExpansion(messageID: message.id)
                    } label: {
                        HStack(spacing: 6) {
                            Text(t("생각중", "Thinking", "思考中"))
                                .font(.caption2.weight(.semibold))
                                .foregroundStyle(.secondary)
                            Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                                .font(.caption2.weight(.semibold))
                                .foregroundStyle(.secondary.opacity(0.85))
                        }
                    }
                    .buttonStyle(.plain)

                    if let headline = latestTraceHeadline(for: message), !headline.isEmpty {
                        Text(headline)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                    if isExpanded, !traceEvents.isEmpty {
                        ScrollView {
                            thinkingTraceTimeline(traceEvents)
                        }
                        .frame(maxHeight: 140)
                        .plosGlassInputFrame(radius: 10)
                    }
                }
            } else {
                Button {
                    toggleReasoningExpansion(for: message)
                } label: {
                    Text(t("생각됨", "Thought", "思考済み"))
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)

                if expandedReasoningMessageIDs.contains(message.id) {
                    ScrollView {
                        if !traceEvents.isEmpty {
                            thinkingTraceTimeline(traceEvents)
                        } else {
                            Text(reasoningDetailText(for: message))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                                .textSelection(.enabled)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 8)
                        }
                    }
                    .frame(maxHeight: 180)
                    .plosGlassInputFrame(radius: 10)
                }
            }
        }

        let shouldShowActions: Bool = {
            guard let planType = message.plan?.plan_type.lowercased() else { return true }
            return planType != "conversation"
        }()
        let visibleActions = shouldShowActions
            ? message.actions.filter { action in
                action.kind != .askFollowup && !action.label.contains("다음 질문")
            }
            : []

        if !visibleActions.isEmpty {
            LazyVGrid(columns: [GridItem(.adaptive(minimum: 120), spacing: 8)], spacing: 8) {
                ForEach(visibleActions) { action in
                    Button(action.label) {
                        Task { await viewModel.executeAction(action) }
                    }
                    .buttonStyle(.plain)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background {
                        Capsule(style: .continuous)
                            .fill(.clear)
                            .glassEffect(
                                .regular.tint(Color.white.opacity(0.04)),
                                in: Capsule(style: .continuous)
                            )
                    }
                    .overlay {
                        Capsule(style: .continuous)
                            .stroke(Color.white.opacity(0.22), lineWidth: 1)
                    }
                    .clipShape(Capsule(style: .continuous))
                }
            }
            .padding(.top, 2)
        }
    }

    @ViewBuilder
    private func artifactSection(_ artifacts: [GeneratedArtifact]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(t("생성 결과", "Generated artifacts", "生成成果物"))
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)
            ForEach(artifacts) { artifact in
                VStack(alignment: .leading, spacing: 6) {
                    Text(artifact.title)
                        .font(.callout.weight(.semibold))
                    if artifact.mime_type.lowercased().hasPrefix("image/"),
                       let filePath = filePath(from: artifact.file_uri),
                       let image = NSImage(contentsOfFile: filePath)
                    {
                        Image(nsImage: image)
                            .resizable()
                            .scaledToFit()
                            .frame(maxHeight: 220)
                            .clipShape(RoundedRectangle(cornerRadius: 10, style: .continuous))
                    }
                    if let destination = urlForArtifactURI(artifact.file_uri) {
                        Link(artifact.file_uri, destination: destination)
                            .font(.caption2)
                            .foregroundStyle(Color.blue.opacity(0.92))
                    } else {
                        Text(artifact.file_uri)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                    }
                }
                .padding(10)
                .frame(maxWidth: .infinity, alignment: .leading)
                .plosGlassInputFrame(radius: 10)
            }
        }
    }

    private func urlForArtifactURI(_ raw: String) -> URL? {
        let trimmed = raw.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if let url = URL(string: trimmed), url.scheme != nil {
            return url
        }
        return URL(fileURLWithPath: trimmed)
    }

    private func filePath(from uri: String) -> String? {
        let trimmed = uri.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !trimmed.isEmpty else { return nil }
        if trimmed.hasPrefix("file://"), let url = URL(string: trimmed) {
            return url.path
        }
        if trimmed.hasPrefix("/") {
            return trimmed
        }
        return nil
    }

    private var generatingIndicatorRow: some View {
        let hasStreamingAssistantMessage = {
            guard let activeID = viewModel.activeGeneratingMessageID else { return false }
            guard let message = viewModel.chatMessages.first(where: { $0.id == activeID }) else { return false }
            return message.isStreaming && message.source == .local
        }()
        let traceEvents = generatingTraceEvents()
        let showThinking = viewModel.showThinkingProcessInChat && !traceEvents.isEmpty
        let isExpanded = isLiveReasoningExpanded(messageID: viewModel.activeGeneratingMessageID)
        return HStack {
            if hasStreamingAssistantMessage {
                EmptyView()
            } else {
            VStack(alignment: .leading, spacing: 8) {
                if viewModel.sidecarRecoveryState == "recovering" {
                    HStack(spacing: 6) {
                        ProgressView()
                            .controlSize(.small)
                        Text(t("복구 중...", "Recovering...", "復旧中..."))
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.secondary)
                    }
                }
                if showThinking {
                    VStack(alignment: .leading, spacing: 3) {
                        Button {
                            toggleLiveReasoningExpansion(messageID: viewModel.activeGeneratingMessageID)
                        } label: {
                            HStack(spacing: 6) {
                                Text(t("생각중", "Thinking", "思考中"))
                                    .font(.caption2.weight(.semibold))
                                    .foregroundStyle(.secondary)
                                Image(systemName: isExpanded ? "chevron.up" : "chevron.down")
                                    .font(.caption2.weight(.semibold))
                                    .foregroundStyle(.secondary.opacity(0.85))
                            }
                        }
                        .buttonStyle(.plain)

                        if let headline = latestTraceHeadlineForGenerating(), !headline.isEmpty {
                            Text(headline)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }
                    if isExpanded {
                        ScrollView {
                            thinkingTraceTimeline(traceEvents)
                        }
                        .frame(maxHeight: 140)
                        .plosGlassInputFrame(radius: 10)
                    }
                } else {
                    TypingDotsView()
                }
            }
            .padding(.horizontal, 2)
            .padding(.vertical, 4)
            .frame(maxWidth: 840, alignment: .leading)
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func hasReasoningSignal(_ message: ChatMessage) -> Bool {
        guard viewModel.showThinkingProcessInChat else { return false }
        if !thinkingTraceEvents(for: message).isEmpty {
            return true
        }
        let planType = (message.plan?.plan_type ?? "").lowercased()
        if ["summary", "compare", "analysis", "deep", "retrieval", "web"].contains(where: { planType.contains($0) }) {
            return true
        }
        return false
    }

    private func messageTimestampRow(for message: ChatMessage) -> some View {
        Text(Self.messageTimestampFormatter.string(from: message.timestamp))
            .font(.caption2)
            .foregroundStyle(.secondary.opacity(0.82))
            .padding(.horizontal, 2)
    }

    private func latestTraceHeadlineForGenerating() -> String? {
        guard let latest = generatingTraceEvents().last else { return nil }
        return localizedTraceMessage(latest)
    }

    private func generatingTraceEvents() -> [ThinkingTraceEvent] {
        guard viewModel.showThinkingProcessInChat else { return [] }
        if !viewModel.liveThinkingTraceEvents.isEmpty {
            return Array(viewModel.liveThinkingTraceEvents.prefix(24).enumerated()).map { index, item in
                ThinkingTraceEvent(
                    id: "live-\(index)-\(item.id.uuidString)",
                    status: item.status,
                    message: item.message,
                    source: item.source,
                    url: item.url,
                    at: item.at
                )
            }
        }
        guard let activeID = viewModel.activeGeneratingMessageID,
              let activeMessage = viewModel.chatMessages.first(where: { $0.id == activeID })
        else {
            return []
        }
        return thinkingTraceEvents(for: activeMessage)
    }

    private func latestTraceHeadline(for message: ChatMessage) -> String? {
        guard let latest = thinkingTraceEvents(for: message).last else { return nil }
        return localizedTraceMessage(latest)
    }

    private func thinkingTraceEvents(for message: ChatMessage) -> [ThinkingTraceEvent] {
        guard viewModel.showThinkingProcessInChat else { return [] }
        if isReasoningInProgress(for: message) && !viewModel.liveThinkingTraceEvents.isEmpty {
            return Array(viewModel.liveThinkingTraceEvents.prefix(24).enumerated()).map { index, item in
                ThinkingTraceEvent(
                    id: "live-\(index)-\(item.id.uuidString)",
                    status: item.status,
                    message: item.message,
                    source: item.source,
                    url: item.url,
                    at: item.at
                )
            }
        }
        guard let traceValues = message.responseMetadata?["trace_events"]?.arrayValue else {
            return []
        }
        var output: [ThinkingTraceEvent] = []
        for (index, item) in traceValues.prefix(24).enumerated() {
            guard let object = item.objectValue else { continue }
            let status = object["status"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "done"
            let rawMessage = object["message"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if rawMessage.isEmpty || !isSafeThinkingTraceMessage(rawMessage) {
                continue
            }
            let source = object["source"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines) ?? "pipeline"
            if !isMeaningfulThinkingTrace(status: status, source: source, message: rawMessage) {
                continue
            }
            let url = object["url"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines)
            let at = object["at"]?.stringValue?.trimmingCharacters(in: .whitespacesAndNewlines)
            output.append(
                ThinkingTraceEvent(
                    id: "\(message.id.uuidString)-\(index)",
                    status: status,
                    message: rawMessage,
                    source: source,
                    url: (url?.isEmpty == false ? url : nil),
                    at: (at?.isEmpty == false ? at : nil)
                )
            )
        }
        return output
    }

    private func isSafeThinkingTraceMessage(_ text: String) -> Bool {
        let lowered = text.lowercased()
        let blocked = [
            "user:",
            "assistant:",
            "system prompt",
            "사용자 메시지에",
            "바로 반응하세요",
            "final answer",
            "chain of thought",
        ]
        return !blocked.contains(where: { lowered.contains($0) })
    }

    private func isMeaningfulThinkingTrace(status: String, source: String, message: String) -> Bool {
        let statusValue = status.lowercased()
        let sourceValue = source.lowercased()
        if ["retrieving", "retrieved", "warning"].contains(statusValue) {
            return true
        }
        if ["external", "retrieval", "model_reasoning"].contains(sourceValue) {
            return true
        }
        let lowered = message.lowercased()
        let cues = [
            "analysis", "summary", "compare", "web", "search", "fetch",
            "분석", "요약", "비교", "검색", "수집", "웹",
        ]
        return cues.contains(where: { lowered.contains($0) })
    }

    @ViewBuilder
    private func thinkingTraceTimeline(_ events: [ThinkingTraceEvent]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(events) { event in
                HStack(alignment: .top, spacing: 8) {
                    Circle()
                        .fill(traceColor(for: event.status))
                        .frame(width: 6, height: 6)
                        .padding(.top, 5)
                    VStack(alignment: .leading, spacing: 2) {
                        Text(localizedTraceMessage(event))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .fixedSize(horizontal: false, vertical: true)
                        if let url = event.url, !url.isEmpty {
                            if let destination = URL(string: url) {
                                Link(url, destination: destination)
                                    .font(.caption2)
                                    .foregroundStyle(Color.blue.opacity(0.92))
                            } else {
                                Text(url)
                                    .font(.caption2)
                                    .foregroundStyle(Color.blue.opacity(0.92))
                                    .textSelection(.enabled)
                            }
                        }
                    }
                    Spacer(minLength: 0)
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
    }

    private func traceColor(for status: String) -> Color {
        switch status {
        case "retrieving":
            return .orange
        case "retrieved", "done":
            return .green
        case "warning":
            return .red
        default:
            return .secondary
        }
    }

    private func localizedTraceMessage(_ event: ThinkingTraceEvent) -> String {
        let raw = event.message.trimmingCharacters(in: .whitespacesAndNewlines)
        let lowered = raw.lowercased()

        if lowered == "retrieving web search results" {
            return t("웹 검색 결과를 수집 중", "Retrieving web search results", "Web検索結果を取得中")
        }
        if lowered == "web search completed via direct crawler" {
            return t("직접 크롤러로 웹 검색 완료", "Web search completed via direct crawler", "直接クローラでWeb検索完了")
        }
        if lowered.hasPrefix("retrieving ") {
            let tail = String(raw.dropFirst("retrieving ".count))
            return t("수집 중 \(tail)", "Retrieving \(tail)", "\(tail) を取得中")
        }
        if lowered.hasPrefix("retrieved ") {
            let tail = String(raw.dropFirst("retrieved ".count))
            return t("수집 완료 \(tail)", "Retrieved \(tail)", "\(tail) の取得完了")
        }
        if lowered.hasPrefix("search failed") {
            return t(raw.replacingOccurrences(of: "search failed", with: "검색 실패"), raw, raw.replacingOccurrences(of: "search failed", with: "検索失敗"))
        }
        if lowered.hasPrefix("fetch failed") {
            return t(raw.replacingOccurrences(of: "fetch failed", with: "수집 실패"), raw, raw.replacingOccurrences(of: "fetch failed", with: "取得失敗"))
        }
        if lowered.hasPrefix("web search blocked:") {
            let reason = raw.components(separatedBy: ":").dropFirst().joined(separator: ":").trimmingCharacters(in: .whitespacesAndNewlines)
            return t("웹 검색 차단: \(reason)", "Web search blocked: \(reason)", "Web検索がブロックされました: \(reason)")
        }
        if lowered.hasPrefix("intent resolved:") {
            let value = raw.components(separatedBy: ":").dropFirst().joined(separator: ":").trimmingCharacters(in: .whitespacesAndNewlines)
            return t("의도 판단: \(value)", "Intent resolved: \(value)", "意図判定: \(value)")
        }
        if lowered.hasPrefix("execution plan:") {
            let value = raw.components(separatedBy: ":").dropFirst().joined(separator: ":").trimmingCharacters(in: .whitespacesAndNewlines)
            return t("실행 계획: \(value)", "Execution plan: \(value)", "実行計画: \(value)")
        }
        if lowered.hasPrefix("agent step:") {
            let value = raw.components(separatedBy: ":").dropFirst().joined(separator: ":").trimmingCharacters(in: .whitespacesAndNewlines)
            return t("에이전트 단계: \(value)", "Agent step: \(value)", "エージェントステップ: \(value)")
        }
        if lowered.hasPrefix("focus applied:") {
            let value = raw.components(separatedBy: ":").dropFirst().joined(separator: ":").trimmingCharacters(in: .whitespacesAndNewlines)
            return t("포커스 적용: \(value)", "Focus applied: \(value)", "フォーカス適用: \(value)")
        }
        if lowered.hasPrefix("summary scope:") {
            let value = raw.components(separatedBy: ":").dropFirst().joined(separator: ":").trimmingCharacters(in: .whitespacesAndNewlines)
            return t("요약 범위: \(value)", "Summary scope: \(value)", "要約範囲: \(value)")
        }
        if lowered.hasPrefix("assistive retrieval:") {
            let value = raw.components(separatedBy: ":").dropFirst().joined(separator: ":").trimmingCharacters(in: .whitespacesAndNewlines)
            return t("보조 검색: \(value)", "Assistive retrieval: \(value)", "補助検索: \(value)")
        }
        return raw
    }

    private var composer: some View {
        return VStack(spacing: 8) {
            composerInputField()
            if !viewModel.composerAttachments.isEmpty {
                attachmentChips
            }

            HStack(spacing: 8) {
                Button {
                    viewModel.attachFileIntoComposer()
                } label: {
                    Image(systemName: "plus")
                        .font(.system(size: 15, weight: .semibold))
                        .frame(width: 18, height: 18)
                        .frame(width: 40, height: 40)
                        .plosGlassCircle()
                }
                .buttonStyle(.plain)
                .disabled(!viewModel.isChatComposerEnabled || viewModel.isBusy)
                .opacity((!viewModel.isChatComposerEnabled || viewModel.isBusy) ? 0.45 : 1.0)

                Menu {
                    if viewModel.installedModelsSorted.isEmpty {
                        Text(t("설치된 모델이 없습니다.", "No installed models.", "インストール済みモデルがありません。"))
                    }
                    ForEach(viewModel.installedModelsSorted) { model in
                        Button {
                            Task { await viewModel.selectInstalledModel(model) }
                        } label: {
                            HStack {
                                Text(model.file_name)
                                Text(model.engine.title)
                                if viewModel.isInstalledModelActive(model) {
                                    Image(systemName: "checkmark")
                                }
                            }
                        }
                    }
                } label: {
                    Image(systemName: "cpu")
                        .font(.system(size: 15, weight: .semibold))
                        .frame(width: 18, height: 18)
                        .frame(width: 40, height: 40)
                        .plosGlassCircle()
                }
                .menuStyle(.borderlessButton)
                .buttonStyle(.plain)

                Text(viewModel.activeModelDisplayName)
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .lineLimit(1)

                ForEach(viewModel.composerPluginToggles) { toggle in
                    Toggle(
                        toggle.title,
                        isOn: viewModel.pluginToggleBinding(
                            pluginID: toggle.pluginID,
                            toggleID: toggle.toggleID,
                            defaultValue: toggle.defaultValue
                        )
                    )
                    .toggleStyle(.checkbox)
                    .controlSize(.small)
                    .font(.caption)
                    .disabled(!toggle.pluginEnabled)
                    .help(toggle.help ?? toggle.pluginID)
                }

                if viewModel.isModelRuntimeBusy {
                    ProgressView()
                        .controlSize(.small)
                }

                Spacer()

                Button {
                    viewModel.startSystemDictation()
                } label: {
                    Image(systemName: "mic")
                        .font(.system(size: 15, weight: .semibold))
                        .frame(width: 18, height: 18)
                        .frame(width: 40, height: 40)
                        .plosGlassCircle()
                }
                .buttonStyle(.plain)

                Button {
                    viewModel.submitOrStopLocalChatFromComposer()
                } label: {
                    Image(systemName: viewModel.isGeneratingChatResponse ? "stop.fill" : "arrow.up")
                        .font(.system(size: 15, weight: .bold))
                        .foregroundStyle(.white)
                        .frame(width: 18, height: 18)
                        .frame(width: 40, height: 40)
                        .plosGlassCircle()
                }
                .buttonStyle(.plain)
                .disabled(
                    viewModel.isGeneratingChatResponse
                        ? false
                        : (!viewModel.isChatComposerEnabled || !viewModel.canSubmitChatInput || viewModel.isBusy)
                )
                .opacity(
                    viewModel.isGeneratingChatResponse
                        ? 1.0
                        : (!viewModel.isChatComposerEnabled || !viewModel.canSubmitChatInput || viewModel.isBusy ? 0.45 : 1.0)
                )
            }
        }
        .padding(12)
        .frame(maxWidth: 920)
        .plosGlassPanel(radius: 14)
    }

    private var attachmentChips: some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(viewModel.composerAttachments) { attachment in
                    HStack(spacing: 6) {
                        Image(systemName: chipSymbol(for: attachment.kind))
                            .font(.system(size: 10, weight: .semibold))
                        Text(attachment.fileName)
                            .lineLimit(1)
                        Button {
                            viewModel.removeComposerAttachment(attachment.id)
                        } label: {
                            Image(systemName: "xmark")
                                .font(.system(size: 9, weight: .bold))
                        }
                        .buttonStyle(.plain)
                        .foregroundStyle(.secondary)
                    }
                    .font(.caption)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .background(Color.secondary.opacity(0.12), in: Capsule())
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func chipSymbol(for kind: ChatAttachmentKind) -> String {
        switch kind {
        case .file:
            return "doc"
        case .image:
            return "photo"
        case .audio:
            return "waveform"
        }
    }

    private func composerInputField() -> some View {
        let placeholder = viewModel.isChatComposerEnabled
            ? t("무엇이든 부탁하세요", "Ask anything", "何でも聞いてください")
            : t("로컬 서버 준비 중입니다...", "Preparing local server...", "ローカルサーバーを準備中です...")
        return ZStack(alignment: .topLeading) {
            ComposerMultilineTextView(
                text: $viewModel.inputQuery,
                onSubmit: {
                    if viewModel.isGeneratingChatResponse {
                        viewModel.stopActiveLocalChatGeneration()
                        return
                    }
                    guard viewModel.isChatComposerEnabled, !viewModel.isBusy else { return }
                    viewModel.submitOrStopLocalChatFromComposer()
                }
            )
            .frame(height: 44)
            .padding(.horizontal, 12)
            .padding(.vertical, 6)
            .disabled(!viewModel.isChatComposerEnabled || viewModel.isBusy)

            if viewModel.inputQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                Text(placeholder)
                    .font(.body)
                    .foregroundStyle(.secondary.opacity(0.75))
                    .padding(.horizontal, 12)
                    .padding(.vertical, 10)
                    .allowsHitTesting(false)
            }
        }
        .plosGlassInputFrame(radius: 14)
    }

    @ViewBuilder
    private func markdownText(_ raw: String, font: Font) -> some View {
        let normalized = ChatPanelMarkdownFormatter.normalizeMarkdownForRender(raw)
        #if canImport(MarkdownUI)
            Markdown(normalized)
                .textSelection(.enabled)
        #else
        if let attributed = try? AttributedString(markdown: normalized) {
            let hasLink = attributed.runs.contains(where: { $0.link != nil })
            if hasLink {
                AnyView(
                    Text(attributed)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.disabled)
                )
            } else {
                AnyView(
                    Text(attributed)
                        .fixedSize(horizontal: false, vertical: true)
                        .textSelection(.enabled)
                )
            }
        } else {
            Text(normalized)
                .font(font)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }
        #endif
    }

    @ViewBuilder
    private func markdownRichContent(_ raw: String, font: Font, keyPrefix: String) -> some View {
        let segments = ChatPanelMarkdownFormatter.markdownSegments(from: raw, keyPrefix: keyPrefix)
        VStack(alignment: .leading, spacing: 8) {
            ForEach(segments) { segment in
                switch segment.kind {
                case .text:
                    if !segment.content.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        markdownText(segment.content, font: font)
                    }
                case .code:
                    codeBlockView(
                        code: segment.content,
                        language: segment.language,
                        blockID: segment.id
                    )
                }
            }
        }
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder
    private func codeBlockView(code: String, language: String?, blockID: String) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack(spacing: 8) {
                if let language, !language.isEmpty {
                    Text(language)
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
                Spacer(minLength: 0)
                Button {
                    copyTextToPasteboard(code)
                } label: {
                    Image(systemName: "doc.on.doc")
                        .font(.system(size: 11, weight: .semibold))
                        .frame(width: 14, height: 14)
                        .frame(width: 28, height: 28)
                        .plosGlassCircle()
                }
                .buttonStyle(.plain)
                .contentShape(Circle())
                .opacity(hoveredCodeBlockIDs.contains(blockID) ? 1.0 : 0.58)
                .onHover { isHovering in
                    if isHovering {
                        hoveredCodeBlockIDs.insert(blockID)
                    } else {
                        hoveredCodeBlockIDs.remove(blockID)
                    }
                }
                .help("코드 복사")
            }
            Text(highlightedCodeAttributedString(code, language: language))
                .frame(maxWidth: .infinity, alignment: .leading)
                .textSelection(.enabled)
                .fixedSize(horizontal: false, vertical: true)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .plosGlassInputFrame(radius: 10)
    }

    private func highlightedCodeAttributedString(_ code: String, language: String?) -> AttributedString {
        let text = code.isEmpty ? " " : code
        let ns = text as NSString
        let output = NSMutableAttributedString(
            string: text,
            attributes: [
                .font: NSFont.monospacedSystemFont(ofSize: 13, weight: .regular),
                .foregroundColor: NSColor.labelColor,
            ]
        )

        func apply(pattern: String, color: NSColor, options: NSRegularExpression.Options = []) {
            guard let regex = try? NSRegularExpression(pattern: pattern, options: options) else { return }
            let range = NSRange(location: 0, length: ns.length)
            regex.matches(in: text, options: [], range: range).forEach { match in
                output.addAttribute(.foregroundColor, value: color, range: match.range)
            }
        }

        let normalizedLanguage = (language ?? "").trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let keywordPattern: String = {
            if normalizedLanguage.contains("py") {
                return #"\b(?:def|class|import|from|as|if|elif|else|for|while|try|except|finally|return|with|lambda|yield|async|await|True|False|None|and|or|not|in|is|pass|break|continue|raise)\b"#
            }
            if normalizedLanguage.contains("swift") {
                return #"\b(?:func|let|var|struct|class|enum|protocol|extension|import|if|else|guard|for|while|switch|case|default|return|throw|throws|try|catch|async|await|nil|true|false|self|init|deinit)\b"#
            }
            return #"\b(?:function|const|let|var|class|import|export|from|if|else|for|while|switch|case|default|return|try|catch|finally|throw|new|async|await|true|false|null|undefined)\b"#
        }()

        apply(pattern: keywordPattern, color: NSColor.systemBlue)
        apply(pattern: #"\b\d+(?:\.\d+)?\b"#, color: NSColor.systemOrange)
        apply(pattern: #"\"(?:\\.|[^\"\\])*\"|'(?:\\.|[^'\\])*'"#, color: NSColor.systemRed)
        apply(pattern: #"//.*$"#, color: NSColor.systemGreen, options: [.anchorsMatchLines])
        apply(pattern: #"#.*$"#, color: NSColor.systemGreen, options: [.anchorsMatchLines])
        apply(pattern: #"/\*[\s\S]*?\*/"#, color: NSColor.systemGreen)

        if let converted = try? AttributedString(output, including: \.appKit) {
            return converted
        }
        return AttributedString(text)
    }

    @ViewBuilder
    private func assistantMessageCopyButton(for message: ChatMessage) -> some View {
        let payload = assistantMessageBodyText(for: message)
        if !payload.isEmpty {
            Button {
                copyTextToPasteboard(payload)
            } label: {
                Image(systemName: "doc.on.doc")
                    .font(.system(size: 11, weight: .semibold))
                    .frame(width: 14, height: 14)
                    .frame(width: 28, height: 28)
                    .plosGlassCircle()
            }
            .buttonStyle(.plain)
            .contentShape(Circle())
            .opacity(hoveredAssistantCopyMessageIDs.contains(message.id) ? 1.0 : 0.56)
            .onHover { isHovering in
                if isHovering {
                    hoveredAssistantCopyMessageIDs.insert(message.id)
                } else {
                    hoveredAssistantCopyMessageIDs.remove(message.id)
                }
            }
            .help("메시지 복사")
        }
    }

    private func assistantMessageBodyText(for message: ChatMessage) -> String {
        switch message.source {
        case .user:
            return ""
        case .external:
            return nfc(message.text ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
        case .local:
            let text = nfc(message.text ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            let lead = nfc(message.lead ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            let summary = nfc(message.resultSummary ?? "").trimmingCharacters(in: .whitespacesAndNewlines)
            return [text, lead, summary]
                .filter { !$0.isEmpty }
                .joined(separator: "\n\n")
                .trimmingCharacters(in: .whitespacesAndNewlines)
        }
    }

    private func copyTextToPasteboard(_ raw: String) {
        let text = nfc(raw).trimmingCharacters(in: .whitespacesAndNewlines)
        guard !text.isEmpty else { return }
        let pasteboard = NSPasteboard.general
        pasteboard.clearContents()
        pasteboard.setString(text, forType: .string)
    }

    private func animatedText(for message: ChatMessage, field: String, raw: String) -> String {
        if message.isStreaming {
            return raw
        }
        let key = revealKey(messageID: message.id, field: field)
        if let revealed = revealedCharactersByKey[key] {
            let safeCount = max(0, min(revealed, raw.count))
            if safeCount < raw.count, !animatingRevealKeys.contains(key) {
                return raw
            }
            return String(raw.prefix(safeCount))
        }
        return shouldAnimateMessageReveal(message) ? "" : raw
    }

    private func primeRevealState(for message: ChatMessage) {
        guard message.source != .user else { return }

        if let text = message.text, !text.isEmpty {
            startReveal(for: message, field: "text", raw: nfc(text))
        }
        if let lead = message.lead, !lead.isEmpty {
            startReveal(for: message, field: "lead", raw: nfc(lead))
        }
        if let summary = message.resultSummary, !summary.isEmpty {
            startReveal(for: message, field: "summary", raw: nfc(summary))
        }
    }

    private func startReveal(for message: ChatMessage, field: String, raw: String) {
        let key = revealKey(messageID: message.id, field: field)
        let total = raw.count
        guard total > 0 else {
            revealedCharactersByKey[key] = 0
            return
        }

        if revealedCharactersByKey[key] != nil || animatingRevealKeys.contains(key) {
            return
        }

        guard shouldAnimateMessageReveal(message), shouldAnimateFieldReveal(raw: raw) else {
            revealedCharactersByKey[key] = total
            return
        }

        revealedCharactersByKey[key] = 0
        animatingRevealKeys.insert(key)

        Task { @MainActor in
            var current = 0
            let step = max(1, min(6, total / 140))
            let delayNs: UInt64
            if total > 900 {
                delayNs = 8_000_000
            } else if total > 300 {
                delayNs = 12_000_000
            } else {
                delayNs = 16_000_000
            }

            while current < total {
                current = min(total, current + step)
                revealedCharactersByKey[key] = current
                try? await Task.sleep(nanoseconds: delayNs)
            }

            animatingRevealKeys.remove(key)
            revealedCharactersByKey[key] = total
        }
    }

    private func shouldAnimateMessageReveal(_ message: ChatMessage) -> Bool {
        guard message.source != .user else { return false }
        let isFresh = Date().timeIntervalSince(message.timestamp) < 2.2
        let isLatestAssistant = message.id == latestAssistantMessageID
        return isFresh && isLatestAssistant
    }

    private func shouldAnimateFieldReveal(raw: String) -> Bool {
        let normalized = nfc(raw)
        if normalized.count > 520 {
            return false
        }
        if normalized.contains("```") || normalized.contains("~~~") {
            return false
        }
        return true
    }

    private func revealKey(messageID: UUID, field: String) -> String {
        "\(messageID.uuidString):\(field)"
    }

    private func synchronizeRevealState(for message: ChatMessage) {
        guard message.source != .user else { return }
        let fields: [(String, String)] = [
            ("text", nfc(message.text ?? "")),
            ("lead", nfc(message.lead ?? "")),
            ("summary", nfc(message.resultSummary ?? "")),
        ]
        for (field, raw) in fields {
            let key = revealKey(messageID: message.id, field: field)
            if raw.isEmpty {
                revealedCharactersByKey[key] = nil
                animatingRevealKeys.remove(key)
                continue
            }
            revealedCharactersByKey[key] = raw.count
            animatingRevealKeys.remove(key)
        }
    }

    private var latestLocalMessageID: UUID? {
        viewModel.chatMessages.last(where: { $0.source == .local })?.id
    }

    private var latestAssistantMessageID: UUID? {
        viewModel.chatMessages.last(where: { $0.source == .local || $0.source == .external })?.id
    }

    private var latestAssistantRevealSignature: String {
        guard let message = viewModel.chatMessages.last(where: { $0.source == .local || $0.source == .external }) else {
            return ""
        }
        let textCount = nfc(message.text ?? "").count
        let leadCount = nfc(message.lead ?? "").count
        let summaryCount = nfc(message.resultSummary ?? "").count
        return "\(message.id.uuidString)|\(message.isStreaming ? 1 : 0)|\(textCount)|\(leadCount)|\(summaryCount)"
    }

    private func citationsForMessage(_ message: ChatMessage) -> [Citation] {
        guard message.source == .local, message.id == latestLocalMessageID else {
            return []
        }
        return Array(viewModel.citations.prefix(3))
    }

    private func miniCitationStrip(_ citations: [Citation]) -> some View {
        ScrollView(.horizontal, showsIndicators: false) {
            HStack(spacing: 6) {
                ForEach(Array(citations.enumerated()), id: \.offset) { index, citation in
                    Button {
                        viewModel.highlightedCitationPath = citation.file_path
                    } label: {
                        HStack(spacing: 4) {
                            Text("[\(index + 1)]")
                            Text(nfc(URL(fileURLWithPath: citation.file_path).lastPathComponent))
                                .lineLimit(1)
                        }
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                    }
                    .buttonStyle(.plain)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .plosGlassChip()
                    .help(citation.file_path)
                }
            }
        }
        .padding(.top, 2)
    }

    private func nfc(_ value: String) -> String {
        ChatPanelMarkdownFormatter.nfc(value)
    }
}
