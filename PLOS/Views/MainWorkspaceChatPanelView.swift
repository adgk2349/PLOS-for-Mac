import Foundation
import SwiftUI

struct ChatPanelView: View {
    @ObservedObject var viewModel: AppViewModel
    @Binding var isSidebarCollapsed: Bool
    @State private var showPresetMenu = false
    @State private var expandedReasoningMessageIDs: Set<UUID> = []
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        ZStack(alignment: .top) {
            conversationColumn
            topChatBlendOverlay

            header
                .padding(.top, 12)
                .padding(.horizontal, 10)
        }
        .padding(10)
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .confirmationDialog("외부 AI 호출 전 확인", isPresented: $viewModel.needsExternalConfirmation) {
            Button("승인하고 실행") {
                viewModel.confirmExternalCall()
            }
            Button("취소", role: .cancel) {
                viewModel.cancelExternalCall()
            }
        } message: {
            Text("선택된 자료 일부가 외부 제공자에 전달될 수 있습니다.")
        }
        .confirmationDialog(
            "시스템 액션 승인",
            isPresented: Binding(
                get: { viewModel.pendingSystemAction != nil },
                set: { shown in
                    if !shown {
                        viewModel.cancelPendingSystemAction()
                    }
                }
            )
        ) {
            Button("허용하고 실행") {
                viewModel.confirmPendingSystemAction()
            }
            Button("취소", role: .cancel) {
                viewModel.cancelPendingSystemAction()
            }
        } message: {
            if let path = viewModel.pendingSystemAction?.payload["file_path"], !path.isEmpty {
                Text("로컬 파일을 엽니다: \(path)")
            } else {
                Text("시스템 액션 실행 권한이 필요합니다.")
            }
        }
    }

    private var header: some View {
        HStack(spacing: 10) {
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
                        .plosGlassCircle()
                }
                .buttonStyle(.plain)
            }

            Button {
                showPresetMenu.toggle()
            } label: {
                HStack(spacing: 8) {
                    Text("PLOS")
                        .font(.headline.weight(.semibold))
                    Text(viewModel.quickInferencePreset.title)
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
                            .regular.tint(Color.white.opacity(0.10)),
                            in: Capsule(style: .continuous)
                        )
                }
                .overlay {
                    Capsule(style: .continuous)
                        .stroke(Color.white.opacity(0.28), lineWidth: 1)
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
                                    Text(preset.title)
                                        .font(.subheadline.weight(.semibold))
                                    Text(preset.detail)
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
                }
                .padding(10)
                .frame(width: 280)
                .background(PLOSGlassBackground())
            }

            Spacer(minLength: 12)

            HStack(spacing: 10) {
                ShareLink(item: viewModel.currentChatTranscript.isEmpty ? "PLOS" : viewModel.currentChatTranscript) {
                    Image(systemName: "square.and.arrow.up")
                        .font(.system(size: 13, weight: .semibold))
                        .frame(width: 18, height: 18)
                        .frame(width: 40, height: 40)
                        .plosGlassCircle()
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
                        .plosGlassCircle()
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
        ZStack(alignment: .bottom) {
            ScrollView {
                LazyVStack(spacing: 12) {
                    ForEach(viewModel.chatMessages) { message in
                        messageRow(message)
                    }
                }
                .frame(maxWidth: 920)
                .frame(maxWidth: .infinity)
                .padding(.top, 72)
                .padding(.horizontal, 6)
                .padding(.bottom, 170)
            }
            .scrollIndicators(.hidden)
            .frame(maxWidth: .infinity, maxHeight: .infinity)

            composer
                .padding(.horizontal, 6)
                .padding(.bottom, 8)
        }
    }

    private var topChatBlendOverlay: some View {
        let tint = PLOSGlassTheme.chromeTint(for: colorScheme)

        return Rectangle()
            .fill(.ultraThinMaterial)
            .overlay(tint)
            .mask(
                LinearGradient(
                    colors: [
                        Color.white,
                        Color.white.opacity(0.82),
                        Color.white.opacity(0.48),
                        Color.clear,
                    ],
                    startPoint: .top,
                    endPoint: .bottom
                )
            )
            .blur(radius: 6)
            .frame(height: 108)
            .offset(y: -8)
            .allowsHitTesting(false)
    }

    private func messageRow(_ message: ChatMessage) -> some View {
        HStack {
            messageContainer(message)
        }
        .frame(maxWidth: .infinity, alignment: message.source == .user ? .trailing : .leading)
    }

    @ViewBuilder
    private func messageContainer(_ message: ChatMessage) -> some View {
        switch message.source {
        case .user:
            let raw = nfc(message.text ?? "")
            let compact = isCompactUserBubble(raw)

            if compact {
                VStack(alignment: .trailing, spacing: 0) {
                    markdownText(raw, font: .body)
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
                    markdownText(raw, font: .body)
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
                Text("External")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
                markdownText(nfc(message.text ?? ""), font: .body)
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
        viewModel.isBusy && message.id == latestLocalMessageID
    }

    private func toggleReasoningExpansion(for message: ChatMessage) {
        if expandedReasoningMessageIDs.contains(message.id) {
            expandedReasoningMessageIDs.remove(message.id)
        } else {
            expandedReasoningMessageIDs.insert(message.id)
        }
    }

    private func reasoningDetailText(for message: ChatMessage) -> String {
        var sections: [String] = []

        if let brief = message.reasoningBrief,
           !brief.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            sections.append("요약 판단\n\(brief)")
        }

        if let parsed = message.parsedIntent {
            var rows: [String] = []
            rows.append("의도: \(parsed.intent.rawValue)")
            rows.append(String(format: "의도 신뢰도: %.2f", parsed.confidence))
            if let operation = parsed.operation, !operation.isEmpty {
                rows.append("작업 연산: \(operation)")
            }
            if let scope = parsed.scope, !scope.isEmpty {
                rows.append("요청 범위: \(scope)")
            }
            if let target = parsed.target, !target.isEmpty {
                rows.append("요청 대상: \(target)")
            }
            if let ambiguity = parsed.ambiguity, !ambiguity.isEmpty {
                rows.append("모호성 판단: \(ambiguity)")
            }

            if !parsed.entities.file_names.isEmpty {
                rows.append("파일 엔티티: \(parsed.entities.file_names.joined(separator: ", "))")
            }
            if !parsed.entities.tags.isEmpty {
                rows.append("태그 엔티티: \(parsed.entities.tags.joined(separator: ", "))")
            }
            if !parsed.entities.topics.isEmpty {
                rows.append("주제 엔티티: \(parsed.entities.topics.joined(separator: ", "))")
            }
            if !parsed.entities.projects.isEmpty {
                rows.append("프로젝트 엔티티: \(parsed.entities.projects.joined(separator: ", "))")
            }
            if let year = parsed.time_filters.year {
                rows.append("연도 필터: \(year)")
            }
            if let from = parsed.time_filters.year_from, let to = parsed.time_filters.year_to {
                rows.append("기간 필터: \(from)-\(to)")
            }
            if let days = parsed.time_filters.relative_days {
                rows.append("상대 기간: 최근 \(days)일")
            }
            if !parsed.workspace_filters.included_paths.isEmpty {
                rows.append("포함 경로 수: \(parsed.workspace_filters.included_paths.count)")
            }
            if !parsed.workspace_filters.excluded_paths.isEmpty {
                rows.append("제외 경로 수: \(parsed.workspace_filters.excluded_paths.count)")
            }

            sections.append(rows.joined(separator: "\n"))
        }

        if let plan = message.plan {
            var rows: [String] = []
            rows.append("계획 유형: \(plan.plan_type)")
            rows.append("응답 전략: \(plan.response_strategy)")
            rows.append("선택 파일 수: \(plan.selected_files.count)")
            rows.append("선택 청크 수: \(plan.selected_chunks.count)")
            if !plan.allowed_actions.isEmpty {
                rows.append("허용 액션: \(plan.allowed_actions.map(\.rawValue).joined(separator: ", "))")
            }
            rows.append("외부 추론 필요: \(plan.external_reasoning_needed ? "예" : "아니오")")
            sections.append(rows.joined(separator: "\n"))
        }

        if let verify = message.verification {
            var rows: [String] = []
            rows.append("검증 유효성: \(verify.is_valid ? "유효" : "재검토")")
            rows.append(String(format: "검증 신뢰도: %.2f", verify.confidence))
            rows.append(String(format: "모호성: %.2f", verify.ambiguity_level))
            rows.append("후보 모드: \(verify.candidate_mode ? "예" : "아니오")")
            if !verify.issues.isEmpty {
                rows.append("이슈: \(verify.issues.joined(separator: ", "))")
            }
            sections.append(rows.joined(separator: "\n"))
        }

        if sections.isEmpty {
            return "추론 상세 정보가 없습니다."
        }

        return sections.joined(separator: "\n\n")
    }

    @ViewBuilder
    private func localHeader(_ message: ChatMessage) -> some View {
        HStack(spacing: 8) {
            Text("Local AI")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.secondary)

            if message.responseMetadata?["conversation_path"] == .string("external_escalated") {
                Text("External Escalated")
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 8)
                    .padding(.vertical, 3)
                    .background(Color.orange.opacity(0.24), in: Rectangle())
            }
        }
    }

    @ViewBuilder
    private func localBody(_ message: ChatMessage) -> some View {
        if let lead = message.lead, !lead.isEmpty {
            markdownText(nfc(lead), font: .body.weight(.semibold))
        }
        if let summary = message.resultSummary, !summary.isEmpty {
            markdownText(nfc(summary), font: .body)
        }

        let smallCitations = citationsForMessage(message)
        if !smallCitations.isEmpty {
            miniCitationStrip(smallCitations)
        }

        if hasReasoningSignal(message) {
            if isReasoningInProgress(for: message) {
                Text("생각중")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
            } else {
                Button {
                    toggleReasoningExpansion(for: message)
                } label: {
                    Text("생각됨")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
                .buttonStyle(.plain)

                if expandedReasoningMessageIDs.contains(message.id) {
                    ScrollView {
                        Text(reasoningDetailText(for: message))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 8)
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

    private func hasReasoningSignal(_ message: ChatMessage) -> Bool {
        if message.parsedIntent != nil || message.plan != nil || message.verification != nil {
            return true
        }
        if let brief = message.reasoningBrief,
           !brief.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        {
            return true
        }
        return false
    }

    private var composer: some View {
        VStack(spacing: 8) {
            TextField("무엇이든 부탁하세요", text: $viewModel.inputQuery, axis: .vertical)
                .lineLimit(1 ... 8)
                .textFieldStyle(.plain)
                .padding(.horizontal, 14)
                .padding(.vertical, 10)
                .submitLabel(.send)
                .onSubmit {
                    guard !viewModel.isBusy else { return }
                    Task { await viewModel.askLocal() }
                }
                .plosGlassCapsule(tint: Color.white.opacity(0.02))

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

                Menu {
                    if viewModel.installedModelsSorted.isEmpty {
                        Text("설치된 모델이 없습니다.")
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

                if viewModel.isBusy {
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
                    Task { await viewModel.askLocal() }
                } label: {
                    Image(systemName: "arrow.up")
                        .font(.system(size: 15, weight: .bold))
                        .foregroundStyle(.white)
                        .frame(width: 18, height: 18)
                        .frame(width: 40, height: 40)
                        .plosGlassCircle()
                }
                .buttonStyle(.plain)
                .disabled(viewModel.inputQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || viewModel.isBusy)
                .opacity(viewModel.inputQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || viewModel.isBusy ? 0.45 : 1.0)
            }
        }
        .padding(12)
        .frame(maxWidth: 920)
        .plosGlassPanel(radius: 14)
    }

    @ViewBuilder
    private func markdownText(_ raw: String, font: Font) -> some View {
        let normalized = nfc(raw)
        if let attributed = try? AttributedString(markdown: normalized) {
            Text(attributed)
                .font(font)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        } else {
            Text(normalized)
                .font(font)
                .fixedSize(horizontal: false, vertical: true)
                .textSelection(.enabled)
        }
    }

    private var latestLocalMessageID: UUID? {
        viewModel.chatMessages.last(where: { $0.source == .local })?.id
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
        value.precomposedStringWithCanonicalMapping
    }
}
