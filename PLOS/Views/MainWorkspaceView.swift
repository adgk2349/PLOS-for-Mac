import SwiftUI

private enum WorkspaceSection: String, CaseIterable, Identifiable {
    case chat
    case status
    case settings

    var id: String { rawValue }

    var title: String {
        switch self {
        case .chat:
            return "Chat"
        case .status:
            return "상태"
        case .settings:
            return "설정"
        }
    }

    var systemImage: String {
        switch self {
        case .chat:
            return "message"
        case .status:
            return "waveform.path.ecg"
        case .settings:
            return "gearshape"
        }
    }
}

struct MainWorkspaceView: View {
    @ObservedObject var viewModel: AppViewModel
    @State private var selectedSection: WorkspaceSection = .chat
    @State private var sidebarSearch = ""

    var body: some View {
        NavigationSplitView {
            sidebar
                .padding(12)
        } detail: {
            VStack(spacing: 12) {
                detailHeader
                detailContent
            }
            .padding(12)
        }
        .navigationSplitViewStyle(.balanced)
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 12) {
                Circle()
                    .fill(Color.red)
                    .frame(width: 12, height: 12)
                Circle()
                    .fill(Color.yellow)
                    .frame(width: 12, height: 12)
                Circle()
                    .fill(Color.green)
                    .frame(width: 12, height: 12)

                Spacer()

                Image(systemName: "rectangle.leadinghalf.inset.filled")
                    .foregroundStyle(.secondary)
                Image(systemName: "square.and.pencil")
                    .foregroundStyle(.secondary)
            }

            HStack(spacing: 8) {
                Image(systemName: "magnifyingglass")
                    .foregroundStyle(.secondary)
                TextField("검색", text: $sidebarSearch)
                    .textFieldStyle(.plain)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 10)
            .glassCard(cornerRadius: 12)

            VStack(alignment: .leading, spacing: 8) {
                sidebarButton(.chat)
                sidebarButton(.status)
                sidebarButton(.settings)
            }

            Divider()

            ScrollView {
                VStack(alignment: .leading, spacing: 6) {
                    Text("최근 대화")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.bottom, 4)

                    ForEach(filteredRecentPrompts, id: \.self) { prompt in
                        Text(prompt)
                            .font(.subheadline)
                            .lineLimit(1)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 8)
                            .glassEffect(
                                .regular.tint(Color.white.opacity(0.08)),
                                in: RoundedRectangle(cornerRadius: 10, style: .continuous)
                            )
                    }
                }
            }

            Spacer()

            HStack(spacing: 10) {
                Circle()
                    .fill(Color.orange)
                    .frame(width: 30, height: 30)
                    .overlay(
                        Text("LM")
                            .font(.caption2.weight(.bold))
                            .foregroundStyle(.white)
                    )
                Text("Lee Seung Min")
                    .font(.headline)
            }
            .padding(.top, 6)
        }
        .padding(14)
        .frame(minWidth: 280, idealWidth: 320, maxWidth: 360)
        .glassCard(cornerRadius: 24)
    }

    private var detailHeader: some View {
        HStack {
            Text("ChatGPT Auto")
                .font(.title3.weight(.semibold))
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .glassCard(cornerRadius: 16)

            Spacer()

            HStack(spacing: 8) {
                routeBadge
                Button {
                    selectedSection = .chat
                } label: {
                    Image(systemName: "square.and.arrow.up")
                }
                .buttonStyle(.glass)

                Button {
                    selectedSection = .chat
                } label: {
                    Image(systemName: "doc.on.doc")
                }
                .buttonStyle(.glass)
            }
        }
        .padding(10)
        .glassCard(cornerRadius: 18)
    }

    @ViewBuilder
    private var routeBadge: some View {
        switch viewModel.currentRoute {
        case .local:
            Label("로컬", systemImage: "lock.shield.fill")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.green)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .glassCard(cornerRadius: 14)
        case .external:
            Label("외부", systemImage: "cloud.fill")
                .font(.caption.weight(.semibold))
                .foregroundStyle(.orange)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .glassCard(cornerRadius: 14)
        }
    }

    @ViewBuilder
    private var detailContent: some View {
        switch selectedSection {
        case .chat:
            ChatPanelView(viewModel: viewModel)
                .glassCard(cornerRadius: 22)
        case .status:
            StatusPanelView(viewModel: viewModel)
                .glassCard(cornerRadius: 22)
        case .settings:
            SettingsPanelView(viewModel: viewModel)
                .glassCard(cornerRadius: 22)
        }
    }

    private var recentPrompts: [String] {
        var seen = Set<String>()
        let prompts = viewModel.chatMessages
            .filter { $0.source == .user }
            .compactMap(\.text)
            .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        var unique: [String] = []
        for prompt in prompts.reversed() where !seen.contains(prompt) {
            unique.append(prompt)
            seen.insert(prompt)
        }
        return unique
    }

    private var filteredRecentPrompts: [String] {
        let needle = sidebarSearch.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        if needle.isEmpty {
            return Array(recentPrompts.prefix(14))
        }
        return recentPrompts
            .filter { $0.lowercased().contains(needle) }
            .prefix(14)
            .map { $0 }
    }

    private func sidebarButton(_ section: WorkspaceSection) -> some View {
        let selected = selectedSection == section
        return Button {
            selectedSection = section
        } label: {
            HStack(spacing: 10) {
                Image(systemName: section.systemImage)
                    .frame(width: 18)
                Text(section.title)
                    .font(.headline)
                Spacer()
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .frame(maxWidth: .infinity, alignment: .leading)
            .if(selected) { view in
                view.glassEffect(
                    .regular.tint(Color.white.opacity(0.16)),
                    in: RoundedRectangle(cornerRadius: 10, style: .continuous)
                )
            }
        }
        .buttonStyle(.plain)
    }
}

struct ChatPanelView: View {
    @ObservedObject var viewModel: AppViewModel

    var body: some View {
        GeometryReader { proxy in
            let useSplitLayout = proxy.size.width >= 1060

            VStack(spacing: 12) {
                controls
                    .glassCard(cornerRadius: 14)

                if useSplitLayout {
                    HStack(alignment: .top, spacing: 12) {
                        messageAndComposer
                        citationPanel
                            .frame(width: 320)
                    }
                } else {
                    messageAndComposer
                    if viewModel.isCitationDrawerVisible {
                        citationPanel
                            .transition(.move(edge: .bottom).combined(with: .opacity))
                    }
                }
            }
            .animation(.easeInOut(duration: 0.2), value: viewModel.isCitationDrawerVisible)
            .padding(16)
        }
        .confirmationDialog("외부 AI 호출 전 확인", isPresented: $viewModel.needsExternalConfirmation) {
            Button("승인하고 실행") {
                Task {
                    await viewModel.performDeepAnalysis(userConfirmed: true)
                }
            }
            Button("취소", role: .cancel) {}
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

    private var controls: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Picker("작업 모드", selection: $viewModel.selectedMode) {
                    ForEach(WorkMode.allCases) { mode in
                        Text(mode.title).tag(mode)
                    }
                }
                .frame(maxWidth: 420)

                Picker("제공자", selection: $viewModel.selectedProvider) {
                    Text("OpenAI").tag("openai")
                    Text("Anthropic").tag("anthropic")
                }
                .frame(width: 180)

                Spacer()

                Button("더 깊게 분석") {
                    viewModel.deepAnalyzeTapped()
                }
                .buttonStyle(.glassProminent)
                .disabled(viewModel.chatMessages.isEmpty || viewModel.isBusy)
            }

            HStack(spacing: 8) {
                Picker("카테고리", selection: $viewModel.chatFilterCategory) {
                    Text("전체").tag("")
                    ForEach(AppViewModel.fixedCategories, id: \.self) { category in
                        Text(category).tag(category)
                    }
                }
                .frame(width: 160)

                TextField("태그 (예: Swift,RAG)", text: $viewModel.chatFilterTags)
                    .textFieldStyle(.roundedBorder)

                TextField("연도", text: $viewModel.chatFilterYear)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 80)

                TextField("프로젝트", text: $viewModel.chatFilterProject)
                    .textFieldStyle(.roundedBorder)

                Button(viewModel.isCitationDrawerVisible ? "출처 숨기기" : "출처 보기") {
                    viewModel.isCitationDrawerVisible.toggle()
                }
                .buttonStyle(.glass)
            }
        }
        .padding(12)
    }

    private var messageAndComposer: some View {
        VStack(spacing: 10) {
            ScrollView {
                LazyVStack(alignment: .leading, spacing: 10) {
                    ForEach(viewModel.chatMessages) { message in
                        HStack(alignment: .top, spacing: 10) {
                            Text(label(for: message.source))
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)
                                .frame(width: 54, alignment: .leading)

                            messageBody(message)
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(12)
                                .glassTint(background(for: message.source), cornerRadius: 12)
                        }
                    }
                }
            }
            .glassCard(cornerRadius: 14)

            composer
        }
    }

    private var citationPanel: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                Text("출처")
                    .font(.headline)
                Spacer()
                Text("\(viewModel.citations.count)")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if viewModel.citations.isEmpty {
                Text("출처가 아직 없습니다.")
                    .foregroundStyle(.secondary)
                    .padding(.vertical, 12)
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 8) {
                        ForEach(viewModel.citations) { citation in
                            VStack(alignment: .leading, spacing: 6) {
                                Text(citation.file_path)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                                Text(citation.snippet)
                                    .font(.subheadline)
                                    .lineLimit(3)
                                Text("카테고리 \(citation.category) · 점수 \(String(format: "%.3f", citation.score))")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                if !citation.tags.isEmpty {
                                    Text(citation.tags.joined(separator: ", "))
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(2)
                                }
                            }
                            .padding(10)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .glassEffect(
                                .regular.tint(viewModel.highlightedCitationPath == citation.file_path ? Color.white.opacity(0.20) : Color.white.opacity(0.08)),
                                in: RoundedRectangle(cornerRadius: 10, style: .continuous)
                            )
                        }
                    }
                }
            }
        }
        .padding(12)
        .glassCard(cornerRadius: 14)
    }

    private var composer: some View {
        HStack {
            TextField("무엇이든 부탁하세요", text: $viewModel.inputQuery, axis: .vertical)
                .lineLimit(1 ... 6)
                .textFieldStyle(.plain)
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .glassEffect(
                    .regular.tint(Color.white.opacity(0.10)),
                    in: RoundedRectangle(cornerRadius: 10, style: .continuous)
                )

            Button("전송") {
                Task {
                    await viewModel.askLocal()
                }
            }
            .buttonStyle(.glassProminent)
            .disabled(viewModel.inputQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || viewModel.isBusy)
        }
        .padding(10)
        .glassCard(cornerRadius: 14)
    }

    private func label(for source: ChatMessage.Source) -> String {
        switch source {
        case .user:
            return "USER"
        case .local:
            return "LOCAL"
        case .external:
            return "EXT"
        }
    }

    @ViewBuilder
    private func messageBody(_ message: ChatMessage) -> some View {
        if message.source == .local {
            VStack(alignment: .leading, spacing: 8) {
                if let lead = message.lead, !lead.isEmpty {
                    Text(lead)
                        .font(.body.weight(.semibold))
                }
                if let summary = message.resultSummary, !summary.isEmpty {
                    Text(summary)
                        .font(.body)
                }
                if let reasoning = message.reasoningBrief, !reasoning.isEmpty {
                    Text(reasoning)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .padding(.top, 2)
                }
                if !message.actions.isEmpty {
                    LazyVGrid(columns: [GridItem(.adaptive(minimum: 96), spacing: 8)], spacing: 8) {
                        ForEach(message.actions) { action in
                            Button(action.label) {
                                Task {
                                    await viewModel.executeAction(action)
                                }
                            }
                            .buttonStyle(.glass)
                        }
                    }
                    .padding(.top, 4)
                }
            }
        } else {
            Text(message.text ?? "")
        }
    }

    private func background(for source: ChatMessage.Source) -> Color {
        switch source {
        case .user:
            return GlassTheme.userTint
        case .local:
            return GlassTheme.localTint
        case .external:
            return GlassTheme.externalTint
        }
    }
}

private extension View {
    @ViewBuilder
    func `if`<Transformed: View>(_ condition: Bool, transform: (Self) -> Transformed) -> some View {
        if condition {
            transform(self)
        } else {
            self
        }
    }
}

