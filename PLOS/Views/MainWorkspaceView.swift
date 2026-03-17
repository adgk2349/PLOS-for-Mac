import SwiftUI

struct MainWorkspaceView: View {
    @ObservedObject var viewModel: AppViewModel

    var body: some View {
        TabView {
            ChatPanelView(viewModel: viewModel)
                .tabItem {
                    Label("질의응답", systemImage: "message")
                }

            StatusPanelView(viewModel: viewModel)
                .tabItem {
                    Label("상태", systemImage: "gauge")
                }

            SettingsPanelView(viewModel: viewModel)
                .tabItem {
                    Label("설정", systemImage: "gearshape")
                }
        }
        .padding(8)
        .glassCard(cornerRadius: 16)
    }
}

struct ChatPanelView: View {
    @ObservedObject var viewModel: AppViewModel

    var body: some View {
        GeometryReader { proxy in
            let useSplitLayout = proxy.size.width >= 980

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
                .buttonStyle(.bordered)
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
                                .background(background(for: message.source), in: RoundedRectangle(cornerRadius: 12, style: .continuous))
                                .overlay(
                                    RoundedRectangle(cornerRadius: 12, style: .continuous)
                                        .stroke(Color.white.opacity(0.18), lineWidth: 1)
                                )
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
                            .background(
                                (viewModel.highlightedCitationPath == citation.file_path ? Color.white.opacity(0.18) : Color.white.opacity(0.08)),
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
            TextField("질문을 입력하세요", text: $viewModel.inputQuery, axis: .vertical)
                .lineLimit(1 ... 5)
                .textFieldStyle(.plain)
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .background(Color.white.opacity(0.10), in: RoundedRectangle(cornerRadius: 10, style: .continuous))

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
                            .buttonStyle(.bordered)
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
