import SwiftUI

struct ChatView: View {
    @ObservedObject var viewModel: AppViewModel

    var body: some View {
        VStack(spacing: 12) {
            topControls

            messageList

            citationList

            composer
        }
        .padding(16)
        .confirmationDialog(
            "외부 AI 호출 전 확인",
            isPresented: $viewModel.needsExternalConfirmation,
            titleVisibility: .visible
        ) {
            Button("승인하고 실행") {
                Task { await viewModel.performDeepAnalysis(userConfirmed: true) }
            }
            Button("취소", role: .cancel) {}
        } message: {
            Text("선택된 자료 일부가 외부 제공자에 전달될 수 있습니다.")
        }
    }

    private var topControls: some View {
        HStack {
            Picker("작업 모드", selection: $viewModel.selectedMode) {
                ForEach(WorkMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .frame(maxWidth: 480)

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
    }

    private var messageList: some View {
        ScrollView {
            LazyVStack(alignment: .leading, spacing: 10) {
                ForEach(viewModel.chatMessages) { message in
                    HStack(alignment: .top) {
                        Text(label(for: message.source))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .frame(width: 52, alignment: .leading)

                        Text(message.text)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .padding(10)
                            .background(background(for: message.source))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                    }
                }
            }
        }
        .frame(maxHeight: 320)
    }

    private var citationList: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("출처")
                .font(.headline)

            if viewModel.citations.isEmpty {
                Text("출처가 아직 없습니다.")
                    .foregroundStyle(.secondary)
            } else {
                ScrollView {
                    LazyVStack(alignment: .leading, spacing: 8) {
                        ForEach(viewModel.citations) { citation in
                            VStack(alignment: .leading, spacing: 4) {
                                Text(citation.file_path)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                                Text(citation.snippet)
                                    .font(.subheadline)
                                    .lineLimit(3)
                                Text(String(format: "score %.3f", citation.score))
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(10)
                            .frame(maxWidth: .infinity, alignment: .leading)
                            .background(Color.gray.opacity(0.1))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                    }
                }
                .frame(maxHeight: 200)
            }
        }
    }

    private var composer: some View {
        HStack {
            TextField("질문을 입력하세요", text: $viewModel.inputQuery, axis: .vertical)
                .lineLimit(1...5)
                .textFieldStyle(.roundedBorder)

            Button("전송") {
                Task {
                    await viewModel.askLocal()
                }
            }
            .buttonStyle(.borderedProminent)
            .disabled(viewModel.inputQuery.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || viewModel.isBusy)
        }
    }

    private func label(for source: ChatMessage.Source) -> String {
        switch source {
        case .user: return "USER"
        case .local: return "LOCAL"
        case .external: return "EXT"
        }
    }

    private func background(for source: ChatMessage.Source) -> Color {
        switch source {
        case .user:
            return Color.blue.opacity(0.15)
        case .local:
            return Color.green.opacity(0.12)
        case .external:
            return Color.orange.opacity(0.16)
        }
    }
}
