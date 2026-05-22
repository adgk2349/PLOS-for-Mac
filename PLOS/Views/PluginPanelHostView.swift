import SwiftUI

struct PluginPanelHostView: View {
    @ObservedObject var viewModel: AppViewModel

    private var language: AppLanguage { viewModel.appLanguage }

    private func t(_ ko: String, _ en: String, _ ja: String) -> String {
        L10n.text(ko, en, ja, language: language)
    }

    var body: some View {
        VStack(spacing: 0) {
            header
            Divider().opacity(0.25)
            content
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
        .background(Color.clear)
        .task(id: viewModel.selectedPluginPanelCompositeID) {
            guard let selection = viewModel.selectedPluginPanelSelection else { return }
            let activePluginID = viewModel.activePluginPanel?.plugin_id.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let activePanelID = viewModel.activePluginPanel?.panel_id.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            let targetPanelID = selection.panelID?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if activePluginID != selection.pluginID || activePanelID != targetPanelID {
                await viewModel.openPluginPanel(pluginID: selection.pluginID, panelID: selection.panelID)
            }
        }
    }

    private var header: some View {
        HStack(spacing: 10) {
            Button {
                viewModel.switchToChatPanel()
            } label: {
                Label(t("채팅으로", "Back to chats", "チャットへ"), systemImage: "bubble.left.and.bubble.right")
                    .font(.caption.weight(.semibold))
            }
            .buttonStyle(.plain)

            Spacer(minLength: 0)

            if let panel = viewModel.activePluginPanel {
                Text(panel.title)
                    .font(.headline.weight(.semibold))
                    .lineLimit(1)
            } else {
                Text(t("플러그인 패널", "Plugin panel", "プラグインパネル"))
                    .font(.headline.weight(.semibold))
            }

            Spacer(minLength: 0)

            if viewModel.pluginPanelIsBusy {
                ProgressView()
                    .controlSize(.small)
            }
        }
        .padding(.horizontal, 16)
        .padding(.vertical, 12)
    }

    @ViewBuilder
    private var content: some View {
        if let panel = viewModel.activePluginPanel {
            switch panel.view_type {
            case .imageStudio:
                imageStudio
            case .customForm:
                unsupportedView
            }
        } else {
            loadingView
        }
    }

    private var loadingView: some View {
        VStack(spacing: 12) {
            ProgressView()
            Text(t("플러그인 패널을 여는 중...", "Opening plugin panel...", "プラグインパネルを開いています..."))
                .foregroundStyle(.secondary)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var unsupportedView: some View {
        VStack(spacing: 10) {
            Text(t("아직 지원되지 않는 패널 타입입니다.", "This panel type is not supported yet.", "未対応のパネルタイプです。"))
                .foregroundStyle(.secondary)
            Button(t("채팅으로 돌아가기", "Return to chats", "チャットへ戻る")) {
                viewModel.switchToChatPanel()
            }
            .buttonStyle(.plain)
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity)
    }

    private var imageStudio: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 12) {
                Text(t("모델 관리", "Model management", "モデル管理"))
                    .font(.subheadline.weight(.semibold))

                TextField(t("모델 ID", "Model ID", "モデルID"), text: $viewModel.pluginPanelModelID)
                    .textFieldStyle(.roundedBorder)

                TextField(t("Hugging Face repo_id", "Hugging Face repo_id", "Hugging Face repo_id"), text: $viewModel.pluginPanelRepoID)
                    .textFieldStyle(.roundedBorder)

                TextField(t("파일명(선택)", "Filename (optional)", "ファイル名（任意）"), text: $viewModel.pluginPanelFilename)
                    .textFieldStyle(.roundedBorder)

                HStack(spacing: 10) {
                    Button {
                        Task { await viewModel.downloadPluginPanelModel() }
                    } label: {
                        Label(t("모델 다운로드", "Download model", "モデルをダウンロード"), systemImage: "arrow.down.circle")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(viewModel.pluginPanelIsBusy || viewModel.pluginPanelModelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                    Button {
                        Task { await viewModel.setActivePluginPanelModel() }
                    } label: {
                        Label(t("활성 모델로 설정", "Set active model", "アクティブモデルに設定"), systemImage: "checkmark.circle")
                    }
                    .buttonStyle(.bordered)
                    .disabled(viewModel.pluginPanelIsBusy || viewModel.pluginPanelModelID.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)

                    Button {
                        Task { await viewModel.refreshPluginPanelModels() }
                    } label: {
                        Label(t("목록 새로고침", "Refresh list", "一覧更新"), systemImage: "arrow.clockwise")
                    }
                    .buttonStyle(.bordered)
                    .disabled(viewModel.pluginPanelIsBusy)
                }

                if !viewModel.pluginPanelActiveModelID.isEmpty {
                    Text(t("활성 모델", "Active model", "アクティブモデル") + ": " + viewModel.pluginPanelActiveModelID)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                if !viewModel.pluginPanelInstalledModels.isEmpty {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(t("설치된 모델", "Installed models", "インストール済みモデル"))
                            .font(.caption.weight(.semibold))
                        ForEach(viewModel.pluginPanelInstalledModels, id: \.self) { model in
                            Text(model)
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(1)
                        }
                    }
                }

                if !viewModel.pluginPanelDownloadMessage.isEmpty {
                    Text(viewModel.pluginPanelDownloadMessage)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }

                Divider().padding(.vertical, 2)

                Text(t("이미지 생성", "Image generation", "画像生成"))
                    .font(.subheadline.weight(.semibold))

                TextField(t("프롬프트", "Prompt", "プロンプト"), text: $viewModel.pluginPanelPrompt)
                    .textFieldStyle(.roundedBorder)

                TextField(t("네거티브 프롬프트", "Negative prompt", "ネガティブプロンプト"), text: $viewModel.pluginPanelNegativePrompt)
                    .textFieldStyle(.roundedBorder)

                HStack(spacing: 10) {
                    Stepper("\(t("너비", "Width", "幅")): \(viewModel.pluginPanelWidth)", value: $viewModel.pluginPanelWidth, in: 256...2048, step: 64)
                    Stepper("\(t("높이", "Height", "高さ")): \(viewModel.pluginPanelHeight)", value: $viewModel.pluginPanelHeight, in: 256...2048, step: 64)
                }

                HStack(spacing: 10) {
                    Stepper("\(t("스텝", "Steps", "ステップ")): \(viewModel.pluginPanelSteps)", value: $viewModel.pluginPanelSteps, in: 1...60)
                    Stepper("\(t("배치", "Batch", "バッチ")): \(viewModel.pluginPanelBatch)", value: $viewModel.pluginPanelBatch, in: 1...4)
                }

                TextField(t("시드(선택)", "Seed (optional)", "シード（任意）"), text: $viewModel.pluginPanelSeedText)
                    .textFieldStyle(.roundedBorder)

                HStack {
                    Button {
                        Task { await viewModel.submitActivePluginPanelAction() }
                    } label: {
                        Label(t("생성", "Generate", "生成"), systemImage: "sparkles")
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(viewModel.pluginPanelPrompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty || viewModel.pluginPanelIsBusy)

                    if let jobID = viewModel.pluginPanelLastJobID, !jobID.isEmpty {
                        Text("job: \(jobID)")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }

                if !viewModel.pluginPanelImages.isEmpty {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(t("결과 이미지", "Generated images", "生成結果"))
                            .font(.subheadline.weight(.semibold))
                        ForEach(viewModel.pluginPanelImages, id: \.self) { uri in
                            if let filePath = filePath(from: uri) {
                                VStack(alignment: .leading, spacing: 6) {
                                    Text(filePath)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                    Image(nsImage: NSImage(contentsOfFile: filePath) ?? NSImage())
                                        .resizable()
                                        .scaledToFit()
                                        .frame(maxHeight: 280)
                                        .background(Color.white.opacity(0.02))
                                        .clipShape(RoundedRectangle(cornerRadius: 10))
                                }
                            } else {
                                Text(uri)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                        }
                    }
                    .padding(.top, 8)
                }
            }
            .padding(16)
            .frame(maxWidth: 860, alignment: .leading)
        }
    }

    private func filePath(from uri: String) -> String? {
        guard let url = URL(string: uri), url.isFileURL else { return nil }
        return url.path
    }
}
