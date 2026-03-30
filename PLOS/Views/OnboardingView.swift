import Foundation
import SwiftUI

struct OnboardingView: View {
    @ObservedObject var viewModel: AppViewModel
    private var language: AppLanguage { viewModel.appLanguage }

    private func t(_ ko: String, _ en: String, _ ja: String) -> String {
        L10n.text(ko, en, ja, language: language)
    }

    var body: some View {
        GeometryReader { proxy in
            HStack(spacing: 16) {
                stepSidebar
                    .frame(width: min(280, max(230, proxy.size.width * 0.28)))

                stepContent
            }
            .padding(16)
        }
    }

    private var stepSidebar: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(t("PLOS 설정", "PLOS Setup", "PLOS セットアップ"))
                .font(.title3.weight(.bold))

            ForEach(OnboardingStep.allCases, id: \.rawValue) { step in
                HStack(spacing: 8) {
                    Image(systemName: viewModel.onboardingStep.rawValue >= step.rawValue ? "checkmark.circle.fill" : "circle")
                        .foregroundStyle(viewModel.onboardingStep.rawValue >= step.rawValue ? .green : .secondary)
                    Text(stepTitle(step))
                        .font(.subheadline.weight(viewModel.onboardingStep == step ? .semibold : .regular))
                    Spacer()
                }
                .padding(.horizontal, 8)
                .padding(.vertical, 8)
                .background(
                    viewModel.onboardingStep == step ? Color.white.opacity(0.12) : Color.clear,
                    in: Rectangle()
                )
            }

            Spacer()
            Text(t("로컬 우선 · 필요할 때만 외부 호출", "Local first · external only when needed", "ローカル優先 · 必要時のみ外部呼び出し"))
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .padding(14)
        .plosGlassPanel()
    }

    @ViewBuilder
    private var stepContent: some View {
        VStack(alignment: .leading, spacing: 16) {
            Text(stepTitle(viewModel.onboardingStep))
                .font(.title.weight(.bold))
            Text(stepDescription(viewModel.onboardingStep))
                .foregroundStyle(.secondary)

            Group {
                switch viewModel.onboardingStep {
                case .welcome:
                    welcomeView
                case .dataSelection:
                    dataSelectionView
                case .startProfile:
                    startProfileView
                case .privacyInfo:
                    privacyView
                case .indexing:
                    indexingView
                case .ready:
                    readyView
                }
            }

            Spacer(minLength: 0)
        }
        .padding(20)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .plosGlassPanel()
    }

    private var welcomeView: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(t("선택한 자료만 로컬에서 정리하고, 필요할 때만 외부 AI를 사용합니다.", "Use selected data locally, and use external AI only when necessary.", "選択した資料のみローカルで処理し、必要時のみ外部AIを使います。"))
            Button(t("시작하기", "Get started", "開始する")) {
                viewModel.onboardingStep = .dataSelection
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .plosGlassControl()
        }
    }

    private var dataSelectionView: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(spacing: 8) {
                presetFolderButton(t("문서", "Documents", "書類"), path: homePath("Documents"))
                presetFolderButton(t("데스크탑", "Desktop", "デスクトップ"), path: homePath("Desktop"))
                presetFolderButton(t("다운로드", "Downloads", "ダウンロード"), path: homePath("Downloads"))
            }

            HStack(spacing: 8) {
                Button(t("특정 폴더 추가", "Add custom folder", "特定フォルダを追加")) {
                    viewModel.addFolder()
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 12)
                .padding(.vertical, 9)
                .plosGlassControl()

                Button(t("다음", "Next", "次へ")) {
                    guard !viewModel.includedFolderURLs.isEmpty else {
                        viewModel.lastError = t("최소 1개 이상의 폴더를 선택해 주세요.", "Please select at least one folder.", "少なくとも1つ以上のフォルダを選択してください。")
                        return
                    }
                    viewModel.onboardingStep = .startProfile
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 12)
                .padding(.vertical, 9)
                .plosGlassControl()
            }

            if viewModel.includedFolderURLs.isEmpty {
                Text(t("아직 선택된 폴더가 없습니다.", "No folders selected yet.", "まだ選択されたフォルダはありません。"))
                    .foregroundStyle(.secondary)
            } else {
                ForEach(viewModel.includedFolderURLs, id: \.path) { url in
                    HStack {
                        Text(url.path)
                            .lineLimit(1)
                        Spacer()
                        Button(t("제거", "Remove", "削除")) {
                            viewModel.removeFolder(url.path)
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .plosGlassInputFrame()
                }
            }
        }
    }

    private var startProfileView: some View {
        VStack(alignment: .leading, spacing: 12) {
            Picker(t("시작 방식", "Startup profile", "開始プロファイル"), selection: $viewModel.startupProfile) {
                ForEach(StartupProfile.allCases) { profile in
                    Text(profile.title(language: language)).tag(profile)
                }
            }
            .pickerStyle(.menu)
            .id("onboarding-startup-profile-\(language.rawValue)")
            .frame(maxWidth: 280)

            Text(t("나중에 설정에서 변경할 수 있습니다.", "You can change this later in settings.", "後で設定から変更できます。"))
                .foregroundStyle(.secondary)

            Button(t("다음", "Next", "次へ")) {
                viewModel.onboardingStep = .privacyInfo
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .plosGlassControl()
        }
    }

    private var privacyView: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(t("기본적으로 로컬에서 처리하며 외부 AI 사용 여부는 설정에서 바꿀 수 있습니다.", "By default everything runs locally; external AI usage can be changed in settings.", "既定ではローカル処理で、外部AI利用は設定で変更できます。"))
            Button(t("인덱싱 시작", "Start indexing", "インデックス開始")) {
                Task {
                    await viewModel.startOnboardingIndexingFlow()
                }
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .plosGlassControl()
            .disabled(viewModel.isBusy)
        }
    }

    private var indexingView: some View {
        VStack(alignment: .leading, spacing: 12) {
            ProgressView(value: viewModel.indexProgress)
                .progressViewStyle(.linear)
            Text(viewModel.indexStageText)
                .font(.subheadline)
                .foregroundStyle(.secondary)
            if viewModel.isBusy {
                Text(t("문서 분석 중입니다…", "Analyzing documents…", "文書を分析中…"))
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var readyView: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(t("준비가 끝났습니다. 이제 자료 기반 질의응답을 시작할 수 있습니다.", "Setup is complete. You can start grounded Q&A now.", "準備が完了しました。資料ベースのQ&Aを開始できます。"))
            Button(t("시작", "Start", "開始")) {
                viewModel.finalizeOnboarding()
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .plosGlassControl()

            VStack(alignment: .leading, spacing: 6) {
                Text(t("추천 질문", "Suggested prompts", "おすすめ質問"))
                    .font(.headline)
                Text(t("• 이 프로젝트의 핵심 목표가 뭐였지?", "• What is the core goal of this project?", "• このプロジェクトの核心目標は何？"))
                Text(t("• 이 폴더 문서들 핵심만 요약해줘", "• Summarize key points from this folder", "• このフォルダ文書の要点を要約して"))
                Text(t("• 지난번 메모 기준으로 다음 할 일 정리해줘", "• Organize next actions from previous notes", "• 前回メモを基に次の作業を整理して"))
            }
            .padding(.top, 4)
        }
    }

    private func stepTitle(_ step: OnboardingStep) -> String {
        switch step {
        case .welcome:
            return t("당신의 Mac에서 시작되는 개인 AI", "Personal AI that starts on your Mac", "あなたのMacで始まる個人AI")
        case .dataSelection:
            return t("어떤 자료를 참고할까요?", "Which data should be used?", "どの資料を使いますか？")
        case .startProfile:
            return t("어떤 방식으로 시작할까요?", "How would you like to start?", "どの方式で始めますか？")
        case .privacyInfo:
            return t("기본적으로 로컬에서 처리합니다", "Local processing by default", "既定ではローカル処理")
        case .indexing:
            return t("로컬 인덱싱 진행", "Local indexing in progress", "ローカルインデックス進行中")
        case .ready:
            return t("준비가 끝났습니다", "Ready to go", "準備完了")
        }
    }

    private func stepDescription(_ step: OnboardingStep) -> String {
        switch step {
        case .welcome:
            return t("선택한 자료만 로컬에서 정리하고, 필요할 때만 외부 AI를 사용합니다.", "Only selected data is processed locally, with optional external AI.", "選択した資料だけをローカル処理し、必要時のみ外部AIを使います。")
        case .dataSelection:
            return t("선택한 자료만 인덱싱됩니다. 나중에 언제든 변경할 수 있습니다.", "Only selected data will be indexed. You can change it anytime later.", "選択した資料のみインデックスされます。後でいつでも変更できます。")
        case .startProfile:
            return t("빠른 시작 / 추천 설정 / 깊은 분석 중에서 선택하세요.", "Choose fast / recommended / deep analysis.", "高速 / おすすめ / 深い分析 から選択してください。")
        case .privacyInfo:
            return t("외부 AI 사용 정책은 설정에서 언제든 조정할 수 있습니다.", "External AI policy can be adjusted in settings at any time.", "外部AI利用ポリシーは設定でいつでも調整できます。")
        case .indexing:
            return t("문서를 스캔하고 검색/응답 준비를 진행합니다.", "Scanning documents and preparing retrieval/response.", "文書をスキャンし、検索/応答準備を進めます。")
        case .ready:
            return t("첫 질문을 입력해 작업을 시작하세요.", "Enter your first prompt to begin.", "最初の質問を入力して開始してください。")
        }
    }

    private func homePath(_ component: String) -> String {
        URL(fileURLWithPath: NSHomeDirectory()).appendingPathComponent(component).path
    }

    private func presetFolderButton(_ title: String, path: String) -> some View {
        let isSelected = viewModel.includedFolderURLs.contains(where: { $0.path == path })
        return Button {
            if isSelected {
                viewModel.removeFolder(path)
            } else {
                viewModel.includedFolderURLs.append(URL(fileURLWithPath: path))
                viewModel.persistBookmarks()
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: isSelected ? "checkmark.square.fill" : "square")
                Text(title)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .plosGlassControl()
        }
        .buttonStyle(.plain)
    }
}
