import SwiftUI

struct OnboardingView: View {
    @ObservedObject var viewModel: AppViewModel

    var body: some View {
        VStack(alignment: .leading, spacing: 20) {
            Text(stepTitle)
                .font(.largeTitle.weight(.bold))

            Group {
                switch viewModel.onboardingStep {
                case .welcome:
                    welcomeStep
                case .dataSelection:
                    dataSelectionStep
                case .startProfile:
                    startupProfileStep
                case .privacyInfo:
                    privacyStep
                case .indexing:
                    indexingStep
                case .ready:
                    firstQuestionStep
                }
            }
            .frame(maxHeight: .infinity, alignment: .top)
        }
        .padding(28)
    }

    private var stepTitle: String {
        switch viewModel.onboardingStep {
        case .welcome:
            return "당신의 Mac에서 시작되는 개인 AI"
        case .dataSelection:
            return "어떤 자료를 참고할까요?"
        case .startProfile:
            return "어떤 방식으로 시작할까요?"
        case .privacyInfo:
            return "기본적으로 로컬에서 처리합니다"
        case .indexing:
            return "로컬 인덱싱 준비"
        case .ready:
            return "준비가 끝났습니다"
        }
    }

    private var welcomeStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("선택한 자료만 로컬에서 정리하고, 필요할 때만 외부 AI를 사용합니다.")
            Button("시작하기") {
                viewModel.onboardingStep = .dataSelection
            }
            .buttonStyle(.borderedProminent)
            .padding(.top, 8)
        }
    }

    private var dataSelectionStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("선택한 자료만 로컬 인덱싱됩니다. 나중에 언제든 변경할 수 있습니다.")
                .foregroundStyle(.secondary)

            HStack {
                Button("Documents") {
                    appendDefaultDirectory("Documents")
                }
                Button("Desktop") {
                    appendDefaultDirectory("Desktop")
                }
                Button("Downloads") {
                    appendDefaultDirectory("Downloads")
                }
                Button("특정 폴더 추가") {
                    viewModel.addFolder()
                }
            }

            List {
                ForEach(viewModel.includedFolderURLs, id: \.path) { url in
                    HStack {
                        Text(url.path)
                            .lineLimit(1)
                        Spacer()
                        Button("제거") {
                            viewModel.removeFolder(url.path)
                        }
                    }
                }
            }
            .frame(minHeight: 240)

            HStack {
                Button("뒤로") { viewModel.goToPreviousOnboardingStep() }
                Spacer()
                Button("다음") {
                    viewModel.onboardingStep = .startProfile
                }
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.includedFolderURLs.isEmpty)
            }
        }
    }

    private var startupProfileStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            Picker("시작 방식", selection: $viewModel.startupProfile) {
                ForEach(StartupProfile.allCases) { profile in
                    Text(profile.title).tag(profile)
                }
            }
            .pickerStyle(.radioGroup)

            Text("나중에 설정에서 변경할 수 있습니다.")
                .foregroundStyle(.secondary)

            HStack {
                Button("뒤로") { viewModel.goToPreviousOnboardingStep() }
                Spacer()
                Button("다음") { viewModel.onboardingStep = .privacyInfo }
                    .buttonStyle(.borderedProminent)
            }
        }
    }

    private var privacyStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("외부 AI 사용 여부는 설정에서 언제든 바꿀 수 있습니다.")

            Picker("프라이버시 모드", selection: $viewModel.privacyMode) {
                ForEach(PrivacyMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            HStack {
                Button("뒤로") { viewModel.goToPreviousOnboardingStep() }
                Spacer()
                Button("계속") { viewModel.onboardingStep = .indexing }
                    .buttonStyle(.borderedProminent)
            }
        }
    }

    private var indexingStep: some View {
        VStack(alignment: .leading, spacing: 14) {
            Text(viewModel.indexStageText)
            ProgressView(value: viewModel.indexProgress)

            Text("단순 로딩이 아닌 작업 준비 상태를 단계적으로 표시합니다.")
                .foregroundStyle(.secondary)

            HStack {
                Button("뒤로") {
                    viewModel.goToPreviousOnboardingStep()
                }
                Spacer()
                Button("인덱싱 시작") {
                    Task {
                        await viewModel.startOnboardingIndexingFlow()
                    }
                }
                .buttonStyle(.borderedProminent)
                .disabled(viewModel.isBusy)
            }
        }
    }

    private var firstQuestionStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("이제 당신의 자료를 기반으로 질문할 수 있습니다.")

            Group {
                suggestionButton("이 프로젝트의 핵심 목표가 뭐였지?")
                suggestionButton("이 폴더 문서들 핵심만 요약해줘")
                suggestionButton("지난번 메모 기준으로 다음 할 일 정리해줘")
            }

            Picker("작업 모드", selection: $viewModel.selectedMode) {
                ForEach(WorkMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.segmented)

            HStack {
                Spacer()
                Button("작업 시작") {
                    viewModel.finalizeOnboarding()
                }
                .buttonStyle(.borderedProminent)
            }
        }
    }

    private func suggestionButton(_ text: String) -> some View {
        Button(text) {
            viewModel.inputQuery = text
        }
        .buttonStyle(.bordered)
    }

    private func appendDefaultDirectory(_ name: String) {
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else { return }
        if !viewModel.includedFolderURLs.contains(where: { $0.path == url.path }) {
            viewModel.includedFolderURLs.append(url)
        }
    }
}
