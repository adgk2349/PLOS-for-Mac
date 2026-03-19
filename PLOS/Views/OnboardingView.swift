import Foundation
import SwiftUI

struct OnboardingView: View {
    @ObservedObject var viewModel: AppViewModel

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
            Text("PLOS Setup")
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
            Text("로컬 우선 · 필요할 때만 외부 호출")
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
            Text("선택한 자료만 로컬에서 정리하고, 필요할 때만 외부 AI를 사용합니다.")
            Button("시작하기") {
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
                presetFolderButton("Documents", path: homePath("Documents"))
                presetFolderButton("Desktop", path: homePath("Desktop"))
                presetFolderButton("Downloads", path: homePath("Downloads"))
            }

            HStack(spacing: 8) {
                Button("특정 폴더 추가") {
                    viewModel.addFolder()
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 12)
                .padding(.vertical, 9)
                .plosGlassControl()

                Button("다음") {
                    guard !viewModel.includedFolderURLs.isEmpty else {
                        viewModel.lastError = "최소 1개 이상의 폴더를 선택해 주세요."
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
                Text("아직 선택된 폴더가 없습니다.")
                    .foregroundStyle(.secondary)
            } else {
                ForEach(viewModel.includedFolderURLs, id: \.path) { url in
                    HStack {
                        Text(url.path)
                            .lineLimit(1)
                        Spacer()
                        Button("제거") {
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
            Picker("시작 방식", selection: $viewModel.startupProfile) {
                ForEach(StartupProfile.allCases) { profile in
                    Text(profile.title).tag(profile)
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: 280)

            Text("나중에 설정에서 변경할 수 있습니다.")
                .foregroundStyle(.secondary)

            Button("다음") {
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
            Text("기본적으로 로컬에서 처리하며 외부 AI 사용 여부는 설정에서 바꿀 수 있습니다.")
            Button("인덱싱 시작") {
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
                Text("문서 분석 중입니다…")
                    .foregroundStyle(.secondary)
            }
        }
    }

    private var readyView: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("준비가 끝났습니다. 이제 자료 기반 질의응답을 시작할 수 있습니다.")
            Button("시작") {
                viewModel.finalizeOnboarding()
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 12)
            .padding(.vertical, 9)
            .plosGlassControl()

            VStack(alignment: .leading, spacing: 6) {
                Text("추천 질문")
                    .font(.headline)
                Text("• 이 프로젝트의 핵심 목표가 뭐였지?")
                Text("• 이 폴더 문서들 핵심만 요약해줘")
                Text("• 지난번 메모 기준으로 다음 할 일 정리해줘")
            }
            .padding(.top, 4)
        }
    }

    private func stepTitle(_ step: OnboardingStep) -> String {
        switch step {
        case .welcome:
            return "당신의 Mac에서 시작되는 개인 AI"
        case .dataSelection:
            return "어떤 자료를 참고할까요?"
        case .startProfile:
            return "어떤 방식으로 시작할까요?"
        case .privacyInfo:
            return "기본적으로 로컬에서 처리합니다"
        case .indexing:
            return "로컬 인덱싱 진행"
        case .ready:
            return "준비가 끝났습니다"
        }
    }

    private func stepDescription(_ step: OnboardingStep) -> String {
        switch step {
        case .welcome:
            return "선택한 자료만 로컬에서 정리하고, 필요할 때만 외부 AI를 사용합니다."
        case .dataSelection:
            return "선택한 자료만 인덱싱됩니다. 나중에 언제든 변경할 수 있습니다."
        case .startProfile:
            return "빠른 시작 / 추천 설정 / 깊은 분석 중에서 선택하세요."
        case .privacyInfo:
            return "외부 AI 사용 정책은 설정에서 언제든 조정할 수 있습니다."
        case .indexing:
            return "문서를 스캔하고 검색/응답 준비를 진행합니다."
        case .ready:
            return "첫 질문을 입력해 작업을 시작하세요."
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
