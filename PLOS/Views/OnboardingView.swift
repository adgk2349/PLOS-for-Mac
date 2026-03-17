import AppKit
import SwiftUI

struct OnboardingView: View {
    @ObservedObject var viewModel: AppViewModel

    var body: some View {
        GeometryReader { proxy in
            HStack(spacing: 22) {
                installerSidebar
                    .frame(width: min(290, max(240, proxy.size.width * 0.3)))

                installerContent
            }
            .padding(24)
        }
    }

    private var installerSidebar: some View {
        VStack(alignment: .leading, spacing: 16) {
            HStack(spacing: 8) {
                Image(systemName: "checkmark.shield.fill")
                    .foregroundStyle(Color.green)
                Text("Installer")
                    .font(.headline.weight(.semibold))
            }

            Text("Local AI Core for Mac")
                .font(.title3.weight(.bold))

            ForEach(OnboardingStep.allCases, id: \.rawValue) { step in
                stepRow(step)
            }

            Spacer()

            Text("로컬 우선 · 선택형 외부 호출")
                .font(.footnote)
                .foregroundStyle(.secondary)
        }
        .padding(20)
        .frame(maxHeight: .infinity, alignment: .top)
        .glassCard(cornerRadius: 18)
    }

    private var installerContent: some View {
        VStack(alignment: .leading, spacing: 18) {
            HStack {
                Text("Step \(viewModel.onboardingStep.rawValue + 1) of \(OnboardingStep.allCases.count)")
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)

                Spacer()

                Text(stepBadgeText)
                    .font(.caption2.weight(.semibold))
                    .padding(.horizontal, 10)
                    .padding(.vertical, 4)
                    .background(Color.green.opacity(0.15))
                    .clipShape(Capsule())
            }

            Text(stepTitle)
                .font(.title.weight(.bold))

            Text(stepDescription)
                .foregroundStyle(.secondary)

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
                    readyStep
                }
            }
            .frame(maxHeight: .infinity, alignment: .top)
        }
        .padding(28)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .glassCard(cornerRadius: 20)
    }

    private var stepBadgeText: String {
        switch viewModel.onboardingStep {
        case .welcome:
            return "WELCOME"
        case .dataSelection:
            return "DATA"
        case .startProfile:
            return "PROFILE"
        case .privacyInfo:
            return "PRIVACY"
        case .indexing:
            return "INDEX"
        case .ready:
            return "READY"
        }
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

    private var stepDescription: String {
        switch viewModel.onboardingStep {
        case .welcome:
            return "선택한 자료만 정리하고, 필요한 경우에만 외부 AI를 호출합니다."
        case .dataSelection:
            return "인덱싱할 자료 범위를 명확히 선택해 데이터 통제권을 유지합니다."
        case .startProfile:
            return "속도/품질 성향을 시작 프로필로 지정합니다."
        case .privacyInfo:
            return "외부 호출 정책을 먼저 고정하면 이후 동작이 예측 가능해집니다."
        case .indexing:
            return "로컬 검색과 응답 품질을 위해 데이터 준비를 수행합니다."
        case .ready:
            return "이제 설치가 완료되었습니다. 첫 질문으로 바로 시작할 수 있습니다."
        }
    }

    private func stepRow(_ step: OnboardingStep) -> some View {
        let isCurrent = step == viewModel.onboardingStep
        let isDone = step.rawValue < viewModel.onboardingStep.rawValue

        return HStack(spacing: 10) {
            ZStack {
                Circle()
                    .fill(isDone || isCurrent ? Color.green : Color.gray.opacity(0.25))
                    .frame(width: 18, height: 18)
                if isDone {
                    Image(systemName: "checkmark")
                        .font(.system(size: 10, weight: .bold))
                        .foregroundStyle(.white)
                } else {
                    Circle()
                        .stroke(Color.white.opacity(0.8), lineWidth: 1)
                        .frame(width: 8, height: 8)
                }
            }

            Text(stepName(step))
                .font(.subheadline.weight(isCurrent ? .semibold : .regular))
                .foregroundStyle(isCurrent ? .primary : .secondary)
        }
    }

    private func stepName(_ step: OnboardingStep) -> String {
        switch step {
        case .welcome:
            return "환영"
        case .dataSelection:
            return "자료 선택"
        case .startProfile:
            return "시작 방식"
        case .privacyInfo:
            return "프라이버시"
        case .indexing:
            return "인덱싱"
        case .ready:
            return "완료"
        }
    }

    private var welcomeStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("웹 검색 중심 도구가 아니라, 내 Mac 안의 문서/노트/프로젝트 맥락을 복원하는 작업 코어입니다.")
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(14)
                .background(Color.green.opacity(0.12), in: RoundedRectangle(cornerRadius: 12))

            Button("시작하기") {
                viewModel.onboardingStep = .dataSelection
            }
            .buttonStyle(.glassProminent)
        }
    }

    private var dataSelectionStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("선택한 자료만 로컬 인덱싱됩니다. 나중에 언제든 변경 가능합니다.")
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
            .buttonStyle(.bordered)

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
            .frame(minHeight: 220)

            HStack {
                Button("뒤로") {
                    viewModel.onboardingStep = .welcome
                }
                Spacer()
                Button("다음") {
                    viewModel.onboardingStep = .startProfile
                }
                .buttonStyle(.glassProminent)
                .disabled(viewModel.includedFolderURLs.isEmpty)
            }
        }
    }

    private var startupProfileStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 12) {
                profileCard(.fast, subtitle: "빠른 설치, 경량 응답")
                profileCard(.recommended, subtitle: "속도·품질 균형")
                profileCard(.deep, subtitle: "느리지만 더 깊은 분석")
            }
            .frame(maxWidth: .infinity)

            Text("나중에 설정에서 변경할 수 있습니다.")
                .foregroundStyle(.secondary)

            HStack {
                Button("뒤로") {
                    viewModel.onboardingStep = .dataSelection
                }
                Spacer()
                Button("다음") {
                    viewModel.onboardingStep = .privacyInfo
                }
                .buttonStyle(.glassProminent)
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
                Button("뒤로") {
                    viewModel.onboardingStep = .startProfile
                }
                Spacer()
                Button("계속") {
                    viewModel.onboardingStep = .indexing
                }
                .buttonStyle(.glassProminent)
            }
        }
    }

    private var indexingStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text(viewModel.indexStageText)
                .font(.headline)

            ProgressView(value: viewModel.indexProgress)
                .tint(.green)

            Text("단순 로딩이 아닌 준비 상태를 단계적으로 표시합니다.")
                .foregroundStyle(.secondary)

            HStack {
                Button("뒤로") {
                    viewModel.onboardingStep = .privacyInfo
                }
                Spacer()
                Button("인덱싱 시작") {
                    Task {
                        await viewModel.startOnboardingIndexingFlow()
                    }
                }
                .buttonStyle(.glassProminent)
                .disabled(viewModel.isBusy)
            }
        }
    }

    private var readyStep: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("이제 당신의 자료를 기반으로 질문할 수 있습니다.")
                .font(.headline)

            suggestionButton("이 프로젝트의 핵심 목표가 뭐였지?")
            suggestionButton("이 폴더 문서들 핵심만 요약해줘")
            suggestionButton("지난번 메모 기준으로 다음 할 일 정리해줘")

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
                .buttonStyle(.glassProminent)
            }
        }
    }

    private func suggestionButton(_ text: String) -> some View {
        Button(text) {
            viewModel.inputQuery = text
        }
        .buttonStyle(.bordered)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func profileCard(_ profile: StartupProfile, subtitle: String) -> some View {
        let isSelected = viewModel.startupProfile == profile
        return Button {
            viewModel.startupProfile = profile
        } label: {
            VStack(alignment: .leading, spacing: 6) {
                Text(profile.title)
                    .font(.headline)
                Text(subtitle)
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }
            .padding(12)
            .frame(maxWidth: .infinity, alignment: .leading)
            .background(
                RoundedRectangle(cornerRadius: 12)
                    .fill(isSelected ? Color.green.opacity(0.18) : Color.gray.opacity(0.10))
            )
            .overlay(
                RoundedRectangle(cornerRadius: 12)
                    .stroke(isSelected ? Color.green : Color.clear, lineWidth: 1.5)
            )
        }
        .buttonStyle(.plain)
    }

    private func appendDefaultDirectory(_ name: String) {
        let url = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path) else {
            return
        }
        if !viewModel.includedFolderURLs.contains(where: { $0.path == url.path }) {
            viewModel.includedFolderURLs.append(url)
            viewModel.persistBookmarks()
        }
    }
}

