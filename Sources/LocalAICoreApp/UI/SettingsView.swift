import SwiftUI

struct SettingsView: View {
    @ObservedObject var viewModel: AppViewModel
    @State private var newExcludePath: String = ""

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("설정")
                    .font(.title2.weight(.bold))

                VStack(alignment: .leading, spacing: 8) {
                    Text("프라이버시")
                        .font(.headline)
                    Picker("프라이버시 모드", selection: $viewModel.privacyMode) {
                        ForEach(PrivacyMode.allCases) { mode in
                            Text(mode.title).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("시작 방식")
                        .font(.headline)
                    Picker("시작 방식", selection: $viewModel.startupProfile) {
                        ForEach(StartupProfile.allCases) { profile in
                            Text(profile.title).tag(profile)
                        }
                    }
                    .pickerStyle(.radioGroup)
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("기본 작업 모드")
                        .font(.headline)
                    Picker("기본 작업 모드", selection: $viewModel.defaultWorkMode) {
                        ForEach(WorkMode.allCases) { mode in
                            Text(mode.title).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("인덱싱 대상 폴더")
                        .font(.headline)

                    HStack {
                        Button("폴더 추가") {
                            viewModel.addFolder()
                        }
                        Button("전체 재인덱싱") {
                            Task {
                                await viewModel.triggerFullReindex()
                            }
                        }
                        .disabled(viewModel.isBusy)
                    }

                    ForEach(viewModel.includedFolderURLs, id: \.path) { url in
                        HStack {
                            Text(url.path)
                                .lineLimit(1)
                            Spacer()
                            Button("삭제") {
                                viewModel.removeFolder(url.path)
                            }
                        }
                    }
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text("제외 폴더")
                        .font(.headline)

                    HStack {
                        TextField("제외 경로", text: $newExcludePath)
                        Button("추가") {
                            let trimmed = newExcludePath.trimmingCharacters(in: .whitespacesAndNewlines)
                            guard !trimmed.isEmpty else { return }
                            if !viewModel.excludedPaths.contains(trimmed) {
                                viewModel.excludedPaths.append(trimmed)
                            }
                            newExcludePath = ""
                        }
                    }

                    ForEach(viewModel.excludedPaths, id: \.self) { path in
                        HStack {
                            Text(path)
                                .lineLimit(1)
                            Spacer()
                            Button("제거") {
                                viewModel.excludedPaths.removeAll { $0 == path }
                            }
                        }
                    }
                }

                HStack {
                    Spacer()
                    Button("설정 저장") {
                        Task {
                            await viewModel.updateSettings()
                        }
                    }
                    .buttonStyle(.borderedProminent)
                    .disabled(viewModel.isBusy)
                }
            }
            .padding(16)
        }
    }
}
