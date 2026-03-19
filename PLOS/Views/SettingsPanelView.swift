import SwiftUI

struct SettingsPanelView: View {
    @ObservedObject var viewModel: AppViewModel
    var onOpenStatusPanel: (() -> Void)? = nil

    @State private var excludeInput = ""
    @State private var showMemoryViewer = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header

                privacySection
                behaviorSection
                runtimeSection
                apiKeySection
                foldersSection
                memorySection
                maintenanceSection
            }
            .padding(14)
        }
        .sheet(isPresented: $showMemoryViewer) {
            MemoryViewerSheet(viewModel: viewModel)
                .frame(minWidth: 760, minHeight: 560)
                .padding(16)
        }
    }

    private var header: some View {
        HStack {
            Text("설정")
                .font(.title2.weight(.bold))
            Spacer()
            if viewModel.isBusy {
                ProgressView()
            }
            if let onOpenStatusPanel {
                Button("상태 패널") {
                    onOpenStatusPanel()
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 12)
                .padding(.vertical, 8)
                .plosGlassControl()
            }
            Button("저장") {
                Task { await viewModel.saveSettingsAndWorkspace() }
            }
            .buttonStyle(.plain)
            .padding(.horizontal, 12)
            .padding(.vertical, 8)
            .plosGlassControl()
        }
    }

    private var privacySection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("프라이버시 / 응답 경로")
                .font(.headline)

            Picker("프라이버시 모드", selection: $viewModel.privacyMode) {
                ForEach(PrivacyMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: 280)

            Picker("응답 경로", selection: $viewModel.chatResponseRoute) {
                ForEach(ChatResponseRoute.allCases) { route in
                    Text(route.title).tag(route)
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: 280)
            .onChange(of: viewModel.chatResponseRoute) { _, newValue in
                viewModel.setChatResponseRoute(newValue)
            }

            Picker("시스템 액션 승인", selection: $viewModel.actionPermissionMode) {
                ForEach(ActionPermissionMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: 280)
        }
        .padding(12)
        .plosGlassPanel()
    }

    private var behaviorSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("기본 작업 동작")
                .font(.headline)

            Picker("추론 프리셋", selection: $viewModel.quickInferencePreset) {
                ForEach(QuickInferencePreset.allCases) { preset in
                    Text(preset.title).tag(preset)
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: 280)
            .onChange(of: viewModel.quickInferencePreset) { _, newValue in
                viewModel.applyQuickInferencePreset(newValue)
            }

            Picker("기본 모드", selection: $viewModel.defaultWorkMode) {
                ForEach(WorkMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: 280)

            Picker("시작 프로필", selection: $viewModel.startupProfile) {
                ForEach(StartupProfile.allCases) { profile in
                    Text(profile.title).tag(profile)
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: 280)
        }
        .padding(12)
        .plosGlassPanel()
    }

    private var runtimeSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("로컬 모델 / 엔진")
                .font(.headline)

            Picker("엔진", selection: $viewModel.localEngine) {
                ForEach(LocalEngine.allCases) { engine in
                    Text(engine.title).tag(engine)
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: 220)

            HStack(spacing: 8) {
                TextField("MLX 모델 경로", text: $viewModel.mlxModelPath)
                    .textFieldStyle(.plain)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .plosGlassInputFrame()

                Button("선택") {
                    viewModel.chooseModelFile(for: .mlx)
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassControl()
            }

            HStack(spacing: 8) {
                TextField("llama.cpp 모델 경로", text: $viewModel.llamaModelPath)
                    .textFieldStyle(.plain)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .plosGlassInputFrame()

                Button("선택") {
                    viewModel.chooseModelFile(for: .llamaCPP)
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassControl()
            }

            if !viewModel.installedModelsSorted.isEmpty {
                Text("설치된 모델")
                    .font(.subheadline.weight(.semibold))

                ForEach(viewModel.installedModelsSorted.prefix(8)) { model in
                    HStack {
                        VStack(alignment: .leading, spacing: 2) {
                            Text(model.file_name)
                                .lineLimit(1)
                            Text(model.engine.title)
                                .font(.caption)
                                .foregroundStyle(.secondary)
                        }
                        Spacer()
                        Button(viewModel.isInstalledModelActive(model) ? "사용중" : "사용") {
                            Task { await viewModel.selectInstalledModel(model) }
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .plosGlassInputFrame()
                }
            }

            HStack(spacing: 8) {
                Button("런타임 준비") {
                    Task { await viewModel.prepareRuntimeNow() }
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassControl()

                if !viewModel.localRuntimeDetail.isEmpty {
                    Text(viewModel.localRuntimeDetail)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
            }
        }
        .padding(12)
        .plosGlassPanel()
    }

    private var apiKeySection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("외부 API 키")
                .font(.headline)

            SecureField("OpenAI API Key", text: $viewModel.openAIAPIKey)
                .textFieldStyle(.plain)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassInputFrame()

            SecureField("Anthropic API Key", text: $viewModel.anthropicAPIKey)
                .textFieldStyle(.plain)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassInputFrame()

            Text("저장은 macOS Keychain에 보관됩니다.")
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(12)
        .plosGlassPanel()
    }

    private var foldersSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("인덱싱 폴더")
                .font(.headline)

            HStack(spacing: 8) {
                Button("폴더 추가") {
                    viewModel.addFolder()
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassControl()

                Button("전체 재인덱싱") {
                    Task { await viewModel.triggerFullReindex() }
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassControl()
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
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassInputFrame()
            }

            HStack(spacing: 8) {
                TextField("제외 폴더 경로", text: $excludeInput)
                    .textFieldStyle(.plain)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .plosGlassInputFrame()

                Button("추가") {
                    let trimmed = excludeInput.trimmingCharacters(in: .whitespacesAndNewlines)
                    guard !trimmed.isEmpty else { return }
                    if !viewModel.excludedPaths.contains(trimmed) {
                        viewModel.excludedPaths.append(trimmed)
                    }
                    excludeInput = ""
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassControl()
            }

            if !viewModel.excludedPaths.isEmpty {
                ForEach(viewModel.excludedPaths, id: \.self) { path in
                    HStack {
                        Text(path)
                            .lineLimit(1)
                        Spacer()
                        Button("제거") {
                            viewModel.excludedPaths.removeAll { $0 == path }
                        }
                        .buttonStyle(.plain)
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .plosGlassInputFrame()
                }
            }
        }
        .padding(12)
        .plosGlassPanel()
    }

    private var memorySection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("로컬 메모리")
                .font(.headline)

            Toggle("Adaptive personalization", isOn: $viewModel.adaptivePersonalizationEnabled)
            Toggle("Session memory", isOn: $viewModel.sessionMemoryEnabled)
            Toggle("Workspace memory", isOn: $viewModel.workspaceMemoryEnabled)
            Toggle("Local memory only", isOn: $viewModel.localMemoryOnly)

            Picker("Workspace memory mode", selection: $viewModel.workspaceMemoryMode) {
                ForEach(WorkspaceMemoryMode.allCases) { mode in
                    Text(mode.title).tag(mode)
                }
            }
            .pickerStyle(.menu)
            .frame(maxWidth: 260)

            HStack(spacing: 8) {
                Button("메모리 보기") {
                    showMemoryViewer = true
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassControl()

                Button("세션 초기화") {
                    Task { await viewModel.clearMemory(scope: .session) }
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassControl()

                Button("전체 초기화") {
                    Task { await viewModel.clearMemory(scope: .all) }
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassControl()
            }
        }
        .padding(12)
        .plosGlassPanel()
    }

    private var maintenanceSection: some View {
        VStack(alignment: .leading, spacing: 8) {
            Text("유지보수")
                .font(.headline)

            HStack(spacing: 8) {
                Button("원격 상태 새로고침") {
                    Task {
                        do {
                            try await viewModel.refreshRemoteState()
                        } catch {
                            viewModel.lastError = error.localizedDescription
                        }
                    }
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassControl()

                Button("설정 저장") {
                    Task { await viewModel.saveSettingsAndWorkspace() }
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassControl()
            }
        }
        .padding(12)
        .plosGlassPanel()
    }
}

private struct MemoryViewerSheet: View {
    @ObservedObject var viewModel: AppViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text("메모리 보기")
                    .font(.title2.weight(.bold))

                memoryBlock("Session", items: viewModel.sessionMemoryItems.map { "\($0.key): \($0.value_json)" })
                memoryBlock("Workspace", items: viewModel.workspaceMemoryItems.map { "\($0.memory_type) / \($0.key): \($0.value_json)" })
                memoryBlock("Preferences", items: viewModel.preferenceMemoryItems.map { "\($0.key): \($0.value_json)" })
                memoryBlock("Episodic", items: viewModel.episodicMemoryItems.map { "\($0.event_type): \($0.summary)" })

                VStack(alignment: .leading, spacing: 8) {
                    Text("Pinned")
                        .font(.headline)
                    if viewModel.pinnedMemoryItems.isEmpty {
                        Text("고정 메모리가 없습니다.")
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(viewModel.pinnedMemoryItems) { item in
                            HStack(alignment: .top) {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(item.title)
                                        .font(.subheadline.weight(.semibold))
                                    Text(item.content)
                                        .font(.subheadline)
                                        .foregroundStyle(.secondary)
                                }
                                Spacer()
                                Button("해제") {
                                    Task { await viewModel.unpinMemory(memoryID: item.id) }
                                }
                                .buttonStyle(.plain)
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 8)
                            .plosGlassInputFrame()
                        }
                    }
                }
                .padding(12)
                .plosGlassPanel()
            }
            .padding(12)
        }
        .task {
            do {
                try await viewModel.refreshMemoryState()
            } catch {
                if !(error is CancellationError) {
                    viewModel.lastError = error.localizedDescription
                }
            }
        }
    }

    private func memoryBlock(_ title: String, items: [String]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            Text(title)
                .font(.headline)
            if items.isEmpty {
                Text("비어 있음")
                    .foregroundStyle(.secondary)
            } else {
                ForEach(Array(items.prefix(20).enumerated()), id: \.offset) { _, line in
                    Text(line)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .plosGlassInputFrame()
                }
            }
        }
        .padding(12)
        .plosGlassPanel()
    }
}
