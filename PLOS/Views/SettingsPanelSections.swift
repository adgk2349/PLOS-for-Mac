import Foundation
import SwiftUI

extension SettingsPanelView {
    var privacySection: some View {
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

    var behaviorSection: some View {
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

    var runtimeSection: some View {
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

            VStack(alignment: .leading, spacing: 4) {
                Text("모델 성능 가이드")
                    .font(.caption.weight(.semibold))
                Text("16GB급(3B~8B): 기본 대화/검색 중심, 요약 작업은 API 필요")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text("32GB급(14B~32B): 일반 요약/정리는 로컬 가능, 고난도는 API 권장")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text("64GB+급(70B): 긴 문서 요약/분석도 로컬 가능(속도 비용 큼)")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
                Text("정책 권장: 16GB급 모델에서 '요약/정리' 요청은 API 경로 우선")
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .padding(.horizontal, 10)
            .padding(.vertical, 8)
            .plosGlassInputFrame()
        }
        .padding(12)
        .plosGlassPanel()
    }

    var modelCatalogSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("모델 다운로드 선택지")
                    .font(.headline)
                Spacer()
                Button("새로고침") {
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
                .disabled(viewModel.isCatalogBusy || viewModel.isBusy)
            }

            Text("현재 시스템 메모리: 약 \(systemMemoryGB)GB · 권장 사양 미만 모델은 다운로드가 비활성화됩니다.")
                .font(.caption)
                .foregroundStyle(.secondary)

            if catalogVisibleModels.isEmpty {
                Text("카탈로그 모델을 불러오지 못했습니다. sidecar 상태를 확인한 뒤 새로고침해 주세요.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(catalogTierBuckets, id: \.0) { tier, models in
                    VStack(alignment: .leading, spacing: 8) {
                        Text(tier)
                            .font(.subheadline.weight(.semibold))
                            .foregroundStyle(.secondary)

                        ForEach(models) { model in
                            catalogModelRow(model)
                        }
                    }
                }
            }
        }
        .padding(12)
        .plosGlassPanel()
    }

    var apiKeySection: some View {
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

    var foldersSection: some View {
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

    var memorySection: some View {
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

    var maintenanceSection: some View {
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

    var catalogTierBuckets: [(String, [ModelCatalogItem])] {
        let sorted = catalogVisibleModels.sorted {
            if $0.recommended_memory_gb != $1.recommended_memory_gb {
                return $0.recommended_memory_gb < $1.recommended_memory_gb
            }
            return $0.size_gb < $1.size_gb
        }
        let tiers = ["16GB 이상", "64GB 이상", "256GB 이상", "500GB 이상"]
        return tiers.compactMap { tier in
            let models = sorted.filter { memoryTierTitle(for: $0.recommended_memory_gb) == tier }
            return models.isEmpty ? nil : (tier, models)
        }
    }

    func memoryTierTitle(for memoryGB: Int) -> String {
        if memoryGB <= 16 { return "16GB 이상" }
        if memoryGB <= 64 { return "64GB 이상" }
        if memoryGB <= 256 { return "256GB 이상" }
        return "500GB 이상"
    }

    var systemMemoryGB: Int {
        max(1, Int(ProcessInfo.processInfo.physicalMemory / 1_073_741_824))
    }

    var catalogVisibleModels: [ModelCatalogItem] {
        // Keep catalog concise: hide tiny presets and focus on practical tiers.
        viewModel.catalogModels.filter { $0.recommended_memory_gb >= 16 }
    }

    func canDownloadCatalogModel(_ model: ModelCatalogItem) -> Bool {
        systemMemoryGB >= model.recommended_memory_gb
    }

    @ViewBuilder
    func catalogModelRow(_ model: ModelCatalogItem) -> some View {
        let status = effectiveCatalogStatus(for: model)
        let canDownload = canDownloadCatalogModel(model)
        VStack(alignment: .leading, spacing: 6) {
            HStack(alignment: .top, spacing: 8) {
                VStack(alignment: .leading, spacing: 2) {
                    Text(model.name)
                        .font(.subheadline.weight(.semibold))
                    Text("\(model.profileTitle) · \(model.engine.title) · 약 \(String(format: "%.1f", model.size_gb))GB · 권장 \(model.recommended_memory_gb)GB+")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Text(model.description)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                }
                Spacer(minLength: 8)
                Text(status.title)
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .plosGlassChip()
            }

            if !canDownload {
                Text("이 모델은 최소 \(model.recommended_memory_gb)GB RAM 권장입니다. 현재: \(systemMemoryGB)GB")
                    .font(.caption2)
                    .foregroundStyle(.red.opacity(0.9))
            }

            HStack(spacing: 8) {
                switch status {
                case .notInstalled:
                    Button("다운로드") {
                        Task { await viewModel.installCatalogModel(model.id) }
                    }
                    .disabled(viewModel.isCatalogBusy || viewModel.isBusy || !canDownload)
                case .downloading:
                    HStack(spacing: 8) {
                        Text("다운로드 중")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                        ProgressView()
                            .progressViewStyle(.linear)
                            .frame(width: 130)
                            .controlSize(.small)
                    }
                case .installed:
                    Button(model.active ? "사용 중" : "사용하기") {
                        Task { await viewModel.activateCatalogModel(model.id) }
                    }
                    .disabled(viewModel.isCatalogBusy || viewModel.isBusy || model.active)
                case .active:
                    Button("사용 중") {}
                        .disabled(true)
                case .failed:
                    Button("재시도") {
                        Task { await viewModel.installCatalogModel(model.id) }
                    }
                    .disabled(viewModel.isCatalogBusy || viewModel.isBusy || !canDownload)
                }

                if status != .notInstalled && status != .downloading {
                    Button("삭제") {
                        Task { await viewModel.deleteCatalogModel(model.id) }
                    }
                    .disabled(viewModel.isCatalogBusy || viewModel.isBusy)
                }
            }
            .buttonStyle(.plain)
            .padding(.top, 2)
        }
        .padding(.horizontal, 10)
        .padding(.vertical, 8)
        .plosGlassInputFrame()
    }

    func effectiveCatalogStatus(for model: ModelCatalogItem) -> ModelInstallStatus {
        if viewModel.catalogInstallingModelID == model.id {
            return .downloading
        }
        return model.status
    }
}
