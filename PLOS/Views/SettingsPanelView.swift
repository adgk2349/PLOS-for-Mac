import SwiftUI

struct SettingsPanelView: View {
    @ObservedObject var viewModel: AppViewModel
    @State private var newExcludePath = ""
    @State private var editingDocument: DocumentMetadata?
    @State private var editCategory = ""
    @State private var editSubcategory = ""
    @State private var editDocumentType = ""
    @State private var editTags = ""
    @State private var editYear = ""
    @State private var editProject = ""
    @State private var editImportance = 0.5
    @State private var editExcluded = false

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
                .padding(12)
                .glassCard(cornerRadius: 12)

                VStack(alignment: .leading, spacing: 8) {
                    Text("시스템 액션 권한")
                        .font(.headline)
                    Picker("권한 정책", selection: $viewModel.actionPermissionMode) {
                        ForEach(ActionPermissionMode.allCases) { mode in
                            Text(mode.title).tag(mode)
                        }
                    }
                    .pickerStyle(.segmented)
                    Text("파일 열기 같은 시스템 액션 실행 시 확인 정책입니다.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                }
                .padding(12)
                .glassCard(cornerRadius: 12)

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
                .padding(12)
                .glassCard(cornerRadius: 12)

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
                .padding(12)
                .glassCard(cornerRadius: 12)

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
                .padding(12)
                .glassCard(cornerRadius: 12)

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
                .padding(12)
                .glassCard(cornerRadius: 12)

                modelRuntimeSection

                documentMetadataSection

                HStack {
                    Spacer()
                    Button("설정 저장") {
                        Task {
                            await viewModel.saveSettingsAndWorkspace()
                        }
                    }
                    .buttonStyle(.glassProminent)
                    .disabled(viewModel.isBusy)
                }
            }
            .padding(16)
        }
        .sheet(item: $editingDocument) { document in
            documentEditSheet(document)
        }
        .task {
            do {
                try await viewModel.refreshRemoteState()
            } catch {
                if !(error is CancellationError) {
                    viewModel.lastError = error.localizedDescription
                }
            }
            await viewModel.refreshDocuments()
        }
    }

    private var documentMetadataSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("문서 메타데이터")
                .font(.headline)

            HStack {
                TextField("검색", text: $viewModel.documentSearchText)
                    .textFieldStyle(.roundedBorder)
                Picker("카테고리", selection: $viewModel.documentFilterCategory) {
                    Text("전체").tag("")
                    ForEach(AppViewModel.fixedCategories, id: \.self) { category in
                        Text(category).tag(category)
                    }
                }
                .frame(width: 170)
                TextField("태그", text: $viewModel.documentFilterTag)
                    .textFieldStyle(.roundedBorder)
                TextField("연도", text: $viewModel.documentFilterYear)
                    .textFieldStyle(.roundedBorder)
                    .frame(width: 80)
                TextField("프로젝트", text: $viewModel.documentFilterProject)
                    .textFieldStyle(.roundedBorder)
                Toggle("제외 포함", isOn: $viewModel.showExcludedDocuments)
                    .toggleStyle(.switch)
                Button("조회") {
                    Task { await viewModel.refreshDocuments() }
                }
            }

            Text("총 \(viewModel.documentsTotal)개")
                .font(.caption)
                .foregroundStyle(.secondary)

            ForEach(viewModel.documents.prefix(40)) { document in
                VStack(alignment: .leading, spacing: 4) {
                    Text(document.path)
                        .font(.caption)
                        .foregroundStyle(.secondary)
                        .lineLimit(1)
                    Text("\(document.category) · \(document.subcategory.isEmpty ? "-" : document.subcategory) · 중요도 \(String(format: "%.2f", document.importance))")
                        .font(.subheadline)
                    if !document.tags.isEmpty {
                        Text(document.tags.joined(separator: ", "))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }

                    HStack {
                        Button("편집") {
                            openEditor(for: document)
                        }
                        Button("재분류") {
                            Task { await viewModel.reclassifyDocument(docID: document.doc_id) }
                        }
                    }
                }
                .padding(10)
                .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
            }
        }
        .padding(12)
        .glassCard(cornerRadius: 12)
    }

    private var modelRuntimeSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("로컬 모델/엔진")
                .font(.headline)

            Text("일반 사용자는 아래 추천 모델에서 프로필만 골라 설치/사용하세요.")
                .font(.caption)
                .foregroundStyle(.secondary)

            if viewModel.catalogModels.isEmpty {
                Text("추천 모델 정보를 불러오는 중이거나 아직 없습니다.")
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                ForEach(["fast", "balanced", "advanced"], id: \.self) { profile in
                    let models = viewModel.catalogModels.filter { $0.profile == profile }
                    if !models.isEmpty {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(models.first?.profileTitle ?? profile)
                                .font(.subheadline.weight(.semibold))

                            ForEach(models) { model in
                                VStack(alignment: .leading, spacing: 6) {
                                    HStack {
                                        Text(model.name)
                                            .font(.subheadline.weight(.semibold))
                                        Spacer()
                                        Text(model.status.title)
                                            .font(.caption2)
                                            .padding(.horizontal, 8)
                                            .padding(.vertical, 3)
                                            .background(Color.white.opacity(0.12), in: Capsule())
                                    }

                                    Text(model.description)
                                        .font(.caption)
                                        .foregroundStyle(.secondary)

                                    HStack(spacing: 10) {
                                        Text("약 \(String(format: "%.1f", model.size_gb))GB")
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                        Text("권장 메모리 \(model.recommended_memory_gb)GB")
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                    }

                                    HStack {
                                        if model.status == .notInstalled || model.status == .failed {
                                            Button(model.status == .failed ? "재다운로드" : "다운로드") {
                                                Task { await viewModel.installCatalogModel(model.id) }
                                            }
                                            .buttonStyle(.glassProminent)
                                            .disabled(viewModel.isCatalogBusy)
                                        }

                                        if model.status == .installed {
                                            Button("사용하기") {
                                                Task { await viewModel.activateCatalogModel(model.id) }
                                            }
                                            .buttonStyle(.glassProminent)
                                            .disabled(viewModel.isCatalogBusy)
                                        }

                                        if model.status == .active {
                                            Text("현재 사용 중")
                                                .font(.caption)
                                                .foregroundStyle(.green)
                                        }

                                        if model.status == .installed || model.status == .active || model.status == .failed {
                                            Button("삭제") {
                                                Task { await viewModel.deleteCatalogModel(model.id) }
                                            }
                                            .buttonStyle(.bordered)
                                            .disabled(viewModel.isCatalogBusy)
                                        }
                                    }

                                    if let reason = model.failure_reason, !reason.isEmpty {
                                        Text(reason)
                                            .font(.caption2)
                                            .foregroundStyle(.red)
                                            .lineLimit(2)
                                    }
                                }
                                .padding(10)
                                .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
                            }
                        }
                    }
                }
            }

            Divider()

            Toggle("고급 모델 설정 보기", isOn: $viewModel.showAdvancedModelDetails)
                .toggleStyle(.switch)

            if viewModel.showAdvancedModelDetails {
                Picker("추론 엔진", selection: $viewModel.localEngine) {
                    ForEach(LocalEngine.allCases) { engine in
                        Text(engine.title).tag(engine)
                    }
                }
                .pickerStyle(.segmented)

                HStack {
                    Button("엔진 준비") {
                        Task { await viewModel.prepareRuntimeNow() }
                    }
                    .buttonStyle(.glassProminent)
                    .disabled(viewModel.isBusy)

                    if !viewModel.localRuntimeDetail.isEmpty {
                        Text(viewModel.localRuntimeDetail)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                    }
                }

                HStack {
                    Text("MLX 모델 경로")
                        .frame(width: 110, alignment: .leading)
                    TextField("예: /path/to/mlx-model", text: $viewModel.mlxModelPath)
                        .textFieldStyle(.roundedBorder)
                    Button("파일 선택") {
                        viewModel.chooseModelFile(for: .mlx)
                    }
                }

                HStack {
                    Text("llama GGUF 경로")
                        .frame(width: 110, alignment: .leading)
                    TextField("예: /path/to/model.gguf", text: $viewModel.llamaModelPath)
                        .textFieldStyle(.roundedBorder)
                    Button("파일 선택") {
                        viewModel.chooseModelFile(for: .llamaCPP)
                    }
                }

                Divider()

                Text("수동 URL 다운로드")
                    .font(.subheadline.weight(.semibold))

                HStack {
                    Picker("엔진", selection: $viewModel.modelDownloadEngine) {
                        ForEach(LocalEngine.allCases) { engine in
                            Text(engine.title).tag(engine)
                        }
                    }
                    .frame(width: 140)

                    TextField("다운로드 URL", text: $viewModel.modelDownloadURL)
                        .textFieldStyle(.roundedBorder)

                    TextField("파일명(선택)", text: $viewModel.modelDownloadFilename)
                        .textFieldStyle(.roundedBorder)
                        .frame(width: 180)

                    Button(viewModel.isDownloadingModel ? "다운로드 중..." : "다운로드") {
                        Task { await viewModel.downloadModel() }
                    }
                    .buttonStyle(.glassProminent)
                    .disabled(viewModel.isDownloadingModel || viewModel.modelDownloadURL.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                }

                if viewModel.availableModels.isEmpty {
                    Text("다운로드된 모델이 없습니다.")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                } else {
                    ForEach(viewModel.availableModels.prefix(24)) { model in
                        HStack {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(model.file_name)
                                    .font(.subheadline)
                                Text("\(model.engine.title) · \(ByteCountFormatter.string(fromByteCount: Int64(model.size_bytes), countStyle: .file))")
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                Text(model.path)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                            }
                            Spacer()
                            Button("경로 적용") {
                                viewModel.applyDownloadedModel(model)
                            }
                            .buttonStyle(.bordered)
                        }
                        .padding(8)
                        .background(Color.white.opacity(0.08), in: RoundedRectangle(cornerRadius: 10))
                    }
                }
            }

            if !viewModel.localRuntimeDetail.isEmpty {
                Text(viewModel.localRuntimeDetail)
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .glassCard(cornerRadius: 12)
    }

    private func openEditor(for document: DocumentMetadata) {
        editCategory = document.category
        editSubcategory = document.subcategory
        editDocumentType = document.document_type
        editTags = document.tags.joined(separator: ", ")
        editYear = document.year.map(String.init) ?? ""
        editProject = document.project ?? ""
        editImportance = document.importance
        editExcluded = document.excluded
        editingDocument = document
    }

    private func documentEditSheet(_ document: DocumentMetadata) -> some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("문서 메타데이터 편집")
                .font(.headline)
            Text(document.path)
                .font(.caption)
                .foregroundStyle(.secondary)
                .lineLimit(1)

            Picker("카테고리", selection: $editCategory) {
                ForEach(AppViewModel.fixedCategories, id: \.self) { category in
                    Text(category).tag(category)
                }
            }
            TextField("서브카테고리", text: $editSubcategory)
            TextField("문서 타입", text: $editDocumentType)
            TextField("태그(콤마)", text: $editTags)
            TextField("연도", text: $editYear)
            TextField("프로젝트", text: $editProject)
            HStack {
                Text("중요도 \(String(format: "%.2f", editImportance))")
                Slider(value: $editImportance, in: 0 ... 1)
            }
            Toggle("검색 제외", isOn: $editExcluded)

            HStack {
                Spacer()
                Button("취소") {
                    editingDocument = nil
                }
                Button("저장") {
                    Task {
                        await viewModel.updateDocumentMetadata(
                            docID: document.doc_id,
                            category: editCategory,
                            subcategory: editSubcategory,
                            documentType: editDocumentType,
                            tags: editTags
                                .split(separator: ",")
                                .map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
                                .filter { !$0.isEmpty },
                            year: Int(editYear),
                            project: editProject.isEmpty ? nil : editProject,
                            importance: editImportance,
                            excluded: editExcluded
                        )
                        editingDocument = nil
                    }
                }
                .buttonStyle(.glassProminent)
            }
        }
        .padding(18)
        .frame(width: 520)
    }
}
