import Foundation
import SwiftUI
import UniformTypeIdentifiers

extension SettingsPanelView {
    private var language: AppLanguage { viewModel.appLanguage }

    private func t(_ ko: String, _ en: String, _ ja: String) -> String {
        L10n.text(ko, en, ja, language: language)
    }

    var privacySection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(t("프라이버시", "Privacy", "プライバシー"))
                .font(.headline)

            Picker(t("프라이버시 모드", "Privacy mode", "プライバシーモード"), selection: $viewModel.privacyMode) {
                ForEach(PrivacyMode.allCases) { mode in
                    Text(mode.title(language: language)).tag(mode)
                }
            }
            .pickerStyle(.menu)
            .id("privacy-mode-\(language.rawValue)")
            .frame(maxWidth: .infinity, alignment: .leading)
            .onChange(of: viewModel.privacyMode) { _, _ in
                Task { await viewModel.saveSettingsAndWorkspace() }
            }

            if viewModel.privacyMode == .hybrid {
                Toggle(t("하이브리드에서 웹검색(인터넷 경로) 허용", "Allow web search in hybrid mode", "ハイブリッド時にWeb検索を許可"), isOn: $viewModel.hybridWebSearchEnabled)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .onChange(of: viewModel.hybridWebSearchEnabled) { _, _ in
                        Task { await viewModel.saveSettingsAndWorkspace() }
                    }
            }

            Picker(t("파일 권한 설정", "File permission", "ファイル権限"), selection: $viewModel.systemFilePermission) {
                ForEach(SystemFilePermission.allCases) { permission in
                    Text(permission.title(language: language)).tag(permission)
                }
            }
            .pickerStyle(.menu)
            .id("file-permission-\(language.rawValue)")
            .frame(maxWidth: .infinity, alignment: .leading)

            Picker(t("시스템 액션 승인", "System action approval", "システム操作の承認"), selection: $viewModel.actionPermissionMode) {
                ForEach(ActionPermissionMode.allCases) { mode in
                    Text(mode.title(language: language)).tag(mode)
                }
            }
            .pickerStyle(.menu)
            .id("action-permission-\(language.rawValue)")
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(12)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .plosGlassPanel()
    }

    var advancedSettingsSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            DisclosureGroup(isExpanded: $showAdvancedSettingsAccordion) {
                VStack(alignment: .leading, spacing: 12) {
                    advancedMultimodalVisionSection
                    pluginSection
                    webSearchSection
                    apiKeySection
                    foldersSection
                }
                .padding(.top, 8)
            } label: {
                Text(t("고급 설정", "Advanced settings", "詳細設定"))
                    .font(.headline)
            }
        }
        .padding(12)
        .plosGlassPanel()
    }

    var advancedMultimodalVisionSection: some View {
        DisclosureGroup(isExpanded: $showAdvancedMultimodalRuntime) {
            VStack(alignment: .leading, spacing: 8) {
                Toggle(
                    t("이미지 비전 해석 활성화", "Enable image vision analysis", "画像ビジョン解析を有効化"),
                    isOn: $viewModel.sidecarVisionEnabled
                )
                Toggle(
                    t("채팅에 생각 과정 표시", "Show thinking process in chat", "チャットに思考過程を表示"),
                    isOn: $viewModel.showThinkingProcessInChat
                )

                Divider()
                    .overlay(Color.white.opacity(0.1))

                Toggle(
                    t("MLX KV 캐시 TurboQuant 실험 모드", "MLX KV cache TurboQuant experimental mode", "MLX KVキャッシュ TurboQuant 実験モード"),
                    isOn: $viewModel.sidecarMlxKVQEnabled
                )

                Toggle(
                    t("대화 Turbo 모드(응답 길이/후처리 완화)", "Conversation Turbo mode (relax length/postprocess)", "会話Turboモード（長さ/後処理の緩和）"),
                    isOn: $viewModel.sidecarConversationTurboEnabled
                )

                Toggle(
                    t("추론 제한시간 비활성화(고급)", "Disable inference timeout (advanced)", "推論タイムアウト無効化（上級）"),
                    isOn: $viewModel.sidecarInferenceTimeoutDisabled
                )

                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text(t("본답변 제한시간(초)", "Main response timeout (sec)", "本回答タイムアウト（秒）"))
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Spacer(minLength: 8)
                        Text(
                            viewModel.sidecarInferenceTimeoutDisabled
                            ? t("무제한", "Unlimited", "無制限")
                            : "\(viewModel.sidecarMainResponseTimeoutSeconds)s"
                        )
                        .font(.caption2.monospacedDigit())
                        .foregroundStyle(.secondary)
                    }
                    Stepper(
                        value: $viewModel.sidecarMainResponseTimeoutSeconds,
                        in: 30 ... 3600,
                        step: 30
                    ) {
                        EmptyView()
                    }
                    .disabled(viewModel.sidecarInferenceTimeoutDisabled)
                }

                VStack(alignment: .leading, spacing: 6) {
                    HStack {
                        Text(t("보조경로 제한시간(초)", "Auxiliary path timeout (sec)", "補助経路タイムアウト（秒）"))
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Spacer(minLength: 8)
                        Text("\(viewModel.sidecarAuxiliaryTimeoutSeconds)s")
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    Stepper(
                        value: $viewModel.sidecarAuxiliaryTimeoutSeconds,
                        in: 4 ... 120,
                        step: 1
                    ) {
                        EmptyView()
                    }
                }

                if viewModel.sidecarInferenceTimeoutDisabled {
                    Text(
                        t(
                            "주의: 응답이 무한 대기 상태가 될 수 있습니다. 필요 시 sidecar 재시작으로 중단하세요.",
                            "Caution: responses may wait indefinitely. Restart sidecar to force-stop if needed.",
                            "注意: 応答が無期限待機になる場合があります。必要時はsidecar再起動で停止してください。"
                        )
                    )
                    .font(.caption2)
                    .foregroundStyle(.orange.opacity(0.95))
                }

                HStack(spacing: 8) {
                    Text(t("KVQ 모드", "KVQ mode", "KVQモード"))
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Picker("", selection: $viewModel.sidecarMlxKVQMode) {
                        Text("turbo3").tag(SidecarMlxKVQMode.turbo3)
                        Text("turbo4").tag(SidecarMlxKVQMode.turbo4)
                        Text("off").tag(SidecarMlxKVQMode.off)
                    }
                    .pickerStyle(.segmented)
                }
                .disabled(!viewModel.sidecarMlxKVQEnabled)

                VStack(alignment: .leading, spacing: 4) {
                    HStack {
                        Text(t("KV 인덱스 비트", "KV index bits", "KVインデックスビット"))
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        Spacer(minLength: 8)
                        Text("\(viewModel.sidecarMlxKVQBits)bit")
                            .font(.caption2.monospacedDigit())
                            .foregroundStyle(.secondary)
                    }
                    Slider(
                        value: Binding(
                            get: { Double(viewModel.sidecarMlxKVQBits) },
                            set: { viewModel.sidecarMlxKVQBits = Int($0.rounded()) }
                        ),
                        in: 2 ... 8,
                        step: 1
                    )
                }
                .disabled(!viewModel.sidecarMlxKVQEnabled)

                VStack(alignment: .leading, spacing: 4) {
                    Text(t("캡션 모델 ID", "Caption model ID", "キャプションモデルID"))
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    TextField("microsoft/git-base-coco", text: $viewModel.sidecarVisionCaptionModel)
                        .textFieldStyle(.plain)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .plosGlassInputFrame()
                }

                VStack(alignment: .leading, spacing: 4) {
                    Text(t("분류 모델 ID", "Classification model ID", "分類モデルID"))
                        .font(.caption.weight(.semibold))
                        .foregroundStyle(.secondary)
                    TextField("google/vit-base-patch16-224", text: $viewModel.sidecarVisionClassifyModel)
                        .textFieldStyle(.plain)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .plosGlassInputFrame()
                }

                Text(t("고급 옵션: 저장 시 sidecar 재시작 후 적용됩니다.", "Advanced option: applied after sidecar restart on save.", "詳細オプション: 保存時にsidecar再起動後に適用されます。"))
                    .font(.caption2)
                    .foregroundStyle(.secondary)
            }
            .padding(.top, 6)
        } label: {
            Text(t("고급: 멀티모달/비전", "Advanced: multimodal/vision", "詳細: マルチモーダル/ビジョン"))
                .font(.subheadline.weight(.semibold))
                .foregroundStyle(.secondary)
        }
    }

    var behaviorSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(t("기본 작업 동작", "Default behavior", "基本動作"))
                .font(.headline)

            Picker(t("추론 프리셋", "Inference preset", "推論プリセット"), selection: $viewModel.quickInferencePreset) {
                ForEach(QuickInferencePreset.allCases) { preset in
                    Text(preset.title(language: language)).tag(preset)
                }
            }
            .pickerStyle(.menu)
            .id("inference-preset-\(language.rawValue)")
            .frame(maxWidth: .infinity, alignment: .leading)
            .onChange(of: viewModel.quickInferencePreset) { _, newValue in
                viewModel.applyQuickInferencePreset(newValue)
            }

            Picker(t("특화 모드", "Specialized mode", "特化モード"), selection: $viewModel.defaultWorkMode) {
                ForEach(WorkMode.allCases) { mode in
                    Text(mode.title(language: language)).tag(mode)
                }
            }
            .pickerStyle(.menu)
            .id("default-mode-\(language.rawValue)")
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .padding(12)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .plosGlassPanel()
    }

    var runtimeSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(t("로컬 모델 / 엔진", "Local model / engine", "ローカルモデル / エンジン"))
                .font(.headline)

            if !viewModel.installedModelsSorted.isEmpty {
                DisclosureGroup(isExpanded: $showInstalledModelsList) {
                    VStack(alignment: .leading, spacing: 8) {
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
                                Button(viewModel.isInstalledModelActive(model) ? t("사용중", "Active", "使用中") : t("사용", "Use", "使用")) {
                                    Task { await viewModel.selectInstalledModel(model) }
                                }
                                .buttonStyle(.plain)
                            }
                            .padding(.horizontal, 10)
                            .padding(.vertical, 8)
                            .plosGlassInputFrame()
                        }
                    }
                    .padding(.top, 6)
                } label: {
                    HStack(spacing: 8) {
                        Text(t("설치된 모델", "Installed models", "インストール済みモデル"))
                            .font(.subheadline.weight(.semibold))
                        Text("\(min(viewModel.installedModelsSorted.count, 8))")
                            .font(.caption2.weight(.semibold))
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 7)
                            .padding(.vertical, 3)
                            .plosGlassChip()
                    }
                }
            } else {
                Text(t("설치된 모델이 없습니다. 아래 카탈로그에서 먼저 다운로드해 주세요.", "No installed model. Download one from catalog first.", "インストール済みモデルがありません。カタログから先にダウンロードしてください。"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            }

            if !viewModel.localRuntimeDetail.isEmpty {
                Text(viewModel.localRuntimeDetail)
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .lineLimit(2)
            }

            VStack(alignment: .leading, spacing: 8) {
                Text(t("저장소 경로", "Storage paths", "保存先パス"))
                    .font(.subheadline.weight(.semibold))

                HStack(alignment: .top, spacing: 10) {
                    VStack(alignment: .leading, spacing: 6) {
                        Text(t("모델 저장 경로", "Model storage path", "モデル保存パス"))
                            .font(.caption.weight(.semibold))
                        Text(viewModel.modelsStorageDirectoryPath)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                        HStack(spacing: 8) {
                            Button(t("폴더 선택", "Choose folder", "フォルダ選択")) {
                                Task { await viewModel.chooseModelsStorageDirectory() }
                            }
                            .buttonStyle(.plain)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .plosGlassControl()
                            .disabled(viewModel.isBusy)

                            Button(t("기본값 복원", "Reset default", "既定値に戻す")) {
                                Task { await viewModel.resetModelsStorageDirectoryToDefault() }
                            }
                            .buttonStyle(.plain)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .plosGlassControl()
                            .disabled(viewModel.isBusy)
                        }
                        Text(
                            t("적용 경로", "Effective path", "適用パス")
                            + ": "
                            + (viewModel.effectiveModelsStorageDirectoryPath.isEmpty ? viewModel.modelsStorageDirectoryPath : viewModel.effectiveModelsStorageDirectoryPath)
                        )
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                    }
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .plosGlassInputFrame()

                    VStack(alignment: .leading, spacing: 6) {
                        Text(t("런타임(venv) 경로", "Runtime (venv) path", "ランタイム(venv)パス"))
                            .font(.caption.weight(.semibold))
                        Text(viewModel.runtimeStorageDirectoryPath)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(2)
                        HStack(spacing: 8) {
                            Button(t("폴더 선택", "Choose folder", "フォルダ選択")) {
                                Task { await viewModel.chooseRuntimeStorageDirectory() }
                            }
                            .buttonStyle(.plain)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .plosGlassControl()
                            .disabled(viewModel.isBusy)

                            Button(t("기본값 복원", "Reset default", "既定値に戻す")) {
                                Task { await viewModel.resetRuntimeStorageDirectoryToDefault() }
                            }
                            .buttonStyle(.plain)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 6)
                            .plosGlassControl()
                            .disabled(viewModel.isBusy)
                        }
                        Text(
                            t("적용 경로", "Effective path", "適用パス")
                            + ": "
                            + (viewModel.effectiveRuntimeStorageDirectoryPath.isEmpty ? viewModel.runtimeStorageDirectoryPath : viewModel.effectiveRuntimeStorageDirectoryPath)
                        )
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .lineLimit(2)
                    }
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .plosGlassInputFrame()
                }

                if !viewModel.storageDirectoryWarning.isEmpty {
                    Text(viewModel.storageDirectoryWarning)
                        .font(.caption2)
                        .foregroundStyle(.orange.opacity(0.95))
                        .lineLimit(3)
                }
            }
            .padding(.horizontal, 2)

            VStack(alignment: .leading, spacing: 8) {
                Text(t("모델 성능 가이드", "Model capability guide", "モデル性能ガイド"))
                    .font(.caption.weight(.semibold))
                HStack(alignment: .top, spacing: 10) {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("16GB")
                            .font(.caption.weight(.semibold))
                        Text(t("3B~8B 중심. 기본 대화/검색에는 적합하고, 긴 요약/정리는 외부 API 경로가 안정적입니다.", "Best with 3B~8B. Good for basic chat/search; external API is safer for long summaries.", "3B~8B中心。基本会話/検索向けで、長い要約は外部APIが安定します。"))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .plosGlassInputFrame()

                    VStack(alignment: .leading, spacing: 4) {
                        Text("32GB")
                            .font(.caption.weight(.semibold))
                        Text(t("14B~32B까지 일반 요약/정리를 로컬에서 처리 가능. 고난도 추론은 외부 API를 권장합니다.", "Can handle general local summarization up to 14B~32B. Use external API for harder reasoning.", "14B~32Bで一般的な要約はローカル可能。高難度推論は外部API推奨。"))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .plosGlassInputFrame()

                    VStack(alignment: .leading, spacing: 4) {
                        Text("64GB+")
                            .font(.caption.weight(.semibold))
                        Text(t("70B급도 로컬 처리 가능하지만 속도 비용이 큽니다. 장시간 작업은 발열/전력까지 고려하세요.", "Even 70B can run locally, but latency is high. Consider sustained heat/power for long runs.", "70B級もローカル動作可能ですが遅延コストが大きいです。長時間運用では発熱/電力も考慮してください。"))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    .frame(maxWidth: .infinity, alignment: .topLeading)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .plosGlassInputFrame()
                }

                HStack(alignment: .top, spacing: 10) {
                    Text(t("주의: Gemma/DeepSeek 같은 8B+ 모델을 RAM 부족 상태에서 오래 쓰면 swap I/O 증가로 SSD 수명에 영향을 줄 수 있습니다.", "Caution: long swap-heavy runs with 8B+ models can increase SSD wear.", "注意: 8B+モデルをメモリ不足で長時間使うとSSD負荷が増える可能性があります。"))
                        .font(.caption2)
                        .foregroundStyle(.orange.opacity(0.95))
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .plosGlassInputFrame()
                    Text(t("정책 권장: 16GB급 모델에서 '요약/정리' 요청은 API 경로 우선", "Policy: on 16GB-class models, prefer API for summary/organization", "推奨: 16GB級モデルでは要約/整理はAPI優先"))
                        .font(.caption2)
                        .foregroundStyle(.secondary)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .plosGlassInputFrame()
                }
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
                Text(t("모델 다운로드 선택지", "Model catalog", "モデルカタログ"))
                    .font(.headline)
                Spacer()
                Button(t("새로고침", "Refresh", "更新")) {
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

            Text(t("현재 시스템 메모리: 약 \(systemMemoryGB)GB · 권장 사양 미만 모델은 다운로드가 비활성화됩니다.", "System memory: ~\(systemMemoryGB)GB · downloads are disabled below recommended spec.", "現在のシステムメモリ: 約\(systemMemoryGB)GB · 推奨未満モデルはダウンロード不可です。"))
                .font(.caption)
                .foregroundStyle(.secondary)

            DisclosureGroup(isExpanded: $showModelCatalogList) {
                VStack(alignment: .leading, spacing: 8) {
                    if catalogVisibleModels.isEmpty {
                        Text(t("카탈로그 모델을 불러오지 못했습니다. sidecar 상태를 확인한 뒤 새로고침해 주세요.", "Failed to load catalog models. Check sidecar status and refresh.", "カタログモデルを取得できませんでした。sidecar状態を確認して再読み込みしてください。"))
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    } else {
                        ForEach(catalogTierBuckets, id: \.key) { bucket in
                            DisclosureGroup(isExpanded: catalogTierExpansionBinding(for: bucket.key)) {
                                VStack(alignment: .leading, spacing: 8) {
                                    ForEach(bucket.models) { model in
                                        catalogModelRow(model)
                                    }
                                }
                                .padding(.top, 4)
                                .frame(maxWidth: .infinity, alignment: .leading)
                            } label: {
                                HStack(spacing: 8) {
                                    Text(bucket.title)
                                        .font(.subheadline.weight(.semibold))
                                        .foregroundStyle(.secondary)
                                    Text("\(bucket.models.count)")
                                        .font(.caption2.weight(.semibold))
                                        .foregroundStyle(.secondary)
                                        .padding(.horizontal, 7)
                                        .padding(.vertical, 3)
                                        .plosGlassChip()
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)
                        }
                    }
                }
                .padding(.top, 4)
                .frame(maxWidth: .infinity, alignment: .leading)
            } label: {
                Text(t("목록 보기", "Show list", "一覧を表示"))
                    .font(.subheadline.weight(.semibold))
                    .foregroundStyle(.secondary)
            }
        }
        .padding(12)
        .plosGlassPanel()
    }

    var pluginSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text(t("확장 / 플러그인", "Extensions / Plugins", "拡張 / プラグイン"))
                    .font(.headline)
                Spacer()
                Button(t("새로고침", "Refresh", "更新")) {
                    Task { await viewModel.refreshExtensionsNow() }
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassControl()
                .disabled(viewModel.isPluginBusy || viewModel.isBusy)
            }

            if viewModel.extensionCapabilities.isEmpty {
                Text(t("현재 capability source는 built-in 기본값입니다.", "Capability source is currently built-in by default.", "現在のcapability sourceはbuilt-inが既定です。"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                let capabilityOverrides = viewModel.extensionCapabilities.filter { item in
                    let hasPluginID = !(item.plugin_id ?? "").isEmpty
                    let hasIssue = item.error_code != nil || !(item.blocked_reason ?? "").isEmpty
                    return item.source != .builtIn || hasPluginID || hasIssue
                }
                if capabilityOverrides.isEmpty {
                    HStack(spacing: 8) {
                        Text(t("내장 패키지", "Built-in package", "内蔵パッケージ"))
                            .font(.subheadline.weight(.semibold))
                        Spacer()
                        Text(t("모든 기능이 내장 기본값으로 동작 중", "All capabilities are using built-in defaults", "すべての機能が内蔵既定値で動作中"))
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .plosGlassInputFrame()
                } else {
                    ForEach(capabilityOverrides) { item in
                        HStack(spacing: 8) {
                            Text(item.capability.title(language: language))
                                .font(.subheadline.weight(.semibold))
                            Spacer()
                            Text(item.source.title(language: language))
                                .font(.caption2.weight(.semibold))
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 5)
                                .plosGlassChip()
                            if let pluginID = item.plugin_id, !pluginID.isEmpty {
                                Text(pluginID)
                                    .font(.caption2)
                                    .foregroundStyle(.secondary)
                                    .lineLimit(1)
                            }
                        }
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .plosGlassInputFrame()
                    }
                }
            }

            Divider()
                .overlay(Color.white.opacity(0.08))
                .padding(.vertical, 2)

            Text(t("플러그인 추가", "Add plugin", "プラグイン追加"))
                .font(.subheadline.weight(.semibold))

            HStack(spacing: 8) {
                Button(t("플러그인 추가(파일/폴더)", "Add plugin (file/folder)", "プラグイン追加（ファイル/フォルダ）")) {
                    Task { await viewModel.registerPluginFromManifestFile() }
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassControl()
                .disabled(viewModel.isPluginBusy || viewModel.isBusy)
            }

            Text(
                t(
                    "폴더 내부 json 파일 선택 방식 지원: 폴더 선택 시 내부 plugin.json이 자동 사용됩니다.",
                    "Folder-internal JSON selection is supported: selecting a folder automatically uses its plugin.json.",
                    "フォルダ内JSON選択方式に対応: フォルダ選択時は内部のplugin.jsonが自動適用されます。"
                )
            )
            .font(.caption2)
            .foregroundStyle(.secondary)

            VStack(alignment: .leading, spacing: 6) {
                Text(t("플러그인 폴더 또는 manifest JSON 파일을 여기로 드래그 앤 드롭", "Drag & drop plugin folder or manifest JSON here", "プラグインフォルダまたはmanifest JSONをここにドラッグ＆ドロップ"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
            .padding(.horizontal, 12)
            .padding(.vertical, 14)
            .background(
                RoundedRectangle(cornerRadius: 10, style: .continuous)
                    .stroke(
                        isPluginDropTargeted ? Color.accentColor.opacity(0.7) : Color.white.opacity(0.12),
                        style: StrokeStyle(lineWidth: isPluginDropTargeted ? 1.8 : 1.0, dash: [5, 5])
                    )
            )
            .onDrop(of: [UTType.fileURL.identifier], isTargeted: $isPluginDropTargeted) { providers in
                Task { await viewModel.registerPluginFromDroppedItemProviders(providers) }
                return !providers.isEmpty
            }

            DisclosureGroup(isExpanded: $showAdvancedPluginRegistration) {
                VStack(alignment: .leading, spacing: 8) {
                    HStack(spacing: 8) {
                        TextField(t("플러그인 ID", "Plugin ID", "プラグインID"), text: $viewModel.pluginDraftID)
                            .textFieldStyle(.plain)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 8)
                            .plosGlassInputFrame()

                        TextField(t("버전", "Version", "バージョン"), text: $viewModel.pluginDraftVersion)
                            .textFieldStyle(.plain)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 8)
                            .frame(maxWidth: 140)
                            .plosGlassInputFrame()
                    }

                    TextField(t("entrypoint (예: python -m my_plugin.main)", "entrypoint (e.g., python -m my_plugin.main)", "entrypoint (例: python -m my_plugin.main)"), text: $viewModel.pluginDraftEntrypoint)
                        .textFieldStyle(.plain)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .plosGlassInputFrame()

                    HStack(spacing: 8) {
                        Picker(t("빌드 타겟", "Build target", "ビルドターゲット"), selection: $viewModel.pluginDraftBuildTarget) {
                            ForEach(PluginBuildTarget.allCases) { target in
                                Text(target.title(language: language)).tag(target)
                            }
                        }
                        .pickerStyle(.menu)
                        .id("plugin-build-target-\(language.rawValue)")
                        .frame(maxWidth: 200)

                        Picker(t("프라이버시", "Privacy", "プライバシー"), selection: $viewModel.pluginDraftPrivacyMode) {
                            ForEach(PluginPrivacyMode.allCases) { mode in
                                Text(mode.title(language: language)).tag(mode)
                            }
                        }
                        .pickerStyle(.menu)
                        .id("plugin-privacy-mode-\(language.rawValue)")
                        .frame(maxWidth: 220)

                        Toggle(t("등록 즉시 활성화", "Enable immediately", "登録後すぐ有効化"), isOn: $viewModel.pluginDraftEnabled)
                    }

                    TextField(
                        t(
                            "권한(쉼표로 구분)",
                            "Permissions (comma-separated)",
                            "権限（カンマ区切り）"
                        ),
                        text: $viewModel.pluginDraftPermissions
                    )
                    .textFieldStyle(.plain)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .plosGlassInputFrame()

                    TextField(
                        t(
                            "서명(선택)",
                            "Signature (optional)",
                            "署名（任意）"
                        ),
                        text: $viewModel.pluginDraftSignature
                    )
                    .textFieldStyle(.plain)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .plosGlassInputFrame()

                    VStack(alignment: .leading, spacing: 6) {
                        Text(t("기능", "Capabilities", "機能"))
                            .font(.caption.weight(.semibold))
                            .foregroundStyle(.secondary)
                        ForEach(ExtensionCapability.allCases) { capability in
                            Toggle(capability.title, isOn: viewModel.pluginDraftCapabilityBinding(capability))
                        }
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .plosGlassInputFrame()

                    HStack(spacing: 8) {
                        Button(t("등록", "Register", "登録")) {
                            Task { await viewModel.registerPluginFromDraft() }
                        }
                        .buttonStyle(.plain)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .plosGlassControl()
                        .disabled(viewModel.isPluginBusy || viewModel.isBusy)

                        Button(t("초기화", "Reset", "リセット")) {
                            viewModel.resetPluginDraft()
                        }
                        .buttonStyle(.plain)
                        .padding(.horizontal, 10)
                        .padding(.vertical, 8)
                        .plosGlassControl()
                        .disabled(viewModel.isPluginBusy || viewModel.isBusy)
                    }
                }
                .padding(.top, 6)
            } label: {
                Text(t("고급: 수동 등록", "Advanced: manual registration", "詳細: 手動登録"))
                    .font(.caption.weight(.semibold))
                    .foregroundStyle(.secondary)
            }

            if viewModel.pluginEntries.isEmpty {
                Text(t("등록된 플러그인이 없습니다.", "No registered plugins.", "登録済みプラグインはありません。"))
                    .font(.caption)
                    .foregroundStyle(.secondary)
            } else {
                Text(t("등록된 플러그인", "Registered plugins", "登録済みプラグイン"))
                    .font(.subheadline.weight(.semibold))
                ForEach(viewModel.pluginEntries) { entry in
                    let isBuiltIn = viewModel.isBuiltInPluginEntry(entry)
                    DisclosureGroup(isExpanded: pluginExpansionBinding(for: entry.plugin_id)) {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(t("엔트리포인트", "Entrypoint", "エントリポイント") + ": \(entry.manifest.entrypoint)")
                                .font(.caption2)
                                .foregroundStyle(.secondary)
                                .lineLimit(2)

                            Text(t("포함 기능", "Included capabilities", "含まれる機能"))
                                .font(.caption.weight(.semibold))
                                .foregroundStyle(.secondary)

                            VStack(alignment: .leading, spacing: 4) {
                                ForEach(entry.manifest.capabilities, id: \.rawValue) { capability in
                                    HStack(spacing: 6) {
                                        Image(systemName: "checkmark.circle.fill")
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                        Text(capability.title(language: language))
                                            .font(.caption2)
                                            .foregroundStyle(.secondary)
                                    }
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                }
                            }
                            .frame(maxWidth: .infinity, alignment: .leading)

                            if !isBuiltIn {
                                HStack(spacing: 6) {
                                    if entry.enabled {
                                        Button(t("비활성화", "Disable", "無効化")) {
                                            Task { await viewModel.disablePlugin(entry.plugin_id) }
                                        }
                                    } else {
                                        Button(t("활성화", "Enable", "有効化")) {
                                            Task { await viewModel.enablePlugin(entry.plugin_id) }
                                        }
                                    }
                                    Button(t("삭제", "Delete", "削除")) {
                                        Task { await viewModel.deletePlugin(entry.plugin_id) }
                                    }
                                }
                                .buttonStyle(.plain)
                            }

                            if let validationError = entry.validation_error, !validationError.isEmpty {
                                Text(validationError)
                                    .font(.caption2)
                                    .foregroundStyle(.red.opacity(0.9))
                            }
                        }
                        .padding(.top, 6)
                        .frame(maxWidth: .infinity, alignment: .leading)
                    } label: {
                        HStack(alignment: .top, spacing: 8) {
                            VStack(alignment: .leading, spacing: 2) {
                                Text(isBuiltIn ? t("builtin.core (내장 패키지)", "builtin.core (Built-in package)", "builtin.core（内蔵パッケージ）") : entry.plugin_id)
                                    .font(.subheadline.weight(.semibold))
                                Text(
                                    isBuiltIn
                                    ? t("항상 활성 · 내장 기능 묶음", "Always enabled · bundled capabilities", "常時有効 · 内蔵機能バンドル")
                                    : "v\(entry.manifest.version) · \(entry.state) · \(entry.enabled ? t("활성", "enabled", "有効") : t("비활성", "disabled", "無効"))"
                                )
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            }
                            Spacer()
                            Text(isBuiltIn ? t("내장", "Built-in", "内蔵") : (entry.enabled ? t("활성", "enabled", "有効") : t("비활성", "disabled", "無効")))
                                .font(.caption2.weight(.semibold))
                                .foregroundStyle(.secondary)
                                .padding(.horizontal, 8)
                                .padding(.vertical, 4)
                                .plosGlassChip()
                        }
                    }
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .plosGlassInputFrame()
                    .frame(maxWidth: .infinity, alignment: .leading)
                }
            }
        }
        .padding(12)
        .plosGlassPanel()
    }

    private func pluginExpansionBinding(for pluginID: String) -> Binding<Bool> {
        Binding(
            get: { expandedPluginIDs.contains(pluginID) },
            set: { isExpanded in
                if isExpanded {
                    expandedPluginIDs.insert(pluginID)
                } else {
                    expandedPluginIDs.remove(pluginID)
                }
            }
        )
    }

    var webSearchSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(t("웹 검색 (SearXNG)", "Web Search (SearXNG)", "Web検索 (SearXNG)"))
                .font(.headline)

            TextField(t("SearXNG URL (예: http://localhost:8080)", "SearXNG URL (e.g., http://localhost:8080)", "SearXNG URL (例: http://localhost:8080)"), text: $viewModel.searxngURL)
                .textFieldStyle(.plain)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassInputFrame()

            Toggle(t("SearXNG Docker 자동 실행", "Auto-start SearXNG Docker", "SearXNG Dockerを自動起動"), isOn: $viewModel.autoStartSearXNG)
                .help(t("설정을 저장할 때 sidecar에서 SearXNG Docker 컨테이너를 시작하거나 중지합니다.", "Starts/stops SearXNG Docker container from sidecar when saving settings.", "設定保存時にsidecarからSearXNG Dockerコンテナを起動/停止します。"))

            Text(t("SearXNG는 프라이버시가 강화된 로컬 검색엔진 메타검색기입니다. Docker가 설치되어 있어야 자동 실행이 가능합니다.", "SearXNG is a privacy-respecting metasearch engine. Docker must be installed for auto-start.", "SearXNGはプライバシー重視のメタ検索エンジンです。自動起動にはDockerのインストールが必要です。"))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(12)
        .plosGlassPanel()
    }

    var apiKeySection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(t("외부 API 키", "External API keys", "外部APIキー"))
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

            Text(t("저장은 macOS Keychain에 보관됩니다.", "Saved in macOS Keychain.", "macOS Keychainに保存されます。"))
                .font(.caption)
                .foregroundStyle(.secondary)
        }
        .padding(12)
        .plosGlassPanel()
    }

    var foldersSection: some View {
        VStack(alignment: .leading, spacing: 10) {
            Text(t("인덱싱 폴더", "Indexing folders", "インデックス対象フォルダ"))
                .font(.headline)

            HStack(spacing: 8) {
                Button(t("폴더 추가", "Add folder", "フォルダ追加")) {
                    viewModel.addFolder()
                }
                .buttonStyle(.plain)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassControl()

                Button(t("전체 재인덱싱", "Full reindex", "フル再インデックス")) {
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
                    Button(t("삭제", "Remove", "削除")) {
                        viewModel.removeFolder(url.path)
                    }
                    .buttonStyle(.plain)
                }
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassInputFrame()
            }

            HStack(spacing: 8) {
                TextField(t("제외 폴더 경로", "Excluded path", "除外パス"), text: $excludeInput)
                    .textFieldStyle(.plain)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .plosGlassInputFrame()

                Button(t("추가", "Add", "追加")) {
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
                        Button(t("제거", "Remove", "削除")) {
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
            Text(t("로컬 메모리", "Local memory", "ローカルメモリ"))
                .font(.headline)

            Toggle(t("적응형 개인화", "Adaptive personalization", "適応型パーソナライズ"), isOn: $viewModel.adaptivePersonalizationEnabled)
            Toggle(t("세션 메모리", "Session memory", "セッションメモリ"), isOn: $viewModel.sessionMemoryEnabled)
            Toggle(t("워크스페이스 메모리", "Workspace memory", "ワークスペースメモリ"), isOn: $viewModel.workspaceMemoryEnabled)
            Toggle(t("로컬 메모리만 사용", "Local memory only", "ローカルメモリのみ"), isOn: $viewModel.localMemoryOnly)

            Picker(t("워크스페이스 메모리 모드", "Workspace memory mode", "ワークスペースメモリモード"), selection: $viewModel.workspaceMemoryMode) {
                ForEach(WorkspaceMemoryMode.allCases) { mode in
                    Text(mode.title(language: language)).tag(mode)
                }
            }
            .pickerStyle(.menu)
            .id("workspace-memory-mode-\(language.rawValue)")
            .frame(maxWidth: 260)

            HStack(spacing: 8) {
                Button(t("메모리 보기", "View memory", "メモリ表示")) {
                    showMemoryViewer = true
                }
                .buttonStyle(.plain)
                .frame(maxWidth: .infinity, minHeight: 20)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassControl()

                Button(t("세션 초기화", "Clear session", "セッション初期化")) {
                    Task { await viewModel.clearMemory(scope: .session) }
                }
                .buttonStyle(.plain)
                .frame(maxWidth: .infinity, minHeight: 20)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassControl()

                Button(t("전체 초기화", "Clear all", "全体初期化")) {
                    Task { await viewModel.clearMemory(scope: .all) }
                }
                .buttonStyle(.plain)
                .frame(maxWidth: .infinity, minHeight: 20)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassControl()
            }
        }
        .padding(12)
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
        .plosGlassPanel()
    }

    var catalogTierBuckets: [(key: String, title: String, models: [ModelCatalogItem])] {
        let sorted = catalogVisibleModels.sorted {
            if $0.recommended_memory_gb != $1.recommended_memory_gb {
                return $0.recommended_memory_gb < $1.recommended_memory_gb
            }
            return $0.size_gb < $1.size_gb
        }
        let tiers = ["tier16", "tier64", "tier256", "tier500"]
        return tiers.compactMap { tier in
            let models = sorted.filter { memoryTierTitle(for: $0.recommended_memory_gb) == tier }
            return models.isEmpty ? nil : (tier, displayTierTitle(tier), models)
        }
    }

    func memoryTierTitle(for memoryGB: Int) -> String {
        if memoryGB <= 16 { return "tier16" }
        if memoryGB <= 64 { return "tier64" }
        if memoryGB <= 256 { return "tier256" }
        return "tier500"
    }

    func displayTierTitle(_ tierKey: String) -> String {
        switch tierKey {
        case "tier16":
            return t("16GB 이상", "16GB+", "16GB以上")
        case "tier64":
            return t("64GB 이상", "64GB+", "64GB以上")
        case "tier256":
            return t("256GB 이상", "256GB+", "256GB以上")
        default:
            return t("500GB 이상", "500GB+", "500GB以上")
        }
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

    private func catalogTierExpansionBinding(for tierKey: String) -> Binding<Bool> {
        Binding(
            get: { !collapsedCatalogTierKeys.contains(tierKey) },
            set: { isExpanded in
                if isExpanded {
                    collapsedCatalogTierKeys.remove(tierKey)
                } else {
                    collapsedCatalogTierKeys.insert(tierKey)
                }
            }
        )
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
                Text(status.title(language: language))
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
                    .padding(.horizontal, 8)
                    .padding(.vertical, 5)
                    .plosGlassChip()
            }

            if !canDownload {
                Text(t("이 모델은 최소 \(model.recommended_memory_gb)GB RAM 권장입니다. 현재: \(systemMemoryGB)GB", "Recommended RAM for this model is at least \(model.recommended_memory_gb)GB. Current: \(systemMemoryGB)GB", "このモデルの推奨RAMは最低\(model.recommended_memory_gb)GBです。現在: \(systemMemoryGB)GB"))
                    .font(.caption2)
                    .foregroundStyle(.red.opacity(0.9))
            }

            HStack(spacing: 8) {
                switch status {
                case .notInstalled:
                    Button(t("다운로드", "Download", "ダウンロード")) {
                        Task { await viewModel.installCatalogModel(model.id) }
                    }
                    .disabled(viewModel.isCatalogBusy || viewModel.isBusy || !canDownload)
                case .downloading:
                    if let progress = catalogProgressValue(for: model) {
                        HStack(spacing: 8) {
                            Text(String(format: t("다운로드 중 %.1f%%", "Downloading %.1f%%", "ダウンロード中 %.1f%%"), progress * 100.0))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            ProgressView(value: progress, total: 1.0)
                                .progressViewStyle(.linear)
                                .frame(width: 130)
                                .controlSize(.small)
                        }
                    } else {
                        HStack(spacing: 8) {
                            Text(t("다운로드 중 (크기 정보 없음)", "Downloading (size unknown)", "ダウンロード中 (サイズ不明)"))
                                .font(.caption)
                                .foregroundStyle(.secondary)
                            ProgressView()
                                .controlSize(.small)
                        }
                    }
                case .installed:
                    Button(model.active ? t("사용 중", "Active", "使用中") : t("사용하기", "Use", "使用する")) {
                        Task { await viewModel.activateCatalogModel(model.id) }
                    }
                    .disabled(viewModel.isCatalogBusy || viewModel.isBusy || model.active)
                case .active:
                    Button(t("사용 중", "Active", "使用中")) {}
                        .disabled(true)
                case .failed:
                    Button(t("재시도", "Retry", "再試行")) {
                        Task { await viewModel.installCatalogModel(model.id) }
                    }
                    .disabled(viewModel.isCatalogBusy || viewModel.isBusy || !canDownload)
                }

                if status != .notInstalled && status != .downloading {
                    Button(t("삭제", "Delete", "削除")) {
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

    func catalogProgressValue(for model: ModelCatalogItem) -> Double? {
        if let fromLive = viewModel.catalogInstallProgress[model.id] {
            return min(max(fromLive, 0.0), 1.0)
        }
        if
            model.status == .downloading,
            let downloaded = model.downloaded_bytes,
            let total = model.total_bytes,
            total > 0
        {
            return min(max(Double(downloaded) / Double(total), 0.0), 1.0)
        }
        if model.status == .downloading, let fromModel = model.progress_percent {
            return min(max(fromModel / 100.0, 0.0), 1.0)
        }
        return nil
    }
}
