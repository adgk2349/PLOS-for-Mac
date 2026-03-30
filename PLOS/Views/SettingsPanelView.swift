import SwiftUI
import Foundation

struct SettingsPanelView: View {
    @ObservedObject var viewModel: AppViewModel
    var onOpenStatusPanel: (() -> Void)? = nil

    @State var excludeInput = ""
    @State var showMemoryViewer = false
    @State var expandedPluginIDs: Set<String> = []
    @State var showAdvancedPluginRegistration = false
    @State var isPluginDropTargeted = false
    @State var showModelCatalogList = true
    @State var collapsedCatalogTierKeys: Set<String> = []
    @State private var baselineFingerprint = ""
    private let topPreferenceCardHeight: CGFloat = 280

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header

                topPreferenceRow
                runtimeSection
                modelCatalogSection
                pluginSection
                webSearchSection
                apiKeySection
                foldersSection
            }
            .padding(14)
        }
        .overlay(alignment: .bottomTrailing) {
            HStack(spacing: 10) {
                Button {
                    Task {
                        do {
                            try await viewModel.refreshRemoteState()
                        } catch {
                            viewModel.lastError = error.localizedDescription
                        }
                    }
                } label: {
                    Image(systemName: "arrow.clockwise")
                        .font(.system(size: 16, weight: .semibold))
                        .frame(width: 40, height: 40)
                }
                .buttonStyle(.plain)
                .frame(width: 40, height: 40)
                .contentShape(Circle())
                .plosGlassCircle()
                .disabled(viewModel.isBusy)

                Button {
                    Task { await viewModel.saveSettingsAndWorkspace() }
                } label: {
                    Image(systemName: "square.and.arrow.down")
                        .font(.system(size: 16, weight: .semibold))
                        .frame(width: 40, height: 40)
                }
                .buttonStyle(.plain)
                .frame(width: 40, height: 40)
                .contentShape(Circle())
                .plosGlassCircle()
                .disabled(!hasPendingSettingChanges || viewModel.isBusy)
            }
            .padding(18)
        }
        .onAppear {
            L10n.reloadLanguages()
            if viewModel.appLanguage != .auto, !L10n.availableLanguageOptions().contains(viewModel.appLanguage) {
                viewModel.appLanguage = .auto
            }
            baselineFingerprint = currentSettingsFingerprint
        }
        .onChange(of: viewModel.lastSettingsSavedAt) { _, _ in
            baselineFingerprint = currentSettingsFingerprint
        }
        .sheet(isPresented: $showMemoryViewer) {
            MemoryViewerSheet(viewModel: viewModel)
                .frame(minWidth: 760, minHeight: 560)
                .padding(16)
        }
    }

    private var topPreferenceRow: some View {
        HStack(alignment: .top, spacing: 16) {
            privacySection
                .frame(maxWidth: .infinity, minHeight: topPreferenceCardHeight, maxHeight: topPreferenceCardHeight, alignment: .topLeading)
            behaviorSection
                .frame(maxWidth: .infinity, minHeight: topPreferenceCardHeight, maxHeight: topPreferenceCardHeight, alignment: .topLeading)
            memorySection
                .frame(maxWidth: .infinity, minHeight: topPreferenceCardHeight, maxHeight: topPreferenceCardHeight, alignment: .topLeading)
        }
    }

    private var hasPendingSettingChanges: Bool {
        baselineFingerprint != currentSettingsFingerprint
    }

    private var currentSettingsFingerprint: String {
        let included = viewModel.includedFolderURLs.map(\.path).sorted().joined(separator: "|")
        let excluded = viewModel.excludedPaths.sorted().joined(separator: "|")
        return [
            viewModel.appLanguage.rawValue,
            viewModel.privacyMode.rawValue,
            "\(viewModel.hybridWebSearchEnabled)",
            viewModel.actionPermissionMode.rawValue,
            viewModel.quickInferencePreset.rawValue,
            viewModel.defaultWorkMode.rawValue,
            viewModel.localEngine.rawValue,
            viewModel.mlxModelPath,
            viewModel.llamaModelPath,
            viewModel.modelsStorageDirectoryPath,
            viewModel.runtimeStorageDirectoryPath,
            viewModel.openAIAPIKey,
            viewModel.anthropicAPIKey,
            "\(viewModel.adaptivePersonalizationEnabled)",
            "\(viewModel.sessionMemoryEnabled)",
            "\(viewModel.workspaceMemoryEnabled)",
            "\(viewModel.localMemoryOnly)",
            viewModel.workspaceMemoryMode.rawValue,
            viewModel.searxngURL,
            "\(viewModel.autoStartSearXNG)",
            included,
            excluded,
        ].joined(separator: "\n")
    }

    private var header: some View {
        HStack {
            Text(L10n.tr("settings.title", language: viewModel.appLanguage, fallbackKo: "설정", fallbackEn: "Settings", fallbackJa: "設定"))
                .font(.title2.weight(.bold))
            Spacer()
            if viewModel.isBusy {
                ProgressView()
            }
            Menu {
                ForEach(L10n.availableLanguageOptions()) { language in
                    Button {
                        viewModel.appLanguage = language
                    } label: {
                        HStack {
                            Text(L10n.languageOptionTitle(language, language: viewModel.appLanguage))
                            if viewModel.appLanguage == language {
                                Image(systemName: "checkmark")
                            }
                        }
                    }
                }
            } label: {
                Image(systemName: "globe")
                    .font(.system(size: 13, weight: .semibold))
                    .frame(width: 40, height: 40)
            }
            .buttonStyle(.plain)
            .contentShape(Circle())
            .plosGlassCircle()
            if let onOpenStatusPanel {
                Button {
                    onOpenStatusPanel()
                } label: {
                    Image(systemName: "chart.bar")
                        .font(.system(size: 13, weight: .semibold))
                        .frame(width: 40, height: 40)
                }
                .buttonStyle(.plain)
                .contentShape(Circle())
                .plosGlassCircle()
            }
        }
    }
}
