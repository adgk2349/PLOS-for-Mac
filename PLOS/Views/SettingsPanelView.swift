import SwiftUI
import Foundation

struct SettingsPanelView: View {
    @ObservedObject var viewModel: AppViewModel
    var onOpenStatusPanel: (() -> Void)? = nil

    @State var excludeInput = ""
    @State var showMemoryViewer = false

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header

                privacySection
                behaviorSection
                runtimeSection
                modelCatalogSection
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
}
