import Foundation
import SwiftUI

struct MemoryViewerSheet: View {
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
