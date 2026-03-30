import Foundation
import SwiftUI

struct MemoryViewerSheet: View {
    @ObservedObject var viewModel: AppViewModel
    private var language: AppLanguage { viewModel.appLanguage }

    private func t(_ ko: String, _ en: String, _ ja: String) -> String {
        L10n.text(ko, en, ja, language: language)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text(t("메모리 보기", "Memory viewer", "メモリビューア"))
                    .font(.title2.weight(.bold))

                memoryBlock(t("세션", "Session", "セッション"), items: viewModel.sessionMemoryItems.map { "\($0.key): \($0.value_json)" })
                memoryBlock(t("워크스페이스", "Workspace", "ワークスペース"), items: viewModel.workspaceMemoryItems.map { "\($0.memory_type) / \($0.key): \($0.value_json)" })
                memoryBlock(t("환경설정", "Preferences", "設定"), items: viewModel.preferenceMemoryItems.map { "\($0.key): \($0.value_json)" })
                memoryBlock(t("이벤트", "Episodic", "イベント"), items: viewModel.episodicMemoryItems.map { "\($0.event_type): \($0.summary)" })

                VStack(alignment: .leading, spacing: 8) {
                    Text(t("고정", "Pinned", "固定"))
                        .font(.headline)
                    if viewModel.pinnedMemoryItems.isEmpty {
                        Text(t("고정 메모리가 없습니다.", "No pinned memory.", "固定メモリはありません。"))
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
                                Button(t("해제", "Unpin", "解除")) {
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
                Text(t("비어 있음", "Empty", "空です"))
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
