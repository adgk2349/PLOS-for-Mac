import SwiftUI

struct StatusPanelView: View {
    @ObservedObject var viewModel: AppViewModel
    private var language: AppLanguage { viewModel.appLanguage }

    private func t(_ ko: String, _ en: String, _ ja: String) -> String {
        L10n.text(ko, en, ja, language: language)
    }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text(t("상태 패널", "Status panel", "ステータスパネル"))
                    .font(.title2.weight(.bold))

                if let snapshot = viewModel.statusSnapshot {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(t("인덱싱 문서 수", "Indexed docs", "インデックス文書数") + ": \(snapshot.indexed_docs)")
                        Text(t("마지막 인덱싱", "Last indexing", "最終インデックス") + ": \(snapshot.last_indexed_at ?? "-")")
                        Text(t("현재 프라이버시 모드", "Current privacy mode", "現在のプライバシーモード") + ": \(snapshot.privacy_mode.title(language: language))")
                        Text(t("최근 외부 호출", "Latest external call", "最近の外部呼び出し") + ": \(snapshot.latest_external_call?.provider ?? t("없음", "None", "なし"))")
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .plosGlassPanel()
                } else {
                    Text(t("상태 정보를 불러오는 중입니다.", "Loading status...", "ステータスを読み込み中です。"))
                        .foregroundStyle(.secondary)
                }

                VStack(alignment: .leading, spacing: 8) {
                    Text(t("인덱싱 대상 폴더", "Indexed folders", "インデックス対象フォルダ"))
                        .font(.headline)
                    if viewModel.includedFolderURLs.isEmpty {
                        Text(t("선택된 폴더가 없습니다.", "No selected folders.", "選択されたフォルダはありません。"))
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(viewModel.includedFolderURLs, id: \.path) { url in
                            Text(url.path)
                                .lineLimit(1)
                                .font(.subheadline)
                                .foregroundStyle(.secondary)
                        }
                    }
                }
                .padding(12)
                .plosGlassPanel()

                VStack(alignment: .leading, spacing: 8) {
                    Text(t("실패 파일 목록", "Failed files", "失敗ファイル一覧"))
                        .font(.headline)
                    if viewModel.failureItems.isEmpty {
                        Text(t("실패한 파일이 없습니다.", "No failed files.", "失敗したファイルはありません。"))
                            .foregroundStyle(.secondary)
                    } else {
                        ForEach(viewModel.failureItems) { item in
                            VStack(alignment: .leading, spacing: 3) {
                                Text(item.path)
                                    .lineLimit(1)
                                Text(item.reason)
                                    .font(.caption)
                                    .foregroundStyle(.secondary)
                            }
                            .padding(8)
                            .background(Color.red.opacity(0.12), in: Rectangle())
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
                try await viewModel.refreshRemoteState()
            } catch {
                if !(error is CancellationError) {
                    viewModel.lastError = error.localizedDescription
                }
            }
        }
    }
}
