import SwiftUI

struct StatusPanelView: View {
    @ObservedObject var viewModel: AppViewModel

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                Text("상태 패널")
                    .font(.title2.weight(.bold))

                if let snapshot = viewModel.statusSnapshot {
                    VStack(alignment: .leading, spacing: 8) {
                        Text("인덱싱 문서 수: \(snapshot.indexed_docs)")
                        Text("마지막 인덱싱: \(snapshot.last_indexed_at ?? "-")")
                        Text("현재 프라이버시 모드: \(snapshot.privacy_mode.title)")
                        Text("최근 외부 호출: \(snapshot.latest_external_call?.provider ?? "없음")")
                    }
                    .padding(12)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .glassCard(cornerRadius: 12)
                } else {
                    Text("상태 정보를 불러오는 중입니다.")
                        .foregroundStyle(.secondary)
                }

                VStack(alignment: .leading, spacing: 6) {
                    Text("인덱싱 대상 폴더")
                        .font(.headline)
                    ForEach(viewModel.includedFolderURLs, id: \.path) { url in
                        Text(url.path)
                            .font(.subheadline)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                }
                .padding(12)
                .glassCard(cornerRadius: 12)

                VStack(alignment: .leading, spacing: 8) {
                    Text("실패 파일 목록")
                        .font(.headline)
                    if viewModel.failureItems.isEmpty {
                        Text("실패한 파일이 없습니다.")
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
                            .background(Color.red.opacity(0.12))
                            .clipShape(RoundedRectangle(cornerRadius: 8))
                        }
                    }
                }
                .padding(12)
                .glassCard(cornerRadius: 12)
            }
            .padding(16)
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

