import SwiftUI

struct ContentView: View {
    @ObservedObject var viewModel: AppViewModel

    var body: some View {
        VStack(spacing: 0) {
            headerBar

            Divider()

            if viewModel.hasFinishedOnboarding {
                MainWorkspaceView(viewModel: viewModel)
            } else {
                OnboardingView(viewModel: viewModel)
            }

            if let error = viewModel.lastError {
                Divider()
                Text(error)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 16)
                    .padding(.vertical, 8)
            }
        }
    }

    private var headerBar: some View {
        HStack {
            Label("Local AI Core", systemImage: "brain")
                .font(.headline)

            Spacer()

            Text("Privacy: \(viewModel.currentPrivacyBadge)")
                .font(.caption)
                .padding(.horizontal, 8)
                .padding(.vertical, 4)
                .background(.thinMaterial)
                .clipShape(Capsule())

            if let status = viewModel.statusSnapshot, status.latest_external_call != nil {
                Image(systemName: "network")
                    .foregroundStyle(.orange)
                    .help("최근 외부 호출 있음")
            } else {
                Image(systemName: "desktopcomputer")
                    .foregroundStyle(.green)
                    .help("로컬 처리")
            }
        }
        .padding(12)
    }
}
