import SwiftUI

struct ContentView: View {
    @ObservedObject var viewModel: AppViewModel

    var body: some View {
        ZStack {
            AnimatedGlassBackground()

            VStack(spacing: 10) {
                if viewModel.hasFinishedOnboarding {
                    MainWorkspaceView(viewModel: viewModel)
                } else {
                    headerBar
                    OnboardingView(viewModel: viewModel)
                }

                if let error = viewModel.lastError {
                    Text(error)
                        .font(.footnote)
                        .foregroundStyle(.red)
                        .frame(maxWidth: .infinity, alignment: .leading)
                        .padding(.horizontal, 14)
                        .padding(.vertical, 10)
                        .glassCard(cornerRadius: 12)
                }
            }
            .padding(12)
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
                .background(.regularMaterial)
                .clipShape(Capsule())

            routeBadge
        }
        .padding(12)
        .glassCard(cornerRadius: 14)
    }

    @ViewBuilder
    private var routeBadge: some View {
        switch viewModel.currentRoute {
        case .local:
            Image(systemName: "lock.shield.fill")
                .foregroundStyle(.green)
                .help("현재 로컬 처리 경로")
        case .external:
            Image(systemName: "cloud.fill")
                .foregroundStyle(.orange)
                .help("최근 응답은 외부 AI 경로")
        }
    }
}
