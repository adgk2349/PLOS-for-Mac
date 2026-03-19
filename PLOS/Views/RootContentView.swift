import SwiftUI

struct ContentView: View {
    @ObservedObject var viewModel: AppViewModel

    var body: some View {
        ZStack {
            PLOSGlassBackground()

            if viewModel.hasFinishedOnboarding {
                MainWorkspaceView(viewModel: viewModel)
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            } else {
                OnboardingView(viewModel: viewModel)
                    .padding(14)
            }
        }
        .background(
            WindowChromeConfigurator()
                .allowsHitTesting(false)
        )
        .overlay(alignment: .bottom) {
            if let error = viewModel.lastError, !error.isEmpty {
                Text(error)
                    .font(.footnote)
                    .foregroundStyle(.red)
                    .lineLimit(2)
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 12)
                    .padding(.vertical, 8)
                    .background(Color.black.opacity(0.34), in: Rectangle())
            }
        }
    }
}
