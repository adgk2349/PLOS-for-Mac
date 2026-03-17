import SwiftUI

@main
struct PLOSApp: App {
    @StateObject private var viewModel = AppViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView(viewModel: viewModel)
                .task {
                    await viewModel.bootstrap()
                }
                .onDisappear {
                    viewModel.shutdown()
                }
                .frame(minWidth: 1100, minHeight: 760)
        }
    }
}
