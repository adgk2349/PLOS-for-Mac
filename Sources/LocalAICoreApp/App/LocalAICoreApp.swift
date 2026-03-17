import SwiftUI

@main
struct LocalAICoreApp: App {
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
        }
        .defaultSize(width: 1200, height: 820)
    }
}
