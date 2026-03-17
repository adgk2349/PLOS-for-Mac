import AppKit
import SwiftUI

final class PLOSApplicationDelegate: NSObject, NSApplicationDelegate {
    var onWillTerminate: (() -> Void)?

    func applicationWillTerminate(_ notification: Notification) {
        onWillTerminate?()
    }
}

@main
struct PLOSApp: App {
    @NSApplicationDelegateAdaptor(PLOSApplicationDelegate.self) private var appDelegate
    @StateObject private var viewModel = AppViewModel()

    var body: some Scene {
        WindowGroup {
            ContentView(viewModel: viewModel)
                .task {
                    await viewModel.bootstrap()
                }
                .onAppear {
                    appDelegate.onWillTerminate = {
                        viewModel.shutdown()
                    }
                }
                .frame(minWidth: 1100, minHeight: 760)
        }
    }
}
