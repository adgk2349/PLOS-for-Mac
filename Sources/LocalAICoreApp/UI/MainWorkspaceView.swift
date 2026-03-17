import SwiftUI

struct MainWorkspaceView: View {
    @ObservedObject var viewModel: AppViewModel

    var body: some View {
        TabView {
            ChatView(viewModel: viewModel)
                .tabItem {
                    Label("질의응답", systemImage: "message")
                }

            StatusPanelView(viewModel: viewModel)
                .tabItem {
                    Label("상태", systemImage: "gauge")
                }

            SettingsView(viewModel: viewModel)
                .tabItem {
                    Label("설정", systemImage: "gearshape")
                }
        }
    }
}
