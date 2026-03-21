import Foundation
import SwiftUI

struct MainWorkspaceView: View {
    private enum SidebarFolder: String, CaseIterable, Identifiable {
        case chats
        case archive

        var id: String { rawValue }

        var title: String {
            switch self {
            case .chats:
                return "대화"
            case .archive:
                return "보관함"
            }
        }

        var icon: String {
            switch self {
            case .chats:
                return "bubble.left.and.bubble.right"
            case .archive:
                return "archivebox"
            }
        }
    }

    @ObservedObject var viewModel: AppViewModel
    @State private var sidebarSearch = ""
    @State private var showSettingsPanel = false
    @State private var showStatusPanel = false
    @State private var isSidebarCollapsed = false
    @State private var selectedSidebarFolder: SidebarFolder = .chats
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        GeometryReader { proxy in
            let sidebarWidth = min(300, max(250, proxy.size.width * 0.24))
            ZStack {
                HStack(spacing: 0) {
                    if !isSidebarCollapsed {
                        sidebar
                            .frame(width: sidebarWidth)
                    }

                    ChatPanelView(viewModel: viewModel, isSidebarCollapsed: $isSidebarCollapsed)
                        .frame(maxWidth: .infinity, maxHeight: .infinity)
                }

                if showSettingsPanel {
                    Color.black.opacity(0.34)
                        .ignoresSafeArea()
                        .onTapGesture {
                            showSettingsPanel = false
                        }

                    SettingsPanelView(
                        viewModel: viewModel,
                        onOpenStatusPanel: {
                            showSettingsPanel = false
                            showStatusPanel = true
                        }
                    )
                        .padding(18)
                        .frame(
                            width: min(920, proxy.size.width * 0.86),
                            height: min(780, proxy.size.height * 0.9)
                        )
                        .plosGlassPanel()
                        .onTapGesture {
                            // consume tap inside settings
                        }
                        .zIndex(3)
                }
            }
        }
        .animation(.easeInOut(duration: 0.18), value: showSettingsPanel)
        .sheet(isPresented: $showStatusPanel) {
            StatusPanelView(viewModel: viewModel)
                .frame(minWidth: 760, minHeight: 560)
                .padding(16)
        }
        .onChange(of: selectedSidebarFolder) { _, newValue in
            if newValue == .chats {
                viewModel.selectFirstInboxRoomIfNeeded()
            }
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                Spacer(minLength: 0)
                iconButton(symbol: "square.and.pencil", help: "새 채팅") {
                    selectedSidebarFolder = .chats
                    viewModel.createChatRoom()
                }
                iconButton(symbol: "gearshape", help: "설정") {
                    showSettingsPanel = true
                }
                iconButton(symbol: "sidebar.left", help: "사이드바 닫기") {
                    withAnimation(.easeInOut(duration: 0.18)) {
                        isSidebarCollapsed = true
                    }
                }
            }

            TextField("대화/질문 검색", text: $sidebarSearch)
                .textFieldStyle(.plain)
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .plosGlassCapsule(tint: Color.white.opacity(0.02))

            HStack(spacing: 8) {
                folderButton(.chats, count: viewModel.inboxChatRooms.count)
                folderButton(.archive, count: viewModel.archivedChatRooms.count)
            }

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 6) {
                    if filteredChatRooms.isEmpty {
                        Text(selectedSidebarFolder == .archive ? "보관된 채팅이 없습니다" : "대화를 시작해보세요")
                            .font(.caption)
                            .foregroundStyle(.secondary)
                            .padding(.horizontal, 10)
                            .padding(.vertical, 8)
                    } else {
                        ForEach(filteredChatRooms) { room in
                            Button {
                                viewModel.selectChatRoom(room.id)
                            } label: {
                                VStack(alignment: .leading, spacing: 3) {
                                    Text(room.title)
                                        .font(.subheadline.weight(.semibold))
                                        .lineLimit(1)
                                    Text(roomPreview(room))
                                        .font(.caption2)
                                        .foregroundStyle(.secondary)
                                        .lineLimit(1)
                                }
                                .frame(maxWidth: .infinity, alignment: .leading)
                                .padding(.horizontal, 10)
                                .padding(.vertical, 8)
                                .background(room.id == viewModel.selectedChatRoomID ? Color.white.opacity(0.08) : Color.clear)
                                .plosGlassInputFrame(radius: 12)
                            }
                            .buttonStyle(.plain)
                            .contextMenu {
                                if room.isArchived {
                                    Button {
                                        selectedSidebarFolder = .chats
                                        viewModel.unarchiveChatRoom(room.id, selectAfterRestore: true)
                                    } label: {
                                        Label("복원", systemImage: "arrow.uturn.backward")
                                    }
                                } else {
                                    Button {
                                        viewModel.archiveChatRoom(room.id)
                                    } label: {
                                        Label("아카이브", systemImage: "archivebox")
                                    }
                                }

                                Divider()

                                Button(role: .destructive) {
                                    viewModel.deleteChatRoom(room.id)
                                } label: {
                                    Label("삭제", systemImage: "trash")
                                }
                            }
                        }
                    }
                }
            }

            Spacer(minLength: 6)
        }
        .padding(12)
        .frame(maxHeight: .infinity, alignment: .top)
        .background(
            sidebarBackground
                .ignoresSafeArea(edges: .top)
        )
    }

    private var sidebarBackground: some View {
        return Rectangle()
            .fill(.ultraThinMaterial)
            .overlay(PLOSGlassTheme.chromeTint(for: colorScheme))
    }

    private func iconButton(symbol: String, help: String, action: @escaping () -> Void) -> some View {
        Button(action: action) {
            ZStack {
                Circle().fill(Color.clear)
                Image(systemName: symbol)
                    .font(.system(size: 13, weight: .semibold))
            }
            .frame(width: 40, height: 40)
            .plosGlassCircle()
        }
        .buttonStyle(.plain)
        .contentShape(Circle())
        .help(help)
    }

    private var filteredChatRooms: [ChatRoom] {
        let rooms: [ChatRoom] = {
            switch selectedSidebarFolder {
            case .chats:
                return viewModel.inboxChatRooms
            case .archive:
                return viewModel.archivedChatRooms
            }
        }()
        let needle = sidebarSearch.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        guard !needle.isEmpty else { return rooms }
        return rooms.filter {
            $0.title.precomposedStringWithCanonicalMapping.lowercased().contains(needle) ||
            roomPreview($0).precomposedStringWithCanonicalMapping.lowercased().contains(needle)
        }
    }

    private func folderButton(_ folder: SidebarFolder, count: Int) -> some View {
        Button {
            selectedSidebarFolder = folder
            if folder == .chats {
                viewModel.selectFirstInboxRoomIfNeeded()
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: folder.icon)
                    .font(.caption.weight(.semibold))
                Text(folder.title)
                    .font(.caption.weight(.semibold))
                Text("\(count)")
                    .font(.caption2.weight(.semibold))
                    .foregroundStyle(.secondary)
            }
            .frame(maxWidth: .infinity)
            .padding(.vertical, 8)
            .padding(.horizontal, 10)
            .background(selectedSidebarFolder == folder ? Color.white.opacity(0.08) : Color.clear)
            .plosGlassInputFrame(radius: 10)
        }
        .buttonStyle(.plain)
    }

    private func roomPreview(_ room: ChatRoom) -> String {
        guard let last = room.messages.last else {
            return "대화를 시작해보세요"
        }
        switch last.source {
        case .user:
            return last.text?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
                ? (last.text ?? "").precomposedStringWithCanonicalMapping
                : "사용자 입력"
        case .local:
            return last.resultSummary?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
                ? (last.resultSummary ?? "").precomposedStringWithCanonicalMapping
                : (last.lead ?? "로컬 응답").precomposedStringWithCanonicalMapping
        case .external:
            return last.text?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
                ? (last.text ?? "").precomposedStringWithCanonicalMapping
                : "외부 분석 응답"
        }
    }
}

