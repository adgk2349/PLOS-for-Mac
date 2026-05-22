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
    @State private var isPluginAccordionExpanded = true
    @Environment(\.colorScheme) private var colorScheme

    var body: some View {
        GeometryReader { proxy in
            let sidebarWidth = min(300, max(250, proxy.size.width * 0.24))
            ZStack(alignment: .topLeading) {
                HStack(spacing: 0) {
                    if !isSidebarCollapsed {
                        sidebar
                            .frame(width: sidebarWidth)
                    }

                    Group {
                        if viewModel.selectedMainPanel == .plugin {
                            PluginPanelHostView(viewModel: viewModel)
                        } else {
                            ChatPanelView(viewModel: viewModel, isSidebarCollapsed: $isSidebarCollapsed)
                        }
                    }
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

                if showStatusPanel {
                    Color.black.opacity(0.34)
                        .ignoresSafeArea()
                        .onTapGesture {
                            showStatusPanel = false
                        }

                    StatusPanelView(viewModel: viewModel)
                        .padding(18)
                        .frame(
                            width: min(920, proxy.size.width * 0.86),
                            height: min(780, proxy.size.height * 0.9)
                        )
                        .plosGlassPanel()
                        .onTapGesture {
                            // consume tap inside status panel
                        }
                        .zIndex(4)
                }
            }
        }
        .animation(.none, value: colorScheme)
        .animation(.easeInOut(duration: 0.18), value: showSettingsPanel)
        .animation(.easeInOut(duration: 0.18), value: showStatusPanel)
        .onChange(of: selectedSidebarFolder) { _, newValue in
            if newValue == .chats {
                viewModel.selectFirstInboxRoomIfNeeded()
                viewModel.switchToChatPanel()
            }
        }
    }

    private var sidebar: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack(spacing: 8) {
                iconButton(symbol: "gearshape", helpText: L10n.tr("settings.title", language: viewModel.appLanguage, fallbackKo: "설정", fallbackEn: "Settings", fallbackJa: "設定")) {
                    showSettingsPanel = true
                }
                Spacer(minLength: 0)
                iconButton(symbol: "square.and.pencil", helpText: L10n.tr("workspace.sidebar.new_chat", language: viewModel.appLanguage, fallbackKo: "새 채팅", fallbackEn: "New chat", fallbackJa: "新しいチャット")) {
                    selectedSidebarFolder = .chats
                    viewModel.createChatRoom()
                }
                iconButton(symbol: "sidebar.left", helpText: L10n.tr("workspace.sidebar.collapse_sidebar", language: viewModel.appLanguage, fallbackKo: "사이드바 닫기", fallbackEn: "Collapse sidebar", fallbackJa: "サイドバーを閉じる")) {
                    withAnimation(.easeInOut(duration: 0.18)) {
                        isSidebarCollapsed = true
                    }
                }
            }

            TextField(L10n.tr("workspace.sidebar.search_placeholder", language: viewModel.appLanguage, fallbackKo: "대화/질문 검색", fallbackEn: "Search chats/questions", fallbackJa: "会話/質問を検索"), text: $sidebarSearch)
                .textFieldStyle(.plain)
                .padding(.horizontal, 12)
                .padding(.vertical, 10)
                .plosGlassCapsule(tint: Color.white.opacity(0.02))

            pluginAccordion

            HStack(spacing: 8) {
                folderButton(.chats, count: viewModel.inboxChatRooms.count)
                folderButton(.archive, count: viewModel.archivedChatRooms.count)
            }

            ScrollView {
                LazyVStack(alignment: .leading, spacing: 6) {
                    if filteredChatRooms.isEmpty {
                        Text(
                            selectedSidebarFolder == .archive
                                ? L10n.tr("workspace.sidebar.no_archived_chats", language: viewModel.appLanguage, fallbackKo: "보관된 채팅이 없습니다", fallbackEn: "No archived chats", fallbackJa: "アーカイブされたチャットはありません")
                                : L10n.tr("workspace.sidebar.start_conversation", language: viewModel.appLanguage, fallbackKo: "대화를 시작해보세요", fallbackEn: "Start a conversation", fallbackJa: "会話を始めましょう")
                        )
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
                                    HStack(spacing: 6) {
                                        Text(room.title)
                                            .font(.subheadline.weight(.semibold))
                                            .lineLimit(1)
                                        if viewModel.roomUsesWorkspaceOverride(room.id) {
                                            Image(systemName: "folder.badge.gearshape")
                                                .font(.caption2)
                                                .foregroundStyle(.secondary)
                                                .help(L10n.tr("workspace.sidebar.room_workspace_override_help", language: viewModel.appLanguage, fallbackKo: "이 채팅방 전용 워크스페이스 경로 사용 중", fallbackEn: "This room uses workspace override", fallbackJa: "このチャットは専用ワークスペースを使用中"))
                                            if let roomState = viewModel.roomIndexStateByRoomID[room.id], !roomState.isEmpty {
                                                Text(roomIndexStateLabel(for: room.id, state: roomState))
                                                    .font(.caption2)
                                                    .foregroundStyle(.secondary)
                                                    .padding(.horizontal, 6)
                                                    .padding(.vertical, 2)
                                                    .background(Color.white.opacity(0.05))
                                                    .clipShape(Capsule())
                                            }
                                        }
                                    }
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
                                if !room.isArchived {
                                    Button {
                                        viewModel.addIncludedFolderToRoom(room.id)
                                    } label: {
                                        Label(L10n.tr("workspace.sidebar.context.add_room_folder", language: viewModel.appLanguage, fallbackKo: "이 방 폴더 추가", fallbackEn: "Add room folder", fallbackJa: "この部屋にフォルダ追加"), systemImage: "folder.badge.plus")
                                    }

                                    Button {
                                        viewModel.addExcludedPathToRoom(room.id)
                                    } label: {
                                        Label(L10n.tr("workspace.sidebar.context.add_excluded_path", language: viewModel.appLanguage, fallbackKo: "이 방 제외 경로 추가", fallbackEn: "Add excluded path", fallbackJa: "除外パスを追加"), systemImage: "minus.circle")
                                    }

                                    if let excludedPaths = room.excludedPaths, !excludedPaths.isEmpty {
                                        Menu {
                                            ForEach(excludedPaths, id: \.self) { path in
                                                Button {
                                                    viewModel.removeExcludedPathFromRoom(room.id, path: path)
                                                } label: {
                                                    Label(path, systemImage: "minus.circle")
                                                }
                                            }
                                        } label: {
                                            Label(L10n.tr("workspace.sidebar.context.remove_excluded_path", language: viewModel.appLanguage, fallbackKo: "이 방 제외 경로 제거", fallbackEn: "Remove excluded path", fallbackJa: "除外パスを削除"), systemImage: "minus.circle.dotted")
                                        }

                                        Button {
                                            viewModel.clearExcludedPathsForRoom(room.id)
                                        } label: {
                                            Label(L10n.tr("workspace.sidebar.context.reset_excluded_paths", language: viewModel.appLanguage, fallbackKo: "이 방 제외 경로 초기화", fallbackEn: "Reset excluded paths", fallbackJa: "除外パスを初期化"), systemImage: "arrow.uturn.backward.circle")
                                        }
                                    }

                                    if viewModel.roomUsesWorkspaceOverride(room.id) {
                                        Button {
                                            viewModel.clearRoomWorkspaceOverride(room.id)
                                        } label: {
                                            Label(L10n.tr("workspace.sidebar.context.reset_room_workspace", language: viewModel.appLanguage, fallbackKo: "이 방 폴더 설정 초기화", fallbackEn: "Reset room workspace", fallbackJa: "部屋ワークスペースを初期化"), systemImage: "arrow.uturn.backward.circle")
                                        }
                                    }

                                    Divider()
                                }

                                if room.isArchived {
                                    Button {
                                        selectedSidebarFolder = .chats
                                        viewModel.unarchiveChatRoom(room.id, selectAfterRestore: true)
                                    } label: {
                                        Label(L10n.tr("workspace.sidebar.context.restore", language: viewModel.appLanguage, fallbackKo: "복원", fallbackEn: "Restore", fallbackJa: "復元"), systemImage: "arrow.uturn.backward")
                                    }
                                } else {
                                    Button {
                                        viewModel.archiveChatRoom(room.id)
                                    } label: {
                                        Label(L10n.tr("workspace.sidebar.context.archive", language: viewModel.appLanguage, fallbackKo: "아카이브", fallbackEn: "Archive", fallbackJa: "アーカイブ"), systemImage: "archivebox")
                                    }
                                }

                                Divider()

                                Button(role: .destructive) {
                                    viewModel.deleteChatRoom(room.id)
                                } label: {
                                    Label(L10n.tr("workspace.sidebar.context.delete", language: viewModel.appLanguage, fallbackKo: "삭제", fallbackEn: "Delete", fallbackJa: "削除"), systemImage: "trash")
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

    private func iconButton(symbol: String, helpText: String, action: @escaping () -> Void) -> some View {
        let topBarCircleTint: Color = colorScheme == .dark
            ? Color.white.opacity(0.08)
            : Color.black.opacity(0.035)
        return Button(action: action) {
            ZStack {
                Circle().fill(Color.clear)
                Image(systemName: symbol)
                    .font(.system(size: 13, weight: .semibold))
            }
            .frame(width: 40, height: 40)
            .plosGlassCircle(tint: topBarCircleTint)
        }
        .buttonStyle(.plain)
        .contentShape(Circle())
        .help(Text(helpText))
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
        let title: String = {
            switch folder {
            case .chats:
                return L10n.tr("workspace.sidebar.folder.chats", language: viewModel.appLanguage, fallbackKo: "대화", fallbackEn: "Chats", fallbackJa: "チャット")
            case .archive:
                return L10n.tr("workspace.sidebar.folder.archive", language: viewModel.appLanguage, fallbackKo: "보관함", fallbackEn: "Archive", fallbackJa: "アーカイブ")
            }
        }()
        return Button {
            selectedSidebarFolder = folder
            if folder == .chats {
                viewModel.selectFirstInboxRoomIfNeeded()
                viewModel.switchToChatPanel()
            }
        } label: {
            HStack(spacing: 6) {
                Image(systemName: folder.icon)
                    .font(.caption.weight(.semibold))
                Text(title)
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

    @ViewBuilder
    private var pluginAccordion: some View {
        VStack(alignment: .leading, spacing: 8) {
            Button {
                withAnimation(.easeInOut(duration: 0.16)) {
                    isPluginAccordionExpanded.toggle()
                }
            } label: {
                HStack(spacing: 8) {
                    Image(systemName: "puzzlepiece.extension")
                        .font(.caption.weight(.semibold))
                    Text(L10n.tr("workspace.sidebar.folder.plugins", language: viewModel.appLanguage, fallbackKo: "플러그인", fallbackEn: "Plugins", fallbackJa: "プラグイン"))
                        .font(.caption.weight(.semibold))
                    Text("\(viewModel.sidebarPluginPanels.count)")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                    Spacer(minLength: 0)
                    Image(systemName: isPluginAccordionExpanded ? "chevron.up" : "chevron.down")
                        .font(.caption2.weight(.semibold))
                        .foregroundStyle(.secondary)
                }
                .frame(maxWidth: .infinity, alignment: .leading)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
                .plosGlassInputFrame(radius: 10)
            }
            .buttonStyle(.plain)

            if isPluginAccordionExpanded {
                ScrollView {
                    VStack(alignment: .leading, spacing: 6) {
                        pluginRows
                    }
                }
                .frame(maxHeight: 188)
            }
        }
        .padding(.bottom, 4)
    }

    private func roomPreview(_ room: ChatRoom) -> String {
        guard let last = room.messages.last else {
            return L10n.tr("workspace.sidebar.start_conversation", language: viewModel.appLanguage, fallbackKo: "대화를 시작해보세요", fallbackEn: "Start a conversation", fallbackJa: "会話を始めましょう")
        }
        switch last.source {
        case .user:
            return last.text?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
                ? (last.text ?? "").precomposedStringWithCanonicalMapping
                : L10n.tr("workspace.sidebar.preview.user_message", language: viewModel.appLanguage, fallbackKo: "사용자 입력", fallbackEn: "User message", fallbackJa: "ユーザー入力")
        case .local:
            return last.resultSummary?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
                ? (last.resultSummary ?? "").precomposedStringWithCanonicalMapping
                : (last.lead ?? L10n.tr("workspace.sidebar.preview.local_response", language: viewModel.appLanguage, fallbackKo: "로컬 응답", fallbackEn: "Local response", fallbackJa: "ローカル応答")).precomposedStringWithCanonicalMapping
        case .external:
            return last.text?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty == false
                ? (last.text ?? "").precomposedStringWithCanonicalMapping
                : L10n.tr("workspace.sidebar.preview.external_response", language: viewModel.appLanguage, fallbackKo: "외부 분석 응답", fallbackEn: "External analysis response", fallbackJa: "外部分析応答")
        }
    }

    @ViewBuilder
    private var pluginRows: some View {
        let panels = filteredPluginPanels
        if panels.isEmpty {
            Text(L10n.tr("workspace.sidebar.no_plugins", language: viewModel.appLanguage, fallbackKo: "표시할 플러그인 패널이 없습니다", fallbackEn: "No plugin panels available", fallbackJa: "表示できるプラグインパネルがありません"))
                .font(.caption)
                .foregroundStyle(.secondary)
                .padding(.horizontal, 10)
                .padding(.vertical, 8)
        } else {
            ForEach(panels) { panel in
                Button {
                    viewModel.selectPluginPanel(pluginID: panel.pluginID, panelID: panel.panelID)
                } label: {
                    VStack(alignment: .leading, spacing: 4) {
                        Text(panel.title)
                            .font(.subheadline.weight(.semibold))
                            .lineLimit(1)
                        Text(panel.subtitle ?? panel.pluginID)
                            .font(.caption2)
                            .foregroundStyle(.secondary)
                            .lineLimit(1)
                    }
                    .frame(maxWidth: .infinity, alignment: .leading)
                    .padding(.horizontal, 10)
                    .padding(.vertical, 8)
                    .background(panel.id == viewModel.selectedPluginPanelCompositeID && viewModel.selectedMainPanel == .plugin ? Color.white.opacity(0.08) : Color.clear)
                    .plosGlassInputFrame(radius: 12)
                }
                .buttonStyle(.plain)
                .disabled(!panel.pluginEnabled)
            }
        }
    }

    private var filteredPluginPanels: [AppViewModel.SidebarPluginPanelItem] {
        let needle = sidebarSearch.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        let panels = viewModel.sidebarPluginPanels
        guard !needle.isEmpty else { return panels }
        return panels.filter { panel in
            panel.title.precomposedStringWithCanonicalMapping.lowercased().contains(needle) ||
            (panel.subtitle ?? "").precomposedStringWithCanonicalMapping.lowercased().contains(needle) ||
            panel.pluginID.lowercased().contains(needle)
        }
    }

    private func roomIndexStateLabel(for roomID: String, state: String) -> String {
        if state == "indexing", let progress = viewModel.roomIndexProgressByRoomID[roomID] {
            let percent = Int((min(max(progress, 0.0), 1.0) * 100.0).rounded())
            let label = L10n.tr(
                "workspace.sidebar.room_index_state.indexing",
                language: viewModel.appLanguage,
                fallbackKo: "인덱싱 중",
                fallbackEn: "Indexing",
                fallbackJa: "インデックス中"
            )
            return "\(label) \(percent)%"
        }
        switch state {
        case "indexing":
            return L10n.tr("workspace.sidebar.room_index_state.indexing", language: viewModel.appLanguage, fallbackKo: "인덱싱 중", fallbackEn: "Indexing", fallbackJa: "インデックス中")
        case "ready":
            return L10n.tr("workspace.sidebar.room_index_state.ready", language: viewModel.appLanguage, fallbackKo: "준비됨", fallbackEn: "Ready", fallbackJa: "準備完了")
        case "failed":
            return L10n.tr("workspace.sidebar.room_index_state.failed", language: viewModel.appLanguage, fallbackKo: "실패", fallbackEn: "Failed", fallbackJa: "失敗")
        default:
            return L10n.tr("workspace.sidebar.room_index_state.idle", language: viewModel.appLanguage, fallbackKo: "대기", fallbackEn: "Idle", fallbackJa: "待機")
        }
    }
}
