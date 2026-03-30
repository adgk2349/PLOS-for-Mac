import Foundation

struct ChatRoomLoadSnapshot {
    let rooms: [ChatRoom]
    let selectedRoomID: String
    let messages: [ChatMessage]
    let citations: [Citation]
}

final class ChatRoomService {
    func makeDefaultRoom() -> ChatRoom {
        ChatRoom.makeDefault()
    }

    func loadChatRooms(
        defaults: UserDefaults,
        roomsKey: String,
        activeRoomKey: String
    ) -> ChatRoomLoadSnapshot {
        let rooms: [ChatRoom]
        if
            let data = defaults.data(forKey: roomsKey),
            let decoded = try? JSONDecoder().decode([ChatRoom].self, from: data),
            !decoded.isEmpty
        {
            let migrated = decoded.map { room -> ChatRoom in
                var updated = room
                updated.title = room.title.precomposedStringWithCanonicalMapping
                let normalized = updated.title
                    .trimmingCharacters(in: .whitespacesAndNewlines)
                    .lowercased()
                if normalized == "plos chat" || normalized == "chatgpt" {
                    updated.title = "새 채팅"
                }
                return updated
            }
            rooms = migrated.sorted { $0.updatedAt > $1.updatedAt }
        } else {
            rooms = [makeDefaultRoom()]
        }

        let selectedRoomID: String
        let preferredRoomID = defaults.string(forKey: activeRoomKey)
        if
            let preferredRoomID,
            let preferred = rooms.first(where: { $0.id == preferredRoomID && !$0.isArchived })
        {
            selectedRoomID = preferred.id
        } else if let firstInbox = rooms.first(where: { !$0.isArchived }) {
            selectedRoomID = firstInbox.id
        } else if let first = rooms.first {
            selectedRoomID = first.id
        } else {
            selectedRoomID = ""
        }

        if let selected = rooms.first(where: { $0.id == selectedRoomID }) {
            return ChatRoomLoadSnapshot(
                rooms: rooms,
                selectedRoomID: selectedRoomID,
                messages: selected.messages,
                citations: selected.citations
            )
        }
        return ChatRoomLoadSnapshot(rooms: rooms, selectedRoomID: selectedRoomID, messages: [], citations: [])
    }

    func persistChatRooms(
        _ rooms: [ChatRoom],
        selectedRoomID: String,
        defaults: UserDefaults,
        roomsKey: String,
        activeRoomKey: String
    ) {
        guard let encoded = try? JSONEncoder().encode(rooms) else {
            return
        }
        defaults.set(encoded, forKey: roomsKey)
        let activeID = selectedRoomID.isEmpty ? rooms.first?.id : selectedRoomID
        if let activeID {
            defaults.set(activeID, forKey: activeRoomKey)
        }
    }

    func summarizeChatRoomTitle(from firstUserText: String) -> String {
        let normalized = firstUserText
            .precomposedStringWithCanonicalMapping
            .replacingOccurrences(of: "\r\n", with: "\n")
            .replacingOccurrences(of: "\r", with: "\n")

        let candidateLines = normalized
            .split(whereSeparator: \.isNewline)
            .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        guard !candidateLines.isEmpty else {
            return "새 채팅"
        }

        let firstLine = candidateLines.first(where: { line in
            let lowered = line.lowercased()
            return !lowered.hasPrefix("첨부 파일:") && !lowered.hasPrefix("attached file:")
        }) ?? candidateLines[0]

        var text = firstLine
        text = text.replacingOccurrences(of: #"^["'“”‘’\[\(\s]+"#, with: "", options: .regularExpression)
        text = text.replacingOccurrences(of: #"["'“”‘’\]\)\s]+$"#, with: "", options: .regularExpression)
        text = text.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        if let questionRange = text.range(of: #"[?？]"#, options: .regularExpression) {
            text = String(text[..<questionRange.lowerBound]).trimmingCharacters(in: .whitespacesAndNewlines)
        }

        text = text.replacingOccurrences(
            of: #"(?:\s*(?:좀|그냥))*\s*(?:해줘|해주세요|해줄래|알려줘|알려주세요|보여줘|찾아줘|요약해줘|정리해줘|추천해줘|가능해|가능할까|어때)\s*$"#,
            with: "",
            options: .regularExpression
        )
        text = text.replacingOccurrences(of: #"[.!?…]+$"#, with: "", options: .regularExpression)
        text = text.replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
        text = text.trimmingCharacters(in: .whitespacesAndNewlines)

        guard !text.isEmpty else {
            return "새 채팅"
        }

        let maxLength = 30
        if text.count > maxLength {
            let tentative = String(text.prefix(maxLength + 6))
            if let range = tentative.range(of: #"\s+\S*$"#, options: .regularExpression) {
                let compact = tentative[..<range.lowerBound].trimmingCharacters(in: .whitespacesAndNewlines)
                if compact.count >= 8 {
                    text = String(compact)
                } else {
                    text = String(text.prefix(maxLength))
                }
            } else {
                text = String(text.prefix(maxLength))
            }
        }

        return text.isEmpty ? "새 채팅" : text
    }

    func summarizeChatRoomTitle(from messages: [ChatMessage]) -> String {
        let userTexts = messages
            .filter { $0.source == .user }
            .compactMap { ($0.text ?? "").trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }

        if let firstMeaningful = userTexts.first(where: { !isGenericGreeting($0) }) {
            return summarizeChatRoomTitle(from: firstMeaningful)
        }
        if let firstUser = userTexts.first {
            return summarizeChatRoomTitle(from: firstUser)
        }

        let assistantTexts = messages
            .filter { $0.source == .local || $0.source == .external }
            .compactMap { ($0.text ?? "").trimmingCharacters(in: .whitespacesAndNewlines) }
            .filter { !$0.isEmpty }
        if let firstAssistant = assistantTexts.first {
            return summarizeChatRoomTitle(from: firstAssistant)
        }
        return "새 채팅"
    }

    func shouldAutoRetitle(_ title: String) -> Bool {
        let trimmed = title.trimmingCharacters(in: .whitespacesAndNewlines)
        if trimmed.isEmpty || trimmed == "새 채팅" {
            return true
        }
        return isGenericGreeting(trimmed)
    }

    private func isGenericGreeting(_ text: String) -> Bool {
        let lowered = text
            .precomposedStringWithCanonicalMapping
            .lowercased()
            .trimmingCharacters(in: .whitespacesAndNewlines)
        let compact = lowered.replacingOccurrences(of: #"[!~?.\s]+"#, with: "", options: .regularExpression)
        if compact.isEmpty {
            return true
        }
        let greetings: Set<String> = [
            "안녕",
            "안녕하세요",
            "반가워",
            "반가워요",
            "hello",
            "hi",
            "hey",
        ]
        return greetings.contains(compact)
    }
}
