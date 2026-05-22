import Foundation

struct ChatMessage: Identifiable, Codable {
    enum Source: String, Codable {
        case user
        case local
        case external
    }

    let id: UUID
    let source: Source
    var text: String?
    var intent: ChatIntent?
    var lead: String?
    var resultSummary: String?
    var structuredResult: StructuredResult?
    var responseMetadata: [String: JSONValue]?
    var parsedIntent: ParsedIntent?
    var plan: LocalPlan?
    var verification: VerificationResult?
    var reasoningBrief: String?
    var artifacts: [GeneratedArtifact]?
    var actions: [SuggestedAction]
    let timestamp: Date
    var isStreaming: Bool = false

    private static func nfc(_ value: String?) -> String? {
        value?.precomposedStringWithCanonicalMapping
    }

    init(id: UUID = UUID(), source: Source, text: String, timestamp: Date) {
        self.id = id
        self.source = source
        self.text = text.precomposedStringWithCanonicalMapping
        intent = nil
        lead = nil
        resultSummary = nil
        structuredResult = nil
        responseMetadata = nil
        parsedIntent = nil
        plan = nil
        verification = nil
        reasoningBrief = nil
        artifacts = nil
        actions = []
        self.timestamp = timestamp
    }

    init(id: UUID = UUID(), local response: LocalChatResponse, timestamp: Date) {
        self.id = id
        source = .local
        text = nil
        intent = response.intent
        lead = Self.nfc(response.lead)
        resultSummary = Self.nfc(response.result_summary)
        structuredResult = nil
        responseMetadata = nil
        parsedIntent = nil
        plan = nil
        verification = nil
        reasoningBrief = Self.nfc(response.reasoning_brief)
        artifacts = nil
        actions = response.actions
        self.timestamp = timestamp
    }

    init(id: UUID = UUID(), localV2 response: ComposedChatResponseV2, timestamp: Date) {
        self.id = id
        source = .local
        let fallbackText = (response.generated_text ?? "").precomposedStringWithCanonicalMapping
        let normalizedFallback = fallbackText.trimmingCharacters(in: .whitespacesAndNewlines)
        let normalizedLead = response.lead.precomposedStringWithCanonicalMapping
        let normalizedSummary = response.structured_result.summary.precomposedStringWithCanonicalMapping
        let isDegraded = response.used_fallback ?? false
        text = normalizedFallback.isEmpty ? nil : fallbackText
        intent = nil
        lead = normalizedLead
        // Prefer rendered generated_text for markdown/code output; avoid duplicate summary body.
        resultSummary = normalizedFallback.isEmpty ? normalizedSummary : nil
        structuredResult = response.structured_result
        var mergedMetadata = response.metadata ?? [:]
        if mergedMetadata["execution_status"] == nil {
            mergedMetadata["execution_status"] = .string(isDegraded ? "degraded" : "ok")
        }
        responseMetadata = mergedMetadata
        parsedIntent = response.parsed_intent
        plan = response.plan
        verification = response.verification
        reasoningBrief = nil
        artifacts = response.artifacts
        actions = response.actions
        if isDegraded {
            let trimmedText = text?.trimmingCharacters(in: .whitespacesAndNewlines) ?? ""
            if !trimmedText.isEmpty {
                lead = "실행 품질 저하"
                resultSummary = trimmedText
                text = nil
            }
        }
        self.timestamp = timestamp
    }
}

struct ChatRoom: Codable, Identifiable {
    var id: String
    var title: String
    var messages: [ChatMessage]
    var citations: [Citation]
    var includedFolderURLs: [URL]?
    var excludedPaths: [String]?
    var lastResolvedIncludedPaths: [String]?
    var lastResolvedExcludedPaths: [String]?
    var lastResolvedRoomScopeHash: String?
    var lastResolvedRoomStorageID: String?
    var latestQueryForDeepAnalysis: String?
    var updatedAt: Date
    var archivedAt: Date?

    var isArchived: Bool { archivedAt != nil }

    static func makeDefault() -> ChatRoom {
        ChatRoom(
            id: UUID().uuidString,
            title: "새 채팅",
            messages: [],
            citations: [],
            includedFolderURLs: nil,
            excludedPaths: nil,
            lastResolvedIncludedPaths: nil,
            lastResolvedExcludedPaths: nil,
            lastResolvedRoomScopeHash: nil,
            lastResolvedRoomStorageID: nil,
            latestQueryForDeepAnalysis: nil,
            updatedAt: Date(),
            archivedAt: nil
        )
    }
}
