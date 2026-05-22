import Foundation

enum StreamTagParser {
    static func sanitizeStreamChunk(
        _ incoming: String,
        isInsideReasoningTag: inout Bool,
        isInsideToolCodeTag: inout Bool,
        pendingTagFragment: inout String
    ) -> String {
        let normalizedIncoming = incoming.precomposedStringWithCanonicalMapping
        if normalizedIncoming.isEmpty {
            return ""
        }
        let source = pendingTagFragment + normalizedIncoming
        pendingTagFragment = ""
        if source.isEmpty {
            return ""
        }

        let reasoningTags: Set<String> = [
            "thought", "think", "thinking", "analysis", "reasoning", "cot", "searching", "tool_running",
        ]
        let controlTags: Set<String> = [
            "channel", "analysis", "final", "assistant", "assistant_response", "system", "user",
            "message", "observation", "plan", "reflection",
        ]
        var output = ""
        var cursor = source.startIndex

        while cursor < source.endIndex {
            let current = source[cursor]
            if current == "<" {
                if let closing = source[cursor...].firstIndex(of: ">") {
                    let tokenRange = source.index(after: cursor) ..< closing
                    let token = source[tokenRange]
                        .trimmingCharacters(in: .whitespacesAndNewlines)
                        .lowercased()
                    let head = token.split(whereSeparator: { $0 == " " || $0 == "\t" || $0 == "\n" || $0 == "\r" }).first
                    let tagName = String(head ?? Substring(token))
                    let isClosingTag = tagName.hasPrefix("/")
                    let bareName = isClosingTag ? String(tagName.dropFirst()) : tagName

                    if reasoningTags.contains(bareName) {
                        isInsideReasoningTag = !isClosingTag
                        cursor = source.index(after: closing)
                        continue
                    }
                    if controlTags.contains(bareName) {
                        cursor = source.index(after: closing)
                        continue
                    }
                    if bareName == "tool_code" {
                        if isClosingTag {
                            if isInsideToolCodeTag {
                                if !output.hasSuffix("\n") {
                                    output.append("\n")
                                }
                                output.append("```\n")
                            }
                            isInsideToolCodeTag = false
                        } else {
                            if !isInsideToolCodeTag {
                                if !output.hasSuffix("\n") {
                                    output.append("\n")
                                }
                                output.append("```\n")
                            }
                            isInsideToolCodeTag = true
                        }
                        cursor = source.index(after: closing)
                        continue
                    }
                } else {
                    pendingTagFragment = String(source[cursor...])
                    break
                }
            }

            if !isInsideReasoningTag {
                output.append(current)
            }
            cursor = source.index(after: cursor)
        }

        output = output.replacingOccurrences(
            of: #"(?is)</?(tool_code|tool_result|action|final_answer|assistant_response|tool_[a-z0-9_]+)\b[^>]*>"#,
            with: "",
            options: .regularExpression
        )
        output = output.replacingOccurrences(
            of: #"(?is)</?(channel|analysis|final|assistant|system|user|message|observation|plan|reflection)\b[^>]*>"#,
            with: "",
            options: .regularExpression
        )
        return output
    }

    static func splitReasoningAndAnswer(from raw: String) -> (reasoningNotes: [String], answerText: String) {
        let text = raw.precomposedStringWithCanonicalMapping
        if text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return ([], "")
        }

        let cleaned = text
            .replacingOccurrences(
                of: #"(?is)<tool_code\b[^>]*>\s*([\s\S]*?)\s*</tool_code>"#,
                with: "\n```\n$1\n```\n",
                options: .regularExpression
            )
            .replacingOccurrences(
                of: #"(?is)<(final_answer|assistant_response)\b[^>]*>(.*?)</\1>"#,
                with: "$2",
                options: .regularExpression
            )
            .replacingOccurrences(
                of: #"(?is)<(thought|think|thinking|analysis|reasoning|cot|searching|tool_running)\b[^>]*>.*?</\1>"#,
                with: "",
                options: .regularExpression
            )
            .replacingOccurrences(
                of: #"(?is)</?(thought|think|thinking|analysis|reasoning|cot|searching|tool_running)\b[^>]*>"#,
                with: "",
                options: .regularExpression
            )
            .replacingOccurrences(
                of: #"(?is)</?(tool_code|tool_result|action|final_answer|assistant_response|tool_[a-z0-9_]+)\b[^>]*>"#,
                with: "",
                options: .regularExpression
            )
            .replacingOccurrences(
                of: #"(?is)</?(channel|analysis|final|assistant|system|user|message|observation|plan|reflection)\b[^>]*>"#,
                with: "",
                options: .regularExpression
            )

        let lines = cleaned.components(separatedBy: .newlines)
        var reasoning: [String] = []
        var answerLines: [String] = []
        var answerStarted = false

        for line in lines {
            let trimmed = line.trimmingCharacters(in: .whitespacesAndNewlines)
            if trimmed.isEmpty {
                if answerStarted {
                    answerLines.append(line)
                }
                continue
            }
            if !answerStarted, isExplicitReasoningLine(trimmed), !containsAnswerAnchor(trimmed) {
                reasoning.append(trimmed)
                continue
            }
            answerStarted = true
            answerLines.append(line)
        }

        let answer = answerLines.joined(separator: "\n").trimmingCharacters(in: .whitespacesAndNewlines)
        if answer.isEmpty {
            return (dedupeReasoningNotes(reasoning), cleaned.trimmingCharacters(in: .whitespacesAndNewlines))
        }
        return (dedupeReasoningNotes(reasoning), answer)
    }

    static func dedupeReasoningNotes(_ notes: [String]) -> [String] {
        var seen = Set<String>()
        var output: [String] = []
        for item in notes {
            let normalized = item
                .precomposedStringWithCanonicalMapping
                .trimmingCharacters(in: .whitespacesAndNewlines)
                .replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
            guard !normalized.isEmpty else { continue }
            let key = normalized.lowercased()
            if seen.contains(key) {
                continue
            }
            seen.insert(key)
            output.append(normalized)
        }
        return output
    }

    private static func isExplicitReasoningLine(_ line: String) -> Bool {
        let lowered = line.lowercased()
        let explicitPrefixes = [
            "thought:",
            "thought process:",
            "thinking:",
            "thinking process:",
            "reasoning:",
            "reasoning process:",
            "analysis:",
            "chain of thought:",
            "internal monologue:",
            "[thinking]",
            "[reasoning]",
            "(thinking)",
            "(reasoning)",
            "생각:",
            "추론:",
            "분석:",
        ]
        return explicitPrefixes.contains(where: { lowered.hasPrefix($0) })
    }

    private static func containsAnswerAnchor(_ line: String) -> Bool {
        let lowered = line.lowercased()
        let anchors = [
            "물론", "좋아요", "아래", "다음", "문제",
            "sure", "certainly", "here", "let's", "answer", "solution",
            "```", "file:",
        ]
        return anchors.contains(where: { lowered.contains($0.lowercased()) })
    }
}
