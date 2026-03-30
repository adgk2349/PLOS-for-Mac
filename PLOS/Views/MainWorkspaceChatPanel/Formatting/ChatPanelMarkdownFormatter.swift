import Foundation

enum ChatPanelMarkdownFormatter {
    static func nfc(_ value: String) -> String {
        value.precomposedStringWithCanonicalMapping
    }

    static func markdownSegments(from raw: String, keyPrefix: String) -> [MarkdownRenderSegment] {
        let normalized = normalizeMarkdownForRender(raw)
        guard normalized.contains("```") else {
            return [MarkdownRenderSegment(id: "\(keyPrefix)-text-0", kind: .text, content: normalized, language: nil)]
        }
        guard let regex = try? NSRegularExpression(pattern: "```([A-Za-z0-9_+.-]*)[ \\t]*\\n?([\\s\\S]*?)```") else {
            return [MarkdownRenderSegment(id: "\(keyPrefix)-text-fallback", kind: .text, content: normalized, language: nil)]
        }
        let ns = normalized as NSString
        let fullRange = NSRange(location: 0, length: ns.length)
        let matches = regex.matches(in: normalized, options: [], range: fullRange)
        if matches.isEmpty {
            return [MarkdownRenderSegment(id: "\(keyPrefix)-text-0", kind: .text, content: normalized, language: nil)]
        }

        var cursor = 0
        var output: [MarkdownRenderSegment] = []
        var index = 0
        for match in matches {
            let matchRange = match.range
            if matchRange.location > cursor {
                let textRange = NSRange(location: cursor, length: matchRange.location - cursor)
                let text = ns.substring(with: textRange)
                output.append(
                    MarkdownRenderSegment(
                        id: "\(keyPrefix)-text-\(index)",
                        kind: .text,
                        content: text,
                        language: nil
                    )
                )
                index += 1
            }

            let language: String? = {
                let langRange = match.range(at: 1)
                guard langRange.location != NSNotFound, langRange.length > 0 else { return nil }
                let value = ns.substring(with: langRange).trimmingCharacters(in: .whitespacesAndNewlines)
                return value.isEmpty ? nil : value
            }()
            let codeRange = match.range(at: 2)
            let code = codeRange.location == NSNotFound ? "" : ns.substring(with: codeRange)
            output.append(
                MarkdownRenderSegment(
                    id: "\(keyPrefix)-code-\(index)",
                    kind: .code,
                    content: code.trimmingCharacters(in: .newlines),
                    language: language
                )
            )
            index += 1
            cursor = matchRange.location + matchRange.length
        }

        if cursor < ns.length {
            let tailRange = NSRange(location: cursor, length: ns.length - cursor)
            let tail = ns.substring(with: tailRange)
            output.append(
                MarkdownRenderSegment(
                    id: "\(keyPrefix)-text-\(index)",
                    kind: .text,
                    content: tail,
                    language: nil
                )
            )
        }

        return output
    }

    static func normalizeMarkdownForRender(_ raw: String) -> String {
        var value = nfc(raw)
        if value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
            return value
        }

        value = value.replacingOccurrences(of: "\r\n", with: "\n")
        value = value.replacingOccurrences(of: "\r", with: "\n")

        value = value.replacingOccurrences(
            of: #"(?m)^([ \t]{0,3})'''([A-Za-z0-9_+.-]+)[ \t]+([^\n'][^\n]*)$"#,
            with: "$1```$2\n$3",
            options: .regularExpression
        )
        value = value.replacingOccurrences(
            of: #"(?m)^([ \t]{0,3})'''[ \t]+([^\n'][^\n]*)$"#,
            with: "$1```\n$2",
            options: .regularExpression
        )
        value = value.replacingOccurrences(
            of: #"(?m)^([ \t]{0,3})'''[ \t]*([A-Za-z0-9_+.-]+)?[ \t]*$"#,
            with: "$1```$2",
            options: .regularExpression
        )

        value = value.replacingOccurrences(
            of: #"```([A-Za-z0-9_+.-]+)[ \t]+([^\n`][^\n]*)"#,
            with: "```$1\n$2",
            options: .regularExpression
        )
        value = value.replacingOccurrences(
            of: #"```[ \t]+([^\n`][^\n]*)"#,
            with: "```\n$1",
            options: .regularExpression
        )
        if value.components(separatedBy: "```").count % 2 == 0 {
            value += "\n```"
        }

        let stashed = stashFencedCodeBlocks(in: value)
        value = stashed.text

        let replacements: [(String, String)] = [
            (#"([^\n])\s+(#{1,6}\s)"#, "$1\n\n$2"),
            (#"([.!?。！？])\s+((?:\d{1,2}[.)]|[-*•])\s+)"#, "$1\n$2"),
            (#"\s+((?:\d{1,2}[.)])(?=[^\d\s]))"#, "\n$1"),
            (#"(?m)^(\d{1,2}[.)])(?=\S)"#, "$1 "),
            (#"\n{3,}"#, "\n\n"),
        ]
        for (pattern, template) in replacements {
            value = value.replacingOccurrences(
                of: pattern,
                with: template,
                options: .regularExpression
            )
        }

        return restoreFencedCodeBlocks(in: value, codeBlocks: stashed.blocks)
    }

    private static func stashFencedCodeBlocks(in text: String) -> (text: String, blocks: [String: String]) {
        guard text.contains("```"),
              let regex = try? NSRegularExpression(pattern: "```[^\\n`]*\\n[\\s\\S]*?```")
        else {
            return (text, [:])
        }

        let ns = text as NSString
        let matches = regex.matches(in: text, range: NSRange(location: 0, length: ns.length))
        if matches.isEmpty {
            return (text, [:])
        }

        let mutable = NSMutableString(string: text)
        var blocks: [String: String] = [:]
        for (index, match) in matches.enumerated().reversed() {
            let token = "__CODE_BLOCK_\(index)__"
            let code = ns.substring(with: match.range)
            blocks[token] = code
            mutable.replaceCharacters(in: match.range, with: token)
        }
        return (mutable as String, blocks)
    }

    private static func restoreFencedCodeBlocks(in text: String, codeBlocks: [String: String]) -> String {
        guard !codeBlocks.isEmpty else { return text }
        var value = text
        for (token, block) in codeBlocks {
            value = value.replacingOccurrences(of: token, with: block)
        }
        return value
    }
}
