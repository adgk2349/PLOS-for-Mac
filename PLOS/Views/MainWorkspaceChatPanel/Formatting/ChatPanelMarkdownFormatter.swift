import Foundation

enum ChatPanelMarkdownFormatter {
    private static var segmentCache: [String: [MarkdownRenderSegment]] = [:]
    private static var segmentCacheOrder: [String] = []
    private static let segmentCacheLock = NSLock()
    private static let segmentCacheLimit = 96

    static func nfc(_ value: String) -> String {
        value.precomposedStringWithCanonicalMapping
    }

    static func markdownSegments(from raw: String, keyPrefix: String) -> [MarkdownRenderSegment] {
        let cacheKey = "\(keyPrefix)|\(cacheFingerprint(raw))"
        if let cached = cachedSegments(forKey: cacheKey) {
            return cached
        }
        let normalized = normalizeMarkdownForRender(raw)
        guard normalized.contains("```") else {
            let output = [MarkdownRenderSegment(id: "\(keyPrefix)-text-0", kind: .text, content: normalized, language: nil)]
            storeSegments(output, forKey: cacheKey)
            return output
        }
        guard let regex = try? NSRegularExpression(pattern: "(`{3,}|~{3,})([A-Za-z0-9_+.-]*)[ \\t]*\\n?([\\s\\S]*?)\\1") else {
            let output = [MarkdownRenderSegment(id: "\(keyPrefix)-text-fallback", kind: .text, content: normalized, language: nil)]
            storeSegments(output, forKey: cacheKey)
            return output
        }
        let ns = normalized as NSString
        let fullRange = NSRange(location: 0, length: ns.length)
        let matches = regex.matches(in: normalized, options: [], range: fullRange)
        if matches.isEmpty {
            let output = [MarkdownRenderSegment(id: "\(keyPrefix)-text-0", kind: .text, content: normalized, language: nil)]
            storeSegments(output, forKey: cacheKey)
            return output
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
                let langRange = match.range(at: 2)
                guard langRange.location != NSNotFound, langRange.length > 0 else { return nil }
                let value = ns.substring(with: langRange).trimmingCharacters(in: .whitespacesAndNewlines)
                return value.isEmpty ? nil : value
            }()
            let codeRange = match.range(at: 3)
            let code = codeRange.location == NSNotFound ? "" : ns.substring(with: codeRange)
            let trimmedCode = code.trimmingCharacters(in: .newlines)
            if trimmedCode.isEmpty {
                cursor = matchRange.location + matchRange.length
                continue
            }
            output.append(
                MarkdownRenderSegment(
                    id: "\(keyPrefix)-code-\(index)",
                    kind: .code,
                    content: trimmedCode,
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

        storeSegments(output, forKey: cacheKey)
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
            of: #"(?is)<(thinking|thought|reasoning|analysis|cot|searching|tool_running|action|tool_result)\b[^>]*>.*?</\1>"#,
            with: "",
            options: .regularExpression
        )
        value = value.replacingOccurrences(
            of: #"(?is)<tool_code\b[^>]*>\s*([\s\S]*?)\s*</tool_code>"#,
            with: "\n```\n$1\n```\n",
            options: .regularExpression
        )
        value = value.replacingOccurrences(
            of: #"(?is)<(final_answer|assistant_response)\b[^>]*>(.*?)</\1>"#,
            with: "$2",
            options: .regularExpression
        )
        value = value.replacingOccurrences(
            of: #"(?is)</?(thinking|thought|reasoning|analysis|cot|searching|tool_running|tool_code|tool_result|action|final_answer|assistant_response|tool_[a-z0-9_]+)\b[^>]*>"#,
            with: "",
            options: .regularExpression
        )
        value = value.replacingOccurrences(
            of: #"(?is)</?(channel|analysis|final|assistant|system|user|message|observation|plan|reflection)\b[^>]*>"#,
            with: "",
            options: .regularExpression
        )
        value = value.replacingOccurrences(
            of: #"(?im)^\s*<\s*/?\s*(?:channel|analysis|final|assistant|system|user|message|observation|plan|reflection)\s*>\s*[.:：-]?\s*$"#,
            with: "",
            options: .regularExpression
        )
        value = value.replacingOccurrences(
            of: #"(?im)^\s*[.:：-]+\s*$"#,
            with: "",
            options: .regularExpression
        )

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
        // Normalize uncommon fence styles so downstream parsing consistently recognizes code blocks.
        value = value.replacingOccurrences(
            of: #"(?m)^([ \t]{0,3})~{3,}[ \t]*([A-Za-z0-9_+.-]+)?[ \t]*$"#,
            with: "$1```$2",
            options: .regularExpression
        )
        value = value.replacingOccurrences(
            of: #"(?m)^([ \t]{0,3})`{4,}[ \t]*([A-Za-z0-9_+.-]+)?[ \t]*$"#,
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
        // Drop empty fenced code blocks that can appear when internal wrappers are sanitized.
        value = value.replacingOccurrences(
            of: #"(?ms)^[ \t]*```[A-Za-z0-9_+.-]*[ \t]*\n[ \t]*```[ \t]*\n?"#,
            with: "",
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
              let regex = try? NSRegularExpression(pattern: "(`{3,}|~{3,})[^\\n`~]*\\n[\\s\\S]*?\\1")
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

    private static func cachedSegments(forKey key: String) -> [MarkdownRenderSegment]? {
        segmentCacheLock.lock()
        defer { segmentCacheLock.unlock() }
        return segmentCache[key]
    }

    private static func cacheFingerprint(_ raw: String) -> String {
        var hasher = Hasher()
        hasher.combine(raw)
        hasher.combine(raw.count)
        return String(hasher.finalize(), radix: 16)
    }

    private static func storeSegments(_ segments: [MarkdownRenderSegment], forKey key: String) {
        segmentCacheLock.lock()
        defer { segmentCacheLock.unlock() }
        segmentCache[key] = segments
        if let existingIndex = segmentCacheOrder.firstIndex(of: key) {
            segmentCacheOrder.remove(at: existingIndex)
        }
        segmentCacheOrder.append(key)
        if segmentCacheOrder.count > segmentCacheLimit {
            let removeCount = segmentCacheOrder.count - segmentCacheLimit
            for staleKey in segmentCacheOrder.prefix(removeCount) {
                segmentCache.removeValue(forKey: staleKey)
            }
            segmentCacheOrder.removeFirst(removeCount)
        }
    }
}
