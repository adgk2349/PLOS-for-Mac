import CryptoKit
import Foundation

final class ChatFlowService {
    func normalizeQuery(_ query: String) -> String {
        query
            .precomposedStringWithCanonicalMapping
            .trimmingCharacters(in: .whitespacesAndNewlines)
    }

    func workspaceID(includedFolderURLs: [URL], excludedPaths: [String]) -> String {
        let included = includedFolderURLs
            .map(\.path)
            .map { NSString(string: $0).expandingTildeInPath }
            .sorted()
        let excluded = excludedPaths
            .map { NSString(string: $0).expandingTildeInPath }
            .sorted()
        let combined = included.map { "+:\($0)" } + excluded.map { "-:\($0)" }
        let joined = combined.joined(separator: "\n")
        let digest = Insecure.SHA1.hash(data: Data(joined.utf8))
        let hex = digest.map { String(format: "%02x", $0) }.joined()
        return String(hex.prefix(16))
    }
}
