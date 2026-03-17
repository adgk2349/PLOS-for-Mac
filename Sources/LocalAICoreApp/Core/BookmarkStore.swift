import Foundation

final class BookmarkStore {
    private let defaultsKey = "local_ai_bookmarks"
    private let defaults = UserDefaults.standard

    struct BookmarkEntry: Codable {
        let path: String
        let bookmarkData: Data
    }

    func loadPaths() -> [String] {
        guard
            let data = defaults.data(forKey: defaultsKey),
            let entries = try? JSONDecoder().decode([BookmarkEntry].self, from: data)
        else {
            return []
        }
        return entries.map(\.path)
    }

    func loadURLs() -> [URL] {
        guard
            let data = defaults.data(forKey: defaultsKey),
            let entries = try? JSONDecoder().decode([BookmarkEntry].self, from: data)
        else {
            return []
        }

        var urls: [URL] = []
        for entry in entries {
            var stale = false
            do {
                let resolved = try URL(
                    resolvingBookmarkData: entry.bookmarkData,
                    options: [.withSecurityScope],
                    relativeTo: nil,
                    bookmarkDataIsStale: &stale
                )
                _ = resolved.startAccessingSecurityScopedResource()
                urls.append(resolved)
            } catch {
                continue
            }
        }
        return urls
    }

    func save(urls: [URL]) {
        var entries: [BookmarkEntry] = []
        for url in urls {
            do {
                let data = try url.bookmarkData(
                    options: [.withSecurityScope],
                    includingResourceValuesForKeys: nil,
                    relativeTo: nil
                )
                entries.append(BookmarkEntry(path: url.path, bookmarkData: data))
            } catch {
                continue
            }
        }

        if let encoded = try? JSONEncoder().encode(entries) {
            defaults.set(encoded, forKey: defaultsKey)
        }
    }
}
