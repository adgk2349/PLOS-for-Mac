import Foundation

struct AppLanguage: RawRepresentable, Hashable, Identifiable, Codable {
    let rawValue: String

    init(rawValue: String) {
        self.rawValue = Self.normalize(rawValue)
    }

    var id: String { rawValue }

    var isAuto: Bool { rawValue == "auto" }

    static let auto = AppLanguage(rawValue: "auto")
    static let kor = AppLanguage(rawValue: "kor")
    static let eng = AppLanguage(rawValue: "eng")
    static let jpn = AppLanguage(rawValue: "jpn")

    static func normalize(_ value: String) -> String {
        let raw = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch raw {
        case "", "auto", "system", "default":
            return "auto"
        case "ko", "ko-kr", "korean", "kor":
            return "kor"
        case "en", "en-us", "english", "eng":
            return "eng"
        case "ja", "ja-jp", "japanese", "jpn":
            return "jpn"
        default:
            return raw
        }
    }
}

struct LanguageDescriptor: Hashable, Codable, Identifiable {
    let id: String
    let iso: String
    let displayName: String
    let nativeName: String
    let isDefault: Bool
}

private struct LanguageFilePayload: Decodable {
    let id: String
    let iso: String
    let displayName: String
    let nativeName: String
    let isDefault: Bool?
    let strings: [String: String]

    enum CodingKeys: String, CodingKey {
        case id
        case iso
        case displayName = "display_name"
        case nativeName = "native_name"
        case isDefault = "is_default"
        case strings
    }
}

private struct LanguageBundle {
    let descriptor: LanguageDescriptor
    let strings: [String: String]
}

private final class LanguageRegistry {
    static let shared = LanguageRegistry()

    private let queue = DispatchQueue(label: "plos.localization.registry", qos: .userInitiated)
    private var loaded = false
    private var bundles: [String: LanguageBundle] = [:]

    private init() {}

    func ensureLoaded() {
        queue.sync {
            if !loaded {
                reloadLocked()
            }
        }
    }

    func reload() {
        queue.sync {
            reloadLocked()
        }
    }

    func availableDescriptors() -> [LanguageDescriptor] {
        queue.sync {
            bundles.values.map(\.descriptor)
        }
    }

    func descriptor(for id: String) -> LanguageDescriptor? {
        queue.sync {
            bundles[id]?.descriptor
        }
    }

    func string(for key: String, languageID: String) -> String? {
        queue.sync {
            bundles[languageID]?.strings[key]
        }
    }

    func containsLanguage(id: String) -> Bool {
        queue.sync {
            bundles[id] != nil
        }
    }

    private func reloadLocked() {
        var nextBundles: [String: LanguageBundle] = [:]
        for descriptor in Self.defaultDescriptors {
            nextBundles[descriptor.id] = LanguageBundle(descriptor: descriptor, strings: [:])
        }

        let fileManager = FileManager.default
        let bundleURLs = Bundle.main.urls(forResourcesWithExtension: "json", subdirectory: "Languages") ?? []
        let candidateURLs = bundleURLs.sorted { $0.lastPathComponent < $1.lastPathComponent }

        for url in candidateURLs {
            guard fileManager.fileExists(atPath: url.path) else { continue }

            do {
                let data = try Data(contentsOf: url)
                let payload = try JSONDecoder().decode(LanguageFilePayload.self, from: data)
                let normalizedID = AppLanguage.normalize(payload.id)
                let fileID = AppLanguage.normalize(url.deletingPathExtension().lastPathComponent)
                guard normalizedID == fileID else {
                    print("[L10n] skip \(url.lastPathComponent): id(\(payload.id)) != filename(\(fileID))")
                    continue
                }
                guard !normalizedID.isEmpty, normalizedID != "auto" else {
                    print("[L10n] skip \(url.lastPathComponent): invalid id=\(payload.id)")
                    continue
                }

                let iso = Self.normalizeISO(payload.iso, languageID: normalizedID)
                let descriptor = LanguageDescriptor(
                    id: normalizedID,
                    iso: iso,
                    displayName: payload.displayName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? normalizedID.uppercased() : payload.displayName,
                    nativeName: payload.nativeName.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty ? payload.displayName : payload.nativeName,
                    isDefault: payload.isDefault ?? Self.defaultIDs.contains(normalizedID)
                )
                let compacted = payload.strings.reduce(into: [String: String]()) { partial, entry in
                    let trimmed = entry.value.trimmingCharacters(in: .whitespacesAndNewlines)
                    if !trimmed.isEmpty {
                        partial[entry.key] = entry.value
                    }
                }
                nextBundles[normalizedID] = LanguageBundle(descriptor: descriptor, strings: compacted)
            } catch {
                print("[L10n] skip \(url.lastPathComponent): \(error.localizedDescription)")
            }
        }

        for descriptor in Self.defaultDescriptors where nextBundles[descriptor.id] == nil {
            nextBundles[descriptor.id] = LanguageBundle(descriptor: descriptor, strings: [:])
        }

        bundles = nextBundles
        loaded = true
    }

    private static let defaultIDs: Set<String> = ["kor", "eng", "jpn"]

    private static let defaultDescriptors: [LanguageDescriptor] = [
        LanguageDescriptor(id: "kor", iso: "ko", displayName: "Korean", nativeName: "한국어", isDefault: true),
        LanguageDescriptor(id: "eng", iso: "en", displayName: "English", nativeName: "English", isDefault: true),
        LanguageDescriptor(id: "jpn", iso: "ja", displayName: "Japanese", nativeName: "日本語", isDefault: true),
    ]

    private static func normalizeISO(_ value: String, languageID: String) -> String {
        let raw = value.trimmingCharacters(in: .whitespacesAndNewlines).lowercased()
        switch raw {
        case "ko", "ko-kr":
            return "ko"
        case "en", "en-us":
            return "en"
        case "ja", "ja-jp":
            return "ja"
        default:
            switch languageID {
            case "kor": return "ko"
            case "eng": return "en"
            case "jpn": return "ja"
            default: return raw.isEmpty ? "en" : raw
            }
        }
    }
}

enum L10n {
    static let userDefaultsKey = "local_ai_app_language_id_v1"
    static let legacyUserDefaultsKey = "local_ai_app_language_v1"

    static func reloadLanguages() {
        LanguageRegistry.shared.reload()
    }

    static func availableLanguageOptions() -> [AppLanguage] {
        LanguageRegistry.shared.ensureLoaded()
        let descriptors = LanguageRegistry.shared.availableDescriptors()
        let priority = ["kor", "eng", "jpn"]
        let prioritized = priority.compactMap { priorityID in
            descriptors.first { $0.id == priorityID }
        }
        let remainder = descriptors
            .filter { !priority.contains($0.id) }
            .sorted { lhs, rhs in
                let lhsKey = lhs.displayName.lowercased()
                let rhsKey = rhs.displayName.lowercased()
                if lhsKey == rhsKey {
                    return lhs.id < rhs.id
                }
                return lhsKey < rhsKey
            }
        return [.auto] + (prioritized + remainder).map { AppLanguage(rawValue: $0.id) }
    }

    static func selectionFromSettings(_ value: String?) -> AppLanguage {
        guard let raw = value?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty else {
            return .auto
        }
        return AppLanguage(rawValue: raw)
    }

    static func saveSelection(_ language: AppLanguage) {
        UserDefaults.standard.set(language.rawValue, forKey: userDefaultsKey)
        UserDefaults.standard.removeObject(forKey: legacyUserDefaultsKey)
    }

    static func loadSelection() -> AppLanguage {
        let defaults = UserDefaults.standard
        if let raw = defaults.string(forKey: userDefaultsKey) {
            return selectionFromSettings(raw)
        }
        if let legacy = defaults.string(forKey: legacyUserDefaultsKey) {
            let migrated = selectionFromSettings(legacy)
            saveSelection(migrated)
            return migrated
        }
        return .auto
    }

    static func sidecarLanguageCode(for language: AppLanguage) -> String {
        if language.isAuto {
            return "auto"
        }
        return isoCode(for: language)
    }

    static func isoCode(for language: AppLanguage) -> String {
        LanguageRegistry.shared.ensureLoaded()
        let id = AppLanguage.normalize(language.rawValue)
        if let descriptor = LanguageRegistry.shared.descriptor(for: id) {
            return descriptor.iso
        }
        switch id {
        case "kor": return "ko"
        case "eng": return "en"
        case "jpn": return "ja"
        default: return "en"
        }
    }

    static func effectiveCode(for language: AppLanguage) -> String {
        isoCode(for: displayLanguage(for: language))
    }

    static func displayLanguage(for language: AppLanguage) -> AppLanguage {
        LanguageRegistry.shared.ensureLoaded()
        let effectiveID = effectiveLanguageID(for: language)
        return AppLanguage(rawValue: effectiveID)
    }

    static func tr(_ key: String, language: AppLanguage? = nil) -> String {
        tr(key, language: language ?? loadSelection(), fallbackKo: nil, fallbackEn: nil, fallbackJa: nil)
    }

    static func tr(
        _ key: String,
        language: AppLanguage,
        fallbackKo: String?,
        fallbackEn: String?,
        fallbackJa: String?
    ) -> String {
        LanguageRegistry.shared.ensureLoaded()
        let selectedID = effectiveLanguageID(for: language)
        var candidates: [String] = [selectedID, "eng", "kor"]
        var seen: Set<String> = []
        candidates = candidates.filter { seen.insert($0).inserted }

        for candidate in candidates {
            if let value = LanguageRegistry.shared.string(for: key, languageID: candidate),
               !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                return value
            }
        }

        if selectedID == "jpn", let fallbackJa, !fallbackJa.isEmpty {
            return fallbackJa
        }
        if selectedID == "eng", let fallbackEn, !fallbackEn.isEmpty {
            return fallbackEn
        }
        if let fallbackKo, !fallbackKo.isEmpty {
            return fallbackKo
        }
        if let fallbackEn, !fallbackEn.isEmpty {
            return fallbackEn
        }
        if let fallbackJa, !fallbackJa.isEmpty {
            return fallbackJa
        }
        return key
    }

    static func text(_ ko: String, _ en: String, _ ja: String, language: AppLanguage) -> String {
        let key = legacyKey(ko: ko, en: en, ja: ja)
        return tr(key, language: language, fallbackKo: ko, fallbackEn: en, fallbackJa: ja)
    }

    static func text(_ ko: String, _ en: String, _ ja: String) -> String {
        text(ko, en, ja, language: loadSelection())
    }

    static func languageOptionTitle(_ option: AppLanguage, language: AppLanguage) -> String {
        if option.isAuto {
            return tr(
                "settings.language.option.auto",
                language: language,
                fallbackKo: "자동",
                fallbackEn: "Auto",
                fallbackJa: "自動"
            )
        }
        if let descriptor = LanguageRegistry.shared.descriptor(for: option.rawValue) {
            if descriptor.nativeName.caseInsensitiveCompare(descriptor.displayName) == .orderedSame {
                return descriptor.nativeName
            }
            return "\(descriptor.nativeName) (\(descriptor.displayName))"
        }
        return option.rawValue
    }

    private static func effectiveLanguageID(for language: AppLanguage) -> String {
        LanguageRegistry.shared.ensureLoaded()
        if !language.isAuto {
            let selectedID = AppLanguage.normalize(language.rawValue)
            if LanguageRegistry.shared.containsLanguage(id: selectedID) {
                return selectedID
            }
            return "kor"
        }

        let systemISO = (Locale.current.language.languageCode?.identifier ?? "ko").lowercased()
        let mapped: String
        switch systemISO {
        case "ko": mapped = "kor"
        case "en": mapped = "eng"
        case "ja": mapped = "jpn"
        default: mapped = "eng"
        }
        if LanguageRegistry.shared.containsLanguage(id: mapped) {
            return mapped
        }
        return "kor"
    }

    private static func legacyKey(ko: String, en: String, ja: String) -> String {
        let source = "\(ko)\u{001F}\(en)\u{001F}\(ja)"
        var hash: UInt64 = 1469598103934665603
        for byte in source.utf8 {
            hash ^= UInt64(byte)
            hash = hash &* 1099511628211
        }
        let digest = String(hash, radix: 16, uppercase: false)
        let slug = en
            .lowercased()
            .replacingOccurrences(of: "[^a-z0-9]+", with: "_", options: .regularExpression)
            .trimmingCharacters(in: CharacterSet(charactersIn: "_"))
        let safeSlug = slug.isEmpty ? "value" : String(slug.prefix(36))
        return "legacy.\(safeSlug).\(digest)"
    }
}
