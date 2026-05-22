import Foundation

enum ExtensionCapability: String, Codable, CaseIterable, Identifiable, Hashable {
    case retrieverSearch = "retriever.search"
    case rerankerRank = "reranker.rank"
    case summarizerGenerate = "summarizer.generate"
    case retrievalQueryTransform = "retrieval.query_transform"
    case retrievalPostFilter = "retrieval.post_filter"
    case chunkingStrategy = "chunking.strategy"
    case embeddingProvider = "embedding.provider"
    case indexingPreprocess = "indexing.preprocess"
    case finetuneJobSubmit = "finetune.job_submit"
    case finetuneJobStatus = "finetune.job_status"
    case finetuneModelPublish = "finetune.model_publish"
    case imageGenerate = "image.generate"

    var id: String { rawValue }

    func title(language: AppLanguage) -> String {
        switch self {
        case .retrieverSearch:
            return L10n.tr("extension_capability.retriever_search", language: language, fallbackKo: "검색 수집기", fallbackEn: "Retriever Search", fallbackJa: "検索リトリーバ")
        case .rerankerRank:
            return L10n.tr("extension_capability.reranker_rank", language: language, fallbackKo: "재정렬기", fallbackEn: "Reranker Rank", fallbackJa: "再ランキング")
        case .summarizerGenerate:
            return L10n.tr("extension_capability.summarizer_generate", language: language, fallbackKo: "요약 생성기", fallbackEn: "Summarizer Generate", fallbackJa: "要約生成")
        case .retrievalQueryTransform:
            return L10n.tr("extension_capability.retrieval_query_transform", language: language, fallbackKo: "검색 질의 변환", fallbackEn: "Query Transform", fallbackJa: "検索クエリ変換")
        case .retrievalPostFilter:
            return L10n.tr("extension_capability.retrieval_post_filter", language: language, fallbackKo: "검색 후처리", fallbackEn: "Post Filter", fallbackJa: "検索後フィルタ")
        case .chunkingStrategy:
            return L10n.tr("extension_capability.chunking_strategy", language: language, fallbackKo: "청킹 전략", fallbackEn: "Chunking Strategy", fallbackJa: "チャンク戦略")
        case .embeddingProvider:
            return L10n.tr("extension_capability.embedding_provider", language: language, fallbackKo: "임베딩 제공자", fallbackEn: "Embedding Provider", fallbackJa: "埋め込みプロバイダ")
        case .indexingPreprocess:
            return L10n.tr("extension_capability.indexing_preprocess", language: language, fallbackKo: "인덱싱 전처리", fallbackEn: "Indexing Preprocess", fallbackJa: "索引前処理")
        case .finetuneJobSubmit:
            return L10n.tr("extension_capability.finetune_job_submit", language: language, fallbackKo: "파인튜닝 제출", fallbackEn: "Finetune Submit", fallbackJa: "微調整ジョブ提出")
        case .finetuneJobStatus:
            return L10n.tr("extension_capability.finetune_job_status", language: language, fallbackKo: "파인튜닝 상태 조회", fallbackEn: "Finetune Status", fallbackJa: "微調整ステータス")
        case .finetuneModelPublish:
            return L10n.tr("extension_capability.finetune_model_publish", language: language, fallbackKo: "파인튜닝 모델 배포", fallbackEn: "Finetune Publish", fallbackJa: "微調整モデル公開")
        case .imageGenerate:
            return L10n.tr("extension_capability.image_generate", language: language, fallbackKo: "이미지 생성", fallbackEn: "Image Generate", fallbackJa: "画像生成")
        }
    }

    var title: String { title(language: L10n.loadSelection()) }
}

enum PluginCapabilitySource: String, Codable, Hashable {
    case builtIn = "built_in"
    case plugin = "plugin"
    case disabled = "disabled"

    func title(language: AppLanguage) -> String {
        switch self {
        case .builtIn:
            return L10n.tr("plugin_capability_source.built_in", language: language, fallbackKo: "내장", fallbackEn: "Built-in", fallbackJa: "内蔵")
        case .plugin:
            return L10n.tr("plugin_capability_source.plugin", language: language, fallbackKo: "플러그인", fallbackEn: "Plugin", fallbackJa: "プラグイン")
        case .disabled:
            return L10n.tr("plugin_capability_source.disabled", language: language, fallbackKo: "비활성화", fallbackEn: "Disabled", fallbackJa: "無効")
        }
    }

    var title: String { title(language: L10n.loadSelection()) }
}

enum PluginBuildTarget: String, Codable, CaseIterable, Identifiable, Hashable {
    case community = "community"
    case enterprise = "enterprise"
    case both = "both"

    var id: String { rawValue }

    func title(language: AppLanguage) -> String {
        switch self {
        case .community:
            return L10n.tr("plugin_build_target.community", language: language, fallbackKo: "커뮤니티", fallbackEn: "Community", fallbackJa: "コミュニティ")
        case .enterprise:
            return L10n.tr("plugin_build_target.enterprise", language: language, fallbackKo: "엔터프라이즈", fallbackEn: "Enterprise", fallbackJa: "エンタープライズ")
        case .both:
            return L10n.tr("plugin_build_target.both", language: language, fallbackKo: "모두", fallbackEn: "Both", fallbackJa: "両方")
        }
    }

    var title: String { title(language: L10n.loadSelection()) }
}

enum PluginErrorCode: String, Codable, Hashable {
    case pluginTimeout = "PLUGIN_TIMEOUT"
    case pluginUnavailable = "PLUGIN_UNAVAILABLE"
    case pluginValidationError = "PLUGIN_VALIDATION_ERROR"
    case pluginPermissionDenied = "PLUGIN_PERMISSION_DENIED"
}

enum PluginPrivacyMode: String, Codable, CaseIterable, Identifiable, Hashable {
    case localOnly = "LOCAL_ONLY"
    case hybrid = "HYBRID"
    case externalAllowed = "EXTERNAL_ALLOWED"

    var id: String { rawValue }

    func title(language: AppLanguage) -> String {
        switch self {
        case .localOnly:
            return L10n.tr("plugin_privacy_mode.local_only", language: language, fallbackKo: "완전 로컬", fallbackEn: "Local only", fallbackJa: "完全ローカル")
        case .hybrid:
            return L10n.tr("plugin_privacy_mode.hybrid", language: language, fallbackKo: "하이브리드", fallbackEn: "Hybrid", fallbackJa: "ハイブリッド")
        case .externalAllowed:
            return L10n.tr("plugin_privacy_mode.external_allowed", language: language, fallbackKo: "외부 호출 허용", fallbackEn: "External allowed", fallbackJa: "外部呼び出し許可")
        }
    }
}

enum PluginUIToggleLocation: String, Codable, Hashable {
    case composerModelControls = "composer.model_controls"
}

struct PluginUIToggleSpec: Codable, Identifiable, Hashable {
    var id: String
    var location: PluginUIToggleLocation
    var default_enabled: Bool = false
    var title_ko: String?
    var title_en: String?
    var title_ja: String?
    var help_ko: String?
    var help_en: String?
    var help_ja: String?

    func title(language: AppLanguage) -> String {
        switch language {
        case .kor:
            return title_ko ?? title_en ?? title_ja ?? id
        case .jpn:
            return title_ja ?? title_en ?? title_ko ?? id
        case .eng:
            return title_en ?? title_ko ?? title_ja ?? id
        case .auto:
            return title_ko ?? title_en ?? title_ja ?? id
        default:
            return title_en ?? title_ko ?? title_ja ?? id
        }
    }

    func help(language: AppLanguage) -> String? {
        switch language {
        case .kor:
            return help_ko ?? help_en ?? help_ja
        case .jpn:
            return help_ja ?? help_en ?? help_ko
        case .eng:
            return help_en ?? help_ko ?? help_ja
        case .auto:
            return help_ko ?? help_en ?? help_ja
        default:
            return help_en ?? help_ko ?? help_ja
        }
    }
}

struct PluginUIManifest: Codable, Hashable {
    var toggles: [PluginUIToggleSpec] = []
    var views: [PluginUIViewSpec] = []
}

enum PluginUIViewLocation: String, Codable, Hashable {
    case mainPanel = "main.panel"
}

enum PluginUIViewType: String, Codable, Hashable {
    case imageStudio = "image_studio"
    case customForm = "custom_form"
}

struct PluginUIViewSpec: Codable, Identifiable, Hashable {
    var id: String
    var location: PluginUIViewLocation
    var view_type: PluginUIViewType
    var title_ko: String?
    var title_en: String?
    var title_ja: String?
    var help_ko: String?
    var help_en: String?
    var help_ja: String?
    var schema: [String: JSONValue]

    init(
        id: String,
        location: PluginUIViewLocation,
        view_type: PluginUIViewType,
        title_ko: String? = nil,
        title_en: String? = nil,
        title_ja: String? = nil,
        help_ko: String? = nil,
        help_en: String? = nil,
        help_ja: String? = nil,
        schema: [String: JSONValue] = [:]
    ) {
        self.id = id
        self.location = location
        self.view_type = view_type
        self.title_ko = title_ko
        self.title_en = title_en
        self.title_ja = title_ja
        self.help_ko = help_ko
        self.help_en = help_en
        self.help_ja = help_ja
        self.schema = schema
    }

    func title(language: AppLanguage) -> String {
        switch language {
        case .kor:
            return title_ko ?? title_en ?? title_ja ?? id
        case .jpn:
            return title_ja ?? title_en ?? title_ko ?? id
        case .eng:
            return title_en ?? title_ko ?? title_ja ?? id
        case .auto:
            return title_ko ?? title_en ?? title_ja ?? id
        default:
            return title_en ?? title_ko ?? title_ja ?? id
        }
    }

    func help(language: AppLanguage) -> String? {
        switch language {
        case .kor:
            return help_ko ?? help_en ?? help_ja
        case .jpn:
            return help_ja ?? help_en ?? help_ko
        case .eng:
            return help_en ?? help_ko ?? help_ja
        case .auto:
            return help_ko ?? help_en ?? help_ja
        default:
            return help_en ?? help_ko ?? help_ja
        }
    }
}

struct PluginManifestV1: Codable, Hashable {
    var plugin_id: String
    var version: String
    var api_version: String
    var capabilities: [ExtensionCapability]
    var privacy_mode: PluginPrivacyMode
    var permissions: [String]
    var entrypoint: String
    var signature: String?
    var build_target: PluginBuildTarget
    var ui: PluginUIManifest?
}

struct ExtensionCapabilityState: Codable, Identifiable, Hashable {
    var capability: ExtensionCapability
    var source: PluginCapabilitySource
    var plugin_enabled: Bool
    var plugin_id: String?
    var error_code: PluginErrorCode?
    var plugin_privacy_mode: PluginPrivacyMode?
    var effective_privacy_mode: PluginPrivacyMode?
    var blocked_reason: String?

    var id: String { capability.rawValue }
}

struct ExtensionCapabilitiesResponse: Codable {
    var version: Int
    var capabilities: [ExtensionCapabilityState]
}

struct PluginRegistryEntry: Codable, Identifiable, Hashable {
    var plugin_id: String
    var manifest: PluginManifestV1
    var enabled: Bool
    var state: String
    var updated_at: Date
    var validation_error: String?
    var is_builtin: Bool?

    var id: String { plugin_id }
}

struct PluginRegistryResponse: Codable {
    var entries: [PluginRegistryEntry]
}

struct PluginRegisterRequest: Codable {
    var manifest: PluginManifestV1
    var enabled: Bool
}

struct PluginEnableResponse: Codable {
    var plugin: PluginRegistryEntry
    var capabilities: [ExtensionCapabilityState]
}

struct PluginPanelOpenRequest: Codable {
    var plugin_id: String
    var panel_id: String?
}

struct PluginPanelOpenResponse: Codable {
    var plugin_id: String
    var panel_id: String
    var view_type: PluginUIViewType
    var title: String
    var help: String?
    var schema: [String: JSONValue]
    var defaults: [String: JSONValue]
}

struct PluginPanelActionRequest: Codable {
    var plugin_id: String
    var panel_id: String
    var action: String
    var payload: [String: JSONValue]
}

struct PluginPanelActionResponse: Codable {
    var plugin_id: String
    var panel_id: String
    var job_id: String
    var status: String
    var result: [String: JSONValue]
    var error: String?
}

struct PluginPanelStatusResponse: Codable {
    var plugin_id: String
    var panel_id: String
    var job_id: String
    var status: String
    var result: [String: JSONValue]
    var error: String?
}

struct ExtensionImageGenerateRequest: Codable {
    var plugin_id: String
    var prompt: String
    var negative_prompt: String?
    var width: Int
    var height: Int
    var steps: Int
    var seed: Int?
    var batch: Int
}

struct ExtensionImageGenerateResponse: Codable {
    var plugin_id: String
    var job_id: String
    var status: String
    var result: [String: JSONValue]
    var error: String?
}
