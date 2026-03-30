import Foundation

// API 모델은 도메인별로 분리되었습니다.
// - Models/API/APIBaseModels.swift
// - Plugins/Models/APIExtensionPluginModels.swift
// - Models/API/APIChatStateModels.swift

// Build-time bridge marker:
// If split files are not included in a build target, fail fast.
typealias _APIModelsSplitBridge = APIBaseModelsSplitMarker
