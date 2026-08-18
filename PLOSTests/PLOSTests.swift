//
//  PLOSTests.swift
//  PLOSTests
//
//  Created by Seung Min Lee on 3/17/26.
//

import Testing
import Foundation
@testable import PLOS

struct PLOSTests {
    @Test
    @MainActor
    func filtersModelArtifactsForPicker() {
        let vm = AppViewModel()
        let now = Date()
        vm.availableModels = [
            ModelListItem(file_name: ".gitignore", path: "/tmp/.gitignore", engine: .llamaCPP, size_bytes: 1, modified_at: now),
            ModelListItem(file_name: "Qwen3-8B-Q4_K_M.gguf.metadata", path: "/tmp/Qwen3-8B-Q4_K_M.gguf.metadata", engine: .llamaCPP, size_bytes: 1, modified_at: now),
            ModelListItem(file_name: "catalog_state.json", path: "/tmp/catalog_state.json", engine: .mlx, size_bytes: 1, modified_at: now),
            ModelListItem(file_name: "readme.txt", path: "/tmp/readme.txt", engine: .llamaCPP, size_bytes: 1, modified_at: now),
            ModelListItem(file_name: "Qwen3-8B-Q4_K_M.gguf", path: "/tmp/Qwen3-8B-Q4_K_M.gguf", engine: .llamaCPP, size_bytes: 1024, modified_at: now),
            ModelListItem(file_name: "mlx-model", path: "/tmp/mlx/model", engine: .mlx, size_bytes: 2048, modified_at: now),
        ]

        let names = Set(vm.installedModelsSorted.map(\.file_name))
        #expect(names.contains("Qwen3-8B-Q4_K_M.gguf"))
        #expect(names.contains("mlx-model"))
        #expect(!names.contains(".gitignore"))
        #expect(!names.contains("Qwen3-8B-Q4_K_M.gguf.metadata"))
        #expect(!names.contains("catalog_state.json"))
        #expect(!names.contains("readme.txt"))
    }

    @Test
    func quickInferencePresetMapping() {
        #expect(QuickInferencePreset.fast.startupProfile == .fast)
        #expect(QuickInferencePreset.quality.startupProfile == .recommended)
        #expect(QuickInferencePreset.highQuality.startupProfile == .deep)
    }

    @Test
    func languageSelectionMigrationAndSidecarMapping() {
        #expect(L10n.selectionFromSettings("ko") == .kor)
        #expect(L10n.selectionFromSettings("en-us") == .eng)
        #expect(L10n.selectionFromSettings("ja-jp") == .jpn)
        #expect(L10n.selectionFromSettings("auto") == .auto)

        #expect(L10n.sidecarLanguageCode(for: .auto) == "auto")
        #expect(L10n.sidecarLanguageCode(for: .kor) == "ko")
        #expect(L10n.sidecarLanguageCode(for: .eng) == "en")
        #expect(L10n.sidecarLanguageCode(for: .jpn) == "ja")
    }

    @Test
    func localizationFallbackOrder() {
        let key = "test.missing_key"
        let resultEN = L10n.tr(
            key,
            language: .eng,
            fallbackKo: "한국어",
            fallbackEn: "English",
            fallbackJa: "日本語"
        )
        let resultKO = L10n.tr(
            key,
            language: .kor,
            fallbackKo: "한국어",
            fallbackEn: "English",
            fallbackJa: "日本語"
        )
        let resultJA = L10n.tr(
            key,
            language: .jpn,
            fallbackKo: "한국어",
            fallbackEn: "English",
            fallbackJa: "日本語"
        )
        #expect(resultEN == "English")
        #expect(resultKO == "한국어")
        #expect(resultJA == "日本語")
    }

    @Test
    func streamingParserPreservesTokenizerBoundaries() {
        let first = StreamTagParser.splitReasoningAndAnswer(from: "첫 번째 문장 ")
        let second = StreamTagParser.splitReasoningAndAnswer(from: "다음 문장입니다.")

        #expect(first.answerText + second.answerText == "첫 번째 문장 다음 문장입니다.")
    }

    @Test
    func markdownFormatterRestoresInlineOutlineBlocks() {
        let raw = "1. 목표 설정 * 핵심 문제에 집중 * MVP 완성도 2. 실행 품질 향상 * 피드백 반영"
        let normalized = ChatPanelMarkdownFormatter.normalizeMarkdownForRender(raw)

        #expect(normalized.hasPrefix("1. 목표 설정"))
        #expect(normalized.contains("\n\n* 핵심 문제에 집중"))
        #expect(normalized.contains("\n\n2. 실행 품질 향상"))
        #expect(normalized.contains("\n\n* 피드백 반영"))
    }

    @Test
    func generatedChatTitleIsReducedToOneSafeLine() {
        let service = ChatRoomService()
        let title = service.normalizeGeneratedChatTitle("제목:  해커톤 우승 전략\n")

        #expect(title == "해커톤 우승 전략")
        #expect(service.normalizeGeneratedChatTitle("-") == nil)
    }
}
