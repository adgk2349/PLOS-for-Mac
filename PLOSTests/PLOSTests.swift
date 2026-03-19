//
//  PLOSTests.swift
//  PLOSTests
//
//  Created by Seung Min Lee on 3/17/26.
//

import Testing
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
}
