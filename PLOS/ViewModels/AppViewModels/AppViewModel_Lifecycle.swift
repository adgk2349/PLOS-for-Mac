import AppKit
import Combine
import CryptoKit
import Foundation

@MainActor
extension AppViewModel {
    func bootstrap() async {
        isSidecarReadyForChat = false
        L10n.reloadLanguages()
        hasFinishedOnboarding = UserDefaults.standard.bool(forKey: UDKey.onboardingFinished)
        includedFolderURLs = bookmarkStore.loadURLs()
        loadApprovedSystemActionKinds()
        loadChatRooms()
        loadChatResponseRoute()
        loadRoleplayPreference()
        loadAppLanguagePreference()
        loadSearXNGPreference()
        loadStorageDirectoryPreferences()
        loadSidecarVisionPreferences()
        loadLocalModelPreferenceSnapshot()
        loadSecretAPIKeys()
        syncQuickInferencePresetFromProfile()

        do {
            try await sidecar.start()
            _ = try await ensureSidecarClient()
            isSidecarReadyForChat = true
            syncStorageDirectoryResolutionFromSidecar()
            if hasFinishedOnboarding {
                try await refreshRemoteState()
                try await syncWorkspaceAndSettings()
                try await refreshRemoteState()
            }
        } catch {
            isSidecarReadyForChat = false
            handleViewModelError(error)
        }
    }


    func shutdown() {
        isSidecarReadyForChat = false
        Task { @MainActor in
            await sidecar.stop()
        }
    }


    func startOnboardingIndexingFlow() async {
        guard !includedFolderURLs.isEmpty else {
            lastError = "최소 1개 이상의 폴더를 선택해 주세요."
            return
        }

        onboardingStep = .indexing
        isBusy = true
        defer { isBusy = false }

        do {
            try await syncWorkspaceAndSettings()
            try await runIndexing(scope: "full")
            onboardingStep = .ready
            try await refreshRemoteState()
        } catch {
            handleViewModelError(error)
        }
    }


    func finalizeOnboarding() {
        hasFinishedOnboarding = true
        UserDefaults.standard.set(true, forKey: UDKey.onboardingFinished)
    }
}
