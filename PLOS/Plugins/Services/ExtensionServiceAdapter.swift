import Foundation

final class ExtensionServiceAdapter {
    func fetchCapabilities(client: SidecarAPIClient) async throws -> ExtensionCapabilitiesResponse {
        try await client.getExtensionCapabilities()
    }

    func fetchPluginRegistry(client: SidecarAPIClient) async throws -> PluginRegistryResponse {
        try await client.listExtensionPlugins()
    }

    func registerPlugin(client: SidecarAPIClient, request: PluginRegisterRequest) async throws -> PluginRegistryResponse {
        try await client.registerExtensionPlugin(request)
    }

    func enablePlugin(client: SidecarAPIClient, pluginID: String) async throws -> PluginEnableResponse {
        try await client.enableExtensionPlugin(pluginID: pluginID)
    }

    func disablePlugin(client: SidecarAPIClient, pluginID: String) async throws -> PluginEnableResponse {
        try await client.disableExtensionPlugin(pluginID: pluginID)
    }

    func deletePlugin(client: SidecarAPIClient, pluginID: String) async throws -> Bool {
        try await client.deleteExtensionPlugin(pluginID: pluginID)
    }

    func openPanel(client: SidecarAPIClient, request: PluginPanelOpenRequest) async throws -> PluginPanelOpenResponse {
        try await client.openExtensionPanel(request)
    }

    func submitPanelAction(client: SidecarAPIClient, request: PluginPanelActionRequest) async throws -> PluginPanelActionResponse {
        try await client.submitExtensionPanelAction(request)
    }

    func panelStatus(client: SidecarAPIClient, jobID: String) async throws -> PluginPanelStatusResponse {
        try await client.getExtensionPanelStatus(jobID: jobID)
    }

    func streamPanelStatus(client: SidecarAPIClient, jobID: String) async throws -> AsyncThrowingStream<PluginPanelStatusResponse, Error> {
        try await client.streamExtensionPanelStatus(jobID: jobID)
    }

    func generateImage(client: SidecarAPIClient, request: ExtensionImageGenerateRequest) async throws -> ExtensionImageGenerateResponse {
        try await client.generateExtensionImage(request)
    }
}
