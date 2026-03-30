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
}
