// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "LocalAICoreForMac",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        .executable(name: "LocalAICoreApp", targets: ["LocalAICoreApp"])
    ],
    targets: [
        .executableTarget(
            name: "LocalAICoreApp",
            path: "Sources/LocalAICoreApp"
        )
    ]
)
