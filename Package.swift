// swift-tools-version: 5.10
import PackageDescription

let package = Package(
    name: "LocalAICoreForMac",
    platforms: [
        .macOS(.v14)
    ],
    products: [
        // Canonical app entry point is the Xcode target in `PLOS.xcodeproj`.
        // Keep SwiftPM only for lightweight shared utilities/tools.
        .library(name: "PLOSPackageSupport", targets: ["PLOSPackageSupport"])
    ],
    targets: [
        .target(
            name: "PLOSPackageSupport",
            path: "PackageSupport"
        )
    ]
)
