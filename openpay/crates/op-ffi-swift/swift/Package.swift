// swift-tools-version:5.9
// OpenPay Swift Package — consumer-facing wrapper around the
// op-ffi-swift xcframework.
//
// Usage in your app's Package.swift:
//
//   dependencies: [
//     .package(path: "../openpay/crates/op-ffi-swift/swift")
//   ],
//   targets: [
//     .target(
//       name: "MyApp",
//       dependencies: [.product(name: "OpenPay", package: "OpenPay")]
//     )
//   ]
//
// Or via SwiftPM directly:
//
//   .package(url: "git@github.com:org/openpay.git", from: "0.7.0")
//
// The package vendors the prebuilt xcframework (produced by
// scripts/build-xcframework.sh) and the generated swift-bridge glue.

import PackageDescription

let package = Package(
    name: "OpenPay",
    platforms: [
        .iOS(.v15),
        .macOS(.v12),
    ],
    products: [
        .library(
            name: "OpenPay",
            targets: ["OpenPay"]
        ),
    ],
    targets: [
        // Prebuilt Rust core as a binary target. The .xcframework is
        // produced out of band by scripts/build-xcframework.sh and
        // dropped into ../target/OpenPay.xcframework.
        .binaryTarget(
            name: "OpenPayRust",
            path: "../target/OpenPay.xcframework"
        ),
        // The Swift module that wraps the C ABI and re-exports the
        // swift-bridge glue. Depends on the binary target for linkage.
        .target(
            name: "OpenPay",
            dependencies: ["OpenPayRust"],
            path: "OpenPay",
            // The swift-bridge glue lives next to the wrapper; SwiftPM
            // picks up the .swift, .h, and .modulemap automatically.
            publicHeadersPath: "include"
        ),
    ]
)
