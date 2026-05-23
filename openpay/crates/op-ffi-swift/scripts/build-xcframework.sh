#!/usr/bin/env bash
# Build OpenPay as an XCFramework for iOS device, iOS simulator (both
# Apple Silicon and Intel slices), and macOS (both arches).
#
# Output layout:
#   target/OpenPay.xcframework/
#     ios-arm64/                 -- device
#     ios-arm64_x86_64-simulator/ -- simulator (fat: arm64 + x86_64)
#     macos-arm64_x86_64/        -- macOS (fat: arm64 + x86_64)
#
# Plus the swift-bridge generated glue:
#   target/swift/OpenPay.swift
#   target/swift/openpay-swift-bridge.h
#   target/swift/module.modulemap
#
# Prerequisites:
#   - macOS host with Xcode (xcodebuild + lipo).
#   - rustup target add aarch64-apple-ios aarch64-apple-ios-sim x86_64-apple-ios \
#                       aarch64-apple-darwin x86_64-apple-darwin
#
# Usage:
#   bash scripts/build-xcframework.sh [--release | --debug]
#
# Verified target triples against doc.rust-lang.org/rustc/platform-support
# entries for *-apple-ios (tier 2) and the iOS Rust platform notes as of
# 2026. The simulator distinction is encoded in the target triple
# (-sim suffix) and is consumed by xcodebuild via the static library's
# embedded LC_BUILD_VERSION command. The lipo + xcodebuild combination
# matches Apple's documented workflow for static libraries.

set -euo pipefail

CRATE_NAME="op-ffi-swift"
LIB_BASENAME="libop_ffi_swift"

PROFILE="${1:-release}"
PROFILE="${PROFILE#--}"
if [[ "$PROFILE" != "release" && "$PROFILE" != "debug" ]]; then
    echo "usage: $0 [--release|--debug]" >&2
    exit 2
fi

CARGO_FLAGS=""
if [[ "$PROFILE" == "release" ]]; then
    CARGO_FLAGS="--release"
fi

# Find the workspace root. Run from anywhere inside it.
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CRATE_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
WORKSPACE_ROOT="$( cd "$CRATE_DIR/../.." && pwd )"

OUT_DIR="$WORKSPACE_ROOT/target/OpenPay.xcframework"
SWIFT_GLUE_DIR="$WORKSPACE_ROOT/target/swift"
TARGET_DIR="$WORKSPACE_ROOT/target"

echo ">>> building $CRATE_NAME for $PROFILE"
echo ">>> workspace: $WORKSPACE_ROOT"
echo ">>> output:    $OUT_DIR"

# Targets we build for.
TARGETS=(
    aarch64-apple-ios       # iOS device
    aarch64-apple-ios-sim   # iOS simulator (Apple Silicon)
    x86_64-apple-ios        # iOS simulator (Intel)
    aarch64-apple-darwin    # macOS arm64
    x86_64-apple-darwin     # macOS x86_64
)

# Build the staticlib for each target.
for t in "${TARGETS[@]}"; do
    echo ">>> cargo build --target $t"
    (cd "$WORKSPACE_ROOT" && cargo build -p "$CRATE_NAME" --target "$t" $CARGO_FLAGS)
done

# Helper: locate the produced static library for a target.
lib_path_for() {
    echo "$TARGET_DIR/$1/$PROFILE/${LIB_BASENAME}.a"
}

# Verify every output exists before we try to lipo or wrap.
for t in "${TARGETS[@]}"; do
    p="$(lib_path_for "$t")"
    if [[ ! -f "$p" ]]; then
        echo "missing: $p" >&2
        exit 1
    fi
done

# Lipo the two simulator arches into a fat archive.
SIM_FAT="$TARGET_DIR/sim-fat/${LIB_BASENAME}.a"
mkdir -p "$TARGET_DIR/sim-fat"
echo ">>> lipo simulator (arm64 + x86_64)"
lipo -create \
    "$(lib_path_for aarch64-apple-ios-sim)" \
    "$(lib_path_for x86_64-apple-ios)" \
    -output "$SIM_FAT"

# Lipo the two macOS arches into a fat archive.
MAC_FAT="$TARGET_DIR/mac-fat/${LIB_BASENAME}.a"
mkdir -p "$TARGET_DIR/mac-fat"
echo ">>> lipo macOS (arm64 + x86_64)"
lipo -create \
    "$(lib_path_for aarch64-apple-darwin)" \
    "$(lib_path_for x86_64-apple-darwin)" \
    -output "$MAC_FAT"

# Locate the swift-bridge generated artifacts. The build script emits
# them under target/<TRIPLE>/<PROFILE>/build/op-ffi-swift-<HASH>/out/.
# Any of the per-target build dirs has them — we just pick the first.
FIRST_TARGET="${TARGETS[0]}"
GLUE_SRC="$(find "$TARGET_DIR/$FIRST_TARGET/$PROFILE/build" -type d -name out -path '*op-ffi-swift-*' | head -1 || true)"
if [[ -z "$GLUE_SRC" ]]; then
    echo "warning: could not locate swift-bridge glue dir; you may be building with --features c-only" >&2
else
    echo ">>> copying swift-bridge glue from $GLUE_SRC"
    mkdir -p "$SWIFT_GLUE_DIR"
    # The swift-bridge output is laid out as:
    #   $GLUE_SRC/SwiftBridgeCore.swift
    #   $GLUE_SRC/SwiftBridgeCore.h
    #   $GLUE_SRC/OpenPay/OpenPay.swift
    #   $GLUE_SRC/OpenPay/OpenPay.h
    # Copy everything in one go.
    cp -R "$GLUE_SRC"/* "$SWIFT_GLUE_DIR/"
fi

# Build the XCFramework. xcodebuild infers platform variant keys from
# each library's embedded LC_BUILD_VERSION command, so we don't pass
# -headers per slice; the headers live in the swift-bridge-generated
# OpenPay/ directory which the Swift package consumes separately.
echo ">>> xcodebuild -create-xcframework"
rm -rf "$OUT_DIR"
xcodebuild -create-xcframework \
    -library "$(lib_path_for aarch64-apple-ios)" \
    -library "$SIM_FAT" \
    -library "$MAC_FAT" \
    -output "$OUT_DIR"

echo ""
echo "=========================================================="
echo "Built: $OUT_DIR"
echo "Swift glue: $SWIFT_GLUE_DIR"
echo ""
echo "Drop OpenPay.xcframework into your Xcode project's Frameworks,"
echo "Libraries, and Embedded Content. Add the swift/ directory as a"
echo "Swift Package or copy the .swift / .h / .modulemap into your"
echo "target's sources."
echo "=========================================================="
