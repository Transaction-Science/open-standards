#!/usr/bin/env bash
# Build OpenPay JNI library for all Android ABIs and stage into the
# Kotlin module's jniLibs directory.
#
# Output layout:
#   kotlin/openpay/src/main/jniLibs/arm64-v8a/libop_ffi_jni.so
#   kotlin/openpay/src/main/jniLibs/armeabi-v7a/libop_ffi_jni.so
#   kotlin/openpay/src/main/jniLibs/x86/libop_ffi_jni.so
#   kotlin/openpay/src/main/jniLibs/x86_64/libop_ffi_jni.so
#
# Prerequisites:
#   - Android NDK r25 or later (set ANDROID_NDK_HOME or
#     ANDROID_NDK_ROOT, or pass --android-platform to cargo-ndk).
#   - rustup target add aarch64-linux-android armv7-linux-androideabi \
#                       x86_64-linux-android i686-linux-android
#   - cargo install cargo-ndk
#
# Usage:
#   bash scripts/build-android.sh [--release|--debug] [--min-sdk N]
#
# The four ABIs map to the Android ABI conventions per Android Developer
# documentation (developer.android.com/ndk/guides/abis):
#   arm64-v8a       — modern 64-bit ARM (all current Android devices)
#   armeabi-v7a     — older 32-bit ARM (pre-2019 devices)
#   x86_64          — emulator on Intel/AMD Macs and Linux dev hosts
#   x86             — older 32-bit emulator (rarely used; included for completeness)

set -euo pipefail

CRATE_NAME="op-ffi-jni"
LIB_BASENAME="libop_ffi_jni"
SO_NAME="${LIB_BASENAME}.so"

PROFILE="release"
MIN_SDK="23"   # Android 6.0 Marshmallow. Matches AndroidX Security 1.1+.

while [[ $# -gt 0 ]]; do
    case "$1" in
        --release) PROFILE="release"; shift ;;
        --debug)   PROFILE="debug"; shift ;;
        --min-sdk) MIN_SDK="$2"; shift 2 ;;
        *) echo "usage: $0 [--release|--debug] [--min-sdk N]" >&2; exit 2 ;;
    esac
done

CARGO_FLAGS=""
[[ "$PROFILE" == "release" ]] && CARGO_FLAGS="--release"

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CRATE_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"
WORKSPACE_ROOT="$( cd "$CRATE_DIR/../.." && pwd )"

JNI_LIBS_DIR="$CRATE_DIR/kotlin/openpay/src/main/jniLibs"

echo ">>> building $CRATE_NAME ($PROFILE, minSdk=$MIN_SDK) for all Android ABIs"

# ABI → Rust target mapping. cargo-ndk uses the Android ABI names; we
# also need the Rust target triples to locate the output .so files.
declare -A ABI_TO_TRIPLE=(
    [arm64-v8a]=aarch64-linux-android
    [armeabi-v7a]=armv7-linux-androideabi
    [x86_64]=x86_64-linux-android
    [x86]=i686-linux-android
)

# Run cargo-ndk for all four ABIs in one invocation.
(cd "$WORKSPACE_ROOT" && \
    cargo ndk \
        --platform "$MIN_SDK" \
        -t arm64-v8a \
        -t armeabi-v7a \
        -t x86_64 \
        -t x86 \
        build -p "$CRATE_NAME" $CARGO_FLAGS)

# Stage outputs into jniLibs/ABI/.
for abi in "${!ABI_TO_TRIPLE[@]}"; do
    triple="${ABI_TO_TRIPLE[$abi]}"
    src="$WORKSPACE_ROOT/target/$triple/$PROFILE/$SO_NAME"
    dst_dir="$JNI_LIBS_DIR/$abi"
    if [[ ! -f "$src" ]]; then
        echo "missing: $src" >&2
        exit 1
    fi
    mkdir -p "$dst_dir"
    cp "$src" "$dst_dir/"
    echo "    staged $abi  $(ls -la "$dst_dir/$SO_NAME" | awk '{print $5}') bytes"
done

echo ""
echo "=========================================================="
echo "Built: $JNI_LIBS_DIR"
echo ""
echo "Open the kotlin/ directory in Android Studio or run:"
echo "    cd $CRATE_DIR/kotlin && ./gradlew :openpay:assembleRelease"
echo ""
echo "Drop the resulting AAR (or include the openpay module as a"
echo "Gradle subproject) in your Android app to depend on the"
echo "dev.openpay package."
echo "=========================================================="
