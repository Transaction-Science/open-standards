#!/usr/bin/env bash
# Build the op-wasm crate via wasm-pack for the target of choice.
#
# Usage:
#   bash scripts/build-wasm.sh                 # default: --target web
#   bash scripts/build-wasm.sh --target nodejs # Node.js / CommonJS
#   bash scripts/build-wasm.sh --target web    # browsers via ESM
#   bash scripts/build-wasm.sh --target bundler   # webpack / rollup
#   bash scripts/build-wasm.sh --target no-modules # legacy script-tag inline
#
# Prerequisites:
#   - cargo install wasm-pack
#   - rustup target add wasm32-unknown-unknown
#
# Output:
#   pkg/op_wasm.js          — generated JS shim
#   pkg/op_wasm_bg.wasm     — compiled wasm binary
#   pkg/op_wasm.d.ts        — TypeScript type definitions
#   pkg/op_wasm_bg.wasm.d.ts — additional type information
#   pkg/package.json        — npm-publishable manifest
#   pkg/README.md           — copied from crate-level README

set -euo pipefail

TARGET="web"
PROFILE="--release"
ENABLE_PANIC_HOOK=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --target)
            TARGET="$2"
            shift 2
            ;;
        --dev)
            PROFILE="--dev"
            shift
            ;;
        --release)
            PROFILE="--release"
            shift
            ;;
        --panic-hook)
            ENABLE_PANIC_HOOK="--features console-panic-hook"
            shift
            ;;
        *)
            echo "unknown flag: $1" >&2
            echo "usage: $0 [--target web|nodejs|bundler|no-modules] [--dev|--release] [--panic-hook]" >&2
            exit 2
            ;;
    esac
done

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
CRATE_DIR="$( cd "$SCRIPT_DIR/.." && pwd )"

echo ">>> building op-wasm (target=$TARGET, profile=${PROFILE#--})"

# wasm-pack is invoked from inside the crate directory.
cd "$CRATE_DIR"

# wasm-pack manages the whole chain: cargo build --target
# wasm32-unknown-unknown → wasm-bindgen → wasm-opt → generated JS shim
# + .d.ts + package.json.
wasm-pack build \
    "$PROFILE" \
    --target "$TARGET" \
    --out-dir pkg \
    --out-name op_wasm \
    $ENABLE_PANIC_HOOK

echo ""
echo "=========================================================="
echo "Output in $CRATE_DIR/pkg/"
echo ""
echo "Consume from JS:"
case "$TARGET" in
    web)
        cat <<'EOF'
    import init, { RustVault, CardData } from './pkg/op_wasm.js';
    await init();
    const vault = new RustVault('checkout');
EOF
        ;;
    nodejs)
        cat <<'EOF'
    const { RustVault, CardData } = require('./pkg/op_wasm.js');
    const vault = new RustVault('checkout');
EOF
        ;;
    bundler)
        cat <<'EOF'
    import { RustVault, CardData } from './pkg/op_wasm.js';
    const vault = new RustVault('checkout');
    // Webpack/Rollup handles wasm loading automatically.
EOF
        ;;
    no-modules)
        cat <<'EOF'
    <script src="./pkg/op_wasm.js"></script>
    <script>
      wasm_bindgen('./pkg/op_wasm_bg.wasm').then(() => {
        const { RustVault, CardData } = wasm_bindgen;
        const vault = new RustVault('checkout');
      });
    </script>
EOF
        ;;
esac
echo "=========================================================="
