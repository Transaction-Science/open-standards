# `eoc-rs` — EOC reference implementation in Rust

Apache-2.0 reference implementation of the **Energy-Optimized Compute** specification (CC-BY-4.0 spec lives in [`../spec/`](../spec/)).

EOC is a four-stage memoizing cascade. Every query is tried at each stage in turn and the first that can answer wins. The four stages, cheapest to most expensive, are:

1. **Cache** (`eoc-cache`) — content-addressed LRU. Cache hits are nearly free.
2. **Key-value** (`eoc-kv`) — exact key match plus cosine-similarity embedding match.
3. **Graph** (`eoc-graph`) — triple-store retrieval. Reference impl is a small in-memory matcher; production deployments swap in a real graph backend driven by DCY (EOC-4).
4. **Neural** (`eoc-neural`) — last-resort inference. Reference impl ships an `EchoBackend`; production deployments wire in llama.cpp / Ollama / Anthropic API / etc.

Joule cost is the unit of accounting. The `eoc-meter` crate reads cumulative micro-joules from whatever hardware counter is available (Linux RAPL, macOS `powermetrics`, NVIDIA NVML behind the `cuda` feature) and falls back to a `StubCounter` everywhere else — including WASM.

## Workspace layout

| Crate          | Purpose                                                      |
|----------------|--------------------------------------------------------------|
| `eoc-core`     | Types: `Query`, `Response`, `Stage`, `JouleCost`, `Receipt`. |
| `eoc-cache`    | Stage 1 — LRU cache + the `Stage` trait.                     |
| `eoc-kv`       | Stage 2 — KV + embedding-similarity lookup.                  |
| `eoc-graph`    | Stage 3 — in-memory triple store.                            |
| `eoc-neural`   | Stage 4 — neural inference trait + `EchoBackend`.            |
| `eoc-meter`    | Hardware energy counters with stub fallback.                 |
| `eoc-cascade`  | Glue: walks the four stages, attributes joule cost.          |
| `eoc`          | Reference CLI.                                               |
| `eoc-wasm`     | `wasm-bindgen` wrapper — browser-runnable cascade.           |

## Running

```sh
cargo test --workspace
cargo clippy --workspace --all-targets
cargo run --bin eoc -- query "what is 2+2"
cargo run --bin eoc -- meter
```

Example output:

```
$ cargo run --bin eoc -- query "what is 2+2"
stage  : kv
payload: 4
cost   : 50 µJ (estimated)
receipt: <64-char hex>
```

## CUDA / NVML

NVIDIA NVML support is feature-gated and **off by default** so the workspace builds without `libnvidia-ml`. To enable:

```sh
cargo build -p eoc-meter --features cuda
```

## WASM

The cascade compiles to `wasm32-unknown-unknown`. The meter does not — RAPL, NVML, and `powermetrics` are all absent in the browser, so WASM builds use `StubCounter` and report estimated joule cost.

```sh
rustup target add wasm32-unknown-unknown
cargo build -p eoc-wasm --target wasm32-unknown-unknown --release
wasm-bindgen target/wasm32-unknown-unknown/release/eoc_wasm.wasm \
    --out-dir ./pkg --target web
```

Then from JavaScript:

```js
import init, { WasmCascade } from "./pkg/eoc_wasm.js";
await init();
const cascade = new WasmCascade();
const r = await cascade.resolve("ping");
console.log(r); // { stage: "kv", payload: "pong", microjoules: 50, ... }
```

## License

Apache-2.0 for the Rust source in this workspace. The EOC specification text in [`../spec/`](../spec/) is CC-BY-4.0. See [`../LICENSE`](../LICENSE).
