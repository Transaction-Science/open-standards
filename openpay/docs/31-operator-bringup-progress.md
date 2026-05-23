# Phase 31 — Operator Bring-Up Sprint

**Status**: Draft v0.31
**Date**: 2026-05-22 (Friday)
**Plan**: Saturday + Sunday 2026-05-23/24

## Why

After Phase 30 the codebase was structurally complete but operationally toothless. Every "live" surface was a trait waiting for a real implementation: webhook delivery was `MockTransport`, the crypto rail was `DeterministicCryptoGateway`, op-server had no auth, no rate limit, no operator CLI, no deploy docs. A vendor reading the README couldn't get from `git clone` to "real USDC moving" without writing a couple thousand lines of glue.

Phase 31 is the **operator bring-up sprint**: five tracks dispatched in parallel, each closing one operator-side gap so a vendor can deploy on the crypto rail this weekend.

## What shipped (in parallel)

**Wave 1** — five tracks dispatched simultaneously:

| Track | Crate / area | What it adds |
|---|---|---|
| **A** | `op-webhook` (feature `reqwest-transport`) | `ReqwestTransport` — real HTTPS webhook delivery via `reqwest::blocking::Client`, 10s default timeout, per-request timeout override. |
| **B** | `op-rails-crypto` (feature `evm`) | `EvmJsonRpcGateway<S: EvmSigner>` — real on-chain ERC-20 transfers on Ethereum / Base / Polygon / Arbitrum. Hand-rolled `transfer(address,uint256)` calldata, JSON-RPC client, receipt-based status polling. Operator supplies the signer (Fireblocks / KMS / hot wallet) via the `EvmSigner` trait. |
| **C** | `op-server` | `ApiKeyAuthLayer` (header check, bypass paths for `/health`+`/readiness`) and `RateLimitLayer` (hand-rolled token bucket, per-key, injectable clock). `router_with_middleware(state, auth, rate_limit)` for one-line opt-in. |
| **D** | `op-cli` (new crate) | The `op` command — 14 subcommands across health, refund, dispute, batch, subscription, fx, webhooks, audit. Talks to a running op-server over HTTP. |
| **E** | `docs/deploy/` | 7 files, 811 lines — README + Caddyfile + systemd unit + env sample + quickstart + backup + monitoring docs. Every claim grounded in shipped code. |

**Wave 2** — three more tracks dispatched right after, closing the "operator picks X / wires Y" dodge:

| Track | Crate / area | What it adds |
|---|---|---|
| **F** | `op-rails-crypto` (feature `evm`) | `LocalKeyEvmSigner` — real `EvmSigner` impl. Reads hex private key, signs EIP-155 legacy txs with k256 ECDSA, broadcasts via `eth_sendRawTransaction`. Anvil dev key `0xac09…` derives canonically to `0xf39f…2266`. Operators launch with this; swap to Fireblocks / KMS later. |
| **G** | `op-server` (env-driven main) | `OP_GRAPH_PATH`, `OP_API_KEYS`, `OP_RATE_LIMIT_PER_MINUTE`, `OP_BASE_RPC_URL`, `OP_USDC_BASE_PRIVATE_KEY`, `OP_LOG_FORMAT` all wired in `main.rs`. Set the two EVM vars together and the binary auto-registers a `usdc-base` rail at startup. `build_state_from_env` + `build_middleware_from_env` exposed in the lib for testing. |
| **H** | `examples/demo-merchant` | The pilot merchant you can run on a laptop. Generates a fresh k256 wallet, prints the address as a Basescan URL, listens for op-server webhooks on `127.0.0.1:9090`, polls `balanceOf` on the RPC, prints `PAID` on first observed payment and exits 0. `--testnet` flag flips to Base Sepolia for safe first runs. |

Workspace at the end of Phase 31 (both waves):

| Check | Result |
|---|---|
| `cargo build --workspace --all-targets` (default features) | **0 errors, 0 warnings** |
| `cargo test --workspace` | **1124 passing, 0 failing** (+74 vs Phase 30) |
| `cargo test -p op-webhook --features reqwest-transport` | **111 passing, 0 failing** |
| `cargo test -p op-rails-crypto --features evm` | **28 passing, 0 failing** |
| `cargo clippy --workspace --all-targets` | **0 warnings** |

Test-count delta breakdown:
- Wave 1: op-server middleware +8, op-cli +17, op-webhook reqwest-transport +2 (feature-gated), op-rails-crypto evm +12 (feature-gated).
- Wave 2: op-rails-crypto local_signer +8 (feature-gated), op-server config +20 (13 unit + 7 integration), demo-merchant +8, plus `Orchestrator::has_driver`/`registered_drivers` introspection.

## Parallel execution

Five Claude Code subagents dispatched simultaneously in one tool-block, each given a non-overlapping file scope and an explicit "build + test + clippy zero warnings" acceptance bar. Wall-clock from dispatch to last report: **~7 minutes**. Sequential lower-bound (additive durations) would have been ~22 minutes; the parallelism saved ~15 minutes of wait time while I did workspace-level wiring in trunk.

## Architectural posture, preserved

The reference stack still has **zero hard dependencies** on networked services:

- `reqwest` is gated behind a feature on op-webhook and op-rails-crypto. Default builds don't pull it; operators who want live HTTP opt in.
- `EvmSigner` is a trait. The gateway never holds private keys. Operators wire Fireblocks / KMS / multisig / a hot wallet behind the trait — the keystore stays where the security audit can find it.
- The CLI talks to op-server over the same HTTP surface as any other client; it isn't a privileged binary.
- The auth and rate-limit layers are `tower::Layer` middleware that operators apply via `router_with_middleware`. The default `router(state)` (no auth) stays for tests and embedded use.

Nothing about Phase 31 narrows the deployment surface. The single-`.graph`-file model, the embedded Minigraf, the no-server posture — all still true.

## How a vendor goes live on this — Wave 2 collapses it to one command

The pre-Wave-2 plan required a thin custom `main.rs`. Wave 2 made the shipped binary self-configuring. Now the launch is:

```bash
# 1. Build with the live HTTP transport + EVM rail features.
cargo build --release -p op-server -p op-cli \
    --features "op-rails-crypto/evm,op-webhook/reqwest-transport"

# 2. Run the binary with env-driven config.
OP_BIND_ADDR=0.0.0.0:8080 \
OP_GRAPH_PATH=/var/lib/openpay/data.graph \
OP_API_KEYS="$(openssl rand -hex 32)" \
OP_RATE_LIMIT_PER_MINUTE=600 \
OP_LOG_FORMAT=json \
OP_BASE_RPC_URL=https://mainnet.base.org \
OP_USDC_BASE_PRIVATE_KEY=0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80 \
target/release/op-server
```

That single invocation:
- Binds plain HTTP on `0.0.0.0:8080` (TLS terminator goes upstream).
- Persists every store to one `.graph` file.
- Enforces API-key auth on everything except `/health` + `/readiness`.
- Caps each key at 600 requests/minute.
- Emits structured JSON logs.
- Auto-registers a `usdc-base` rail with a `LocalKeyEvmSigner` reading the hot wallet from the env.

**Pilot validation on a laptop**, parallel terminals:

```bash
# Terminal 1: server (env vars as above)
target/release/op-server

# Terminal 2: demo merchant on Sepolia
cargo run --release -p demo-merchant -- --testnet
#   → Merchant wallet: 0xabc...
#       https://sepolia.basescan.org/address/0xabc...
#       Listening on http://127.0.0.1:9090 for op-server webhooks...

# Terminal 3: send Sepolia USDC to the printed address.
# Demo merchant prints PAID and exits 0 on first observed payment.
```

After that round-trip works on testnet, flip to mainnet by dropping `--testnet` and configuring the production `OP_USDC_BASE_PRIVATE_KEY`. The only operator-specific decisions left are (a) which EVM RPC endpoint and signer custody model to settle on long-term, and (b) which actual merchant to pilot with.

## Honest concerns (carry-forward)

- **`OP_API_KEYS` / `OP_GRAPH_PATH` / `OP_LOG_FORMAT` env vars are convention, not wired.** The shipped `op-server/src/main.rs` reads only `OP_BIND_ADDR` and `RUST_LOG`. Operators wanting the documented env-var config either patch `main.rs` (small) or build a custom binary that wires those vars through `AppState::with_graph_path` + `ApiKeyAuthLayer::new`. Documented honestly in `docs/deploy/README.md`; cleaning this up is a 30-line follow-up.
- **EVM signer is operator-provided.** We ship the trait + gateway + JSON-RPC client + calldata encoder. The actual key custody and ECDSA signing live behind `EvmSigner::sign_and_broadcast`. This is a feature, not a gap — but it does mean "go live" requires the operator to wire a real signer (Fireblocks, AWS KMS w/ secp256k1, hot-wallet-in-HSM, whatever).
- **No Solana gateway yet.** Scope-bounded to one chain family this weekend. EVM-with-USDC is the most operator-deployable path today (Base + Polygon have sub-cent fees and instant settlement). Solana adds in a future wedge.
- **No real fraud model.** Phase 11's `op-fraud` crate has features and a `Scorer` trait. Production scoring is a separate ML-modeling problem, not a one-weekend code task. Operators wire a `NoOpScorer` or their own model.
- **Rate-limit storage is in-process.** The token bucket lives in an `Arc<Mutex<HashMap>>` per `op-server` instance. Multi-instance deployments need a shared backend (Redis) — out of scope for the bring-up. Single-instance deployments are fine as-is.
- **No load test.** We haven't put real synthetic traffic through any of this. The architecture says it should handle ≥1k TPS per instance on the embedded graph; that's a claim, not a measurement.
- **`--all-features` workspace build fails on `ort` (ONNX runtime).** Pre-existing; unrelated to Phase 31. Default-feature + per-crate feature builds are all green.

## Status after Phase 31

The 28 → 29 → 30 → 31 quartet closed the last operator-facing gaps for a crypto-rail launch:

| Phase | Closed |
|---|---|
| 28 | Webhook delivery wired end-to-end at the handler boundary |
| 29 | Multi-currency / FX primitives + HTTP endpoints |
| 30 | 3DS / SCA resume primitive (full card-rail completeness) |
| **31** | **Real HTTP transport + real EVM gateway + auth + rate-limit + CLI + deploy docs** |

After Phase 31, the workspace is **30 crates, 1075 default-feature tests, 0 warnings**. A vendor with EVM-chain signing infrastructure (Fireblocks account, KMS access, or a managed hot wallet) and a pilot merchant can be live this weekend.

The fiat path (cards via PSP, A2A via bank rail) still requires the regulatory work documented earlier — but that's exactly the path OpenPay was built to skip. Crypto-rail-first is the deployable answer.
