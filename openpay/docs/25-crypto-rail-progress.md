# Phase 25 — Crypto rail: stablecoin settlement at sub-cent cost

**Status**: Draft v0.25
**Date**: 2026-05-22

## Why

The mission has always been "bring margin back to the vendors."
Card networks charge 100–300 basis points per transaction. A USDC
transfer on Base or Solana costs **single-cents or fractions of a
cent**. For a $1M/month vendor that's the difference between
$30,000 and $5 in fees. That is the entire point.

Phase 25 adds the crypto / stablecoin rail to OpenPay. First of
three sequenced phases (25 → 26 → 27): crypto first because it's
the most direct path to the margin win, then persistent backends
(deployable), then subscriptions (feature surface).

## What shipped

| # | Item | Where |
|--:|---|---|
| 1 | `RailKind::Crypto` variant in op-core | `op-core/src/rails.rs` |
| 2 | `PaymentMethod::Crypto(CryptoAddress)` variant in op-core | `op-core/src/method.rs` |
| 3 | `CryptoAddress { chain, address }` type with chain-agnostic constructor | `op-core/src/method.rs` |
| 4 | `op-rails-crypto` crate — `CryptoGateway` trait, `CryptoTransferReq` / `CryptoDecision` / `CryptoStatus`, `StableToken` curated catalog | `crates/op-rails-crypto/` |
| 5 | `op-orchestrator::CryptoAdapter` — wraps any `CryptoGateway` as a `RailAdapter` | `op-orchestrator/src/adapters/crypto.rs` |
| 6 | `PolicyRouter::with_crypto_drivers` + `rail_order` carve-out (Crypto is method-exclusive) | `op-orchestrator/src/router.rs` |
| 7 | `DeterministicCryptoGateway` in op-driver-sdk — per-key overrides, amount rules, transport-error mode, history | `op-driver-sdk/src/crypto.rs` |
| 8 | `conformance::run_crypto` harness — `CryptoMissingTxHashOnFinalized`, `CryptoCrossChainAccepted` checks | `op-driver-sdk/src/conformance.rs` |
| 9 | Orchestrator integration: crypto-routed approve + reject flows, all-three-rails conformance | `op-driver-sdk/tests/orchestrator_integration.rs` |
| 10 | `RailKind` codec in op-graph rail_telemetry covers `Crypto` | `op-graph/src/rail_telemetry.rs` |

Workspace at the end of Phase 25:

| Check | Result |
|---|---|
| `cargo build --workspace --all-targets` | **0 errors, 0 warnings** |
| `cargo test --workspace` | **957 passing, 0 failing** (+27 vs Phase 24) |
| `cargo clippy --workspace --all-targets` | **0 warnings** |

Test-count delta: op-rails-crypto +7 (token / status), op-orchestrator
adapters/crypto +7, op-driver-sdk +11 (8 mock + 3 conformance), e2e +2.

## Architecture

```text
                                          on-chain
                                              ▲
   PaymentIntent ──► PolicyRouter ──► CryptoAdapter ──► CryptoGateway ──► chain client
        │                                              (driver-owned)        │
        │                                                                    ▼
        └──Idempotency key flows through unchanged──►       signer / RPC / Fireblocks
```

Three trait layers, mirroring the existing card / A2A pattern:

1. **`CryptoGateway`** (in `op-rails-crypto`) — the only thing
   driver authors implement. Methods: `submit_transfer`,
   `query_status`, plus accessors (`name`, `chain`, `token`,
   `supports`).
2. **`CryptoAdapter`** (in `op-orchestrator`) — wraps any
   `CryptoGateway` as a `RailAdapter` the orchestrator can route
   through. Status → outcome mapping:
   `Finalized → Success`, `Rejected → HardDecline`,
   `Pending|Confirming → SoftFailure`, `Transient → SoftFailure`.
3. **`DeterministicCryptoGateway`** (in `op-driver-sdk`) —
   programmable mock for operator tests + driver-author reference.

## Why no chain SDK dependency

The reference stack does NOT pull in `solana-sdk` or `ethers-rs`.
Three reasons:

1. **Footprint.** `ethers-rs` alone is ~250 crates.
2. **Signing security.** Operators sign with HSMs, KMS, multisig,
   Fireblocks. Baking in a software keystore would be a footgun.
3. **Chain evolution.** New chains and L2s appear quarterly; the
   reference stack stays neutral by exposing only the trait
   surface.

Operators wire their preferred client behind a `CryptoGateway`
impl — a `RpcClient` for Solana, an `EthersProvider` for EVM,
Fireblocks for custody, whatever fits their security posture.

## StableToken catalog

Curated `(chain, contract, decimals, symbol)` constants for the
common deployments:

| Constructor | Chain | Symbol | Decimals |
|---|---|---|---|
| `StableToken::UsdcSolana` | solana | USDC | 6 |
| `StableToken::UsdcBase` | base | USDC | 6 |
| `StableToken::UsdcEthereum` | ethereum | USDC | 6 |
| `StableToken::UsdcPolygon` | polygon | USDC | 6 |
| `StableToken::UsdcArbitrum` | arbitrum | USDC | 6 |
| `StableToken::EurcBase` | base | EURC | 6 |
| `StableToken::PyusdEthereum` | ethereum | PYUSD | 6 |

Drivers **must** re-validate contract addresses against the
issuer's official docs before broadcasting — token contracts have
been upgraded / re-deployed in the past, and minting a transfer to
a stale contract burns funds.

## CryptoStatus taxonomy

```rust
Pending      // submitted, not yet in a block
Confirming   // in a block, below operator confirmation threshold
Finalized    // meets operator finality threshold — settled
Rejected     // chain refused (revert, sim failure, invalid sig)
Transient    // RPC/network — may or may not have broadcast
```

`funds_moved()` is true only for `Finalized`. Operator-acceptable
finality is driver-side (1 confirmation on Solana, 12 on Ethereum
mainnet, varies on L2s).

## Routing changes

`PolicyRouter::rail_order` short-circuits: any
`PaymentMethod::Crypto(_)` routes exclusively to crypto drivers.
No fallback to card or A2A — those rails physically can't fulfill
a crypto destination, so the orchestrator doesn't try.

```rust
let router = PolicyRouter::new(vec![], vec![])
    .with_crypto_drivers(vec!["usdc-base".to_owned(), "usdc-solana".to_owned()]);
```

The chain-level routing (which `(chain, token)` driver) is the
operator's choice — they register one adapter per pair and rely on
`supports()` to filter by destination chain.

## Conformance additions

```rust
let gateway = MyChainGateway::sandbox(rpc_url, signer);
let report = op_driver_sdk::conformance::run_crypto(&gateway);
assert!(report.is_clean(), "{:?}", report.failures);
```

New failure modes:

- `CryptoMissingTxHashOnFinalized` — finalized without a tx_hash;
  callers have no handle for reconciliation.
- `CryptoCrossChainAccepted { gateway_chain, address_chain }` —
  driver accepted a destination on a chain it doesn't service. A
  Solana driver shouldn't broadcast EVM addresses.

## Honest concerns (carry-forward)

- **No on-chain signing.** That's operator-side by design (see
  "Why no chain SDK dependency" above).
- **No bridging.** Cross-chain transfers (USDC@Base → USDC@Solana)
  are out of scope. Operators handle multi-chain by registering
  one driver per `(chain, token)` and routing on destination chain.
- **Confirmation polling is operator-side.** A transfer that
  comes back `Confirming` is reported as such; the operator drives
  the polling loop via `gateway.query_status(...)` and decides
  when depth meets their finality threshold.
- **Fiat ⇄ stablecoin conversion is operator-side.** The intent's
  `Money` flows through unchanged. Operators quoting in USD and
  settling in USDC (1:1 by definition) handle the 2-decimal vs
  6-decimal scaling in their quote layer.
- **Memo support varies by chain.** Solana has the memo program;
  EVM has no standard memo. Drivers either emit an indexed event
  log or skip memos entirely.
- **Reorg handling is operator-side.** If a chain reorgs and a
  previously-finalized tx un-confirms, the operator's polling
  loop is the right layer to detect it and post a compensating
  ledger entry.
- **No L2-specific routing logic.** Drivers register per `(chain,
  token)`; cost-optimization across L2s (route Base when fees are
  low, Arbitrum when not) is a higher-layer concern operators
  build on top.

## Test totals

```
op-rails-crypto             7  (token catalog 3, status predicates 4)
op-orchestrator/crypto      7  (adapter status mapping + idempotency + chain check)
op-driver-sdk crypto mock   8  (default / overrides / amount / transport / xchain / etc.)
op-driver-sdk conformance   3  (deterministic passes, missing tx_hash, cross-chain accept)
e2e integration             2  (crypto routed through Orchestrator, rejection path)
                                                              ----
                                                              +27 net
```

`cargo test --workspace`: **957 passing, 0 failing.**
`cargo build --workspace --all-targets`: **0 warnings.**
`cargo clippy --workspace --all-targets`: **0 warnings.**
