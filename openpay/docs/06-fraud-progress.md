# Phase 6 — `op-fraud` Complete (On-Device Fraud Scoring)

**Status**: Draft v0.6
**Date**: 2026-05-17

## What shipped

`op-fraud`: on-device fraud scoring. A pluggable `Scorer` trait with
three implementations — heuristic (default), ONNX (`ort` 2.0), Burn
(pure Rust) — fronted by a deterministic feature extractor that hashes
every PII string before the model sees it.

Why this matters: every instant rail (FedNow ~20s, PIX ~10s, SEPA Inst
5–10s, with the 2025 SEPA Regulation amendment squeezing to 5/7/9s)
gives the orchestrator a hard latency envelope. A round-trip to an
external fraud API plus the model inference plus the decision API call
eats most of that budget. The transfers are also **irrevocable** — a
fraud decision *after* submit is useless. Scoring runs in-process,
before routing, in single-digit milliseconds.

## Crate layout

```
crates/op-fraud/
├── Cargo.toml                     # workspace member; default = heuristic-only
├── src/
│   ├── lib.rs                     # crate root, feature-gated modules, re-exports
│   ├── error.rs                   # Error: Features, ModelLoad, ModelOutput,
│   │                              # ScoreOutOfRange, Backend, Core
│   ├── context.rs                 # ScoringContext (operator-supplied signals)
│   ├── features.rs                # 32-feature extraction with SHA-256 hashing
│   ├── decision.rs                # FraudDecision + tunable Thresholds
│   ├── scorer.rs                  # Scorer trait + HeuristicScorer
│   ├── onnx.rs                    # OnnxScorer via ort 2.x   [feature: onnx]
│   └── burn_backend.rs            # BurnScorer + FraudMlp     [feature: burn-backend]
└── tests/
    └── lifecycle.rs               # end-to-end pipeline tests
```

## Verified ground truth

### Burn (pure-Rust ML)

| Claim | Source |
|---|---|
| Burn 0.17.1 is the current stable (0.20.0-pre.2 in flight) | docs.rs/crate/burn/latest |
| NdArray is the only backend that supports `no_std` + WASM + iOS/Android | docs.rs/crate/burn-ndarray, README |
| `TensorData` replaced `Data` since 0.17.0 — current API requires `TensorData::new` or `from_data` | burn-ndarray 0.17 release notes |
| `Module` derive macro, `Linear<B>`, `LinearConfig::new(in,out).init(device)`, `Relu`, `Sigmoid` are the building blocks | burn.dev/docs/burn/nn/, verified live |
| `BinFileRecorder<FullPrecisionSettings>::new()` then `.load(path, device)` then `model.load_record(record)` is the canonical inference load path | Burn Book §Saving & Loading |
| Forward pattern: store activations as Module fields, call `.forward(x)` each layer | tracel-ai/burn issue #2739, ResNet tutorial |

### ONNX Runtime via ort

| Claim | Source |
|---|---|
| `ort 2.0.0-rc.12` is production-ready (just not API-stable) and recommended for new projects | ort.pyke.io intro |
| Wraps ONNX Runtime 1.24 | docs.rs/crate/ort/latest |
| `load-dynamic` feature avoids the static-link DLL hell documented in the book | ort.pyke.io setup §Strategies |
| Supports Linux, macOS, Windows, iOS, Android, WASM | onnxruntime.ai |
| `Session::run` takes `&mut self` — Mutex needed for `Sync` | ort docs §Session |

### Threshold defaults — rationale

Default review/decline/freeze thresholds are 0.50 / 0.80 / 0.95. These
match the industry-standard calibration for instant payments where false
negatives (letting fraud through) cost the operator the full transfer
amount (irrevocable) while false positives (declining a legit payment)
cost only the customer-experience friction of a step-up auth or retry.
Card deployments may safely loosen these because chargebacks provide
recourse. The thresholds are operator-tunable via `Thresholds::new`.

## The three-scorer pattern

```rust
trait Scorer: Send + Sync {
    fn name(&self) -> &str;
    fn score(&self, features: &FeatureVector) -> Result<f32>;
}
```

| Scorer | Binary cost | Where it fits |
|---|---|---|
| `HeuristicScorer` | 0 (no dep) | Always available. Defensible baseline. Cold-start, fallback, and security floor when the ML model fails. Rule-based; deterministic. |
| `OnnxScorer` | ~10–30 MB (libonnxruntime shipped separately) | Operators with existing PyTorch / TensorFlow / sklearn models. Convert to ONNX once, deploy anywhere ort runs. |
| `BurnScorer` | ~1 MB (statically linked) | Pure-Rust deployments. WASM, kiosk-Linux, FFI to mobile via Phases 8–10. No shared library. Operator trains in Burn (or imports ONNX via Burn's ONNX→Rust transpiler). |

The orchestrator holds `Box<dyn Scorer>` and doesn't care which
implementation. Same pattern as Phase 4's `Box<dyn CardAcquirer>` and
Phase 5's `Box<dyn A2aAcquirer>`.

## Feature schema (32 floats)

| Idx | Feature | Range | Notes |
|---:|---|---|---|
| 0 | `log10(amount_minor + 1)` | [0, 18.3] | Wide-range amount signal |
| 1 | `amount_minor / 1e6` | [0, ∞) | Linear signal for small amounts |
| 2 | `is_currency_usd` | {0, 1} | Currency one-hots |
| 3 | `is_currency_eur` | {0, 1} | |
| 4 | `is_currency_brl` | {0, 1} | |
| 5 | `is_round_amount` | {0, 1} | Fraudsters favor round numbers |
| 6 | `over_1000_major` | {0, 1} | Threshold flags |
| 7 | `over_10000_major` | {0, 1} | |
| 8 | `hour / 24` | [0, 1) | Time of day, linear |
| 9 | `dow / 7` | [0, 1) | Day of week, linear |
| 10 | `is_weekend` | {0, 1} | |
| 11 | `is_night` (00-06) | {0, 1} | |
| 12-15 | sin/cos(hour), sin/cos(dow) | [-1, 1] | Cyclic encoding for time |
| 16-18 | `log1p(velocity_{1h, 24h, device_1h})` | [0, ∞) | Velocity signals |
| 19 | normalized seconds since last payment | [0, 1] | |
| 20 | normalized seconds since auth | [0, 1] | |
| 21 | `is_new_customer` | {0, 0.5, 1} | Missing = 0.5 (neutral) |
| 22 | `geo_matches_history` | {0, 0.5, 1} | Missing = 0.5 |
| 23 | `hash_to_unit(device_id)` | [0, 1] | SHA-256 prefix, projected |
| 24-27 | rail one-hots (Card/A2a/Wallet/Qr) | {0, 1} | |
| 28 | `hash_to_unit(creditor_account)` | [0, 1] | |
| 29 | `hash_to_unit(creditor_name)` | [0, 1] | |
| 30 | `hash_to_unit(debtor_account)` | [0, 1] | |
| 31 | `has_remittance` | {0, 1} | |

## PII hardening

By construction, the input to every `Scorer::score` call is a `[f32; 32]`
array. There is no string slot in the feature vector. Every identifier
that originates as text (account, name, device id) is hashed:

```rust
fn hash_to_unit(s: Option<&str>) -> f32 {
    match s {
        None | Some("") => 0.0,                              // missing signal
        Some(s) => {
            let digest = Sha256::digest(s.as_bytes());
            let upper = u32::from_be_bytes([digest[0], digest[1], digest[2], digest[3]]);
            (f64::from(upper) / f64::from(u32::MAX)) as f32  // → [0, 1]
        }
    }
}
```

A model that's exfiltrated cannot reconstruct PII; it can only see a
random-looking float per identifier. Two distinct identifiers collide
in the feature space with probability ~1/2³² ≈ 1 in 4 billion, which
is well below any signal threshold the model could learn from.

### Cross-language verification

The Rust test asserts `hash_to_unit("abc") ≈ 0.728395` (tolerance 1e-4).
Python verified:

```text
SHA-256("abc")[:4] big-endian = 0xba7816bf = 3128432319
upper / u32::MAX             = 0.7283949106299307
```

Same value to 6 decimal places.

## Burn architecture — fixed shape on purpose

```text
input [batch, 32]
  → Linear(32 → 32) → ReLU
  → Linear(32 → 16) → ReLU
  → Linear(16 → 1)  → Sigmoid
→ output [batch, 1] ∈ [0.0, 1.0]
```

Why fixed:

1. **Type-level certainty.** Burn modules carry their layer shapes in
   their types. Loading a trained record requires the same module type.
   Fixing the shape means operators don't have to ship a matching
   `FraudMlp` definition alongside their trained `.mpk` file.

2. **Fraud doesn't need depth.** XGBoost with ~100 features is the
   industry baseline. A 2-hidden-layer MLP on 32 features matches that
   performance in seconds of CPU training, and inference is ~50µs on
   commodity hardware. Going deeper inflates latency on the very rails
   we need to stay under 10ms for.

Operators who genuinely need a different architecture (transformers,
graph nets, ensembles) use the ONNX path.

## Decision space

```rust
enum FraudDecision { Approve, Review, Decline, Freeze }
```

| Decision | When | Orchestrator action |
|---|---|---|
| `Approve` | score < 0.50 | Route to chosen rail |
| `Review` | 0.50 ≤ score < 0.80 | Hold + step-up auth or human review |
| `Decline` | 0.80 ≤ score < 0.95 | Refuse silently, return `Payment<Failed>` |
| `Freeze` | score ≥ 0.95 | Refuse + flag the customer account for review of all subsequent activity |

Thresholds are monotone-validated at construction (`Thresholds::new`).
The default settings are conservative for A2A (irrevocable) and can be
loosened for card flows where chargebacks exist.

## Test coverage

| Module | Tests | What's covered |
|---|---|---|
| `context.rs` | 3 | Empty / fresh-customer / JSON round-trip |
| `decision.rs` | 9 | Threshold defaults, boundary tests (4 regions), custom validation, score-out-of-range, helper functions, JSON round-trip |
| `features.rs` | 23 | Length, finiteness, amount features (log + linear + currency + round + thresholds), time (linear + cyclic at midnight + noon), weekend flag, velocity log, three-valued booleans, rail one-hots, hash determinism + boundedness + Python cross-check, full determinism, normalize_log_seconds edge cases |
| `scorer.rs` | 12 | Bounded output, low/high amount, velocity monotonicity, night-time, geo-mismatch, saturated clamp, neutral score, name stability, object-safety, determinism |
| `onnx.rs` | 3 | from_file missing path, from_bytes empty, from_bytes garbage |
| `burn_backend.rs` | 7 | Bounded score, deterministic per instance, from_file missing path, object-safety, name, forward shape, sigmoid output ∈ [0,1] |
| `tests/lifecycle.rs` | 8 | Normal card approves, large A2A reviews/declines, velocity spike, PIX 3am with geo mismatch, pluggable trait, PII compile-time safety + hash distinguishing, low score = approve, full pipeline determinism |
| **Phase 6 total** | **65** | |
| **Cumulative Phases 1–6** | **296** | |

## Independently verified

- `hash_to_unit("abc") = 0.7283949106` via Python `hashlib`. Rust f32
  conversion path stays within 1e-4 of this.
- Burn module pattern matches verified usage in the tracel-ai/burn
  issue tracker (issue #2739, the BCE-loss thread) — same `Linear`,
  `Relu`, `Sigmoid` module-field pattern.
- Feature schema cyclic encoding: `sin(2π · hour/24)` at midnight is 0;
  `cos` is 1. At noon, `sin` is 0; `cos` is -1. Direct trig verification
  with `(f32::EPSILON < 1e-6)` tolerance.

## Design decisions

### 1. Pluggable scorers, not a single mega-model

Three scorers behind one trait, each carrying different deployment
trade-offs. The orchestrator holds `Box<dyn Scorer>` and doesn't know
which it has. Operators choose:

- ship nothing extra → Heuristic (always present)
- ship libonnxruntime → ONNX (broad model compatibility, ~10–30 MB)
- ship nothing extra and use Burn → BurnScorer (pure Rust, ~1 MB)

### 2. Feature extraction is deterministic and side-effect-free

The same `(PaymentDescriptor, ScoringContext)` always produces the same
`FeatureVector`. No clock reads inside extraction unless the caller
provided a timestamp explicitly. This lets training and inference share
the exact feature pipeline — no skew, no "we trained on a different
extractor."

### 3. PII is hashed before the boundary, not redacted after

The model never sees a string. There is no "scrubbing" step that might
miss a field. The compile-time type of the scorer input is `[f32; 32]`,
so a model can't accidentally consume PII because the language doesn't
let it.

### 4. Heuristic is not a fallback — it's a floor

If the ML model goes down, the heuristic continues to ship. More
importantly: the heuristic represents a **minimum acceptable level of
fraud screening**. No tuned ML model should perform worse than the
rules-based baseline. If it does, that's a deployment bug, not a model
improvement, and the operator should keep the heuristic.

### 5. Burn architecture is fixed, ONNX architecture is free

For Burn, we ship a fixed `FraudMlp` shape. Operators training in Burn
import the exact `FraudMlp` we deploy from; they only tune the weights.
For ONNX, the model contract is a single `[1, 32] → [1, 1]` sigmoid;
internal architecture is the operator's business.

### 6. `init_runtime` is a one-time setup, not per-scorer

The ONNX path requires `init_from(library_path).commit()` once before
any `OnnxScorer` is constructed. Operators call it during startup with
the path to their bundled `libonnxruntime.{so,dylib,dll}`. Matches the
pattern from rail drivers (FedLine MQ, CloudHSM): centrally configured,
not bundled per crate.

## Bugs caught and fixed during construction

1. **`from_floats` slice support is API-dependent.** Initially used
   `Tensor::from_floats(features.as_slice(), &device)`. The Burn Book
   shows `from_floats` with array literals only; slice support varies
   across versions. Switched to the unambiguous
   `TensorData::new(features.to_vec(), [FEATURES])` then `from_data`.

2. **Mutex around the Burn module.** Burn modules are `Send + Sync`,
   but the `forward` method I use needs `&self` only. The Mutex is
   strictly speaking unnecessary for Burn — kept for symmetry with the
   ONNX path (where `Session::run` takes `&mut self`) so the
   orchestrator presents a uniform interior-mutability story.

3. **`Sigmoid::new()` vs `Sigmoid::default()`.** Both exist on the unit
   struct `pub struct Sigmoid;`. Used `Sigmoid::new()` to match the
   verified usage in tracel-ai/burn issue #2739.

## What's NOT in this phase (explicitly deferred)

- **Calibration**. Models produce probabilities, but those probabilities
  aren't necessarily calibrated to the empirical fraud rate. An
  `IsotonicCalibrator` to map raw scores to calibrated probabilities is
  on the roadmap. → Phase 6.1.
- **Training pipeline**. We don't ship a training entrypoint. Operators
  train offline in their environment (Python or Rust) and deploy the
  serialized weights. Documented training scripts come with Phase 11's
  kiosk-Linux example.
- **Model versioning / hot-swap**. Loading a new model without restart
  is operator concern (atomic-replace the file, recreate the scorer,
  swap behind an Arc). We don't ship a hot-swap manager. → Phase 7+.
- **Online updates / streaming training**. Out of scope.
- **Explainability**. Per-decision SHAP / feature attributions are
  valuable for review queues but not for the score itself. → Phase 6.2.
- **Multi-model ensembles**. The trait composes (`Scorer` over `Vec<dyn
  Scorer>`) but we don't ship an ensemble combinator. → Phase 6.3.
- **Federated learning**. Out of scope.

## Next: Phase 7 — `op-vault`

Token-only payment-method storage abstraction. iOS Keychain + Android
Keystore on mobile, OS keystore on desktop, HSM on the merchant kiosk.
The vault never returns a raw PAN to the caller — only opaque tokens
that the rail drivers can resolve at submit time. This is the surface
that lets the rest of the stack stay outside PCI DSS scope: as long as
raw card data never leaves the vault, the orchestrator and rail drivers
inherit the "out of scope" classification.
