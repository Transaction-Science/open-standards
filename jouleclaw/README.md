# JouleClaw — Energy-Optimized AI Runtime

The substrate for AI that costs Joules, not tokens.

**Capability per Joule, not capability per parameter.**

## What JouleClaw is

JouleClaw is a pure-Rust, omni-modal AI runtime + harness + agentic layer
governed by Transaction Science as an open standard (5th in the open-standards
quartet, alongside [OpenPay](../openpay/), [Smart Byte](../smart-byte/),
[EOC](../eoc/), and [WAI](../wai/)).

Every operation JouleClaw performs walks an explicit energy-tiered cascade,
top to bottom, stopping at the first stage that closes the query:

| Tier | Name        | Cost class       | What runs here                                            |
|------|-------------|------------------|-----------------------------------------------------------|
| L0   | **Cache**   | picojoules       | Content-addressed cache hits                              |
| L1   | **Lawful**  | nanojoules       | Deterministic, pre-compiled primitive (text/code only)    |
| L2   | **Embed**   | sub-millijoules  | Matryoshka embeddings + nearest-neighbor + hybrid search  |
| L3   | **Model**   | joules           | Local SSM / ternary / multimodal / diffusion / etc.       |
| L4   | **Wire**    | tens of joules   | Remote frontier RPC (escape hatch only)                   |

L1 (Lawful) is only meaningful for text and code modalities. For image, audio,
video and 3D generation, the cascade collapses to L0 / L2 / L3 / L4 — the
energy savings come from caching plus dispatch to the cheapest model that
meets the quality bar.

## Why JouleClaw exists

The mainstream "agentic AI" ecosystem treats inference as the entry point.
Every state shift, every tool call, every routing decision runs through a
70-billion-parameter stochastic engine, burning kilojoules to do work that
deterministic compute or a hot cache could resolve for microjoules.

JouleClaw inverts the default. Inference is the **last resort**, not the
first. Frozen model weights are **less trustworthy than current world state**;
fresh, provenance-stamped retrieval beats synthesis from training cuts.
Every operation is metered in microjoules where the hardware permits it, and
in millijoules where it doesn't — the spec is honest about the floor on every
platform.

The thermodynamic circuit breaker is the safety mechanism. Set an energy
budget; when measured consumption exceeds the budget, the breaker trips and
the operation halts. Hallucination loops cannot survive an honest joule
ledger.

## What's here

| Path                              | Contents                                                    |
|-----------------------------------|-------------------------------------------------------------|
| [`SPEC.md`](SPEC.md)              | JouleClaw v1 spec — cascade tiers, energy contract, receipt |
| [`CHARTER.md`](CHARTER.md)        | Stewardship pattern (shared across open-standards)          |
| [`jouleclaw-rs/`](jouleclaw-rs/)  | Apache-2.0 Rust reference implementation                    |
| [`spec/`](spec/)                  | Tier specifications, energy provenance, receipt format      |
| [`conformance/v1/`](conformance/) | Signed wire vectors for cross-implementation verification   |

## Reference implementation

`jouleclaw-rs/` is the Apache-2.0 reference implementation in Rust. It
runs on Apple Silicon (M3/M4/M5 with UMA + Metal 4 Tensor APIs), AMD Strix
Halo / Ryzen AI Max, Intel Lunar Lake, NVIDIA Jetson, and discrete
NVIDIA / AMD GPUs.

It is omni-modal: text, code, vision, audio, video, image diffusion, 3D
Gaussian splatting, and cross-modal fusion all dispatch through the same
energy-tiered cascade.

## Status

v0.1.0 reference implementation in active development. The spec is draft
status. Cross-implementation conformance vectors will land with v1.0.0.

## License

- **Code:** Apache-2.0 (see [`LICENSE`](LICENSE))
- **Spec text:** CC-BY-4.0 (see [`spec/LICENSE`](spec/LICENSE))
