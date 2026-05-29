# ARL — AI Readiness Level

A universal, vendor-neutral standard for measuring the readiness of an AI system to perform a defined task in a defined context.

ARL is to AI what the **Technology Readiness Level** scale (NASA, 1974) is to technology: a portable yardstick that says *how ready* a thing is, independent of who built it, what it is made of, or which lab is making the claim. It is tied to no model, no runtime, no vendor. Any AI system — a frontier API, a local model, an agent harness — gets an ARL score, and the score means the same thing everywhere because each axis is anchored in math or physics that does not change across time, languages, or political regimes.

Every ARL score has **four required parts**. None summarizes the others.

1. **Validation Depth (1–9)** — how thoroughly the readiness claim has been tested. *Statistics.*
2. **Convergence Class (A–E)** — how stochastic the system is on the certified task. *Stochastic process theory.*
3. **Energy Profile (joules)** — training-amortized, per-task inference, total cost of operation. *Thermodynamics.*
4. **Security Class (S0–S4)** — adversarial robustness, output integrity, confidentiality, auditability. *Information theory and cryptography.*

A score is assigned to a specific *system + task + context* on specific hardware. Change any of them and you score again. The four axes cover what the system **is**, what it **does**, what it **costs**, and how it **holds up under attack**.

The teeth of the framework are the **cross-axis gates**: a high validation depth is unreachable without a matching convergence class and security class (ARL ≥ 4 ⇒ Class D + S1; ≥ 6 ⇒ C + S2; ≥ 8 ⇒ B + S3; 9 ⇒ S4), refusing to disclose energy caps the score at ARL 3, refusing security methodology caps it at S0, and the published methodology must predate the published claim. A claim that omits any of the four parts is incomplete by definition.

## Contents

- [`ARL.md`](ARL.md) — the four-axis readiness scoring framework.
- [`ARL-S.md`](ARL-S.md) — the ARL Sandbox: the testing environment, isolation tiers, telemetry, attestation, and replay requirements inside which scores are measured. Composes named open-source components (seccomp/gVisor/Firecracker for isolation; RAPL/NVML/Zeus for energy; Ed25519-over-JCS + SHA-256 for attestation) — no component is mandated by brand, only by capability.
- [`LEXICON.md`](LEXICON.md) — the controlled vocabulary for ARL claims, so a stated score has one meaning.
- [`RATIONALE.md`](RATIONALE.md) — *Why ARL*: the trends in AI that make a shared readiness yardstick necessary, the reasoning behind the four axes, and how ARL composes with the governance frameworks and benchmarks already in use worldwide. Background, not part of the normative spec.

The first three documents are self-contained: a reader can understand and use the framework with only those. `RATIONALE.md` is the *why* behind them.

## Status

Document stack **v1.3** (May 2026). The standalone, system-agnostic reference implementation has begun: [`arl-rs/`](arl-rs/) ships **`arl-core`** — the four-axis claim model with every cross-axis gate enforced (a high validation depth is unreachable without matching convergence and security; energy non-disclosure caps the score at ARL 3; security non-disclosure caps the class at S0; methodology must predate the claim) and the LEXICON controlled vocabulary enforced (terms with no single operational definition, such as *AGI* or *consciousness*, cannot anchor a claim because they cannot be measured; terms with a measurable operational sense are flagged to confirm that sense is intended — the lexicon takes no position on the terms themselves). 22 tests, zero coupling to any runtime. **`arl-sandbox`** ships the ARL-S Supervisor core: the evaluation-session model, the tier↔ARL↔telemetry structural gates (isolation must match the ARL range, all three telemetry categories required above Tier 0, Tier 2/3 must be replayable, detected tampering invalidates), and the **attestation** — Ed25519 (RFC 8032) over JCS-canonicalized JSON (RFC 8785) with SHA-256 — so a measurement is tamper-evident and content-addressable, verifiable by anyone with no trust in the issuer. It also ships a trait-based **`Supervisor`** orchestration layer: drive a `Harness` (the task runner — reference `EchoHarness` needs no model), sample a `PhysicalTelemetrySource` (energy; `NullPhysicalSource` / `FixedPhysicalSource`, with RAPL+NVML as the Linux deployment slot), assemble the session with the telemetry categories actually captured, and sign it — `evaluate()` returns a signed, validated `Evaluation`. An environment with no energy meter honestly reports physical telemetry absent, which caps the achievable tier. 20 tests. (Launching the real isolation tiers — seccomp/gVisor/Firecracker — and reading RAPL/NVML/TPM is OS- and hardware-specific deployment glue behind the traits, composed from the neutral components ARL-S names; not in this crate.) And **`arl-cli`** — the `arl` reference checker: `arl validate <claim.json>` runs the cross-axis gates and exits 0/1 so a claim that fails the ARL gates fails your build; `arl lint` reports controlled-vocabulary findings; `arl verify <session.json> <attestation.json>` checks an attestation; `arl explain` prints the axes and gates. A `Claim` is plain JSON — missing fields default to the uncharacterized floor, and an out-of-range validation depth is rejected at the parse boundary.

And **`arl-wasm`** — the same `arl-core` gates compiled to WebAssembly (single-sourced; no JS reimplementation to drift), so a browser playground validates a claim live. The playground + spec site lives in the separate `arl-web/` project (not this repo), an Astro site that renders the spec and bundles the WASM checker. The compiled WASM was verified executing in a real runtime — a downgraded claim reports its gate violations, a term with no operational definition cannot anchor the claim — identical to the CLI.

`arl-rs/` is **55 tests, 0 failures**, depends on nothing but `arl-core` + neutral crypto/wasm crates. The multi-node sandbox, edge tier, non-NVIDIA accelerator telemetry, and continuous-evaluation framework remain deferred.

## Lineage

ARL adapts the Technology Readiness Level scale (NASA 1974; nine levels per Mankins 1995; ISO 16290:2013) and applies measurement disciplines from ENERGY STAR (energy), pharmaceutical efficacy / radar Pd-Pfa / bit-error-rate practice (convergence), and the CIA + non-repudiation discipline of NIST SP 800-53, ISO/IEC 27001, and Common Criteria EALs (security). Hardware documentation discipline draws from DoD SWaP-C2 reporting.

## Licence

CC-BY-4.0 (see [`LICENSE`](LICENSE)) for the specification text. The forthcoming reference implementation will carry Apache-2.0.

ARL is owned by no one. Transaction Science is one steward — it publishes the specification and writes the reference implementation; the standard itself is public and forkable.
