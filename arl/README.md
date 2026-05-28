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

The three documents are self-contained: a reader can understand and use the framework with only these.

## Status

Document stack **v1.2** (May 2026). Specification only — a standalone, system-agnostic reference implementation (`arl-core`: the four-axis claim model with the cross-axis gates and LEXICON vocabulary enforced, so an invalid claim cannot be constructed; then an `arl-sandbox` Supervisor) is in progress and will land alongside these documents. No reference implementation, playground, multi-node sandbox, edge tier, non-NVIDIA accelerator telemetry, or continuous-evaluation framework ships in v1.2.

## Lineage

ARL adapts the Technology Readiness Level scale (NASA 1974; nine levels per Mankins 1995; ISO 16290:2013) and applies measurement disciplines from ENERGY STAR (energy), pharmaceutical efficacy / radar Pd-Pfa / bit-error-rate practice (convergence), and the CIA + non-repudiation discipline of NIST SP 800-53, ISO/IEC 27001, and Common Criteria EALs (security). Hardware documentation discipline draws from DoD SWaP-C2 reporting.

## Licence

CC-BY-4.0 (see [`LICENSE`](LICENSE)) for the specification text. The forthcoming reference implementation will carry Apache-2.0.

ARL is owned by no one. Transaction Science is one steward — it publishes the specification and writes the reference implementation; the standard itself is public and forkable.
