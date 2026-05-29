# AI Readiness Level (ARL)

**Version 1.3** — May 2026

AI Readiness Level (ARL) is a measurement system used to assess the readiness of an AI system to perform a defined task in a defined context. Every ARL score has four parts: a validation depth from 1 to 9, a convergence class from A to E, an energy profile in joules, and a security class from S0 to S4. All four parts are required. None summarizes the others.

A score is assigned to a specific system performing a specific task on specific hardware. The score is for that combination. Change the system, the task, or the hardware, and you score again.

---

## Validation Depth

A scale from 1 to 9 describing how thoroughly the readiness claim has been tested.

**ARL 1.** Principle observed. The capability is hypothesized with theoretical or adjacent-domain support.

**ARL 2.** Capability formulated as a measurable claim with a published evaluation methodology.

**ARL 3.** Proof of concept demonstrated on a defined set of test cases.

**ARL 4.** Component validated. Measured on a defined benchmark with error bars from N ≥ 3 runs, adversarial test results, and a documented failure-mode catalog.

**ARL 5.** Integrated validation. Demonstrated in a realistic synthetic environment with multi-step tasks and adversarial conditions.

**ARL 6.** Operational prototype. Demonstrated in actual deployment with real users or tasks, narrow scope, active human oversight on every instance.

**ARL 7.** Operational at scale. Production deployment with documented reliability metrics, drift monitoring, incident reporting, quarterly revalidation.

**ARL 8.** Qualified. Sustained at scale for 12+ months without per-action oversight, within a defined operational envelope, with independent third-party assessment.

**ARL 9.** Proven. Sustained track record across diverse operational contexts for 24+ months, with public incident disclosure.

Above ARL 4, error bars and failure modes must be published. Above ARL 5, the evaluation methodology must be published before the claim is made.

---

## Convergence Class

A class from A to E describing how stochastic the system is on the certified task.

**Class A.** Deterministic equivalent. Identical or semantically equivalent outputs across at least 100 runs under identical conditions.

**Class B.** Bounded convergent. Variance characterized, failure rate quantified with confidence interval, failure modes enumerated. N ≥ 30 under operational conditions.

**Class C.** Bounded with characterized failures. Reliable inside an operational envelope, fails predictably outside it. The envelope is documented and enforced.

**Class D.** Divergent on extension. Stable on short tasks, diverges as task length, context size, or compositional depth increases.

**Class E.** Uncharacterized. Variance never measured. Default class.

ARL 4 and above require Class D or better. ARL 6 and above require Class C. ARL 8 and above require Class B.

---

## Energy Profile

Three numbers, all in joules.

**Training, amortized.** Total training energy divided by expected deployment lifetime. Stated in MWh per deployment year. Reported with the training facility's PUE and grid carbon intensity.

**Per task, inference.** Mean and standard deviation of joules consumed to complete one instance of the certified task, including all retries, tool calls, and reasoning. Stated in kJ/task with N ≥ 100.

**Total cost of operation.** Per-task inference energy multiplied by the deployment facility's PUE. Stated in kJ/task at PUE X.X with grid carbon intensity at the deployment location.

Refusing to disclose energy figures caps the achievable score at ARL 3.

---

## Security Class

A class from S0 to S4 describing the system's measured resistance to adversarial conditions. Four properties combine into the class: adversarial robustness, output integrity, input and state confidentiality, and auditability.

**S0.** Uncharacterized. Default class. No security measurement performed.

**S1.** Adversarial robustness measured. Attack success rate published from a documented attack corpus (jailbreaks, prompt injection, adversarial inputs, Constrained Decoding Attack patterns). N ≥ 100 attacks per category. No requirements on the other three properties.

**S2.** Adversarial robustness measured. Output integrity cryptographically attested: every output signed by the inference operator with a hardware-backed key, every artifact content-addressable, replay reconstructs the session deterministically from telemetry. Ed25519 over JCS-canonicalized JSON (RFC 8032 + RFC 8785 + SHA-256) is the reference signing primitive set.

**S3.** S2 plus measured input and state confidentiality. Training data extraction attack rate published. System prompt extraction rate published. Tool credential leak rate published. Side-channel leak rate published. Cross-user context leak rate published where multi-tenant.

**S4.** S3 plus complete auditability. Every output traceable to its inputs, tool calls, model version, harness configuration, and operator identity. Audit trail completeness measured against an adversary attempting to make actions un-traceable. Retention period documented.

Each level requires the published measurement methodology to predate the published claim. Post-hoc selection of favorable attack corpora invalidates the level.

ARL 4 and above require S1 or better. ARL 6 and above require S2. ARL 8 and above require S3. ARL 9 requires S4.

Security claims at S3 and above must be independently reproducible by an adversarial third party. Refusing to disclose security measurement methodology caps the achievable score at S0.

---

## A complete ARL score

```
System:      [model version] + [harness version] + [config hash]
Task:        [specific task definition]
Context:     [deployment envelope — scope, supervision, exclusions]

ARL:         6
Class:       D
Energy:      training    38 MWh/deployment-year
             inference   12.3 ± 7.1 kJ/task (N=500)
             total       18.5 kJ/task at PUE 1.5, 420 gCO₂eq/kWh
Security:    S2
             adversarial    27% attack success rate (N=1000, corpus v2026.03)
             integrity      Ed25519 over JCS, hardware-key signing
             confidentiality not measured
             auditability   session replay verified, identity audit pending

Envelope:    [operational limits — what the claim covers and what it does not]
Methodology: [link to published evaluation methodology and results]
Measured on: 8× H200, 141GB HBM/accelerator, NVLink, FP8 (E4M3),
             vLLM 0.7.2, continuous batching, safety filter v2026.02
Measured:    [date]    Valid through: [date + 12 months]
```

A claim missing any of the four parts is incomplete. The hardware on which the score was measured is documentation alongside the date, methodology link, and validity window — required for reproducibility, but not a peer axis with the four measured properties of the system.

---

## Companion specification

ARL scores are measured inside the ARL Sandbox (ARL-S). The sandbox specifies the testing environment, isolation tiers, telemetry, and replay requirements. See `ARL-S.md`.

---

## Lineage

ARL adapts the Technology Readiness Level scale (NASA, 1974; nine levels per Mankins 1995; ISO 16290:2013) and applies measurement disciplines from ENERGY STAR (energy profile), stochastic system characterization practice from pharmaceutical efficacy reporting, radar Pd/Pfa specification, and bit error rate methodology (convergence class), and the confidentiality/integrity/availability/non-repudiation discipline of NIST SP 800-53, ISO/IEC 27001, and Common Criteria Evaluation Assurance Levels (security class). Hardware documentation discipline draws from DoD SWaP-C2 reporting practice.
