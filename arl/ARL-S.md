# ARL Sandbox (ARL-S)

**Version 1.2** — May 2026
**Companion to:** ARL v1.2

The ARL Sandbox specifies the testing environment in which ARL scores are measured. The sandbox is the constant. The system under test and the task are the variables.

Every ARL-S evaluation involves three entities, kept separated.

**System Under Test (SUT).** The AI system being evaluated. Sealed inside the sandbox. Cannot read or influence anything outside it.

**Harness.** The runtime inside the sandbox alongside the SUT. Provides task inputs, tools, and any agentic scaffolding the task requires. The Harness is itself subject to telemetry.

**Supervisor.** The process outside the sandbox. Orchestrates the evaluation, reads physical telemetry, signs the attestation. The signing key is stored in hardware unreachable by the SUT or Harness operator.

---

## Isolation tiers

Higher ARL scores require stronger isolation. Each tier maps to an ARL range.

**Tier 0 — Research (ARL 1–3).** No isolation required. Run the evaluation in any environment that produces the required telemetry.

**Tier 1 — Process (ARL 4).** SUT runs as an isolated process. Linux seccomp-bpf for syscall filtering, namespaces for resource isolation, cgroups v2 for resource limits. gVisor is acceptable for stronger kernel boundary.

**Tier 2 — Container (ARL 5–6).** SUT runs inside a container with content-addressable image, controlled mounts, controlled network namespace, controlled resource budget. Podman, Docker, or Kata Containers. Tool execution runs in a sub-sandbox; Wasmtime is preferred for WASI-compatible tools.

**Tier 3 — MicroVM (ARL 7–9).** SUT runs inside a microVM with dedicated CPU cores via cgroups v2 cpuset and accelerators allocated by PCI passthrough. Firecracker, Cloud Hypervisor, or crosvm. Dedicated allocation is required for clean energy attribution.

---

## Telemetry

Three categories. All three are required above Tier 0.

**Logical.** Session identifier, SUT identifier (version, weights hash, configuration hash), Harness identifier, input, intermediate states (tool calls, reasoning chain), output, full transcript capable of replay.

**Resource.** CPU time, resident memory peak, forward pass count, token counts, I/O bytes, network bytes, recording proxy log of all outbound requests.

**Physical.** CPU energy via Intel RAPL through Linux powercap, or AMD energy at `/sys/devices/platform/amd_energy/`. GPU energy via NVIDIA NVML `nvmlDeviceGetTotalEnergyConsumption()`, AMD ROCm SMI, or Intel oneAPI Level Zero. Memory bandwidth via Intel PCM. Thermal events, throttle events, voltage and frequency transitions. Host-side perf counters via eBPF.

Higher-level energy measurement frameworks built on these primitives include Zeus (open-source per-request energy measurement on production stacks including vLLM), TokenPowerBench (phase-aligned attribution to prefill and decode stages), MELODI (CPU and GPU instantaneous power aligned to inference phase), and PIE-P. The ML.ENERGY benchmark uses Zeus on H100 and B200 hardware. These compose with RAPL and NVML rather than replace them.

Telemetry is signed by the Supervisor with a hardware-backed key (TPM, HSM, or secure enclave). The reference signing primitive set is Ed25519 over JCS-canonicalized JSON (RFC 8032 + RFC 8785 + SHA-256). These are the same primitives used by the Microsoft Agent Governance Toolkit (April 2026, MIT-licensed) and Mastercard Verifiable Intent (March 2026), so attestations are compatible with the broader receipt ecosystem at the crypto layer. The SUT has no access to the telemetry stream. The Harness has no path to modify it. Sigstore is an acceptable artifact signing infrastructure on top of these primitives.

Retention: Tier 2 sessions for at least 6 months. Tier 3 sessions for at least 24 months.

---

## Replay

Every Tier 2 or Tier 3 session must be replayable from telemetry. Given the recorded inputs, recorded tool responses, and recorded random seeds, the Harness must be able to reconstruct the session as it occurred.

Replay is for audit. Re-running, which produces a new sample of the system's stochastic behavior, is what convergence class measures.

The sandbox environment itself must be reproducible. All sandbox images are content-addressable. Harness code and Supervisor code are version-controlled with versions recorded in the attestation. Complete configuration is recorded in a form an independent party could reconstruct.

For Tier 3 evaluations supporting ARL 7 or higher, an independent third party with access to equivalent hardware must be able to reconstruct the sandbox from public artifacts and reproduce the evaluation.

---

## Anti-evasion

Hostname, paths, and process tree visible to the SUT are normalized to deployment-realistic values. Time of day, timezone, and locale are normalized. Network responses, when network is enabled, are served from a recording proxy with realistic latencies.

The Supervisor monitors for probing — repeated reads of `/proc/cpuinfo`, hardware enumeration attempts, DNS queries to suspicious domains, filesystem walks of the root directory. Probing is recorded in the attestation.

The Harness can communicate with the SUT but cannot read from outside the sandbox into the SUT context. The Harness's communications to the Supervisor occur via a unidirectional channel the SUT cannot inspect.

Scoring decisions are made outside the SUT's reachable scope. Tool responses do not depend on the eventual score. Timing channels are jittered. Tampering attempts (privilege escalation, VM-escape patterns, memory writes outside allocation) are detected and recorded; detected tampering invalidates the session.

The Harness configuration itself is an attack surface. Constrained Decoding Attack (CDA) and similar schema-level techniques embed evasion in grammar rules while the user prompt stays benign, with documented success rates of 94 to 99 percent across major models. Harness grammars, system prompts, and tool schemas submitted by the system operator must be policy-audited and content-addressable before the session begins. Mutations to the Harness configuration during a session invalidate the session.

---

## Hardware documentation

Every session records the hardware on which it ran, as documentation of the measurement substrate. This is required for reproducibility and for energy claim verification, but it is not a peer axis with the four measured properties of the system (validation depth, convergence class, energy profile, security class). Hardware documentation includes: CPU model and microcode version, accelerator models and driver versions, memory configuration, network interface, storage configuration, precision used (FP16, BF16, FP8, INT8, INT4, mixed), inference stack (serving framework and version, scheduler discipline, safety filter), and agentic harness or scaffolding where applicable.

Per-task inference energy is measured directly via RAPL for CPU and NVML for NVIDIA accelerators (or vendor equivalent). Tier 3 requires dedicated cpuset and PCI passthrough for clean per-microVM attribution.

Total cost of operation is per-task inference energy multiplied by deployment facility PUE. PUE is reported separately at the facility level.

For Tier 3 evaluations, energy claims must be independently verifiable through direct re-execution on equivalent hardware, audit of the signed telemetry stream, or wall-plug power meter cross-check during a sample of sessions.

---

## Reference stack

A reference implementation composes the following open-source components.

**Supervisor:** Rust, Tokio async runtime, TOML or YAML configuration.

**Tier 0–1:** seccomp-bpf, cgroups v2, gVisor.

**Tier 2:** Podman or Docker, Kata Containers, Bottlerocket as the container host, Wasmtime for tool sub-sandbox.

**Tier 3:** Firecracker (Apache 2.0, Rust, KVM-based), Cloud Hypervisor as alternative, dedicated CPU pinning via cpuset, PCI passthrough for accelerators.

**Telemetry:** Linux powercap (RAPL), NVML, ROCm SMI, Intel oneAPI Level Zero, Intel PCM, eBPF. Higher-level frameworks: Zeus, TokenPowerBench, MELODI, PIE-P.

**Attestation:** TPM, HSM, or vendor secure enclave for signing keys. Ed25519 (RFC 8032) over JCS-canonicalized JSON (RFC 8785) with SHA-256. Sigstore for artifact signing infrastructure. Microsoft Agent Governance Toolkit (MIT-licensed, April 2026) is an adjacent reference implementation of the same primitives.

**Tasks plug in as Harnesses.** Inspect (UK AISI), HCAST (METR), GAIA 2 / Agents Research Environments (Meta), AI Verify (Singapore IMDA), or custom corpora.
