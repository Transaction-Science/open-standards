//! ARL-S Supervisor core — the measurement environment in which ARL
//! scores are produced, reduced to its testable, neutral essence.
//!
//! ARL-S keeps three entities separated: the **System Under Test (SUT)**,
//! the **Harness** (runtime + tools beside it), and the **Supervisor**
//! (outside the sandbox, reads physical telemetry, signs the
//! attestation with a hardware-backed key the SUT cannot reach). This
//! crate models an evaluation **session**, enforces the structural rules
//! the spec puts on one (isolation tier ↔ ARL range, telemetry presence,
//! replayability, tampering invalidation), and implements the
//! attestation:
//!
//! > Ed25519 (RFC 8032) over **JCS-canonicalized JSON** (RFC 8785) with
//! > SHA-256 (FIPS 180-4).
//!
//! These are neutral industry primitives — the same set the Microsoft
//! Agent Governance Toolkit and Mastercard Verifiable Intent use — so an
//! ARL attestation is verifiable by anyone, with no trust relationship to
//! the issuer. This crate depends only on [`arl_core`] (for the ARL
//! level type) and the crypto primitives; it is tied to no runtime.
//!
//! What this crate does **not** do: launch the actual isolation tiers
//! (seccomp/gVisor/Firecracker), read RAPL/NVML, or talk to a TPM. Those
//! are deployment glue and OS/hardware-specific; ARL-S names the neutral
//! components to compose, and a deployment wires them. This is the
//! session/attestation core that the wiring serializes and signs.

#![forbid(unsafe_code)]

pub mod attest;
pub mod session;
pub mod supervisor;

pub use attest::{attest_session, verify_attestation, AttestError, Attestation};
pub use session::{
    IsolationTier, Session, SessionViolation, TelemetryPresence,
};
pub use supervisor::{
    EchoHarness, Evaluation, FixedPhysicalSource, Harness, HarnessOutcome, NullPhysicalSource,
    PhysicalTelemetry, PhysicalTelemetrySource, ResourceTelemetry, Supervisor,
};
