//! The Supervisor — orchestrates one evaluation and emits a signed
//! [`Session`].
//!
//! The Supervisor sits outside the sandbox. It drives a [`Harness`] (which
//! runs the task beside the SUT and yields logical + resource telemetry),
//! samples a [`PhysicalTelemetrySource`] (energy, outside the SUT's
//! reach), assembles the [`Session`] with the telemetry categories it
//! actually captured, and signs it.
//!
//! The transport-specific pieces are traits, so this layer runs and is
//! tested on any OS:
//! - **[`Harness`]** — your task runner. The reference [`EchoHarness`]
//!   needs no model.
//! - **[`PhysicalTelemetrySource`]** — energy. [`NullPhysicalSource`]
//!   (no meter → physical telemetry absent → caps the achievable tier);
//!   [`FixedPhysicalSource`] for a known meter. A Linux deployment plugs
//!   in RAPL (powercap) + NVML here; that backend is OS/hardware-specific
//!   and is the deployment's slot, not this crate's.
//!
//! Likewise the signing key is held directly here for the reference
//! implementation; a deployment supplies a TPM/HSM-backed key the SUT and
//! Harness operator cannot reach.

use ed25519_dalek::SigningKey;
use serde::{Deserialize, Serialize};

use crate::attest::{attest_session, AttestError, Attestation};
use crate::session::{IsolationTier, Session, TelemetryPresence};

/// Resource telemetry captured by the Harness wrapper (the "resource"
/// category). Counts, not energy.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceTelemetry {
    pub cpu_ms: u64,
    pub mem_peak_bytes: u64,
    pub forward_passes: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub io_bytes: u64,
    pub net_bytes: u64,
}

/// Physical telemetry sampled by the Supervisor (the "physical"
/// category) — measured energy, outside the SUT's reach.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PhysicalTelemetry {
    pub cpu_joules: f64,
    pub gpu_joules: f64,
    /// What produced these figures, e.g. `"RAPL+NVML"` or `"fixed"`.
    pub source: String,
}

/// What a Harness produces from one task run: the output, the replayable
/// transcript (intermediate states / reasoning chain), and resource
/// counters.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HarnessOutcome {
    pub output: String,
    pub transcript: Vec<String>,
    pub resource: ResourceTelemetry,
}

/// The task runner inside the sandbox beside the SUT.
pub trait Harness {
    /// Run the task for `input`; produce the output, transcript, and
    /// resource counters.
    fn run(&mut self, input: &str) -> HarnessOutcome;
    /// Stable Harness identity (version + config) for the session record.
    fn id(&self) -> &str;
}

/// Reference Harness: echoes the input and records a one-step transcript.
/// Deterministic, dependency-free — for tests and smoke runs.
pub struct EchoHarness {
    id: String,
}

impl EchoHarness {
    pub fn new(id: impl Into<String>) -> Self {
        Self { id: id.into() }
    }
}

impl Harness for EchoHarness {
    fn run(&mut self, input: &str) -> HarnessOutcome {
        let output = format!("echo: {input}");
        HarnessOutcome {
            transcript: vec![format!("received: {input}"), output.clone()],
            resource: ResourceTelemetry {
                input_tokens: input.split_whitespace().count() as u64,
                output_tokens: output.split_whitespace().count() as u64,
                forward_passes: 1,
                ..Default::default()
            },
            output,
        }
    }

    fn id(&self) -> &str {
        &self.id
    }
}

/// Samples physical (energy) telemetry. `None` means no meter is
/// available — the physical telemetry category is then absent.
pub trait PhysicalTelemetrySource {
    fn sample(&mut self) -> Option<PhysicalTelemetry>;
}

/// No physical meter — physical telemetry is absent. Honest default for
/// environments (e.g. macOS dev, sandboxed CI) without RAPL/NVML; caps
/// the achievable isolation tier because Tier ≥ 1 requires all three
/// telemetry categories.
pub struct NullPhysicalSource;
impl PhysicalTelemetrySource for NullPhysicalSource {
    fn sample(&mut self) -> Option<PhysicalTelemetry> {
        None
    }
}

/// A fixed physical reading — for a deployment with a known meter, and
/// for tests.
pub struct FixedPhysicalSource {
    pub cpu_joules: f64,
    pub gpu_joules: f64,
    pub source: String,
}
impl PhysicalTelemetrySource for FixedPhysicalSource {
    fn sample(&mut self) -> Option<PhysicalTelemetry> {
        Some(PhysicalTelemetry {
            cpu_joules: self.cpu_joules,
            gpu_joules: self.gpu_joules,
            source: self.source.clone(),
        })
    }
}

/// The complete result of one evaluation: the signed session plus the
/// telemetry the caller folds into an ARL claim (the energy figures go
/// into the claim's `EnergyProfile`; the transcript is the replay log).
#[derive(Debug, Clone)]
pub struct Evaluation {
    pub session: Session,
    pub attestation: Attestation,
    pub outcome: HarnessOutcome,
    pub physical: Option<PhysicalTelemetry>,
}

/// The Supervisor. Generic over the physical telemetry source.
pub struct Supervisor<P: PhysicalTelemetrySource> {
    supervisor_id: String,
    signing_key: SigningKey,
    tier: IsolationTier,
    physical: P,
    next_session: u64,
}

impl<P: PhysicalTelemetrySource> Supervisor<P> {
    pub fn new(
        supervisor_id: impl Into<String>,
        signing_key: SigningKey,
        tier: IsolationTier,
        physical: P,
    ) -> Self {
        Self {
            supervisor_id: supervisor_id.into(),
            signing_key,
            tier,
            physical,
            next_session: 0,
        }
    }

    /// The Supervisor's public key (hex) — the key a verifier checks an
    /// attestation against via [`Attestation::signer_is`].
    pub fn public_key_hex(&self) -> String {
        let mut s = String::with_capacity(64);
        for b in self.signing_key.verifying_key().as_bytes() {
            s.push_str(&format!("{b:02x}"));
        }
        s
    }

    /// Run one evaluation: drive the Harness, sample physical telemetry,
    /// assemble the [`Session`] with the categories actually captured, and
    /// sign it.
    ///
    /// `probing_detected` / `tampering_detected` are the anti-evasion
    /// verdicts the deployment's monitor supplies (the OS-level detection
    /// is the deployment's slot); they are recorded in the session and a
    /// tampering verdict invalidates it under [`Session::validate`].
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate(
        &mut self,
        sut_id: impl Into<String>,
        harness: &mut dyn Harness,
        input: &str,
        arl_claim_sha256_hex: impl Into<String>,
        measured_unix: u64,
        valid_through_unix: u64,
        probing_detected: bool,
        tampering_detected: bool,
    ) -> Result<Evaluation, AttestError> {
        let outcome = harness.run(input);
        let physical = self.physical.sample();

        let telemetry = TelemetryPresence {
            // Logical telemetry (ids, input, transcript, output) is always
            // recorded by the Supervisor.
            logical: true,
            // Resource telemetry comes from the Harness wrapper.
            resource: true,
            // Physical telemetry is present only if a meter sampled.
            physical: physical.is_some(),
        };

        let session_id = format!("{}::{}", self.supervisor_id, self.next_session);
        self.next_session += 1;

        let session = Session {
            session_id,
            sut_id: sut_id.into(),
            harness_id: harness.id().to_string(),
            supervisor_id: self.supervisor_id.clone(),
            tier: self.tier,
            telemetry,
            // A full logical transcript was recorded → replayable.
            replayable: !outcome.transcript.is_empty(),
            probing_detected,
            tampering_detected,
            arl_claim_sha256_hex: arl_claim_sha256_hex.into(),
            measured_unix,
            valid_through_unix,
        };

        let attestation = attest_session(&session, &self.signing_key)?;

        Ok(Evaluation {
            session,
            attestation,
            outcome,
            physical,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attest::verify_attestation;
    use arl_core::ValidationDepth;

    fn key() -> SigningKey {
        SigningKey::from_bytes(&[5u8; 32])
    }

    #[test]
    fn tier3_evaluation_with_a_meter_supports_arl9() {
        let mut sup = Supervisor::new(
            "sup-1",
            key(),
            IsolationTier::Tier3,
            FixedPhysicalSource {
                cpu_joules: 1200.0,
                gpu_joules: 18_500.0,
                source: "fixed".into(),
            },
        );
        let mut harness = EchoHarness::new("harness v1");
        let eval = sup
            .evaluate(
                "model-x v2",
                &mut harness,
                "what is the capital of france",
                &"ab".repeat(32),
                1_900_000_000,
                1_931_536_000,
                false,
                false,
            )
            .unwrap();

        // All three telemetry categories captured.
        assert!(eval.session.telemetry.all());
        assert!(eval.session.replayable);
        assert_eq!(eval.outcome.output, "echo: what is the capital of france");
        assert!(eval.physical.is_some());

        // The session validates as the substrate for an ARL 9 claim …
        assert!(eval.session.validate(ValidationDepth::new(9).unwrap()).is_ok());
        // … and its attestation verifies and names this Supervisor.
        assert!(verify_attestation(&eval.session, &eval.attestation).unwrap());
        assert!(eval.attestation.signer_is(&sup.public_key_hex()));
    }

    #[test]
    fn no_meter_means_physical_telemetry_absent() {
        let mut sup = Supervisor::new("sup-1", key(), IsolationTier::Tier1, NullPhysicalSource);
        let mut harness = EchoHarness::new("harness v1");
        let eval = sup
            .evaluate("sut", &mut harness, "hi", &"0".repeat(64), 1, 2, false, false)
            .unwrap();
        assert!(!eval.session.telemetry.physical);
        // Tier 1 (ARL 4) requires all three categories → fails honestly.
        let errs = eval.session.validate(ValidationDepth::new(4).unwrap()).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, crate::session::SessionViolation::TelemetryIncomplete)));
    }

    #[test]
    fn session_ids_are_unique_per_supervisor() {
        let mut sup = Supervisor::new("sup-1", key(), IsolationTier::Tier0, NullPhysicalSource);
        let mut h = EchoHarness::new("h");
        let a = sup.evaluate("s", &mut h, "x", &"0".repeat(64), 1, 2, false, false).unwrap();
        let b = sup.evaluate("s", &mut h, "y", &"0".repeat(64), 1, 2, false, false).unwrap();
        assert_ne!(a.session.session_id, b.session.session_id);
    }

    #[test]
    fn tampering_verdict_flows_through_and_invalidates() {
        let mut sup = Supervisor::new(
            "sup-1",
            key(),
            IsolationTier::Tier3,
            FixedPhysicalSource { cpu_joules: 1.0, gpu_joules: 1.0, source: "fixed".into() },
        );
        let mut h = EchoHarness::new("h");
        let eval = sup
            .evaluate("s", &mut h, "x", &"0".repeat(64), 1, 2, false, /* tampering */ true)
            .unwrap();
        assert!(eval.session.tampering_detected);
        // Attestation still verifies (it faithfully signs that tampering
        // happened) — but the session is invalid for any ARL level.
        assert!(verify_attestation(&eval.session, &eval.attestation).unwrap());
        assert!(eval.session.validate(ValidationDepth::new(7).unwrap()).is_err());
    }
}
