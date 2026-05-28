//! The evaluation session and the structural rules ARL-S puts on it.

use arl_core::ValidationDepth;
use serde::{Deserialize, Serialize};
use thiserror::Error;

/// Isolation strength, mapped to an ARL range. Higher scores require
/// stronger isolation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum IsolationTier {
    /// Tier 0 — Research (ARL 1–3). No isolation required.
    Tier0,
    /// Tier 1 — Process (ARL 4). seccomp-bpf / namespaces / cgroups v2.
    Tier1,
    /// Tier 2 — Container (ARL 5–6). Content-addressable image, sub-sandbox tools.
    Tier2,
    /// Tier 3 — MicroVM (ARL 7–9). Dedicated cpuset + PCI passthrough.
    Tier3,
}

impl IsolationTier {
    /// The minimum tier ARL `level` demands.
    pub fn required_for(level: u8) -> IsolationTier {
        match level {
            0..=3 => IsolationTier::Tier0,
            4 => IsolationTier::Tier1,
            5..=6 => IsolationTier::Tier2,
            _ => IsolationTier::Tier3, // 7–9
        }
    }

    fn rank(self) -> u8 {
        match self {
            IsolationTier::Tier0 => 0,
            IsolationTier::Tier1 => 1,
            IsolationTier::Tier2 => 2,
            IsolationTier::Tier3 => 3,
        }
    }

    /// True if `self` is at least as strong as `floor`.
    pub fn at_least(self, floor: IsolationTier) -> bool {
        self.rank() >= floor.rank()
    }
}

/// Which telemetry categories were captured. Above Tier 0 all three are
/// required.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelemetryPresence {
    /// Session/SUT/Harness ids, inputs, intermediate states, transcript.
    pub logical: bool,
    /// CPU time, memory, token counts, I/O, network.
    pub resource: bool,
    /// RAPL/NVML energy, thermal/throttle/voltage events, perf counters.
    pub physical: bool,
}

impl TelemetryPresence {
    pub fn all(self) -> bool {
        self.logical && self.resource && self.physical
    }
}

/// One ARL-S evaluation session. Designed to serialize to a **float-free**
/// JSON value (ids, an integer tier, booleans, integer timestamps, hex
/// hashes) so its [`attest`](crate::attest) canonical form is
/// unambiguous. The ARL claim it attests is referenced by content hash,
/// not embedded, keeping the signed record stable and the energy figures
/// (which carry floats) in the claim.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Session {
    /// Unique session identifier.
    pub session_id: String,
    /// SUT identity (version + weights hash + config hash).
    pub sut_id: String,
    /// Harness identity (version + config).
    pub harness_id: String,
    /// Supervisor identity.
    pub supervisor_id: String,
    /// Isolation tier the session actually ran in.
    pub tier: IsolationTier,
    /// Which telemetry categories were captured.
    pub telemetry: TelemetryPresence,
    /// Whether the session is replayable from telemetry (required at
    /// Tier 2/3).
    pub replayable: bool,
    /// Probing by the SUT (cpuinfo reads, hardware enumeration, root
    /// walks) was observed — recorded, not necessarily invalidating.
    pub probing_detected: bool,
    /// Tampering (privilege escalation, VM-escape, out-of-allocation
    /// writes, or mid-session Harness-config mutation) was detected —
    /// this invalidates the session.
    pub tampering_detected: bool,
    /// SHA-256 (hex) of the ARL claim this session attests.
    pub arl_claim_sha256_hex: String,
    /// Measurement time, unix seconds.
    pub measured_unix: u64,
    /// Validity horizon, unix seconds.
    pub valid_through_unix: u64,
}

/// A reason a session does not satisfy ARL-S for a given ARL level.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
pub enum SessionViolation {
    #[error("ARL {level} requires isolation {need:?} or stronger, but the session ran in {have:?}")]
    IsolationTooWeak {
        level: u8,
        have: IsolationTier,
        need: IsolationTier,
    },
    #[error("isolation above Tier 0 requires all three telemetry categories (logical, resource, physical)")]
    TelemetryIncomplete,
    #[error("Tier 2/3 sessions must be replayable from telemetry")]
    NotReplayable,
    #[error("tampering was detected — the session is invalid")]
    TamperingInvalidates,
    #[error("session references no ARL claim (arl_claim_sha256_hex is empty)")]
    NoClaimReference,
}

impl Session {
    /// Validate the session as the measurement substrate for an ARL claim
    /// at `level`. Returns every violation found.
    pub fn validate(&self, level: ValidationDepth) -> Result<(), Vec<SessionViolation>> {
        let mut v = Vec::new();
        let lvl = level.level();

        // Tampering invalidates unconditionally.
        if self.tampering_detected {
            v.push(SessionViolation::TamperingInvalidates);
        }

        // Tier vs ARL.
        let need = IsolationTier::required_for(lvl);
        if !self.tier.at_least(need) {
            v.push(SessionViolation::IsolationTooWeak {
                level: lvl,
                have: self.tier,
                need,
            });
        }

        // Telemetry: all three required above Tier 0.
        if self.tier != IsolationTier::Tier0 && !self.telemetry.all() {
            v.push(SessionViolation::TelemetryIncomplete);
        }

        // Replay required at Tier 2/3.
        if self.tier.at_least(IsolationTier::Tier2) && !self.replayable {
            v.push(SessionViolation::NotReplayable);
        }

        // Must reference the claim it attests.
        if self.arl_claim_sha256_hex.trim().is_empty() {
            v.push(SessionViolation::NoClaimReference);
        }

        if v.is_empty() {
            Ok(())
        } else {
            Err(v)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tier3_session() -> Session {
        Session {
            session_id: "s-1".into(),
            sut_id: "model-x v2 + weights:abc + cfg:def".into(),
            harness_id: "harness v1".into(),
            supervisor_id: "supervisor v1".into(),
            tier: IsolationTier::Tier3,
            telemetry: TelemetryPresence {
                logical: true,
                resource: true,
                physical: true,
            },
            replayable: true,
            probing_detected: false,
            tampering_detected: false,
            arl_claim_sha256_hex: "0".repeat(64),
            measured_unix: 1_900_000_000,
            valid_through_unix: 1_931_536_000,
        }
    }

    #[test]
    fn tier_required_for_level() {
        assert_eq!(IsolationTier::required_for(3), IsolationTier::Tier0);
        assert_eq!(IsolationTier::required_for(4), IsolationTier::Tier1);
        assert_eq!(IsolationTier::required_for(6), IsolationTier::Tier2);
        assert_eq!(IsolationTier::required_for(9), IsolationTier::Tier3);
    }

    #[test]
    fn clean_tier3_supports_arl9() {
        let s = tier3_session();
        assert!(s.validate(ValidationDepth::new(9).unwrap()).is_ok());
    }

    #[test]
    fn weak_isolation_for_high_arl_fails() {
        let mut s = tier3_session();
        s.tier = IsolationTier::Tier1;
        s.replayable = false; // tier1 doesn't need replay; isolate the tier gate
        let errs = s.validate(ValidationDepth::new(7).unwrap()).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, SessionViolation::IsolationTooWeak { need: IsolationTier::Tier3, .. })));
    }

    #[test]
    fn telemetry_required_above_tier0() {
        let mut s = tier3_session();
        s.telemetry.physical = false;
        let errs = s.validate(ValidationDepth::new(7).unwrap()).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, SessionViolation::TelemetryIncomplete)));
    }

    #[test]
    fn tier0_does_not_require_telemetry() {
        let s = Session {
            tier: IsolationTier::Tier0,
            telemetry: TelemetryPresence::default(),
            replayable: false,
            ..tier3_session()
        };
        // ARL 3 in Tier 0 with no telemetry is fine.
        assert!(s.validate(ValidationDepth::new(3).unwrap()).is_ok());
    }

    #[test]
    fn tier2_requires_replay() {
        let mut s = tier3_session();
        s.tier = IsolationTier::Tier2;
        s.replayable = false;
        let errs = s.validate(ValidationDepth::new(6).unwrap()).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, SessionViolation::NotReplayable)));
    }

    #[test]
    fn tampering_invalidates() {
        let mut s = tier3_session();
        s.tampering_detected = true;
        let errs = s.validate(ValidationDepth::new(9).unwrap()).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, SessionViolation::TamperingInvalidates)));
    }

    #[test]
    fn missing_claim_reference_fails() {
        let mut s = tier3_session();
        s.arl_claim_sha256_hex = "".into();
        let errs = s.validate(ValidationDepth::new(9).unwrap()).unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, SessionViolation::NoClaimReference)));
    }
}
