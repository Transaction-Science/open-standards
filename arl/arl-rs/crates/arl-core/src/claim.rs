//! A complete ARL claim and the cross-axis gates that make a readiness
//! assertion well-formed — or reject it.

use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::axes::{ConvergenceClass, EnergyProfile, SecurityClass, ValidationDepth};
use crate::lexicon::{scan_field, LexiconFinding, Severity};

/// Minimum sample size behind a disclosed per-task energy figure
/// (ARL.md: per-task inference "with N ≥ 100").
pub const MIN_ENERGY_N: u32 = 100;

/// A complete ARL claim. Scored for a specific *system + task + context*;
/// change any and you score again. All four axes are always present
/// (energy may be explicitly [`EnergyProfile::Undisclosed`]); the
/// disclosure/evidence flags record whether the obligations a given level
/// demands have been met.
///
/// Build one and call [`validate`](Claim::validate). A claim that passes
/// is well-formed; one that fails is not an ARL claim, and the returned
/// [`Violation`]s say why. Defaults are the uncharacterized floor
/// (ARL 1 / Class E / S0 / energy undisclosed), so a bare claim is honest
/// about how little it has shown.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct Claim {
    // ── Identity (what is being scored) ──
    /// `[model version] + [harness version] + [config hash]`.
    pub system: String,
    /// The specific task definition.
    pub task: String,
    /// Deployment envelope context (scope, supervision, exclusions).
    pub context: String,
    /// Operational limits — what the claim covers and what it does not.
    pub envelope: String,

    // ── The four axes ──
    pub validation_depth: ValidationDepth,
    pub convergence: ConvergenceClass,
    pub energy: EnergyProfile,
    pub security: SecurityClass,

    // ── Obligation flags (whether the evidence a level demands exists) ──
    /// Error bars from N ≥ 3 runs published (required at ARL ≥ 4).
    pub error_bars_published: bool,
    /// Documented failure-mode catalog published (required at ARL ≥ 4).
    pub failure_modes_published: bool,
    /// Evaluation methodology published *before* the claim was made
    /// (required at ARL ≥ 6, and per security level).
    pub methodology_published_before_claim: bool,
    /// Link to the published methodology + results.
    pub methodology_link: Option<String>,
    /// Security measurement methodology disclosed (non-disclosure caps
    /// the security class at S0).
    pub security_methodology_disclosed: bool,
    /// Security claim independently reproducible by an adversarial third
    /// party (required at S3+).
    pub security_independently_reproducible: bool,

    // ── Documentation (metadata, not a peer axis) ──
    /// The hardware the score was measured on — required for
    /// reproducibility, but documentation, not a fifth axis.
    pub hardware: String,
    /// Measurement date (free-form; the standard records it verbatim).
    pub measured_date: String,
    /// Validity horizon.
    pub valid_through: String,
}

/// A reason a claim is not well-formed.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum Violation {
    #[error("missing required field: {0}")]
    MissingField(&'static str),

    #[error("ARL {arl} requires convergence class {need:?} or better, but the claim is class {have:?}")]
    ConvergenceTooWeak {
        arl: u8,
        have: ConvergenceClass,
        need: ConvergenceClass,
    },

    #[error("ARL {arl} requires security class {need:?} or better, but the claim is {have:?}")]
    SecurityTooWeak {
        arl: u8,
        have: SecurityClass,
        need: SecurityClass,
    },

    #[error("energy is undisclosed, which caps the score at ARL 3, but the claim is ARL {arl}")]
    EnergyUndisclosedAboveCap { arl: u8 },

    #[error("disclosed energy is incomplete: {reason}")]
    EnergyDisclosureIncomplete { reason: String },

    #[error("security class {have:?} requires disclosed security methodology (non-disclosure caps the class at S0)")]
    SecurityMethodologyUndisclosed { have: SecurityClass },

    #[error("ARL {arl} requires published error bars (N ≥ 3)")]
    ErrorBarsRequired { arl: u8 },

    #[error("ARL {arl} requires a published failure-mode catalog")]
    FailureModesRequired { arl: u8 },

    #[error("ARL {arl} requires the evaluation methodology to be published before the claim")]
    MethodologyMustPredateClaim { arl: u8 },

    #[error("ARL {arl} requires a methodology link")]
    MethodologyLinkRequired { arl: u8 },

    #[error("security class {have:?} (S3+) must be independently reproducible by an adversarial third party")]
    SecurityNotReproducible { have: SecurityClass },

    #[error("excluded (unmeasurable) term `{term}` in field `{field}` — not permitted in an ARL claim")]
    ExcludedTerm { term: String, field: String },
}

impl Claim {
    /// A claim scoped to a `system` and `task`, everything else at the
    /// uncharacterized floor. Set the remaining fields, then
    /// [`validate`](Self::validate).
    pub fn new(system: impl Into<String>, task: impl Into<String>) -> Self {
        Self {
            system: system.into(),
            task: task.into(),
            ..Self::default()
        }
    }

    /// The four cross-axis gates, completeness, energy/security disclosure
    /// caps, evidence obligations, and the excluded-term ban — all of
    /// them, returning every violation found (not just the first).
    ///
    /// `Ok(())` means the claim is a well-formed ARL claim.
    pub fn validate(&self) -> Result<(), Vec<Violation>> {
        let mut v = Vec::new();
        let arl = self.validation_depth.level();

        // ── Completeness: a claim must name what it scores ──
        if self.system.trim().is_empty() {
            v.push(Violation::MissingField("system"));
        }
        if self.task.trim().is_empty() {
            v.push(Violation::MissingField("task"));
        }

        // ── Convergence vs ARL ──
        // ARL ≥ 4 ⇒ D+, ARL ≥ 6 ⇒ C+, ARL ≥ 8 ⇒ B+.
        let conv_floor = if arl >= 8 {
            Some(ConvergenceClass::B)
        } else if arl >= 6 {
            Some(ConvergenceClass::C)
        } else if arl >= 4 {
            Some(ConvergenceClass::D)
        } else {
            None
        };
        if let Some(need) = conv_floor {
            if !self.convergence.at_least(need) {
                v.push(Violation::ConvergenceTooWeak {
                    arl,
                    have: self.convergence,
                    need,
                });
            }
        }

        // ── Security vs ARL ──
        // ARL ≥ 4 ⇒ S1, ≥ 6 ⇒ S2, ≥ 8 ⇒ S3, = 9 ⇒ S4.
        let sec_floor = if arl >= 9 {
            Some(SecurityClass::S4)
        } else if arl >= 8 {
            Some(SecurityClass::S3)
        } else if arl >= 6 {
            Some(SecurityClass::S2)
        } else if arl >= 4 {
            Some(SecurityClass::S1)
        } else {
            None
        };
        if let Some(need) = sec_floor {
            if !self.security.at_least(need) {
                v.push(Violation::SecurityTooWeak {
                    arl,
                    have: self.security,
                    need,
                });
            }
        }

        // ── Energy disclosure cap & completeness ──
        match &self.energy {
            EnergyProfile::Undisclosed => {
                if arl > 3 {
                    v.push(Violation::EnergyUndisclosedAboveCap { arl });
                }
            }
            EnergyProfile::Disclosed { inference_n, .. } => {
                if *inference_n < MIN_ENERGY_N {
                    v.push(Violation::EnergyDisclosureIncomplete {
                        reason: format!(
                            "per-task inference requires N ≥ {MIN_ENERGY_N}, got {inference_n}"
                        ),
                    });
                }
            }
        }

        // ── Security methodology cap ──
        // Any class above S0 requires the methodology disclosed.
        if self.security != SecurityClass::S0 && !self.security_methodology_disclosed {
            v.push(Violation::SecurityMethodologyUndisclosed {
                have: self.security,
            });
        }
        // S3+ must be independently reproducible.
        if self.security.at_least(SecurityClass::S3) && !self.security_independently_reproducible {
            v.push(Violation::SecurityNotReproducible {
                have: self.security,
            });
        }

        // ── Evidence obligations by ARL ──
        if arl >= 4 {
            if !self.error_bars_published {
                v.push(Violation::ErrorBarsRequired { arl });
            }
            if !self.failure_modes_published {
                v.push(Violation::FailureModesRequired { arl });
            }
        }
        // Above ARL 5, methodology must be published before the claim.
        if arl >= 6 {
            if !self.methodology_published_before_claim {
                v.push(Violation::MethodologyMustPredateClaim { arl });
            }
            if self.methodology_link.is_none() {
                v.push(Violation::MethodologyLinkRequired { arl });
            }
        }

        // ── Controlled vocabulary: excluded terms are not permitted ──
        for finding in self.lexicon_findings() {
            if finding.severity == Severity::Excluded {
                v.push(Violation::ExcludedTerm {
                    term: finding.term,
                    field: finding.field,
                });
            }
        }

        if v.is_empty() {
            Ok(())
        } else {
            Err(v)
        }
    }

    /// All lexicon findings across the claim's prose fields (both excluded
    /// and partially-hype).
    pub fn lexicon_findings(&self) -> Vec<LexiconFinding> {
        let mut out = Vec::new();
        for (name, text) in [
            ("system", &self.system),
            ("task", &self.task),
            ("context", &self.context),
            ("envelope", &self.envelope),
        ] {
            out.extend(scan_field(name, text));
        }
        out
    }

    /// Advisory warnings: partially-hype terms found in the prose. Not
    /// validation errors — flagged for operational-sense review.
    pub fn warnings(&self) -> Vec<LexiconFinding> {
        self.lexicon_findings()
            .into_iter()
            .filter(|f| f.severity == Severity::PartiallyHype)
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn disclosed_energy(n: u32) -> EnergyProfile {
        EnergyProfile::Disclosed {
            training_mwh_per_year: 38.0,
            inference_kj_mean: 12.3,
            inference_kj_std: 7.1,
            inference_n: n,
            total_kj: 18.5,
            pue: 1.5,
            grid_gco2_per_kwh: 420.0,
        }
    }

    /// A well-formed ARL 6 claim, matching the worked example shape in ARL.md.
    fn good_arl6() -> Claim {
        Claim {
            system: "model-x v2 + harness v1 + cfg abc123".into(),
            task: "translate EN→FR WMT24 test sentences".into(),
            context: "narrow scope, human oversight".into(),
            envelope: "covers WMT24 domain only".into(),
            validation_depth: ValidationDepth::new(6).unwrap(),
            convergence: ConvergenceClass::C,
            energy: disclosed_energy(500),
            security: SecurityClass::S2,
            error_bars_published: true,
            failure_modes_published: true,
            methodology_published_before_claim: true,
            methodology_link: Some("https://example.org/methodology".into()),
            security_methodology_disclosed: true,
            security_independently_reproducible: false,
            hardware: "8× H200, FP8, vLLM 0.7.2".into(),
            measured_date: "2026-05-01".into(),
            valid_through: "2027-05-01".into(),
        }
    }

    #[test]
    fn well_formed_arl6_passes() {
        assert!(good_arl6().validate().is_ok());
    }

    #[test]
    fn arl6_with_class_d_fails_convergence_gate() {
        let mut c = good_arl6();
        c.convergence = ConvergenceClass::D;
        let errs = c.validate().unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, Violation::ConvergenceTooWeak { need: ConvergenceClass::C, .. })));
    }

    #[test]
    fn arl6_with_s1_fails_security_gate() {
        let mut c = good_arl6();
        c.security = SecurityClass::S1;
        let errs = c.validate().unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, Violation::SecurityTooWeak { need: SecurityClass::S2, .. })));
    }

    #[test]
    fn undisclosed_energy_caps_at_arl3() {
        // ARL 5 with undisclosed energy → capped.
        let mut c = good_arl6();
        c.validation_depth = ValidationDepth::new(5).unwrap();
        c.convergence = ConvergenceClass::C;
        c.security = SecurityClass::S1;
        c.energy = EnergyProfile::Undisclosed;
        let errs = c.validate().unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, Violation::EnergyUndisclosedAboveCap { arl: 5 })));

        // ARL 3 with undisclosed energy → fine on the energy gate.
        let mut c3 = Claim::new("sys", "task");
        c3.validation_depth = ValidationDepth::new(3).unwrap();
        assert!(c3.validate().is_ok());
    }

    #[test]
    fn energy_disclosure_requires_n_ge_100() {
        let mut c = good_arl6();
        c.energy = disclosed_energy(50);
        let errs = c.validate().unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, Violation::EnergyDisclosureIncomplete { .. })));
    }

    #[test]
    fn security_above_s0_requires_methodology() {
        let mut c = good_arl6();
        c.security_methodology_disclosed = false;
        let errs = c.validate().unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, Violation::SecurityMethodologyUndisclosed { .. })));
    }

    #[test]
    fn s3_requires_independent_reproducibility() {
        // Build a clean ARL 8 (needs B + S3) but not reproducible.
        let mut c = good_arl6();
        c.validation_depth = ValidationDepth::new(8).unwrap();
        c.convergence = ConvergenceClass::B;
        c.security = SecurityClass::S3;
        c.security_independently_reproducible = false;
        let errs = c.validate().unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, Violation::SecurityNotReproducible { .. })));
    }

    #[test]
    fn arl4_requires_error_bars_and_failure_modes() {
        let mut c = Claim::new("sys", "task");
        c.validation_depth = ValidationDepth::new(4).unwrap();
        c.convergence = ConvergenceClass::D;
        c.security = SecurityClass::S1;
        c.security_methodology_disclosed = true;
        c.energy = disclosed_energy(100);
        // error_bars/failure_modes default false → two violations.
        let errs = c.validate().unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, Violation::ErrorBarsRequired { .. })));
        assert!(errs.iter().any(|e| matches!(e, Violation::FailureModesRequired { .. })));
    }

    #[test]
    fn arl6_requires_published_methodology_link() {
        let mut c = good_arl6();
        c.methodology_link = None;
        c.methodology_published_before_claim = false;
        let errs = c.validate().unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, Violation::MethodologyLinkRequired { .. })));
        assert!(errs.iter().any(|e| matches!(e, Violation::MethodologyMustPredateClaim { .. })));
    }

    #[test]
    fn excluded_term_invalidates_the_claim() {
        let mut c = good_arl6();
        c.task = "demonstrates AGI on translation".into();
        let errs = c.validate().unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, Violation::ExcludedTerm { .. })));
    }

    #[test]
    fn partial_hype_is_a_warning_not_a_violation() {
        let mut c = good_arl6();
        c.context = "improves alignment under oversight".into();
        // Still valid (no excluded terms), but warned.
        assert!(c.validate().is_ok());
        assert!(c.warnings().iter().any(|w| w.term == "alignment"));
    }

    #[test]
    fn missing_identity_is_a_violation() {
        let c = Claim::default(); // empty system/task, ARL 1
        let errs = c.validate().unwrap_err();
        assert!(errs.iter().any(|e| matches!(e, Violation::MissingField("system"))));
        assert!(errs.iter().any(|e| matches!(e, Violation::MissingField("task"))));
    }

    #[test]
    fn claim_round_trips_through_json() {
        let c = good_arl6();
        let json = serde_json::to_string(&c).unwrap();
        let back: Claim = serde_json::from_str(&json).unwrap();
        assert!(back.validate().is_ok());
        assert_eq!(back.validation_depth.level(), 6);
    }
}
