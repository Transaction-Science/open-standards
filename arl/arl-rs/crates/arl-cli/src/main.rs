//! `arl` — the ARL reference checker.
//!
//! A thin, dependency-light command line over `arl-core` and
//! `arl-sandbox`. The gate logic lives in the libraries; this binary is
//! a faithful front end so a claim can be checked in CI:
//!
//! ```text
//! arl validate claim.json      # cross-axis gates → exit 0 (pass) / 1 (fail)
//! arl lint     claim.json      # controlled-vocabulary findings
//! arl verify   session.json attestation.json   # attestation check
//! arl explain                  # the axes and the gates, as a cheat sheet
//! ```
//!
//! `validate` and `verify` set the exit code so the binary drops into a
//! pipeline gate: a claim that fails the ARL gates fails the build.

#![forbid(unsafe_code)]

use std::io::Write;
use std::process::ExitCode;

use arl_core::Claim;
use arl_sandbox::{verify_attestation, Attestation, Session};

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut out = std::io::stdout();
    let code = run(&args, &mut out);
    let _ = out.flush();
    ExitCode::from(code)
}

/// Dispatch a parsed argument vector, writing human output to `out` and
/// returning the process exit code (0 = ok / pass, 1 = fail, 2 = usage).
fn run(args: &[String], out: &mut impl Write) -> u8 {
    match args.first().map(String::as_str) {
        Some("validate") => match read_arg(args, 1) {
            Ok(json) => {
                let (report, code) = validate_claim_json(&json);
                let _ = writeln!(out, "{report}");
                code
            }
            Err(e) => usage(out, &e),
        },
        Some("lint") => match read_arg(args, 1) {
            Ok(json) => {
                let (report, code) = lint_claim_json(&json);
                let _ = writeln!(out, "{report}");
                code
            }
            Err(e) => usage(out, &e),
        },
        Some("verify") => match (read_arg(args, 1), read_arg(args, 2)) {
            (Ok(session), Ok(att)) => {
                let (report, code) = verify_json(&session, &att);
                let _ = writeln!(out, "{report}");
                code
            }
            _ => usage(out, "verify needs <session.json> <attestation.json>"),
        },
        Some("explain") => {
            let _ = writeln!(out, "{}", explain());
            0
        }
        Some("--help") | Some("-h") | Some("help") | None => {
            let _ = writeln!(out, "{}", help_text());
            0
        }
        Some(other) => usage(out, &format!("unknown command `{other}`")),
    }
}

fn usage(out: &mut impl Write, msg: &str) -> u8 {
    let _ = writeln!(out, "error: {msg}\n\n{}", help_text());
    2
}

/// Read the Nth positional argument as a file path and return its
/// contents. (Kept separate so the core logic functions take strings and
/// stay unit-testable without touching the filesystem.)
fn read_arg(args: &[String], n: usize) -> Result<String, String> {
    let path = args.get(n).ok_or_else(|| format!("missing argument {n}"))?;
    std::fs::read_to_string(path).map_err(|e| format!("cannot read `{path}`: {e}"))
}

fn help_text() -> String {
    "arl — the ARL reference checker\n\
     \n\
     USAGE:\n\
     \u{20}\u{20}arl validate <claim.json>                   check the four-axis cross-axis gates\n\
     \u{20}\u{20}arl lint     <claim.json>                   report controlled-vocabulary findings\n\
     \u{20}\u{20}arl verify   <session.json> <attest.json>   verify a session attestation\n\
     \u{20}\u{20}arl explain                                 print the axes and the gates\n\
     \n\
     EXIT CODES: 0 = pass, 1 = fail, 2 = usage error."
        .to_string()
}

/// `arl validate` core: parse a Claim and run the cross-axis gates.
fn validate_claim_json(json: &str) -> (String, u8) {
    let claim: Claim = match serde_json::from_str(json) {
        Ok(c) => c,
        Err(e) => return (format!("error: invalid claim JSON: {e}"), 2),
    };
    match claim.validate() {
        Ok(()) => {
            let mut s = String::from("PASS — well-formed ARL claim.");
            let warnings = claim.warnings();
            if !warnings.is_empty() {
                s.push_str("\n\nwarnings (partially-hype terms — review for operational sense):");
                for w in warnings {
                    s.push_str(&format!("\n  · `{}` in {}", w.term, w.field));
                }
            }
            (s, 0)
        }
        Err(violations) => {
            let mut s = format!("FAIL — {} gate violation(s):", violations.len());
            for v in violations {
                s.push_str(&format!("\n  ✗ {v}"));
            }
            (s, 1)
        }
    }
}

/// `arl lint` core: report all lexicon findings (excluded + partial).
fn lint_claim_json(json: &str) -> (String, u8) {
    let claim: Claim = match serde_json::from_str(json) {
        Ok(c) => c,
        Err(e) => return (format!("error: invalid claim JSON: {e}"), 2),
    };
    let findings = claim.lexicon_findings();
    if findings.is_empty() {
        return ("clean — no controlled-vocabulary findings.".to_string(), 0);
    }
    let mut excluded = 0u32;
    let mut s = String::from("lexicon findings:");
    for f in &findings {
        let tag = match f.severity {
            arl_core::Severity::Excluded => {
                excluded += 1;
                "EXCLUDED"
            }
            arl_core::Severity::PartiallyHype => "warn",
        };
        s.push_str(&format!("\n  [{tag}] `{}` in {}", f.term, f.field));
    }
    // Any excluded term means the prose is not ARL-claim-eligible.
    (s, if excluded > 0 { 1 } else { 0 })
}

/// `arl verify` core: verify an attestation over a session.
fn verify_json(session_json: &str, attestation_json: &str) -> (String, u8) {
    let session: Session = match serde_json::from_str(session_json) {
        Ok(s) => s,
        Err(e) => return (format!("error: invalid session JSON: {e}"), 2),
    };
    let att: Attestation = match serde_json::from_str(attestation_json) {
        Ok(a) => a,
        Err(e) => return (format!("error: invalid attestation JSON: {e}"), 2),
    };
    match verify_attestation(&session, &att) {
        Ok(true) => (
            format!(
                "VERIFIED — attestation is valid for this session.\n  signer: {}",
                att.public_key_hex
            ),
            0,
        ),
        Ok(false) => (
            "FAILED — attestation does not match this session (tampered or wrong key).".to_string(),
            1,
        ),
        Err(e) => (format!("error: {e}"), 2),
    }
}

/// `arl explain`: the axes and the cross-axis gates, as a reference.
fn explain() -> String {
    "ARL — AI Readiness Level. Four required axes; none summarizes the others.\n\
     \n\
     1. Validation Depth (1–9)   — how thoroughly tested        [statistics]\n\
     2. Convergence Class (A–E)  — how stochastic on the task   [stochastic process theory]\n\
     3. Energy Profile (joules)  — train / per-task / total     [thermodynamics]\n\
     4. Security Class (S0–S4)   — robustness/integrity/etc.    [info theory + crypto]\n\
     \n\
     Cross-axis gates:\n\
     \u{20}\u{20}ARL ≥ 4  requires Convergence D+ and Security S1\n\
     \u{20}\u{20}ARL ≥ 6  requires Convergence C+ and Security S2\n\
     \u{20}\u{20}ARL ≥ 8  requires Convergence B+ and Security S3\n\
     \u{20}\u{20}ARL = 9  requires Security S4\n\
     \u{20}\u{20}Undisclosed energy caps the score at ARL 3.\n\
     \u{20}\u{20}Undisclosed security methodology caps the security class at S0.\n\
     \u{20}\u{20}ARL ≥ 4 requires published error bars + failure modes.\n\
     \u{20}\u{20}ARL ≥ 6 requires the methodology published before the claim.\n\
     \u{20}\u{20}Security S3+ must be independently reproducible.\n\
     \n\
     Excluded (unmeasurable) terms — AGI, superintelligence, consciousness,\n\
     sentience, singularity, human-level — are not permitted in a claim."
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arl_core::{ConvergenceClass, EnergyProfile, SecurityClass, ValidationDepth};

    fn good_arl6_json() -> String {
        let claim = Claim {
            system: "model-x v2 + harness v1 + cfg abc".into(),
            task: "translate EN→FR WMT24 sentences".into(),
            context: "narrow scope, human oversight".into(),
            envelope: "WMT24 domain only".into(),
            validation_depth: ValidationDepth::new(6).unwrap(),
            convergence: ConvergenceClass::C,
            energy: EnergyProfile::Disclosed {
                training_mwh_per_year: 38.0,
                inference_kj_mean: 12.3,
                inference_kj_std: 7.1,
                inference_n: 500,
                total_kj: 18.5,
                pue: 1.5,
                grid_gco2_per_kwh: 420.0,
            },
            security: SecurityClass::S2,
            error_bars_published: true,
            failure_modes_published: true,
            methodology_published_before_claim: true,
            methodology_link: Some("https://example.org/m".into()),
            security_methodology_disclosed: true,
            security_independently_reproducible: false,
            hardware: "8× H200".into(),
            measured_date: "2026-05-01".into(),
            valid_through: "2027-05-01".into(),
        };
        serde_json::to_string(&claim).unwrap()
    }

    #[test]
    fn validate_passes_good_claim_exit_0() {
        let (report, code) = validate_claim_json(&good_arl6_json());
        assert_eq!(code, 0, "{report}");
        assert!(report.starts_with("PASS"));
    }

    #[test]
    fn validate_fails_downgraded_convergence_exit_1() {
        let mut v: serde_json::Value = serde_json::from_str(&good_arl6_json()).unwrap();
        v["convergence"] = serde_json::json!("D");
        let (report, code) = validate_claim_json(&v.to_string());
        assert_eq!(code, 1);
        assert!(report.starts_with("FAIL"));
        assert!(report.contains("convergence"));
    }

    #[test]
    fn validate_bad_json_exit_2() {
        let (_r, code) = validate_claim_json("{not valid");
        assert_eq!(code, 2);
    }

    #[test]
    fn lint_flags_excluded_term_exit_1() {
        let mut v: serde_json::Value = serde_json::from_str(&good_arl6_json()).unwrap();
        v["task"] = serde_json::json!("demonstrates AGI on translation");
        let (report, code) = lint_claim_json(&v.to_string());
        assert_eq!(code, 1);
        assert!(report.contains("EXCLUDED"));
        assert!(report.contains("agi"));
    }

    #[test]
    fn lint_clean_claim_exit_0() {
        let (report, code) = lint_claim_json(&good_arl6_json());
        assert_eq!(code, 0);
        assert!(report.starts_with("clean"));
    }

    #[test]
    fn verify_round_trips_a_real_attestation() {
        use arl_sandbox::{
            attest_session, EchoHarness, FixedPhysicalSource, IsolationTier, SigningKey, Supervisor,
        };
        // Build a session + attestation via the Supervisor, serialize both.
        let mut sup = Supervisor::new(
            "sup",
            SigningKey::from_bytes(&[3u8; 32]),
            IsolationTier::Tier3,
            FixedPhysicalSource { cpu_joules: 1.0, gpu_joules: 1.0, source: "fixed".into() },
        );
        let mut h = EchoHarness::new("h");
        let eval = sup
            .evaluate("sut", &mut h, "q", &"ab".repeat(32), 1, 2, false, false)
            .unwrap();
        let session_json = serde_json::to_string(&eval.session).unwrap();
        let att_json = serde_json::to_string(&eval.attestation).unwrap();

        let (report, code) = verify_json(&session_json, &att_json);
        assert_eq!(code, 0, "{report}");
        assert!(report.starts_with("VERIFIED"));

        // Tamper with the serialized session → verify fails.
        let mut tampered: serde_json::Value = serde_json::from_str(&session_json).unwrap();
        tampered["sut_id"] = serde_json::json!("different-sut");
        let (treport, tcode) = verify_json(&tampered.to_string(), &att_json);
        assert_eq!(tcode, 1, "{treport}");
        assert!(treport.starts_with("FAILED"));

        // Keep the unused import honest.
        let _ = attest_session;
    }

    #[test]
    fn explain_and_help_render() {
        assert!(explain().contains("Cross-axis gates"));
        assert!(help_text().contains("arl validate"));
    }

    #[test]
    fn run_dispatches_explain() {
        let mut buf = Vec::new();
        let code = run(&["explain".to_string()], &mut buf);
        assert_eq!(code, 0);
        assert!(String::from_utf8(buf).unwrap().contains("AI Readiness Level"));
    }

    #[test]
    fn run_unknown_command_is_usage_error() {
        let mut buf = Vec::new();
        let code = run(&["frobnicate".to_string()], &mut buf);
        assert_eq!(code, 2);
    }
}
