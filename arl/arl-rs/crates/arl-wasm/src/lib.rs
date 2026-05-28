//! WASM bindings over `arl-core` for the browser playground.
//!
//! These functions run the *same* cross-axis gates and lexicon checks as
//! the `arl` CLI — the logic lives in `arl-core`, compiled to WASM here,
//! so the playground can never drift from the standard. Each returns a
//! JSON string the page parses and renders.

#![forbid(unsafe_code)]

use arl_core::{Claim, Severity};
use wasm_bindgen::prelude::*;

/// Validate a claim (JSON) against the cross-axis gates.
///
/// Returns JSON: `{ "ok": bool, "violations": [string], "warnings": [string] }`.
/// A parse error returns `ok: false` with the error in `violations`.
#[wasm_bindgen]
pub fn validate(claim_json: &str) -> String {
    let claim: Claim = match serde_json::from_str(claim_json) {
        Ok(c) => c,
        Err(e) => {
            return result_json(
                false,
                &[format!("invalid claim JSON: {e}")],
                &[],
            );
        }
    };
    let warnings: Vec<String> = claim
        .warnings()
        .iter()
        .map(|w| format!("`{}` in {}", w.term, w.field))
        .collect();
    match claim.validate() {
        Ok(()) => result_json(true, &[], &warnings),
        Err(violations) => {
            let v: Vec<String> = violations.iter().map(|x| x.to_string()).collect();
            result_json(false, &v, &warnings)
        }
    }
}

/// Lint a claim's prose against the controlled vocabulary.
///
/// Returns JSON: `{ "findings": [{ "term", "field", "severity" }], "excluded": n }`.
#[wasm_bindgen]
pub fn lint(claim_json: &str) -> String {
    let claim: Claim = match serde_json::from_str(claim_json) {
        Ok(c) => c,
        Err(e) => {
            return serde_json::json!({ "error": format!("invalid claim JSON: {e}") }).to_string();
        }
    };
    let mut excluded = 0u32;
    let findings: Vec<serde_json::Value> = claim
        .lexicon_findings()
        .into_iter()
        .map(|f| {
            let sev = match f.severity {
                Severity::Excluded => {
                    excluded += 1;
                    "excluded"
                }
                Severity::PartiallyHype => "warn",
            };
            serde_json::json!({ "term": f.term, "field": f.field, "severity": sev })
        })
        .collect();
    serde_json::json!({ "findings": findings, "excluded": excluded }).to_string()
}

fn result_json(ok: bool, violations: &[String], warnings: &[String]) -> String {
    serde_json::json!({
        "ok": ok,
        "violations": violations,
        "warnings": warnings,
    })
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn validate_returns_ok_json_for_minimal_floor_claim() {
        // ARL 1 floor claim with identity → passes.
        let v = validate(r#"{"system":"s","task":"t","validation_depth":1}"#);
        let parsed: serde_json::Value = serde_json::from_str(&v).unwrap();
        assert_eq!(parsed["ok"], true);
    }

    #[test]
    fn validate_flags_gate_and_excluded_term() {
        let v = validate(
            r#"{"system":"s","task":"hits AGI","validation_depth":6,"convergence":"D","energy":"Undisclosed","security":"S0"}"#,
        );
        let parsed: serde_json::Value = serde_json::from_str(&v).unwrap();
        assert_eq!(parsed["ok"], false);
        let viols = parsed["violations"].as_array().unwrap();
        assert!(viols.iter().any(|x| x.as_str().unwrap().contains("agi")));
        assert!(viols.iter().any(|x| x.as_str().unwrap().contains("convergence")));
    }

    #[test]
    fn validate_reports_bad_json() {
        let v = validate("{not json");
        let parsed: serde_json::Value = serde_json::from_str(&v).unwrap();
        assert_eq!(parsed["ok"], false);
        assert!(parsed["violations"][0].as_str().unwrap().contains("invalid claim JSON"));
    }

    #[test]
    fn lint_counts_excluded() {
        let l = lint(r#"{"system":"s","task":"demonstrates AGI","validation_depth":1}"#);
        let parsed: serde_json::Value = serde_json::from_str(&l).unwrap();
        assert_eq!(parsed["excluded"], 1);
    }
}
