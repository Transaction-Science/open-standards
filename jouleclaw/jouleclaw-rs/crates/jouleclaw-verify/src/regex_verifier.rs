//! Regex-based deterministic verifier.
//!
//! Wraps a compiled [`regex::Regex`]. The output is decoded as UTF-8;
//! non-UTF-8 input fails with `"output not utf-8"`. Passes iff
//! `regex.is_match(s) == must_match`.

use crate::error::VerifyError;
use crate::verifier::{OutputVerifier, VerifyResult};

/// Default microjoule cost charged to a regex verifier touch. Small
/// — the verifier runs in-process and is bounded by the input length
/// times the regex DFA — but non-zero so receipts always have at
/// least the verifier's name and cost in the ledger.
pub const DEFAULT_REGEX_COST_UJ: u64 = 50;

/// A verifier that checks whether the output matches (or does not
/// match) a regular expression.
#[derive(Debug)]
pub struct RegexVerifier {
    /// Compiled pattern.
    pattern: regex::Regex,
    /// `true` → Pass when the regex matches. `false` → Pass when it
    /// does *not* match.
    must_match: bool,
    /// Verifier name as it appears in the receipt
    /// (`verify:<tag>`).
    name: String,
    /// Declared microjoule cost.
    cost_uj: u64,
}

impl RegexVerifier {
    /// Build a verifier that passes when the output *matches* `pattern`.
    pub fn must_match(pattern: &str) -> Result<Self, VerifyError> {
        Ok(Self {
            pattern: regex::Regex::new(pattern)?,
            must_match: true,
            name: "verify:regex".to_string(),
            cost_uj: DEFAULT_REGEX_COST_UJ,
        })
    }

    /// Build a verifier that passes when the output does *not* match
    /// `pattern` (e.g. "no swear words", "no SSN-shaped substring").
    pub fn must_not_match(pattern: &str) -> Result<Self, VerifyError> {
        Ok(Self {
            pattern: regex::Regex::new(pattern)?,
            must_match: false,
            name: "verify:regex".to_string(),
            cost_uj: DEFAULT_REGEX_COST_UJ,
        })
    }

    /// Override the verifier name (used in receipts). Convention:
    /// prefix with `verify:`.
    pub fn named(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Override the declared microjoule cost.
    pub fn with_cost_uj(mut self, cost_uj: u64) -> Self {
        self.cost_uj = cost_uj;
        self
    }
}

impl OutputVerifier for RegexVerifier {
    fn name(&self) -> &str {
        &self.name
    }

    fn verify(&self, output: &[u8]) -> VerifyResult {
        let s = match std::str::from_utf8(output) {
            Ok(s) => s,
            Err(_) => return VerifyResult::fail("output not utf-8"),
        };
        let matched = self.pattern.is_match(s);
        if matched == self.must_match {
            VerifyResult::Pass
        } else if self.must_match {
            VerifyResult::fail(format!("regex `{}` did not match", self.pattern.as_str()))
        } else {
            VerifyResult::fail(format!("regex `{}` matched but must not", self.pattern.as_str()))
        }
    }

    fn declared_cost_uj(&self) -> u64 {
        self.cost_uj
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn must_match_passes_when_pattern_matches() {
        let v = RegexVerifier::must_match("^[0-9]+$")
            .expect("compile")
            .named("verify:digits");
        assert_eq!(v.verify(b"12345"), VerifyResult::Pass);
        assert_eq!(v.name(), "verify:digits");
    }

    #[test]
    fn must_match_fails_when_pattern_does_not_match() {
        let v = RegexVerifier::must_match("^[0-9]+$").expect("compile");
        match v.verify(b"abc") {
            VerifyResult::Fail { reason } => assert!(reason.contains("did not match")),
            VerifyResult::Pass => panic!("expected Fail"),
        }
    }

    #[test]
    fn must_not_match_passes_when_pattern_is_absent() {
        let v = RegexVerifier::must_not_match("password=")
            .expect("compile")
            .named("verify:no-creds");
        assert_eq!(v.verify(b"username=alice"), VerifyResult::Pass);
    }

    #[test]
    fn must_not_match_fails_when_pattern_is_present() {
        let v = RegexVerifier::must_not_match("password=").expect("compile");
        match v.verify(b"username=alice password=hunter2") {
            VerifyResult::Fail { reason } => assert!(reason.contains("must not")),
            VerifyResult::Pass => panic!("expected Fail"),
        }
    }

    #[test]
    fn bad_utf8_fails() {
        let v = RegexVerifier::must_match(".*").expect("compile");
        // 0xFF is not a valid leading UTF-8 byte
        let bad = [0xFFu8, 0xFE, 0xFD];
        match v.verify(&bad) {
            VerifyResult::Fail { reason } => assert_eq!(reason, "output not utf-8"),
            VerifyResult::Pass => panic!("expected Fail"),
        }
    }

    #[test]
    fn invalid_regex_pattern_errors() {
        // unbalanced bracket → regex::Error
        let err = RegexVerifier::must_match("[unbalanced").unwrap_err();
        matches!(err, VerifyError::InvalidRegex(_));
    }

    #[test]
    fn declared_cost_is_reported() {
        let v = RegexVerifier::must_match("x")
            .expect("compile")
            .with_cost_uj(123);
        assert_eq!(v.declared_cost_uj(), 123);
    }
}
