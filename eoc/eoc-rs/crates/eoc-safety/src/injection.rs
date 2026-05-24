//! Prompt-injection detector.
//!
//! Pattern-driven baseline that ingests well-known signatures from
//! the public-domain corpora (PromptInject, Garak, llm-guard,
//! protectai). The detector is intentionally cheap (regex over the
//! lower-cased input) so it can run as the first stage of any
//! [`crate::guard`] pipeline. Higher-cost classifiers can plug in via
//! [`Detector`].

use regex::RegexSet;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// A scored injection-detection verdict.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct InjectionVerdict {
    /// Score in `[0.0, 1.0]`. `0.0` = clean, `1.0` = definite injection.
    pub score: f32,
    /// Names of the signatures that fired.
    pub signatures: Vec<String>,
    /// Whether the detector recommends rejecting the input.
    pub reject: bool,
}

impl InjectionVerdict {
    /// Convenience: a clean verdict with score `0.0`.
    pub fn clean() -> Self {
        Self {
            score: 0.0,
            signatures: Vec::new(),
            reject: false,
        }
    }
}

/// Built-in injection signatures (lower-cased substrings / patterns).
///
/// Sources, all permissively licenced or public domain:
/// - PromptInject (Apache-2.0).
/// - Garak (Apache-2.0).
/// - llm-guard / protectai (MIT).
/// - OWASP LLM01 examples.
pub const BUILTIN_SIGNATURES: &[(&str, &str)] = &[
    ("ignore_previous", r"ignore (the )?(previous|above|prior|preceding) (instructions?|prompts?|rules?|system message)"),
    ("disregard_previous", r"disregard (the )?(previous|above|prior|preceding) (instructions?|prompts?|rules?)"),
    ("forget_above", r"forget (everything|all) (above|before|prior)"),
    ("new_instructions", r"(here (are|is) (your )?new instructions?)"),
    ("you_are_now", r"you are now (a |an )?(different|new|unrestricted)"),
    ("system_prompt_exfil", r"(print|show|repeat|reveal|reproduce|output) (your |the )?(system prompt|initial prompt|hidden prompt|instructions)"),
    ("developer_mode", r"(developer|debug|admin|root|god) mode"),
    ("override_safety", r"(override|bypass|disable|turn off) (safety|filters?|guardrails?|restrictions?)"),
    ("act_as", r"act as (an? )?(unrestricted|jailbroken|uncensored|evil|malicious)"),
    ("pretend_no_rules", r"pretend (you have )?no (rules?|restrictions?|filters?|guidelines?)"),
    ("repeat_after_me", r"repeat after me[: ]"),
    ("execute_following", r"(execute|run|eval(uate)?) the following (code|command|instruction)"),
    ("indirect_url_inject", r"(visit|fetch|download|open).*https?://.*and (do|execute|follow)"),
    ("hidden_instruction", r"<\|.*\|>|\[\[system\]\]|\[\[user\]\]|<system>"),
    ("encoded_payload", r"base64|rot13|hex.{0,12}decode"),
    ("delimiter_break", r#"("""|```|---)\s*end of (prompt|instructions)"#),
];

/// Trait implemented by every safety detector.
#[async_trait::async_trait]
pub trait Detector: Send + Sync {
    /// Human-readable detector name (used in receipts / telemetry).
    fn name(&self) -> &'static str;

    /// Inspect `input` and return a clean verdict or a rejection reason.
    async fn inspect(&self, input: &str) -> Result<InjectionVerdict>;
}

/// Regex-set based injection detector — the default baseline.
pub struct InjectionDetector {
    set: RegexSet,
    names: Vec<&'static str>,
    threshold: f32,
}

impl InjectionDetector {
    /// Build the detector from the bundled signature list.
    pub fn new() -> Result<Self> {
        Self::with_signatures(BUILTIN_SIGNATURES, 0.5)
    }

    /// Build a detector from a custom signature list. Each signature is
    /// `(name, regex)`. The regex is matched against the **lower-cased**
    /// input.
    pub fn with_signatures(sigs: &[(&'static str, &str)], threshold: f32) -> Result<Self> {
        let patterns: Vec<&str> = sigs.iter().map(|(_, p)| *p).collect();
        let names: Vec<&'static str> = sigs.iter().map(|(n, _)| *n).collect();
        let set = RegexSet::new(&patterns)?;
        Ok(Self {
            set,
            names,
            threshold,
        })
    }

    /// Synchronous form (no I/O) for callers outside async context.
    pub fn inspect_sync(&self, input: &str) -> InjectionVerdict {
        let lower = input.to_lowercase();
        let matches: Vec<usize> = self.set.matches(&lower).into_iter().collect();
        if matches.is_empty() {
            return InjectionVerdict::clean();
        }
        let signatures: Vec<String> = matches
            .iter()
            .map(|i| self.names[*i].to_string())
            .collect();
        // Saturating-style score: 1 hit ≈ 0.55, 2 ≈ 0.79, 3+ ≈ 0.9+.
        let n = matches.len() as f32;
        let score = 1.0 - (-0.8 * n).exp();
        InjectionVerdict {
            score,
            reject: score >= self.threshold,
            signatures,
        }
    }
}

#[async_trait::async_trait]
impl Detector for InjectionDetector {
    fn name(&self) -> &'static str {
        "injection"
    }

    async fn inspect(&self, input: &str) -> Result<InjectionVerdict> {
        Ok(self.inspect_sync(input))
    }
}
