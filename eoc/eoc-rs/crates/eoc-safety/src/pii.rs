//! PII redaction.
//!
//! Detects and redacts emails, US SSNs, phone numbers, credit cards
//! (Luhn-validated), US street addresses (heuristic), IP addresses
//! (v4 / v6), and a small list of common given names. Patterns ingested
//! from Microsoft Presidio and Google Cloud DLP (both Apache-2.0).
//!
//! The redactor is regex-driven and deterministic; it never makes a
//! network call. Replacement tokens follow the Presidio convention,
//! e.g. `<EMAIL>`, `<SSN>`, `<CREDIT_CARD>`.

use regex::Regex;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// One detected PII span.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiiSpan {
    /// Category label (uppercase).
    pub category: String,
    /// Byte offset where the span starts in the original input.
    pub start: usize,
    /// Byte offset where the span ends in the original input.
    pub end: usize,
    /// The raw matched text.
    pub text: String,
}

/// Result of running the redactor over a string.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PiiReport {
    /// Redacted output (with the spans replaced by category tokens).
    pub redacted: String,
    /// Every span that was matched (in left-to-right order).
    pub spans: Vec<PiiSpan>,
}

impl PiiReport {
    /// True if any PII was detected.
    pub fn has_pii(&self) -> bool {
        !self.spans.is_empty()
    }
}

struct CompiledRule {
    category: &'static str,
    re: Regex,
    luhn: bool,
}

/// PII redactor.
pub struct PiiRedactor {
    rules: Vec<CompiledRule>,
}

impl PiiRedactor {
    /// Build the redactor with the default rule set.
    pub fn new() -> Result<Self> {
        let raw: &[(&str, &str, bool)] = &[
            ("EMAIL", r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}", false),
            ("SSN", r"\b\d{3}-\d{2}-\d{4}\b", false),
            ("PHONE", r"(?:\+?1[-. ]?)?\(?\d{3}\)?[-. ]?\d{3}[-. ]?\d{4}\b", false),
            ("CREDIT_CARD", r"\b(?:\d[ -]?){13,19}\b", true),
            ("IPV4", r"\b(?:\d{1,3}\.){3}\d{1,3}\b", false),
            ("IPV6", r"\b(?:[A-Fa-f0-9]{1,4}:){7}[A-Fa-f0-9]{1,4}\b", false),
            ("US_ADDRESS", r"\b\d{1,5} [A-Z][a-zA-Z]+(?: [A-Z][a-zA-Z]+){0,3} (?:St|Street|Ave|Avenue|Rd|Road|Blvd|Boulevard|Ln|Lane|Dr|Drive|Way|Ct|Court|Pl|Place)\b", false),
            ("GIVEN_NAME", r"\b(?:John|Jane|Alice|Bob|Carol|David|Eve|Frank|Grace|Henry|Ivy|Jack|Karen|Larry|Mary|Nancy|Oscar|Peggy|Quinn|Rachel|Sam|Tina|Uma|Victor|Wendy|Xavier|Yvonne|Zach)\b", false),
        ];
        let mut rules = Vec::with_capacity(raw.len());
        for (cat, pat, luhn) in raw {
            rules.push(CompiledRule {
                category: cat,
                re: Regex::new(pat)?,
                luhn: *luhn,
            });
        }
        Ok(Self { rules })
    }

    /// Run the redactor over `input`.
    pub fn redact(&self, input: &str) -> PiiReport {
        let mut spans: Vec<PiiSpan> = Vec::new();
        for rule in &self.rules {
            for m in rule.re.find_iter(input) {
                if rule.luhn && !luhn_valid(m.as_str()) {
                    continue;
                }
                spans.push(PiiSpan {
                    category: rule.category.to_string(),
                    start: m.start(),
                    end: m.end(),
                    text: m.as_str().to_string(),
                });
            }
        }
        spans.sort_by_key(|s| (s.start, std::cmp::Reverse(s.end)));
        // Drop overlapping spans (keep the leftmost-longest).
        let mut deduped: Vec<PiiSpan> = Vec::with_capacity(spans.len());
        for span in spans {
            if let Some(last) = deduped.last() {
                if span.start < last.end {
                    continue;
                }
            }
            deduped.push(span);
        }

        let mut out = String::with_capacity(input.len());
        let mut cursor = 0usize;
        for span in &deduped {
            out.push_str(&input[cursor..span.start]);
            out.push('<');
            out.push_str(&span.category);
            out.push('>');
            cursor = span.end;
        }
        out.push_str(&input[cursor..]);

        PiiReport {
            redacted: out,
            spans: deduped,
        }
    }
}

/// Luhn check used for credit-card validation.
fn luhn_valid(s: &str) -> bool {
    let digits: Vec<u32> = s.chars().filter_map(|c| c.to_digit(10)).collect();
    if digits.len() < 13 || digits.len() > 19 {
        return false;
    }
    let mut sum = 0u32;
    let mut alt = false;
    for d in digits.iter().rev() {
        let mut x = *d;
        if alt {
            x *= 2;
            if x > 9 {
                x -= 9;
            }
        }
        sum += x;
        alt = !alt;
    }
    sum % 10 == 0
}
