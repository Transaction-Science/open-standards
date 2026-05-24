//! Jailbreak detector — DAN, DUDE, Grandma, STAN, AIM, Developer-Mode,
//! and the broader "do anything now" family.
//!
//! Sources: jailbreakchat.com archive (public domain), ChatML / Anthropic
//! red-team disclosures, and the Garak `dan` probe set (Apache-2.0).

use regex::RegexSet;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::injection::Detector;

/// Verdict from the jailbreak detector.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct JailbreakVerdict {
    /// Score in `[0.0, 1.0]`.
    pub score: f32,
    /// Identified jailbreak family names (e.g. `"dan"`, `"grandma"`).
    pub families: Vec<String>,
    /// Whether the detector recommends rejecting the input.
    pub reject: bool,
}

impl JailbreakVerdict {
    /// Clean verdict — no jailbreak family detected.
    pub fn clean() -> Self {
        Self {
            score: 0.0,
            families: Vec::new(),
            reject: false,
        }
    }
}

/// Known jailbreak families. Each entry is `(family_name, regex)`.
pub const FAMILIES: &[(&str, &str)] = &[
    ("dan", r"\bdan\b|do anything now"),
    ("dude", r"\bdude\b.*(no restrictions?|anything)"),
    ("stan", r"\bstan\b.*strive to avoid norms"),
    ("aim", r"\baim\b.*always intelligent and machiavellian"),
    ("grandma", r"(my )?(grand[mp]a|grandmother|grandfather).*(used to|would).*(tell|read|recite|sing).*(napalm|malware|exploit|password|key|recipe)"),
    ("developer_mode", r"developer mode (enabled|on)|enable developer mode"),
    ("evil_confidant", r"evil (confidant|trusted|friend|assistant)"),
    ("token_smuggling", r"<\|im_(start|end)\|>|<\|endoftext\|>"),
    ("roleplay_unrestricted", r"role[- ]?play.*(no (rules?|restrictions?|filters?))"),
    ("opposite_day", r"opposite day|invert (your )?(rules?|guidelines?)"),
    ("hypothetical", r"hypothetical(ly)?.*(harmful|illegal|dangerous|unsafe)"),
    ("translator_loophole", r"translate.*(into|to).*(then|and).*(ignore|disregard)"),
    ("fictional_universe", r"in a (fictional|alternate|parallel) (universe|world|reality).*(no laws?|no rules?)"),
];

/// Default jailbreak detector.
pub struct JailbreakDetector {
    set: RegexSet,
    names: Vec<&'static str>,
    threshold: f32,
}

impl JailbreakDetector {
    /// Build the detector from the bundled family list.
    pub fn new() -> Result<Self> {
        Self::with_families(FAMILIES, 0.5)
    }

    /// Build a detector from a custom family list.
    pub fn with_families(fams: &[(&'static str, &str)], threshold: f32) -> Result<Self> {
        let patterns: Vec<&str> = fams.iter().map(|(_, p)| *p).collect();
        let names: Vec<&'static str> = fams.iter().map(|(n, _)| *n).collect();
        Ok(Self {
            set: RegexSet::new(&patterns)?,
            names,
            threshold,
        })
    }

    /// Synchronous inspection (regex-only).
    pub fn inspect_sync(&self, input: &str) -> JailbreakVerdict {
        let lower = input.to_lowercase();
        let hits: Vec<usize> = self.set.matches(&lower).into_iter().collect();
        if hits.is_empty() {
            return JailbreakVerdict::clean();
        }
        let families: Vec<String> = hits.iter().map(|i| self.names[*i].to_string()).collect();
        let n = hits.len() as f32;
        let score = 1.0 - (-0.9 * n).exp();
        JailbreakVerdict {
            score,
            reject: score >= self.threshold,
            families,
        }
    }
}

#[async_trait::async_trait]
impl Detector for JailbreakDetector {
    fn name(&self) -> &'static str {
        "jailbreak"
    }

    async fn inspect(&self, input: &str) -> Result<crate::injection::InjectionVerdict> {
        let v = self.inspect_sync(input);
        Ok(crate::injection::InjectionVerdict {
            score: v.score,
            signatures: v.families,
            reject: v.reject,
        })
    }
}
