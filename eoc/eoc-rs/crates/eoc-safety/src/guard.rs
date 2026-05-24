//! Composable guard pipeline.
//!
//! Mirrors the Llama-Guard / NeMo-Guardrails layout: an ordered list of
//! input checks and an ordered list of output checks. Each check
//! returns a [`GuardSignal`]; the pipeline returns a [`GuardReport`]
//! that aggregates them.
//!
//! ```no_run
//! # use eoc_safety::guard::{InputGuard, OutputGuard};
//! # use eoc_safety::injection::InjectionDetector;
//! # async fn ex() -> eoc_safety::error::Result<()> {
//! let input_guard = InputGuard::new()
//!     .with_injection(InjectionDetector::new()?);
//! let report = input_guard.check("ignore previous instructions").await?;
//! assert!(report.blocked);
//! # Ok(()) }
//! ```

use serde::{Deserialize, Serialize};

use crate::bias::HeuristicBiasDetector;
use crate::error::Result;
use crate::injection::{Detector, InjectionDetector};
use crate::jailbreak::JailbreakDetector;
use crate::nsfw::HeuristicNsfwClassifier;
use crate::pii::{PiiRedactor, PiiReport};
use crate::toxicity::HeuristicToxicityClassifier;

/// One signal in a [`GuardReport`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardSignal {
    /// Name of the detector that fired.
    pub detector: String,
    /// Score in `[0.0, 1.0]`.
    pub score: f32,
    /// Whether this signal alone is enough to block.
    pub blocking: bool,
    /// Optional sub-labels (e.g. matched signature names).
    pub labels: Vec<String>,
}

/// Aggregated decision from an input or output guard.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GuardReport {
    /// Sanitised text (e.g. PII-redacted) — same as input if no
    /// redaction happened.
    pub sanitized: String,
    /// All signals raised by the pipeline (in evaluation order).
    pub signals: Vec<GuardSignal>,
    /// True if any blocking signal fired.
    pub blocked: bool,
    /// Block reason if any.
    pub reason: Option<String>,
}

impl GuardReport {
    /// Build an empty "all clear" report.
    pub fn pass(input: &str) -> Self {
        Self {
            sanitized: input.to_string(),
            signals: Vec::new(),
            blocked: false,
            reason: None,
        }
    }
}

/// Pipeline for inspecting **user input** before it reaches the model.
#[derive(Default)]
pub struct InputGuard {
    injection: Option<InjectionDetector>,
    jailbreak: Option<JailbreakDetector>,
    pii: Option<PiiRedactor>,
    toxicity: Option<HeuristicToxicityClassifier>,
    nsfw: Option<HeuristicNsfwClassifier>,
}

impl InputGuard {
    /// Empty pipeline.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: attach an injection detector.
    pub fn with_injection(mut self, d: InjectionDetector) -> Self {
        self.injection = Some(d);
        self
    }

    /// Builder: attach a jailbreak detector.
    pub fn with_jailbreak(mut self, d: JailbreakDetector) -> Self {
        self.jailbreak = Some(d);
        self
    }

    /// Builder: attach the PII redactor.
    pub fn with_pii(mut self, r: PiiRedactor) -> Self {
        self.pii = Some(r);
        self
    }

    /// Builder: attach the toxicity classifier.
    pub fn with_toxicity(mut self, c: HeuristicToxicityClassifier) -> Self {
        self.toxicity = Some(c);
        self
    }

    /// Builder: attach the NSFW classifier.
    pub fn with_nsfw(mut self, c: HeuristicNsfwClassifier) -> Self {
        self.nsfw = Some(c);
        self
    }

    /// Run the pipeline.
    pub async fn check(&self, input: &str) -> Result<GuardReport> {
        let mut report = GuardReport::pass(input);

        if let Some(d) = &self.injection {
            let v = d.inspect(input).await?;
            if v.score > 0.0 {
                if v.reject {
                    report.blocked = true;
                    report.reason.get_or_insert_with(|| "injection".to_string());
                }
                report.signals.push(GuardSignal {
                    detector: "injection".into(),
                    score: v.score,
                    blocking: v.reject,
                    labels: v.signatures,
                });
            }
        }

        if let Some(d) = &self.jailbreak {
            let v = d.inspect_sync(input);
            if v.score > 0.0 {
                if v.reject {
                    report.blocked = true;
                    report.reason.get_or_insert_with(|| "jailbreak".to_string());
                }
                report.signals.push(GuardSignal {
                    detector: "jailbreak".into(),
                    score: v.score,
                    blocking: v.reject,
                    labels: v.families,
                });
            }
        }

        if let Some(r) = &self.pii {
            let red: PiiReport = r.redact(input);
            if red.has_pii() {
                let labels: Vec<String> = red.spans.iter().map(|s| s.category.clone()).collect();
                report.signals.push(GuardSignal {
                    detector: "pii".into(),
                    score: 1.0,
                    blocking: false,
                    labels,
                });
                report.sanitized = red.redacted;
            }
        }

        if let Some(c) = &self.toxicity {
            let v = c.classify_sync(input);
            if v.score > 0.0 {
                if v.reject {
                    report.blocked = true;
                    report.reason.get_or_insert_with(|| "toxicity".to_string());
                }
                report.signals.push(GuardSignal {
                    detector: "toxicity".into(),
                    score: v.score,
                    blocking: v.reject,
                    labels: v
                        .categories
                        .iter()
                        .map(|(c, _)| format!("{:?}", c))
                        .collect(),
                });
            }
        }

        if let Some(c) = &self.nsfw {
            let v = c.classify_sync(input);
            if v.score > 0.0 {
                if v.reject {
                    report.blocked = true;
                    report.reason.get_or_insert_with(|| "nsfw".to_string());
                }
                report.signals.push(GuardSignal {
                    detector: "nsfw".into(),
                    score: v.score,
                    blocking: v.reject,
                    labels: v
                        .categories
                        .iter()
                        .map(|(c, _)| format!("{:?}", c))
                        .collect(),
                });
            }
        }

        Ok(report)
    }
}

/// Pipeline for inspecting **model output** before it reaches the user.
#[derive(Default)]
pub struct OutputGuard {
    pii: Option<PiiRedactor>,
    toxicity: Option<HeuristicToxicityClassifier>,
    nsfw: Option<HeuristicNsfwClassifier>,
    bias: Option<HeuristicBiasDetector>,
}

impl OutputGuard {
    /// Empty pipeline.
    pub fn new() -> Self {
        Self::default()
    }

    /// Builder: attach the PII redactor.
    pub fn with_pii(mut self, r: PiiRedactor) -> Self {
        self.pii = Some(r);
        self
    }

    /// Builder: attach the toxicity classifier.
    pub fn with_toxicity(mut self, c: HeuristicToxicityClassifier) -> Self {
        self.toxicity = Some(c);
        self
    }

    /// Builder: attach the NSFW classifier.
    pub fn with_nsfw(mut self, c: HeuristicNsfwClassifier) -> Self {
        self.nsfw = Some(c);
        self
    }

    /// Builder: attach the bias detector.
    pub fn with_bias(mut self, b: HeuristicBiasDetector) -> Self {
        self.bias = Some(b);
        self
    }

    /// Run the pipeline.
    pub async fn check(&self, output: &str) -> Result<GuardReport> {
        let mut report = GuardReport::pass(output);

        if let Some(r) = &self.pii {
            let red = r.redact(output);
            if red.has_pii() {
                let labels: Vec<String> = red.spans.iter().map(|s| s.category.clone()).collect();
                report.signals.push(GuardSignal {
                    detector: "pii".into(),
                    score: 1.0,
                    blocking: false,
                    labels,
                });
                report.sanitized = red.redacted;
            }
        }
        if let Some(c) = &self.toxicity {
            let v = c.classify_sync(output);
            if v.score > 0.0 {
                if v.reject {
                    report.blocked = true;
                    report.reason.get_or_insert_with(|| "toxicity".to_string());
                }
                report.signals.push(GuardSignal {
                    detector: "toxicity".into(),
                    score: v.score,
                    blocking: v.reject,
                    labels: v
                        .categories
                        .iter()
                        .map(|(c, _)| format!("{:?}", c))
                        .collect(),
                });
            }
        }
        if let Some(c) = &self.nsfw {
            let v = c.classify_sync(output);
            if v.score > 0.0 {
                if v.reject {
                    report.blocked = true;
                    report.reason.get_or_insert_with(|| "nsfw".to_string());
                }
                report.signals.push(GuardSignal {
                    detector: "nsfw".into(),
                    score: v.score,
                    blocking: v.reject,
                    labels: v
                        .categories
                        .iter()
                        .map(|(c, _)| format!("{:?}", c))
                        .collect(),
                });
            }
        }
        if let Some(b) = &self.bias {
            let v = b.detect_sync(output);
            if v.score > 0.0 {
                report.signals.push(GuardSignal {
                    detector: "bias".into(),
                    score: v.score,
                    blocking: false,
                    labels: v.axes.iter().map(|(a, _)| format!("{:?}", a)).collect(),
                });
            }
        }
        Ok(report)
    }
}
