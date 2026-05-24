//! Constitutional-AI critique → revise loop.
//!
//! Implements the two-step pattern from Anthropic's
//! *Constitutional AI: Harmlessness from AI Feedback* (Bai et al.,
//! 2022) in an LLM-agnostic, deterministic shell:
//!
//! 1. Run the model to produce a draft answer.
//! 2. For each principle in the [`Constitution`], ask the model to
//!    critique the draft against that principle.
//! 3. If any critique flags a violation, ask the model to revise the
//!    answer.
//!
//! This crate provides the orchestration; the actual LLM calls are
//! delegated to a [`CritiqueModel`] trait so callers can wire in any
//! backend (e.g. `eoc_neural::NeuralBackend`).

use serde::{Deserialize, Serialize};

use crate::error::Result;

/// One principle in a [`Constitution`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Principle {
    /// Short slug used in receipts.
    pub id: String,
    /// Human-readable rule the model must obey.
    pub rule: String,
}

/// A set of [`Principle`]s.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Constitution {
    /// Ordered principles. Earlier principles are critiqued first.
    pub principles: Vec<Principle>,
}

impl Constitution {
    /// Empty constitution.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a principle and return `self` (builder-style).
    pub fn with(mut self, id: impl Into<String>, rule: impl Into<String>) -> Self {
        self.principles.push(Principle {
            id: id.into(),
            rule: rule.into(),
        });
        self
    }

    /// Default constitution adapted from the Anthropic paper.
    pub fn anthropic_default() -> Self {
        Self::new()
            .with("harmless", "The response must not help with harm to humans.")
            .with("honest", "The response must not contain knowingly false statements.")
            .with("helpful", "The response must address the user's actual question.")
            .with("non_violent", "The response must not advocate violence.")
            .with("non_discriminatory", "The response must not stereotype protected groups.")
    }
}

/// Outcome of critiquing a single principle.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CritiqueOutcome {
    /// Principle that was applied.
    pub principle_id: String,
    /// Free-text critique returned by the model.
    pub critique: String,
    /// Did the model flag the draft as violating this principle?
    pub flagged: bool,
}

/// Full report from one critique→revise round.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConstitutionReport {
    /// Original draft answer.
    pub draft: String,
    /// Final (possibly revised) answer.
    pub revised: String,
    /// Per-principle critiques.
    pub critiques: Vec<CritiqueOutcome>,
    /// Whether a revision was actually performed.
    pub revised_applied: bool,
}

/// Pluggable critique / revision model.
#[async_trait::async_trait]
pub trait CritiqueModel: Send + Sync {
    /// Return a free-text critique of `draft` under `principle`. The
    /// implementation must return `(critique, flagged)`.
    async fn critique(
        &self,
        question: &str,
        draft: &str,
        principle: &Principle,
    ) -> Result<(String, bool)>;

    /// Return a revised answer addressing the supplied critiques.
    async fn revise(
        &self,
        question: &str,
        draft: &str,
        critiques: &[CritiqueOutcome],
    ) -> Result<String>;
}

/// Run one critique→revise round.
pub async fn run_round<M: CritiqueModel>(
    model: &M,
    constitution: &Constitution,
    question: &str,
    draft: &str,
) -> Result<ConstitutionReport> {
    let mut critiques: Vec<CritiqueOutcome> = Vec::with_capacity(constitution.principles.len());
    for p in &constitution.principles {
        let (text, flagged) = model.critique(question, draft, p).await?;
        critiques.push(CritiqueOutcome {
            principle_id: p.id.clone(),
            critique: text,
            flagged,
        });
    }
    let needs_revise = critiques.iter().any(|c| c.flagged);
    let revised = if needs_revise {
        model.revise(question, draft, &critiques).await?
    } else {
        draft.to_string()
    };
    Ok(ConstitutionReport {
        draft: draft.to_string(),
        revised,
        critiques,
        revised_applied: needs_revise,
    })
}

/// Reference implementation that flags drafts containing any term in
/// `keywords`. Useful for tests; not for production.
pub struct KeywordCritiqueModel {
    /// Trigger keywords (lower-cased on construction).
    pub keywords: Vec<String>,
    /// Substitution text used when a revision is requested.
    pub safe_response: String,
}

impl KeywordCritiqueModel {
    /// Build with the supplied trigger keywords and a static safe answer.
    pub fn new(keywords: Vec<&str>, safe_response: &str) -> Self {
        Self {
            keywords: keywords.into_iter().map(|s| s.to_lowercase()).collect(),
            safe_response: safe_response.to_string(),
        }
    }
}

#[async_trait::async_trait]
impl CritiqueModel for KeywordCritiqueModel {
    async fn critique(
        &self,
        _question: &str,
        draft: &str,
        principle: &Principle,
    ) -> Result<(String, bool)> {
        let lower = draft.to_lowercase();
        let flagged = self.keywords.iter().any(|k| lower.contains(k));
        let critique = if flagged {
            format!("Draft violates principle '{}'.", principle.id)
        } else {
            format!("Draft satisfies principle '{}'.", principle.id)
        };
        Ok((critique, flagged))
    }

    async fn revise(
        &self,
        _question: &str,
        _draft: &str,
        _critiques: &[CritiqueOutcome],
    ) -> Result<String> {
        Ok(self.safe_response.clone())
    }
}
