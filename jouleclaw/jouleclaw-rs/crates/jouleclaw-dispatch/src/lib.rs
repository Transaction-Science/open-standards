//! Cost-table model-bank dispatcher — the compartment is a *bank*, not a
//! singleton.
//!
//! "AI lives in the compartment" is not a commitment to one model.
//! Different query classes have structurally — not marginally —
//! different cheapest reasoners: a short factual query, a long-context
//! read, a visual question, and a pure embedding belong on different
//! hardware with order-of-magnitude energy spreads. NemoClaw routes
//! inference; joule-state names it rule 14. This crate implements it:
//! classify the query into a [`Modality`], then pick the **cheapest
//! capable** reasoner slot from a [`ModelBank`] by its measured joule
//! cost — minimum-energy search over the cost table, not a default
//! architecture.
//!
//! The headline win is not exotic model selection; it is **avoiding the
//! language-tax on non-language work** — never sending a short or
//! structured query to a frontier text model when a small specialist
//! resolves it for a fraction of the joules.
//!
//! Slots reuse the [`jouleclaw_llm_cheap::LlmBackend`] trait, so any
//! backend already written for the L3 tier drops into a bank. Per-slot
//! `typical_joules` is the cost-table entry — populated in production by
//! `substrate-energy` measurement on the deployed hardware, configured
//! here.

#![forbid(unsafe_code)]

use std::time::Duration;

use jouleclaw_cascade::tier::{Tier, TierEstimate};
use jouleclaw_cascade::types::{
    Answer, AnswerError, AnswerOutput, ExecutionTrace, L3ModelId, Query, QueryInput, RefusalReason,
    TierId,
};
use jouleclaw_cascade::verification::VerificationStatus;
use jouleclaw_llm_cheap::{FinishReason, LlmBackend, LlmRequest};

/// Default char count above which a text query is treated as long-context.
pub const LONG_CONTEXT_CHARS: usize = 4_000;
/// Default per-completion token cap requested from a slot.
pub const DEFAULT_MAX_TOKENS: u32 = 256;

/// The class of work a query represents — the routing key. Slots declare
/// which modalities they can serve; the dispatcher routes to the cheapest
/// capable slot.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Modality {
    /// Short factual / conversational text.
    ShortText,
    /// Long text that needs a large context window.
    LongContext,
    /// Multi-step reasoning over text.
    Reasoning,
    /// Structured / JSON input.
    Structured,
    /// Image input.
    Visual,
    /// Audio input.
    Audio,
}

/// Classifies a query into a [`Modality`]. Returns `None` for inputs no
/// slot can serve (e.g. raw binary).
pub trait ModalityClassifier: Send + Sync {
    fn classify(&self, q: &Query) -> Option<Modality>;
}

/// Deterministic heuristic classifier — length + modality + reasoning
/// cues. Pure; no model.
#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicClassifier;

const REASONING_CUES: &[&str] = &[
    "why", "explain", "prove", "reason", "step by step", "derive", "justify", "how does",
];

impl ModalityClassifier for HeuristicClassifier {
    fn classify(&self, q: &Query) -> Option<Modality> {
        match &q.input {
            QueryInput::Image(_) => Some(Modality::Visual),
            QueryInput::Audio(_) => Some(Modality::Audio),
            QueryInput::Structured(_) => Some(Modality::Structured),
            QueryInput::Binary(_) => None,
            QueryInput::Multimodal { images, audio, text } => {
                if !images.is_empty() {
                    Some(Modality::Visual)
                } else if !audio.is_empty() {
                    Some(Modality::Audio)
                } else {
                    Some(classify_text(text))
                }
            }
            QueryInput::Text(t) => Some(classify_text(t)),
        }
    }
}

fn classify_text(t: &str) -> Modality {
    if t.chars().count() > LONG_CONTEXT_CHARS {
        return Modality::LongContext;
    }
    let lower = t.to_lowercase();
    if REASONING_CUES.iter().any(|c| lower.contains(c)) {
        Modality::Reasoning
    } else {
        Modality::ShortText
    }
}

/// One reasoner in the bank: a backend plus its cost-table entry.
pub struct ReasonerSlot {
    /// Stable slot name (surfaced in traces).
    pub name: String,
    /// Modalities this slot can serve.
    pub handles: Vec<Modality>,
    /// Measured typical joules per call (the cost-table value).
    pub typical_joules: f64,
    /// The backend (reuses the L3 tier's trait).
    pub backend: Box<dyn LlmBackend>,
}

impl ReasonerSlot {
    pub fn new(
        name: impl Into<String>,
        handles: Vec<Modality>,
        typical_joules: f64,
        backend: Box<dyn LlmBackend>,
    ) -> Self {
        Self {
            name: name.into(),
            handles,
            typical_joules,
            backend,
        }
    }

    fn can_serve(&self, m: Modality) -> bool {
        self.handles.contains(&m)
    }
}

/// A bank of reasoner slots with cost-table-driven selection.
#[derive(Default)]
pub struct ModelBank {
    slots: Vec<ReasonerSlot>,
}

impl ModelBank {
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a slot (builder style).
    pub fn with_slot(mut self, slot: ReasonerSlot) -> Self {
        self.slots.push(slot);
        self
    }

    pub fn add_slot(&mut self, slot: ReasonerSlot) {
        self.slots.push(slot);
    }

    pub fn len(&self) -> usize {
        self.slots.len()
    }

    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    /// Index of the cheapest slot that can serve `modality` within
    /// `max_joules`. `None` if no capable, affordable slot exists. Ties
    /// resolve to the earliest-registered slot (deterministic).
    pub fn select(&self, modality: Modality, max_joules: f64) -> Option<usize> {
        self.slots
            .iter()
            .enumerate()
            .filter(|(_, s)| s.can_serve(modality) && s.typical_joules <= max_joules)
            .min_by(|(_, a), (_, b)| {
                a.typical_joules
                    .partial_cmp(&b.typical_joules)
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .map(|(i, _)| i)
    }

    /// Cheapest slot that can serve `modality` ignoring budget (for
    /// `estimate_cost`). `None` if no slot serves it.
    fn cheapest_capable(&self, modality: Modality) -> Option<usize> {
        self.select(modality, f64::INFINITY)
    }
}

/// Confidence implied by a finish reason; mirrors the L3 tier's mapping.
fn confidence_of(reason: &FinishReason) -> Option<f32> {
    match reason {
        FinishReason::Stop => Some(0.8),
        FinishReason::Length => Some(0.6),
        // ContentFilter / Error → no usable answer (refuse).
        FinishReason::ContentFilter | FinishReason::Error(_) => None,
    }
}

/// Errors from the dispatcher.
#[derive(Debug, thiserror::Error)]
pub enum DispatchError {
    #[error("model bank is empty")]
    EmptyBank,
}

/// L3 model-bank tier: classify, then dispatch to the cheapest capable
/// slot. Replaces a single-backend L3 with a cost-table-routed bank.
pub struct ModelBankTier<C: ModalityClassifier = HeuristicClassifier> {
    bank: ModelBank,
    classifier: C,
    max_tokens: u32,
}

impl ModelBankTier<HeuristicClassifier> {
    /// Build a tier over `bank` with the default heuristic classifier.
    pub fn new(bank: ModelBank) -> Result<Self, DispatchError> {
        if bank.is_empty() {
            return Err(DispatchError::EmptyBank);
        }
        Ok(Self {
            bank,
            classifier: HeuristicClassifier,
            max_tokens: DEFAULT_MAX_TOKENS,
        })
    }
}

impl<C: ModalityClassifier> ModelBankTier<C> {
    /// Build with a custom classifier.
    pub fn with_classifier(bank: ModelBank, classifier: C) -> Result<Self, DispatchError> {
        if bank.is_empty() {
            return Err(DispatchError::EmptyBank);
        }
        Ok(Self {
            bank,
            classifier,
            max_tokens: DEFAULT_MAX_TOKENS,
        })
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens.max(1);
        self
    }

    fn prompt_of(q: &Query) -> Option<String> {
        match &q.input {
            QueryInput::Text(t) => Some(t.clone()),
            QueryInput::Multimodal { text, .. } => Some(text.clone()),
            QueryInput::Structured(b) => std::str::from_utf8(b).ok().map(|s| s.to_string()),
            _ => None,
        }
    }
}

impl<C: ModalityClassifier + 'static> Tier for ModelBankTier<C> {
    fn id(&self) -> TierId {
        TierId::L3(L3ModelId(0))
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        let modality = self.classifier.classify(q)?;
        let idx = self.bank.cheapest_capable(modality)?;
        let slot = &self.bank.slots[idx];
        Some(TierEstimate {
            joules: slot.typical_joules,
            latency: Duration::from_secs(3),
            confidence_floor: 0.5,
        })
    }

    fn try_answer(&mut self, q: &Query, budget_remaining: f64) -> Result<Answer, AnswerError> {
        let id = self.id();
        let Some(modality) = self.classifier.classify(q) else {
            return Ok(refused(id, RefusalReason::Inapplicable));
        };
        let Some(prompt) = Self::prompt_of(q) else {
            return Ok(refused(id, RefusalReason::Inapplicable));
        };
        // Cheapest capable slot within the remaining budget.
        let Some(idx) = self.bank.select(modality, budget_remaining) else {
            return Ok(refused(
                id,
                RefusalReason::TierSpecific(format!(
                    "no slot for {modality:?} within {budget_remaining:.3e} J"
                )),
            ));
        };
        let slot = &self.bank.slots[idx];
        let request = LlmRequest::from_prompt(prompt, self.max_tokens);
        match slot.backend.complete(&request) {
            Ok(resp) => {
                let joules = resp.energy_joules.unwrap_or(slot.typical_joules);
                match confidence_of(&resp.finish_reason) {
                    Some(confidence) => Ok(Answer {
                        output: AnswerOutput::Text(resp.text),
                        tier_used: id,
                        joules_spent: joules,
                        confidence,
                        trace: ExecutionTrace::default(),
                        verification: VerificationStatus::Resolved,
                    }),
                    None => Ok(refused_spent(
                        id,
                        joules,
                        RefusalReason::TierSpecific(format!("slot `{}` produced no answer", slot.name)),
                    )),
                }
            }
            Err(e) => Ok(refused_spent(
                id,
                slot.typical_joules,
                RefusalReason::TierSpecific(format!("slot `{}` failed: {e}", slot.name)),
            )),
        }
    }
}

fn refused(tier: TierId, reason: RefusalReason) -> Answer {
    refused_spent(tier, 0.0, reason)
}

fn refused_spent(tier: TierId, joules: f64, reason: RefusalReason) -> Answer {
    Answer {
        output: AnswerOutput::Refused(reason),
        tier_used: tier,
        joules_spent: joules,
        confidence: 0.0,
        trace: ExecutionTrace::default(),
        verification: VerificationStatus::Resolved,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_cascade::types::{ContextRef, JouleBudget, QualityFloor};
    use jouleclaw_llm_cheap::EchoBackend;

    fn text(s: &str) -> Query {
        Query {
            input: QueryInput::Text(s.to_string()),
            budget: JouleBudget::expensive(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    /// A bank: a cheap short-text specialist + an expensive long-context
    /// model. Echo backends with distinct names/costs stand in for real
    /// reasoners.
    fn bank() -> ModelBank {
        ModelBank::new()
            .with_slot(ReasonerSlot::new(
                "small-dense",
                vec![Modality::ShortText, Modality::Reasoning],
                0.5,
                Box::new(EchoBackend::new().with_name("small-dense").with_typical_joules(0.5)),
            ))
            .with_slot(ReasonerSlot::new(
                "long-ctx",
                vec![Modality::LongContext, Modality::ShortText],
                12.0,
                Box::new(EchoBackend::new().with_name("long-ctx").with_typical_joules(12.0)),
            ))
    }

    #[test]
    fn classifier_separates_modalities() {
        let c = HeuristicClassifier;
        assert_eq!(c.classify(&text("capital of france")), Some(Modality::ShortText));
        assert_eq!(c.classify(&text("explain why the sky is blue")), Some(Modality::Reasoning));
        let long = "x".repeat(LONG_CONTEXT_CHARS + 1);
        assert_eq!(c.classify(&text(&long)), Some(Modality::LongContext));
    }

    #[test]
    fn binary_input_is_unclassifiable() {
        let c = HeuristicClassifier;
        let q = Query {
            input: QueryInput::Binary(vec![1, 2, 3]),
            budget: JouleBudget::expensive(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        assert!(c.classify(&q).is_none());
    }

    #[test]
    fn select_picks_cheapest_capable() {
        let b = bank();
        // ShortText: both can serve; small-dense (0.5) is cheaper.
        assert_eq!(b.select(Modality::ShortText, f64::INFINITY), Some(0));
        // LongContext: only long-ctx serves it.
        assert_eq!(b.select(Modality::LongContext, f64::INFINITY), Some(1));
        // Visual: nobody serves it.
        assert_eq!(b.select(Modality::Visual, f64::INFINITY), None);
    }

    #[test]
    fn budget_excludes_unaffordable_slots() {
        let b = bank();
        // A long-context query but only 1 J budget → the 12 J slot is
        // unaffordable, and no cheap slot serves LongContext → None.
        assert_eq!(b.select(Modality::LongContext, 1.0), None);
        // Short text within 1 J → the 0.5 J slot.
        assert_eq!(b.select(Modality::ShortText, 1.0), Some(0));
    }

    #[test]
    fn tier_routes_short_query_to_cheap_slot() {
        let mut tier = ModelBankTier::new(bank()).unwrap();
        let ans = tier.try_answer(&text("capital of france"), 100.0).unwrap();
        match ans.output {
            AnswerOutput::Text(t) => assert!(t.contains("capital of france")),
            other => panic!("expected text, got {other:?}"),
        }
        // Routed to the cheap slot — joules ≈ 0.5, not 12.
        assert!((ans.joules_spent - 0.5).abs() < 1e-6, "spent {}", ans.joules_spent);
    }

    #[test]
    fn tier_routes_long_query_to_long_context_slot() {
        let mut tier = ModelBankTier::new(bank()).unwrap();
        let long = format!("summarize: {}", "word ".repeat(LONG_CONTEXT_CHARS));
        let ans = tier.try_answer(&text(&long), 100.0).unwrap();
        // Only the 12 J long-context slot can serve it.
        assert!((ans.joules_spent - 12.0).abs() < 1e-6, "spent {}", ans.joules_spent);
    }

    #[test]
    fn tier_refuses_when_no_affordable_slot() {
        let mut tier = ModelBankTier::new(bank()).unwrap();
        let long = format!("explain: {}", "word ".repeat(LONG_CONTEXT_CHARS));
        // LongContext needs the 12 J slot; only 1 J remains → refuse.
        let ans = tier.try_answer(&text(&long), 1.0).unwrap();
        assert!(matches!(ans.output, AnswerOutput::Refused(_)));
    }

    #[test]
    fn empty_bank_rejected() {
        assert!(matches!(ModelBankTier::new(ModelBank::new()), Err(DispatchError::EmptyBank)));
    }

    #[test]
    fn estimate_reflects_cheapest_capable_slot() {
        let tier = ModelBankTier::new(bank()).unwrap();
        let est = tier.estimate_cost(&text("hi")).unwrap();
        assert!((est.joules - 0.5).abs() < 1e-6); // cheap slot
        // No slot for raw binary → no estimate.
        let q = Query {
            input: QueryInput::Binary(vec![1]),
            budget: JouleBudget::expensive(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        assert!(tier.estimate_cost(&q).is_none());
    }

    #[test]
    fn tier_via_cascade_runtime() {
        use jouleclaw_cascade::tier::{Cascade, Runtime};
        let mut cascade = Cascade::new();
        cascade.register(Box::new(ModelBankTier::new(bank()).unwrap()));
        let mut rt = Runtime::new_without_l0(cascade);
        let ans = rt.answer(text("a short question")).unwrap();
        assert!(matches!(ans.tier_used, TierId::L3(_)));
        assert!((ans.joules_spent - 0.5).abs() < 1e-6); // cheapest capable
    }
}
