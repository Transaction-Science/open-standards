//! `LiquidTier` — joule cascade tier shell for CfC-based recurrent models.
//!
//! R29.0 lands the kernel: a verified, deterministic CfC cell stack
//! ([`crate::cell::CfcCell`], [`crate::model::LiquidModel`]). The cascade
//! integration point is in place; full token-level inference (tokenizer,
//! embedding, output projection, sampling, weight loading) lands in R29.1.
//!
//! Coordinate:
//!
//!   Z = Z2_3        (small-to-medium statistical inference)
//!   E = Reactive
//!   T = L1_Measure  (well above Landauer, measurable per-op)
//!   I = Tokens
//!   V = Statistical (single samples may be wrong; the distribution is the claim)
//!   R = Facts
//!   P = { StateSpace, Sample }
//!
//! `StateSpace` is the most accurate P-axis label for CfC: a recurrent
//! dynamical system whose state evolves continuously in time. This sits
//! alongside transformer-style `AttentionGrouped` tiers in the cascade,
//! offering an alternative architectural axis for sequence modeling.

use std::time::Duration;

use jouleclaw_cascade::*;

use crate::lm::LiquidLanguageModel;
use crate::model::LiquidModel;

/// Cascade tier wrapping a Liquid (CfC-recurrent) inference path.
pub struct LiquidTier {
    /// R29.0 path: bare CfC stack with no LM-head wrapper. Retained for
    /// register-and-refuse smoke tests; `try_answer` refuses when only
    /// this field is set.
    pub model: Option<LiquidModel>,
    /// R29.1 path: full recurrent language model (embedding + CfC stack
    /// + LM head). When present, `try_answer` runs `generate_greedy` and
    /// returns Text.
    pub lm: Option<LiquidLanguageModel>,
    /// Tokens to generate beyond the prompt. Default 16.
    pub max_new_tokens: usize,
    /// Stable model ID surfaced as `TierId::L3(L3ModelId(model_id))`.
    pub model_id: u32,
}

impl LiquidTier {
    /// Empty tier — coordinate declared, floor cost reported, all queries refused.
    pub fn empty(model_id: u32) -> Self {
        Self { model: None, lm: None, max_new_tokens: 16, model_id }
    }

    pub fn from_model(model_id: u32, model: LiquidModel) -> Self {
        Self { model: Some(model), lm: None, max_new_tokens: 16, model_id }
    }

    /// Build a LiquidTier backed by a full recurrent language model.
    /// Queries on Text input will run the LM and return generated text.
    pub fn from_lm(model_id: u32, lm: LiquidLanguageModel) -> Self {
        Self { model: None, lm: Some(lm), max_new_tokens: 16, model_id }
    }

    pub fn with_max_new_tokens(mut self, n: usize) -> Self {
        self.max_new_tokens = n;
        self
    }

    /// Joule estimate for one forward pass / token-step worth of work.
    pub fn forward_joules(&self) -> f64 {
        if let Some(lm) = &self.lm {
            lm.step_joules() * (self.max_new_tokens as f64).max(1.0)
        } else if let Some(m) = &self.model {
            m.step_joules()
        } else {
            10e-9
        }
    }
}

impl Tier for LiquidTier {
    fn id(&self) -> TierId {
        TierId::L3(L3ModelId(self.model_id))
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        match &q.input {
            QueryInput::Text(_) => Some(TierEstimate {
                joules: self.forward_joules(),
                latency: Duration::from_nanos(500),
                // Placeholder until R29.1.1 calibrates from a held-out eval.
                confidence_floor: 0.5,
            }),
            _ => None,
        }
    }

    fn try_answer(&mut self, q: &Query, _budget: f64) -> Result<Answer, AnswerError> {
        // R29.1 hot path: full LM is loaded.
        if let Some(lm) = self.lm.as_mut() {
            let text = match &q.input {
                QueryInput::Text(s) => s,
                _ => return Ok(refused(self.id(), 0.0, RefusalReason::Inapplicable)),
            };
            let prompt = LiquidLanguageModel::encode_bytes(text);
            let all = match lm.generate_greedy(&prompt, self.max_new_tokens) {
                Ok(v) => v,
                Err(e) => {
                    return Ok(refused(
                        self.id(),
                        0.0,
                        RefusalReason::TierSpecific(format!("liquid generation: {}", e)),
                    ));
                }
            };
            let continuation = &all[prompt.len()..];
            let out_text = LiquidLanguageModel::decode_bytes(continuation);
            let cost = self.forward_joules();
            return Ok(Answer {
                output: AnswerOutput::Text(out_text),
                tier_used: self.id(),
                joules_spent: cost,
                confidence: 0.5,
                trace: hit_trace(self.id(), cost),
                verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
            });
        }

        // R29.0 path: only the bare CfC stack (no LM head). Refuse.
        let cost = self.forward_joules();
        let reason = match &self.model {
            None => RefusalReason::TierSpecific(
                "LiquidTier kernel ready; load weights via from_lm to enable inference".into(),
            ),
            Some(m) => RefusalReason::TierSpecific(format!(
                "LiquidModel ({} layer(s)) loaded as bare CfC stack; use from_lm for inference",
                m.num_layers(),
            )),
        };
        let _ = q;
        Ok(refused(self.id(), cost, reason))
    }

    fn coord(&self) -> Option<jouleclaw_cascade::coord::Coord> {
        use jouleclaw_cascade::coord::{
            Coord, Encoding, Entity, Interface, NamedPrimitive, PrimitiveSet, Thermo,
            Verify, Zone,
        };
        Some(
            Coord::new(
                Zone::Z2_3,
                Entity::Reactive,
                Thermo::L1_Measure,
                Interface::Tokens,
                Verify::Statistical,
                Encoding::Facts,
            )
            .with_primitives(PrimitiveSet::of(&[
                NamedPrimitive::StateSpace,
                NamedPrimitive::Sample,
            ])),
        )
    }
}

fn hit_trace(tier: TierId, joules: f64) -> ExecutionTrace {
    let mut t = ExecutionTrace::default();
    t.attempts.push(TraceEntry {
        tier,
        outcome: TraceOutcome::Hit,
        joules,
    });
    t
}

fn refused(tier: TierId, joules: f64, reason: RefusalReason) -> Answer {
    let mut trace = ExecutionTrace::default();
    trace.attempts.push(TraceEntry {
        tier,
        outcome: TraceOutcome::Refused(reason.clone()),
        joules,
    });
    Answer {
        output: AnswerOutput::Refused(reason),
        tier_used: tier,
        joules_spent: joules,
        confidence: 0.0,
        trace,
        verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cell::CfcCell;

    fn text_query(s: &str) -> Query {
        Query {
            input: QueryInput::Text(s.to_string()),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    #[test]
    fn empty_tier_declares_coord_and_floor_cost() {
        let tier = LiquidTier::empty(7);
        let q = text_query("anything");
        let est = tier.estimate_cost(&q).expect("estimate should be Some");
        assert!(est.joules > 0.0);
        assert!(est.joules < 1e-6);
        assert_eq!(tier.id(), TierId::L3(L3ModelId(7)));
        let coord = tier.coord().expect("coord should be Some");
        assert!(matches!(coord.zone, jouleclaw_cascade::coord::Zone::Z2_3));
        assert!(matches!(coord.verify, jouleclaw_cascade::coord::Verify::Statistical));
        assert!(
            coord.primitives.named.iter()
                .any(|p| matches!(p, jouleclaw_cascade::coord::NamedPrimitive::StateSpace)),
            "StateSpace should be in P axis"
        );
    }

    #[test]
    fn empty_tier_refuses_with_structured_reason() {
        let mut tier = LiquidTier::empty(0);
        let ans = tier.try_answer(&text_query("hello"), 1.0).unwrap();
        match ans.output {
            AnswerOutput::Refused(RefusalReason::TierSpecific(msg)) => {
                assert!(
                    msg.contains("from_lm"),
                    "refusal should point users to from_lm: {}",
                    msg
                );
            }
            other => panic!("expected structured refusal, got {:?}", other),
        }
    }

    #[test]
    fn lm_loaded_tier_hits_with_text_output() {
        use crate::lm::{synthetic_lm, LmConfig};
        let lm = synthetic_lm(LmConfig::tiny_byte(), 0xCAFE).unwrap();
        let mut tier = LiquidTier::from_lm(99, lm).with_max_new_tokens(4);
        let ans = tier.try_answer(&text_query("hi"), 1.0).unwrap();
        match ans.output {
            AnswerOutput::Text(s) => {
                // Random weights → arbitrary continuation, but it must be Text not Refused.
                let _ = s;
            }
            other => panic!("expected Text, got {:?}", other),
        }
        assert_eq!(ans.tier_used, TierId::L3(L3ModelId(99)));
        assert!(ans.joules_spent > 0.0);
    }

    #[test]
    fn lm_path_is_deterministic_per_seed() {
        use crate::lm::{synthetic_lm, LmConfig};
        let lm1 = synthetic_lm(LmConfig::tiny_byte(), 777).unwrap();
        let lm2 = synthetic_lm(LmConfig::tiny_byte(), 777).unwrap();
        let mut t1 = LiquidTier::from_lm(0, lm1).with_max_new_tokens(6);
        let mut t2 = LiquidTier::from_lm(0, lm2).with_max_new_tokens(6);
        let a = t1.try_answer(&text_query("ping"), 1.0).unwrap();
        let b = t2.try_answer(&text_query("ping"), 1.0).unwrap();
        match (a.output, b.output) {
            (AnswerOutput::Text(sa), AnswerOutput::Text(sb)) => {
                assert_eq!(sa, sb, "same seed + same prompt must yield same output");
            }
            _ => panic!("both should be Text"),
        }
    }

    #[test]
    fn loaded_tier_cost_scales_with_layer_count() {
        let one = LiquidModel::new(vec![CfcCell::zeros(16, 16).unwrap()]).unwrap();
        let three = LiquidModel::new(vec![
            CfcCell::zeros(16, 16).unwrap(),
            CfcCell::zeros(16, 16).unwrap(),
            CfcCell::zeros(16, 16).unwrap(),
        ]).unwrap();

        let small = LiquidTier::from_model(1, one);
        let large = LiquidTier::from_model(1, three);

        assert!(large.forward_joules() > small.forward_joules() * 2.0,
            "3-layer should be at least 2× a 1-layer in joule estimate");
        assert!(large.forward_joules() < 1e-3);
    }

    #[test]
    fn cascade_can_register_liquid_tier() {
        let mut cascade = Cascade::new();
        cascade.register(Box::new(LiquidTier::empty(0)));
        let mut rt = Runtime::new_without_l0(cascade);
        // Empty cascade with only a refusing tier → either NoTierSatisfied
        // error or a refusal Answer. We only care that the tier slots in
        // without panicking and that the dispatch path runs.
        let _ = rt.answer(text_query("forecast next value"));
    }
}
