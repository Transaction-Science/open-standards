//! `MrlTier` — joule cascade tier shell for Matryoshka-truncated retrieval.
//!
//! R30.0 lands the dim-picking + cost model. R30.1 wires an actual corpus
//! and nearest-neighbor search. Until then `try_answer` refuses with a
//! structured reason naming R30.1, while `estimate_cost` reports an
//! honest floor that reflects the embed pass cost plus a minimum
//! retrieval-against-zero-docs floor.
//!
//! Coordinate:
//!
//!   Z = Z2          (statistical embedder)
//!   E = Reactive
//!   T = L2_Landauer (small matvec ops, close to physical floor)
//!   I = Signals     (embeddings, not raw text)
//!   V = Statistical
//!   R = Navigation  (embeddings encode position in similarity space)
//!   P = { Embed, NearestNeighbor }

use std::time::Duration;

use jouleclaw_cascade::*;

use crate::embedder::Embedder;
use crate::matryoshka::MatryoshkaEmbedder;
use crate::retrieval::Corpus;

pub struct MrlTier<E: Embedder + 'static> {
    pub corpus: Corpus<E>,
    pub model_id: u32,
    /// Quality floor used by the dim picker on each retrieve. R30.1.1
    /// will replace this with a calibrated value from a held-out eval.
    pub quality_floor: f32,
    /// Per-query retrieval budget in joules.
    pub retrieval_budget: f64,
    /// Top-k hits returned in the answer text. Default 3.
    pub top_k: usize,
}

impl<E: Embedder + 'static> MrlTier<E> {
    /// Empty corpus — declares the cell, will refuse until docs are added.
    pub fn new(model_id: u32, matryoshka: MatryoshkaEmbedder<E>) -> Self {
        Self {
            corpus: Corpus::new(matryoshka),
            model_id,
            quality_floor: 0.95,
            retrieval_budget: 1.0,
            top_k: 3,
        }
    }

    /// Build from a pre-populated corpus. `try_answer` will run the
    /// nearest-neighbor lookup and return Text.
    pub fn from_corpus(model_id: u32, corpus: Corpus<E>) -> Self {
        Self {
            corpus,
            model_id,
            quality_floor: 0.95,
            retrieval_budget: 1.0,
            top_k: 3,
        }
    }

    /// Insert a document into the corpus.
    pub fn add_doc(&mut self, text: impl Into<String>) -> Result<u32, String> {
        self.corpus.add(text).map_err(|e| e.to_string())
    }

    pub fn with_top_k(mut self, k: usize) -> Self {
        self.top_k = k;
        self
    }

    pub fn with_quality_floor(mut self, q: f32) -> Self {
        self.quality_floor = q;
        self
    }

    pub fn with_retrieval_budget(mut self, j: f64) -> Self {
        self.retrieval_budget = j;
        self
    }

    pub fn corpus_size(&self) -> usize {
        self.corpus.len()
    }

    /// Cost floor: one embed pass + nearest-neighbor over the corpus at
    /// the picked dim. Conservative — uses the smallest dim in the ladder.
    pub fn floor_joules(&self) -> f64 {
        let smallest = self.corpus.matryoshka().dims().first().copied().unwrap_or(1);
        self.corpus.retrieve_joules(smallest)
    }
}

impl<E: Embedder + 'static> Tier for MrlTier<E> {
    fn id(&self) -> TierId {
        TierId::L2(L2ModelId(self.model_id))
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        match &q.input {
            QueryInput::Text(_) => Some(TierEstimate {
                joules: self.floor_joules(),
                latency: Duration::from_micros(20),
                // Placeholder until R30.1.1 calibrates from a held-out eval.
                confidence_floor: 0.7,
            }),
            _ => None,
        }
    }

    fn try_answer(&mut self, q: &Query, _budget: f64) -> Result<Answer, AnswerError> {
        // R30.1.0 hot path: real corpus loaded and non-empty.
        if !self.corpus.is_empty() {
            let text = match &q.input {
                QueryInput::Text(s) => s,
                _ => return Ok(refused(self.id(), 0.0, RefusalReason::Inapplicable)),
            };
            match self.corpus.retrieve(text, self.top_k, self.quality_floor, self.retrieval_budget) {
                Ok(hits) => {
                    // Project the top-hit's similarity score into the
                    // Answer's confidence. Without this, MrlTier reports
                    // a hardcoded 0.7 regardless of how good the match
                    // was — a "lava biome" query and a goblin query
                    // both came back at confidence 0.7 in a cascade
                    // smoke (2026-05-24), even though the goblin hits
                    // were unrelated pattern-lang docs. Mapping
                    // confidence ← top_score makes the cascade's
                    // quality floor a real task-fit gate, not just a
                    // tier-existence gate. Negative similarities map
                    // to 0.0; the cascade skips below the floor.
                    let top_score = hits.first().map(|h| h.score).unwrap_or(0.0);
                    let actual_confidence = (top_score as f32).clamp(0.0, 1.0);
                    let cost = self.floor_joules();
                    if actual_confidence < q.quality.min_confidence {
                        return Ok(refused(
                            self.id(),
                            cost,
                            RefusalReason::TierSpecific(format!(
                                "top retrieval score {:.3} below quality floor {:.3}",
                                actual_confidence, q.quality.min_confidence
                            )),
                        ));
                    }
                    let mut out = String::new();
                    out.push_str("Top results:\n");
                    for h in &hits {
                        out.push_str(&format!(
                            "  [{:.4}@dim{}] doc#{}: {}\n",
                            h.score, h.dim, h.doc_id, h.doc_text
                        ));
                    }
                    return Ok(Answer {
                        output: AnswerOutput::Text(out),
                        tier_used: self.id(),
                        joules_spent: cost,
                        confidence: actual_confidence,
                        trace: hit_trace(self.id(), cost),
                        verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
                    });
                }
                Err(e) => {
                    return Ok(refused(
                        self.id(),
                        self.floor_joules(),
                        RefusalReason::TierSpecific(format!("retrieval failed: {}", e)),
                    ));
                }
            }
        }

        // Empty corpus — preserve R30.0 refusal posture.
        let cost = self.floor_joules();
        let reason = RefusalReason::TierSpecific(
            "MrlTier corpus is empty; populate via add_doc to enable retrieval".into(),
        );
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
                Zone::Z2,
                Entity::Reactive,
                Thermo::L2_Landauer,
                Interface::Signals,
                Verify::Statistical,
                Encoding::Navigation,
            )
            .with_primitives(PrimitiveSet::of(&[
                NamedPrimitive::Embed,
                NamedPrimitive::NearestNeighbor,
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
    use crate::embedder::IdentityEmbedder;
    use crate::matryoshka::MatryoshkaEmbedder;

    fn text_query(s: &str) -> Query {
        Query {
            input: QueryInput::Text(s.to_string()),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    fn empty_tier(model_id: u32, full_dim: usize) -> MrlTier<IdentityEmbedder> {
        let m = MatryoshkaEmbedder::with_powers_of_two(IdentityEmbedder::new(full_dim));
        MrlTier::new(model_id, m)
    }

    fn populated_tier(
        model_id: u32,
        full_dim: usize,
        docs: &[&str],
    ) -> MrlTier<IdentityEmbedder> {
        let mut t = empty_tier(model_id, full_dim);
        for d in docs {
            t.add_doc(*d).unwrap();
        }
        t
    }

    #[test]
    fn tier_declares_navigation_encoding_and_l2_id() {
        let tier = empty_tier(5, 512);
        assert_eq!(tier.id(), TierId::L2(L2ModelId(5)));
        let coord = tier.coord().expect("coord set");
        assert!(matches!(coord.encoding, jouleclaw_cascade::coord::Encoding::Navigation));
        assert!(coord.primitives.named.iter().any(|p|
            matches!(p, jouleclaw_cascade::coord::NamedPrimitive::Embed)
        ));
        assert!(coord.primitives.named.iter().any(|p|
            matches!(p, jouleclaw_cascade::coord::NamedPrimitive::NearestNeighbor)
        ));
    }

    #[test]
    fn empty_corpus_refuses_with_structured_reason() {
        let mut tier = empty_tier(0, 256);
        let ans = tier.try_answer(&text_query("hi"), 1.0).unwrap();
        match ans.output {
            AnswerOutput::Refused(RefusalReason::TierSpecific(msg)) => {
                assert!(
                    msg.contains("add_doc"),
                    "refusal should point users at add_doc: {}",
                    msg
                );
            }
            other => panic!("expected refusal, got {:?}", other),
        }
    }

    #[test]
    fn populated_corpus_hits_with_top_results() {
        let mut tier = populated_tier(
            0, 64,
            &[
                "hello world",
                "the lawful synthesizer",
                "ternary quantization",
                "matryoshka embeddings",
            ],
        )
        .with_top_k(2);
        let ans = tier.try_answer(&text_query("the lawful synthesizer"), 1.0).unwrap();
        match ans.output {
            AnswerOutput::Text(s) => {
                assert!(s.contains("Top results:"));
                // Self-match should appear in the output.
                assert!(
                    s.contains("the lawful synthesizer"),
                    "expected self-match in output: {}",
                    s
                );
            }
            other => panic!("expected Text, got {:?}", other),
        }
        assert!(ans.joules_spent > 0.0);
    }

    #[test]
    fn cascade_can_register_mrl_tier() {
        let mut cascade = Cascade::new();
        cascade.register(Box::new(populated_tier(0, 256, &["a", "b"])));
        let mut rt = Runtime::new_without_l0(cascade);
        let _ = rt.answer(text_query("find related"));
    }
}
