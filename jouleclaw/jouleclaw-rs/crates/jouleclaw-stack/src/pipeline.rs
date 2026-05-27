//! RAG pipeline orchestrator: `retrieve → enrich → rerank → read`.
//!
//! The pipeline-stage tiers each speak a different structured-JSON
//! envelope. This module is the glue that drives them in sequence,
//! translating each stage's output into the next stage's input, charging
//! energy as it goes, and stopping early if the joule budget is spent.
//! It ends by feeding the top reranked passages to the L1.5 SSM reader,
//! whose extractive answer is the pipeline's result.
//!
//! Envelope bridging performed here:
//! - **retrieve** — L1 [`crate::JouleClawStack::local_index`] emits
//!   `{hits:[{doc_id,text,score}]}`; L2 [`crate::JouleClawStack::federation`]
//!   emits `{hits:[{title,url,snippet,score,sources}]}`. Both fold into a
//!   common candidate list.
//! - **enrich** — L1.25 [`crate::JouleClawStack::graph_rag`] emits
//!   `{summary,…}`; a non-empty summary is injected as an extra candidate.
//! - **rerank** — L2.5 [`crate::JouleClawStack::rerank`] consumes
//!   `{query,docs:[{id,text}],top_k}` and emits `{reranked:[{doc_id,score}]}`,
//!   reordering the candidates.
//! - **read** — L1.5 [`crate::JouleClawStack::ssm_reader`] consumes
//!   `{question,passages:[{text,source}]}` and emits the extracted answer.
//!
//! Energy is charged from each tier's self-reported `joules_spent`; the
//! pipeline never exceeds the supplied budget (it returns early with
//! `budget_exhausted = true`).

use jouleclaw_cascade::tier::Tier;
use jouleclaw_cascade::types::{
    AnswerOutput, ContextRef, JouleBudget, QualityFloor, Query, QueryInput,
};
use serde_json::{json, Value};

use crate::JouleClawStack;

/// Knobs for a pipeline run.
#[derive(Debug, Clone)]
pub struct RagConfig {
    /// Candidates kept after reranking.
    pub top_k: usize,
    /// Passages actually handed to the reader (≤ `top_k`).
    pub max_passages: usize,
    /// Retrieve from the L1 local index.
    pub use_local_index: bool,
    /// Retrieve from the L2 federation.
    pub use_federation: bool,
    /// Run the L1.25 GraphRAG enrichment stage.
    pub use_graph_enrich: bool,
}

impl Default for RagConfig {
    fn default() -> Self {
        Self {
            top_k: 8,
            max_passages: 4,
            use_local_index: true,
            use_federation: true,
            use_graph_enrich: true,
        }
    }
}

/// One stage's contribution to the run, for observability.
#[derive(Debug, Clone)]
pub struct StageTrace {
    /// Stage label, e.g. `"retrieve.local_index"`, `"rerank"`, `"read"`.
    pub stage: &'static str,
    /// Joules charged by this stage.
    pub joules: f64,
    /// Human-readable note (hit counts, fallbacks, refusals).
    pub note: String,
}

/// The result of a pipeline run.
#[derive(Debug, Clone, Default)]
pub struct RagOutcome {
    /// The extracted answer, or `None` if the reader refused / no
    /// candidates were retrieved.
    pub answer: Option<String>,
    /// Reader confidence in `[0, 1]` (0.0 when there is no answer).
    pub confidence: f32,
    /// Total joules charged across all stages.
    pub joules_spent: f64,
    /// Candidate passages gathered by retrieval + enrichment.
    pub candidates: usize,
    /// Passages actually read.
    pub passages_read: usize,
    /// Per-stage trace, in execution order.
    pub stages: Vec<StageTrace>,
    /// True if the run stopped early because the budget was spent.
    pub budget_exhausted: bool,
}

/// Internal candidate passage.
struct Candidate {
    text: String,
    source: Option<String>,
}

fn text_query(s: &str) -> Query {
    Query {
        input: QueryInput::Text(s.to_string()),
        budget: JouleBudget::expensive(),
        quality: QualityFloor::any(),
        context: ContextRef::fresh(),
        deadline: None,
    }
}

fn structured_query(v: &Value) -> Query {
    Query {
        input: QueryInput::Structured(serde_json::to_vec(v).unwrap_or_default()),
        budget: JouleBudget::expensive(),
        quality: QualityFloor::any(),
        context: ContextRef::fresh(),
        deadline: None,
    }
}

/// Dispatch a tier with a query, returning its structured-JSON output
/// (if any) and the joules it charged. Refusals and non-structured
/// outputs yield `None` for the value but still report joules.
fn dispatch_structured(tier: &mut dyn Tier, q: &Query, remaining: f64) -> (Option<Value>, f64) {
    match tier.try_answer(q, remaining) {
        Ok(answer) => {
            let joules = answer.joules_spent;
            let value = match answer.output {
                AnswerOutput::Structured(bytes) => serde_json::from_slice(&bytes).ok(),
                _ => None,
            };
            (value, joules)
        }
        Err(_) => (None, 0.0),
    }
}

impl JouleClawStack {
    /// Run the default RAG pipeline for `question` within `budget_j`
    /// joules. Convenience for [`rag_with`](Self::rag_with) with
    /// [`RagConfig::default`].
    pub fn rag(&mut self, question: &str, budget_j: f64) -> RagOutcome {
        self.rag_with(question, budget_j, &RagConfig::default())
    }

    /// Run the RAG pipeline: `retrieve → enrich → rerank → read`.
    ///
    /// Drives the stage tiers in order, bridging their envelopes,
    /// charging energy, and stopping early if `budget_j` is exhausted.
    pub fn rag_with(&mut self, question: &str, budget_j: f64, cfg: &RagConfig) -> RagOutcome {
        let mut out = RagOutcome::default();
        let mut cands: Vec<Candidate> = Vec::new();
        let qtext = text_query(question);

        // ── 1. RETRIEVE ─────────────────────────────────────────────
        if cfg.use_local_index {
            let (v, j) = dispatch_structured(
                &mut self.local_index,
                &qtext,
                budget_j - out.joules_spent,
            );
            out.joules_spent += j;
            let n = collect_hits(&v, "text", "doc_id", &mut cands);
            out.stages.push(StageTrace {
                stage: "retrieve.local_index",
                joules: j,
                note: format!("{n} hits"),
            });
        }
        if cfg.use_federation {
            let (v, j) = dispatch_structured(
                &mut self.federation,
                &qtext,
                budget_j - out.joules_spent,
            );
            out.joules_spent += j;
            let n = collect_hits(&v, "snippet", "url", &mut cands);
            out.stages.push(StageTrace {
                stage: "retrieve.federation",
                joules: j,
                note: format!("{n} hits"),
            });
        }

        if out.joules_spent >= budget_j {
            out.budget_exhausted = true;
            out.candidates = cands.len();
            return out;
        }
        if cands.is_empty() {
            out.stages.push(StageTrace {
                stage: "read",
                joules: 0.0,
                note: "no candidates retrieved".into(),
            });
            return out;
        }

        // ── 2. ENRICH ───────────────────────────────────────────────
        if cfg.use_graph_enrich {
            let (v, j) =
                dispatch_structured(&mut self.graph_rag, &qtext, budget_j - out.joules_spent);
            out.joules_spent += j;
            let mut note = "no enrichment".to_string();
            if let Some(summary) = v
                .as_ref()
                .and_then(|v| v.get("summary"))
                .and_then(|s| s.as_str())
            {
                if !summary.trim().is_empty() {
                    cands.push(Candidate {
                        text: summary.to_string(),
                        source: Some("graph".into()),
                    });
                    note = "graph summary injected".into();
                }
            }
            out.stages.push(StageTrace {
                stage: "enrich.graph_rag",
                joules: j,
                note,
            });
        }
        out.candidates = cands.len();

        // ── 3. RERANK ───────────────────────────────────────────────
        let docs: Vec<Value> = cands
            .iter()
            .enumerate()
            .map(|(i, c)| json!({ "id": i.to_string(), "text": c.text }))
            .collect();
        let rerank_q = structured_query(&json!({
            "query": question,
            "docs": docs,
            "top_k": cfg.top_k,
        }));
        let (v, j) =
            dispatch_structured(&mut self.rerank, &rerank_q, budget_j - out.joules_spent);
        out.joules_spent += j;
        let mut order: Vec<usize> = Vec::new();
        if let Some(reranked) = v
            .as_ref()
            .and_then(|v| v.get("reranked"))
            .and_then(|r| r.as_array())
        {
            for r in reranked {
                if let Some(idx) = r
                    .get("doc_id")
                    .and_then(|d| d.as_str())
                    .and_then(|s| s.parse::<usize>().ok())
                {
                    if idx < cands.len() {
                        order.push(idx);
                    }
                }
            }
        }
        if order.is_empty() {
            // Rerank refused or returned nothing — keep retrieval order.
            order = (0..cands.len()).collect();
            out.stages.push(StageTrace {
                stage: "rerank",
                joules: j,
                note: "fallback to retrieval order".into(),
            });
        } else {
            out.stages.push(StageTrace {
                stage: "rerank",
                joules: j,
                note: format!("{} ranked", order.len()),
            });
        }

        if out.joules_spent >= budget_j {
            out.budget_exhausted = true;
            return out;
        }

        // ── 4. READ ─────────────────────────────────────────────────
        let passages: Vec<Value> = order
            .iter()
            .take(cfg.max_passages)
            .map(|&i| json!({ "text": cands[i].text, "source": cands[i].source }))
            .collect();
        out.passages_read = passages.len();
        let read_q = structured_query(&json!({
            "question": question,
            "passages": passages,
        }));
        match self.ssm_reader.try_answer(&read_q, budget_j - out.joules_spent) {
            Ok(answer) => {
                out.joules_spent += answer.joules_spent;
                let note = match answer.output {
                    AnswerOutput::Text(t) => {
                        out.answer = Some(t);
                        out.confidence = answer.confidence;
                        "answer extracted"
                    }
                    AnswerOutput::Structured(_) => "structured (unexpected)",
                    AnswerOutput::Refused(_) => "reader refused (low confidence)",
                };
                out.stages.push(StageTrace {
                    stage: "read",
                    joules: answer.joules_spent,
                    note: note.into(),
                });
            }
            Err(_) => {
                out.stages.push(StageTrace {
                    stage: "read",
                    joules: 0.0,
                    note: "reader backend error".into(),
                });
            }
        }

        out
    }
}

/// Pull hits out of a stage's `{hits:[…]}` output into the candidate
/// list, reading `text_key` for the passage body and `src_key` for the
/// source label. Returns how many were appended.
fn collect_hits(v: &Option<Value>, text_key: &str, src_key: &str, out: &mut Vec<Candidate>) -> usize {
    let Some(hits) = v.as_ref().and_then(|v| v.get("hits")).and_then(|h| h.as_array()) else {
        return 0;
    };
    let mut n = 0;
    for h in hits {
        if let Some(text) = h.get(text_key).and_then(|t| t.as_str()) {
            if text.trim().is_empty() {
                continue;
            }
            let source = h.get(src_key).and_then(|s| s.as_str()).map(String::from);
            out.push(Candidate {
                text: text.to_string(),
                source,
            });
            n += 1;
        }
    }
    n
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_local_index::{Document, InMemoryIndex, LocalIndexTier};

    fn seeded_stack() -> JouleClawStack {
        let mut idx = InMemoryIndex::new();
        idx.insert(Document::new("d1", "The capital of France is Paris."));
        idx.insert(Document::new("d2", "The capital of Germany is Berlin."));
        idx.insert(Document::new("d3", "The capital of Italy is Rome."));
        let mut stack = JouleClawStack::with_defaults();
        stack.local_index = LocalIndexTier::new(idx);
        stack
    }

    #[test]
    fn extracts_answer_from_seeded_index() {
        let mut stack = seeded_stack();
        // Federation's mock hits would add noise; isolate the local index.
        let cfg = RagConfig {
            use_federation: false,
            use_graph_enrich: false,
            ..RagConfig::default()
        };
        let out = stack.rag_with("What is the capital of France?", 100.0, &cfg);
        let answer = out.answer.unwrap_or_default();
        assert!(answer.contains("Paris"), "stages={:?} answer={answer:?}", out.stages);
        assert!(out.joules_spent > 0.0);
        assert!(out.passages_read >= 1);
        // Trace records every stage that ran.
        assert!(out.stages.iter().any(|s| s.stage == "retrieve.local_index"));
        assert!(out.stages.iter().any(|s| s.stage == "rerank"));
        assert!(out.stages.iter().any(|s| s.stage == "read"));
    }

    #[test]
    fn no_candidates_yields_no_answer() {
        // Empty local index + no federation → nothing retrieved.
        let mut stack = JouleClawStack::with_defaults();
        let cfg = RagConfig {
            use_federation: false,
            use_graph_enrich: false,
            ..RagConfig::default()
        };
        let out = stack.rag_with("anything", 100.0, &cfg);
        assert!(out.answer.is_none());
        assert_eq!(out.candidates, 0);
    }

    #[test]
    fn zero_budget_exhausts_immediately() {
        let mut stack = seeded_stack();
        let out = stack.rag("capital of France", 0.0);
        // First retrieval stage already pushes spend to/over the ceiling.
        assert!(out.budget_exhausted || out.answer.is_none());
    }

    #[test]
    fn default_pipeline_runs_with_federation() {
        // With the default MockProvider federation, the pipeline should
        // retrieve mock candidates, rerank, and attempt a read without
        // panicking — even though the mock snippets carry no real answer.
        let mut stack = JouleClawStack::with_defaults();
        let out = stack.rag("hello world", 100.0);
        assert!(out.joules_spent > 0.0);
        assert!(out.candidates > 0, "federation should yield mock candidates");
        assert!(out.stages.iter().any(|s| s.stage == "retrieve.federation"));
    }
}
