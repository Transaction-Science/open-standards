//! HyDE — Hypothetical Document Embeddings (Gao et al. 2022,
//! arXiv:2212.10496).
//!
//! Instead of embedding the (short, often under-specified) user query,
//! HyDE first generates a hypothetical answer with an LLM, then
//! embeds *that* and retrieves against it. The hypothetical document
//! is in the same semantic neighbourhood as the real evidence, which
//! makes dense retrieval more robust to short queries.
//!
//! The reference [`HeuristicHydeGenerator`] is a deterministic
//! fallback used in tests; vendor backends plug in via the
//! [`HydeGenerator`] trait.

use std::sync::Arc;

use async_trait::async_trait;
use eoc_core::JouleCost;

use crate::citation::{CitationEnforcement, CitationPolicy, derive_citations};
use crate::error::{RagError, RagResult};
use crate::pipeline::{Answer, Pipeline, RagRequest, Stage, Trace, TraceEvent};
use crate::store::DocumentStore;

/// Generates a hypothetical answer to a query.
#[async_trait]
pub trait HydeGenerator: Send + Sync {
    /// Produce one (or more) hypothetical documents for `query`.
    /// Implementations should keep these short — a single paragraph.
    async fn hypothesize(&self, query: &str) -> RagResult<Vec<String>>;
}

/// Deterministic fallback generator. Concatenates the query with
/// hard-coded scaffolding text that mimics a typical encyclopedia
/// answer.
pub struct HeuristicHydeGenerator;

#[async_trait]
impl HydeGenerator for HeuristicHydeGenerator {
    async fn hypothesize(&self, query: &str) -> RagResult<Vec<String>> {
        // Expand the query with anchor phrases that often appear in
        // matching evidence. This is enough to give the dense
        // retriever extra surface area in the deterministic
        // in-memory store, mirroring the empirical lift HyDE shows
        // on short queries.
        let stripped = strip_interrogative(query);
        Ok(vec![
            format!(
                "{stripped} refers to a well-known concept. In general, {stripped} can be described in detail as follows."
            ),
        ])
    }
}

fn strip_interrogative(query: &str) -> String {
    let trimmed = query.trim().trim_end_matches('?').trim();
    let lower = trimmed.to_lowercase();
    let prefixes = [
        "what is the ",
        "what is ",
        "how does ",
        "how do ",
        "why is ",
        "why does ",
    ];
    for p in prefixes {
        if lower.starts_with(p) {
            let start = p.len();
            return trimmed[start..].trim().to_string();
        }
    }
    trimmed.to_string()
}

/// HyDE pipeline. Replaces the query at the retrieval step with the
/// concatenation of the hypothetical document(s) plus the original
/// query (so we don't lose exact-match recall).
pub struct HydePipeline {
    /// Store to retrieve against.
    pub store: Arc<dyn DocumentStore>,
    /// The hypothetical-document generator.
    pub generator: Arc<dyn HydeGenerator>,
    /// Citation policy.
    pub citation_policy: CitationPolicy,
    /// Per-stage joule estimates.
    pub hyde_microjoules: u64,
    /// Retrieval cost.
    pub retrieve_microjoules: u64,
    /// Generation cost.
    pub generate_microjoules: u64,
}

impl HydePipeline {
    /// Construct with the deterministic [`HeuristicHydeGenerator`].
    pub fn new(store: Arc<dyn DocumentStore>) -> Self {
        Self {
            store,
            generator: Arc::new(HeuristicHydeGenerator),
            citation_policy: CitationPolicy::Optional,
            hyde_microjoules: 30_000,
            retrieve_microjoules: 5_000,
            generate_microjoules: 50_000,
        }
    }

    /// Construct with a caller-supplied generator.
    pub fn with_generator(store: Arc<dyn DocumentStore>, generator: Arc<dyn HydeGenerator>) -> Self {
        Self {
            store,
            generator,
            citation_policy: CitationPolicy::Optional,
            hyde_microjoules: 30_000,
            retrieve_microjoules: 5_000,
            generate_microjoules: 50_000,
        }
    }

    /// Override citation policy (consumes `self`).
    pub fn with_citation_policy(mut self, policy: CitationPolicy) -> Self {
        self.citation_policy = policy;
        self
    }
}

#[async_trait]
impl Pipeline for HydePipeline {
    async fn answer(&self, req: &RagRequest) -> RagResult<Answer> {
        if req.top_k == 0 {
            return Err(RagError::Config("top_k must be >= 1".into()));
        }
        let mut trace = Trace::new();

        let hypos = self.generator.hypothesize(&req.query).await?;
        trace.record(TraceEvent::new(
            Stage::HypotheticalDocument,
            JouleCost::estimated(self.hyde_microjoules),
            format!("{} hypothetical(s)", hypos.len()),
        ));

        // Concatenate hypotheticals with the original query so we
        // keep exact-match recall.
        let mut composite = req.query.clone();
        for h in &hypos {
            composite.push(' ');
            composite.push_str(h);
        }

        let chunks = self.store.retrieve(&composite, req.top_k).await?;
        trace.record(TraceEvent::new(
            Stage::Retrieve,
            JouleCost::estimated(self.retrieve_microjoules),
            format!("retrieved {} chunks via HyDE", chunks.len()),
        ));
        if chunks.is_empty() {
            return Err(RagError::NoChunks);
        }

        let answer_text = chunks[0].chunk.text.clone();
        trace.record(TraceEvent::new(
            Stage::Generate,
            JouleCost::estimated(self.generate_microjoules),
            "hyde stuffed generation",
        ));

        let citations = derive_citations(&answer_text, &chunks);
        CitationEnforcement::new(self.citation_policy).enforce(&answer_text, &chunks, &citations)?;

        Ok(Answer::new(answer_text, chunks, trace).with_citations(citations))
    }

    fn name(&self) -> &str {
        "hyde"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::{Chunk, InMemoryStore};

    #[tokio::test]
    async fn hyde_returns_chunks_for_short_query() {
        let store: Arc<dyn DocumentStore> = Arc::new(InMemoryStore::from_chunks(
            "test",
            vec![
                Chunk::new(
                    "d1",
                    0,
                    "RRF refers to a well-known consensus algorithm described in detail.",
                ),
                Chunk::new("d2", 0, "Random text about turtles."),
            ],
        ));
        let p = HydePipeline::new(store);
        let req = RagRequest::new("RRF", 2);
        let ans = p.answer(&req).await.expect("ok");
        assert!(!ans.chunks.is_empty());
    }
}
