//! Query rewriting strategies.
//!
//! Three flavours:
//!
//! * [`RewriteKind::MultiQuery`] — emit `n` paraphrases of the query.
//!   The reference implementation deterministically substitutes
//!   synonyms; vendor-API integrations override with an LLM call.
//! * [`RewriteKind::StepBack`] — generate a more abstract query
//!   ("step-back prompting", Zheng et al. 2023). The deterministic
//!   fallback drops question-specific qualifiers and keeps the head
//!   noun phrase.
//! * [`RewriteKind::Decompose`] — break a multi-hop question into
//!   atomic sub-queries ("least-to-most prompting", Zhou et al. 2022,
//!   adapted for retrieval). The deterministic fallback splits on
//!   conjunctions ("and", ";", ",").

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::error::RagResult;

/// Which rewrite strategy to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RewriteKind {
    /// Multi-query expansion.
    MultiQuery,
    /// Step-back (more abstract) rewrite.
    StepBack,
    /// Decomposition into sub-queries.
    Decompose,
}

/// Abstract rewriter trait. Vendor-API backends (OpenAI / Anthropic /
/// Cohere / local) implement this. The bundled [`HeuristicRewriter`]
/// is a deterministic reference fallback.
#[async_trait]
pub trait QueryRewriter: Send + Sync {
    /// Rewrite `query` according to `kind`. The output is a vector of
    /// rewrites — `MultiQuery` typically returns 3-5, `StepBack` and
    /// `Decompose` may return 1+ depending on the query.
    async fn rewrite(&self, query: &str, kind: RewriteKind) -> RagResult<Vec<String>>;
}

/// Deterministic, dependency-free rewriter used as the reference and
/// in tests.
pub struct HeuristicRewriter;

#[async_trait]
impl QueryRewriter for HeuristicRewriter {
    async fn rewrite(&self, query: &str, kind: RewriteKind) -> RagResult<Vec<String>> {
        Ok(match kind {
            RewriteKind::MultiQuery => multi_query(query),
            RewriteKind::StepBack => vec![step_back(query)],
            RewriteKind::Decompose => decompose(query),
        })
    }
}

/// Conservative synonym table used by the heuristic multi-query
/// rewriter. Each entry rewrites the source token; rewrites that
/// change zero tokens are filtered out.
const SYNONYMS: &[(&str, &str)] = &[
    ("what is", "describe"),
    ("how does", "explain how"),
    ("how do", "explain how"),
    ("why", "for what reason"),
    ("best", "optimal"),
    ("fast", "rapid"),
    ("buy", "purchase"),
    ("car", "vehicle"),
    ("doctor", "physician"),
    ("compute", "calculate"),
    ("explain", "describe"),
];

/// Multi-query expansion — emit the original plus up to two synonym
/// substitutions plus an "in detail" anchored variant.
pub fn multi_query(query: &str) -> Vec<String> {
    let mut out: Vec<String> = Vec::new();
    out.push(query.to_string());
    let lower = query.to_lowercase();
    for (from, to) in SYNONYMS {
        if lower.contains(from) {
            // Case-insensitive single substitution.
            let rewritten = case_insensitive_replace(query, from, to);
            if rewritten != query && !out.contains(&rewritten) {
                out.push(rewritten);
                if out.len() >= 4 {
                    break;
                }
            }
        }
    }
    let detail = format!("{query} (in detail)");
    if !out.contains(&detail) {
        out.push(detail);
    }
    out
}

/// Step-back rewriter — drop question-mark and leading interrogative
/// to produce a more abstract query. If no interrogative is present,
/// prepend "Concept:".
pub fn step_back(query: &str) -> String {
    let trimmed = query.trim_end_matches('?').trim();
    let lower = trimmed.to_lowercase();
    let interrogatives = [
        "what is the ",
        "what are the ",
        "what is ",
        "what are ",
        "how does ",
        "how do ",
        "why does ",
        "why do ",
        "why is ",
        "when does ",
        "when is ",
        "where is ",
        "where are ",
    ];
    for prefix in interrogatives {
        if lower.starts_with(prefix) {
            let original_len = prefix.len();
            return trimmed[original_len..].trim().to_string();
        }
    }
    format!("Concept: {trimmed}")
}

/// Decompose by punctuation + conjunctions.
pub fn decompose(query: &str) -> Vec<String> {
    let lower = query.to_lowercase();
    let separators = [";", " and ", ", and "];
    let mut parts: Vec<String> = vec![query.to_string()];
    for sep in separators {
        let mut next: Vec<String> = Vec::new();
        for p in parts {
            if lower.contains(sep) {
                for s in p.split(sep) {
                    let t = s.trim();
                    if !t.is_empty() {
                        next.push(t.to_string());
                    }
                }
            } else {
                next.push(p);
            }
        }
        parts = next;
    }
    parts
        .into_iter()
        .filter(|s| !s.trim().is_empty())
        .collect()
}

fn case_insensitive_replace(haystack: &str, needle: &str, replacement: &str) -> String {
    let haystack_low = haystack.to_lowercase();
    let needle_low = needle.to_lowercase();
    if let Some(pos) = haystack_low.find(&needle_low) {
        let mut out = String::with_capacity(haystack.len() + replacement.len());
        out.push_str(&haystack[..pos]);
        out.push_str(replacement);
        out.push_str(&haystack[pos + needle.len()..]);
        out
    } else {
        haystack.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn multi_query_expands() {
        let r = HeuristicRewriter;
        let out = r
            .rewrite("what is the best way to compute joules", RewriteKind::MultiQuery)
            .await
            .expect("ok");
        assert!(out.len() >= 2);
        assert!(out.contains(&"what is the best way to compute joules".to_string()));
    }

    #[tokio::test]
    async fn step_back_drops_interrogative() {
        let r = HeuristicRewriter;
        let out = r
            .rewrite("What is the capital of France?", RewriteKind::StepBack)
            .await
            .expect("ok");
        assert_eq!(out[0], "capital of France");
    }

    #[tokio::test]
    async fn decompose_splits_conjunctions() {
        let r = HeuristicRewriter;
        let out = r
            .rewrite(
                "what is BM25 and how does HNSW work",
                RewriteKind::Decompose,
            )
            .await
            .expect("ok");
        assert!(out.len() >= 2);
    }
}
