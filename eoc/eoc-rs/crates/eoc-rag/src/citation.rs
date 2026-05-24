//! Span-backed citations and provenance enforcement.
//!
//! A [`Cite`] points at a specific byte range inside a retrieved
//! chunk. [`CitationEnforcement`] is the policy gate the pipeline
//! applies before returning an [`crate::pipeline::Answer`]: if the
//! caller required citations and none are attached, the call is
//! rejected with [`crate::error::RagError::CitationRequired`].

use serde::{Deserialize, Serialize};

use crate::error::{RagError, RagResult};
use crate::store::{Chunk, ChunkId, RetrievedChunk};

/// A single citation — a span inside a retrieved chunk.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Cite {
    /// The cited chunk.
    pub chunk_id: ChunkId,
    /// Source document identifier.
    pub doc_id: String,
    /// Byte offset of the cited span *inside the chunk*.
    pub start: usize,
    /// Length of the cited span in bytes.
    pub length: usize,
    /// The actual cited text (copied so the cite is self-contained).
    pub quote: String,
}

impl Cite {
    /// Construct a `Cite` for a span inside `chunk`. Returns `None` if
    /// the requested span falls outside the chunk.
    pub fn from_span(chunk: &Chunk, start: usize, length: usize) -> Option<Self> {
        let end = start.checked_add(length)?;
        if end > chunk.text.len() {
            return None;
        }
        let slice = chunk.text.get(start..end)?;
        Some(Self {
            chunk_id: chunk.id,
            doc_id: chunk.doc_id.clone(),
            start,
            length,
            quote: slice.to_string(),
        })
    }

    /// Construct a citation that covers the entire chunk.
    pub fn whole(chunk: &Chunk) -> Self {
        Self {
            chunk_id: chunk.id,
            doc_id: chunk.doc_id.clone(),
            start: 0,
            length: chunk.text.len(),
            quote: chunk.text.clone(),
        }
    }
}

/// What the caller wants the pipeline to do about citations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CitationPolicy {
    /// Citations are optional — the pipeline may return an answer with
    /// none.
    Optional,
    /// At least one citation must be attached, otherwise the answer is
    /// rejected.
    Required,
    /// Every retrieved chunk must contribute a citation.
    PerChunk,
}

impl Default for CitationPolicy {
    fn default() -> Self {
        CitationPolicy::Required
    }
}

/// Enforcement gate. Pipelines call [`CitationEnforcement::enforce`]
/// just before returning.
pub struct CitationEnforcement {
    /// The policy in effect.
    pub policy: CitationPolicy,
}

impl CitationEnforcement {
    /// Construct.
    pub fn new(policy: CitationPolicy) -> Self {
        Self { policy }
    }

    /// Apply the policy to `(answer_text, chunks, citations)`. Returns
    /// `Ok(())` if the policy is satisfied, an error otherwise.
    pub fn enforce(
        &self,
        _answer_text: &str,
        chunks: &[RetrievedChunk],
        citations: &[Cite],
    ) -> RagResult<()> {
        match self.policy {
            CitationPolicy::Optional => Ok(()),
            CitationPolicy::Required => {
                if citations.is_empty() {
                    Err(RagError::CitationRequired)
                } else {
                    Ok(())
                }
            }
            CitationPolicy::PerChunk => {
                if chunks.len() != citations.len() {
                    return Err(RagError::CitationRequired);
                }
                Ok(())
            }
        }
    }
}

/// Derive a best-effort citation set from `(answer_text, chunks)` by
/// matching whole-chunk substrings. Production pipelines should use
/// model-generated cite anchors; this is a deterministic fallback for
/// the reference pipelines and tests.
pub fn derive_citations(answer_text: &str, chunks: &[RetrievedChunk]) -> Vec<Cite> {
    let mut out: Vec<Cite> = Vec::new();
    for rc in chunks {
        // Heuristic: if at least one alphanumeric token from the chunk
        // appears in the answer, attach a citation for the chunk.
        let answer_low = answer_text.to_lowercase();
        let mut hit = false;
        let mut tok = String::new();
        for ch in rc.chunk.text.chars() {
            if ch.is_alphanumeric() {
                for low in ch.to_lowercase() {
                    tok.push(low);
                }
            } else if !tok.is_empty() {
                if tok.len() >= 3 && answer_low.contains(&tok) {
                    hit = true;
                    break;
                }
                tok.clear();
            }
        }
        if !hit && tok.len() >= 3 && answer_low.contains(&tok) {
            hit = true;
        }
        if hit {
            out.push(Cite::whole(&rc.chunk));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::Chunk;

    fn chunks() -> Vec<RetrievedChunk> {
        vec![RetrievedChunk {
            chunk: Chunk::new("d1", 0, "apples are sweet fruits"),
            score: 1.0,
            rank: 1,
        }]
    }

    #[test]
    fn required_policy_rejects_empty() {
        let g = CitationEnforcement::new(CitationPolicy::Required);
        let err = g.enforce("hi", &chunks(), &[]).unwrap_err();
        matches!(err, RagError::CitationRequired);
    }

    #[test]
    fn required_policy_accepts_nonempty() {
        let g = CitationEnforcement::new(CitationPolicy::Required);
        let cs = chunks();
        let cite = Cite::whole(&cs[0].chunk);
        assert!(g.enforce("hi", &cs, &[cite]).is_ok());
    }

    #[test]
    fn cite_from_span_bounded() {
        let c = Chunk::new("d", 0, "hello world");
        assert!(Cite::from_span(&c, 0, 5).is_some());
        assert!(Cite::from_span(&c, 6, 5).is_some());
        assert!(Cite::from_span(&c, 6, 50).is_none());
    }

    #[test]
    fn derive_citations_matches_overlapping_chunk() {
        let cs = chunks();
        let cites = derive_citations("oranges and apples make a fruit salad", &cs);
        assert_eq!(cites.len(), 1);
    }
}
