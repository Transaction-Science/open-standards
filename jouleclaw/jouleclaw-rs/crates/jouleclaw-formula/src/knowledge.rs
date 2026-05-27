//! Portable knowledge-store interface for the L0.25 formula-first tier.
//!
//! The donor (`verity-cascade`) hard-coded a dependency on `verity-contrast`'s
//! `KnowledgeStore`. That crate carries OpenIE IP (biological-trait lexicons,
//! the spiking-neural-net store, the contrast formula's actual coefficients),
//! none of which can live in the public JouleClaw standard. This module
//! re-states the *interface* the formula-first tier needs from a knowledge
//! store, so any downstream consumer — including OpenIE's own
//! `verity-contrast` — can adapt itself to it.
//!
//! ## Shape
//!
//! A `KnowledgeStore` is the minimum surface needed to:
//!
//! 1. Look up concepts by name (entity extraction).
//! 2. Find nearest neighbours of a concept (single-entity neighbourhood).
//! 3. Compute pairwise structural contrast between two concepts.
//!
//! Implementations decide what "structural" means — the formula tier does not
//!. It treats `similarity` and `coverage` as opaque scores in `[0, 10_000]`
//! (q14.0 fixed-point, matching the donor's convention).
//!
//! ## In-memory reference impl
//!
//! [`InMemoryKnowledgeStore`] is a small pure-Rust reference implementation
//! sufficient for tests, demos, and the L0.25 conformance vectors. It uses
//! a flat vector of [`Concept`]s, substring search on names, and a trivial
//! cosine-by-overlap similarity over [`Concept::traits`]. Production
//! consumers should plug in their own implementation.

use std::collections::HashMap;

/// Opaque fixed-point similarity score. `0` = nothing in common, `10_000` =
/// identical. Q14.0 quantization matches the donor's `Similarity` type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct Similarity(pub u16);

impl Similarity {
    /// Highest similarity: identical entities.
    pub const MAX: Similarity = Similarity(10_000);
    /// Lowest similarity: nothing in common.
    pub const MIN: Similarity = Similarity(0);

    /// Construct from a `f64` in `[0.0, 1.0]`. Out-of-range values clamp.
    pub fn from_unit(x: f64) -> Self {
        let v = (x.clamp(0.0, 1.0) * 10_000.0) as u16;
        Self(v)
    }

    /// As an `f32` in `[0.0, 1.0]`.
    pub fn as_unit(self) -> f32 {
        self.0 as f32 / 10_000.0
    }
}

/// A single concept (entity) stored in the knowledge store.
///
/// The donor's `Concept` carried a 64-dimension structural embedding. We keep
/// a generic `traits` vector so any implementation can choose its own
/// dimensionality. The formula-first tier itself never inspects individual
/// dimensions — that's the store's job.
#[derive(Debug, Clone, PartialEq)]
pub struct Concept {
    /// Stable canonical identifier (e.g. `wd:Q12345`, `urn:concept:fire`).
    pub id: String,
    /// Human-readable display name (e.g. "fire").
    pub name: String,
    /// Optional structural embedding. Empty when the store uses a different
    /// representation under the hood.
    pub traits: Vec<f32>,
}

/// A single dimension's contribution to a pairwise contrast.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ContrastDimension {
    /// Index of the dimension in the underlying formula.
    pub dimension: u16,
    /// Signed contribution to the overall similarity.
    pub contribution: f32,
    /// Coarse classification of this dimension's status.
    pub relation: ContrastRelation,
}

/// Per-dimension status emitted by a pairwise contrast.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContrastRelation {
    /// Both entities point the same way on this dimension.
    Align,
    /// Both entities point opposite ways.
    Oppose,
    /// Only one entity has a measurement here.
    Partial,
    /// Neither entity has a measurement.
    Unknown,
}

/// Structured pairwise contrast result.
#[derive(Debug, Clone, PartialEq)]
pub struct ContrastMap {
    /// Aggregate similarity in q14.0.
    pub similarity: Similarity,
    /// Knowledge coverage — how confident the store is in the pairing.
    pub coverage: Similarity,
    /// Per-dimension breakdown. May be empty for stores that do not
    /// expose per-dimension structure.
    pub dimensions: Vec<ContrastDimension>,
    /// Convenience counter: how many dimensions resolved as `Align` or
    /// `Oppose` (i.e. both sides had a measurement).
    pub known_count: u16,
    /// Dimensions where only one side had a measurement.
    pub partial_count: u16,
    /// Dimensions where neither side had a measurement.
    pub unknown_count: u16,
}

/// The portable interface the L0.25 formula-first tier needs from a
/// knowledge store.
///
/// Implementations are free to back this with any structural-contrast engine
/// — biological-trait lexicons, learned embeddings, hand-curated tables, …
/// The formula-first tier itself only depends on this trait.
pub trait KnowledgeStore: Send + Sync {
    /// Number of stored concepts. The tier short-circuits to "Skipped" when
    /// the store is empty.
    fn len(&self) -> usize;

    /// Convenience: whether the store has any concepts.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Look up concepts by name. The donor's contract is substring-friendly
    /// (case-insensitive substring matching), so implementations SHOULD do
    /// the same to keep entity extraction working. Returns at most `limit`
    /// concepts, best-match-first.
    fn search_by_name(&self, name: &str, limit: usize) -> Vec<Concept>;

    /// Find the `limit` nearest concepts to the concept with id `id`.
    /// Returns `(similarity, concept)` pairs.
    fn nearest_to(&self, id: &str, limit: usize) -> Vec<(Similarity, Concept)>;

    /// Compute the pairwise structural contrast between two concepts.
    /// Returns `None` when either id is unknown to the store.
    fn contrast(&self, a_id: &str, b_id: &str) -> Option<ContrastMap>;

    /// Sorted list of dimension names. May be empty for stores that do not
    /// expose dimension names. Used only for emitting structured contrast
    /// records.
    fn dimension_names(&self) -> Vec<String> {
        Vec::new()
    }
}

// ─── In-memory reference implementation ──────────────────────────

/// Small pure-Rust knowledge store sufficient for tests, demos, and the
/// L0.25 conformance vectors.
///
/// Storage is a flat `Vec<Concept>` plus a `HashMap<String, usize>` index
/// from canonical ids. Similarity is a trivial cosine-by-overlap over the
/// `Concept::traits` vectors; concepts whose `traits` are empty default to
/// `0.5` similarity (they have a name in common with the query but we know
/// nothing about their structure).
#[derive(Debug, Default, Clone)]
pub struct InMemoryKnowledgeStore {
    concepts: Vec<Concept>,
    by_id: HashMap<String, usize>,
    dim_names: Vec<String>,
}

impl InMemoryKnowledgeStore {
    /// Create an empty store.
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the human-readable dimension names. Optional; only used when
    /// callers want to surface dimension-level contrasts.
    pub fn with_dimension_names<I, S>(mut self, names: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.dim_names = names.into_iter().map(Into::into).collect();
        self
    }

    /// Insert (or replace) a concept by id.
    pub fn insert(&mut self, concept: Concept) -> &mut Self {
        if let Some(&idx) = self.by_id.get(&concept.id) {
            self.concepts[idx] = concept;
        } else {
            let idx = self.concepts.len();
            self.by_id.insert(concept.id.clone(), idx);
            self.concepts.push(concept);
        }
        self
    }

    /// Look up a concept by canonical id.
    pub fn get(&self, id: &str) -> Option<&Concept> {
        self.by_id.get(id).map(|&idx| &self.concepts[idx])
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        if a.is_empty() || b.is_empty() {
            return 0.5;
        }
        let n = a.len().min(b.len());
        let mut dot = 0.0_f32;
        let mut na = 0.0_f32;
        let mut nb = 0.0_f32;
        for i in 0..n {
            dot += a[i] * b[i];
            na += a[i] * a[i];
            nb += b[i] * b[i];
        }
        if na == 0.0 || nb == 0.0 {
            0.5
        } else {
            let denom = na.sqrt() * nb.sqrt();
            // Map cosine ∈ [-1, 1] into [0, 1].
            ((dot / denom) * 0.5 + 0.5).clamp(0.0, 1.0)
        }
    }
}

impl KnowledgeStore for InMemoryKnowledgeStore {
    fn len(&self) -> usize {
        self.concepts.len()
    }

    fn search_by_name(&self, name: &str, limit: usize) -> Vec<Concept> {
        let needle = name.to_lowercase();
        let mut hits: Vec<(usize, &Concept)> = Vec::new();
        for c in &self.concepts {
            let nl = c.name.to_lowercase();
            let il = c.id.to_lowercase();
            // Score by how much of either field overlaps the needle.
            // Lower score = better match (front-of-line semantics).
            let in_name = nl.contains(&needle);
            let in_id = il.contains(&needle);
            let contains_name = needle.contains(&nl);
            if in_name || in_id || contains_name {
                // Exact name match beats substring.
                let score = if nl == needle {
                    0
                } else if in_name {
                    1
                } else if in_id {
                    2
                } else {
                    3
                };
                hits.push((score, c));
            }
        }
        hits.sort_by_key(|(s, _)| *s);
        hits.into_iter()
            .take(limit)
            .map(|(_, c)| c.clone())
            .collect()
    }

    fn nearest_to(&self, id: &str, limit: usize) -> Vec<(Similarity, Concept)> {
        let anchor = match self.get(id) {
            Some(c) => c,
            None => return Vec::new(),
        };
        let mut scored: Vec<(Similarity, Concept)> = self
            .concepts
            .iter()
            .filter(|c| c.id != anchor.id)
            .map(|c| {
                let s = Self::cosine(&anchor.traits, &c.traits);
                (Similarity::from_unit(s as f64), c.clone())
            })
            .collect();
        scored.sort_by(|a, b| b.0.0.cmp(&a.0.0));
        scored.truncate(limit);
        scored
    }

    fn contrast(&self, a_id: &str, b_id: &str) -> Option<ContrastMap> {
        let a = self.get(a_id)?;
        let b = self.get(b_id)?;
        let sim_unit = Self::cosine(&a.traits, &b.traits) as f64;
        let similarity = Similarity::from_unit(sim_unit);

        let n = a.traits.len().min(b.traits.len());
        let mut dimensions: Vec<ContrastDimension> = Vec::with_capacity(n);
        let mut known = 0u16;
        let mut partial = 0u16;
        let mut unknown = 0u16;
        for i in 0..n {
            let ai = a.traits[i];
            let bi = b.traits[i];
            let relation = if ai == 0.0 && bi == 0.0 {
                unknown += 1;
                ContrastRelation::Unknown
            } else if ai == 0.0 || bi == 0.0 {
                partial += 1;
                ContrastRelation::Partial
            } else if ai.signum() == bi.signum() {
                known += 1;
                ContrastRelation::Align
            } else {
                known += 1;
                ContrastRelation::Oppose
            };
            dimensions.push(ContrastDimension {
                dimension: i as u16,
                contribution: ai * bi,
                relation,
            });
        }

        // Coverage: how many dimensions we could measure on at least one side.
        let coverage_unit = if n == 0 {
            0.5
        } else {
            (known as f64 + 0.5 * partial as f64) / n as f64
        };
        let coverage = Similarity::from_unit(coverage_unit);

        Some(ContrastMap {
            similarity,
            coverage,
            dimensions,
            known_count: known,
            partial_count: partial,
            unknown_count: unknown,
        })
    }

    fn dimension_names(&self) -> Vec<String> {
        self.dim_names.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_store() -> InMemoryKnowledgeStore {
        let mut k = InMemoryKnowledgeStore::new()
            .with_dimension_names(["heat", "wet", "alive"]);
        k.insert(Concept {
            id: "urn:fire".into(),
            name: "fire".into(),
            traits: vec![1.0, -1.0, 0.0],
        });
        k.insert(Concept {
            id: "urn:water".into(),
            name: "water".into(),
            traits: vec![-1.0, 1.0, 0.0],
        });
        k.insert(Concept {
            id: "urn:cell".into(),
            name: "cell".into(),
            traits: vec![0.1, 0.5, 1.0],
        });
        k
    }

    #[test]
    fn search_by_name_exact_then_substring() {
        let k = sample_store();
        let r = k.search_by_name("fire", 3);
        assert!(r.first().is_some_and(|c| c.id == "urn:fire"));
    }

    #[test]
    fn nearest_to_excludes_self() {
        let k = sample_store();
        let n = k.nearest_to("urn:fire", 5);
        assert!(!n.iter().any(|(_, c)| c.id == "urn:fire"));
    }

    #[test]
    fn contrast_unknown_returns_none() {
        let k = sample_store();
        assert!(k.contrast("urn:fire", "urn:missing").is_none());
    }

    #[test]
    fn contrast_align_or_oppose_per_dimension() {
        let k = sample_store();
        let m = k.contrast("urn:fire", "urn:water").expect("contrast");
        // Both have heat and wet measured (opposite signs) and alive unknown.
        assert!(m.known_count >= 2);
        assert!(m.unknown_count >= 1);
    }
}
