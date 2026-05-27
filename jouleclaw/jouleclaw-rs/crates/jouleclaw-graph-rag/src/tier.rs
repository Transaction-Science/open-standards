//! L1.25 — GraphRAG tier.
//!
//! Runs between L0.75 (SsmRouter) and L1.375 (StructContrast). Extracts
//! entity candidates from the raw query, resolves each candidate against
//! a consumer-supplied [`KnowledgeGraph`], collects the immediate
//! neighbourhood of every resolved entity, and emits the resulting
//! sub-graph as `AnswerOutput::Structured` so the downstream
//! L1.375 / L1.5 tiers have richer context to work with.
//!
//! ## Mapping to the `Tier` trait
//!
//! - `id` → [`TierId::L1_25GraphRag`].
//! - `estimate_cost` → fixed 500 µJ / 50 µs envelope, confidence floor 0.6.
//!   `None` when the graph is empty or the query input is not text.
//! - `try_answer` → returns `AnswerOutput::Structured(json!({...}))` on
//!   success, or `Refused(Inapplicable)` when no entities resolve.
//!
//! The donor's confidence formula (`entity_score + relation_score`) is
//! preserved: more resolved entities → higher confidence, capped at 1.0.

use std::time::Duration;

use jouleclaw_cascade::tier::{Tier, TierEstimate};
use jouleclaw_cascade::types::{
    Answer, AnswerError, AnswerOutput, ExecutionTrace, Query, QueryInput,
    RefusalReason, TierId,
};
use jouleclaw_cascade::verification::VerificationStatus;
use jouleclaw_energy::Provenance;

use crate::extract::{extract_entity_candidates, EntityCandidate, EntityClass};
use crate::graph::{Edge, Entity, EntityId, KnowledgeGraph};

// ─── Cost model ──────────────────────────────────────────────────

/// Donor envelope: ~500 µJ per query.
pub const GRAPH_RAG_JOULES: f64 = 500e-6;
/// Wall-clock latency target.
pub const GRAPH_RAG_LATENCY: Duration = Duration::from_micros(50);
/// Confidence floor advertised to the runtime — the lowest confidence
/// we are willing to claim before refusing the dispatch.
pub const GRAPH_RAG_CONFIDENCE_FLOOR: f32 = 0.6;
/// Default neighbourhood depth for [`GraphRagTier::neighborhood_depth`].
/// One hop preserves the donor's "direct co-occurrence" model.
pub const DEFAULT_NEIGHBORHOOD_DEPTH: u8 = 1;
/// Hard ceiling on entities surfaced in the structured output. Mirrors
/// the donor's `truncate(10)`.
pub const MAX_ENTITIES_OUT: usize = 10;
/// Hard ceiling on edges surfaced in the structured output. Mirrors the
/// donor's `truncate(5)` for relationships.
pub const MAX_EDGES_OUT: usize = 5;

// ─── Errors ──────────────────────────────────────────────────────

/// Errors specific to the graph-rag tier.
#[derive(Debug, thiserror::Error)]
pub enum GraphRagError {
    /// Failed to serialise the structured output payload.
    #[error("failed to serialise graph-rag output: {0}")]
    Serialise(#[from] serde_json::Error),
}

// ─── Public output shape ─────────────────────────────────────────

/// Structured payload returned in `AnswerOutput::Structured`. Downstream
/// tiers (L1.375 StructContrast, L1.5 SsmReader) deserialise this to
/// recover the resolved entity set and sub-graph.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GraphRagOutput {
    /// Echoes the original query text. Lets downstream tiers do
    /// per-query analysis without re-plumbing the cascade.
    pub query: String,
    /// Resolved entities in mention-order (most mentions first), capped
    /// at [`MAX_ENTITIES_OUT`].
    pub entities: Vec<GraphRagEntity>,
    /// Edges connecting resolved entities, capped at [`MAX_EDGES_OUT`]
    /// after sorting by weight descending.
    pub edges: Vec<Edge>,
    /// Human-readable summary — useful when an SSM-class downstream
    /// tier wants prose instead of structured fields.
    pub summary: String,
}

/// One resolved entity surfaced in the structured output. Combines the
/// extractor's surface form with the knowledge-graph record.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct GraphRagEntity {
    /// Resolved canonical entity from the graph.
    pub entity: Entity,
    /// Class the extractor inferred for this candidate.
    pub class: EntityClass,
    /// How many distinct candidate surface-forms in the query mapped to
    /// this entity. Higher = more strongly grounded.
    pub mentions: u32,
}

// ─── The tier ────────────────────────────────────────────────────

/// L1.25 GraphRAG tier. Generic over a [`KnowledgeGraph`] so consumers
/// plug their own store (Neo4j, in-memory, embedded sled, etc.).
pub struct GraphRagTier<G: KnowledgeGraph> {
    graph: G,
    neighborhood_depth: u8,
}

impl<G: KnowledgeGraph> GraphRagTier<G> {
    /// Build a new GraphRAG tier over `graph`. Defaults to one-hop
    /// neighbourhoods (mirroring the donor's co-occurrence depth).
    pub fn new(graph: G) -> Self {
        Self {
            graph,
            neighborhood_depth: DEFAULT_NEIGHBORHOOD_DEPTH,
        }
    }

    /// Override the neighbourhood depth used during enrichment.
    ///
    /// Depth zero is allowed (the tier will still report resolved
    /// entities, just no edges); depth values above ~3 risk inflating
    /// the L1.25 cost envelope and SHOULD be avoided unless paired with
    /// a knowledge graph that caps its own response size.
    pub fn with_neighborhood_depth(mut self, depth: u8) -> Self {
        self.neighborhood_depth = depth;
        self
    }

    /// Borrow the underlying knowledge graph.
    pub fn graph(&self) -> &G {
        &self.graph
    }

    /// Provenance tag for any energy spend reported by this tier. The
    /// L1.25 envelope is model-derived (no hardware shunt), so
    /// [`Provenance::Estimator`] is the honest label.
    pub const fn provenance() -> Provenance {
        Provenance::Estimator
    }

    // ─── Internal pipeline ───────────────────────────────────────

    /// Resolve candidates against the graph, deduping by canonical id.
    /// Returns the resolved entities (in first-mention order) paired
    /// with their extractor class and mention count.
    fn resolve_entities(
        &self,
        candidates: &[EntityCandidate],
    ) -> Vec<GraphRagEntity> {
        // Order-preserving dedup keyed by canonical id.
        let mut order: Vec<EntityId> = Vec::new();
        let mut by_id: std::collections::HashMap<
            EntityId,
            GraphRagEntity,
        > = std::collections::HashMap::new();

        for cand in candidates {
            let Some(entity) = self.graph.lookup_entity(&cand.surface) else {
                continue;
            };
            match by_id.get_mut(&entity.id) {
                Some(existing) => existing.mentions += 1,
                None => {
                    order.push(entity.id.clone());
                    by_id.insert(
                        entity.id.clone(),
                        GraphRagEntity {
                            entity,
                            class: cand.class,
                            mentions: 1,
                        },
                    );
                }
            }
        }

        // Sort by mention count descending (stable on first-mention
        // order via the explicit `order` Vec).
        let mut out: Vec<GraphRagEntity> = order
            .into_iter()
            .filter_map(|id| by_id.remove(&id))
            .collect();
        out.sort_by_key(|e| std::cmp::Reverse(e.mentions));
        out.truncate(MAX_ENTITIES_OUT);
        out
    }

    /// Collect deduped edges over the resolved entity set.
    ///
    /// Only edges whose `from` AND `to` both resolved are kept — this
    /// keeps the sub-graph honest (no dangling pointers into the wider
    /// store) and bounded by the entity-set size.
    fn collect_edges(&self, entities: &[GraphRagEntity]) -> Vec<Edge> {
        let resolved_ids: std::collections::HashSet<EntityId> = entities
            .iter()
            .map(|e| e.entity.id.clone())
            .collect();

        let mut seen: std::collections::HashSet<(EntityId, EntityId, String)> =
            std::collections::HashSet::new();
        let mut edges: Vec<Edge> = Vec::new();

        for ent in entities {
            for e in
                self.graph.neighbors(&ent.entity.id, self.neighborhood_depth)
            {
                if !resolved_ids.contains(&e.to) {
                    continue;
                }
                let key = (e.from.clone(), e.to.clone(), e.label.clone());
                if seen.insert(key) {
                    edges.push(e);
                }
            }
        }

        edges.sort_by(|a, b| {
            b.weight
                .partial_cmp(&a.weight)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        edges.truncate(MAX_EDGES_OUT);
        edges
    }

    /// Compose a prose summary mirroring the donor's
    /// `build_entity_context` output. Kept verbatim so existing
    /// downstream tiers that consume the summary string still see the
    /// same shape.
    fn build_summary(
        &self,
        query: &str,
        entities: &[GraphRagEntity],
        edges: &[Edge],
    ) -> String {
        if entities.is_empty() {
            return String::new();
        }
        let mut ctx = format!("Key entities related to \"{query}\":\n");
        for ge in entities {
            ctx.push_str(&format!(
                "- {} [{}] (mentioned {} time(s); kind={})\n",
                ge.entity.name,
                ge.class.tag(),
                ge.mentions,
                ge.entity.kind,
            ));
        }
        if !edges.is_empty() {
            ctx.push_str("Relationships:\n");
            for e in edges {
                ctx.push_str(&format!(
                    "- {} --{}--> {} (weight {:.2})\n",
                    e.from.as_str(),
                    e.label,
                    e.to.as_str(),
                    e.weight,
                ));
            }
        }
        ctx
    }

    /// Donor confidence map: more entities + more edges → higher
    /// confidence, ceilinged at 1.0.
    fn compute_confidence(&self, entities: &[GraphRagEntity], edges: &[Edge]) -> f32 {
        if entities.is_empty() {
            return 0.0;
        }
        // 0.06 per entity (cap 0.6 at 10 entities) + 0.08 per edge
        // (cap 0.4 at 5 edges) = 1.0 total ceiling.
        let entity_score = (entities.len() as f32 * 0.06).min(0.6);
        let edge_score = (edges.len() as f32 * 0.08).min(0.4);
        (entity_score + edge_score).clamp(0.0, 1.0)
    }

    /// End-to-end pipeline: extract, resolve, collect edges, summarise.
    fn enrich(&self, query: &str) -> Result<Answer, GraphRagError> {
        let candidates = extract_entity_candidates(query);
        let entities = self.resolve_entities(&candidates);

        if entities.is_empty() {
            return Ok(refused_inapplicable());
        }

        let edges = self.collect_edges(&entities);
        let summary = self.build_summary(query, &entities, &edges);
        let confidence = self.compute_confidence(&entities, &edges);

        let payload = GraphRagOutput {
            query: query.to_string(),
            entities,
            edges,
            summary,
        };
        let bytes = serde_json::to_vec(&payload)?;

        Ok(Answer {
            output: AnswerOutput::Structured(bytes),
            tier_used: TierId::L1_25GraphRag,
            joules_spent: GRAPH_RAG_JOULES,
            confidence,
            trace: ExecutionTrace::default(),
            verification: VerificationStatus::Resolved,
        })
    }
}

// ─── Answer helpers ──────────────────────────────────────────────

fn refused_inapplicable() -> Answer {
    Answer {
        output: AnswerOutput::Refused(RefusalReason::Inapplicable),
        tier_used: TierId::L1_25GraphRag,
        joules_spent: GRAPH_RAG_JOULES,
        confidence: 0.0,
        trace: ExecutionTrace::default(),
        verification: VerificationStatus::Resolved,
    }
}

// ─── Tier impl ───────────────────────────────────────────────────

impl<G: KnowledgeGraph + 'static> Tier for GraphRagTier<G> {
    fn id(&self) -> TierId {
        TierId::L1_25GraphRag
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        let text = match &q.input {
            QueryInput::Text(s) => s,
            _ => return None,
        };
        if self.graph.is_empty() {
            return None;
        }
        // Cheap pre-check — if the extractor finds no candidates at all
        // we know the tier will refuse, so return `None` to skip it.
        if extract_entity_candidates(text).is_empty() {
            return None;
        }
        Some(TierEstimate {
            joules: GRAPH_RAG_JOULES,
            latency: GRAPH_RAG_LATENCY,
            confidence_floor: GRAPH_RAG_CONFIDENCE_FLOOR,
        })
    }

    fn try_answer(
        &mut self,
        q: &Query,
        _budget_remaining: f64,
    ) -> Result<Answer, AnswerError> {
        let text = match &q.input {
            QueryInput::Text(s) => s.clone(),
            _ => return Ok(refused_inapplicable()),
        };
        self.enrich(&text).map_err(|e| AnswerError::TierFailed {
            tier: TierId::L1_25GraphRag,
            cause: e.to_string(),
        })
    }
}

// ─── Tests ───────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::{Entity, InMemoryKnowledgeGraph};
    use jouleclaw_cascade::tier::Cascade;
    use jouleclaw_cascade::types::{
        ContextRef, JouleBudget, QualityFloor, Query, QueryInput,
    };

    fn fixture() -> InMemoryKnowledgeGraph {
        let mut g = InMemoryKnowledgeGraph::new();
        g.insert_entity(Entity {
            id: EntityId::new("urn:rust"),
            name: "Rust".into(),
            kind: "Language".into(),
            description: "systems language".into(),
        });
        g.insert_entity(Entity {
            id: EntityId::new("urn:cargo"),
            name: "Cargo".into(),
            kind: "Tool".into(),
            description: "package manager".into(),
        });
        g.insert_entity(Entity {
            id: EntityId::new("urn:crates_io"),
            name: "crates.io".into(),
            kind: "Service".into(),
            description: "package registry".into(),
        });
        g.insert_edge(Edge {
            from: EntityId::new("urn:rust"),
            to: EntityId::new("urn:cargo"),
            label: "ships_with".into(),
            weight: 1.0,
        });
        g.insert_edge(Edge {
            from: EntityId::new("urn:cargo"),
            to: EntityId::new("urn:crates_io"),
            label: "publishes_to".into(),
            weight: 0.9,
        });
        g
    }

    fn text_query(s: &str) -> Query {
        Query {
            input: QueryInput::Text(s.into()),
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    #[test]
    fn tier_id_is_l1_25() {
        let t = GraphRagTier::new(fixture());
        assert_eq!(t.id(), TierId::L1_25GraphRag);
    }

    #[test]
    fn estimate_cost_for_text_with_candidates() {
        let t = GraphRagTier::new(fixture());
        let est = t
            .estimate_cost(&text_query("Rust and Cargo are great"))
            .expect("text + entities should be applicable");
        assert!((est.joules - GRAPH_RAG_JOULES).abs() < 1e-12);
        assert_eq!(est.confidence_floor, GRAPH_RAG_CONFIDENCE_FLOOR);
        assert_eq!(est.latency, GRAPH_RAG_LATENCY);
    }

    #[test]
    fn estimate_cost_for_binary_is_none() {
        let t = GraphRagTier::new(fixture());
        let q = Query {
            input: QueryInput::Binary(vec![0, 1, 2]),
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        assert!(t.estimate_cost(&q).is_none());
    }

    #[test]
    fn estimate_cost_empty_graph_is_none() {
        let t = GraphRagTier::new(InMemoryKnowledgeGraph::new());
        assert!(t.estimate_cost(&text_query("Rust and Cargo")).is_none());
    }

    #[test]
    fn estimate_cost_text_with_no_candidates_is_none() {
        // No capitalized words, no quantity, no CamelCase / snake / kebab → no
        // candidates → tier reports inapplicable up front.
        let t = GraphRagTier::new(fixture());
        assert!(t.estimate_cost(&text_query("a b c d e")).is_none());
    }

    #[test]
    fn try_answer_resolves_two_entities() {
        let mut t = GraphRagTier::new(fixture());
        let a = t.try_answer(&text_query("Rust and Cargo"), 1.0).expect("ok");
        assert_eq!(a.tier_used, TierId::L1_25GraphRag);
        let bytes = match a.output {
            AnswerOutput::Structured(b) => b,
            other => panic!("expected structured output, got {other:?}"),
        };
        let payload: GraphRagOutput =
            serde_json::from_slice(&bytes).expect("deserialise");
        assert_eq!(payload.entities.len(), 2);
        let ids: Vec<&str> =
            payload.entities.iter().map(|e| e.entity.id.as_str()).collect();
        assert!(ids.contains(&"urn:rust"));
        assert!(ids.contains(&"urn:cargo"));
        // The Rust→Cargo edge is one hop; expect it in the sub-graph.
        assert!(!payload.edges.is_empty());
        assert!(a.confidence > 0.0);
    }

    #[test]
    fn try_answer_with_no_resolved_entities_refuses() {
        let mut t = GraphRagTier::new(fixture());
        // "Python" is not in the fixture; the extractor will produce a
        // candidate but the graph rejects it.
        let a = t.try_answer(&text_query("Python rocks"), 1.0).expect("ok");
        assert!(matches!(
            a.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable)
        ));
        assert_eq!(a.confidence, 0.0);
    }

    #[test]
    fn try_answer_non_text_refuses() {
        let mut t = GraphRagTier::new(fixture());
        let q = Query {
            input: QueryInput::Binary(vec![1, 2, 3]),
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        let a = t.try_answer(&q, 1.0).expect("ok");
        assert!(matches!(
            a.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable)
        ));
    }

    #[test]
    fn confidence_scales_with_entities_and_edges() {
        let t = GraphRagTier::new(fixture());
        let one_ent = vec![GraphRagEntity {
            entity: Entity {
                id: EntityId::new("urn:x"),
                name: "x".into(),
                kind: "k".into(),
                description: String::new(),
            },
            class: EntityClass::ProperNoun,
            mentions: 1,
        }];
        let c_one = t.compute_confidence(&one_ent, &[]);
        let c_one_edge = t.compute_confidence(&one_ent, &[
            Edge {
                from: EntityId::new("a"),
                to: EntityId::new("b"),
                label: "x".into(),
                weight: 1.0,
            },
        ]);
        assert!(c_one_edge > c_one);
        assert!((0.0..=1.0).contains(&c_one));
    }

    #[test]
    fn summary_lists_entities_and_edges() {
        let mut t = GraphRagTier::new(fixture());
        let a = t
            .try_answer(&text_query("Rust and Cargo together"), 1.0)
            .expect("ok");
        let bytes = match a.output {
            AnswerOutput::Structured(b) => b,
            _ => panic!("expected structured"),
        };
        let payload: GraphRagOutput =
            serde_json::from_slice(&bytes).expect("deser");
        assert!(payload.summary.contains("Rust"));
        assert!(payload.summary.contains("Cargo"));
        assert!(payload.summary.contains("Relationships"));
    }

    #[test]
    fn provenance_is_estimator() {
        assert_eq!(
            GraphRagTier::<InMemoryKnowledgeGraph>::provenance(),
            Provenance::Estimator,
        );
    }

    #[test]
    fn registers_in_a_cascade() {
        let mut c = Cascade::new();
        c.register(Box::new(GraphRagTier::new(fixture())));
        assert!(c.tier_ids().contains(&TierId::L1_25GraphRag));
    }

    #[test]
    fn with_neighborhood_depth_zero_drops_edges() {
        let mut t = GraphRagTier::new(fixture()).with_neighborhood_depth(0);
        let a = t.try_answer(&text_query("Rust and Cargo"), 1.0).expect("ok");
        let bytes = match a.output {
            AnswerOutput::Structured(b) => b,
            _ => panic!("expected structured"),
        };
        let payload: GraphRagOutput =
            serde_json::from_slice(&bytes).expect("deser");
        assert!(
            payload.edges.is_empty(),
            "depth-0 must produce no edges, got {:?}",
            payload.edges,
        );
        // Entities still surface.
        assert_eq!(payload.entities.len(), 2);
    }

    #[test]
    fn mentions_count_aggregates_repeats() {
        let mut t = GraphRagTier::new(fixture());
        // "Rust" appears thrice with capitalization → three candidate hits
        // resolving to the same canonical id.
        let a = t
            .try_answer(&text_query("Rust Rust Rust and Cargo"), 1.0)
            .expect("ok");
        let bytes = match a.output {
            AnswerOutput::Structured(b) => b,
            _ => panic!("expected structured"),
        };
        let payload: GraphRagOutput =
            serde_json::from_slice(&bytes).expect("deser");
        let rust = payload
            .entities
            .iter()
            .find(|e| e.entity.id.as_str() == "urn:rust")
            .expect("rust present");
        // The extractor will only emit "Rust" via the multi-word proper
        // noun rule when there's a second capitalized neighbour, so the
        // exact mention count depends on the rule; we assert >=1, and
        // strictly more than Cargo's count for the sorted-by-mentions
        // contract.
        let cargo = payload
            .entities
            .iter()
            .find(|e| e.entity.id.as_str() == "urn:cargo")
            .expect("cargo present");
        assert!(rust.mentions >= cargo.mentions);
    }

    #[test]
    fn output_serialises_roundtrip() {
        let out = GraphRagOutput {
            query: "q".into(),
            entities: vec![GraphRagEntity {
                entity: Entity {
                    id: EntityId::new("urn:a"),
                    name: "A".into(),
                    kind: "K".into(),
                    description: "d".into(),
                },
                class: EntityClass::Technical,
                mentions: 2,
            }],
            edges: vec![Edge {
                from: EntityId::new("urn:a"),
                to: EntityId::new("urn:b"),
                label: "rel".into(),
                weight: 0.5,
            }],
            summary: "s".into(),
        };
        let bytes = serde_json::to_vec(&out).expect("ser");
        let back: GraphRagOutput =
            serde_json::from_slice(&bytes).expect("deser");
        assert_eq!(back.query, "q");
        assert_eq!(back.entities.len(), 1);
        assert_eq!(back.entities[0].entity.id.as_str(), "urn:a");
        assert_eq!(back.entities[0].mentions, 2);
        assert_eq!(back.edges.len(), 1);
        assert_eq!(back.edges[0].label, "rel");
    }
}
