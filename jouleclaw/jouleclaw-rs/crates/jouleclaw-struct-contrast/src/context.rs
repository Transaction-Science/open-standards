//! Graph-enriched context payload contract for L1.375.
//!
//! L1.25 GraphRag (the previous tier) extracts entities from the query and
//! enriches them via the knowledge graph. Rather than have L1.375 re-do
//! entity extraction, L1.25 hands off a small JSON envelope carrying the
//! curated entity list.
//!
//! The envelope is intentionally minimal — just enough to drive the
//! pairwise contrast formula. Future extensions can add fields without
//! breaking older readers as long as the `kind` tag is preserved.

use serde::{Deserialize, Serialize};

/// Stable wire tag for the L1.375 input envelope. Routers and producers
/// MUST set this verbatim. Versioned via the `/vN` suffix so future
/// schema bumps can coexist with older readers.
pub const STRUCT_CONTRAST_KIND: &str = "jouleclaw.struct_contrast/v1";

/// Graph-enriched input carried inside `QueryInput::Structured`.
///
/// Producers (typically the L1.25 GraphRag tier) serialise this to JSON
/// and place the bytes in the `Structured` variant. L1.375 deserialises
/// it on entry and refuses any other shape.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StructContrastInput {
    /// Discriminator tag. Must equal [`STRUCT_CONTRAST_KIND`].
    pub kind: String,
    /// The original query text. Used for the human-readable summary
    /// and the structural-vs-factual short-query heuristic.
    pub query: String,
    /// Graph-resolved entity *names*. The tier looks these up in the
    /// knowledge store; unknown names are dropped silently.
    pub entities: Vec<String>,
}

impl StructContrastInput {
    /// Build a new envelope with the canonical [`STRUCT_CONTRAST_KIND`]
    /// tag pre-filled. Convenience for producers — saves them from
    /// hard-coding the string in two places.
    pub fn new<S, I, E>(query: S, entities: I) -> Self
    where
        S: Into<String>,
        I: IntoIterator<Item = E>,
        E: Into<String>,
    {
        Self {
            kind: STRUCT_CONTRAST_KIND.to_string(),
            query: query.into(),
            entities: entities.into_iter().map(Into::into).collect(),
        }
    }

    /// Whether this envelope's `kind` tag matches the canonical
    /// [`STRUCT_CONTRAST_KIND`]. Producers that fail this check should
    /// be rejected by readers without further deserialisation.
    pub fn kind_matches(&self) -> bool {
        self.kind == STRUCT_CONTRAST_KIND
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_sets_canonical_kind() {
        let env = StructContrastInput::new(
            "compare fire and water",
            ["fire", "water"],
        );
        assert_eq!(env.kind, STRUCT_CONTRAST_KIND);
        assert!(env.kind_matches());
        assert_eq!(env.entities, vec!["fire".to_string(), "water".to_string()]);
    }

    #[test]
    fn round_trips_through_json() {
        let env = StructContrastInput::new("q", ["a", "b", "c"]);
        let bytes = serde_json::to_vec(&env).expect("serialise");
        let back: StructContrastInput =
            serde_json::from_slice(&bytes).expect("deserialise");
        assert_eq!(env, back);
    }

    #[test]
    fn kind_mismatch_is_detected() {
        let bad = StructContrastInput {
            kind: "not.us/v1".into(),
            query: "q".into(),
            entities: vec![],
        };
        assert!(!bad.kind_matches());
    }
}
