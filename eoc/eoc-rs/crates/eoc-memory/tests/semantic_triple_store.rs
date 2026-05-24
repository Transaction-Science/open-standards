//! Semantic-graph triple store integration tests.

use eoc_memory::memory::Memory;
use eoc_memory::semantic::{SemanticGraph, Triple};

#[test]
fn assert_and_index_by_spo() {
    let mut g = SemanticGraph::new();
    g.assert(Triple::new("alice", "knows", "bob", 1));
    g.assert(Triple::new("alice", "knows", "carol", 2));
    g.assert(Triple::new("bob", "knows", "carol", 3));

    assert_eq!(g.subject("alice").len(), 2);
    assert_eq!(g.predicate("knows").len(), 3);
    assert_eq!(g.object("carol").len(), 2);
}

#[test]
fn duplicate_assertions_are_idempotent() {
    let mut g = SemanticGraph::new();
    g.assert(Triple::new("a", "p", "b", 1));
    g.assert(Triple::new("a", "p", "b", 2));
    g.assert(Triple::new("a", "p", "b", 3));
    assert_eq!(g.len(), 1);
    // Timestamp should have advanced to the latest.
    assert_eq!(g.all()[0].timestamp_ms, 3);
}

#[test]
fn triple_id_is_stable() {
    let t1 = Triple::new("alice", "knows", "bob", 1);
    let t2 = Triple::new("alice", "knows", "bob", 999);
    assert_eq!(t1.id(), t2.id(), "id must not depend on timestamp");
}

#[test]
fn recent_returns_newest_by_timestamp() {
    let mut g = SemanticGraph::new();
    g.assert(Triple::new("a", "p", "b", 10));
    g.assert(Triple::new("c", "p", "d", 30));
    g.assert(Triple::new("e", "p", "f", 20));

    let recent = g.recent(2).expect("recent");
    assert_eq!(recent.len(), 2);
    assert_eq!(recent[0].timestamp_ms, 30);
    assert_eq!(recent[1].timestamp_ms, 20);
}
