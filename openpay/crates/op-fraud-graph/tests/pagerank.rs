//! Integration tests for [`op_fraud_graph::PageRank`].

use op_fraud_graph::{
    EdgeKind, Entity, EntityKind, FraudGraph, PageRank,
};

fn acc(g: &mut FraudGraph, name: &str) -> op_fraud_graph::VertexId {
    g.upsert_entity(Entity::new(EntityKind::Account, name), 0, true)
}

#[test]
fn star_topology_hub_wins() {
    // Build a star: one hub connected to 6 leaves.
    let mut g = FraudGraph::new();
    let hub = acc(&mut g, "hub");
    let leaves: Vec<_> = (0..6).map(|i| acc(&mut g, &format!("leaf-{i}"))).collect();
    for l in &leaves {
        g.add_edge(hub, *l, EdgeKind::CoTransaction, 1.0).expect("ok");
    }

    let pr = PageRank::default().run(&g).expect("run ok");
    let hub_score = pr.score(hub).expect("hub scored");
    for l in &leaves {
        let leaf_score = pr.score(*l).expect("leaf scored");
        assert!(
            hub_score > leaf_score,
            "hub {hub_score} should exceed leaf {leaf_score}"
        );
    }
}

#[test]
fn scores_sum_close_to_one() {
    let mut g = FraudGraph::new();
    let a = acc(&mut g, "a");
    let b = acc(&mut g, "b");
    let c = acc(&mut g, "c");
    g.add_edge(a, b, EdgeKind::CoTransaction, 1.0).expect("ok");
    g.add_edge(b, c, EdgeKind::CoTransaction, 1.0).expect("ok");

    let pr = PageRank::default().run(&g).expect("ok");
    let sum: f32 = pr.scores.iter().sum();
    assert!((sum - 1.0).abs() < 0.01, "PageRank sum {sum} should be ~= 1");
}

#[test]
fn rejects_bad_config() {
    let g = FraudGraph::new();
    let bad = PageRank {
        damping: 1.5,
        ..PageRank::default()
    };
    assert!(bad.run(&g).is_err());
}

#[test]
fn empty_graph_is_ok() {
    let g = FraudGraph::new();
    let pr = PageRank::default().run(&g).expect("ok");
    assert!(pr.scores.is_empty());
    assert!(pr.converged);
}

#[test]
fn top_returns_sorted_results() {
    let mut g = FraudGraph::new();
    let hub = acc(&mut g, "hub");
    let nearby = acc(&mut g, "nearby");
    for i in 0..8 {
        let l = acc(&mut g, &format!("l-{i}"));
        g.add_edge(hub, l, EdgeKind::CoTransaction, 1.0).expect("ok");
    }
    g.add_edge(hub, nearby, EdgeKind::CoTransaction, 1.0).expect("ok");

    let pr = PageRank::default().run(&g).expect("ok");
    let top = pr.top(3);
    assert_eq!(top.len(), 3);
    assert_eq!(top[0].0, hub);
    assert!(top[0].1 >= top[1].1);
    assert!(top[1].1 >= top[2].1);
}
