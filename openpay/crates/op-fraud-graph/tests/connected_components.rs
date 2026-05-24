//! Integration tests for [`op_fraud_graph::ConnectedComponents`].

use op_fraud_graph::{
    ConnectedComponents, EdgeKind, Entity, EntityKind, FraudGraph,
};

fn ent(k: EntityKind, raw: &str) -> Entity {
    Entity::new(k, raw)
}

#[test]
fn isolated_vertices_each_get_their_own_component() {
    let mut g = FraudGraph::new();
    for i in 0..5 {
        let _ = g.upsert_entity(ent(EntityKind::Account, &format!("a-{i}")), 0, false);
    }
    let cc = ConnectedComponents::from_graph(&g);
    assert_eq!(cc.component_count(), 5);
}

#[test]
fn three_islands_three_components() {
    let mut g = FraudGraph::new();
    // Island 1: a-b-c
    let a = g.upsert_entity(ent(EntityKind::Account, "a"), 0, false);
    let b = g.upsert_entity(ent(EntityKind::Account, "b"), 0, false);
    let c = g.upsert_entity(ent(EntityKind::Account, "c"), 0, false);
    g.add_edge(a, b, EdgeKind::CoTransaction, 1.0).expect("ok");
    g.add_edge(b, c, EdgeKind::CoTransaction, 1.0).expect("ok");

    // Island 2: d-e
    let d = g.upsert_entity(ent(EntityKind::Account, "d"), 0, false);
    let e = g.upsert_entity(ent(EntityKind::Account, "e"), 0, false);
    g.add_edge(d, e, EdgeKind::CoTransaction, 1.0).expect("ok");

    // Island 3: f
    let _f = g.upsert_entity(ent(EntityKind::Account, "f"), 0, false);

    let cc = ConnectedComponents::from_graph(&g);
    assert_eq!(cc.component_count(), 3);
    assert_eq!(cc.component_of(a), cc.component_of(c));
    assert_ne!(cc.component_of(a), cc.component_of(d));
}

#[test]
fn filter_by_edge_kind() {
    let mut g = FraudGraph::new();
    let a = g.upsert_entity(ent(EntityKind::Account, "a"), 0, false);
    let b = g.upsert_entity(ent(EntityKind::Account, "b"), 0, false);
    let c = g.upsert_entity(ent(EntityKind::Account, "c"), 0, false);
    g.add_edge(a, b, EdgeKind::CoTransaction, 1.0).expect("ok");
    g.add_edge(b, c, EdgeKind::SharesInstrument, 1.0).expect("ok");

    // All edges: one component (a-b-c).
    let cc_all = ConnectedComponents::from_graph(&g);
    assert_eq!(cc_all.component_count(), 1);

    // Only SharesInstrument: two components (a alone, b-c together).
    let cc_si = ConnectedComponents::from_graph_filtered(&g, |k| {
        k == EdgeKind::SharesInstrument
    });
    assert_eq!(cc_si.component_count(), 2);
}

#[test]
fn components_by_size_is_descending() {
    let mut g = FraudGraph::new();
    // 4-clique
    let big: Vec<_> = (0..4)
        .map(|i| g.upsert_entity(ent(EntityKind::Account, &format!("b-{i}")), 0, false))
        .collect();
    for i in 0..big.len() {
        for j in (i + 1)..big.len() {
            g.add_edge(big[i], big[j], EdgeKind::CoTransaction, 1.0).expect("ok");
        }
    }
    // 2-pair
    let s1 = g.upsert_entity(ent(EntityKind::Account, "s-1"), 0, false);
    let s2 = g.upsert_entity(ent(EntityKind::Account, "s-2"), 0, false);
    g.add_edge(s1, s2, EdgeKind::CoTransaction, 1.0).expect("ok");

    let cc = ConnectedComponents::from_graph(&g);
    let sizes: Vec<u32> = cc.components_by_size().iter().map(|(_, s)| *s).collect();
    assert_eq!(sizes, vec![4, 2]);
}
