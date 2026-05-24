//! Integration tests for [`op_fraud_graph::RingDetector`].

use op_fraud_graph::{
    EdgeKind, Entity, EntityKind, FraudGraph, RingDetector,
};

#[test]
fn detects_shared_card_across_many_accounts() {
    let mut g = FraudGraph::new();
    let card = g.upsert_entity(Entity::new(EntityKind::CardHash, "PAN-1"), 0, true);
    let mut accounts = Vec::new();
    for i in 0..7 {
        let a = g.upsert_entity(
            Entity::new(EntityKind::Account, &format!("acc-{i}")),
            0,
            true,
        );
        g.add_edge(card, a, EdgeKind::SharesInstrument, 1.0).expect("ok");
        accounts.push(a);
    }

    let rings = RingDetector::default().detect(&g);
    let ring = rings
        .iter()
        .find(|r| r.hub == card)
        .expect("card-hub ring detected");
    assert_eq!(ring.size(), 7);
    assert!(ring.score() > 0.0);
}

#[test]
fn ignores_card_below_threshold() {
    let mut g = FraudGraph::new();
    let card = g.upsert_entity(Entity::new(EntityKind::CardHash, "PAN-2"), 0, true);
    for i in 0..2 {
        let a = g.upsert_entity(
            Entity::new(EntityKind::Account, &format!("a-{i}")),
            0,
            true,
        );
        g.add_edge(card, a, EdgeKind::SharesInstrument, 1.0).expect("ok");
    }

    let rings = RingDetector::default().detect(&g);
    assert!(rings.iter().all(|r| r.hub != card));
}

#[test]
fn ignores_wrong_edge_kind() {
    let mut g = FraudGraph::new();
    let card = g.upsert_entity(Entity::new(EntityKind::CardHash, "PAN-3"), 0, true);
    for i in 0..10 {
        let a = g.upsert_entity(
            Entity::new(EntityKind::Account, &format!("x-{i}")),
            0,
            true,
        );
        g.add_edge(card, a, EdgeKind::CoTransaction, 1.0).expect("ok");
    }

    let rings = RingDetector::default().detect(&g);
    assert!(rings.iter().all(|r| r.hub != card));
}

#[test]
fn respects_min_edge_weight() {
    let mut g = FraudGraph::new();
    let card = g.upsert_entity(Entity::new(EntityKind::CardHash, "PAN-4"), 0, true);
    for i in 0..10 {
        let a = g.upsert_entity(
            Entity::new(EntityKind::Account, &format!("y-{i}")),
            0,
            true,
        );
        // Weight 0.1 < default min_edge_weight (1.0)
        g.add_edge(card, a, EdgeKind::SharesInstrument, 0.1).expect("ok");
    }

    let rings = RingDetector::default().detect(&g);
    assert!(rings.iter().all(|r| r.hub != card));
}
