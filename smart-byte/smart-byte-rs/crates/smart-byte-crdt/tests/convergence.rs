//! End-to-end convergence tests for the Smart Byte CRDT engine.

use proptest::prelude::*;
use smart_byte_crdt::{
    apply,
    document::{CrdtNode, DocumentId, Value},
    hlc::{HlcClock, ReplicaId},
    sync::{clock_of, diff},
    types::{Crdt, CrdtId, GCounter, LwwRegister, OrSet, PnCounter, UniqueTag},
    CrdtDocument,
};

fn rid(n: u128) -> ReplicaId {
    ReplicaId::new(n)
}

#[test]
fn three_replicas_text_edit_converges() {
    let id = DocumentId::from_bytes(b"doc");
    let mut docs: Vec<(ReplicaId, CrdtDocument, HlcClock)> = (1u128..=3)
        .map(|n| {
            let r = rid(n);
            (
                r,
                CrdtDocument::new(id, r),
                HlcClock::with_manual_wall(r, n as u64),
            )
        })
        .collect();

    // Replica 1 inserts "AB" at /text.
    let (r1, mut d1, mut c1) = docs.remove(0);
    let (_o1, p_a) = d1.text_insert("/text", None, 'A', &mut c1, 1).unwrap();
    let (_o2, p_b) = d1.text_insert("/text", Some(p_a), 'B', &mut c1, 2).unwrap();

    // Replica 2 starts from R1's state, then inserts "X" after A.
    let (r2, mut d2, mut c2) = docs.remove(0);
    let ops_to_2 = diff(&d1, &clock_of(&d2));
    apply(&mut d2, &ops_to_2).unwrap();
    let (_o3, _p_x) = d2.text_insert("/text", Some(p_a), 'X', &mut c2, 1).unwrap();

    // Replica 3 starts from R1's state, then inserts "Y" after B.
    let (r3, mut d3, mut c3) = docs.remove(0);
    let ops_to_3 = diff(&d1, &clock_of(&d3));
    apply(&mut d3, &ops_to_3).unwrap();
    let (_o4, _p_y) = d3.text_insert("/text", Some(p_b), 'Y', &mut c3, 1).unwrap();

    // Cross-merge.
    let ops_d2_to_d1 = diff(&d2, &clock_of(&d1));
    let ops_d3_to_d1 = diff(&d3, &clock_of(&d1));
    apply(&mut d1, &ops_d2_to_d1).unwrap();
    apply(&mut d1, &ops_d3_to_d1).unwrap();

    let ops_d1_to_d2 = diff(&d1, &clock_of(&d2));
    apply(&mut d2, &ops_d1_to_d2).unwrap();
    let ops_d1_to_d3 = diff(&d1, &clock_of(&d3));
    apply(&mut d3, &ops_d1_to_d3).unwrap();

    let s1 = collect_text(&d1, "/text");
    let s2 = collect_text(&d2, "/text");
    let s3 = collect_text(&d3, "/text");
    assert_eq!(s1, s2);
    assert_eq!(s2, s3);
    assert!(s1.contains('A'));
    assert!(s1.contains('B'));
    assert!(s1.contains('X'));
    assert!(s1.contains('Y'));

    let _ = (r1, r2, r3);
}

fn collect_text(doc: &CrdtDocument, path: &str) -> String {
    match doc.get_node(path) {
        Some(CrdtNode::Text(t)) => t.iter().collect(),
        _ => String::new(),
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(40))]

    #[test]
    fn gcounter_arbitrary_ops_converge(
        ops in proptest::collection::vec((0u8..3u8, 1u64..50u64), 1..50)
    ) {
        // Each entry: (replica index 0..3, increment amount).
        let id = CrdtId::new(7);
        let mut a = GCounter::new(id);
        let mut b = GCounter::new(id);
        let mut c = GCounter::new(id);

        for (which, amt) in &ops {
            let r = ReplicaId::new(*which as u128 + 1);
            match which {
                0 => a.increment(r, *amt),
                1 => b.increment(r, *amt),
                _ => c.increment(r, *amt),
            }
        }

        // Merge in two different orders; both must yield identical total.
        let mut order1 = a.clone();
        order1.merge(&b);
        order1.merge(&c);

        let mut order2 = c.clone();
        order2.merge(&a);
        order2.merge(&b);

        prop_assert_eq!(order1.value(), order2.value());

        // Idempotency.
        let v = order1.value();
        order1.merge(&b);
        order1.merge(&c);
        prop_assert_eq!(order1.value(), v);
    }

    #[test]
    fn pncounter_arbitrary_ops_converge(
        ops in proptest::collection::vec((0u8..3u8, -50i64..50i64), 1..50)
    ) {
        let id = CrdtId::new(7);
        let mut a = PnCounter::new(id);
        let mut b = PnCounter::new(id);
        let mut c = PnCounter::new(id);

        for (which, delta) in &ops {
            let r = ReplicaId::new(*which as u128 + 1);
            let target = match which {
                0 => &mut a,
                1 => &mut b,
                _ => &mut c,
            };
            if *delta >= 0 {
                target.increment(r, *delta as u64);
            } else {
                target.decrement(r, delta.unsigned_abs());
            }
        }

        let mut order1 = a.clone();
        order1.merge(&b);
        order1.merge(&c);

        let mut order2 = c.clone();
        order2.merge(&a);
        order2.merge(&b);

        prop_assert_eq!(order1.value(), order2.value());
    }

    #[test]
    fn lww_register_converges_regardless_of_order(
        writes in proptest::collection::vec((0u8..3u8, 0u64..100u64, 0i64..1_000i64), 2..20)
    ) {
        // Each write: (replica, wall_ms, value)
        let id = CrdtId::new(11);
        let r0 = ReplicaId::new(1);
        let mut a = LwwRegister::new(id, 0i64, smart_byte_crdt::HybridLogicalClock { wall: 0, logical: 0, node: r0 }, r0);
        let mut b = a.clone();

        for (which, wall, value) in &writes {
            let r = ReplicaId::new(*which as u128 + 1);
            let ts = smart_byte_crdt::HybridLogicalClock { wall: *wall, logical: 0, node: r };
            match which {
                0 => a.write(*value, ts, r),
                _ => b.write(*value, ts, r),
            }
        }

        let mut ab = a.clone();
        ab.merge(&b);
        let mut ba = b.clone();
        ba.merge(&a);
        prop_assert_eq!(*ab.get(), *ba.get());
    }

    #[test]
    fn orset_observed_remove_converges(
        sequence in proptest::collection::vec((0u8..3u8, 0u8..1u8, 0u8..5u8), 1..30)
    ) {
        // Each entry: (replica, op (0=add, 1=remove), item id)
        let id = CrdtId::new(13);
        let mut a: OrSet<u8> = OrSet::new(id);
        let mut b: OrSet<u8> = OrSet::new(id);
        let mut c: OrSet<u8> = OrSet::new(id);
        for (i, (which, op, item)) in sequence.iter().enumerate() {
            let r = ReplicaId::new(*which as u128 + 1);
            let target = match which {
                0 => &mut a,
                1 => &mut b,
                _ => &mut c,
            };
            let nonce = (i as u64) + 1;
            let tag = UniqueTag {
                hlc: smart_byte_crdt::HybridLogicalClock { wall: nonce, logical: 0, node: r },
                replica: r,
                nonce,
            };
            match op {
                0 => target.add(*item, tag),
                _ => target.remove(item),
            }
        }

        let mut order1 = a.clone();
        order1.merge(&b);
        order1.merge(&c);
        let mut order2 = c.clone();
        order2.merge(&a);
        order2.merge(&b);

        prop_assert_eq!(order1.snapshot(), order2.snapshot());
    }
}

#[test]
fn document_set_via_path_round_trips_value() {
    let id = DocumentId::from_bytes(b"d");
    let r = rid(1);
    let mut doc = CrdtDocument::new(id, r);
    let mut clock = HlcClock::with_manual_wall(r, 5);
    doc.write_at("/x/y/z", Value::Text("hello".into()), &mut clock)
        .unwrap();
    match doc.get_node("/x/y/z") {
        Some(CrdtNode::Register(reg)) => assert_eq!(*reg.get(), Value::Text("hello".into())),
        _ => panic!("expected register"),
    }
}
