//! Integration tests for [`op_fraud_graph::VelocityCounter`].

use op_fraud_graph::{EntityKey, EntityKind, VelocityCounter, VelocityWindow};

fn k(s: &str) -> EntityKey {
    EntityKey::from_raw(EntityKind::Account, s)
}

#[test]
fn counts_within_window() {
    let mut c = VelocityCounter::new(VelocityWindow::default()).expect("ok");
    let key = k("acc-A");
    for t in 0..20 {
        c.record(key, t);
    }
    assert_eq!(c.count(key, 19), 20);
}

#[test]
fn drops_events_outside_window() {
    let mut c = VelocityCounter::new(VelocityWindow {
        window_secs: 10,
        bucket_secs: 1,
    })
    .expect("ok");
    let key = k("acc-B");
    for t in 0..10 {
        c.record(key, t);
    }
    assert_eq!(c.count(key, 9), 10);
    // Advance halfway: ts=14 → buckets at t=0..4 should have rolled off
    // (window covers ts-9..=ts).
    let later = c.count(key, 14);
    assert!(later < 10 && later >= 5, "got {later}");
}

#[test]
fn unknown_key_is_zero() {
    let mut c = VelocityCounter::new(VelocityWindow::default()).expect("ok");
    assert_eq!(c.count(k("missing"), 0), 0);
}

#[test]
fn evict_stale_removes_old_entries() {
    let mut c = VelocityCounter::new(VelocityWindow {
        window_secs: 5,
        bucket_secs: 1,
    })
    .expect("ok");
    let key = k("acc-C");
    c.record(key, 0);
    assert_eq!(c.count(key, 0), 1);
    c.evict_stale(1000);
    // After eviction, the key should report zero (no state).
    assert_eq!(c.count(key, 1000), 0);
}

#[test]
fn multiple_keys_isolated() {
    let mut c = VelocityCounter::new(VelocityWindow::default()).expect("ok");
    let a = k("acc-X");
    let b = k("acc-Y");
    c.record(a, 0);
    c.record(a, 1);
    c.record(b, 0);
    assert_eq!(c.count(a, 1), 2);
    assert_eq!(c.count(b, 1), 1);
}
