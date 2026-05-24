//! Episodic log + temporal index integration tests.

use eoc_memory::episodic::{Episode, EpisodicLog};
use eoc_memory::memory::Memory;

#[test]
fn appends_are_monotonic_and_queryable() {
    let mut log = EpisodicLog::new();
    log.append(Episode::new(1_000, "user", "hello"))
        .expect("append 1");
    log.append(Episode::new(2_000, "assistant", "hi"))
        .expect("append 2");
    log.append(Episode::new(5_000, "user", "still there?"))
        .expect("append 3");

    let in_window = log.range(1_500, 4_000);
    assert_eq!(in_window.len(), 1);
    assert_eq!(in_window[0].payload, "hi");

    let all = log.range(0, 10_000);
    assert_eq!(all.len(), 3);
}

#[test]
fn non_monotonic_timestamp_is_rejected() {
    let mut log = EpisodicLog::new();
    log.append(Episode::new(10_000, "user", "first"))
        .expect("first");
    let err = log.append(Episode::new(5_000, "user", "back-dated"));
    assert!(err.is_err(), "non-monotonic timestamp should error");
}

#[test]
fn recent_returns_newest_first() {
    let mut log = EpisodicLog::new();
    log.append(Episode::new(1, "u", "a")).expect("a");
    log.append(Episode::new(2, "u", "b")).expect("b");
    log.append(Episode::new(3, "u", "c")).expect("c");

    let recent = log.recent(2).expect("recent ok");
    assert_eq!(recent.len(), 2);
    assert!(recent[0].text.contains('c'));
    assert!(recent[1].text.contains('b'));
}

#[test]
fn episode_ids_are_deterministic() {
    let a = Episode::new(42, "user", "same payload");
    let b = Episode::new(42, "user", "same payload");
    assert_eq!(a.id, b.id);
    let c = Episode::new(43, "user", "same payload");
    assert_ne!(a.id, c.id);
}
