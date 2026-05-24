//! Dream-cycle consolidation tests.

use eoc_memory::consolidate::{ConsolidateConfig, consolidate};
use eoc_memory::episodic::{Episode, EpisodicLog};
use eoc_memory::memory::Memory;
use eoc_memory::semantic::{SemanticGraph, Triple};

#[test]
fn consolidates_episodes_to_triples_and_skips_dupes() {
    let mut log = EpisodicLog::new();
    log.append(Episode::new(100, "user", "alice knows bob"))
        .expect("ep1");
    log.append(Episode::new(200, "user", "bob knows carol"))
        .expect("ep2");
    log.append(Episode::new(300, "user", "alice knows bob"))
        .expect("ep3 dup payload");

    let mut graph = SemanticGraph::new();
    let cfg = ConsolidateConfig::new((0, 1_000), 64).expect("cfg");

    // Extractor: parse "X knows Y" into a triple.
    let extractor = |ep: &Episode| -> Vec<Triple> {
        let parts: Vec<&str> = ep.payload.split_whitespace().collect();
        if parts.len() == 3 && parts[1] == "knows" {
            vec![Triple::new(parts[0], "knows", parts[2], ep.timestamp_ms)]
        } else {
            Vec::new()
        }
    };

    let report = consolidate(&log, &mut graph, &cfg, extractor).expect("consolidate");
    assert_eq!(report.episodes_scanned, 3);
    assert_eq!(report.triples_asserted, 2);
    assert_eq!(report.triples_duplicate, 1);
    assert_eq!(graph.len(), 2);
}

#[test]
fn max_triples_caps_output() {
    let mut log = EpisodicLog::new();
    for i in 0..10 {
        log.append(Episode::new(
            1_000 + i,
            "u",
            format!("a{i} knows b{i}"),
        ))
        .expect("ep");
    }
    let mut graph = SemanticGraph::new();
    let cfg = ConsolidateConfig::new((0, 100_000), 3).expect("cfg");
    let report = consolidate(&log, &mut graph, &cfg, |ep| {
        let parts: Vec<&str> = ep.payload.split_whitespace().collect();
        vec![Triple::new(parts[0], "knows", parts[2], ep.timestamp_ms)]
    })
    .expect("consolidate");
    assert_eq!(report.triples_asserted, 3);
    assert!(graph.len() <= 3);
}

#[test]
fn empty_window_yields_empty_report() {
    let log = EpisodicLog::new();
    let mut graph = SemanticGraph::new();
    let cfg = ConsolidateConfig::new((0, 1_000), 16).expect("cfg");
    let report = consolidate(&log, &mut graph, &cfg, |_| Vec::new()).expect("ok");
    assert_eq!(report.episodes_scanned, 0);
    assert_eq!(report.triples_asserted, 0);
}
