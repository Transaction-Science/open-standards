//! End-to-end integration tests against fixture data.
//!
//! Tests never hit a real list endpoint. The parser tests under
//! `src/updater.rs` cover schema-level concerns; this file covers
//! the cross-module path: parse fixture → build index → screen →
//! audit-log integrity.

use chrono::Utc;
use op_screening::{
    AuditLog, EntityType, OfacUpdater, SanctionedEntity, SanctionsIndex, SanctionsList,
    ScreenDecision, ScreenRequest, Screener, ScreenerConfig,
};

fn synthetic_entities(n: usize) -> Vec<SanctionedEntity> {
    let mut out = Vec::with_capacity(n);
    for i in 0..n {
        out.push(SanctionedEntity {
            id: format!("id-{i:04}"),
            name: format!("Person Number {i}"),
            name_aliases: vec![format!("Alias of {i}")],
            entity_type: EntityType::Individual,
            dob: None,
            place_of_birth: None,
            addresses: vec![],
            nationalities: vec![],
            identifications: vec![],
            programs: vec!["TEST".to_string()],
            last_updated: Utc::now(),
            source_list: SanctionsList::OfacSdn,
        });
    }
    // Splice in a single distinguished record we can hit.
    out.push(SanctionedEntity {
        id: "TARGET".to_string(),
        name: "Vladimir Konstantin Petrov".to_string(),
        name_aliases: vec!["V.K. Petrov".to_string(), "Vlad Petrov".to_string()],
        entity_type: EntityType::Individual,
        dob: None,
        place_of_birth: None,
        addresses: vec![],
        nationalities: vec![],
        identifications: vec![],
        programs: vec!["RUSSIA-EO14024".to_string()],
        last_updated: Utc::now(),
        source_list: SanctionsList::OfacSdn,
    });
    out
}

#[tokio::test]
async fn end_to_end_near_match_hits() {
    let entities = synthetic_entities(1_000);
    let idx = SanctionsIndex::build(entities);
    assert_eq!(idx.len(), 1_001);

    let screener = Screener::new(
        idx,
        ScreenerConfig {
            threshold: 0.80,
            ..ScreenerConfig::default()
        },
        AuditLog::new(),
    );

    // Near-match (single-character typo) of the distinguished record.
    let r = screener
        .screen(&ScreenRequest {
            name: "Vladimir Konstantin Petrove".to_string(),
            dob: None,
            address: None,
            additional_ids: vec![],
        })
        .await
        .expect("screen ok");

    assert!(!r.hits.is_empty(), "expected near-match to land");
    assert_eq!(r.hits[0].entity.id, "TARGET");
    assert!(matches!(
        r.decision,
        ScreenDecision::Hit | ScreenDecision::AmbiguousNeedsReview
    ));
}

#[tokio::test]
async fn end_to_end_missing_returns_clear() {
    let entities = synthetic_entities(1_000);
    let idx = SanctionsIndex::build(entities);
    let screener = Screener::new(idx, ScreenerConfig::default(), AuditLog::new());

    let r = screener
        .screen(&ScreenRequest {
            name: "Definitively Not Anyone Sanctioned".to_string(),
            dob: None,
            address: None,
            additional_ids: vec![],
        })
        .await
        .expect("screen ok");

    assert_eq!(r.decision, ScreenDecision::Clear);
    assert!(r.hits.is_empty());
}

#[tokio::test]
async fn audit_log_records_every_call_and_verifies() {
    let entities = synthetic_entities(50);
    let idx = SanctionsIndex::build(entities);
    let screener = Screener::new(idx, ScreenerConfig::default(), AuditLog::new());

    for _ in 0..5 {
        screener
            .screen(&ScreenRequest {
                name: "Some Random Name".to_string(),
                dob: None,
                address: None,
                additional_ids: vec![],
            })
            .await
            .expect("screen ok");
    }
    screener.with_audit_log(|log| {
        assert_eq!(log.entries().len(), 5);
        log.verify().expect("audit chain intact");
    });
}

#[test]
fn parse_fixture_ofac() {
    let xml = include_str!("fixtures/sdn_sample.xml");
    let parsed = OfacUpdater::parse_sdn_xml(xml).expect("parse fixture");
    assert!(!parsed.is_empty(), "fixture must parse to >=1 entity");
    // At least one entry must carry a program.
    assert!(parsed.iter().any(|p| !p.programs.is_empty()));
}
