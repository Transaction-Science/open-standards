//! End-to-end: Visa 10.4 fraud chargeback → representment.
//!
//! Walks the lifecycle from the issuer's first chargeback through a
//! sealed evidence package and into a representment filing. This is
//! the canonical happy-path the directive's CE3.0-eligible 10.4
//! flow targets.

use op_dispute_evidence::{
    EvidenceItem, EvidencePackageBuilder, EvidenceRequirement, LifecycleEvent, LifecycleMachine,
    Network, Phase, ReasonCode, ReasonCodeCatalog, VisaReasonCode, WinScore, WinScoreBand,
};
use time::OffsetDateTime;

fn t(s: i64) -> OffsetDateTime {
    OffsetDateTime::from_unix_timestamp(s).expect("valid unix time")
}

#[test]
fn visa_10_4_fraud_representment_flow() {
    let reason = ReasonCode::Visa(VisaReasonCode::F1040);

    // Sanity-check the catalog: 10.4 lives on Visa and demands the
    // CE3.0-style "qualifying history" item.
    assert_eq!(reason.network(), Network::Visa);
    assert_eq!(reason.code(), "10.4");
    let required = ReasonCodeCatalog::required_evidence(reason);
    assert!(required.contains(&EvidenceRequirement::QualifyingHistory));
    assert!(required.contains(&EvidenceRequirement::ThreeDsAuthValue));

    // 1. Issuer files the chargeback. Start the machine straight in
    //    FirstChargeback (skip the rare retrieval-request step).
    let mut machine = LifecycleMachine::starting_at(Phase::FirstChargeback);
    assert_eq!(machine.phase(), Phase::FirstChargeback);

    // 2. Operator packages evidence. Must cover every required
    //    item or the seal will fail.
    let mut builder = EvidencePackageBuilder::new(reason);
    for req in required {
        let item = EvidenceItem::new(
            *req,
            format!("{req:?}"),
            "application/octet-stream",
            b"opaque".to_vec(),
            t(1_700_000_000),
        )
        .expect("payload under cap");
        builder = builder.add(item);
    }
    assert!(builder.missing().is_empty(), "all required satisfied");

    let package = builder
        .seal(t(1_700_000_500))
        .expect("seal complete package");
    assert!(package.total_bytes() > 0);
    assert_eq!(package.items().count(), required.len());

    // 3. Score it. Full 10.4 bundle including 3DS + CE3 history
    //    should land in Likely or Strong.
    let score = WinScore::evaluate(reason, &package);
    assert_eq!(score.missing_required, 0);
    assert!(
        matches!(score.band, WinScoreBand::Likely | WinScoreBand::Strong),
        "expected Likely/Strong, got {:?} (p={})",
        score.band,
        score.probability
    );

    // 4. File the representment, then receive the network's "won"
    //    ruling. Each transition must be accepted.
    machine
        .apply(LifecycleEvent::RepresentmentFiled { at: t(1_700_000_600) })
        .expect("first chargeback -> representment");
    assert_eq!(machine.phase(), Phase::Representment);

    machine
        .apply(LifecycleEvent::Won { at: t(1_700_001_000) })
        .expect("representment -> won");
    assert_eq!(machine.phase(), Phase::FinalWon);
    assert!(machine.phase().is_terminal());

    // 5. Any further event must be rejected — terminal phases are
    //    closed.
    let trailing = machine.apply(LifecycleEvent::Lost { at: t(1_700_001_500) });
    assert!(trailing.is_err());

    // 6. The full transition log captures both moves.
    assert_eq!(machine.history().len(), 2);
}
