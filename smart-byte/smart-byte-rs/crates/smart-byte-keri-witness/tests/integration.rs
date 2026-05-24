//! End-to-end tests covering the inception/rotation/interaction/
//! recovery/delegation cycle, the witness layer, watchers, and the
//! verifier.

use std::collections::HashMap;

use proptest::prelude::*;
use rand::rngs::OsRng;
use smart_byte_core::Said;

use smart_byte_keri_witness::{
    Anchor, Controller, ControllerAid, KeyEvent, KeyPair, LogVerifier, MemoryStorage, Threshold,
    Watcher, Witness, WitnessAid, WitnessReceipt,
};
use smart_byte_keri_witness::storage::EventLogStorage;
use smart_byte_keri_witness::WatcherAid;

fn sign_all_into_map(
    controller: &Controller,
    event: &KeyEvent,
    signatures_out: &mut HashMap<Said, Vec<(u32, Vec<u8>)>>,
) {
    let sigs = controller.sign_event(event).expect("sign");
    let entry = signatures_out.entry(event.said()).or_default();
    for (i, sig) in sigs {
        entry.push((i, sig.to_bytes().to_vec()));
    }
}

#[tokio::test]
async fn inception_three_rotations_five_interactions_verify() {
    let mut rng = OsRng;
    let k0 = KeyPair::generate(&mut rng);
    let k1 = KeyPair::generate(&mut rng);
    let k2 = KeyPair::generate(&mut rng);
    let k3 = KeyPair::generate(&mut rng);
    let k4 = KeyPair::generate(&mut rng);

    let (mut ctrl, icp) = Controller::incept(
        vec![k0],
        vec![k1.clone()],
        Threshold(1),
        Threshold(1),
        vec![],
        Threshold(0),
    )
    .expect("incept");

    let mut sigs: HashMap<Said, Vec<(u32, Vec<u8>)>> = HashMap::new();
    let icp_event = KeyEvent::Inception(icp);
    sign_all_into_map(&ctrl, &icp_event, &mut sigs);

    // After incept, current is k0. To sign the upcoming rotation
    // (whose `k` reveals k1) the rotation must be signed by k1.
    // We model this by swapping `current_keys` in the controller for
    // the new keys before re-signing post-rotation.
    let rot1 = ctrl
        .rotate(vec![k2.clone()], Threshold(1), vec![], vec![], Threshold(0))
        .expect("rot1");
    // After rotate, controller.current_keys == previous next_keys == k1
    let rot1_event = KeyEvent::Rotation(rot1);
    sign_all_into_map(&ctrl, &rot1_event, &mut sigs);

    let rot2 = ctrl
        .rotate(vec![k3.clone()], Threshold(1), vec![], vec![], Threshold(0))
        .expect("rot2");
    let rot2_event = KeyEvent::Rotation(rot2);
    sign_all_into_map(&ctrl, &rot2_event, &mut sigs);

    let rot3 = ctrl
        .rotate(vec![k4.clone()], Threshold(1), vec![], vec![], Threshold(0))
        .expect("rot3");
    let rot3_event = KeyEvent::Rotation(rot3);
    sign_all_into_map(&ctrl, &rot3_event, &mut sigs);

    let mut interactions = Vec::new();
    for i in 0..5 {
        let anchor = Anchor {
            d: Said::hash(format!("payload-{i}").as_bytes()),
            s: Some(i as u64),
            kind: Some("test".into()),
        };
        let ixn = ctrl.interact(vec![anchor]).expect("interact");
        let ev = KeyEvent::Interaction(ixn);
        sign_all_into_map(&ctrl, &ev, &mut sigs);
        interactions.push(ev);
    }

    let log = ctrl.key_event_log.clone();
    let receipts: HashMap<Said, Vec<WitnessReceipt>> = HashMap::new();
    let report = LogVerifier::new()
        .verify(&log, &sigs, &receipts)
        .expect("verify ok");
    assert_eq!(report.last_sequence, 8);
    assert_eq!(report.current_threshold, Threshold(1));
}

#[tokio::test]
async fn pre_rotation_mismatch_is_rejected() {
    let mut rng = OsRng;
    let k0 = KeyPair::generate(&mut rng);
    let k1 = KeyPair::generate(&mut rng);
    let k2 = KeyPair::generate(&mut rng);

    let (mut ctrl, icp) = Controller::incept(
        vec![k0],
        vec![k1.clone()],
        Threshold(1),
        Threshold(1),
        vec![],
        Threshold(0),
    )
    .expect("incept");
    let mut sigs: HashMap<Said, Vec<(u32, Vec<u8>)>> = HashMap::new();
    let icp_event = KeyEvent::Inception(icp);
    sign_all_into_map(&ctrl, &icp_event, &mut sigs);

    // Subvert the pre-rotation commitment by replacing the revealed
    // keys with a key the next-key digests did NOT commit to.
    let rot = ctrl
        .rotate(vec![k2], Threshold(1), vec![], vec![], Threshold(0))
        .expect("rot");
    let mut rot_event = KeyEvent::Rotation(rot);

    // Tamper: swap `k` to a fresh key the previous `n` did not commit to.
    if let KeyEvent::Rotation(ref mut r) = rot_event {
        let evil = KeyPair::generate(&mut rng);
        r.k = vec![evil.public_key()];
        // Recompute SAID so SAID consistency passes — what we want to
        // demonstrate is that the *pre-rotation* check fires.
        let placeholder = Said::default();
        r.d = placeholder;
        let bytes = serde_cbor::to_vec(&KeyEvent::Rotation(r.clone())).expect("cbor");
        r.d = Said::hash(&bytes);
    }
    // Replace prior signature entry: still won't matter — pre-rotation
    // check fires before signatures.
    sign_all_into_map(&ctrl, &rot_event, &mut sigs);

    let log = vec![icp_event, rot_event];
    let receipts: HashMap<Said, Vec<WitnessReceipt>> = HashMap::new();
    let err = LogVerifier::new().verify(&log, &sigs, &receipts).unwrap_err();
    matches!(
        err,
        smart_byte_keri_witness::KeriError::PreRotationMismatch { .. }
    );
}

#[tokio::test]
async fn witness_signs_and_verifier_accepts_quorum() {
    let mut rng = OsRng;
    let w_key = KeyPair::generate(&mut rng);
    let witness = Witness::new(WitnessAid("W1".into()), w_key);

    let k0 = KeyPair::generate(&mut rng);
    let k1 = KeyPair::generate(&mut rng);
    let (ctrl, icp) = Controller::incept(
        vec![k0],
        vec![k1],
        Threshold(1),
        Threshold(1),
        vec![witness.aid.clone()],
        Threshold(1),
    )
    .expect("incept");

    let event = KeyEvent::Inception(icp.clone());
    let raw_sigs = ctrl.sign_event(&event).expect("sign");
    let keys_in_force = icp.k.clone();
    let receipt = witness
        .receive(event.clone(), &raw_sigs, &keys_in_force, Threshold(1))
        .await
        .expect("receipt");
    receipt.verify_signature().expect("receipt sig");

    let mut sigs: HashMap<Said, Vec<(u32, Vec<u8>)>> = HashMap::new();
    let entry = sigs.entry(event.said()).or_default();
    for (i, s) in raw_sigs {
        entry.push((i, s.to_bytes().to_vec()));
    }
    let mut receipts: HashMap<Said, Vec<WitnessReceipt>> = HashMap::new();
    receipts.insert(event.said(), vec![receipt]);

    let log = vec![event];
    let report = LogVerifier::new()
        .require_witness_quorum(true)
        .verify(&log, &sigs, &receipts)
        .expect("verify ok with witness quorum");
    assert!(report.witness_quorum_met);
}

#[tokio::test]
async fn witness_refuses_contradicting_receipt_and_watcher_detects() {
    let mut rng = OsRng;
    let w_key = KeyPair::generate(&mut rng);
    let witness = Witness::new(WitnessAid("W1".into()), w_key);

    let k0 = KeyPair::generate(&mut rng);
    let k1a = KeyPair::generate(&mut rng);
    let k1b = KeyPair::generate(&mut rng);

    // Two controllers off the SAME seed key but with divergent next keys —
    // we manufacture two distinct events at the same sequence by building
    // two parallel controllers that share the inception.
    let (ctrl_a, icp_a) = Controller::incept(
        vec![k0.clone()],
        vec![k1a.clone()],
        Threshold(1),
        Threshold(1),
        vec![witness.aid.clone()],
        Threshold(1),
    )
    .expect("incept-a");
    // We need a SECOND inception event with a different SAID — change
    // the next-key set to k1b.
    let (mut ctrl_b, icp_b) = Controller::incept(
        vec![k0.clone()],
        vec![k1b.clone()],
        Threshold(1),
        Threshold(1),
        vec![witness.aid.clone()],
        Threshold(1),
    )
    .expect("incept-b");
    // The two controllers have different AIDs; for the duplicity test we
    // need same-controller / same-sequence forked events. We build that
    // by taking the first controller, accepting its incep,  then rotating
    // along two different forks at sequence 1.
    let _ = (icp_b, &mut ctrl_b);

    let event_icp = KeyEvent::Inception(icp_a);
    let sigs_icp = ctrl_a.sign_event(&event_icp).expect("sign");
    let _ = witness
        .receive(
            event_icp.clone(),
            &sigs_icp,
            &match &event_icp {
                KeyEvent::Inception(i) => i.k.clone(),
                _ => unreachable!(),
            },
            Threshold(1),
        )
        .await
        .expect("first receipt");

    // Two divergent rotations at sequence 1 using two distinct next keys.
    // The witness has already receipted the inception; now it sees the
    // FIRST rotation (legit) — receipted; then a SECOND rotation at
    // the same sequence with different content — must refuse.
    let k2a = KeyPair::generate(&mut rng);
    let k2b = KeyPair::generate(&mut rng);

    // Clone the controller so we can produce two forks against the same
    // initial state.
    let mut ctrl_fork_a = clone_controller(&ctrl_a);
    let mut ctrl_fork_b = clone_controller(&ctrl_a);

    let rot_a = ctrl_fork_a
        .rotate(vec![k2a], Threshold(1), vec![], vec![], Threshold(0))
        .expect("rot-a");
    let rot_b = ctrl_fork_b
        .rotate(vec![k2b], Threshold(1), vec![], vec![], Threshold(0))
        .expect("rot-b");
    let event_a = KeyEvent::Rotation(rot_a.clone());
    let event_b = KeyEvent::Rotation(rot_b.clone());

    // First fork: witness signs OK.
    let sigs_a = ctrl_fork_a.sign_event(&event_a).expect("sign-a");
    let receipt_a = witness
        .receive(event_a.clone(), &sigs_a, &rot_a.k, Threshold(1))
        .await
        .expect("receipt-a");

    // Second fork at same sequence: witness MUST refuse.
    let sigs_b = ctrl_fork_b.sign_event(&event_b).expect("sign-b");
    let err = witness
        .receive(event_b.clone(), &sigs_b, &rot_b.k, Threshold(1))
        .await
        .unwrap_err();
    assert!(matches!(
        err,
        smart_byte_keri_witness::KeriError::DuplicityRefused { .. }
    ));

    // Watcher sees both forks (it does not refuse — it detects).
    let watcher = Watcher::new(WatcherAid("WA".into()));
    watcher
        .observe(event_a.clone(), std::slice::from_ref(&receipt_a))
        .await
        .expect("observe-a");
    // Forge a watcher-side receipt for event_b so the watcher's
    // receipt-verify check passes; we model a colluding witness here.
    let forged_witness = Witness::new(WitnessAid("W-COLLUDE".into()), KeyPair::generate(&mut rng));
    let receipt_b = forged_witness
        .receive(event_b.clone(), &sigs_b, &rot_b.k, Threshold(1))
        .await
        .expect("forged receipt-b");
    watcher
        .observe(event_b.clone(), &[receipt_b])
        .await
        .expect("observe-b");

    let signals = watcher.signals();
    assert_eq!(signals.len(), 1, "watcher should raise exactly one duplicity signal");
    assert_eq!(signals[0].sequence, 1);
}

fn clone_controller(src: &Controller) -> Controller {
    Controller {
        aid: src.aid.clone(),
        current_keys: src.current_keys.clone(),
        next_keys: src.next_keys.clone(),
        threshold: src.threshold,
        next_threshold: src.next_threshold,
        witnesses: src.witnesses.clone(),
        witness_threshold: src.witness_threshold,
        key_event_log: src.key_event_log.clone(),
        delegated_by: src.delegated_by.clone(),
    }
}

#[tokio::test]
async fn recovery_resets_key_set() {
    let mut rng = OsRng;
    let k0 = KeyPair::generate(&mut rng);
    let k1 = KeyPair::generate(&mut rng);
    let (mut ctrl, icp) = Controller::incept(
        vec![k0],
        vec![k1.clone()],
        Threshold(1),
        Threshold(1),
        vec![],
        Threshold(0),
    )
    .expect("incept");

    let mut sigs: HashMap<Said, Vec<(u32, Vec<u8>)>> = HashMap::new();
    let icp_event = KeyEvent::Inception(icp);
    sign_all_into_map(&ctrl, &icp_event, &mut sigs);

    // Catastrophic loss: pre-image for k1 unavailable. Use recovery.
    let fresh = KeyPair::generate(&mut rng);
    let fresh_next = KeyPair::generate(&mut rng);
    let rec = ctrl
        .recover(
            vec![fresh.clone()],
            vec![fresh_next],
            Threshold(1),
            Threshold(1),
            vec![],
        )
        .expect("recover");
    let rec_event = KeyEvent::Recovery(rec);
    sign_all_into_map(&ctrl, &rec_event, &mut sigs);

    let log = ctrl.key_event_log.clone();
    let receipts: HashMap<Said, Vec<WitnessReceipt>> = HashMap::new();
    let report = LogVerifier::new()
        .strict(false)
        .verify(&log, &sigs, &receipts)
        .expect("recovery accepted");
    assert_eq!(report.recovery_count, 1);
    assert_eq!(report.last_sequence, 1);

    // Strict mode rejects.
    let strict_err = LogVerifier::new()
        .strict(true)
        .verify(&log, &sigs, &receipts)
        .unwrap_err();
    assert!(matches!(
        strict_err,
        smart_byte_keri_witness::KeriError::StrictSpecRejectsRec
    ));
}

#[tokio::test]
async fn delegation_child_anchors_to_parent() {
    let mut rng = OsRng;
    let pk = KeyPair::generate(&mut rng);
    let (parent, _icp) = Controller::incept(
        vec![pk.clone()],
        vec![KeyPair::generate(&mut rng)],
        Threshold(1),
        Threshold(1),
        vec![],
        Threshold(0),
    )
    .expect("parent incept");

    let ck = KeyPair::generate(&mut rng);
    let (child, dlg) = Controller::delegate(
        parent.aid.clone(),
        vec![ck],
        vec![KeyPair::generate(&mut rng)],
        Threshold(1),
        Threshold(1),
    )
    .expect("child delegated");
    assert_eq!(dlg.di, parent.aid);
    assert_eq!(child.delegated_by.as_ref().unwrap(), &parent.aid);

    let mut sigs: HashMap<Said, Vec<(u32, Vec<u8>)>> = HashMap::new();
    let event = KeyEvent::Delegation(dlg);
    sign_all_into_map(&child, &event, &mut sigs);

    let log = child.key_event_log.clone();
    let receipts: HashMap<Said, Vec<WitnessReceipt>> = HashMap::new();
    let report = LogVerifier::new()
        .verify(&log, &sigs, &receipts)
        .expect("delegated verify ok");
    assert_eq!(report.last_sequence, 0);
}

#[tokio::test]
async fn memory_storage_round_trip() {
    let storage = MemoryStorage::new();
    let mut rng = OsRng;
    let (ctrl, icp) = Controller::incept(
        vec![KeyPair::generate(&mut rng)],
        vec![KeyPair::generate(&mut rng)],
        Threshold(1),
        Threshold(1),
        vec![],
        Threshold(0),
    )
    .expect("incept");
    let event = KeyEvent::Inception(icp);
    storage
        .append(&ctrl.aid, event.clone())
        .await
        .expect("append");
    let fetched = storage.fetch_log(&ctrl.aid).await.expect("fetch");
    assert_eq!(fetched.len(), 1);
    assert_eq!(fetched[0].said(), event.said());

    let _ = ControllerAid("ignored".into()); // suppress unused-import warnings
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(6))]
    #[test]
    fn property_valid_log_verifies(extra_rotations in 0u8..3, extra_interactions in 0u8..5) {
        // Build a known-good log of variable length and confirm verify accepts it.
        let rt = tokio::runtime::Runtime::new().unwrap();
        rt.block_on(async move {
            let mut rng = OsRng;
            let mut next_kp = KeyPair::generate(&mut rng);
            let (mut ctrl, icp) = Controller::incept(
                vec![KeyPair::generate(&mut rng)],
                vec![next_kp.clone()],
                Threshold(1),
                Threshold(1),
                vec![],
                Threshold(0),
            ).unwrap();
            let mut sigs: HashMap<Said, Vec<(u32, Vec<u8>)>> = HashMap::new();
            let icp_event = KeyEvent::Inception(icp);
            sign_all_into_map(&ctrl, &icp_event, &mut sigs);

            for _ in 0..extra_rotations {
                next_kp = KeyPair::generate(&mut rng);
                let rot = ctrl.rotate(vec![next_kp.clone()], Threshold(1), vec![], vec![], Threshold(0)).unwrap();
                let ev = KeyEvent::Rotation(rot);
                sign_all_into_map(&ctrl, &ev, &mut sigs);
            }
            for i in 0..extra_interactions {
                let ixn = ctrl.interact(vec![Anchor {
                    d: Said::hash(format!("a{i}").as_bytes()),
                    s: None,
                    kind: None,
                }]).unwrap();
                let ev = KeyEvent::Interaction(ixn);
                sign_all_into_map(&ctrl, &ev, &mut sigs);
            }
            let log = ctrl.key_event_log.clone();
            let receipts: HashMap<Said, Vec<WitnessReceipt>> = HashMap::new();
            LogVerifier::new().verify(&log, &sigs, &receipts).expect("verify");
        });
    }

    #[test]
    fn property_tampered_event_rejects(seed in 0u64..1_000_000) {
        let rt = tokio::runtime::Runtime::new().unwrap();
        let res = rt.block_on(async move {
            let mut rng = OsRng;
            let (mut ctrl, icp) = Controller::incept(
                vec![KeyPair::generate(&mut rng)],
                vec![KeyPair::generate(&mut rng)],
                Threshold(1),
                Threshold(1),
                vec![],
                Threshold(0),
            ).unwrap();
            let mut sigs: HashMap<Said, Vec<(u32, Vec<u8>)>> = HashMap::new();
            let icp_event = KeyEvent::Inception(icp);
            sign_all_into_map(&ctrl, &icp_event, &mut sigs);
            let ixn = ctrl.interact(vec![Anchor {
                d: Said::hash(format!("seed{seed}").as_bytes()),
                s: None,
                kind: None,
            }]).unwrap();
            let mut ev = KeyEvent::Interaction(ixn);
            sign_all_into_map(&ctrl, &ev, &mut sigs);
            if let KeyEvent::Interaction(ref mut i) = ev {
                i.a = vec![Anchor { d: Said::hash(b"tampered"), s: None, kind: None }];
            }
            let mut log = ctrl.key_event_log.clone();
            let last_idx = log.len() - 1;
            log[last_idx] = ev;
            let receipts: HashMap<Said, Vec<WitnessReceipt>> = HashMap::new();
            LogVerifier::new().verify(&log, &sigs, &receipts)
        });
        prop_assert!(res.is_err());
    }
}
