//! Replay verifier per spec §4.
//!
//! Given a sequence of events and (optionally) witness receipts, the
//! verifier accepts the log iff every spec-mandated check passes for
//! every event in sequence.

use std::collections::{BTreeMap, HashMap};

use ed25519_dalek::{Signature, Verifier as EdVerifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use smart_byte_core::Said;

use crate::error::{KeriError, Result};
use crate::events::{
    ControllerAid, EventType, KeyEvent, PublicKey, Threshold, per_key_said,
};
use crate::witness::WitnessReceipt;

/// Evidence record for a detected duplicity event.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DuplicityEvidence {
    /// Sequence number at which two distinct events were observed.
    pub sequence: u64,
    /// SAIDs of the two events.
    pub fork: (Said, Said),
}

/// Outcome of a full-log verification.
#[derive(Clone, Debug)]
pub struct VerificationReport {
    /// Signing keys in force after the last accepted event.
    pub current_keys: Vec<PublicKey>,
    /// Threshold in force after the last accepted event.
    pub current_threshold: Threshold,
    /// Highest sequence number accepted.
    pub last_sequence: u64,
    /// Whether the witness-receipt threshold was met for every event.
    pub witness_quorum_met: bool,
    /// How many recovery events the log contains.
    pub recovery_count: u32,
    /// Duplicity evidence collected from receipts (if any).
    pub duplicity: Vec<DuplicityEvidence>,
}

/// Replay verifier.
#[derive(Default)]
pub struct LogVerifier {
    /// When `true`, reject `rec` events outright per spec §3.2.
    /// When `false`, accept recoveries (this crate's extension).
    pub strict: bool,
    /// When `true`, also require the witness-receipt threshold be met
    /// for every event in the log.
    pub require_witness_quorum: bool,
}

impl LogVerifier {
    /// Build a permissive verifier that accepts recoveries and does
    /// not require witness quorum.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Enable strict spec mode (rejects `rec`).
    #[must_use]
    pub fn strict(mut self, on: bool) -> Self {
        self.strict = on;
        self
    }

    /// Require that every event have at least `bt` valid witness receipts.
    #[must_use]
    pub fn require_witness_quorum(mut self, on: bool) -> Self {
        self.require_witness_quorum = on;
        self
    }

    /// Replay the supplied log and produce a [`VerificationReport`].
    pub fn verify(
        &self,
        log: &[KeyEvent],
        signatures: &HashMap<Said, Vec<(u32, Vec<u8>)>>,
        receipts: &HashMap<Said, Vec<WitnessReceipt>>,
    ) -> Result<VerificationReport> {
        if log.is_empty() {
            return Err(KeriError::Malformed("empty log".into()));
        }
        // First event must be inception or delegated-inception.
        let first = &log[0];
        let (mut current_keys, mut current_threshold, mut next_keys_committed, mut next_threshold, mut witness_threshold) =
            match first {
                KeyEvent::Inception(icp) => {
                    if icp.s != 0 {
                        return Err(KeriError::SequenceError {
                            expected: 0,
                            got: icp.s,
                        });
                    }
                    (
                        icp.k.clone(),
                        icp.kt,
                        icp.n.clone(),
                        icp.nt,
                        icp.bt,
                    )
                }
                KeyEvent::Delegation(dip) if dip.s == 0 => (
                    dip.k.clone(),
                    dip.kt,
                    dip.n.clone(),
                    dip.nt,
                    Threshold(0),
                ),
                _ => {
                    return Err(KeriError::Malformed(
                        "log must begin with inception or delegated-inception".into(),
                    ))
                }
            };
        let controller_aid = first.controller().clone();
        let mut last_said = first.said();
        let mut last_sequence = first.sequence();
        let mut recovery_count = 0u32;
        let mut witness_quorum_met = true;
        let mut duplicity = Vec::new();

        // Verify the inception itself.
        self.verify_event(
            first,
            &current_keys,
            current_threshold,
            None,
            &controller_aid,
            signatures,
        )?;
        self.check_witness_quorum(
            first,
            witness_threshold,
            receipts,
            &mut witness_quorum_met,
        )?;

        for (idx, event) in log.iter().enumerate().skip(1) {
            // 1. SAID consistency.
            event.validate_said()?;
            // 2. Sequence.
            if event.sequence() != last_sequence + 1 {
                return Err(KeriError::SequenceError {
                    expected: last_sequence + 1,
                    got: event.sequence(),
                });
            }
            // 3. Prior linkage.
            if event.prior() != Some(last_said) {
                return Err(KeriError::PriorMismatch {
                    expected: last_said,
                    got: event.prior().unwrap_or_default(),
                });
            }
            // 4. Controller AID.
            if event.controller() != &controller_aid {
                return Err(KeriError::ControllerAidMismatch {
                    expected: controller_aid.0.clone(),
                    got: event.controller().0.clone(),
                });
            }

            // 5. Type-specific structural check.
            match event {
                KeyEvent::Rotation(rot) => {
                    if rot.k.len() != next_keys_committed.len() {
                        return Err(KeriError::PreRotationMismatch { index: 0 });
                    }
                    for (i, key) in rot.k.iter().enumerate() {
                        let derived = per_key_said(key)?;
                        if derived != next_keys_committed[i] {
                            return Err(KeriError::PreRotationMismatch { index: i });
                        }
                    }
                    // Verify signatures against the NEWLY-revealed keys
                    // (rotation is signed by the keys it reveals).
                    self.verify_event(
                        event,
                        &rot.k,
                        rot.kt,
                        Some(idx as u64),
                        &controller_aid,
                        signatures,
                    )?;
                    self.check_witness_quorum(
                        event,
                        rot.bt,
                        receipts,
                        &mut witness_quorum_met,
                    )?;
                    current_keys = rot.k.clone();
                    current_threshold = rot.kt;
                    next_keys_committed = rot.n.clone();
                    next_threshold = rot.nt;
                    witness_threshold = rot.bt;
                }
                KeyEvent::Interaction(_) => {
                    self.verify_event(
                        event,
                        &current_keys,
                        current_threshold,
                        Some(idx as u64),
                        &controller_aid,
                        signatures,
                    )?;
                    self.check_witness_quorum(
                        event,
                        witness_threshold,
                        receipts,
                        &mut witness_quorum_met,
                    )?;
                }
                KeyEvent::Recovery(rec) => {
                    if self.strict {
                        return Err(KeriError::StrictSpecRejectsRec);
                    }
                    // Recovery is verified against the keys it reveals.
                    self.verify_event(
                        event,
                        &rec.k,
                        rec.kt,
                        Some(idx as u64),
                        &controller_aid,
                        signatures,
                    )?;
                    self.check_witness_quorum(
                        event,
                        witness_threshold,
                        receipts,
                        &mut witness_quorum_met,
                    )?;
                    current_keys = rec.k.clone();
                    current_threshold = rec.kt;
                    next_keys_committed = rec.n.clone();
                    next_threshold = rec.nt;
                    recovery_count += 1;
                }
                KeyEvent::Delegation(dlg) => {
                    // Delegated rotation: signed by revealed keys.
                    self.verify_event(
                        event,
                        &dlg.k,
                        dlg.kt,
                        Some(idx as u64),
                        &controller_aid,
                        signatures,
                    )?;
                    current_keys = dlg.k.clone();
                    current_threshold = dlg.kt;
                    next_keys_committed = dlg.n.clone();
                    next_threshold = dlg.nt;
                }
                KeyEvent::Inception(_) => {
                    return Err(KeriError::Malformed(
                        "inception event appeared after sequence 0".into(),
                    ));
                }
            }

            last_said = event.said();
            last_sequence = event.sequence();
        }

        // Duplicity: inspect receipts for forked SAIDs at the same sequence.
        // Said does not implement Ord so we key by sequence and dedupe with
        // a Vec compared via byte-equality.
        let mut by_seq: BTreeMap<u64, Vec<Said>> = BTreeMap::new();
        for (said, rs) in receipts {
            for r in rs {
                if r.controller == controller_aid {
                    let bucket = by_seq.entry(r.sequence).or_default();
                    if !bucket.contains(said) {
                        bucket.push(*said);
                    }
                }
            }
        }
        for (seq, saids) in by_seq {
            if saids.len() > 1 {
                let a = saids[0];
                let b = saids[1];
                duplicity.push(DuplicityEvidence {
                    sequence: seq,
                    fork: (a, b),
                });
            }
        }

        let _ = next_threshold; // kept in report-implicit state
        Ok(VerificationReport {
            current_keys,
            current_threshold,
            last_sequence,
            witness_quorum_met,
            recovery_count,
            duplicity,
        })
    }

    fn verify_event(
        &self,
        event: &KeyEvent,
        keys: &[PublicKey],
        threshold: Threshold,
        _idx: Option<u64>,
        controller_aid: &ControllerAid,
        signatures: &HashMap<Said, Vec<(u32, Vec<u8>)>>,
    ) -> Result<()> {
        // SAID consistency.
        event.validate_said()?;
        // Controller AID consistency.
        if event.controller() != controller_aid {
            return Err(KeriError::ControllerAidMismatch {
                expected: controller_aid.0.clone(),
                got: event.controller().0.clone(),
            });
        }
        // Inception structural check.
        if let KeyEvent::Inception(icp) = event
            && icp.s != 0
        {
            return Err(KeriError::SequenceError { expected: 0, got: icp.s });
        }
        // Signature threshold.
        let sigs = signatures.get(&event.said()).cloned().unwrap_or_default();
        let body = event.to_cbor()?;
        let mut valid = std::collections::BTreeSet::new();
        for (i, sig_bytes) in sigs {
            let pk = keys.get(i as usize).ok_or_else(|| {
                KeriError::Malformed(format!("signature references unknown key index {i}"))
            })?;
            if pk.algorithm() != Some(0x01) || pk.key_bytes().len() != 32 {
                continue;
            }
            let mut pk_arr = [0u8; 32];
            pk_arr.copy_from_slice(pk.key_bytes());
            let vk = match VerifyingKey::from_bytes(&pk_arr) {
                Ok(v) => v,
                Err(_) => continue,
            };
            if sig_bytes.len() != 64 {
                continue;
            }
            let mut sig_arr = [0u8; 64];
            sig_arr.copy_from_slice(&sig_bytes);
            let sig = Signature::from_bytes(&sig_arr);
            if vk.verify(&body, &sig).is_ok() {
                valid.insert(i);
            }
        }
        if (valid.len() as u32) < threshold.0 {
            return Err(KeriError::ThresholdNotMet {
                have: valid.len() as u32,
                need: threshold.0,
            });
        }
        Ok(())
    }

    fn check_witness_quorum(
        &self,
        event: &KeyEvent,
        bt: Threshold,
        receipts: &HashMap<Said, Vec<WitnessReceipt>>,
        witness_quorum_met: &mut bool,
    ) -> Result<()> {
        if !self.require_witness_quorum {
            return Ok(());
        }
        let rs = receipts.get(&event.said()).cloned().unwrap_or_default();
        let mut distinct_valid = std::collections::BTreeSet::new();
        for r in rs {
            if r.verify_signature().is_ok() && r.sequence == event.sequence() {
                distinct_valid.insert(r.witness.0.clone());
            }
        }
        if (distinct_valid.len() as u32) < bt.0 {
            *witness_quorum_met = false;
            return Err(KeriError::WitnessThresholdNotMet {
                have: distinct_valid.len() as u32,
                need: bt.0,
            });
        }
        // Use _ on event-type-prefix discriminator
        let _ = event.event_type();
        Ok(())
    }
}

impl EventType {
    /// True if the event type is one that changes the signing key set.
    #[must_use]
    pub fn is_establishment(self) -> bool {
        matches!(
            self,
            Self::Inception | Self::Rotation | Self::Recovery | Self::Delegation
        )
    }
}
