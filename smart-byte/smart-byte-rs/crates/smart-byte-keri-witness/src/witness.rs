//! Witness layer: peers who observe and counter-sign each event so a
//! controller cannot rewrite their own history.
//!
//! A witness:
//!
//! 1. Receives an event plus the controller's threshold-met signatures.
//! 2. Verifies the controller signatures against the keys in force.
//! 3. If valid, records the event and emits a [`WitnessReceipt`] over
//!    the event's SAID.
//! 4. Refuses to issue a contradicting receipt for the same
//!    `(controller, sequence)` — duplicity defence.
//!
//! Witness signatures are themselves Ed25519. A future revision may
//! upgrade witnesses to hybrid (Ed25519 + ML-DSA) via the same
//! `smart_byte_pq::Signer` plumbing used elsewhere in the substrate.

use async_trait::async_trait;
use chrono::{DateTime, Utc};
use dashmap::DashMap;
use ed25519_dalek::{Signature, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use smart_byte_core::Said;

use crate::controller::KeyPair;
use crate::error::{KeriError, Result};
use crate::events::{ControllerAid, KeyEvent, PublicKey, Threshold, WitnessAid};

/// A receipt signed by a witness attesting that they observed the
/// given event at the given sequence.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct WitnessReceipt {
    /// Witness AID that emitted this receipt.
    pub witness: WitnessAid,
    /// Controller AID the receipted event belongs to.
    pub controller: ControllerAid,
    /// Sequence number of the receipted event.
    pub sequence: u64,
    /// SAID of the receipted event body.
    pub signed_event_said: Said,
    /// Wall-clock timestamp at which the receipt was emitted.
    pub timestamp: DateTime<Utc>,
    /// Witness's Ed25519 verifying key (so verifiers can validate the receipt).
    #[serde(with = "serde_bytes")]
    pub witness_pubkey: Vec<u8>,
    /// Witness's Ed25519 signature over the SAID bytes.
    #[serde(with = "serde_bytes")]
    pub signature: Vec<u8>,
}

impl WitnessReceipt {
    /// Cryptographically verify the witness's signature against the
    /// receipted event SAID.
    pub fn verify_signature(&self) -> Result<()> {
        let mut pk_arr = [0u8; 32];
        if self.witness_pubkey.len() != 32 {
            return Err(KeriError::MalformedKey(format!(
                "witness pubkey must be 32 bytes, got {}",
                self.witness_pubkey.len()
            )));
        }
        pk_arr.copy_from_slice(&self.witness_pubkey);
        let vk = VerifyingKey::from_bytes(&pk_arr)
            .map_err(|e| KeriError::MalformedKey(e.to_string()))?;
        if self.signature.len() != 64 {
            return Err(KeriError::MalformedKey(format!(
                "witness signature must be 64 bytes, got {}",
                self.signature.len()
            )));
        }
        let mut sig_arr = [0u8; 64];
        sig_arr.copy_from_slice(&self.signature);
        let sig = Signature::from_bytes(&sig_arr);
        vk.verify(self.signed_event_said.as_bytes(), &sig)
            .map_err(|_| KeriError::BadSignature)
    }
}

/// A witness peer.
pub struct Witness {
    /// Witness AID — typically the base32 of the witness pubkey
    /// prefixed with `W`, but treated opaquely by this crate.
    pub aid: WitnessAid,
    /// Witness signing key.
    pub key: KeyPair,
    /// Observed event logs, keyed by controller AID. Each entry is
    /// the in-order sequence of events the witness has receipted.
    pub observed_logs: DashMap<ControllerAid, Vec<KeyEvent>>,
}

impl Witness {
    /// Construct a fresh witness wrapping the given AID and Ed25519
    /// key pair.
    #[must_use]
    pub fn new(aid: WitnessAid, key: KeyPair) -> Self {
        Self {
            aid,
            key,
            observed_logs: DashMap::new(),
        }
    }

    /// Receive an event and emit a [`WitnessReceipt`] if the
    /// controller signatures meet the threshold and no contradicting
    /// receipt has previously been issued.
    ///
    /// `controller_keys_in_force` is the signing key set in force at
    /// the time the event was authored (the inception event's own
    /// `k` for inception, or the most recent establishment event's
    /// `k` for everything else). `threshold` is the matching `kt`.
    ///
    /// Witness signatures are NOT computed over an algorithm byte;
    /// they are computed over the raw 32-byte SAID, mirroring the
    /// behaviour of `smart_byte_core::sign`.
    pub async fn receive(
        &self,
        event: KeyEvent,
        controller_signatures: &[(u32, Signature)],
        controller_keys_in_force: &[PublicKey],
        threshold: Threshold,
    ) -> Result<WitnessReceipt> {
        // 1. SAID consistency.
        event.validate_said()?;

        // 2. Controller-signature threshold.
        let body = event.to_cbor()?;
        let mut valid_distinct = std::collections::BTreeSet::new();
        for (idx, sig) in controller_signatures {
            let pk = controller_keys_in_force.get(*idx as usize).ok_or_else(|| {
                KeriError::Malformed(format!("signature references unknown key index {idx}"))
            })?;
            // v1 only validates Ed25519-prefix keys at the witness layer;
            // PQ keys would be routed through smart_byte_pq here.
            if pk.algorithm() != Some(0x01) {
                continue;
            }
            let pk_bytes = pk.key_bytes();
            if pk_bytes.len() != 32 {
                continue;
            }
            let mut pk_arr = [0u8; 32];
            pk_arr.copy_from_slice(pk_bytes);
            let vk = VerifyingKey::from_bytes(&pk_arr)
                .map_err(|e| KeriError::MalformedKey(e.to_string()))?;
            if vk.verify(&body, sig).is_ok() {
                valid_distinct.insert(*idx);
            }
        }
        if (valid_distinct.len() as u32) < threshold.0 {
            return Err(KeriError::ThresholdNotMet {
                have: valid_distinct.len() as u32,
                need: threshold.0,
            });
        }

        // 3. Duplicity defence — refuse to receipt a different event at
        //    the same (controller, sequence).
        let controller = event.controller().clone();
        let sequence = event.sequence();
        let said = event.said();
        if let Some(existing) = self.observed_logs.get(&controller)
            && let Some(prev) = existing.iter().find(|e| e.sequence() == sequence)
            && prev.said() != said
        {
            return Err(KeriError::DuplicityRefused { sequence });
        }

        // 4. Record + sign.
        self.observed_logs.entry(controller.clone()).or_default().push(event);

        let sig = self.key.sign(said.as_bytes());
        Ok(WitnessReceipt {
            witness: self.aid.clone(),
            controller,
            sequence,
            signed_event_said: said,
            timestamp: Utc::now(),
            witness_pubkey: self.key.verifying.as_bytes().to_vec(),
            signature: sig.to_bytes().to_vec(),
        })
    }
}

/// Abstract witness-network interface so the substrate can swap
/// in-memory witnesses for HTTP/Iroh-backed ones without changing the
/// callers.
#[async_trait]
pub trait WitnessClient: Send + Sync {
    /// Push an event to the witness and return its receipt.
    async fn push(
        &self,
        event: KeyEvent,
        controller_signatures: &[(u32, Signature)],
        controller_keys_in_force: &[PublicKey],
        threshold: Threshold,
    ) -> Result<WitnessReceipt>;
}

/// In-process witness client that wraps a [`Witness`] directly.
pub struct InProcessWitnessClient {
    /// Underlying witness.
    pub witness: std::sync::Arc<Witness>,
}

#[async_trait]
impl WitnessClient for InProcessWitnessClient {
    async fn push(
        &self,
        event: KeyEvent,
        controller_signatures: &[(u32, Signature)],
        controller_keys_in_force: &[PublicKey],
        threshold: Threshold,
    ) -> Result<WitnessReceipt> {
        self.witness
            .receive(event, controller_signatures, controller_keys_in_force, threshold)
            .await
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[tokio::test]
    async fn witness_signs_well_formed_event() {
        let mut rng = OsRng;
        let kp = KeyPair::generate(&mut rng);
        let w_aid = WitnessAid("W1".into());
        let w = Witness::new(w_aid.clone(), KeyPair::generate(&mut rng));

        let (mut ctrl, icp) = crate::controller::Controller::incept(
            vec![kp.clone()],
            vec![KeyPair::generate(&mut rng)],
            Threshold(1),
            Threshold(1),
            vec![w_aid.clone()],
            Threshold(1),
        )
        .expect("incept");

        let event = KeyEvent::Inception(icp.clone());
        let sigs = ctrl.sign_event(&event).expect("sign");
        let keys_in_force = icp.k.clone();
        let receipt = w
            .receive(event, &sigs, &keys_in_force, Threshold(1))
            .await
            .expect("receipt");
        receipt.verify_signature().expect("receipt verify");
        let _ = &mut ctrl; // silence unused-mut
    }
}
