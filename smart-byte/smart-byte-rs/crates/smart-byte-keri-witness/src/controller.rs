//! In-memory controller state and event emission.
//!
//! A [`Controller`] holds the current and pre-rotated signing keys, the
//! growing key-event log, the witness set, and the signing/witness
//! thresholds. The four primary lifecycle operations are:
//!
//! * [`Controller::incept`] — bootstrap a new controller AID;
//! * [`Controller::rotate`] — reveal the pre-rotated keys and commit
//!   a fresh next set;
//! * [`Controller::interact`] — anchor off-log content without
//!   rotating;
//! * [`Controller::recover`] — catastrophic-key-loss reset.
//!
//! Signing keys are wrapped in [`KeyPair`], a thin newtype around
//! `ed25519_dalek::SigningKey` that also exposes the `smart_byte_pq`
//! signer trait so the substrate can transition to post-quantum or
//! hybrid keys without a controller-API break.

use std::collections::BTreeSet;

use ed25519_dalek::{Signer as EdSignerTrait, SigningKey, VerifyingKey};
use smart_byte_core::Said;

use crate::error::{KeriError, Result};
use crate::events::{
    Anchor, ControllerAid, DelegationEvent, EventType, InceptionEvent, InteractionEvent,
    KeyEvent, PublicKey, RecoveryEvent, RotationEvent, Threshold, WitnessAid, finalize_event,
    per_key_said,
};
use crate::VERSION_STRING;

/// Algorithm byte used for Ed25519 signing keys (mirrors
/// `smart_byte_pq::SignatureAlgorithm::Ed25519`).
const ED25519_ALG: u8 = 0x01;

/// A signing key plus its public counterpart.
///
/// The post-quantum migration story is handled by the wider
/// `smart_byte_pq` crate; this newtype intentionally exposes only
/// Ed25519 today and will grow algorithm variants when callers need
/// them for rotations.
#[derive(Clone)]
pub struct KeyPair {
    /// Ed25519 signing key.
    pub signing: SigningKey,
    /// Ed25519 verifying key.
    pub verifying: VerifyingKey,
}

impl KeyPair {
    /// Wrap an existing signing key.
    #[must_use]
    pub fn from_signing(signing: SigningKey) -> Self {
        let verifying = signing.verifying_key();
        Self { signing, verifying }
    }

    /// Generate a fresh Ed25519 key pair from the given RNG.
    pub fn generate<R: rand::CryptoRng + rand::RngCore>(rng: &mut R) -> Self {
        let signing = SigningKey::generate(rng);
        Self::from_signing(signing)
    }

    /// Algorithm-byte-prefixed public-key encoding used in events.
    #[must_use]
    pub fn public_key(&self) -> PublicKey {
        PublicKey::new(ED25519_ALG, self.verifying.as_bytes())
    }

    /// Sign an arbitrary byte slice.
    pub fn sign(&self, message: &[u8]) -> ed25519_dalek::Signature {
        self.signing.sign(message)
    }
}

impl core::fmt::Debug for KeyPair {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("KeyPair")
            .field("signing", &"<redacted>")
            .field("verifying", &self.verifying)
            .finish()
    }
}

/// Controller state machine.
#[derive(Debug)]
pub struct Controller {
    /// Controller AID (set at inception).
    pub aid: ControllerAid,
    /// Current signing keys (revealed by the most recent establishment event).
    pub current_keys: Vec<KeyPair>,
    /// Pre-rotated next keys (committed by digest in the most recent event's `n`).
    pub next_keys: Vec<KeyPair>,
    /// Current signing-key threshold.
    pub threshold: Threshold,
    /// Next-rotation threshold.
    pub next_threshold: Threshold,
    /// Witness set in force.
    pub witnesses: BTreeSet<WitnessAid>,
    /// Witness-receipt threshold.
    pub witness_threshold: Threshold,
    /// Growing key-event log (newest at the tail).
    pub key_event_log: Vec<KeyEvent>,
    /// Optional delegating parent (set for delegated controllers only).
    pub delegated_by: Option<ControllerAid>,
}

impl Controller {
    /// Bootstrap a new controller.
    ///
    /// The returned [`Controller`] holds the inception event in its
    /// log; the caller may immediately call
    /// [`Controller::sign_event`] to obtain the signatures and ship
    /// the event to its witnesses.
    pub fn incept(
        initial_keys: Vec<KeyPair>,
        next_keys: Vec<KeyPair>,
        threshold: Threshold,
        next_threshold: Threshold,
        witnesses: Vec<WitnessAid>,
        witness_threshold: Threshold,
    ) -> Result<(Self, InceptionEvent)> {
        if initial_keys.is_empty() {
            return Err(KeriError::Malformed("inception requires >= 1 key".into()));
        }
        if (threshold.0 as usize) > initial_keys.len() {
            return Err(KeriError::Malformed(
                "threshold exceeds initial key count".into(),
            ));
        }
        let k: Vec<PublicKey> = initial_keys.iter().map(KeyPair::public_key).collect();
        let n: Vec<Said> = next_keys
            .iter()
            .map(|kp| per_key_said(&kp.public_key()))
            .collect::<Result<_>>()?;
        let witness_set: BTreeSet<WitnessAid> = witnesses.iter().cloned().collect();
        let placeholder_aid = ControllerAid(String::new());
        let placeholder_said = Said::default();

        let raw = InceptionEvent {
            v: VERSION_STRING.to_string(),
            t: EventType::Inception,
            d: placeholder_said,
            i: placeholder_aid,
            s: 0,
            kt: threshold,
            k,
            nt: next_threshold,
            n,
            c: Vec::new(),
            b: witnesses.clone(),
            bt: witness_threshold,
            a: Vec::new(),
        };
        // First pass: SAID with d zeroed and i empty
        let bytes = serde_cbor::to_vec(&KeyEvent::Inception(raw.clone()))?;
        let said = Said::hash(&bytes);
        let aid = ControllerAid::from_inception_said(&said);
        let mut finalized = raw;
        finalized.d = said;
        finalized.i = aid.clone();

        let controller = Self {
            aid,
            current_keys: initial_keys,
            next_keys,
            threshold,
            next_threshold,
            witnesses: witness_set,
            witness_threshold,
            key_event_log: vec![KeyEvent::Inception(finalized.clone())],
            delegated_by: None,
        };
        Ok((controller, finalized))
    }

    /// Produce a [`RotationEvent`] revealing the pre-committed next
    /// keys and committing a fresh pre-rotation set.
    pub fn rotate(
        &mut self,
        new_next_keys: Vec<KeyPair>,
        new_next_threshold: Threshold,
        add_witnesses: Vec<WitnessAid>,
        remove_witnesses: Vec<WitnessAid>,
        new_witness_threshold: Threshold,
    ) -> Result<RotationEvent> {
        let prior = self
            .key_event_log
            .last()
            .ok_or_else(|| KeriError::Malformed("cannot rotate empty log".into()))?
            .clone();
        let prior_said = prior.said();
        let next_sequence = prior.sequence() + 1;

        let revealed: Vec<PublicKey> = self.next_keys.iter().map(KeyPair::public_key).collect();
        let new_n: Vec<Said> = new_next_keys
            .iter()
            .map(|kp| per_key_said(&kp.public_key()))
            .collect::<Result<_>>()?;

        // Update witness set per br/ba.
        for w in &remove_witnesses {
            self.witnesses.remove(w);
        }
        for w in &add_witnesses {
            self.witnesses.insert(w.clone());
        }
        self.witness_threshold = new_witness_threshold;

        let event = RotationEvent {
            v: VERSION_STRING.to_string(),
            t: EventType::Rotation,
            d: Said::default(),
            i: self.aid.clone(),
            s: next_sequence,
            p: prior_said,
            kt: self.next_threshold,
            k: revealed,
            nt: new_next_threshold,
            n: new_n,
            br: remove_witnesses,
            ba: add_witnesses,
            bt: new_witness_threshold,
            a: Vec::new(),
        };
        let finalized = finalize_event(KeyEvent::Rotation(event))?;
        // After commit: previous next_keys are now current; new next set is committed.
        self.current_keys = std::mem::take(&mut self.next_keys);
        self.next_keys = new_next_keys;
        self.threshold = self.next_threshold;
        self.next_threshold = new_next_threshold;
        self.key_event_log.push(finalized.clone());
        if let KeyEvent::Rotation(rot) = finalized {
            Ok(rot)
        } else {
            Err(KeriError::Malformed("finalize lost event type".into()))
        }
    }

    /// Produce an [`InteractionEvent`] anchoring off-log content.
    pub fn interact(&mut self, anchors: Vec<Anchor>) -> Result<InteractionEvent> {
        let prior = self
            .key_event_log
            .last()
            .ok_or_else(|| KeriError::Malformed("cannot interact on empty log".into()))?
            .clone();
        let event = InteractionEvent {
            v: VERSION_STRING.to_string(),
            t: EventType::Interaction,
            d: Said::default(),
            i: self.aid.clone(),
            s: prior.sequence() + 1,
            p: prior.said(),
            a: anchors,
        };
        let finalized = finalize_event(KeyEvent::Interaction(event))?;
        self.key_event_log.push(finalized.clone());
        if let KeyEvent::Interaction(ixn) = finalized {
            Ok(ixn)
        } else {
            Err(KeriError::Malformed("finalize lost event type".into()))
        }
    }

    /// Produce a [`RecoveryEvent`] resetting the signing key set after
    /// catastrophic key loss. The recovery operation does *not* require
    /// the previous `n` commitment to be honoured — that is the entire
    /// point of recovery. Verifiers in non-strict mode accept the
    /// reset; strict-spec verifiers reject. See
    /// [`crate::verifier::LogVerifier::strict`].
    pub fn recover(
        &mut self,
        new_keys: Vec<KeyPair>,
        new_next_keys: Vec<KeyPair>,
        new_threshold: Threshold,
        new_next_threshold: Threshold,
        anchors: Vec<Anchor>,
    ) -> Result<RecoveryEvent> {
        let prior = self
            .key_event_log
            .last()
            .ok_or_else(|| KeriError::Malformed("cannot recover on empty log".into()))?
            .clone();
        let event = RecoveryEvent {
            v: VERSION_STRING.to_string(),
            t: EventType::Recovery,
            d: Said::default(),
            i: self.aid.clone(),
            s: prior.sequence() + 1,
            p: prior.said(),
            kt: new_threshold,
            k: new_keys.iter().map(KeyPair::public_key).collect(),
            nt: new_next_threshold,
            n: new_next_keys
                .iter()
                .map(|kp| per_key_said(&kp.public_key()))
                .collect::<Result<_>>()?,
            a: anchors,
        };
        let finalized = finalize_event(KeyEvent::Recovery(event))?;
        self.current_keys = new_keys;
        self.next_keys = new_next_keys;
        self.threshold = new_threshold;
        self.next_threshold = new_next_threshold;
        self.key_event_log.push(finalized.clone());
        if let KeyEvent::Recovery(rec) = finalized {
            Ok(rec)
        } else {
            Err(KeriError::Malformed("finalize lost event type".into()))
        }
    }

    /// Bootstrap a delegated child controller. The child's inception
    /// references the parent via the `di` field and is anchored back
    /// to the parent's log via a caller-issued interaction event on
    /// the parent.
    pub fn delegate(
        parent_aid: ControllerAid,
        initial_keys: Vec<KeyPair>,
        next_keys: Vec<KeyPair>,
        threshold: Threshold,
        next_threshold: Threshold,
    ) -> Result<(Self, DelegationEvent)> {
        if initial_keys.is_empty() {
            return Err(KeriError::Malformed("delegation requires >= 1 key".into()));
        }
        let k: Vec<PublicKey> = initial_keys.iter().map(KeyPair::public_key).collect();
        let n: Vec<Said> = next_keys
            .iter()
            .map(|kp| per_key_said(&kp.public_key()))
            .collect::<Result<_>>()?;
        let raw = DelegationEvent {
            v: VERSION_STRING.to_string(),
            t: EventType::Delegation,
            d: Said::default(),
            i: ControllerAid(String::new()),
            s: 0,
            p: Said::default(),
            di: parent_aid.clone(),
            kt: threshold,
            k,
            nt: next_threshold,
            n,
        };
        let bytes = serde_cbor::to_vec(&KeyEvent::Delegation(raw.clone()))?;
        let said = Said::hash(&bytes);
        let aid = ControllerAid::from_inception_said(&said);
        let mut finalized = raw;
        finalized.d = said;
        finalized.i = aid.clone();
        let controller = Self {
            aid,
            current_keys: initial_keys,
            next_keys,
            threshold,
            next_threshold,
            witnesses: BTreeSet::new(),
            witness_threshold: Threshold(0),
            key_event_log: vec![KeyEvent::Delegation(finalized.clone())],
            delegated_by: Some(parent_aid),
        };
        Ok((controller, finalized))
    }

    /// Sign an event with the current signing key set, returning a
    /// vector of `(key_index, signature)` tuples sized to the
    /// threshold. For an inception event the keys revealed *in* the
    /// event itself are the signers; the caller passes those in via
    /// `self.current_keys`.
    pub fn sign_event(&self, event: &KeyEvent) -> Result<Vec<(u32, ed25519_dalek::Signature)>> {
        let body = event.to_cbor()?;
        let mut out = Vec::with_capacity(self.current_keys.len());
        for (i, kp) in self.current_keys.iter().enumerate() {
            let sig = kp.sign(&body);
            out.push((i as u32, sig));
        }
        Ok(out)
    }
}
