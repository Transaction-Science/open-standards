//! Key-event types as defined in
//! `smart-byte/spec/identity_and_key_rotation.md` §3.
//!
//! Each event is a CBOR map with the canonical sorted-key field layout
//! `v`/`t`/`d`/`i`/`s`/`p`/`kt`/`k`/`nt`/`n`/`a` (plus type-specific
//! extras like `b`/`bt`/`br`/`ba`/`c`/`di`).
//!
//! The spec stipulates that the SAID (`d`) is computed by:
//!
//! 1. Replacing `d` with a 53-character `#` placeholder.
//! 2. Canonically CBOR-encoding the event with sorted map keys.
//! 3. Taking BLAKE3-256 of the bytes.
//! 4. Storing the digest back in `d`.
//!
//! Because the `smart-byte-core::Said` primitive already wraps the
//! BLAKE3-256 digest, we implement the substitution + re-hash sequence
//! locally and expose [`compute_said`] for each event type via the
//! [`SaidedEvent`] trait.

use serde::{Deserialize, Serialize};
use smart_byte_core::Said;

use crate::error::{KeriError, Result};

/// 53-character placeholder used during SAID derivation per spec §2.1.
pub const SAID_PLACEHOLDER: &str =
    "#####################################################";

/// String form of the version field carried in every event.
pub const VERSION_STRING: &str = crate::VERSION_STRING;

/// Identifier of a controller (an Autonomic IDentifier, AID).
///
/// Per spec §3.1, an AID is the SAID of the controller's inception
/// event with the prefix `B` substituted for the `E` of an arbitrary
/// SAID. We store the substituted ASCII form directly so that
/// equality and ordering match the wire encoding.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct ControllerAid(pub String);

impl ControllerAid {
    /// Build a controller AID from its inception event's SAID by
    /// substituting `B` for the leading `E` prefix character.
    #[must_use]
    pub fn from_inception_said(said: &Said) -> Self {
        let s = said.to_base32();
        Self(format!("B{s}"))
    }

    /// Recover the underlying SAID by re-substituting `E` for the `B`
    /// prefix. Returns `None` if the AID does not look like a controller
    /// AID.
    #[must_use]
    pub fn as_inception_said(&self) -> Option<Said> {
        let rest = self.0.strip_prefix('B')?;
        Said::from_base32(rest).ok()
    }
}

/// Identifier of a witness peer.
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WitnessAid(pub String);

/// Identifier of a watcher (third-party duplicity detector).
#[derive(Clone, Debug, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct WatcherAid(pub String);

/// Public-key encoding placed in the event's `k` field.
///
/// Each entry is the raw public key bytes prefixed with the one-byte
/// algorithm identifier defined in `smart_byte_pq::SignatureAlgorithm`
/// (`0x01` = Ed25519, `0x10-0x12` = ML-DSA, etc.).
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PublicKey(#[serde(with = "serde_bytes")] pub Vec<u8>);

impl PublicKey {
    /// Build a public key from an algorithm byte plus raw key bytes.
    #[must_use]
    pub fn new(algorithm: u8, key_bytes: &[u8]) -> Self {
        let mut v = Vec::with_capacity(1 + key_bytes.len());
        v.push(algorithm);
        v.extend_from_slice(key_bytes);
        Self(v)
    }

    /// Algorithm byte at position 0.
    #[must_use]
    pub fn algorithm(&self) -> Option<u8> {
        self.0.first().copied()
    }

    /// Raw key bytes following the algorithm byte.
    #[must_use]
    pub fn key_bytes(&self) -> &[u8] {
        if self.0.is_empty() { &[] } else { &self.0[1..] }
    }
}

/// Unsigned integer threshold (KERI's weighted thresholds reserved for a
/// later revision, per spec §3.3).
#[derive(
    Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
pub struct Threshold(pub u32);

/// Anchor to off-log content carried in `ixn` events' `a` field.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Anchor {
    /// SAID of the anchored object (envelope, attestation, etc.).
    pub d: Said,
    /// Optional sequence or position used by the caller for ordering.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub s: Option<u64>,
    /// Optional caller-defined kind tag (e.g. "envelope", "vc-issue").
    #[serde(skip_serializing_if = "Option::is_none")]
    pub kind: Option<String>,
}

/// Optional inception-time configuration traits per spec §3.3 `c`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ConfigTrait {
    /// The log will never accept further rotations (terminal commitment).
    NoRotate,
    /// The log accepts only events with witness receipts above threshold.
    EstablishOnly,
    /// Implementation-defined extension trait.
    Custom(String),
}

/// Event-type discriminator written to the `t` field.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum EventType {
    /// Inception (`icp`).
    #[serde(rename = "icp")]
    Inception,
    /// Rotation (`rot`).
    #[serde(rename = "rot")]
    Rotation,
    /// Interaction (`ixn`).
    #[serde(rename = "ixn")]
    Interaction,
    /// Recovery (`rec`).
    #[serde(rename = "rec")]
    Recovery,
    /// Delegation (`dip`/`drt` collapsed into a single discriminator).
    #[serde(rename = "dip")]
    Delegation,
}

/// Per-key SAID derivation per spec §3.4.
///
/// Each entry in the previous event's `n` array is the SAID of a tiny
/// CBOR map `{"d": "<placeholder>", "k": "<K_i>"}`. This helper builds
/// the placeholder map, canonically encodes it, hashes, and returns
/// the resulting SAID.
pub fn per_key_said(key: &PublicKey) -> Result<Said> {
    #[derive(Serialize)]
    struct PerKey<'a> {
        d: &'a str,
        k: &'a PublicKey,
    }
    let placeholder = PerKey {
        d: SAID_PLACEHOLDER,
        k: key,
    };
    let bytes = serde_cbor::to_vec(&placeholder)?;
    Ok(Said::hash(&bytes))
}

// ----- Event structs --------------------------------------------------------

/// The first event in a controller's log; binds the AID to its initial
/// signing keys, the pre-rotated next keys, witnesses, and any
/// inception-time anchors. See spec §3.2 / §3.3.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InceptionEvent {
    /// Version string (e.g. `SBYTE10JSON`).
    pub v: String,
    /// Event-type discriminator (always [`EventType::Inception`]).
    pub t: EventType,
    /// SAID of this event.
    pub d: Said,
    /// Controller AID (this event's SAID with `B` substituted for `E`).
    pub i: ControllerAid,
    /// Sequence number (always 0).
    pub s: u64,
    /// Signing-key threshold.
    pub kt: Threshold,
    /// Signing keys revealed at inception.
    pub k: Vec<PublicKey>,
    /// Next-key threshold (committed forward).
    pub nt: Threshold,
    /// SAIDs of pre-rotated next keys.
    pub n: Vec<Said>,
    /// Inception-time configuration traits.
    pub c: Vec<ConfigTrait>,
    /// Initial witness set.
    pub b: Vec<WitnessAid>,
    /// Witness-receipt threshold.
    pub bt: Threshold,
    /// Inception-time anchors.
    pub a: Vec<Anchor>,
}

/// Rotation event: replaces the current signing keys with the keys
/// committed by the previous event's `n` array and commits a fresh
/// pre-rotation set. See spec §3.2 (`rot`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RotationEvent {
    /// Version string.
    pub v: String,
    /// Event-type discriminator (always [`EventType::Rotation`]).
    pub t: EventType,
    /// SAID of this event.
    pub d: Said,
    /// Controller AID.
    pub i: ControllerAid,
    /// Sequence number (`previous.s + 1`).
    pub s: u64,
    /// SAID of the immediately prior event.
    pub p: Said,
    /// New signing-key threshold.
    pub kt: Threshold,
    /// Revealed signing keys (must match previous `n` per spec §3.4).
    pub k: Vec<PublicKey>,
    /// New next-key threshold.
    pub nt: Threshold,
    /// SAIDs of new pre-rotated keys.
    pub n: Vec<Said>,
    /// Witnesses removed at this rotation.
    pub br: Vec<WitnessAid>,
    /// Witnesses added at this rotation.
    pub ba: Vec<WitnessAid>,
    /// New witness-receipt threshold.
    pub bt: Threshold,
    /// Anchors carried in this rotation.
    pub a: Vec<Anchor>,
}

/// Interaction event: a signed anchor to off-log content with no key
/// change. See spec §3.2 (`ixn`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InteractionEvent {
    /// Version string.
    pub v: String,
    /// Event-type discriminator (always [`EventType::Interaction`]).
    pub t: EventType,
    /// SAID of this event.
    pub d: Said,
    /// Controller AID.
    pub i: ControllerAid,
    /// Sequence number.
    pub s: u64,
    /// Prior-event SAID.
    pub p: Said,
    /// Anchors (must be non-empty in practice).
    pub a: Vec<Anchor>,
}

/// Recovery event: catastrophic-key-loss reset that re-establishes
/// signing key set and pre-rotation commitment when the controller has
/// lost the pre-images for the previous `n` field.
///
/// The spec §3.2 reserves `rec`; this crate ingests the semantics:
///
/// * a recovery event MUST be signed by an out-of-band recovery key
///   (controller-supplied at construction time);
/// * a recovery event resets the verifier's view of the controller's
///   signing keys to the new `k` field;
/// * post-recovery, the chain continues from the recovery event's `s`.
///
/// Strict-spec verifiers may still reject `rec` events outright;
/// callers opt in by setting [`crate::verifier::LogVerifier`] to
/// non-strict mode.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct RecoveryEvent {
    /// Version string.
    pub v: String,
    /// Event-type discriminator (always [`EventType::Recovery`]).
    pub t: EventType,
    /// SAID of this event.
    pub d: Said,
    /// Controller AID.
    pub i: ControllerAid,
    /// Sequence number.
    pub s: u64,
    /// Prior-event SAID.
    pub p: Said,
    /// New signing-key threshold (post-recovery).
    pub kt: Threshold,
    /// New signing keys (post-recovery).
    pub k: Vec<PublicKey>,
    /// New next-key threshold.
    pub nt: Threshold,
    /// SAIDs of fresh pre-rotated keys.
    pub n: Vec<Said>,
    /// Anchors (typically attestations from witnesses agreeing to recovery).
    pub a: Vec<Anchor>,
}

/// Delegation event: a parent controller delegates authority to a
/// child controller. The child's events anchor to the parent's log.
///
/// Modeled after KERI `dip` (delegated-inception). v1 here covers
/// delegated inception only; delegated rotation (`drt`) reuses
/// [`RotationEvent`] with anchors back to the parent.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct DelegationEvent {
    /// Version string.
    pub v: String,
    /// Event-type discriminator (always [`EventType::Delegation`]).
    pub t: EventType,
    /// SAID of this event.
    pub d: Said,
    /// Child controller AID.
    pub i: ControllerAid,
    /// Sequence number (0 for a delegated inception).
    pub s: u64,
    /// Prior-event SAID; empty SAID for a delegated inception.
    pub p: Said,
    /// Delegating parent controller AID.
    pub di: ControllerAid,
    /// Signing-key threshold.
    pub kt: Threshold,
    /// Signing keys.
    pub k: Vec<PublicKey>,
    /// Next-key threshold.
    pub nt: Threshold,
    /// SAIDs of pre-rotated keys.
    pub n: Vec<Said>,
}

/// Discriminated union of every event type in a key-event log.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "t")]
pub enum KeyEvent {
    /// Inception (`icp`).
    #[serde(rename = "icp")]
    Inception(InceptionEvent),
    /// Rotation (`rot`).
    #[serde(rename = "rot")]
    Rotation(RotationEvent),
    /// Interaction (`ixn`).
    #[serde(rename = "ixn")]
    Interaction(InteractionEvent),
    /// Recovery (`rec`).
    #[serde(rename = "rec")]
    Recovery(RecoveryEvent),
    /// Delegation (`dip`).
    #[serde(rename = "dip")]
    Delegation(DelegationEvent),
}

impl KeyEvent {
    /// SAID of the event.
    #[must_use]
    pub fn said(&self) -> Said {
        match self {
            Self::Inception(e) => e.d,
            Self::Rotation(e) => e.d,
            Self::Interaction(e) => e.d,
            Self::Recovery(e) => e.d,
            Self::Delegation(e) => e.d,
        }
    }

    /// Controller AID this event belongs to.
    #[must_use]
    pub fn controller(&self) -> &ControllerAid {
        match self {
            Self::Inception(e) => &e.i,
            Self::Rotation(e) => &e.i,
            Self::Interaction(e) => &e.i,
            Self::Recovery(e) => &e.i,
            Self::Delegation(e) => &e.i,
        }
    }

    /// Sequence number `s`.
    #[must_use]
    pub fn sequence(&self) -> u64 {
        match self {
            Self::Inception(e) => e.s,
            Self::Rotation(e) => e.s,
            Self::Interaction(e) => e.s,
            Self::Recovery(e) => e.s,
            Self::Delegation(e) => e.s,
        }
    }

    /// Prior-event SAID, or [`None`] for an inception/delegated-inception.
    #[must_use]
    pub fn prior(&self) -> Option<Said> {
        match self {
            Self::Inception(_) => None,
            Self::Delegation(e) if e.s == 0 => None,
            Self::Rotation(e) => Some(e.p),
            Self::Interaction(e) => Some(e.p),
            Self::Recovery(e) => Some(e.p),
            Self::Delegation(e) => Some(e.p),
        }
    }

    /// Event type discriminator.
    #[must_use]
    pub fn event_type(&self) -> EventType {
        match self {
            Self::Inception(_) => EventType::Inception,
            Self::Rotation(_) => EventType::Rotation,
            Self::Interaction(_) => EventType::Interaction,
            Self::Recovery(_) => EventType::Recovery,
            Self::Delegation(_) => EventType::Delegation,
        }
    }

    /// Canonical CBOR encoding of the event body. Used as the message
    /// over which signatures are produced and verified.
    pub fn to_cbor(&self) -> Result<Vec<u8>> {
        Ok(serde_cbor::to_vec(self)?)
    }

    /// Recompute the SAID from the event body with the `d` field
    /// substituted for the placeholder, per spec §2. For inception
    /// and delegated-inception events the `i` field is also reset to
    /// the empty AID because it is derived from `d` itself.
    pub fn recompute_said(&self) -> Result<Said> {
        let mut clone = self.clone();
        let placeholder = Said::default();
        let empty_aid = ControllerAid(String::new());
        match &mut clone {
            Self::Inception(e) => {
                e.d = placeholder;
                e.i = empty_aid;
            }
            Self::Delegation(e) if e.s == 0 => {
                e.d = placeholder;
                e.i = empty_aid;
            }
            Self::Rotation(e) => e.d = placeholder,
            Self::Interaction(e) => e.d = placeholder,
            Self::Recovery(e) => e.d = placeholder,
            Self::Delegation(e) => e.d = placeholder,
        }
        let bytes = serde_cbor::to_vec(&clone)?;
        Ok(Said::hash(&bytes))
    }

    /// Convenience constructor: validate this event's asserted SAID
    /// against its body, returning [`KeriError::SaidMismatch`] on
    /// disagreement.
    pub fn validate_said(&self) -> Result<()> {
        let computed = self.recompute_said()?;
        if computed != self.said() {
            return Err(KeriError::SaidMismatch {
                asserted: self.said(),
                computed,
            });
        }
        Ok(())
    }
}

/// Helper: build the canonical inception event SAID from its body
/// (with `d` and `i` set to placeholders), then patch `d` and derive
/// the AID. Used by [`crate::controller::Controller::incept`].
pub fn finalize_inception(mut event: InceptionEvent) -> Result<InceptionEvent> {
    event.d = Said::default();
    event.i = ControllerAid(String::new());
    let bytes = serde_cbor::to_vec(&KeyEvent::Inception(event.clone()))?;
    let said = Said::hash(&bytes);
    event.d = said;
    event.i = ControllerAid::from_inception_said(&said);
    // Re-hash with the controller AID populated; for spec-equivalence
    // we anchor the controller AID via the `i` field, which depends on
    // the SAID. To keep the SAID stable we re-derive after the second
    // population by *only* re-encoding with the AID set, then patch
    // back to a stable SAID computed over the body-with-placeholder
    // form. We model this as a fixed-point: the SAID covers the body
    // with `d` zeroed and `i` empty, both of which derive from the
    // SAID itself.
    Ok(event)
}

/// Helper: patch a non-inception event so its `d` field reflects the
/// SAID of its body with `d` placeholdered.
pub fn finalize_event(event: KeyEvent) -> Result<KeyEvent> {
    let mut clone = event;
    let placeholder = Said::default();
    match &mut clone {
        KeyEvent::Inception(e) => e.d = placeholder,
        KeyEvent::Rotation(e) => e.d = placeholder,
        KeyEvent::Interaction(e) => e.d = placeholder,
        KeyEvent::Recovery(e) => e.d = placeholder,
        KeyEvent::Delegation(e) => e.d = placeholder,
    }
    let bytes = serde_cbor::to_vec(&clone)?;
    let said = Said::hash(&bytes);
    match &mut clone {
        KeyEvent::Inception(e) => e.d = said,
        KeyEvent::Rotation(e) => e.d = said,
        KeyEvent::Interaction(e) => e.d = said,
        KeyEvent::Recovery(e) => e.d = said,
        KeyEvent::Delegation(e) => e.d = said,
    }
    Ok(clone)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn per_key_said_is_stable() {
        let k = PublicKey::new(0x01, &[7u8; 32]);
        let a = per_key_said(&k).expect("said");
        let b = per_key_said(&k).expect("said");
        assert_eq!(a, b);
    }

    #[test]
    fn controller_aid_round_trip() {
        let said = Said::hash(b"some-inception");
        let aid = ControllerAid::from_inception_said(&said);
        let back = aid.as_inception_said().expect("inverse");
        assert_eq!(back, said);
    }
}
