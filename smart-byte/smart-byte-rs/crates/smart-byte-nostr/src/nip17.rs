//! NIP-17 private direct messages via NIP-59 gift-wrap + sealed-event.
//!
//! Layered structure:
//!
//! 1. **Rumor** (kind 14) — the actual DM content, NOT signed.
//! 2. **Seal** (kind 13) — the rumor JSON is NIP-44 encrypted to the
//!    recipient and wrapped in a signed event from the real sender.
//! 3. **Gift wrap** (kind 1059) — the seal JSON is NIP-44 encrypted
//!    again, this time signed by a fresh ephemeral key so the sender's
//!    real pubkey never appears on the wire.

use crate::error::NostrError;
use crate::event::{Event, UnsignedEvent};
use crate::keys::{NostrPublicKey, NostrSecretKey};
use crate::nip44;
use serde::{Deserialize, Serialize};

/// Kind 14 — unsigned DM rumor.
pub const KIND_DM_RUMOR: u32 = 14;
/// Kind 13 — sealed event.
pub const KIND_SEAL: u32 = 13;
/// Kind 1059 — gift wrap.
pub const KIND_GIFT_WRAP: u32 = 1059;

/// The raw rumor that the recipient ultimately sees. The rumor is
/// represented as JSON with the same field set as a NIP-01 event but
/// without `sig`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Rumor {
    /// Hex id (SHA-256 of canonical serialization).
    pub id: String,
    /// Real sender pubkey hex.
    pub pubkey: String,
    /// Creation time (seconds).
    pub created_at: i64,
    /// Always [`KIND_DM_RUMOR`] for plain DMs (callers may use other kinds).
    pub kind: u32,
    /// Tags (`["p", <recipient pubkey hex>]` at minimum).
    pub tags: Vec<Vec<String>>,
    /// DM body in cleartext.
    pub content: String,
}

impl Rumor {
    /// Build a rumor addressed `to_pubkey` with the given content.
    pub fn new(
        sender: &NostrPublicKey,
        to_pubkey: &NostrPublicKey,
        content: impl Into<String>,
        now: i64,
    ) -> Self {
        let unsigned = UnsignedEvent::new(*sender, KIND_DM_RUMOR, content, now)
            .with_tag(vec!["p".into(), to_pubkey.to_hex()]);
        let id = unsigned.id();
        Self {
            id: crate::keys::hex_encode(&id),
            pubkey: unsigned.pubkey,
            created_at: unsigned.created_at,
            kind: unsigned.kind,
            tags: unsigned.tags,
            content: unsigned.content,
        }
    }
}

/// Wrap a rumor for `recipient_pk` using `sender_sk`. The resulting
/// [`Event`] is a kind-1059 gift wrap signed by a fresh ephemeral key
/// (which is returned so the caller may publish or discard it).
pub fn wrap(
    sender_sk: &NostrSecretKey,
    recipient_pk: &NostrPublicKey,
    rumor: &Rumor,
    now: i64,
) -> Result<Event, NostrError> {
    let rumor_json = serde_json::to_string(rumor)?;
    let sealed_ct = nip44::encrypt(sender_sk, recipient_pk, rumor_json.as_bytes())?;
    let sender_pk = sender_sk.public_key();
    let seal = UnsignedEvent::new(sender_pk, KIND_SEAL, sealed_ct, now).sign(sender_sk)?;

    let seal_json = serde_json::to_string(&seal)?;
    let wrap_sk = NostrSecretKey::generate();
    let wrap_pk = wrap_sk.public_key();
    let wrap_ct = nip44::encrypt(&wrap_sk, recipient_pk, seal_json.as_bytes())?;
    let wrap_event = UnsignedEvent::new(wrap_pk, KIND_GIFT_WRAP, wrap_ct, now)
        .with_tag(vec!["p".into(), recipient_pk.to_hex()])
        .sign(&wrap_sk)?;
    Ok(wrap_event)
}

/// Unwrap a gift wrap. Returns the recovered rumor, panicking is never
/// emitted — all crypto failures collapse into a [`NostrError`].
pub fn unwrap(recipient_sk: &NostrSecretKey, wrap_event: &Event) -> Result<Rumor, NostrError> {
    if wrap_event.kind != KIND_GIFT_WRAP {
        return Err(NostrError::InvalidEvent("not a gift wrap kind".into()));
    }
    wrap_event.verify()?;
    let wrap_pk = wrap_event.public_key()?;
    let seal_json_bytes = nip44::decrypt(recipient_sk, &wrap_pk, &wrap_event.content)?;
    let seal_json = String::from_utf8(seal_json_bytes)
        .map_err(|e| NostrError::Crypto(e.to_string()))?;
    let seal: Event = serde_json::from_str(&seal_json)?;
    if seal.kind != KIND_SEAL {
        return Err(NostrError::InvalidEvent("inner seal kind mismatch".into()));
    }
    seal.verify()?;
    let seal_pk = seal.public_key()?;
    let rumor_json_bytes = nip44::decrypt(recipient_sk, &seal_pk, &seal.content)?;
    let rumor_json = String::from_utf8(rumor_json_bytes)
        .map_err(|e| NostrError::Crypto(e.to_string()))?;
    let rumor: Rumor = serde_json::from_str(&rumor_json)?;
    if rumor.pubkey != seal.pubkey {
        return Err(NostrError::InvalidEvent(
            "rumor pubkey does not match seal signer".into(),
        ));
    }
    Ok(rumor)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gift_wrap_roundtrip() {
        let alice = NostrSecretKey::generate();
        let bob = NostrSecretKey::generate();
        let rumor = Rumor::new(
            &alice.public_key(),
            &bob.public_key(),
            "secret message",
            1_700_000_000,
        );
        let wrapped = wrap(&alice, &bob.public_key(), &rumor, 1_700_000_000).expect("wrap");
        let got = unwrap(&bob, &wrapped).expect("unwrap");
        assert_eq!(got.content, "secret message");
        assert_eq!(got.pubkey, alice.public_key().to_hex());
    }
}
