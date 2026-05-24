//! NIP-01 event format: signing, verification, and canonical serialization.
//!
//! A Nostr event has seven fields: `id`, `pubkey`, `created_at`, `kind`,
//! `tags`, `content`, `sig`. The `id` is the SHA-256 of the canonical
//! JSON `[0, pubkey, created_at, kind, tags, content]`. The `sig` is a
//! BIP-340 Schnorr signature of the `id` under the event's `pubkey`.

use crate::error::NostrError;
use crate::keys::{NostrPublicKey, NostrSecretKey, hex_decode, hex_encode, schnorr_sign, schnorr_verify};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// A signed Nostr event ready for relay submission.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Event {
    /// 32-byte event id (hex of SHA-256 over the canonical serialization).
    pub id: String,
    /// 32-byte x-only public key (hex).
    pub pubkey: String,
    /// Unix timestamp in seconds.
    pub created_at: i64,
    /// Event kind. Common kinds: 0 metadata, 1 text note, 4 DM, 5 deletion,
    /// 1059 gift-wrap, 13 seal, 14 unsigned DM, 10002 relay list, etc.
    pub kind: u32,
    /// Tag list: each tag is a list of strings.
    pub tags: Vec<Vec<String>>,
    /// Content payload (UTF-8). May be ciphertext for DM / encrypted kinds.
    pub content: String,
    /// 64-byte Schnorr signature (hex).
    pub sig: String,
}

/// An event prior to signing — id and signature have not been computed.
#[derive(Debug, Clone)]
pub struct UnsignedEvent {
    /// x-only public key of the signer (hex).
    pub pubkey: String,
    /// Unix timestamp in seconds.
    pub created_at: i64,
    /// Event kind.
    pub kind: u32,
    /// Tag list.
    pub tags: Vec<Vec<String>>,
    /// Content payload.
    pub content: String,
}

impl UnsignedEvent {
    /// Construct a new unsigned event at `now` (caller supplies the clock).
    pub fn new(pubkey: NostrPublicKey, kind: u32, content: impl Into<String>, now: i64) -> Self {
        Self {
            pubkey: pubkey.to_hex(),
            created_at: now,
            kind,
            tags: Vec::new(),
            content: content.into(),
        }
    }

    /// Append a tag, builder-style.
    pub fn with_tag(mut self, tag: Vec<String>) -> Self {
        self.tags.push(tag);
        self
    }

    /// Compute the canonical 32-byte id (SHA-256 of the serialization).
    pub fn id(&self) -> [u8; 32] {
        let canonical = canonical_serialize(
            &self.pubkey,
            self.created_at,
            self.kind,
            &self.tags,
            &self.content,
        );
        let mut hasher = Sha256::new();
        hasher.update(canonical.as_bytes());
        let out = hasher.finalize();
        let mut id = [0u8; 32];
        id.copy_from_slice(&out);
        id
    }

    /// Sign this event with `sk`. Verifies that the supplied secret key's
    /// derived public key matches `pubkey`.
    pub fn sign(self, sk: &NostrSecretKey) -> Result<Event, NostrError> {
        let derived = sk.public_key().to_hex();
        if derived != self.pubkey {
            return Err(NostrError::InvalidKey(
                "secret key does not match event pubkey".into(),
            ));
        }
        let id = self.id();
        let sig = schnorr_sign(sk, &id)?;
        Ok(Event {
            id: hex_encode(&id),
            pubkey: self.pubkey,
            created_at: self.created_at,
            kind: self.kind,
            tags: self.tags,
            content: self.content,
            sig: hex_encode(&sig),
        })
    }
}

impl Event {
    /// Recompute the canonical id from the event's fields.
    pub fn computed_id(&self) -> [u8; 32] {
        let canonical = canonical_serialize(
            &self.pubkey,
            self.created_at,
            self.kind,
            &self.tags,
            &self.content,
        );
        let mut hasher = Sha256::new();
        hasher.update(canonical.as_bytes());
        let out = hasher.finalize();
        let mut id = [0u8; 32];
        id.copy_from_slice(&out);
        id
    }

    /// Verify that `id` matches the canonical serialization AND that `sig`
    /// is a valid Schnorr signature under `pubkey`.
    pub fn verify(&self) -> Result<(), NostrError> {
        let computed = self.computed_id();
        let claimed = hex_decode(&self.id)?;
        if claimed.len() != 32 || claimed != computed {
            return Err(NostrError::IdMismatch);
        }
        let pk_bytes = hex_decode(&self.pubkey)?;
        if pk_bytes.len() != 32 {
            return Err(NostrError::InvalidKey("pubkey must be 32 bytes".into()));
        }
        let mut pk_arr = [0u8; 32];
        pk_arr.copy_from_slice(&pk_bytes);
        let pk = NostrPublicKey::from_bytes(&pk_arr)?;

        let sig_bytes = hex_decode(&self.sig)?;
        if sig_bytes.len() != 64 {
            return Err(NostrError::BadSignature);
        }
        let mut sig_arr = [0u8; 64];
        sig_arr.copy_from_slice(&sig_bytes);

        schnorr_verify(&pk, &computed, &sig_arr)
    }

    /// Return the `pubkey` parsed into an [`NostrPublicKey`].
    pub fn public_key(&self) -> Result<NostrPublicKey, NostrError> {
        NostrPublicKey::from_hex(&self.pubkey)
    }
}

/// Build the canonical JSON `[0, pubkey, created_at, kind, tags, content]`
/// used to compute the event id. Uses `serde_json` so escaping matches the
/// NIP-01 reference.
pub fn canonical_serialize(
    pubkey: &str,
    created_at: i64,
    kind: u32,
    tags: &[Vec<String>],
    content: &str,
) -> String {
    // serde_json produces compact output with the same escaping rules NIP-01
    // requires (forward slashes are not escaped; control chars are escaped).
    let value = serde_json::json!([0, pubkey, created_at, kind, tags, content]);
    serde_json::to_string(&value).unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::keys::NostrSecretKey;

    #[test]
    fn sign_then_verify_roundtrip() {
        let sk = NostrSecretKey::generate();
        let pk = sk.public_key();
        let ev = UnsignedEvent::new(pk, 1, "hello nostr", 1_700_000_000)
            .with_tag(vec!["t".into(), "test".into()])
            .sign(&sk)
            .expect("sign");
        ev.verify().expect("verify");
    }

    #[test]
    fn tamper_detection() {
        let sk = NostrSecretKey::generate();
        let pk = sk.public_key();
        let mut ev = UnsignedEvent::new(pk, 1, "hello", 1_700_000_000)
            .sign(&sk)
            .expect("sign");
        ev.content = "tampered".into();
        assert!(ev.verify().is_err());
    }

    #[test]
    fn canonical_format_is_array_with_zero_prefix() {
        let s = canonical_serialize("ab", 1, 1, &[], "hi");
        assert!(s.starts_with("[0,\"ab\","));
    }
}
