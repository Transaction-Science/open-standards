//! DIF Encrypted Data Vaults v0.10 wire types.
//!
//! Mirrors the structures defined by
//! <https://identity.foundation/edv-spec/> § 4 — Data model:
//!
//! * [`EncryptedDocument`] — the at-rest record stored by a vault.
//! * [`Config`] — vault configuration (controller, key agreement key,
//!   capability invocation key, content metadata, supported key agreement
//!   methods).
//! * [`Provider`] — vault provider metadata (URL, supported features).
//! * [`Hmac`] — HMAC key descriptor used to blind index values (the vault
//!   sees only HMACs, never plaintext, so equality queries are possible
//!   without disclosing values).
//! * [`Stream`] — descriptor for chunked stream encryption, used to break
//!   large payloads into AEAD-sealed chunks that decrypt independently.
//!
//! Concrete JWE / index / capability content sits in [`crate::jwe`],
//! [`crate::index`], and [`crate::zcap`].

use serde::{Deserialize, Serialize};

/// Vault configuration record (DIF EDV v0.10 § 4.1).
///
/// A `Config` is the public face of a vault — it advertises which key
/// agreement keys can be used to wrap content-encryption keys for that
/// vault, who controls the vault, and what kinds of indexed lookup are
/// supported.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Config {
    /// `@context` URIs (DIF EDV v0.10 § 4.1).
    #[serde(rename = "@context")]
    pub context: Vec<String>,
    /// Vault identifier — typically a URN or DID URL.
    pub id: String,
    /// DID of the entity that controls vault configuration.
    pub controller: String,
    /// Sequence number used for optimistic concurrency on the config.
    #[serde(default)]
    pub sequence: u64,
    /// Public key reference used as the recipient for JWE wrapping.
    pub key_agreement_key: KeyDescriptor,
    /// Public key used to verify capability invocations (HTTP signatures).
    pub hmac_key: KeyDescriptor,
    /// Descriptors for supported HMAC key (used for encrypted indexes).
    #[serde(default)]
    pub indexed: Vec<KeyDescriptor>,
}

impl Config {
    /// Construct a minimal v0.10 config.
    pub fn new<I: Into<String>>(
        id: I,
        controller: I,
        key_agreement_key: KeyDescriptor,
        hmac_key: KeyDescriptor,
    ) -> Self {
        Self {
            context: vec![
                "https://w3id.org/security/v2".into(),
                "https://w3id.org/edv/v1".into(),
            ],
            id: id.into(),
            controller: controller.into(),
            sequence: 0,
            key_agreement_key,
            hmac_key,
            indexed: Vec::new(),
        }
    }
}

/// A reference to a key, by `id` and `type`, as carried in vault configs
/// and key agreement key descriptors.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct KeyDescriptor {
    /// Key identifier (DID URL with fragment, or URN).
    pub id: String,
    /// JWK / verification method type (e.g. `JsonWebKey2020`).
    #[serde(rename = "type")]
    pub key_type: String,
    /// Controller DID, if distinct from the parent record's controller.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub controller: Option<String>,
}

/// A vault provider's discovery record (DIF EDV v0.10 § 6 — HTTP API).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Provider {
    /// Base URL of the vault provider's HTTP API.
    pub url: String,
    /// Supported content-encryption suites (`A256GCM`, etc.).
    #[serde(default)]
    pub enc: Vec<String>,
    /// Supported key agreement algorithms (`ECDH-ES+A256KW`, etc.).
    #[serde(default)]
    pub alg: Vec<String>,
    /// Whether the provider supports chunked streams.
    #[serde(default)]
    pub streams: bool,
    /// Whether the provider supports ZCAP-LD capability invocation.
    #[serde(default)]
    pub zcap: bool,
}

impl Default for Provider {
    fn default() -> Self {
        Self {
            url: String::new(),
            enc: vec!["A256GCM".into()],
            alg: vec!["ECDH-ES+A256KW".into()],
            streams: true,
            zcap: true,
        }
    }
}

/// An HMAC key descriptor used to blind index values (DIF EDV v0.10
/// § 4.4 — Indexing).
///
/// The vault stores HMAC tags rather than plaintext attribute values so it
/// can answer equality queries (does any document have attribute `X` equal
/// to value `Y`?) without ever seeing `Y` in the clear.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Hmac {
    /// HMAC key id (DID URL with fragment).
    pub id: String,
    /// HMAC algorithm — always `HS256` for v0.10.
    #[serde(rename = "type")]
    pub key_type: String,
}

impl Hmac {
    /// Construct an HMAC descriptor with the canonical type.
    pub fn new<I: Into<String>>(id: I) -> Self {
        Self {
            id: id.into(),
            key_type: "Sha256HmacKey2019".into(),
        }
    }
}

/// Descriptor for a chunked stream of ciphertext (DIF EDV v0.10 § 4.5).
///
/// Large payloads are split into fixed-size chunks, each independently
/// sealed with AES-256-GCM. The descriptor records the chunk size and
/// total chunk count so a consumer can fetch chunks in any order.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Stream {
    /// Stream identifier (URN).
    pub id: String,
    /// Number of chunks in the stream.
    pub chunk_count: usize,
    /// Plaintext bytes per chunk (the final chunk may be smaller).
    pub chunk_size: usize,
    /// Sequence number on the stream descriptor (concurrency control).
    #[serde(default)]
    pub sequence: u64,
}

/// An indexed attribute on an encrypted document. Stored as `name`+`value`
/// where both fields are HMAC tags (base64url-encoded) — the vault never
/// sees the plaintext name or value.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexedAttribute {
    /// HMAC tag over the attribute name.
    pub name: String,
    /// HMAC tag over the attribute value.
    pub value: String,
    /// Whether duplicates of this `name` HMAC are permitted on this doc.
    #[serde(default)]
    pub unique: bool,
}

/// A group of indexed attributes derived from one HMAC key.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct IndexedEntry {
    /// HMAC key that produced these tags.
    pub hmac: Hmac,
    /// Attribute tags.
    pub attributes: Vec<IndexedAttribute>,
}

/// An at-rest encrypted document (DIF EDV v0.10 § 4.2).
///
/// The body is a JWE in flattened JSON serialisation; the optional
/// `indexed` block carries the encrypted index entries used for lookup.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EncryptedDocument {
    /// Document id (URN).
    pub id: String,
    /// Sequence number — incremented on each update for optimistic
    /// concurrency control.
    #[serde(default)]
    pub sequence: u64,
    /// The JWE-wrapped content (flattened JSON serialisation, base64url).
    pub jwe: serde_json::Value,
    /// Optional encrypted index entries.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub indexed: Vec<IndexedEntry>,
    /// Optional stream descriptor, present when the body is chunked.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub stream: Option<Stream>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn config_round_trip() {
        let kd = KeyDescriptor {
            id: "did:example:alice#kex-1".into(),
            key_type: "JsonWebKey2020".into(),
            controller: None,
        };
        let hk = KeyDescriptor {
            id: "did:example:alice#hmac-1".into(),
            key_type: "Sha256HmacKey2019".into(),
            controller: None,
        };
        let cfg = Config::new(
            "urn:uuid:vault-1",
            "did:example:alice",
            kd,
            hk,
        );
        let s = serde_json::to_string(&cfg).expect("serialize");
        let d: Config = serde_json::from_str(&s).expect("deserialize");
        assert_eq!(cfg, d);
    }

    #[test]
    fn provider_defaults() {
        let p = Provider::default();
        assert!(p.zcap);
        assert!(p.streams);
        assert_eq!(p.enc, vec!["A256GCM".to_string()]);
    }

    #[test]
    fn hmac_canonical_type() {
        let h = Hmac::new("did:example:vault#hmac");
        assert_eq!(h.key_type, "Sha256HmacKey2019");
    }
}
