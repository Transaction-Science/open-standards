//! Entity resolution: the rules for turning raw payment identifiers into
//! canonical, hashed [`Entity`] records that can be safely stored as
//! graph vertices.
//!
//! ## Why hash everything
//!
//! Fraud graphs are sensitive: an attacker who exfiltrates the vertex
//! set learns *who* a merchant transacts with. We sidestep the problem
//! by storing only SHA-256 truncations of the canonical form. The graph
//! can still answer "do these two payments share a card?" because equal
//! inputs hash to equal keys, but it cannot recover the PAN.

use core::fmt;

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// What kind of identifier a vertex represents.
///
/// Each kind has a slightly different canonicalisation rule. Storing the
/// kind alongside the hash lets us refuse "is this email the same as
/// this phone?" comparisons that would otherwise collide in hash-space.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum EntityKind {
    /// Hashed Primary Account Number (full PAN).
    CardHash,
    /// BIN (first 6) + last 4 digits — looser identity than `CardHash`.
    /// Used when full PAN is unavailable (e.g. tokenised transactions).
    BinLast4,
    /// Lower-cased, trimmed email address, then hashed.
    EmailHash,
    /// Device fingerprint string (vendor-defined: FingerprintJS, etc.).
    DeviceFingerprint,
    /// IPv4 or IPv6 in canonical text form.
    Ip,
    /// Normalised postal address (uppercased, single-spaced, no punctuation).
    Address,
    /// E.164 phone number (digits only, leading `+`).
    PhoneHash,
    /// A natural-key string the caller wants to track verbatim
    /// (merchant id, account id, organisation id). Hashed as-is.
    Account,
}

impl fmt::Display for EntityKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::CardHash => "card",
            Self::BinLast4 => "bin_last4",
            Self::EmailHash => "email",
            Self::DeviceFingerprint => "device",
            Self::Ip => "ip",
            Self::Address => "address",
            Self::PhoneHash => "phone",
            Self::Account => "account",
        };
        f.write_str(s)
    }
}

/// The canonical, hashed identity of an entity.
///
/// Two payments referring to the same underlying identifier (after
/// canonicalisation) MUST produce the same `EntityKey`. This is the
/// invariant the graph relies on for entity resolution.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub struct EntityKey {
    /// Tag so two different kinds don't collide.
    pub kind: EntityKind,
    /// First 16 bytes (128 bits) of SHA-256 of the canonical form.
    /// 128 bits is comfortably enough for the cardinalities seen in
    /// payment graphs (tens of billions of entities); the saved memory
    /// per vertex matters at scale.
    pub digest: [u8; 16],
}

impl EntityKey {
    /// Build a key from a raw value, applying the canonicalisation rule
    /// appropriate for `kind`.
    pub fn from_raw(kind: EntityKind, raw: &str) -> Self {
        let canonical = canonicalize(kind, raw);
        let mut hasher = Sha256::new();
        hasher.update(canonical.as_bytes());
        let full = hasher.finalize();
        let mut digest = [0u8; 16];
        digest.copy_from_slice(&full[..16]);
        Self { kind, digest }
    }

    /// Hex-encoded display form. Stable across runs; useful in logs.
    pub fn to_hex(self) -> String {
        let mut out = String::with_capacity(32);
        for b in self.digest {
            // No `format!` allocations per byte: just push two hex chars.
            const HEX: &[u8; 16] = b"0123456789abcdef";
            out.push(HEX[(b >> 4) as usize] as char);
            out.push(HEX[(b & 0x0f) as usize] as char);
        }
        out
    }
}

impl fmt::Display for EntityKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:{}", self.kind, self.to_hex())
    }
}

/// A vertex in the graph: a hashed identifier plus a coarse kind tag.
///
/// All "first seen" / "last seen" / "tx count" enrichment lives in the
/// graph layer (see [`crate::graph::VertexMeta`]), not here. This struct
/// is the immutable identity-key shape.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Entity {
    /// Canonical key — the only field used for equality.
    pub key: EntityKey,
}

impl Entity {
    /// Convenience constructor that delegates to [`EntityKey::from_raw`].
    pub fn new(kind: EntityKind, raw: &str) -> Self {
        Self {
            key: EntityKey::from_raw(kind, raw),
        }
    }
}

/// Apply the canonicalisation rule for a given [`EntityKind`].
///
/// Exposed so callers can implement bespoke entity resolution paths
/// (e.g. fuzzy address match) and feed the already-canonical string
/// into [`EntityKey::from_raw`] with a matching `kind`.
pub fn canonicalize(kind: EntityKind, raw: &str) -> String {
    match kind {
        EntityKind::CardHash | EntityKind::BinLast4 | EntityKind::DeviceFingerprint => {
            // Already an opaque token — strip whitespace only.
            raw.trim().to_string()
        }
        EntityKind::EmailHash => raw.trim().to_ascii_lowercase(),
        EntityKind::Ip => raw.trim().to_string(),
        EntityKind::Address => {
            let upper = raw.to_ascii_uppercase();
            let mut out = String::with_capacity(upper.len());
            let mut last_was_space = true;
            for ch in upper.chars() {
                if ch.is_ascii_alphanumeric() {
                    out.push(ch);
                    last_was_space = false;
                } else if !last_was_space {
                    out.push(' ');
                    last_was_space = true;
                }
            }
            out.trim().to_string()
        }
        EntityKind::PhoneHash => {
            let mut out = String::with_capacity(raw.len());
            // Preserve leading '+' if present, then strip everything non-digit.
            let mut chars = raw.trim().chars().peekable();
            if let Some(&'+') = chars.peek() {
                out.push('+');
                let _ = chars.next();
            }
            for ch in chars {
                if ch.is_ascii_digit() {
                    out.push(ch);
                }
            }
            out
        }
        EntityKind::Account => raw.trim().to_string(),
    }
}

/// True if `billing` and `shipping` addresses would resolve to *different*
/// entity keys — the classic "shipping mismatch" fraud signal.
///
/// Inputs are raw; canonicalisation is applied internally.
pub fn billing_shipping_mismatch(billing: &str, shipping: &str) -> bool {
    canonicalize(EntityKind::Address, billing) != canonicalize(EntityKind::Address, shipping)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn email_is_case_insensitive() {
        let a = EntityKey::from_raw(EntityKind::EmailHash, "Foo@Example.com");
        let b = EntityKey::from_raw(EntityKind::EmailHash, "foo@example.com");
        assert_eq!(a, b);
    }

    #[test]
    fn different_kinds_dont_collide() {
        let a = EntityKey::from_raw(EntityKind::EmailHash, "foo");
        let b = EntityKey::from_raw(EntityKind::Account, "foo");
        assert_ne!(a, b);
    }

    #[test]
    fn phone_normalises_punctuation() {
        let a = EntityKey::from_raw(EntityKind::PhoneHash, "+1 (415) 555-1212");
        let b = EntityKey::from_raw(EntityKind::PhoneHash, "+14155551212");
        assert_eq!(a, b);
    }

    #[test]
    fn address_normalises_whitespace_and_case() {
        let a = canonicalize(EntityKind::Address, "123 Main St, Apt #4");
        let b = canonicalize(EntityKind::Address, "  123  main  st   apt  4 ");
        assert_eq!(a, b);
    }

    #[test]
    fn shipping_mismatch_detected() {
        assert!(billing_shipping_mismatch(
            "1 First Ave",
            "9999 Different Rd"
        ));
        assert!(!billing_shipping_mismatch(
            "1 First Ave",
            "1 first ave"
        ));
    }
}
