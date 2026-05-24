//! Cryptographic audit log.
//!
//! Each call to [`crate::screener::Screener::screen`] appends one row.
//! Every row is:
//!
//! - **Hashed** with BLAKE3 against the previous row's hash (Merkle-chained).
//! - **Signed** with Ed25519 using a key the [`AuditLog`] owns.
//!
//! Two invariants regulators care about:
//!
//! 1. Once written, an entry cannot be silently edited (the signature
//!    breaks).
//! 2. An entry cannot be silently removed or reordered (the chain
//!    hash breaks).
//!
//! [`AuditLog::verify`] checks both.

use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use rand_core::OsRng;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::error::{Error, Result};
use crate::matching::MatchScore;

/// One row in the audit log.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AuditEntry {
    /// Sequence index within the log (0-based).
    pub seq: usize,
    /// UTC timestamp.
    pub at: DateTime<Utc>,
    /// Query name (raw, untruncated — the regulator reads it).
    pub query: String,
    /// `clear` / `hit` / `review`.
    pub decision: String,
    /// Number of hits returned to the caller.
    pub hit_count: usize,
    /// Top entity ID and source list, if any.
    pub top_hit: Option<(String, String)>,
    /// Hex-encoded BLAKE3 hash of `(prev_hash || serialised body)`.
    pub chain_hash: String,
    /// Hex-encoded Ed25519 signature over `chain_hash`.
    pub signature: String,
    /// Globally unique row id (uuid v4 string).
    pub id: String,
}

/// Errors specific to verification.
#[derive(Debug, Error)]
pub enum AuditVerifyError {
    /// The chain hash at `seq` does not match the recomputed value.
    #[error("chain hash mismatch at seq {0}")]
    ChainMismatch(usize),
    /// Signature verification failed at `seq`.
    #[error("signature invalid at seq {0}")]
    BadSignature(usize),
}

/// Chained, signed in-memory audit log.
#[derive(Debug)]
pub struct AuditLog {
    signing_key: SigningKey,
    verifying_key: VerifyingKey,
    entries: Vec<AuditEntry>,
}

impl AuditLog {
    /// Build a fresh log with a new keypair.
    #[must_use]
    pub fn new() -> Self {
        let signing_key = SigningKey::generate(&mut OsRng);
        let verifying_key = signing_key.verifying_key();
        Self { signing_key, verifying_key, entries: Vec::new() }
    }

    /// Public verifying key (regulators ingest this out-of-band).
    #[must_use]
    pub fn verifying_key(&self) -> VerifyingKey {
        self.verifying_key
    }

    /// All entries.
    #[must_use]
    pub fn entries(&self) -> &[AuditEntry] {
        &self.entries
    }

    /// All entries, mutably. ONLY for tests that intentionally tamper.
    #[cfg(test)]
    pub fn entries_mut(&mut self) -> &mut Vec<AuditEntry> {
        &mut self.entries
    }

    /// Append one row. Returns the new row's id.
    pub fn record(
        &mut self,
        query: &str,
        hits: &[MatchScore],
        decision: &str,
    ) -> Result<String> {
        let seq = self.entries.len();
        let at = Utc::now();
        let prev_hash = self
            .entries
            .last()
            .map(|e| e.chain_hash.clone())
            .unwrap_or_default();
        let top_hit = hits.first().map(|h| {
            (
                h.entity.id.clone(),
                h.entity.source_list.label().to_string(),
            )
        });
        let id = uuid_v4();

        // Serialise body deterministically. Bincode would be smaller, but
        // JSON's deterministic-ish output is fine and lets us debug failures
        // by hand. Field order is fixed by serde derive ordering.
        let body = serde_json::to_string(&AuditBody {
            seq,
            at,
            query,
            decision,
            hit_count: hits.len(),
            top_hit: top_hit.clone(),
            id: id.clone(),
        })?;

        // chain_hash = BLAKE3(prev_hash || body)
        let mut hasher = blake3::Hasher::new();
        hasher.update(prev_hash.as_bytes());
        hasher.update(body.as_bytes());
        let chain_hash = hex(hasher.finalize().as_bytes());

        // Sign chain_hash.
        let sig: Signature = self.signing_key.sign(chain_hash.as_bytes());
        let signature = hex(&sig.to_bytes());

        let entry = AuditEntry {
            seq,
            at,
            query: query.to_string(),
            decision: decision.to_string(),
            hit_count: hits.len(),
            top_hit,
            chain_hash,
            signature,
            id: id.clone(),
        };
        self.entries.push(entry);
        Ok(id)
    }

    /// Recompute every entry's chain hash and verify every signature.
    ///
    /// Returns `Ok(())` only if the entire log is intact.
    pub fn verify(&self) -> Result<()> {
        let mut prev_hash = String::new();
        for (i, e) in self.entries.iter().enumerate() {
            let body = serde_json::to_string(&AuditBody {
                seq: e.seq,
                at: e.at,
                query: &e.query,
                decision: &e.decision,
                hit_count: e.hit_count,
                top_hit: e.top_hit.clone(),
                id: e.id.clone(),
            })?;
            let mut hasher = blake3::Hasher::new();
            hasher.update(prev_hash.as_bytes());
            hasher.update(body.as_bytes());
            let recomputed = hex(hasher.finalize().as_bytes());
            if recomputed != e.chain_hash {
                return Err(Error::AuditChainBroken(i));
            }
            let sig_bytes = unhex(&e.signature).ok_or(Error::AuditSignatureInvalid)?;
            let sig_array: [u8; 64] = sig_bytes
                .try_into()
                .map_err(|_| Error::AuditSignatureInvalid)?;
            let sig = Signature::from_bytes(&sig_array);
            self.verifying_key
                .verify(e.chain_hash.as_bytes(), &sig)
                .map_err(|_| Error::AuditSignatureInvalid)?;
            prev_hash = e.chain_hash.clone();
        }
        Ok(())
    }
}

impl Default for AuditLog {
    fn default() -> Self {
        Self::new()
    }
}

/// Internal body shape — distinct from [`AuditEntry`] so that
/// `chain_hash` / `signature` are not part of their own input.
#[derive(Debug, Serialize)]
struct AuditBody<'a> {
    seq: usize,
    at: DateTime<Utc>,
    query: &'a str,
    decision: &'a str,
    hit_count: usize,
    top_hit: Option<(String, String)>,
    id: String,
}

/// Minimal hex encoder so we don't need an extra dep.
fn hex(bytes: &[u8]) -> String {
    const TABLE: &[u8; 16] = b"0123456789abcdef";
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push(TABLE[(b >> 4) as usize] as char);
        out.push(TABLE[(b & 0x0f) as usize] as char);
    }
    out
}

fn unhex(s: &str) -> Option<Vec<u8>> {
    if s.len() % 2 != 0 {
        return None;
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 2);
    for chunk in bytes.chunks(2) {
        let hi = hex_digit(chunk[0])?;
        let lo = hex_digit(chunk[1])?;
        out.push((hi << 4) | lo);
    }
    Some(out)
}

const fn hex_digit(b: u8) -> Option<u8> {
    match b {
        b'0'..=b'9' => Some(b - b'0'),
        b'a'..=b'f' => Some(b - b'a' + 10),
        b'A'..=b'F' => Some(b - b'A' + 10),
        _ => None,
    }
}

/// Tiny standalone uuid v4 generator using BLAKE3 of fresh OS bytes.
fn uuid_v4() -> String {
    use rand_core::RngCore;
    let mut bytes = [0u8; 16];
    OsRng.fill_bytes(&mut bytes);
    bytes[6] = (bytes[6] & 0x0f) | 0x40; // version 4
    bytes[8] = (bytes[8] & 0x3f) | 0x80; // variant 1
    format!(
        "{:02x}{:02x}{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}-{:02x}{:02x}{:02x}{:02x}{:02x}{:02x}",
        bytes[0], bytes[1], bytes[2], bytes[3],
        bytes[4], bytes[5],
        bytes[6], bytes[7],
        bytes[8], bytes[9],
        bytes[10], bytes[11], bytes[12], bytes[13], bytes[14], bytes[15],
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_log_verifies() {
        AuditLog::new().verify().expect("empty");
    }

    #[test]
    fn record_then_verify() {
        let mut log = AuditLog::new();
        log.record("alpha", &[], "clear").unwrap();
        log.record("beta", &[], "clear").unwrap();
        log.record("gamma", &[], "clear").unwrap();
        log.verify().expect("intact");
    }

    #[test]
    fn tamper_breaks_chain() {
        let mut log = AuditLog::new();
        log.record("alpha", &[], "clear").unwrap();
        log.record("beta", &[], "clear").unwrap();
        log.record("gamma", &[], "clear").unwrap();
        // Tamper with entry 1's query field. Chain rehash will fail.
        log.entries_mut()[1].query = "MUTATED".to_string();
        let err = log.verify().expect_err("must fail");
        match err {
            Error::AuditChainBroken(i) => assert_eq!(i, 1),
            other => panic!("wrong err: {other:?}"),
        }
    }

    #[test]
    fn tamper_signature_only_breaks() {
        let mut log = AuditLog::new();
        log.record("alpha", &[], "clear").unwrap();
        // Replace signature with garbage.
        log.entries_mut()[0].signature =
            "00".repeat(64);
        let err = log.verify().expect_err("must fail");
        match err {
            Error::AuditSignatureInvalid => {}
            other => panic!("wrong err: {other:?}"),
        }
    }
}
