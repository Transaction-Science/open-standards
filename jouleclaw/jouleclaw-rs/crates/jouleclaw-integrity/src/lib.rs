//! # jouleclaw-integrity
//!
//! Two-tier page-and-record integrity, the shape Postgres 18,
//! RocksDB, Pebble (CockroachDB), and InnoDB have all converged on:
//!
//! - **Fast path — CRC32C (Castagnoli polynomial)**. Mathematically
//!   proven detection of all 1–3-bit errors and burst errors ≤ 32
//!   bits in blocks < 2^32 bits. Hardware-accelerated on x86
//!   (SSE4.2 / AVX-512 VPCLMULQDQ) and ARM. Collision rate ≈ 2⁻³².
//!   Use for per-page corruption detection in the hot path.
//! - **Strong path — blake3**. Cryptographic 256-bit hash;
//!   collision-resistant under standard assumptions. Use for the
//!   tamper-evident ledger layer (journal hash chain, snapshot
//!   manifests, cross-source agreement).
//!
//! Wave-4 SOTA brief: "CRC32C for the fast-path bulk page integrity
//! + separate cryptographic hash for the tamper-evident layer" is
//! the durable doctrine; no major database has flipped its default
//! to xxh3 or blake3 for page integrity as of May 2026 (RocksDB
//! supports xxh3 as an opt-in but ships CRC32C by default; Postgres
//! 18 enables CRC32C `data_checksums` by default).
//!
//! ## Honest scope
//!
//! - CRC32C detects **random** corruption, not adversarial
//!   tampering. The doctrine pairs it with blake3 so the consumer
//!   has both.
//! - blake3 is overkill for transient bit-flips. The two-tier split
//!   is the point.
//! - Neither addresses **replay** attacks; that needs an ordered
//!   chain ([`jouleclaw_graph::Journal`]).
//! - `crc32fast` (IEEE polynomial) is the WRONG crate for DB-format
//!   work — we depend on `crc32c` (Castagnoli) explicitly.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(unexpected_cfgs)]

use jouleclaw_bounded::{Bounded, BoundedError, FastStrong};
use serde::{Deserialize, Serialize};

// ─────────────────────────────────────────────────────────────────────
// Evidence types
// ─────────────────────────────────────────────────────────────────────

/// Evidence produced by [`Integrity::fast`] or [`Integrity::strong`].
/// Two-variant enum so the consumer can detect divergence by
/// variant.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "kind", content = "value", rename_all = "snake_case")]
pub enum IntegrityEvidence {
    /// 32-bit Castagnoli CRC. Hex-lowercase, zero-padded to 8.
    Crc32c(String),
    /// 256-bit blake3 digest. Hex-lowercase, 64 chars.
    Blake3(String),
}

impl IntegrityEvidence {
    /// True iff this evidence is the strong (blake3) tier.
    pub fn is_blake3(&self) -> bool {
        matches!(self, Self::Blake3(_))
    }
    /// Extract the hex string.
    pub fn as_hex(&self) -> &str {
        match self {
            Self::Crc32c(h) | Self::Blake3(h) => h,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// The two-tier Integrity primitive
// ─────────────────────────────────────────────────────────────────────

/// The two-tier integrity primitive. Holds no state — a typed
/// adapter around the canonical Castagnoli CRC and blake3
/// implementations.
#[derive(Debug, Default, Clone, Copy)]
pub struct Integrity;

impl Integrity {
    /// Construct an instance. State-free; identical to `default()`.
    pub fn new() -> Self {
        Self
    }

    /// Compute the bare CRC32C (Castagnoli) of `input`. Convenience
    /// alias for the fast tier.
    pub fn crc32c(&self, input: &[u8]) -> u32 {
        crc32c::crc32c(input)
    }

    /// Compute the blake3 digest of `input`. Convenience alias for
    /// the strong tier.
    pub fn blake3(&self, input: &[u8]) -> [u8; 32] {
        *blake3::hash(input).as_bytes()
    }
}

impl FastStrong for Integrity {
    type Evidence = IntegrityEvidence;
    type Input = [u8];

    /// Fast path — CRC32C. Use for per-page corruption detection in
    /// hot loops.
    fn fast(&self, input: &[u8]) -> Self::Evidence {
        IntegrityEvidence::Crc32c(format!("{:08x}", self.crc32c(input)))
    }

    /// Strong path — blake3. Use for tamper-evident ledger layers;
    /// journal hash chain, snapshot manifests, cross-source
    /// agreement.
    fn strong(&self, input: &[u8]) -> Self::Evidence {
        IntegrityEvidence::Blake3(hex_lower(&self.blake3(input)))
    }
}

impl Bounded for Integrity {
    /// We report the FAST tier's bound here since `bound()` is
    /// single-valued; consumers asking for the strong tier
    /// explicitly know it's 2⁻¹²⁸ class.
    fn bound(&self) -> BoundedError {
        BoundedError {
            epsilon: 2f64.powi(-32),
            delta: 0.0,
            memory_bytes: Some(0),
            kind: jouleclaw_bounded::BoundKind::Absolute,
        }
    }
}

/// Cross-source agreement check — used for "did two backends
/// produce the same answer" bug detection. Two `Integrity` evidence
/// values agree iff they have the same variant AND hex string.
pub fn agree(a: &IntegrityEvidence, b: &IntegrityEvidence) -> bool {
    a == b
}

fn hex_lower(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        out.push_str(&format!("{:02x}", b));
    }
    out
}

// ─────────────────────────────────────────────────────────────────────
// Pebble-style CRC32C with rotation
// ─────────────────────────────────────────────────────────────────────

/// Pebble (CockroachDB) noticed a coincidental-collision hazard
/// when a page format includes a CRC slot and the slot itself
/// gets CRC'd along with the rest of the bytes. The fix is to
/// rotate the CRC before storing it. We expose the rotation as a
/// helper so callers writing page-format checksums can apply the
/// same trick.
///
/// Returns `crc.rotate_right(15).wrapping_add(0xa282ead8)`. The
/// constants come straight from Pebble's `internal/crc` package.
pub fn pebble_crc32c(payload: &[u8]) -> u32 {
    let c = crc32c::crc32c(payload);
    c.rotate_right(15).wrapping_add(0xa282ead8)
}

// ─────────────────────────────────────────────────────────────────────
// Kani harnesses
// ─────────────────────────────────────────────────────────────────────

/// `crc32c` is deterministic — the same input MUST produce the same
/// CRC.
#[cfg(kani)]
#[kani::proof]
fn kani_crc32c_deterministic() {
    let bytes: [u8; 4] = kani::any();
    let a = crc32c::crc32c(&bytes);
    let b = crc32c::crc32c(&bytes);
    kani::assert(a == b, "CRC32C deterministic");
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn integrity_fast_returns_crc32c_evidence() {
        let i = Integrity::new();
        let e = i.fast(b"hello world");
        match e {
            IntegrityEvidence::Crc32c(h) => assert_eq!(h.len(), 8),
            other => panic!("expected Crc32c, got {other:?}"),
        }
    }

    #[test]
    fn integrity_strong_returns_blake3_evidence() {
        let i = Integrity::new();
        let e = i.strong(b"hello world");
        match e {
            IntegrityEvidence::Blake3(h) => assert_eq!(h.len(), 64),
            other => panic!("expected Blake3, got {other:?}"),
        }
    }

    #[test]
    fn integrity_both_tiers_deterministic() {
        let i = Integrity::new();
        assert_eq!(i.fast(b"x"), i.fast(b"x"));
        assert_eq!(i.strong(b"x"), i.strong(b"x"));
        // Variants differ → PartialEq false by construction.
        assert_ne!(i.fast(b"x"), i.strong(b"x"));
    }

    #[test]
    fn integrity_evidence_round_trips_through_json() {
        let e = IntegrityEvidence::Crc32c("deadbeef".into());
        let j = serde_json::to_value(&e).unwrap();
        assert_eq!(j["kind"], "crc32c");
        let back: IntegrityEvidence = serde_json::from_value(j).unwrap();
        assert_eq!(back, e);
    }

    #[test]
    fn agree_detects_cross_source_disagreement() {
        let i = Integrity::new();
        let a = i.fast(b"hello");
        let b = i.fast(b"world");
        assert!(!agree(&a, &b));
        let c = i.fast(b"hello");
        assert!(agree(&a, &c));
    }

    #[test]
    fn bound_reports_crc32c_collision_rate() {
        let i = Integrity::new();
        let b = i.bound();
        assert!(b.epsilon > 0.0 && b.epsilon < 1e-9);
        assert_eq!(b.memory_bytes, Some(0));
    }

    #[test]
    fn pebble_crc_differs_from_bare_crc() {
        let bare = crc32c::crc32c(b"page-bytes");
        let pebble = pebble_crc32c(b"page-bytes");
        assert_ne!(bare, pebble);
        assert_eq!(pebble_crc32c(b"page-bytes"), pebble);
    }

    #[test]
    fn known_blake3_vector_for_empty_input() {
        let i = Integrity::new();
        let h = i.strong(b"");
        assert_eq!(
            h.as_hex(),
            "af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262"
        );
    }

    #[test]
    fn known_crc32c_vector_from_rfc_3720() {
        // RFC 3720 iSCSI test vector: CRC32C("123456789") = 0xe3069283.
        let i = Integrity::new();
        let h = i.fast(b"123456789");
        assert_eq!(h.as_hex(), "e3069283");
    }

    #[test]
    fn integrity_is_blake3_predicate_correct() {
        let i = Integrity::new();
        assert!(!i.fast(b"x").is_blake3());
        assert!(i.strong(b"x").is_blake3());
    }
}
