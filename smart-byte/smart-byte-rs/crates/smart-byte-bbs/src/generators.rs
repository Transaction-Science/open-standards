//! Public BBS+ message generators.
//!
//! Per `draft-irtf-cfrg-bbs-signatures-08` § 4.1.1
//! (`create_generators`), the generators used to bind messages into a
//! signature are derived deterministically from a domain-separation
//! tag and an integer index. The draft uses RFC 9380 `hash_to_curve`
//! over G1 (suite `BLS12381G1_XMD:SHA-256_SSWU_RO_`) with the input
//! `seed || I2OSP(index, 8)`.
//!
//! Because `bls12_381` 0.8's `hash_to_curve` API is bound to the
//! `digest` 0.9 trait family it accepts `sha2 0.9::Sha256` (re-exported
//! locally as `sha2_v9::Sha256`). This is the same SHA-256 algorithm
//! the IETF draft prescribes; the version split is purely a crate
//! dependency artefact.
//!
//! The generator set produced here is ordered:
//!
//! ```text
//! [ H_0, H_1, H_2, ..., H_{count-1} ]
//! ```
//!
//! Conventionally `H_0` is the "blinding" generator paired with the
//! signature's `s` term and `H_1 .. H_{count-1}` are the per-message
//! generators. Smart Byte's sign/verify treats the first
//! `message_count + 2` generators as `(H0, H_msg_0..H_msg_{n-1}, Q)`.

use bls12_381::{
    G1Projective,
    hash_to_curve::{ExpandMsgXmd, HashToCurve},
};

/// Per-instance domain-separation tag prefix used when deriving
/// generators.
pub const GENERATOR_DST_PREFIX: &[u8] =
    b"SMART_BYTE_BBS_2026_BLS12381G1_XMD:SHA-256_SSWU_RO_MSG_GEN_";

/// Derive `count` independent G1 generators from a public domain
/// string.
///
/// The output is deterministic: callers passing the same `domain` and
/// `count` always get the same generator vector. This is critical for
/// interop with other BBS+ implementations and for the W3C
/// `bbs-2023` cryptosuite which makes the generator set part of the
/// public signing parameters.
pub fn message_generators(domain: &[u8], count: usize) -> Vec<G1Projective> {
    let mut dst = Vec::with_capacity(GENERATOR_DST_PREFIX.len() + domain.len());
    dst.extend_from_slice(GENERATOR_DST_PREFIX);
    dst.extend_from_slice(domain);

    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        // I2OSP(i, 8) — eight big-endian bytes.
        let i_bytes = (i as u64).to_be_bytes();
        let g = <G1Projective as HashToCurve<ExpandMsgXmd<sha2_v9::Sha256>>>::hash_to_curve(
            i_bytes, &dst,
        );
        out.push(g);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_same_inputs() {
        let a = message_generators(b"domain-1", 5);
        let b = message_generators(b"domain-1", 5);
        assert_eq!(a, b);
    }

    #[test]
    fn distinct_for_different_domains() {
        let a = message_generators(b"domain-1", 5);
        let b = message_generators(b"domain-2", 5);
        for (x, y) in a.iter().zip(b.iter()) {
            assert_ne!(x, y);
        }
    }

    #[test]
    fn distinct_indices_distinct_points() {
        let g = message_generators(b"d", 8);
        for i in 0..g.len() {
            for j in (i + 1)..g.len() {
                assert_ne!(g[i], g[j]);
            }
        }
    }

    #[test]
    fn no_generator_is_identity() {
        let g = message_generators(b"d", 12);
        for p in &g {
            assert!(!bool::from(p.is_identity()));
        }
    }

    #[test]
    fn count_zero_is_empty() {
        assert!(message_generators(b"d", 0).is_empty());
    }
}
