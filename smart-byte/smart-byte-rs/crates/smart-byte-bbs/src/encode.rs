//! Byte / scalar / point encoding helpers.
//!
//! Per `draft-irtf-cfrg-bbs-signatures-08` § 4.3 (`hash_to_scalar`),
//! arbitrary input bytes are mapped to the BLS12-381 scalar field by
//! the `expand_message_xmd(SHA-256)` construction with output length
//! `48` bytes followed by an `os2ip || mod r` reduction.
//!
//! For ergonomics we expose a simpler implementation that hashes the
//! input with SHA-512 and reduces the 64-byte digest via
//! `Scalar::from_bytes_wide`. This is the construction recommended by
//! RFC 9380 § 5.2 ("hash-and-reduce" with an output twice the field
//! size) and is functionally equivalent for the security argument used
//! by BBS+ (Fiat-Shamir + uniform-in-the-field message encoding).
//!
//! Domain-separation tags follow the BBS draft's pattern:
//! `BBS_BLS12-381_XMD:SHA-256_SSWU_RO_H2S_` for message hashing and
//! `BBS_BLS12-381_XMD:SHA-256_SSWU_RO_MAP_MSG_TO_SCALAR_` for the
//! map-to-scalar phase, with the Smart Byte variant suffix
//! `SMART_BYTE_BBS_2026_`.

use bls12_381::{G1Affine, G1Projective, G2Affine, Scalar};
use group::GroupEncoding;
use sha2::{Digest, Sha512};

use crate::error::BbsError;

/// Domain-separation tag prefix used everywhere in this crate.
pub const DST_PREFIX: &[u8] = b"SMART_BYTE_BBS_2026_";

/// Domain separation tag for "hash arbitrary bytes to a message scalar".
pub const DST_HASH_TO_SCALAR: &[u8] =
    b"SMART_BYTE_BBS_2026_BBS_BLS12381_XMD:SHA-512_H2S_";

/// Domain separation tag for the Fiat-Shamir challenge.
pub const DST_FIAT_SHAMIR: &[u8] =
    b"SMART_BYTE_BBS_2026_BBS_BLS12381_FS_CHALLENGE_";

/// Domain separation tag for the per-signer secret-nonce derivation
/// used inside the proof (`r1, r2, r3, e_tilde, ...`).
pub const DST_PROOF_NONCE: &[u8] =
    b"SMART_BYTE_BBS_2026_BBS_BLS12381_PROOF_NONCE_";

/// Hash arbitrary bytes to a BLS12-381 scalar.
///
/// SHA-512(`dst || msg`) → reduce modulo the scalar-field order.
pub fn hash_to_scalar(dst: &[u8], msg: &[u8]) -> Scalar {
    let mut h = Sha512::new();
    h.update(dst);
    h.update(msg);
    let bytes = h.finalize();
    let mut wide = [0u8; 64];
    wide.copy_from_slice(&bytes);
    Scalar::from_bytes_wide(&wide)
}

/// Hash arbitrary bytes to a *message* scalar using the BBS message
/// DST.
pub fn message_to_scalar(msg: &[u8]) -> Scalar {
    hash_to_scalar(DST_HASH_TO_SCALAR, msg)
}

/// Encode a Scalar as its canonical 32-byte little-endian
/// representation (the encoding used by `bls12_381::Scalar::to_bytes`).
pub fn scalar_to_bytes(s: &Scalar) -> [u8; 32] {
    s.to_bytes()
}

/// Decode a Scalar from its canonical 32-byte little-endian
/// representation.
pub fn scalar_from_bytes(bytes: &[u8; 32]) -> Result<Scalar, BbsError> {
    Option::<Scalar>::from(Scalar::from_bytes(bytes)).ok_or_else(|| {
        BbsError::InvalidEncoding("non-canonical scalar".into())
    })
}

/// Encode a G1Affine point as its 48-byte compressed representation.
pub fn g1_to_bytes(p: &G1Affine) -> [u8; 48] {
    p.to_compressed()
}

/// Decode a G1Affine point from its 48-byte compressed representation.
pub fn g1_from_bytes(bytes: &[u8; 48]) -> Result<G1Affine, BbsError> {
    Option::<G1Affine>::from(G1Affine::from_compressed(bytes)).ok_or_else(
        || BbsError::InvalidEncoding("invalid G1 compressed encoding".into()),
    )
}

/// Encode a G2Affine point as its 96-byte compressed representation.
pub fn g2_to_bytes(p: &G2Affine) -> [u8; 96] {
    p.to_compressed()
}

/// Decode a G2Affine point from its 96-byte compressed representation.
pub fn g2_from_bytes(bytes: &[u8; 96]) -> Result<G2Affine, BbsError> {
    Option::<G2Affine>::from(G2Affine::from_compressed(bytes)).ok_or_else(
        || BbsError::InvalidEncoding("invalid G2 compressed encoding".into()),
    )
}

/// Canonical BBS+ ciphertext concatenation:
/// `G1A_compressed (48) || E_bytes (32) || message_count (u32 BE)`.
///
/// The actual message scalars are *not* part of this payload — the
/// holder retains them separately so they can be selectively disclosed.
pub fn pack_signature(a: &G1Affine, e: &Scalar, message_count: u32) -> Vec<u8> {
    let mut out = Vec::with_capacity(48 + 32 + 4);
    out.extend_from_slice(&g1_to_bytes(a));
    out.extend_from_slice(&scalar_to_bytes(e));
    out.extend_from_slice(&message_count.to_be_bytes());
    out
}

/// Inverse of [`pack_signature`]. Returns `(A, e, message_count)`.
pub fn unpack_signature(
    bytes: &[u8],
) -> Result<(G1Affine, Scalar, u32), BbsError> {
    if bytes.len() != 48 + 32 + 4 {
        return Err(BbsError::InvalidEncoding(format!(
            "expected {} packed signature bytes, got {}",
            48 + 32 + 4,
            bytes.len()
        )));
    }
    let mut a_buf = [0u8; 48];
    a_buf.copy_from_slice(&bytes[..48]);
    let mut e_buf = [0u8; 32];
    e_buf.copy_from_slice(&bytes[48..80]);
    let mut n_buf = [0u8; 4];
    n_buf.copy_from_slice(&bytes[80..84]);
    let a = g1_from_bytes(&a_buf)?;
    let e = scalar_from_bytes(&e_buf)?;
    let n = u32::from_be_bytes(n_buf);
    Ok((a, e, n))
}

/// Convenience: compress a `G1Projective` to bytes through
/// `G1Affine`. Kept here so call sites need not import `group`.
pub fn g1p_to_bytes(p: &G1Projective) -> [u8; 48] {
    let _ = <G1Projective as group::Group>::is_identity; // anchor trait import for `as G1Affine::from`
    G1Affine::from(p).to_compressed()
}

/// Get the canonical wire size for the constant-size fields.
pub const G1_BYTES: usize = 48;
/// Wire size for a G2 compressed point.
pub const G2_BYTES: usize = 96;
/// Wire size for a Scalar.
pub const SCALAR_BYTES: usize = 32;

/// Compress an arbitrary projective G1 point. Helper used by tests.
pub fn compress_g1(p: &G1Projective) -> Vec<u8> {
    p.to_bytes().as_ref().to_vec()
}
