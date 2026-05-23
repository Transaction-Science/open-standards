//! Signature-algorithm registry.
//!
//! The single byte returned by [`SignatureAlgorithm::algorithm_byte`]
//! is what the Smart Byte envelope places before the signature blob
//! (`§4.1 / §8.3 / §8.4`). v1 reserves `0x01` for Ed25519 and leaves
//! the remainder of the namespace for post-quantum successors. This
//! crate uses:
//!
//! | byte   | algorithm           |
//! |--------|---------------------|
//! | `0x01` | Ed25519             |
//! | `0x10` | ML-DSA-44 (FIPS 204) |
//! | `0x11` | ML-DSA-65 (FIPS 204) |
//! | `0x12` | ML-DSA-87 (FIPS 204) |
//! | `0x20` | SLH-DSA-SHA2-128s (FIPS 205) |
//! | `0x21` | SLH-DSA-SHA2-128f (FIPS 205) |
//! | `0x22` | SLH-DSA-SHA2-192s (FIPS 205) |
//! | `0x23` | SLH-DSA-SHA2-192f (FIPS 205) |
//! | `0x24` | SLH-DSA-SHA2-256s (FIPS 205) |
//! | `0x25` | SLH-DSA-SHA2-256f (FIPS 205) |
//! | `0x26` | SLH-DSA-SHAKE-128s (FIPS 205) |
//! | `0x27` | SLH-DSA-SHAKE-128f (FIPS 205) |
//! | `0x28` | SLH-DSA-SHAKE-192s (FIPS 205) |
//! | `0x29` | SLH-DSA-SHAKE-192f (FIPS 205) |
//! | `0x2A` | SLH-DSA-SHAKE-256s (FIPS 205) |
//! | `0x2B` | SLH-DSA-SHAKE-256f (FIPS 205) |
//! | `0x30` | FN-DSA-512 (draft FIPS 206)  |
//! | `0x31` | FN-DSA-1024 (draft FIPS 206) |
//!
//! The exact byte assignments above are local to this implementation;
//! the substrate-wide registry will pin them when ML-DSA enters
//! `§19.5` "Phase 2" status. Until then, the helpers in this module
//! are the single source of truth callers should consult.

use core::fmt;

use crate::error::{PqError, Result};

// ----- Length constants pulled directly from FIPS 204 / 205 / 206. -----
//
// These are the byte lengths produced by the PQClean reference
// implementations vendored by the pqcrypto crates; they are also the
// lengths fixed by the standards and the NIST submission packages.

// FIPS 204 (ML-DSA)
const MLDSA44_PK: usize = 1312;
const MLDSA44_SK: usize = 2560;
const MLDSA44_SIG: usize = 2420;

const MLDSA65_PK: usize = 1952;
const MLDSA65_SK: usize = 4032;
const MLDSA65_SIG: usize = 3309;

const MLDSA87_PK: usize = 2592;
const MLDSA87_SK: usize = 4896;
const MLDSA87_SIG: usize = 4627;

// FIPS 205 (SLH-DSA), "simple" instantiations, both SHA-2 and SHAKE
// share the same key/signature sizes per parameter set.
const SLHDSA_128S_PK: usize = 32;
const SLHDSA_128S_SK: usize = 64;
const SLHDSA_128S_SIG: usize = 7856;

const SLHDSA_128F_PK: usize = 32;
const SLHDSA_128F_SK: usize = 64;
const SLHDSA_128F_SIG: usize = 17088;

const SLHDSA_192S_PK: usize = 48;
const SLHDSA_192S_SK: usize = 96;
const SLHDSA_192S_SIG: usize = 16224;

const SLHDSA_192F_PK: usize = 48;
const SLHDSA_192F_SK: usize = 96;
const SLHDSA_192F_SIG: usize = 35664;

const SLHDSA_256S_PK: usize = 64;
const SLHDSA_256S_SK: usize = 128;
const SLHDSA_256S_SIG: usize = 29792;

const SLHDSA_256F_PK: usize = 64;
const SLHDSA_256F_SK: usize = 128;
const SLHDSA_256F_SIG: usize = 49856;

// Draft FIPS 206 (FN-DSA / Falcon). Falcon signatures are
// variable-length; the value below is the maximum allocated by the
// PQClean wrapper for the non-padded variant, which is what this
// crate exposes. Callers that need fixed-length encodings should
// re-pad downstream.
const FNDSA512_PK: usize = 897;
const FNDSA512_SK: usize = 1281;
const FNDSA512_SIG_MAX: usize = 752;

const FNDSA1024_PK: usize = 1793;
const FNDSA1024_SK: usize = 2305;
const FNDSA1024_SIG_MAX: usize = 1462;

// Ed25519 (RFC 8032).
const ED25519_PK: usize = 32;
const ED25519_SK: usize = 32;
const ED25519_SIG: usize = 64;

/// The complete set of signature algorithms Smart Byte understands.
///
/// The default for v1 envelopes is [`SignatureAlgorithm::Ed25519`];
/// the remaining variants are selectable per envelope via the
/// algorithm-identifier byte at the head of the signature field.
// The FIPS-205 naming convention uses an underscore between the hash
// family ("Shake") and the parameter ("128s") which trips the
// default non_camel_case_types lint; allow it for readability and
// alignment with the published specification.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[non_exhaustive]
pub enum SignatureAlgorithm {
    /// Edwards-curve digital signature using Curve25519 (RFC 8032).
    /// The v1 default. Quantum-vulnerable.
    Ed25519,

    /// FIPS 204 ML-DSA, parameter set 44 (NIST security category 2).
    MlDsa44,
    /// FIPS 204 ML-DSA, parameter set 65 (NIST security category 3).
    MlDsa65,
    /// FIPS 204 ML-DSA, parameter set 87 (NIST security category 5).
    MlDsa87,

    /// FIPS 205 SLH-DSA, SHA-2 family, 128-bit security, small/slow.
    SlhDsaSha2_128s,
    /// FIPS 205 SLH-DSA, SHA-2 family, 128-bit security, fast/large.
    SlhDsaSha2_128f,
    /// FIPS 205 SLH-DSA, SHA-2 family, 192-bit security, small/slow.
    SlhDsaSha2_192s,
    /// FIPS 205 SLH-DSA, SHA-2 family, 192-bit security, fast/large.
    SlhDsaSha2_192f,
    /// FIPS 205 SLH-DSA, SHA-2 family, 256-bit security, small/slow.
    SlhDsaSha2_256s,
    /// FIPS 205 SLH-DSA, SHA-2 family, 256-bit security, fast/large.
    SlhDsaSha2_256f,
    /// FIPS 205 SLH-DSA, SHAKE family, 128-bit security, small/slow.
    SlhDsaShake_128s,
    /// FIPS 205 SLH-DSA, SHAKE family, 128-bit security, fast/large.
    SlhDsaShake_128f,
    /// FIPS 205 SLH-DSA, SHAKE family, 192-bit security, small/slow.
    SlhDsaShake_192s,
    /// FIPS 205 SLH-DSA, SHAKE family, 192-bit security, fast/large.
    SlhDsaShake_192f,
    /// FIPS 205 SLH-DSA, SHAKE family, 256-bit security, small/slow.
    SlhDsaShake_256s,
    /// FIPS 205 SLH-DSA, SHAKE family, 256-bit security, fast/large.
    SlhDsaShake_256f,

    /// Draft FIPS 206 FN-DSA, parameter set 512 (NIST security
    /// category 1). Available only with the `falcon` cargo feature.
    FnDsa512,
    /// Draft FIPS 206 FN-DSA, parameter set 1024 (NIST security
    /// category 5). Available only with the `falcon` cargo feature.
    FnDsa1024,
}

impl SignatureAlgorithm {
    /// The byte that appears in the envelope's algorithm-identifier
    /// field (`§4.1 / §8.3` of the Smart Byte spec).
    #[must_use]
    pub const fn algorithm_byte(&self) -> u8 {
        match self {
            Self::Ed25519 => 0x01,
            Self::MlDsa44 => 0x10,
            Self::MlDsa65 => 0x11,
            Self::MlDsa87 => 0x12,
            Self::SlhDsaSha2_128s => 0x20,
            Self::SlhDsaSha2_128f => 0x21,
            Self::SlhDsaSha2_192s => 0x22,
            Self::SlhDsaSha2_192f => 0x23,
            Self::SlhDsaSha2_256s => 0x24,
            Self::SlhDsaSha2_256f => 0x25,
            Self::SlhDsaShake_128s => 0x26,
            Self::SlhDsaShake_128f => 0x27,
            Self::SlhDsaShake_192s => 0x28,
            Self::SlhDsaShake_192f => 0x29,
            Self::SlhDsaShake_256s => 0x2A,
            Self::SlhDsaShake_256f => 0x2B,
            Self::FnDsa512 => 0x30,
            Self::FnDsa1024 => 0x31,
        }
    }

    /// Recover a [`SignatureAlgorithm`] from its on-the-wire identifier
    /// byte. Returns [`PqError::UnsupportedAlgorithm`] for unknown bytes
    /// (callers may want to map this to `MalformedSignature` instead).
    pub fn from_algorithm_byte(byte: u8) -> Result<Self> {
        Ok(match byte {
            0x01 => Self::Ed25519,
            0x10 => Self::MlDsa44,
            0x11 => Self::MlDsa65,
            0x12 => Self::MlDsa87,
            0x20 => Self::SlhDsaSha2_128s,
            0x21 => Self::SlhDsaSha2_128f,
            0x22 => Self::SlhDsaSha2_192s,
            0x23 => Self::SlhDsaSha2_192f,
            0x24 => Self::SlhDsaSha2_256s,
            0x25 => Self::SlhDsaSha2_256f,
            0x26 => Self::SlhDsaShake_128s,
            0x27 => Self::SlhDsaShake_128f,
            0x28 => Self::SlhDsaShake_192s,
            0x29 => Self::SlhDsaShake_192f,
            0x2A => Self::SlhDsaShake_256s,
            0x2B => Self::SlhDsaShake_256f,
            0x30 => Self::FnDsa512,
            0x31 => Self::FnDsa1024,
            other => {
                // We borrow the FnDsa512 variant as a placeholder so we
                // can populate the error's argument, but the message is
                // self-evidently about an unknown byte.
                return Err(PqError::MalformedSignature(
                    Self::Ed25519,
                    alloc_format(other),
                ));
            }
        })
    }

    /// Maximum signature length (in bytes) for this algorithm. For
    /// FN-DSA the value is the worst-case length; the actual signature
    /// length is variable. For every other algorithm the value is
    /// exact.
    #[must_use]
    pub const fn signature_bytes_len(&self) -> usize {
        match self {
            Self::Ed25519 => ED25519_SIG,
            Self::MlDsa44 => MLDSA44_SIG,
            Self::MlDsa65 => MLDSA65_SIG,
            Self::MlDsa87 => MLDSA87_SIG,
            Self::SlhDsaSha2_128s | Self::SlhDsaShake_128s => SLHDSA_128S_SIG,
            Self::SlhDsaSha2_128f | Self::SlhDsaShake_128f => SLHDSA_128F_SIG,
            Self::SlhDsaSha2_192s | Self::SlhDsaShake_192s => SLHDSA_192S_SIG,
            Self::SlhDsaSha2_192f | Self::SlhDsaShake_192f => SLHDSA_192F_SIG,
            Self::SlhDsaSha2_256s | Self::SlhDsaShake_256s => SLHDSA_256S_SIG,
            Self::SlhDsaSha2_256f | Self::SlhDsaShake_256f => SLHDSA_256F_SIG,
            Self::FnDsa512 => FNDSA512_SIG_MAX,
            Self::FnDsa1024 => FNDSA1024_SIG_MAX,
        }
    }

    /// Public-key length in bytes.
    #[must_use]
    pub const fn public_key_bytes_len(&self) -> usize {
        match self {
            Self::Ed25519 => ED25519_PK,
            Self::MlDsa44 => MLDSA44_PK,
            Self::MlDsa65 => MLDSA65_PK,
            Self::MlDsa87 => MLDSA87_PK,
            Self::SlhDsaSha2_128s | Self::SlhDsaShake_128s => SLHDSA_128S_PK,
            Self::SlhDsaSha2_128f | Self::SlhDsaShake_128f => SLHDSA_128F_PK,
            Self::SlhDsaSha2_192s | Self::SlhDsaShake_192s => SLHDSA_192S_PK,
            Self::SlhDsaSha2_192f | Self::SlhDsaShake_192f => SLHDSA_192F_PK,
            Self::SlhDsaSha2_256s | Self::SlhDsaShake_256s => SLHDSA_256S_PK,
            Self::SlhDsaSha2_256f | Self::SlhDsaShake_256f => SLHDSA_256F_PK,
            Self::FnDsa512 => FNDSA512_PK,
            Self::FnDsa1024 => FNDSA1024_PK,
        }
    }

    /// Secret-key length in bytes (note: the raw seed/expanded form
    /// reported by the FIPS specs and the PQClean implementations).
    #[must_use]
    pub const fn secret_key_bytes_len(&self) -> usize {
        match self {
            Self::Ed25519 => ED25519_SK,
            Self::MlDsa44 => MLDSA44_SK,
            Self::MlDsa65 => MLDSA65_SK,
            Self::MlDsa87 => MLDSA87_SK,
            Self::SlhDsaSha2_128s | Self::SlhDsaShake_128s => SLHDSA_128S_SK,
            Self::SlhDsaSha2_128f | Self::SlhDsaShake_128f => SLHDSA_128F_SK,
            Self::SlhDsaSha2_192s | Self::SlhDsaShake_192s => SLHDSA_192S_SK,
            Self::SlhDsaSha2_192f | Self::SlhDsaShake_192f => SLHDSA_192F_SK,
            Self::SlhDsaSha2_256s | Self::SlhDsaShake_256s => SLHDSA_256S_SK,
            Self::SlhDsaSha2_256f | Self::SlhDsaShake_256f => SLHDSA_256F_SK,
            Self::FnDsa512 => FNDSA512_SK,
            Self::FnDsa1024 => FNDSA1024_SK,
        }
    }
}

impl fmt::Display for SignatureAlgorithm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let name = match self {
            Self::Ed25519 => "Ed25519",
            Self::MlDsa44 => "ML-DSA-44",
            Self::MlDsa65 => "ML-DSA-65",
            Self::MlDsa87 => "ML-DSA-87",
            Self::SlhDsaSha2_128s => "SLH-DSA-SHA2-128s",
            Self::SlhDsaSha2_128f => "SLH-DSA-SHA2-128f",
            Self::SlhDsaSha2_192s => "SLH-DSA-SHA2-192s",
            Self::SlhDsaSha2_192f => "SLH-DSA-SHA2-192f",
            Self::SlhDsaSha2_256s => "SLH-DSA-SHA2-256s",
            Self::SlhDsaSha2_256f => "SLH-DSA-SHA2-256f",
            Self::SlhDsaShake_128s => "SLH-DSA-SHAKE-128s",
            Self::SlhDsaShake_128f => "SLH-DSA-SHAKE-128f",
            Self::SlhDsaShake_192s => "SLH-DSA-SHAKE-192s",
            Self::SlhDsaShake_192f => "SLH-DSA-SHAKE-192f",
            Self::SlhDsaShake_256s => "SLH-DSA-SHAKE-256s",
            Self::SlhDsaShake_256f => "SLH-DSA-SHAKE-256f",
            Self::FnDsa512 => "FN-DSA-512",
            Self::FnDsa1024 => "FN-DSA-1024",
        };
        f.write_str(name)
    }
}

// Small helper to build a "0xNN" description without pulling in
// std::fmt where unnecessary.
fn alloc_format(b: u8) -> String {
    format!("unknown algorithm identifier byte 0x{b:02X}")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn algorithm_byte_round_trips() {
        let all = [
            SignatureAlgorithm::Ed25519,
            SignatureAlgorithm::MlDsa44,
            SignatureAlgorithm::MlDsa65,
            SignatureAlgorithm::MlDsa87,
            SignatureAlgorithm::SlhDsaSha2_128s,
            SignatureAlgorithm::SlhDsaSha2_128f,
            SignatureAlgorithm::SlhDsaSha2_192s,
            SignatureAlgorithm::SlhDsaSha2_192f,
            SignatureAlgorithm::SlhDsaSha2_256s,
            SignatureAlgorithm::SlhDsaSha2_256f,
            SignatureAlgorithm::SlhDsaShake_128s,
            SignatureAlgorithm::SlhDsaShake_128f,
            SignatureAlgorithm::SlhDsaShake_192s,
            SignatureAlgorithm::SlhDsaShake_192f,
            SignatureAlgorithm::SlhDsaShake_256s,
            SignatureAlgorithm::SlhDsaShake_256f,
            SignatureAlgorithm::FnDsa512,
            SignatureAlgorithm::FnDsa1024,
        ];
        for a in all {
            let byte = a.algorithm_byte();
            let recovered = SignatureAlgorithm::from_algorithm_byte(byte)
                .expect("known byte should round-trip");
            assert_eq!(a, recovered, "round-trip mismatch for {a}");
        }
    }

    #[test]
    fn unknown_algorithm_byte_errors() {
        let err = SignatureAlgorithm::from_algorithm_byte(0xFF)
            .expect_err("0xFF must not be assigned in v1");
        match err {
            PqError::MalformedSignature(_, msg) => {
                assert!(msg.contains("0xFF"), "error should name the bad byte");
            }
            other => panic!("wrong error variant: {other:?}"),
        }
    }

    #[test]
    fn fips_lengths_match_published_values() {
        // ML-DSA spot check (FIPS 204).
        assert_eq!(SignatureAlgorithm::MlDsa44.public_key_bytes_len(), 1312);
        assert_eq!(SignatureAlgorithm::MlDsa65.signature_bytes_len(), 3309);
        assert_eq!(SignatureAlgorithm::MlDsa87.secret_key_bytes_len(), 4896);

        // SLH-DSA spot check (FIPS 205).
        assert_eq!(SignatureAlgorithm::SlhDsaSha2_128s.public_key_bytes_len(), 32);
        assert_eq!(SignatureAlgorithm::SlhDsaSha2_256f.signature_bytes_len(), 49856);
    }
}
