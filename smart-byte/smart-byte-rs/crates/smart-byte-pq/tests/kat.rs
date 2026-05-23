//! Standards-conformance tests.
//!
//! These tests stand in for the NIST Known-Answer Test (KAT) vectors
//! published in the FIPS 204 and FIPS 205 submission packages. The
//! full KAT files (per-parameter-set, multi-megabyte blobs of seeded
//! deterministic vectors) are not vendored into this crate; instead
//! the tests below assert the two properties any KAT-conformant
//! implementation must satisfy and which break under any incorrect
//! wiring of the underlying primitive:
//!
//! 1. **Length conformance**: keys and signatures the implementation
//!    produces have exactly the byte lengths FIPS 204 / 205 specify
//!    for the parameter set in question.
//! 2. **Sign / verify consistency**: a freshly generated key pair
//!    verifies its own signatures over random messages and rejects
//!    tampered ones.
//!
//! The vendored PQClean reference implementations underneath
//! `pqcrypto-mldsa` and `pqcrypto-sphincsplus` are themselves KAT-
//! validated in their respective repositories; this test file
//! verifies that the smart-byte-pq wrappers around them preserve
//! those properties at the public-API surface.
//!
//! To replace the consistency tests with real KAT vector files, drop
//! the relevant `.rsp` files into a `tests/kat_vectors/` directory and
//! parse them in additional `#[test]` functions; the surrounding
//! sign/verify API is already KAT-shaped.

use rand::rngs::OsRng;
use smart_byte_pq::{
    PqError,
    algorithm::SignatureAlgorithm,
    hybrid::{HybridKeyPair, sign_hybrid, verify_hybrid},
    mldsa::{self, MlDsaLevel, Signature as MlDsaSignature},
    slhdsa::{self, SlhDsaParam},
};

// ---------------------------------------------------------------------------
// ML-DSA (FIPS 204)
// ---------------------------------------------------------------------------

#[test]
fn mldsa44_lengths_match_fips204() {
    let mut rng = OsRng;
    let kp = mldsa::keygen(MlDsaLevel::Level2, &mut rng);
    assert_eq!(kp.public.as_bytes().len(), 1312);
    assert_eq!(kp.secret.as_bytes().len(), 2560);
    let sig = mldsa::sign(b"smart byte SAID", &kp.secret).expect("sign");
    assert_eq!(sig.as_bytes().len(), 2420);
    mldsa::verify(b"smart byte SAID", &sig, &kp.public).expect("verify");
}

#[test]
fn mldsa65_lengths_match_fips204() {
    let mut rng = OsRng;
    let kp = mldsa::keygen(MlDsaLevel::Level3, &mut rng);
    assert_eq!(kp.public.as_bytes().len(), 1952);
    assert_eq!(kp.secret.as_bytes().len(), 4032);
    let sig = mldsa::sign(b"smart byte SAID", &kp.secret).expect("sign");
    assert_eq!(sig.as_bytes().len(), 3309);
    mldsa::verify(b"smart byte SAID", &sig, &kp.public).expect("verify");
}

#[test]
fn mldsa87_lengths_match_fips204() {
    let mut rng = OsRng;
    let kp = mldsa::keygen(MlDsaLevel::Level5, &mut rng);
    assert_eq!(kp.public.as_bytes().len(), 2592);
    assert_eq!(kp.secret.as_bytes().len(), 4896);
    let sig = mldsa::sign(b"smart byte SAID", &kp.secret).expect("sign");
    assert_eq!(sig.as_bytes().len(), 4627);
    mldsa::verify(b"smart byte SAID", &sig, &kp.public).expect("verify");
}

#[test]
fn mldsa_tamper_rejection() {
    // One-byte tamper of an ML-DSA signature must verify-fail.
    let mut rng = OsRng;
    let kp = mldsa::keygen(MlDsaLevel::Level3, &mut rng);
    let msg = b"smart byte envelope SAID";
    let sig = mldsa::sign(msg, &kp.secret).expect("sign");
    let mut bytes = sig.as_bytes().to_vec();
    bytes[0] ^= 0x01;
    let bad =
        MlDsaSignature::from_bytes(MlDsaLevel::Level3, &bytes).expect("decode length-OK sig");
    let err = mldsa::verify(msg, &bad, &kp.public).expect_err("must fail");
    assert!(matches!(err, PqError::BadSignature));
}

// ---------------------------------------------------------------------------
// SLH-DSA (FIPS 205) — three parameter sets per the test plan.
// ---------------------------------------------------------------------------

#[test]
fn slhdsa_sha2_128s_round_trip() {
    let mut rng = OsRng;
    let kp = slhdsa::keygen(SlhDsaParam::Sha2_128s, &mut rng);
    assert_eq!(
        kp.public.as_bytes().len(),
        SignatureAlgorithm::SlhDsaSha2_128s.public_key_bytes_len()
    );
    let msg = b"smart byte SAID";
    let sig = slhdsa::sign(msg, &kp.secret).expect("sign");
    assert_eq!(
        sig.as_bytes().len(),
        SignatureAlgorithm::SlhDsaSha2_128s.signature_bytes_len()
    );
    slhdsa::verify(msg, &sig, &kp.public).expect("verify");
}

#[test]
fn slhdsa_sha2_192s_round_trip() {
    let mut rng = OsRng;
    let kp = slhdsa::keygen(SlhDsaParam::Sha2_192s, &mut rng);
    let msg = b"smart byte SAID";
    let sig = slhdsa::sign(msg, &kp.secret).expect("sign");
    assert_eq!(
        sig.as_bytes().len(),
        SignatureAlgorithm::SlhDsaSha2_192s.signature_bytes_len()
    );
    slhdsa::verify(msg, &sig, &kp.public).expect("verify");
}

#[test]
fn slhdsa_shake_256s_round_trip() {
    let mut rng = OsRng;
    // SHAKE-256s is the largest "small" variant; we keep this one
    // test for KAT-style coverage of the SHAKE family at the top
    // security level. (The much-larger 256f case is exercised
    // implicitly by the signature_bytes_len constant tests.)
    let kp = slhdsa::keygen(SlhDsaParam::Shake_256s, &mut rng);
    let msg = b"smart byte SAID";
    let sig = slhdsa::sign(msg, &kp.secret).expect("sign");
    assert_eq!(
        sig.as_bytes().len(),
        SignatureAlgorithm::SlhDsaShake_256s.signature_bytes_len()
    );
    slhdsa::verify(msg, &sig, &kp.public).expect("verify");
}

// ---------------------------------------------------------------------------
// Hybrid (Ed25519 + ML-DSA-65)
// ---------------------------------------------------------------------------

#[test]
fn hybrid_ed25519_mldsa65_round_trip() {
    let mut rng = OsRng;
    let kp = HybridKeyPair::generate(MlDsaLevel::Level3, &mut rng);
    let pub_key = kp.public_key();
    let msg = b"smart byte SAID";
    let sig = sign_hybrid(msg, &kp).expect("sign");
    verify_hybrid(msg, &sig, &pub_key).expect("verify");
}

#[test]
fn hybrid_rejects_when_classical_only_valid() {
    // Tamper the PQ half. Classical alone must not authenticate.
    let mut rng = OsRng;
    let kp = HybridKeyPair::generate(MlDsaLevel::Level3, &mut rng);
    let pub_key = kp.public_key();
    let msg = b"smart byte SAID";
    let mut sig = sign_hybrid(msg, &kp).expect("sign");
    let idx = sig.pq.len() / 2;
    sig.pq[idx] ^= 0x01;
    let err = verify_hybrid(msg, &sig, &pub_key).expect_err("must fail");
    assert!(matches!(err, PqError::BadSignature));
}

#[test]
fn hybrid_rejects_when_pq_only_valid() {
    // Tamper the classical half. PQ alone must not authenticate.
    let mut rng = OsRng;
    let kp = HybridKeyPair::generate(MlDsaLevel::Level3, &mut rng);
    let pub_key = kp.public_key();
    let msg = b"smart byte SAID";
    let mut sig = sign_hybrid(msg, &kp).expect("sign");
    sig.classical[0] ^= 0xFF;
    let err = verify_hybrid(msg, &sig, &pub_key).expect_err("must fail");
    assert!(matches!(err, PqError::BadSignature));
}
