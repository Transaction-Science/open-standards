//! Condition + fulfillment binding tests: round-trip, mismatch, and a
//! cross-check against a hand-computed SHA-256.

use sha2::{Digest, Sha256};
use smart_byte_ilp::{Condition, Fulfillment, Result};

#[test]
fn random_fulfillment_opens_its_condition() -> Result<()> {
    let f = Fulfillment::new([0x42; 32]);
    let c = f.condition();
    f.verify(&c)?;
    Ok(())
}

#[test]
fn mismatched_condition_rejected() {
    let f = Fulfillment::new([0x01; 32]);
    let wrong = Condition::new([0x02; 32]);
    assert!(f.verify(&wrong).is_err());
}

#[test]
fn condition_matches_external_sha256() {
    let preimage = [0xab; 32];
    let f = Fulfillment::new(preimage);
    let c = f.condition();
    let mut h = Sha256::new();
    h.update(preimage);
    let expected = h.finalize();
    assert_eq!(c.as_bytes(), expected.as_slice());
}

#[test]
fn fulfillment_bit_flip_invalidates() {
    let secret = Fulfillment::new([0xa5; 32]);
    let cond = secret.condition();
    let mut bad = secret.0;
    bad[0] ^= 0x01;
    let tampered = Fulfillment::new(bad);
    assert!(tampered.verify(&cond).is_err());
}
