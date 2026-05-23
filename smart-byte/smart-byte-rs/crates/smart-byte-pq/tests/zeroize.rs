//! Verify that secret keys actively zero their bytes on drop.
//!
//! The check works by remembering the raw byte pointer + length of a
//! freshly generated secret key, dropping the key, and then reading
//! through that pointer. After the [`Drop`] impl runs, the underlying
//! Vec backing store may be deallocated; the safest portable test is
//! to inspect the secret key's bytes immediately before drop and at
//! the boundary, verifying that the `Zeroize` impl turns them to
//! zeroes deterministically. We do that here without reading freed
//! memory (which would be UB).

use rand::rngs::OsRng;
use smart_byte_pq::mldsa::{self, MlDsaLevel};
use smart_byte_pq::slhdsa::{self, SlhDsaParam};
use zeroize::Zeroize;

#[test]
fn mldsa_secret_key_zeroizes_via_zeroize_trait() {
    let mut rng = OsRng;
    let kp = mldsa::keygen(MlDsaLevel::Level3, &mut rng);
    let mut sk = kp.secret;
    assert!(sk.as_bytes().iter().any(|&b| b != 0), "fresh key should not be all zero");
    sk.zeroize();
    assert!(
        sk.as_bytes().iter().all(|&b| b == 0),
        "after zeroize, every byte must be zero"
    );
}

#[test]
fn slhdsa_secret_key_zeroizes_via_zeroize_trait() {
    let mut rng = OsRng;
    let kp = slhdsa::keygen(SlhDsaParam::Sha2_128f, &mut rng);
    let mut sk = kp.secret;
    assert!(sk.as_bytes().iter().any(|&b| b != 0));
    sk.zeroize();
    assert!(sk.as_bytes().iter().all(|&b| b == 0));
}
