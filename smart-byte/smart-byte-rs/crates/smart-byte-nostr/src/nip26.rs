//! NIP-26 delegated event signing.
//!
//! Delegation token is a Schnorr signature over the SHA-256 of
//! `nostr:delegation:<delegatee pubkey hex>:<conditions string>`,
//! produced by the delegator. The delegated event carries a
//! `["delegation", <delegator pubkey hex>, <conditions>, <token hex>]`
//! tag.

use crate::error::NostrError;
use crate::event::Event;
use crate::keys::{NostrPublicKey, NostrSecretKey, hex_decode, hex_encode, schnorr_sign, schnorr_verify};
use sha2::{Digest, Sha256};

/// Conditions string describing what the delegatee may sign.
///
/// NIP-26 conditions are a `&`-separated list of `key=value` pairs.
/// Common keys: `kind`, `created_at<` (max), `created_at>` (min).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DelegationConditions(pub String);

impl DelegationConditions {
    /// Parse a conditions string. We do not enforce semantic
    /// well-formedness here — the spec leaves conditions intentionally
    /// extensible.
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrow the underlying string.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Compute the digest a delegator must sign to issue a delegation.
pub fn delegation_digest(delegatee: &NostrPublicKey, conditions: &DelegationConditions) -> [u8; 32] {
    let s = format!(
        "nostr:delegation:{}:{}",
        delegatee.to_hex(),
        conditions.as_str()
    );
    let mut h = Sha256::new();
    h.update(s.as_bytes());
    let out = h.finalize();
    let mut id = [0u8; 32];
    id.copy_from_slice(&out);
    id
}

/// Issue a delegation token: delegator signs the digest.
pub fn create_delegation(
    delegator_sk: &NostrSecretKey,
    delegatee: &NostrPublicKey,
    conditions: &DelegationConditions,
) -> Result<String, NostrError> {
    let digest = delegation_digest(delegatee, conditions);
    let sig = schnorr_sign(delegator_sk, &digest)?;
    Ok(hex_encode(&sig))
}

/// Verify a `["delegation", ...]` tag on a signed event.
///
/// Returns the delegator pubkey on success. Does NOT enforce the
/// conditions string semantically (callers can layer that on top by
/// parsing [`DelegationConditions::as_str`]).
pub fn verify_delegation_tag(event: &Event) -> Result<NostrPublicKey, NostrError> {
    let tag = event
        .tags
        .iter()
        .find(|t| t.first().map(|s| s.as_str()) == Some("delegation"))
        .ok_or_else(|| NostrError::InvalidDelegation("no delegation tag".into()))?;
    if tag.len() < 4 {
        return Err(NostrError::InvalidDelegation("malformed tag".into()));
    }
    let delegator_hex = &tag[1];
    let conditions = DelegationConditions::new(tag[2].clone());
    let token_hex = &tag[3];

    let delegator = NostrPublicKey::from_hex(delegator_hex)?;
    let delegatee = event.public_key()?;
    let digest = delegation_digest(&delegatee, &conditions);
    let sig_bytes = hex_decode(token_hex)?;
    if sig_bytes.len() != 64 {
        return Err(NostrError::InvalidDelegation("token not 64 bytes".into()));
    }
    let mut sig = [0u8; 64];
    sig.copy_from_slice(&sig_bytes);
    schnorr_verify(&delegator, &digest, &sig)
        .map_err(|_| NostrError::InvalidDelegation("bad signature".into()))?;
    Ok(delegator)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::UnsignedEvent;

    #[test]
    fn delegation_create_then_verify() {
        let delegator = NostrSecretKey::generate();
        let delegatee = NostrSecretKey::generate();
        let cond = DelegationConditions::new("kind=1&created_at<2000000000");
        let token = create_delegation(&delegator, &delegatee.public_key(), &cond).expect("token");

        let event = UnsignedEvent::new(delegatee.public_key(), 1, "hi", 1_700_000_000)
            .with_tag(vec![
                "delegation".into(),
                delegator.public_key().to_hex(),
                cond.0.clone(),
                token,
            ])
            .sign(&delegatee)
            .expect("sign");

        let recovered = verify_delegation_tag(&event).expect("verify");
        assert_eq!(recovered.to_hex(), delegator.public_key().to_hex());
    }
}
