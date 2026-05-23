//! [`InMemoryVault`] — reference implementation backed by AES-256-GCM-SIV.
//!
//! ## Status
//!
//! This is a **reference implementation**. It is correct, well-tested,
//! and exhibits the exact behavior the [`Vault`] trait promises. It is
//! NOT a PCI-compliant production vault on its own:
//!
//! - **State is in RAM only.** A restart loses every token. Real vaults
//!   persist ciphertext to durable storage (DB, HSM, KMS-encrypted file).
//! - **The encryption key is operator-supplied.** Real vaults integrate
//!   with KMS / HSM for key management with rotation, dual control,
//!   split knowledge, and FIPS 140-2 Level 2/3 modules.
//! - **No audit logging.** PCI DSS requires log emission for every
//!   detokenize call with user identity, timestamp, and outcome.
//! - **No rate limiting.** Real vaults bound detokenize throughput per
//!   caller to limit blast radius on credential compromise.
//!
//! Use this in tests, CI, and local development. For production deploy
//! a platform vault (iOS Keychain / Android Keystore / AWS KMS /
//! `HashiCorp` Vault / on-prem HSM) behind the same [`Vault`] trait.
//!
//! ## Cryptography
//!
//! AES-256-GCM-SIV per RFC 8452. Misuse-resistant: a nonce collision
//! doesn't catastrophically break confidentiality the way it does in
//! plain AES-GCM. Each ciphertext carries its own 96-bit random nonce
//! prepended to the bytes.
//!
//! ## Token format
//!
//! Tokens are UUID v7 strings prefixed with `tok_v7_`. UUID v7 carries
//! a timestamp prefix that makes tokens sortable by issuance time —
//! useful for vault analytics — without revealing the underlying PAN
//! mapping (the random suffix dominates).

use std::collections::HashMap;
use std::sync::Mutex;

use aes_gcm_siv::aead::{Aead, KeyInit, OsRng};
use aes_gcm_siv::{AeadCore, Aes256GcmSiv, Key, Nonce};
use op_core::VaultRef;
use rand_core::RngCore;
use time::OffsetDateTime;
use uuid::Uuid;
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::card_data::CardData;
use crate::error::{Error, Result};
use crate::policy::{TokenLifetime, TokenizationPolicy};
use crate::vault::Vault;

/// Token id prefix. Picked so tokens are visually distinguishable from
/// PANs (which are all-digit) — per PCI DSS Tokenization Guidelines
/// §3.3, tokens must not be confusable with their underlying PAN.
pub const TOKEN_PREFIX: &str = "tok_v7_";

/// Plaintext payload that gets encrypted into the vault.
///
/// JSON-serialized then encrypted. We use JSON for forward-compat
/// (adding a field doesn't break existing ciphertexts) at the cost of
/// some size overhead. The encryption itself doesn't care about the
/// payload format; this is a vault-internal choice.
#[derive(serde::Serialize, serde::Deserialize, Zeroize, ZeroizeOnDrop)]
struct StoredCard {
    pan: String,
    exp_month: u8,
    exp_year: u16,
}

/// Per-token metadata held alongside the ciphertext. Not encrypted —
/// these fields are used to enforce policy before attempting decryption.
struct Metadata {
    /// 96-bit nonce used to encrypt this entry.
    nonce: [u8; 12],
    /// Encrypted JSON of `StoredCard`.
    ciphertext: Vec<u8>,
    /// Original policy at issuance.
    policy: TokenizationPolicy,
    /// Unix-seconds at which the token was minted.
    issued_at: i64,
    /// Single-use tokens flip this on first successful detokenize.
    consumed: bool,
}

/// AES-256-GCM-SIV-backed reference vault.
pub struct InMemoryVault {
    name: String,
    cipher: Aes256GcmSiv,
    store: Mutex<HashMap<String, Metadata>>,
}

impl InMemoryVault {
    /// Construct with a fresh random key. Useful for tests and ephemeral
    /// development. The key is lost on drop; tokens issued by this
    /// vault are unrecoverable across instances.
    #[must_use]
    pub fn ephemeral(name: impl Into<String>) -> Self {
        let key = Aes256GcmSiv::generate_key(&mut OsRng);
        Self::with_key(name, &key)
    }

    /// Construct with an operator-supplied 32-byte key. The operator
    /// is responsible for key management, rotation, and persistence;
    /// the vault stores only ciphertext.
    #[must_use]
    pub fn with_key(name: impl Into<String>, key: &Key<Aes256GcmSiv>) -> Self {
        let cipher = Aes256GcmSiv::new(key);
        Self {
            name: name.into(),
            cipher,
            store: Mutex::new(HashMap::new()),
        }
    }

    /// Number of tokens currently held. Test-only utility.
    #[cfg(test)]
    pub(crate) fn len(&self) -> usize {
        self.store.lock().expect("poisoned").len()
    }
}

impl Vault for InMemoryVault {
    fn name(&self) -> &str {
        &self.name
    }

    fn tokenize(&self, card: CardData, policy: TokenizationPolicy) -> Result<VaultRef> {
        // Serialize the card payload.
        let stored = StoredCard {
            pan: core::str::from_utf8(card.raw().pan_bytes())
                .map_err(|_| Error::Backend("non-utf8 pan".into()))?
                .to_owned(),
            exp_month: card.exp_month(),
            exp_year: card.exp_year(),
        };
        let plaintext =
            serde_json::to_vec(&stored).map_err(|e| Error::Backend(format!("serialize: {e}")))?;

        // Random 96-bit nonce. GCM-SIV is misuse-resistant so the
        // catastrophic failure on accidental reuse doesn't apply, but
        // we still want fresh nonces per entry as a defense in depth.
        let nonce = Aes256GcmSiv::generate_nonce(&mut OsRng);

        let ciphertext = self
            .cipher
            .encrypt(&nonce, plaintext.as_ref())
            .map_err(|_| Error::Backend("encrypt".into()))?;

        // Mint a token id. UUID v7 carries a millisecond timestamp
        // prefix; the random suffix is what makes it cryptographically
        // independent from the PAN.
        let token_id = format!("{TOKEN_PREFIX}{}", Uuid::now_v7().simple());

        let issued_at = OffsetDateTime::now_utc().unix_timestamp();

        let mut nonce_arr = [0u8; 12];
        nonce_arr.copy_from_slice(nonce.as_slice());
        let metadata = Metadata {
            nonce: nonce_arr,
            ciphertext,
            policy,
            issued_at,
            consumed: false,
        };

        self.store
            .lock()
            .map_err(|_| Error::Backend("lock poisoned".into()))?
            .insert(token_id.clone(), metadata);

        Ok(VaultRef::new(token_id))
    }

    fn detokenize(&self, token: &VaultRef) -> Result<CardData> {
        let id = token.as_str();
        if !id.starts_with(TOKEN_PREFIX) {
            return Err(Error::InvalidToken);
        }

        let mut store = self
            .store
            .lock()
            .map_err(|_| Error::Backend("lock poisoned".into()))?;

        let metadata = store.get_mut(id).ok_or(Error::NotFound)?;

        // Check expiration.
        if let Some(ttl) = metadata.policy.ttl_seconds {
            let now = OffsetDateTime::now_utc().unix_timestamp();
            // `ttl` is u64, `now - issued_at` is i64. A negative age
            // (clock skew / backdated issue) can't be expired.
            let age = now.saturating_sub(metadata.issued_at);
            if u64::try_from(age).is_ok_and(|a| a >= ttl) {
                return Err(Error::Expired);
            }
        }

        // Check single-use.
        if metadata.policy.lifetime == TokenLifetime::SingleUse && metadata.consumed {
            return Err(Error::AlreadyConsumed);
        }

        // Decrypt.
        let nonce = Nonce::from_slice(&metadata.nonce);
        let plaintext = self
            .cipher
            .decrypt(nonce, metadata.ciphertext.as_ref())
            .map_err(|_| Error::AuthFailed)?;

        let stored: StoredCard = serde_json::from_slice(&plaintext)
            .map_err(|e| Error::Backend(format!("deserialize: {e}")))?;

        // Flip single-use marker AFTER successful decryption so a
        // failed attempt doesn't burn the token. Real vaults debate
        // both orderings; we choose forgiveness on the in-memory
        // reference.
        if metadata.policy.lifetime == TokenLifetime::SingleUse {
            metadata.consumed = true;
        }

        // Re-construct CardData by going through the validated
        // constructor — this means we re-run Luhn on detokenize, which
        // catches ciphertext tampering that somehow produced a valid
        // decryption but invalid PAN. Belt and braces.
        CardData::new(stored.pan.clone(), stored.exp_month, stored.exp_year)
    }

    fn exists(&self, token: &VaultRef) -> Result<bool> {
        let id = token.as_str();
        if !id.starts_with(TOKEN_PREFIX) {
            return Ok(false);
        }
        Ok(self
            .store
            .lock()
            .map_err(|_| Error::Backend("lock poisoned".into()))?
            .contains_key(id))
    }

    fn delete(&self, token: &VaultRef) -> Result<bool> {
        let id = token.as_str();
        if !id.starts_with(TOKEN_PREFIX) {
            return Ok(false);
        }
        Ok(self
            .store
            .lock()
            .map_err(|_| Error::Backend("lock poisoned".into()))?
            .remove(id)
            .is_some())
    }
}

/// Generate a fresh 32-byte key. Convenience helper; operators should
/// derive keys from KMS / HSM in production.
#[must_use]
pub fn generate_key() -> [u8; 32] {
    let mut key = [0u8; 32];
    OsRng.fill_bytes(&mut key);
    key
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_VISA: &str = "4242424242424242";
    const VALID_MC: &str = "5555555555554444";

    fn vault() -> InMemoryVault {
        InMemoryVault::ephemeral("test")
    }

    fn card() -> CardData {
        CardData::new(VALID_VISA, 12, 2030).unwrap()
    }

    #[test]
    fn tokenize_returns_well_formed_token() {
        let v = vault();
        let token = v.tokenize(card(), TokenizationPolicy::default()).unwrap();
        assert!(token.as_str().starts_with(TOKEN_PREFIX));
        assert!(token.as_str().len() > TOKEN_PREFIX.len() + 10);
    }

    #[test]
    fn tokenize_then_detokenize_round_trips() {
        let v = vault();
        let original = card();
        let token = v
            .tokenize(original.clone(), TokenizationPolicy::default())
            .unwrap();
        let recovered = v.detokenize(&token).unwrap();
        // Verify last4 / first6 match — never compare full PAN in test logs.
        assert_eq!(original.last_four(), recovered.last_four());
        assert_eq!(original.first_six(), recovered.first_six());
        assert_eq!(original.exp_month(), recovered.exp_month());
        assert_eq!(original.exp_year(), recovered.exp_year());
    }

    #[test]
    fn detokenize_unknown_token_returns_not_found() {
        let v = vault();
        let fake = VaultRef::new(format!("{TOKEN_PREFIX}{}", Uuid::now_v7().simple()));
        let err = v.detokenize(&fake).unwrap_err();
        assert!(matches!(err, Error::NotFound));
    }

    #[test]
    fn detokenize_malformed_token_returns_invalid() {
        let v = vault();
        let bad = VaultRef::new("4242424242424242"); // looks like a PAN
        let err = v.detokenize(&bad).unwrap_err();
        assert!(matches!(err, Error::InvalidToken));
    }

    #[test]
    fn delete_removes_mapping() {
        let v = vault();
        let token = v.tokenize(card(), TokenizationPolicy::default()).unwrap();
        assert!(v.exists(&token).unwrap());
        let removed = v.delete(&token).unwrap();
        assert!(removed);
        assert!(!v.exists(&token).unwrap());
        // Second delete is a no-op (idempotent).
        let removed = v.delete(&token).unwrap();
        assert!(!removed);
    }

    #[test]
    fn exists_returns_false_for_unknown() {
        let v = vault();
        let fake = VaultRef::new(format!("{TOKEN_PREFIX}deadbeef"));
        assert!(!v.exists(&fake).unwrap());
    }

    #[test]
    fn exists_returns_false_for_non_token_string() {
        let v = vault();
        let not_a_token = VaultRef::new("definitely-not-a-token");
        assert!(!v.exists(&not_a_token).unwrap());
    }

    #[test]
    fn single_use_token_consumed_on_first_detokenize() {
        let v = vault();
        let policy = TokenizationPolicy::single_use(60);
        let token = v.tokenize(card(), policy).unwrap();
        // First call succeeds.
        let _ = v.detokenize(&token).unwrap();
        // Second call fails.
        let err = v.detokenize(&token).unwrap_err();
        assert!(matches!(err, Error::AlreadyConsumed));
    }

    #[test]
    fn reusable_token_survives_multiple_detokenizes() {
        let v = vault();
        let token = v.tokenize(card(), TokenizationPolicy::default()).unwrap();
        for _ in 0..5 {
            assert!(v.detokenize(&token).is_ok());
        }
    }

    #[test]
    fn tokens_for_same_pan_are_unique_under_random_policy() {
        let v = vault();
        let policy = TokenizationPolicy::default();
        let a = v.tokenize(card(), policy).unwrap();
        let b = v.tokenize(card(), policy).unwrap();
        assert_ne!(
            a.as_str(),
            b.as_str(),
            "Random format must produce distinct tokens for identical PANs"
        );
    }

    #[test]
    fn different_cards_produce_different_tokens() {
        let v = vault();
        let c1 = CardData::new(VALID_VISA, 12, 2030).unwrap();
        let c2 = CardData::new(VALID_MC, 12, 2030).unwrap();
        let t1 = v.tokenize(c1, TokenizationPolicy::default()).unwrap();
        let t2 = v.tokenize(c2, TokenizationPolicy::default()).unwrap();
        assert_ne!(t1.as_str(), t2.as_str());
    }

    #[test]
    fn vault_with_different_key_cannot_decrypt() {
        let v1 = vault();
        let v2 = vault(); // different ephemeral key
        let token = v1.tokenize(card(), TokenizationPolicy::default()).unwrap();
        // v2 doesn't have this token at all (separate HashMap), so it's
        // NotFound rather than AuthFailed. This is the right behavior —
        // AuthFailed would leak existence.
        let err = v2.detokenize(&token).unwrap_err();
        assert!(matches!(err, Error::NotFound));
    }

    #[test]
    fn with_key_constructor_is_deterministic_per_key() {
        // Same key → same vault can decrypt tokens issued by the other.
        // We achieve this by sharing the key array.
        let key_bytes = generate_key();
        let key = Key::<Aes256GcmSiv>::from_slice(&key_bytes);

        let v1 = InMemoryVault::with_key("v1", key);
        // Issue a token, manually copy ciphertext+nonce to a fresh
        // vault with the same key.
        let token = v1.tokenize(card(), TokenizationPolicy::default()).unwrap();

        // Build a new vault with the same key and inject the metadata.
        let v2 = InMemoryVault::with_key("v2", key);
        {
            let mut s1 = v1.store.lock().unwrap();
            let mut s2 = v2.store.lock().unwrap();
            for (k, m) in s1.drain() {
                s2.insert(
                    k,
                    Metadata {
                        nonce: m.nonce,
                        ciphertext: m.ciphertext,
                        policy: m.policy,
                        issued_at: m.issued_at,
                        consumed: m.consumed,
                    },
                );
            }
        }
        // v2 can decrypt v1's token.
        let recovered = v2.detokenize(&token).unwrap();
        assert_eq!(recovered.last_four(), "4242");
    }

    #[test]
    fn vault_is_object_safe() {
        let v: Box<dyn Vault> = Box::new(vault());
        assert_eq!(v.name(), "test");
    }

    #[test]
    fn generate_key_produces_32_bytes_of_entropy() {
        let k1 = generate_key();
        let k2 = generate_key();
        assert_eq!(k1.len(), 32);
        assert_ne!(k1, k2, "two random keys must differ");
        // Sanity: not all zeros (extremely unlikely from CSPRNG).
        assert!(k1.iter().any(|&b| b != 0));
    }

    #[test]
    fn tokenize_len_increases_then_delete_decreases() {
        let v = vault();
        assert_eq!(v.len(), 0);
        let t1 = v.tokenize(card(), TokenizationPolicy::default()).unwrap();
        assert_eq!(v.len(), 1);
        let t2 = v.tokenize(card(), TokenizationPolicy::default()).unwrap();
        assert_eq!(v.len(), 2);
        v.delete(&t1).unwrap();
        assert_eq!(v.len(), 1);
        v.delete(&t2).unwrap();
        assert_eq!(v.len(), 0);
    }

    #[test]
    fn health_check_default_impl_returns_ok() {
        assert!(vault().health_check().is_ok());
    }

    #[test]
    fn token_prefix_makes_tokens_unlike_pans() {
        let v = vault();
        let token = v.tokenize(card(), TokenizationPolicy::default()).unwrap();
        // PCI Tokenization Guidelines §3.3: tokens must not be
        // confusable with PANs. Our prefix is non-numeric.
        assert!(
            token.as_str().chars().any(|c| !c.is_ascii_digit()),
            "token must contain non-digit characters to be distinguishable from PAN"
        );
    }

    #[test]
    fn ttl_expired_token_returns_expired() {
        // Tokenize with a 60-second TTL, then mutate the issued_at
        // timestamp backwards by 120 seconds to simulate the passage
        // of time without actually sleeping.
        let v = vault();
        let policy = TokenizationPolicy {
            ttl_seconds: Some(60),
            ..Default::default()
        };
        let token = v.tokenize(card(), policy).unwrap();

        // Rewind the issued_at to make this entry "old".
        {
            let mut store = v.store.lock().unwrap();
            let metadata = store.get_mut(token.as_str()).unwrap();
            metadata.issued_at -= 120;
        }

        let err = v.detokenize(&token).unwrap_err();
        assert!(matches!(err, Error::Expired));
    }

    #[test]
    fn ttl_zero_means_expired_immediately() {
        let v = vault();
        let policy = TokenizationPolicy {
            ttl_seconds: Some(0),
            ..Default::default()
        };
        let token = v.tokenize(card(), policy).unwrap();
        let err = v.detokenize(&token).unwrap_err();
        assert!(matches!(err, Error::Expired));
    }

    #[test]
    fn clock_skew_does_not_falsely_expire() {
        // If now < issued_at (clock went backwards), do NOT expire.
        let v = vault();
        let policy = TokenizationPolicy {
            ttl_seconds: Some(60),
            ..Default::default()
        };
        let token = v.tokenize(card(), policy).unwrap();

        // Push issued_at into the future.
        {
            let mut store = v.store.lock().unwrap();
            let metadata = store.get_mut(token.as_str()).unwrap();
            metadata.issued_at += 1000;
        }

        // Should NOT be expired (age is negative).
        assert!(v.detokenize(&token).is_ok());
    }
}
