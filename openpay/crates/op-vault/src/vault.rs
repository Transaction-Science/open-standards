//! The [`Vault`] trait.
//!
//! ## What a Vault does
//!
//! - **`tokenize`**: accept [`CardData`], persist the encrypted PAN
//!   somewhere (Keychain, Keystore, HSM, KMS-encrypted blob in a DB,
//!   etc.), return an opaque [`VaultRef`] that the rest of the stack
//!   can pass around freely.
//! - **`detokenize`**: given a [`VaultRef`], return [`CardData`] to the
//!   rail driver at submit time. This is the only operation that
//!   reconstitutes PAN, and it must happen inside the vault's process
//!   boundary (or via mTLS to a remote vault service).
//! - **`delete`**: drop the mapping. Used when the customer removes a
//!   saved card or when retention policy expires.
//! - **`exists`**: probe without decrypting. Used by the orchestrator
//!   to validate a token before initiating a flow.
//!
//! ## What a Vault MUST NOT do
//!
//! - Log PAN in any form.
//! - Return PAN to anything other than a rail driver in a submit path.
//! - Allow the same token to be resolved if it was issued single-use.
//! - Allow tokens to outlive their TTL.
//! - Reveal *why* a detokenize attempt failed beyond the four error
//!   variants. Distinguishing "token unknown" from "auth failed" is an
//!   oracle.
//!
//! ## Threading
//!
//! `Vault: Send + Sync` so the orchestrator can share `Arc<dyn Vault>`
//! across worker threads. Implementations are expected to handle
//! concurrent calls; the in-memory reference uses a `Mutex` around the
//! ciphertext map.

use op_core::VaultRef;

use crate::card_data::CardData;
use crate::error::Result;
use crate::policy::TokenizationPolicy;

/// The vault interface. Implementations are operator-supplied except
/// for the in-memory reference shipped behind the `in-memory` feature.
pub trait Vault: Send + Sync {
    /// Vault name for telemetry and audit logs.
    fn name(&self) -> &str;

    /// Persist `card` and return an opaque token reference.
    ///
    /// The `policy` parameter is observed at issuance and stored
    /// alongside the encrypted PAN; expiration and single-use semantics
    /// are enforced at `detokenize` time.
    ///
    /// # Errors
    /// - [`Error::Capacity`] if the backend is full or rate-limited.
    /// - [`Error::Backend`] for any other implementation-specific failure.
    ///
    /// [`Error::Capacity`]: crate::Error::Capacity
    /// [`Error::Backend`]: crate::Error::Backend
    fn tokenize(&self, card: CardData, policy: TokenizationPolicy) -> Result<VaultRef>;

    /// Recover the card data for `token`. The rail driver calls this
    /// at submit time to send PAN to the PSP / acquirer / network.
    ///
    /// # Errors
    /// - [`Error::NotFound`] if the token is unknown (or the vault
    ///   chooses to mask another failure as not-found, per oracle
    ///   discipline).
    /// - [`Error::Expired`] if the token has aged past `ttl_seconds`.
    /// - [`Error::AlreadyConsumed`] if single-use and was already used.
    /// - [`Error::AuthFailed`] if decryption fails (tampering, wrong key).
    ///
    /// [`Error::NotFound`]: crate::Error::NotFound
    /// [`Error::Expired`]: crate::Error::Expired
    /// [`Error::AlreadyConsumed`]: crate::Error::AlreadyConsumed
    /// [`Error::AuthFailed`]: crate::Error::AuthFailed
    fn detokenize(&self, token: &VaultRef) -> Result<CardData>;

    /// Check whether a token exists without decrypting.
    ///
    /// # Errors
    /// - [`Error::Backend`] only. Existence checks do NOT distinguish
    ///   expired / consumed from "exists" — that's a detokenize concern.
    ///
    /// [`Error::Backend`]: crate::Error::Backend
    fn exists(&self, token: &VaultRef) -> Result<bool>;

    /// Permanently delete the mapping. Idempotent: deleting an unknown
    /// token returns `Ok(false)`, not an error.
    ///
    /// Returns `true` if a mapping was actually removed, `false` if
    /// the token wasn't present.
    ///
    /// # Errors
    /// - [`Error::Backend`] for implementation-specific failures.
    ///
    /// [`Error::Backend`]: crate::Error::Backend
    fn delete(&self, token: &VaultRef) -> Result<bool>;

    /// Vault-wide health check. Returns `Ok(())` if the backend is
    /// reachable and the encryption key is loaded. Used by the
    /// orchestrator's readiness probe.
    ///
    /// Default impl returns `Ok(())` — implementations that need to
    /// probe a remote service should override.
    ///
    /// # Errors
    /// - [`Error::Backend`] if the backend is unhealthy.
    ///
    /// [`Error::Backend`]: crate::Error::Backend
    fn health_check(&self) -> Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // We can't test the trait without an implementation. The
    // InMemoryVault test module covers behavioral conformance.
    // Here we verify dyn-compatibility.

    struct FakeVault;
    impl Vault for FakeVault {
        fn name(&self) -> &'static str {
            "fake"
        }
        fn tokenize(&self, _card: CardData, _p: TokenizationPolicy) -> Result<VaultRef> {
            Ok(VaultRef::new("fake_tok"))
        }
        fn detokenize(&self, _t: &VaultRef) -> Result<CardData> {
            Err(crate::Error::NotFound)
        }
        fn exists(&self, _t: &VaultRef) -> Result<bool> {
            Ok(false)
        }
        fn delete(&self, _t: &VaultRef) -> Result<bool> {
            Ok(false)
        }
    }

    #[test]
    fn vault_is_object_safe() {
        let v: Box<dyn Vault> = Box::new(FakeVault);
        assert_eq!(v.name(), "fake");
    }

    #[test]
    fn default_health_check_is_ok() {
        let v = FakeVault;
        assert!(v.health_check().is_ok());
    }
}
