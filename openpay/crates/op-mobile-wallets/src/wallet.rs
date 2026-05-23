//! Cross-wallet abstractions.
//!
//! The three wallets share a common downstream shape: each emits an
//! encrypted payload that, once decrypted, yields a normalized
//! [`DecryptedToken`] with a DPAN-as-[`VaultRef`], an expiration, the
//! transaction amount, and a per-transaction cryptogram.
//!
//! Operators usually want one configured decryptor per wallet they
//! accept. The [`decryptor_for`] factory returns a boxed trait object
//! the merchant orchestrator can dispatch through without
//! pattern-matching on the wallet kind at every site.

use op_core::{Currency, Money, VaultRef};
use p256::SecretKey;
use p256::ecdsa::VerifyingKey;
use serde::{Deserialize, Serialize};

use crate::apple_pay::{
    ApplePayConfig, ApplePayDataType, ApplePayDecryptor, ApplePayToken, CertChain, ExpirationDate,
};
use crate::cryptogram::EciIndicator;
use crate::error::{Error, Result};
use crate::google_pay::{
    GooglePayConfig, GooglePayDecryptor, GooglePayProtocolVersion, GooglePayToken,
};
use crate::samsung_pay::{SamsungPayConfig, SamsungPayDecryptor, SamsungPayToken};

/// Marker type for base64-encoded payload bytes coming off the wire.
///
/// Wallets ship their encrypted payloads as base64 strings inside a
/// JSON envelope. We wrap that string in a newtype so it cannot be
/// confused with already-decoded bytes at type-check time.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Base64Encoded(pub String);

impl Base64Encoded {
    /// Construct from a base64 string.
    #[must_use]
    pub fn new(s: impl Into<String>) -> Self {
        Self(s.into())
    }

    /// Borrow the base64-encoded string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Wallet metadata extracted from the *outer* (cleartext) envelope.
///
/// These are the non-secret hints the wallet attaches alongside the
/// encrypted payload — display network, last-4, optional cardholder
/// name — that the merchant UI can surface before authorization.
/// Crucially, these never include the underlying PAN; they describe
/// only the device-PAN that will become a [`VaultRef`] after
/// decryption.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PaymentMethodInfo {
    /// Network short id (`"visa"`, `"mc"`, `"amex"`, `"discover"`).
    pub network: String,
    /// Last four digits of the device-PAN, safe to log per PCI 4.0.1.
    pub last4: String,
    /// Optional cardholder display name as surfaced by the wallet.
    /// Apple Pay populates this when the cardholder has a wallet
    /// display name; Google Pay leaves it `None`.
    pub display_name: Option<String>,
}

/// The normalized output of any wallet decryption.
///
/// **PAN-zero invariant.** The credential reference is
/// [`VaultRef`] — never a `String`, `&str`, `Vec<u8>`, or
/// `&[u8]`. The cleartext device-PAN exists transiently inside the
/// decryption boundary; it is encoded into the `VaultRef` via the
/// operator-supplied [`VaultTokenizer`] and then zeroized. No public
/// field of this struct lets a caller observe the cleartext DPAN
/// outside the decryption boundary.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DecryptedToken {
    /// Vault-bound reference to the device-PAN. Resolving to a usable
    /// credential happens inside the PCI-scoped vault service.
    pub application_primary_account_number: VaultRef,

    /// Device-PAN expiration as `YYYYMM` (Apple Pay format) or the
    /// equivalent normalized form. Bounded by [`ExpirationDate`]'s
    /// range checks at construction.
    pub application_expiration_date: ExpirationDate,

    /// Transaction currency.
    pub currency_code: Currency,

    /// Transaction amount in the currency's minor units.
    pub transaction_amount: Money,

    /// Apple Pay's `paymentDataType` (`"3DSecure"` / `"EMV"`). For
    /// Google Pay and Samsung Pay we synthesize this from the
    /// payload's authentication-mode flag so all three backends share
    /// one shape.
    pub payment_data_type: ApplePayDataType,

    /// Raw cryptogram bytes (the wallet emits this verbatim; the
    /// network-token rail re-encodes to base64 when it forwards). This
    /// is NOT a credential — it is a per-transaction one-shot token
    /// that is useless without the matching DPAN, and `Vec<u8>`
    /// surfaces are appropriate for it.
    pub online_payment_cryptogram: Vec<u8>,

    /// Electronic Commerce Indicator surfaced by the wallet.
    pub eci_indicator: EciIndicator,
}

/// Tokenizer hook: maps the cleartext device-PAN into a [`VaultRef`].
///
/// This is the operator's bridge into their PCI-scoped vault. The
/// closure receives the cleartext DPAN as a borrowed byte slice
/// (which exists only inside the decryption boundary, in a
/// `zeroize`-on-drop buffer) and must return the opaque vault
/// reference that resolves back to it.
///
/// The DPAN slice is **not** retained by this crate after the
/// closure returns. A correct tokenizer copies into its vault
/// service synchronously and returns a stable id; an incorrect
/// tokenizer that retains the slice's contents elsewhere creates a
/// PCI-scope leak in the *operator's* code, not ours.
pub trait VaultTokenizer: Send + Sync {
    /// Tokenize the cleartext device-PAN into a vault reference.
    ///
    /// Implementations should copy `dpan` into their vault service and
    /// return a stable opaque id. If the vault is unavailable, return
    /// an [`Error::VaultTokenizer`] with a human-readable diagnostic.
    fn tokenize(&self, dpan: &[u8]) -> Result<VaultRef>;
}

/// Identity tokenizer used by tests and the conformance harness.
///
/// **Not for production.** The identity tokenizer base64-encodes the
/// DPAN as the vault id. That defeats the PCI-zero posture — the
/// vault id encodes the credential. This exists so a test can show
/// that the API surface alone keeps PAN material out, even when the
/// tokenizer is trivially reversible. Production code MUST wire a
/// real vault tokenizer that calls into the operator's vault.
pub struct IdentityVaultTokenizer;

impl VaultTokenizer for IdentityVaultTokenizer {
    fn tokenize(&self, dpan: &[u8]) -> Result<VaultRef> {
        use base64::Engine as _;
        let b64 = base64::engine::general_purpose::STANDARD.encode(dpan);
        Ok(VaultRef::new(format!("identity:{b64}")))
    }
}

/// Hashing tokenizer used by the no-PAN-leakage tests.
///
/// Returns `VaultRef("sha256:<hex>")`. Demonstrates that the public
/// API surface does not contain the cleartext DPAN even when the
/// tokenizer's output is a deterministic, irreversible function of
/// the DPAN.
pub struct Sha256VaultTokenizer;

impl VaultTokenizer for Sha256VaultTokenizer {
    fn tokenize(&self, dpan: &[u8]) -> Result<VaultRef> {
        use core::fmt::Write as _;
        use sha2::{Digest, Sha256};
        let mut h = Sha256::new();
        h.update(dpan);
        let digest = h.finalize();
        let mut hex = String::with_capacity(2 * digest.len());
        for byte in digest {
            let _ = write!(hex, "{byte:02x}");
        }
        Ok(VaultRef::new(format!("sha256:{hex}")))
    }
}

/// The three supported wallets.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Wallet {
    /// Apple Pay (PaymentToken docs, ECDH P-256 + X9.63-KDF + AES-GCM).
    ApplePay,
    /// Google Pay (tokenization spec, ECDH P-256 + HKDF + AES-CTR + HMAC).
    GooglePay,
    /// Samsung Pay (Apple-Pay-shaped, ECDH P-256 + HKDF + AES-GCM).
    SamsungPay,
}

/// Unified config bundle covering all three wallets.
///
/// Operators populate the variant matching the wallets they accept
/// and leave the rest `None`. The [`decryptor_for`] factory pulls the
/// relevant variant.
pub struct WalletConfig {
    /// Apple Pay configuration. Required for `Wallet::ApplePay`.
    pub apple_pay: Option<ApplePayConfig>,
    /// Google Pay configuration. Required for `Wallet::GooglePay`.
    pub google_pay: Option<GooglePayConfig>,
    /// Samsung Pay configuration. Required for `Wallet::SamsungPay`.
    pub samsung_pay: Option<SamsungPayConfig>,
    /// Operator-provided tokenizer used by every backend at the
    /// decryption boundary to convert cleartext DPAN into a
    /// [`VaultRef`]. Stored as an `Arc<dyn ...>` so a single
    /// tokenizer can serve every wallet.
    pub vault_tokenizer: std::sync::Arc<dyn VaultTokenizer>,
}

/// Decryption interface implemented by each backend.
pub trait WalletDecryptor: Send + Sync {
    /// Decrypt the raw wire payload (the bytes the merchant frontend
    /// received from the wallet SDK).
    ///
    /// The expected format depends on the backend:
    /// - Apple Pay: a UTF-8 JSON serialization of [`ApplePayToken`].
    /// - Google Pay: a UTF-8 JSON serialization of [`GooglePayToken`].
    /// - Samsung Pay: a UTF-8 JSON serialization of [`SamsungPayToken`].
    ///
    /// Backends parse this bytes-blob themselves so a merchant
    /// orchestrator can dispatch through `dyn WalletDecryptor`
    /// without owning a per-wallet token type.
    fn decrypt(&self, raw: &[u8]) -> Result<DecryptedToken>;

    /// The wallet kind this decryptor handles.
    fn wallet(&self) -> Wallet;
}

/// Factory: returns a boxed decryptor for the requested wallet.
///
/// # Errors
///
/// Returns [`Error::Internal`] if the requested wallet's config slot
/// in [`WalletConfig`] is `None`.
pub fn decryptor_for(wallet: Wallet, config: &WalletConfig) -> Result<Box<dyn WalletDecryptor>> {
    match wallet {
        Wallet::ApplePay => {
            let cfg = config
                .apple_pay
                .as_ref()
                .ok_or(Error::Internal("missing ApplePayConfig"))?;
            Ok(Box::new(ApplePayDecryptor::new(
                cfg.clone(),
                config.vault_tokenizer.clone(),
            )))
        }
        Wallet::GooglePay => {
            let cfg = config
                .google_pay
                .as_ref()
                .ok_or(Error::Internal("missing GooglePayConfig"))?;
            Ok(Box::new(GooglePayDecryptor::new(
                cfg.clone(),
                config.vault_tokenizer.clone(),
            )))
        }
        Wallet::SamsungPay => {
            let cfg = config
                .samsung_pay
                .as_ref()
                .ok_or(Error::Internal("missing SamsungPayConfig"))?;
            Ok(Box::new(SamsungPayDecryptor::new(
                cfg.clone(),
                config.vault_tokenizer.clone(),
            )))
        }
    }
}

// Re-export forwarding helpers so a caller doesn't need to import
// from the per-backend modules to call the factory.
pub use op_core::VaultRef as _PanRefAlias;
// Suppress unused warnings on imports the factory pulls in.
use core::marker::PhantomData;
#[doc(hidden)]
pub struct _Unused(PhantomData<(SecretKey, VerifyingKey, ApplePayToken, GooglePayToken, SamsungPayToken, CertChain, GooglePayProtocolVersion)>);
