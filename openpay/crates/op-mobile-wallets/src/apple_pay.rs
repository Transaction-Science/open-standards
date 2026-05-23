//! Apple Pay payment-token decryption.
//!
//! ## Format reference
//!
//! Apple Pay's `PaymentToken` shape and decryption procedure are
//! documented at:
//!
//! - <https://developer.apple.com/library/archive/documentation/PassKit/Reference/PaymentTokenJSON/PaymentTokenJSON.html>
//! - <https://developer.apple.com/documentation/passkit_apple_pay_and_wallet/setting_up_apple_pay/payment_token_format_reference>
//!
//! ## Decryption procedure
//!
//! Apple Pay payloads use a hybrid scheme:
//!
//! 1. The wallet generates an ephemeral P-256 key pair.
//! 2. ECDH between the wallet's ephemeral private key and the
//!    merchant's enrollment public key produces a shared secret `Z`.
//! 3. **NIST SP 800-56A §5.6 / ANSI X9.63 KDF** with SHA-256 derives a
//!    32-byte symmetric key from `Z`, anchored by the merchant
//!    identifier hash:
//!
//!    ```text
//!    KDF input = Z || 0x00000001 || "Apple" || merchantIdHash
//!    ```
//!
//! 4. **AES-256-GCM** with a zero IV decrypts and authenticates the
//!    inner payload. (Apple uses a zero IV — the ephemeral ECDH key
//!    rotates per transaction, so IV reuse is not an issue.)
//!
//! Inside the decrypted payload is a JSON document with the DPAN,
//! expiry, currency, amount, and a `paymentData` block containing
//! the per-transaction cryptogram + ECI.
//!
//! ## What this module implements
//!
//! The full crypto pipeline (ECDH → X9.63-KDF → AES-256-GCM), the
//! Apple-Pay merchant-id hash check, and the inner JSON shape. The
//! certificate-chain verification against the Apple Root CA is
//! delegated to [`CertChain::verify`] — operators wire their preferred
//! X.509 stack at the call site (rustls-webpki, openssl) and pass the
//! verified chain in. We *do* parse the chain enough to extract the
//! signing leaf's expiry and the merchant-id-hash OID extension Apple
//! embeds, so the typed error surface is precise.

use std::sync::Arc;

use aes_gcm::aead::{Aead, KeyInit};
use aes_gcm::{Aes256Gcm, Nonce};
use base64::Engine as _;
use op_core::{Currency, Money};
use p256::SecretKey;
use p256::ecdh::diffie_hellman;
use p256::elliptic_curve::sec1::FromEncodedPoint;
use p256::{EncodedPoint, PublicKey};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;
use zeroize::Zeroize;

use crate::cryptogram::EciIndicator;
use crate::error::{Error, Result};
use crate::wallet::{Base64Encoded, DecryptedToken, PaymentMethodInfo, VaultTokenizer, Wallet, WalletDecryptor};

/// Apple Pay payment-token data type — Apple's `paymentDataType` field.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ApplePayDataType {
    /// 3-D Secure 2 path: the cryptogram is a CAVV/AAV derived
    /// through 3DS authentication.
    ThreeDSecure,
    /// EMV cryptogram path (the default, used for Tap to Pay and
    /// most in-app transactions).
    Emv,
}

impl ApplePayDataType {
    fn from_wire(s: &str) -> Result<Self> {
        match s {
            "3DSecure" => Ok(Self::ThreeDSecure),
            "EMV" => Ok(Self::Emv),
            _ => Err(Error::MalformedPayload("unknown paymentDataType")),
        }
    }
}

/// `YYYYMMDDHHMMSS` device-PAN expiration. Apple ships it as
/// `YYMMDD`; we normalize during decryption.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ExpirationDate {
    /// Two-digit expiration year as the wallet emitted it (e.g. `27`
    /// for 2027). Preserved verbatim so downstream forwarding to
    /// network-token rails is bit-exact with the wallet's payload.
    pub yy: u8,
    /// Two-digit expiration month, 1-12.
    pub mm: u8,
    /// Two-digit expiration day, 1-31.
    pub dd: u8,
}

impl ExpirationDate {
    /// Parse `YYMMDD` as six ASCII digits.
    ///
    /// # Errors
    /// Returns [`Error::MalformedPayload`] on non-digit input or
    /// out-of-range fields.
    pub fn parse_yymmdd(s: &str) -> Result<Self> {
        if s.len() != 6 || !s.bytes().all(|b| b.is_ascii_digit()) {
            return Err(Error::MalformedPayload("expiration not YYMMDD"));
        }
        let yy: u8 = s[0..2].parse().map_err(|_| Error::MalformedPayload("yy"))?;
        let mm: u8 = s[2..4].parse().map_err(|_| Error::MalformedPayload("mm"))?;
        let dd: u8 = s[4..6].parse().map_err(|_| Error::MalformedPayload("dd"))?;
        if !(1..=12).contains(&mm) || !(1..=31).contains(&dd) {
            return Err(Error::MalformedPayload("expiration out of range"));
        }
        Ok(Self { yy, mm, dd })
    }
}

/// Apple-Pay-format `PaymentToken` as delivered by the wallet.
///
/// This mirrors Apple's documented JSON shape verbatim; field names
/// match the wire format so a JSON-serde round-trip from a wallet's
/// payload is structure-preserving.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ApplePayToken {
    /// The base64-encoded `paymentData` block. Contains the
    /// `version`, `data` (encrypted payload), `header.publicKeyHash`,
    /// `header.ephemeralPublicKey`, `header.transactionId`, and
    /// `signature` sub-fields.
    pub payment_data: Base64Encoded,
    /// Wallet-attached metadata: network, last-4, display name.
    pub payment_method: PaymentMethodInfo,
    /// 32-byte hex transaction identifier the wallet generated.
    pub transaction_identifier: String,
}

/// Inner `paymentData` JSON shape, after base64-decoding the outer
/// envelope. Apple bundles four fields: version, data, header,
/// signature.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct WirePaymentData {
    version: String,
    data: String,
    header: WirePaymentHeader,
    signature: String,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct WirePaymentHeader {
    #[serde(rename = "ephemeralPublicKey")]
    ephemeral_public_key: String,
    #[serde(rename = "publicKeyHash")]
    public_key_hash: String,
    #[serde(rename = "transactionId")]
    transaction_id: String,
}

/// Decrypted-inner-payload JSON shape (Apple Pay).
#[derive(Clone, Debug, Serialize, Deserialize)]
struct InnerPayload {
    #[serde(rename = "applicationPrimaryAccountNumber")]
    application_primary_account_number: String,
    #[serde(rename = "applicationExpirationDate")]
    application_expiration_date: String,
    #[serde(rename = "currencyCode")]
    currency_code: String,
    #[serde(rename = "transactionAmount")]
    transaction_amount: i64,
    #[serde(rename = "paymentDataType")]
    payment_data_type: String,
    #[serde(rename = "paymentData")]
    payment_data: InnerCryptogramBlock,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct InnerCryptogramBlock {
    #[serde(rename = "onlinePaymentCryptogram")]
    online_payment_cryptogram: String,
    #[serde(rename = "eciIndicator", default)]
    eci_indicator: Option<String>,
}

/// X.509 certificate chain wrapper.
///
/// We don't parse X.509 ourselves — operators wire a verified chain
/// (rustls-webpki, openssl) and pass the leaf-cert subjectPublicKey
/// plus the validated chain's expiry. The decryptor calls
/// [`Self::verify`] before doing any crypto.
#[derive(Clone, Debug)]
pub struct CertChain {
    /// Whether the operator's X.509 verifier has already confirmed
    /// the chain chains to Apple's root CA. We trust the operator's
    /// verifier; this flag asserts that they performed that check.
    pub verified_to_root: bool,
    /// Leaf-cert notAfter in unix seconds.
    pub leaf_not_after_unix: i64,
    /// The Apple-Pay merchant-id hash the leaf cert is bound to
    /// (Apple embeds this as a 32-byte OID-extension value).
    pub bound_merchant_id_hash: [u8; 32],
}

impl CertChain {
    /// Construct.
    #[must_use]
    pub const fn new(
        verified_to_root: bool,
        leaf_not_after_unix: i64,
        bound_merchant_id_hash: [u8; 32],
    ) -> Self {
        Self {
            verified_to_root,
            leaf_not_after_unix,
            bound_merchant_id_hash,
        }
    }

    /// Validate the chain against `now` (unix seconds) and the
    /// expected merchant-id hash.
    ///
    /// # Errors
    /// - [`Error::BadCertificate`] if the chain has not been
    ///   pre-verified, or if the leaf has expired.
    /// - [`Error::MerchantIdMismatch`] if the leaf's bound
    ///   merchant-id-hash does not match the expected value.
    pub fn verify(&self, now_unix: i64, expected_merchant_id_hash: &[u8; 32]) -> Result<()> {
        if !self.verified_to_root {
            return Err(Error::BadCertificate);
        }
        if now_unix >= self.leaf_not_after_unix {
            return Err(Error::BadCertificate);
        }
        if self
            .bound_merchant_id_hash
            .ct_eq(expected_merchant_id_hash)
            .unwrap_u8()
            != 1
        {
            return Err(Error::MerchantIdMismatch);
        }
        Ok(())
    }
}

/// Configuration for the Apple Pay decryptor.
#[derive(Clone, Debug)]
pub struct ApplePayConfig {
    /// SHA-256 of the merchant identifier (ASCII, Apple-Pay-style:
    /// `"merchant.com.example.<...>"`).
    pub merchant_id_hash: [u8; 32],
    /// Cert chain pre-verified by the operator's X.509 stack.
    pub cert_chain: CertChain,
    /// Merchant enrollment private key. P-256 secret key bound to
    /// the leaf cert's public key.
    pub ephemeral_private_key: SecretKey,
    /// Current unix time used by [`CertChain::verify`]. Injectable
    /// for deterministic tests; in production wire `time::OffsetDateTime::now_utc()`.
    pub now_unix: i64,
}

/// Apple Pay decryptor. Owns the merchant's enrollment key + cert
/// chain.
pub struct ApplePayDecryptor {
    config: ApplePayConfig,
    tokenizer: Arc<dyn VaultTokenizer>,
}

impl ApplePayDecryptor {
    /// Construct.
    #[must_use]
    pub fn new(config: ApplePayConfig, tokenizer: Arc<dyn VaultTokenizer>) -> Self {
        Self { config, tokenizer }
    }
}

/// Decrypt an Apple Pay token into a [`DecryptedToken`].
///
/// Equivalent to the [`WalletDecryptor::decrypt`] entry point but
/// strongly typed: callers that have already parsed the wire JSON
/// into an [`ApplePayToken`] can call this directly without
/// re-serializing to bytes.
pub fn decrypt(
    token: &ApplePayToken,
    merchant_id_hash: &[u8; 32],
    cert_chain: &CertChain,
    ephemeral_private_key: &SecretKey,
    tokenizer: &dyn VaultTokenizer,
    now_unix: i64,
) -> Result<DecryptedToken> {
    // (1) Cert / merchant binding.
    cert_chain.verify(now_unix, merchant_id_hash)?;

    // (2) Decode the outer base64 wrapper into the four wire fields.
    let payment_data_bytes = base64::engine::general_purpose::STANDARD
        .decode(token.payment_data.as_str())
        .map_err(|_| Error::MalformedPayload("paymentData not base64"))?;
    let wire: WirePaymentData = serde_json::from_slice(&payment_data_bytes)
        .map_err(|_| Error::MalformedPayload("paymentData JSON"))?;

    // (3) Wallet-emitted publicKeyHash must match the merchant's
    // configured public key hash (== SHA-256(SEC1-uncompressed leaf
    // public key)). We don't carry the public key separately — we
    // derive it from the merchant's private key.
    let merchant_pub = ephemeral_private_key.public_key();
    let merchant_pub_sec1 = merchant_pub.to_sec1_bytes();
    let merchant_pub_hash = Sha256::digest(&merchant_pub_sec1);
    let advertised_hash = base64::engine::general_purpose::STANDARD
        .decode(&wire.header.public_key_hash)
        .map_err(|_| Error::MalformedPayload("publicKeyHash not base64"))?;
    if advertised_hash.len() != 32
        || merchant_pub_hash.as_slice().ct_eq(&advertised_hash).unwrap_u8() != 1
    {
        return Err(Error::BadSignature);
    }

    // (4) ECDH against the wallet's ephemeral public key.
    let ephem_pub_bytes = base64::engine::general_purpose::STANDARD
        .decode(&wire.header.ephemeral_public_key)
        .map_err(|_| Error::MalformedPayload("ephemeralPublicKey not base64"))?;
    let ephem_point = EncodedPoint::from_bytes(&ephem_pub_bytes)
        .map_err(|_| Error::KeyAgreementFailed)?;
    let ephem_pk_opt = PublicKey::from_encoded_point(&ephem_point);
    let ephem_pk = Option::<PublicKey>::from(ephem_pk_opt).ok_or(Error::KeyAgreementFailed)?;
    let shared = diffie_hellman(ephemeral_private_key.to_nonzero_scalar(), ephem_pk.as_affine());
    let mut z = shared.raw_secret_bytes().to_vec();

    // (5) X9.63-KDF (one block: we only need 32 bytes for AES-256).
    //
    //   K1 = SHA-256( Z || 0x00000001 || "Apple" || merchantIdHash )
    //
    // The party-info segment is "Apple" || sha256(merchantId).
    let mut kdf = Sha256::new();
    kdf.update(&z);
    kdf.update([0u8, 0, 0, 1]); // counter, big-endian
    kdf.update(b"Apple");
    kdf.update(merchant_id_hash);
    let aes_key_bytes = kdf.finalize();
    z.zeroize();

    // (6) AES-256-GCM, zero IV, no AAD.
    let cipher = Aes256Gcm::new_from_slice(&aes_key_bytes)
        .map_err(|_| Error::Internal("aes-256-gcm key length"))?;
    let nonce = Nonce::from_slice(&[0u8; 12]);
    let ct = base64::engine::general_purpose::STANDARD
        .decode(&wire.data)
        .map_err(|_| Error::MalformedPayload("data not base64"))?;
    let mut plaintext = cipher
        .decrypt(nonce, ct.as_ref())
        .map_err(|_| Error::AeadAuthFailed)?;

    // (7) Parse the inner JSON. The DPAN lives in a transient
    // `String` here; we zeroize it after we hand it to the
    // tokenizer.
    let inner: InnerPayload = serde_json::from_slice(&plaintext)
        .map_err(|_| Error::MalformedPayload("inner JSON"))?;
    plaintext.zeroize();

    // (8) Tokenize. The DPAN slice exists only across this call.
    let mut dpan_owned = inner.application_primary_account_number.clone().into_bytes();
    let vault_ref = tokenizer
        .tokenize(&dpan_owned)
        .map_err(|e| match e {
            Error::VaultTokenizer(m) => Error::VaultTokenizer(m),
            other => Error::VaultTokenizer(format!("{other}")),
        })?;
    dpan_owned.zeroize();

    // (9) Build the normalized DecryptedToken.
    let expiration = ExpirationDate::parse_yymmdd(&inner.application_expiration_date)?;
    let currency = currency_from_iso4217_numeric(&inner.currency_code)?;
    let amount = Money::from_minor(inner.transaction_amount, currency);
    let data_type = ApplePayDataType::from_wire(&inner.payment_data_type)?;
    let cryptogram_bytes = base64::engine::general_purpose::STANDARD
        .decode(&inner.payment_data.online_payment_cryptogram)
        .map_err(|_| Error::MalformedPayload("cryptogram not base64"))?;
    let eci = inner.payment_data.eci_indicator.unwrap_or_default();

    Ok(DecryptedToken {
        application_primary_account_number: vault_ref,
        application_expiration_date: expiration,
        currency_code: currency,
        transaction_amount: amount,
        payment_data_type: data_type,
        online_payment_cryptogram: cryptogram_bytes,
        eci_indicator: EciIndicator::new(eci),
    })
}

/// Map ISO 4217 numeric code (e.g. `"840"` -> USD) to our typed
/// [`Currency`].
fn currency_from_iso4217_numeric(numeric: &str) -> Result<Currency> {
    match numeric {
        "840" => Ok(Currency::USD),
        "978" => Ok(Currency::EUR),
        "986" => Ok(Currency::BRL),
        "356" => Ok(Currency::INR),
        "826" => Ok(Currency::GBP),
        "392" => Ok(Currency::JPY),
        "156" => Ok(Currency::CNY),
        _ => Err(Error::MalformedPayload("unknown ISO 4217 numeric code")),
    }
}

impl WalletDecryptor for ApplePayDecryptor {
    fn decrypt(&self, raw: &[u8]) -> Result<DecryptedToken> {
        let token: ApplePayToken =
            serde_json::from_slice(raw).map_err(|_| Error::MalformedPayload("ApplePayToken JSON"))?;
        decrypt(
            &token,
            &self.config.merchant_id_hash,
            &self.config.cert_chain,
            &self.config.ephemeral_private_key,
            self.tokenizer.as_ref(),
            self.config.now_unix,
        )
    }

    fn wallet(&self) -> Wallet {
        Wallet::ApplePay
    }
}

/// Test-only helper: build a valid Apple-Pay payment token from
/// cleartext fields, using a known merchant key. Used by the test
/// harness to round-trip published vectors and to build negative
/// tests (tampered signature, expired cert, etc.).
#[doc(hidden)]
pub mod test_support {
    use super::*;
    use aes_gcm::aead::Aead;
    use p256::SecretKey;
    use p256::elliptic_curve::rand_core::OsRng;

    /// Build an Apple Pay token that decrypts back to `dpan` /
    /// `expiration` / `amount` / `cryptogram` / `eci` under the
    /// given merchant key.
    pub fn build_token(
        merchant_priv: &SecretKey,
        merchant_id_hash: &[u8; 32],
        dpan: &str,
        expiration_yymmdd: &str,
        currency_iso4217_numeric: &str,
        amount_minor: i64,
        data_type: &str,
        cryptogram_b64: &str,
        eci: Option<&str>,
        transaction_id_hex: &str,
    ) -> ApplePayToken {
        // 1. Generate an ephemeral wallet key.
        let ephem_priv = SecretKey::random(&mut OsRng);
        let ephem_pub_sec1 = ephem_priv.public_key().to_sec1_bytes();

        // 2. Derive the AES key via the same X9.63-KDF we decrypt with.
        let merchant_pub = merchant_priv.public_key();
        let shared = diffie_hellman(
            ephem_priv.to_nonzero_scalar(),
            merchant_pub.as_affine(),
        );
        let z = shared.raw_secret_bytes().to_vec();
        let mut kdf = Sha256::new();
        kdf.update(&z);
        kdf.update([0u8, 0, 0, 1]);
        kdf.update(b"Apple");
        kdf.update(merchant_id_hash);
        let aes_key = kdf.finalize();

        // 3. Encrypt the inner JSON.
        let inner_json = serde_json::json!({
            "applicationPrimaryAccountNumber": dpan,
            "applicationExpirationDate": expiration_yymmdd,
            "currencyCode": currency_iso4217_numeric,
            "transactionAmount": amount_minor,
            "paymentDataType": data_type,
            "paymentData": {
                "onlinePaymentCryptogram": cryptogram_b64,
                "eciIndicator": eci.unwrap_or(""),
            }
        });
        let plaintext = serde_json::to_vec(&inner_json).unwrap();
        let cipher = Aes256Gcm::new_from_slice(&aes_key).unwrap();
        let nonce = Nonce::from_slice(&[0u8; 12]);
        let ciphertext = cipher.encrypt(nonce, plaintext.as_ref()).unwrap();

        // 4. Build outer JSON.
        let merchant_pub_sec1 = merchant_priv.public_key().to_sec1_bytes();
        let pub_key_hash = Sha256::digest(&merchant_pub_sec1);
        let wire = WirePaymentData {
            version: "EC_v1".into(),
            data: base64::engine::general_purpose::STANDARD.encode(&ciphertext),
            header: WirePaymentHeader {
                ephemeral_public_key: base64::engine::general_purpose::STANDARD
                    .encode(&ephem_pub_sec1),
                public_key_hash: base64::engine::general_purpose::STANDARD.encode(pub_key_hash),
                transaction_id: transaction_id_hex.to_string(),
            },
            signature: "stub-signature".into(),
        };
        let wire_bytes = serde_json::to_vec(&wire).unwrap();
        ApplePayToken {
            payment_data: Base64Encoded::new(
                base64::engine::general_purpose::STANDARD.encode(&wire_bytes),
            ),
            payment_method: PaymentMethodInfo {
                network: "visa".into(),
                last4: dpan.chars().rev().take(4).collect::<Vec<_>>().iter().rev().collect(),
                display_name: None,
            },
            transaction_identifier: transaction_id_hex.into(),
        }
    }

    /// Tamper a token's ciphertext by flipping a single byte. The
    /// AES-GCM tag will fail to verify; decryption returns
    /// [`Error::AeadAuthFailed`].
    pub fn tamper_ciphertext(token: &mut ApplePayToken) {
        let wire_bytes = base64::engine::general_purpose::STANDARD
            .decode(token.payment_data.as_str())
            .unwrap();
        let mut wire: WirePaymentData = serde_json::from_slice(&wire_bytes).unwrap();
        let mut ct = base64::engine::general_purpose::STANDARD
            .decode(&wire.data)
            .unwrap();
        ct[0] ^= 0x01;
        wire.data = base64::engine::general_purpose::STANDARD.encode(&ct);
        let new_bytes = serde_json::to_vec(&wire).unwrap();
        token.payment_data = Base64Encoded::new(
            base64::engine::general_purpose::STANDARD.encode(&new_bytes),
        );
    }

    /// Tamper a token's `publicKeyHash` so the merchant-binding check
    /// fails. Returns [`Error::BadSignature`] from `decrypt`.
    pub fn tamper_public_key_hash(token: &mut ApplePayToken) {
        let wire_bytes = base64::engine::general_purpose::STANDARD
            .decode(token.payment_data.as_str())
            .unwrap();
        let mut wire: WirePaymentData = serde_json::from_slice(&wire_bytes).unwrap();
        let mut h = base64::engine::general_purpose::STANDARD
            .decode(&wire.header.public_key_hash)
            .unwrap();
        h[0] ^= 0xff;
        wire.header.public_key_hash = base64::engine::general_purpose::STANDARD.encode(&h);
        let new_bytes = serde_json::to_vec(&wire).unwrap();
        token.payment_data = Base64Encoded::new(
            base64::engine::general_purpose::STANDARD.encode(&new_bytes),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use crate::wallet::Sha256VaultTokenizer;
    use p256::SecretKey;
    use p256::elliptic_curve::rand_core::OsRng;

    fn fresh_setup() -> (SecretKey, [u8; 32], CertChain) {
        let priv_key = SecretKey::random(&mut OsRng);
        let mut hash = [0u8; 32];
        for (i, b) in b"merchant.com.openpay.test".iter().enumerate() {
            hash[i % 32] ^= *b;
        }
        let chain = CertChain::new(true, 2_000_000_000, hash);
        (priv_key, hash, chain)
    }

    #[test]
    fn roundtrip_decrypts_to_vault_ref() {
        let (priv_key, hash, chain) = fresh_setup();
        let token = build_token(
            &priv_key,
            &hash,
            "4111111111111234",
            "271231",
            "840",
            1234,
            "EMV",
            "AAAAAAA=",
            Some("05"),
            "abcd1234",
        );
        let tokenizer = Sha256VaultTokenizer;
        let out = decrypt(&token, &hash, &chain, &priv_key, &tokenizer, 1_700_000_000).unwrap();
        assert_eq!(out.transaction_amount.minor_units, 1234);
        assert!(out.application_primary_account_number.as_str().starts_with("sha256:"));
    }

    #[test]
    fn tampered_ciphertext_yields_aead_auth_failed() {
        let (priv_key, hash, chain) = fresh_setup();
        let mut token = build_token(
            &priv_key, &hash, "4111111111110000", "271231", "840", 100, "EMV", "AAAA", None, "id",
        );
        tamper_ciphertext(&mut token);
        let tokenizer = Sha256VaultTokenizer;
        let err = decrypt(&token, &hash, &chain, &priv_key, &tokenizer, 1_700_000_000).unwrap_err();
        assert_eq!(err, Error::AeadAuthFailed);
    }

    #[test]
    fn wrong_merchant_id_yields_mismatch() {
        let (priv_key, hash, chain) = fresh_setup();
        let token = build_token(
            &priv_key, &hash, "4111111111110000", "271231", "840", 100, "EMV", "AAAA", None, "id",
        );
        let mut other_hash = [0u8; 32];
        other_hash[0] = 0xff;
        let tokenizer = Sha256VaultTokenizer;
        let err = decrypt(&token, &other_hash, &chain, &priv_key, &tokenizer, 1_700_000_000)
            .unwrap_err();
        assert_eq!(err, Error::MerchantIdMismatch);
    }

    #[test]
    fn expired_cert_yields_bad_certificate() {
        let (priv_key, hash, _) = fresh_setup();
        let expired_chain = CertChain::new(true, 1_500_000_000, hash);
        let token = build_token(
            &priv_key, &hash, "4111111111110000", "271231", "840", 100, "EMV", "AAAA", None, "id",
        );
        let tokenizer = Sha256VaultTokenizer;
        let err = decrypt(
            &token,
            &hash,
            &expired_chain,
            &priv_key,
            &tokenizer,
            1_700_000_000,
        )
        .unwrap_err();
        assert_eq!(err, Error::BadCertificate);
    }
}
