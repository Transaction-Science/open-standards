//! Google Pay payment-token decryption.
//!
//! ## Format reference
//!
//! Google's tokenization spec is published at
//! <https://developers.google.com/pay/api/web/guides/resources/payment-data-cryptography>.
//!
//! ## Protocol versions
//!
//! - **ECv1** (deprecated). One-tier signing: Google's root key signs
//!   the merchant's `signedMessage` directly. We *do not* support
//!   ECv1; [`GooglePayDecryptor::decrypt`] returns
//!   [`crate::Error::UnsupportedProtocolVersion`].
//! - **ECv2** (current). Two-tier signing: Google's root signing key
//!   signs an `IntermediateSigningKey` (which carries its own
//!   `keyExpiration`); the intermediate key signs the `signedMessage`.
//!   We implement ECv2.
//!
//! ## ECv2 procedure
//!
//! 1. Verify the `intermediateSigningKey.signatures[]` against the
//!    configured root signing keys. The pre-image is
//!    `length-prefixed("Google") || length-prefixed("ECv2") ||
//!    length-prefixed(intermediateSigningKey.signedKey)`.
//! 2. Verify the `signature` field on the outer envelope against
//!    `intermediateSigningKey.signedKey.keyValue`. Pre-image is
//!    `length-prefixed(senderId) || length-prefixed(recipientId) ||
//!    length-prefixed(protocolVersion) || length-prefixed(signedMessage)`.
//! 3. Parse `signedMessage` -> `{ encryptedMessage, ephemeralPublicKey,
//!    tag }`.
//! 4. ECDH P-256 against `ephemeralPublicKey` with the merchant's
//!    recipient private key. Shared secret `Z`.
//! 5. HKDF-SHA256, salt = "", info = "Google", IKM = `ephemeralPublicKey ||
//!    Z`. Expand to 64 bytes; first 32 = AES-256-CTR key, last 32 =
//!    HMAC-SHA256 key.
//! 6. Verify HMAC-SHA256 tag over the ciphertext using the MAC key.
//! 7. AES-256-CTR decrypt with IV = 16 zero bytes. (CTR safety
//!    relies on the per-transaction ephemeral key.)
//! 8. Inner JSON contains `messageExpiration`,
//!    `messageId`, `paymentMethod`, `paymentMethodDetails`. We
//!    enforce `messageExpiration` against `now`.

use std::sync::Arc;

use aes::Aes256;
use aes::cipher::{KeyIvInit, StreamCipher};
use base64::Engine as _;
use ctr::Ctr128BE;
use hkdf::Hkdf;
use hmac::{Hmac, Mac};
use op_core::{Currency, Money};
use p256::SecretKey;
use p256::ecdh::diffie_hellman;
use p256::ecdsa::VerifyingKey;
use p256::ecdsa::signature::Verifier;
use p256::elliptic_curve::sec1::{FromEncodedPoint, ToEncodedPoint};
use p256::{EncodedPoint, PublicKey};
use serde::{Deserialize, Serialize};
use sha2::Sha256;
use subtle::ConstantTimeEq;
use zeroize::Zeroize;

use crate::apple_pay::{ApplePayDataType, ExpirationDate};
use crate::cryptogram::EciIndicator;
use crate::error::{Error, Result};
use crate::wallet::{DecryptedToken, VaultTokenizer, Wallet, WalletDecryptor};

type Aes256Ctr = Ctr128BE<Aes256>;

/// Google Pay protocol version.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum GooglePayProtocolVersion {
    /// ECv1 (deprecated). Stub returns [`Error::UnsupportedProtocolVersion`].
    Ecv1,
    /// ECv2 (current). Implemented end-to-end.
    Ecv2,
}

/// Intermediate signing key block emitted in ECv2 tokens.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct IntermediateSigningKey {
    /// JSON-stringified `{keyValue, keyExpiration}` block.
    #[serde(rename = "signedKey")]
    pub signed_key: String,
    /// Base64-encoded ECDSA-P-256 signatures, one per Google root
    /// signing key the merchant configured.
    pub signatures: Vec<String>,
}

/// Inner `signedKey` JSON shape (Google Pay ECv2).
#[derive(Clone, Debug, Serialize, Deserialize)]
struct SignedKeyInner {
    #[serde(rename = "keyValue")]
    key_value: String,
    #[serde(rename = "keyExpiration")]
    key_expiration: String, // milliseconds since epoch, as a decimal string
}

/// Google Pay payment token as delivered on the wire.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GooglePayToken {
    /// Protocol version. The wire field is a string (`"ECv1"`,
    /// `"ECv2"`); we keep it as a String to preserve provider-quirky
    /// values for diagnostics.
    pub protocol_version: String,
    /// Base64-encoded ECDSA-P-256 signature over the
    /// `signedMessage` field by the intermediate key.
    pub signature: String,
    /// JSON-stringified `signedMessage` block.
    pub signed_message: String,
    /// Intermediate signing key. Required for ECv2.
    pub intermediate_signing_key: Option<IntermediateSigningKey>,
}

/// Inner `signedMessage` JSON (ECv2).
#[derive(Clone, Debug, Serialize, Deserialize)]
struct SignedMessage {
    #[serde(rename = "encryptedMessage")]
    encrypted_message: String,
    #[serde(rename = "ephemeralPublicKey")]
    ephemeral_public_key: String,
    tag: String,
}

/// Inner cleartext payload (ECv2), after AES-CTR decryption.
#[derive(Clone, Debug, Serialize, Deserialize)]
struct InnerEncrypted {
    #[serde(rename = "messageExpiration")]
    message_expiration: String, // millis since epoch, decimal string
    #[serde(rename = "messageId")]
    message_id: String,
    #[serde(rename = "paymentMethod")]
    payment_method: String,
    #[serde(rename = "paymentMethodDetails")]
    payment_method_details: PaymentMethodDetails,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct PaymentMethodDetails {
    #[serde(rename = "expirationYear")]
    expiration_year: u32,
    #[serde(rename = "expirationMonth")]
    expiration_month: u8,
    pan: String,
    #[serde(rename = "authMethod")]
    auth_method: String,
    #[serde(rename = "cryptogram", default)]
    cryptogram: Option<String>,
    #[serde(rename = "eciIndicator", default)]
    eci_indicator: Option<String>,
    /// Optional currency / amount the merchant set on the
    /// `PaymentDataRequest`. Google Pay echoes these back inside the
    /// encrypted blob so the rail driver can assert against the
    /// frontend's quoted total.
    #[serde(rename = "currencyCode", default)]
    currency_code: Option<String>,
    #[serde(rename = "transactionAmount", default)]
    transaction_amount: Option<i64>,
}

/// Google Pay decryptor configuration.
#[derive(Clone, Debug)]
pub struct GooglePayConfig {
    /// Google's root signing keys. In production these are pinned
    /// from <https://payments.developers.google.com/paymentmethodtoken/test/keys.json>
    /// (test) or the production equivalent.
    pub root_signing_keys: Vec<VerifyingKey>,
    /// The merchant's gateway recipient id (e.g.
    /// `"merchant:1234567890"`). Bound into the signature pre-image.
    pub recipient_id: String,
    /// Sender id Google uses ("Google"). Configurable for testing.
    pub sender_id: String,
    /// Recipient private key (P-256). The merchant's enrollment
    /// private half, whose public key is what Google encrypted to.
    pub recipient_private_key: SecretKey,
    /// Current unix milliseconds, used to enforce
    /// `messageExpiration`. Inject for tests; in production wire
    /// `OffsetDateTime::now_utc().unix_timestamp_nanos() / 1_000_000`.
    pub now_unix_millis: i64,
}

/// Google Pay decryptor.
pub struct GooglePayDecryptor {
    config: GooglePayConfig,
    tokenizer: Arc<dyn VaultTokenizer>,
}

impl GooglePayDecryptor {
    /// Construct.
    #[must_use]
    pub fn new(config: GooglePayConfig, tokenizer: Arc<dyn VaultTokenizer>) -> Self {
        Self { config, tokenizer }
    }
}

/// Length-prefixed concatenation as Google Pay specifies for
/// signature pre-images: each piece is `u32-LE length || bytes`.
fn length_prefixed(parts: &[&[u8]]) -> Vec<u8> {
    let mut out = Vec::with_capacity(parts.iter().map(|p| 4 + p.len()).sum());
    for p in parts {
        out.extend_from_slice(&(p.len() as u32).to_le_bytes());
        out.extend_from_slice(p);
    }
    out
}

/// Decrypt a Google Pay token.
pub fn decrypt(
    token: &GooglePayToken,
    root_signing_keys: &[VerifyingKey],
    sender_id: &str,
    recipient_id: &str,
    recipient_private_key: &SecretKey,
    tokenizer: &dyn VaultTokenizer,
    now_unix_millis: i64,
) -> Result<DecryptedToken> {
    // (1) Protocol version.
    match token.protocol_version.as_str() {
        "ECv2" => {}
        "ECv1" => {
            return Err(Error::UnsupportedProtocolVersion("ECv1".into()));
        }
        other => {
            return Err(Error::UnsupportedProtocolVersion(other.into()));
        }
    }

    // (2) Verify the intermediate signing key against the configured
    // root signing keys.
    let intermediate = token
        .intermediate_signing_key
        .as_ref()
        .ok_or(Error::MalformedPayload("missing intermediateSigningKey"))?;
    let pre_image = length_prefixed(&[
        sender_id.as_bytes(),
        b"ECv2",
        intermediate.signed_key.as_bytes(),
    ]);
    let mut chain_ok = false;
    for sig_b64 in &intermediate.signatures {
        let sig_der = base64::engine::general_purpose::STANDARD
            .decode(sig_b64)
            .map_err(|_| Error::MalformedPayload("intermediate sig not base64"))?;
        let Ok(sig) = p256::ecdsa::Signature::from_der(&sig_der) else {
            continue;
        };
        for root in root_signing_keys {
            if root.verify(&pre_image, &sig).is_ok() {
                chain_ok = true;
                break;
            }
        }
        if chain_ok {
            break;
        }
    }
    if !chain_ok {
        return Err(Error::BadSignature);
    }

    // (3) Parse signedKey, check keyExpiration.
    let signed_key: SignedKeyInner = serde_json::from_str(&intermediate.signed_key)
        .map_err(|_| Error::MalformedPayload("signedKey JSON"))?;
    let key_expiration_ms: i64 = signed_key
        .key_expiration
        .parse()
        .map_err(|_| Error::MalformedPayload("keyExpiration parse"))?;
    if now_unix_millis >= key_expiration_ms {
        return Err(Error::BadCertificate);
    }

    // (4) Verify the outer signature against the intermediate
    // signing key.
    let intermediate_pub_bytes = base64::engine::general_purpose::STANDARD
        .decode(&signed_key.key_value)
        .map_err(|_| Error::MalformedPayload("intermediate keyValue not base64"))?;
    let intermediate_point = EncodedPoint::from_bytes(&intermediate_pub_bytes)
        .map_err(|_| Error::MalformedPayload("intermediate keyValue not SEC1"))?;
    let intermediate_pk_opt = PublicKey::from_encoded_point(&intermediate_point);
    let intermediate_pk = Option::<PublicKey>::from(intermediate_pk_opt)
        .ok_or(Error::MalformedPayload("intermediate keyValue point not on curve"))?;
    let intermediate_vk = VerifyingKey::from(&intermediate_pk);

    let outer_pre_image = length_prefixed(&[
        sender_id.as_bytes(),
        recipient_id.as_bytes(),
        b"ECv2",
        token.signed_message.as_bytes(),
    ]);
    let outer_sig_der = base64::engine::general_purpose::STANDARD
        .decode(&token.signature)
        .map_err(|_| Error::MalformedPayload("outer signature not base64"))?;
    let outer_sig = p256::ecdsa::Signature::from_der(&outer_sig_der)
        .map_err(|_| Error::BadSignature)?;
    intermediate_vk
        .verify(&outer_pre_image, &outer_sig)
        .map_err(|_| Error::BadSignature)?;

    // (5) Parse signedMessage.
    let sm: SignedMessage = serde_json::from_str(&token.signed_message)
        .map_err(|_| Error::MalformedPayload("signedMessage JSON"))?;

    // (6) ECDH against ephemeralPublicKey.
    let ephem_pub_bytes = base64::engine::general_purpose::STANDARD
        .decode(&sm.ephemeral_public_key)
        .map_err(|_| Error::MalformedPayload("ephemeralPublicKey not base64"))?;
    let ephem_point = EncodedPoint::from_bytes(&ephem_pub_bytes)
        .map_err(|_| Error::KeyAgreementFailed)?;
    let ephem_pk_opt = PublicKey::from_encoded_point(&ephem_point);
    let ephem_pk = Option::<PublicKey>::from(ephem_pk_opt).ok_or(Error::KeyAgreementFailed)?;
    let shared = diffie_hellman(recipient_private_key.to_nonzero_scalar(), ephem_pk.as_affine());
    let mut z = shared.raw_secret_bytes().to_vec();

    // (7) HKDF-SHA256 with IKM = ephemPubKey || Z, info = sender_id,
    // salt = empty. Expand to 64 bytes: AES key || HMAC key.
    let mut ikm = Vec::with_capacity(ephem_pub_bytes.len() + z.len());
    ikm.extend_from_slice(&ephem_pub_bytes);
    ikm.extend_from_slice(&z);
    z.zeroize();
    let hk = Hkdf::<Sha256>::new(None, &ikm);
    let mut okm = [0u8; 64];
    hk.expand(sender_id.as_bytes(), &mut okm)
        .map_err(|_| Error::Internal("HKDF expand"))?;
    ikm.zeroize();
    let (aes_key, mac_key) = okm.split_at(32);

    // (8) HMAC tag verification (constant-time).
    let ciphertext = base64::engine::general_purpose::STANDARD
        .decode(&sm.encrypted_message)
        .map_err(|_| Error::MalformedPayload("encryptedMessage not base64"))?;
    let mac_tag = base64::engine::general_purpose::STANDARD
        .decode(&sm.tag)
        .map_err(|_| Error::MalformedPayload("tag not base64"))?;
    let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(mac_key)
        .map_err(|_| Error::Internal("HMAC key length"))?;
    mac.update(&ciphertext);
    let expected = mac.finalize().into_bytes();
    if mac_tag.len() != expected.len()
        || expected.as_slice().ct_eq(&mac_tag).unwrap_u8() != 1
    {
        return Err(Error::AeadAuthFailed);
    }

    // (9) AES-256-CTR decrypt (IV = zero block, per spec).
    let mut plaintext = ciphertext;
    let iv = [0u8; 16];
    let mut cipher = Aes256Ctr::new(aes_key.into(), (&iv).into());
    cipher.apply_keystream(&mut plaintext);

    // (10) Parse inner cleartext.
    let inner: InnerEncrypted = serde_json::from_slice(&plaintext)
        .map_err(|_| Error::MalformedPayload("inner CTR-payload JSON"))?;
    plaintext.zeroize();

    // (11) messageExpiration check.
    let msg_exp_ms: i64 = inner
        .message_expiration
        .parse()
        .map_err(|_| Error::MalformedPayload("messageExpiration parse"))?;
    if now_unix_millis >= msg_exp_ms {
        return Err(Error::BadCryptogramExpired);
    }

    // (12) Tokenize.
    let mut dpan_bytes = inner.payment_method_details.pan.clone().into_bytes();
    let vault_ref = tokenizer
        .tokenize(&dpan_bytes)
        .map_err(|e| match e {
            Error::VaultTokenizer(m) => Error::VaultTokenizer(m),
            other => Error::VaultTokenizer(format!("{other}")),
        })?;
    dpan_bytes.zeroize();

    // (13) Build the normalized output.
    let yy = (inner.payment_method_details.expiration_year % 100) as u8;
    let expiration = ExpirationDate {
        yy,
        mm: inner.payment_method_details.expiration_month,
        dd: 1,
    };
    let currency = inner
        .payment_method_details
        .currency_code
        .as_deref()
        .map_or(Ok(Currency::USD), currency_from_iso4217_alpha)?;
    let amount_minor = inner.payment_method_details.transaction_amount.unwrap_or(0);
    let amount = Money::from_minor(amount_minor, currency);
    let data_type = match inner.payment_method_details.auth_method.as_str() {
        "CRYPTOGRAM_3DS" => ApplePayDataType::ThreeDSecure,
        _ => ApplePayDataType::Emv,
    };
    let crypto_b64 = inner
        .payment_method_details
        .cryptogram
        .unwrap_or_default();
    let cryptogram_bytes = base64::engine::general_purpose::STANDARD
        .decode(&crypto_b64)
        .unwrap_or_default();
    let eci = inner
        .payment_method_details
        .eci_indicator
        .unwrap_or_default();

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

fn currency_from_iso4217_alpha(s: &str) -> Result<Currency> {
    match s {
        "USD" => Ok(Currency::USD),
        "EUR" => Ok(Currency::EUR),
        "BRL" => Ok(Currency::BRL),
        "INR" => Ok(Currency::INR),
        "GBP" => Ok(Currency::GBP),
        "JPY" => Ok(Currency::JPY),
        "CNY" => Ok(Currency::CNY),
        _ => Err(Error::MalformedPayload("unknown alpha currency code")),
    }
}

impl WalletDecryptor for GooglePayDecryptor {
    fn decrypt(&self, raw: &[u8]) -> Result<DecryptedToken> {
        let token: GooglePayToken =
            serde_json::from_slice(raw).map_err(|_| Error::MalformedPayload("GooglePayToken JSON"))?;
        decrypt(
            &token,
            &self.config.root_signing_keys,
            &self.config.sender_id,
            &self.config.recipient_id,
            &self.config.recipient_private_key,
            self.tokenizer.as_ref(),
            self.config.now_unix_millis,
        )
    }

    fn wallet(&self) -> Wallet {
        Wallet::GooglePay
    }
}

/// Test-only helpers: build valid ECv2 tokens for round-trip tests.
#[doc(hidden)]
pub mod test_support {
    use super::*;
    use aes::cipher::{KeyIvInit, StreamCipher};
    use p256::SecretKey;
    use p256::ecdsa::{Signature, SigningKey, signature::Signer};
    use p256::elliptic_curve::rand_core::OsRng;

    pub struct GpaySetup {
        pub root_priv: SigningKey,
        pub root_pub: VerifyingKey,
        pub recipient_priv: SecretKey,
    }

    pub fn fresh_setup() -> GpaySetup {
        let root_priv = SigningKey::random(&mut OsRng);
        let root_pub = *root_priv.verifying_key();
        let recipient_priv = SecretKey::random(&mut OsRng);
        GpaySetup {
            root_priv,
            root_pub,
            recipient_priv,
        }
    }

    pub fn build_token(
        setup: &GpaySetup,
        sender_id: &str,
        recipient_id: &str,
        dpan: &str,
        exp_year: u32,
        exp_month: u8,
        auth_method: &str,
        cryptogram_b64: Option<&str>,
        eci: Option<&str>,
        currency_alpha: &str,
        amount_minor: i64,
        message_expiration_millis: i64,
        intermediate_key_expiration_millis: i64,
    ) -> GooglePayToken {
        // 1. Intermediate signing key — fresh.
        let int_priv = SigningKey::random(&mut OsRng);
        let int_pub = *int_priv.verifying_key();
        let int_pub_sec1 = int_pub.to_encoded_point(false).as_bytes().to_vec();
        let signed_key_inner = serde_json::json!({
            "keyValue": base64::engine::general_purpose::STANDARD.encode(&int_pub_sec1),
            "keyExpiration": intermediate_key_expiration_millis.to_string(),
        });
        let signed_key_json = serde_json::to_string(&signed_key_inner).unwrap();
        let pre_image = length_prefixed(&[sender_id.as_bytes(), b"ECv2", signed_key_json.as_bytes()]);
        let int_sig: Signature = setup.root_priv.sign(&pre_image);
        let intermediate = IntermediateSigningKey {
            signed_key: signed_key_json,
            signatures: vec![base64::engine::general_purpose::STANDARD.encode(int_sig.to_der())],
        };

        // 2. Inner cleartext.
        let inner = serde_json::json!({
            "messageExpiration": message_expiration_millis.to_string(),
            "messageId": "msg-test",
            "paymentMethod": "CARD",
            "paymentMethodDetails": {
                "expirationYear": exp_year,
                "expirationMonth": exp_month,
                "pan": dpan,
                "authMethod": auth_method,
                "cryptogram": cryptogram_b64,
                "eciIndicator": eci,
                "currencyCode": currency_alpha,
                "transactionAmount": amount_minor,
            }
        });
        let plaintext = serde_json::to_vec(&inner).unwrap();

        // 3. ECDH against the recipient's public key with an
        // ephemeral key.
        let ephem_priv = SecretKey::random(&mut OsRng);
        let ephem_pub_sec1 = ephem_priv.public_key().to_encoded_point(false).as_bytes().to_vec();
        let shared = diffie_hellman(
            ephem_priv.to_nonzero_scalar(),
            setup.recipient_priv.public_key().as_affine(),
        );
        let z = shared.raw_secret_bytes().to_vec();
        let mut ikm = Vec::new();
        ikm.extend_from_slice(&ephem_pub_sec1);
        ikm.extend_from_slice(&z);
        let hk = Hkdf::<Sha256>::new(None, &ikm);
        let mut okm = [0u8; 64];
        hk.expand(sender_id.as_bytes(), &mut okm).unwrap();
        let (aes_key, mac_key) = okm.split_at(32);

        // 4. AES-256-CTR encrypt.
        let mut ciphertext = plaintext.clone();
        let iv = [0u8; 16];
        let mut cipher = Aes256Ctr::new(aes_key.into(), (&iv).into());
        cipher.apply_keystream(&mut ciphertext);

        // 5. HMAC tag.
        let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(mac_key).unwrap();
        mac.update(&ciphertext);
        let tag = mac.finalize().into_bytes();

        // 6. signedMessage.
        let sm = serde_json::json!({
            "encryptedMessage": base64::engine::general_purpose::STANDARD.encode(&ciphertext),
            "ephemeralPublicKey": base64::engine::general_purpose::STANDARD.encode(&ephem_pub_sec1),
            "tag": base64::engine::general_purpose::STANDARD.encode(tag),
        });
        let sm_json = serde_json::to_string(&sm).unwrap();

        // 7. Outer signature with the intermediate signing key.
        let outer_pre = length_prefixed(&[
            sender_id.as_bytes(),
            recipient_id.as_bytes(),
            b"ECv2",
            sm_json.as_bytes(),
        ]);
        let outer_sig: Signature = int_priv.sign(&outer_pre);

        GooglePayToken {
            protocol_version: "ECv2".into(),
            signature: base64::engine::general_purpose::STANDARD.encode(outer_sig.to_der()),
            signed_message: sm_json,
            intermediate_signing_key: Some(intermediate),
        }
    }

    pub fn tamper_outer_signature(token: &mut GooglePayToken) {
        let mut bytes = base64::engine::general_purpose::STANDARD
            .decode(&token.signature)
            .unwrap();
        let last = bytes.len() - 1;
        bytes[last] ^= 0x01;
        token.signature = base64::engine::general_purpose::STANDARD.encode(&bytes);
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::*;
    use super::*;
    use crate::wallet::Sha256VaultTokenizer;

    const NOW_MS: i64 = 1_700_000_000_000;
    const EXP_MS: i64 = 1_900_000_000_000;

    #[test]
    fn roundtrip_decrypts_to_vault_ref() {
        let setup = fresh_setup();
        let token = build_token(
            &setup,
            "Google",
            "merchant:test",
            "4111111111119876",
            2030,
            12,
            "CRYPTOGRAM_3DS",
            Some("AAAA"),
            Some("05"),
            "USD",
            4242,
            EXP_MS,
            EXP_MS,
        );
        let tokenizer = Sha256VaultTokenizer;
        let out = decrypt(
            &token,
            &[setup.root_pub],
            "Google",
            "merchant:test",
            &setup.recipient_priv,
            &tokenizer,
            NOW_MS,
        )
        .unwrap();
        assert_eq!(out.transaction_amount.minor_units, 4242);
        assert_eq!(out.payment_data_type, ApplePayDataType::ThreeDSecure);
        assert!(out.application_primary_account_number.as_str().starts_with("sha256:"));
    }

    #[test]
    fn ecv1_returns_unsupported() {
        let mut token = build_token(
            &fresh_setup(),
            "Google",
            "merchant:test",
            "4111111111111111",
            2030,
            12,
            "PAN_ONLY",
            None,
            None,
            "USD",
            100,
            EXP_MS,
            EXP_MS,
        );
        token.protocol_version = "ECv1".into();
        let tokenizer = Sha256VaultTokenizer;
        let s = fresh_setup();
        let err = decrypt(
            &token,
            &[s.root_pub],
            "Google",
            "merchant:test",
            &s.recipient_priv,
            &tokenizer,
            NOW_MS,
        )
        .unwrap_err();
        assert!(matches!(err, Error::UnsupportedProtocolVersion(_)));
    }

    #[test]
    fn tampered_outer_signature_yields_bad_signature() {
        let setup = fresh_setup();
        let mut token = build_token(
            &setup,
            "Google",
            "merchant:test",
            "4111111111111111",
            2030,
            12,
            "CRYPTOGRAM_3DS",
            Some("AAAA"),
            Some("05"),
            "USD",
            100,
            EXP_MS,
            EXP_MS,
        );
        tamper_outer_signature(&mut token);
        let tokenizer = Sha256VaultTokenizer;
        let err = decrypt(
            &token,
            &[setup.root_pub],
            "Google",
            "merchant:test",
            &setup.recipient_priv,
            &tokenizer,
            NOW_MS,
        )
        .unwrap_err();
        assert_eq!(err, Error::BadSignature);
    }

    #[test]
    fn expired_intermediate_key_yields_bad_certificate() {
        let setup = fresh_setup();
        let token = build_token(
            &setup,
            "Google",
            "merchant:test",
            "4111111111111111",
            2030,
            12,
            "CRYPTOGRAM_3DS",
            Some("AAAA"),
            Some("05"),
            "USD",
            100,
            EXP_MS,
            1_500_000_000_000, // intermediate key expired well before NOW_MS
        );
        let tokenizer = Sha256VaultTokenizer;
        let err = decrypt(
            &token,
            &[setup.root_pub],
            "Google",
            "merchant:test",
            &setup.recipient_priv,
            &tokenizer,
            NOW_MS,
        )
        .unwrap_err();
        assert_eq!(err, Error::BadCertificate);
    }

    #[test]
    fn expired_message_yields_cryptogram_expired() {
        let setup = fresh_setup();
        let token = build_token(
            &setup,
            "Google",
            "merchant:test",
            "4111111111111111",
            2030,
            12,
            "CRYPTOGRAM_3DS",
            Some("AAAA"),
            Some("05"),
            "USD",
            100,
            1_500_000_000_000, // message expired before NOW_MS
            EXP_MS,
        );
        let tokenizer = Sha256VaultTokenizer;
        let err = decrypt(
            &token,
            &[setup.root_pub],
            "Google",
            "merchant:test",
            &setup.recipient_priv,
            &tokenizer,
            NOW_MS,
        )
        .unwrap_err();
        assert_eq!(err, Error::BadCryptogramExpired);
    }

    #[test]
    fn wrong_root_key_yields_bad_signature() {
        let setup = fresh_setup();
        let token = build_token(
            &setup,
            "Google",
            "merchant:test",
            "4111111111111111",
            2030,
            12,
            "CRYPTOGRAM_3DS",
            Some("AAAA"),
            Some("05"),
            "USD",
            100,
            EXP_MS,
            EXP_MS,
        );
        let wrong_setup = fresh_setup();
        let tokenizer = Sha256VaultTokenizer;
        let err = decrypt(
            &token,
            &[wrong_setup.root_pub],
            "Google",
            "merchant:test",
            &setup.recipient_priv,
            &tokenizer,
            NOW_MS,
        )
        .unwrap_err();
        assert_eq!(err, Error::BadSignature);
    }
}
