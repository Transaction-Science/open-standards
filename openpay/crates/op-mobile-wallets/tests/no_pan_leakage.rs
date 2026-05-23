//! No-PAN-leakage assertions on the public API surface.
//!
//! The PCI-zero invariant for this crate: a cleartext device-PAN must
//! not appear as `String`, `&str`, `Vec<u8>`, or `&[u8]` on any public
//! method's return type after decryption. The only credential exit
//! point is [`op_core::VaultRef`].
//!
//! These tests *behaviorally* show that:
//!
//! 1. After a full successful decrypt, the public surface of
//!    [`DecryptedToken`] does not contain the original cleartext DPAN.
//!    We check this by serializing the entire `DecryptedToken` to JSON
//!    and searching for the DPAN bytes — the rendered bytes must not
//!    contain them.
//! 2. The `Debug` impl on the contained `VaultRef` masks its value
//!    (only first 4 chars + length).
//! 3. Every wallet (Apple, Google, Samsung) round-trips through the
//!    same invariant.
//!
//! Static (type-level) assertions are enforced by the absence of any
//! `pub fn` in this crate that returns the cleartext DPAN — see
//! `cargo doc -p op-mobile-wallets` for the rendered public surface.

use op_mobile_wallets::apple_pay::{ApplePayConfig, CertChain, ExpirationDate, test_support as ap_ts};
use op_mobile_wallets::google_pay::{GooglePayConfig, test_support as gp_ts};
use op_mobile_wallets::samsung_pay::{SamsungPayConfig, test_support as sp_ts};
use op_mobile_wallets::wallet::{IdentityVaultTokenizer, Sha256VaultTokenizer, Wallet, WalletConfig, decryptor_for};
use p256::SecretKey;
use p256::elliptic_curve::rand_core::OsRng;
use sha2::{Digest, Sha256};
use std::sync::Arc;

const DPAN: &str = "4012888888881881";

#[test]
fn apple_pay_no_pan_in_serialized_output() {
    let priv_key = SecretKey::random(&mut OsRng);
    let mut hash = [0u8; 32];
    for (i, b) in b"merchant.com.openpay.test".iter().enumerate() {
        hash[i % 32] ^= *b;
    }
    let chain = CertChain::new(true, 2_000_000_000, hash);
    let cfg = ApplePayConfig {
        merchant_id_hash: hash,
        cert_chain: chain,
        ephemeral_private_key: priv_key.clone(),
        now_unix: 1_700_000_000,
    };
    let tokenizer: Arc<dyn op_mobile_wallets::wallet::VaultTokenizer> =
        Arc::new(Sha256VaultTokenizer);
    let wcfg = WalletConfig {
        apple_pay: Some(cfg),
        google_pay: None,
        samsung_pay: None,
        vault_tokenizer: tokenizer,
    };
    let decryptor = decryptor_for(Wallet::ApplePay, &wcfg).unwrap();
    let token = ap_ts::build_token(
        &priv_key,
        &hash,
        DPAN,
        "271231",
        "840",
        1234,
        "EMV",
        "QUFBQQ==",
        Some("05"),
        "txn-001",
    );
    let raw = serde_json::to_vec(&token).unwrap();
    let out = decryptor.decrypt(&raw).unwrap();

    // 1. Vault ref does NOT contain the cleartext DPAN.
    assert!(!out.application_primary_account_number.as_str().contains(DPAN));
    // 2. Whole-struct JSON does NOT contain the cleartext DPAN.
    let dbg_json = serde_json::to_string(&out).unwrap();
    assert!(!dbg_json.contains(DPAN));
    // 3. Debug rendering masks the vault ref.
    let debug_str = format!("{:?}", out.application_primary_account_number);
    assert!(!debug_str.contains(DPAN));
    assert!(debug_str.starts_with("VaultRef("));
}

#[test]
fn google_pay_no_pan_in_serialized_output() {
    let setup = gp_ts::fresh_setup();
    let cfg = GooglePayConfig {
        root_signing_keys: vec![setup.root_pub],
        sender_id: "Google".into(),
        recipient_id: "merchant:test".into(),
        recipient_private_key: setup.recipient_priv.clone(),
        now_unix_millis: 1_700_000_000_000,
    };
    let tokenizer: Arc<dyn op_mobile_wallets::wallet::VaultTokenizer> =
        Arc::new(Sha256VaultTokenizer);
    let wcfg = WalletConfig {
        apple_pay: None,
        google_pay: Some(cfg),
        samsung_pay: None,
        vault_tokenizer: tokenizer,
    };
    let decryptor = decryptor_for(Wallet::GooglePay, &wcfg).unwrap();
    let token = gp_ts::build_token(
        &setup,
        "Google",
        "merchant:test",
        DPAN,
        2030,
        12,
        "CRYPTOGRAM_3DS",
        Some("QUFBQQ=="),
        Some("05"),
        "USD",
        4242,
        1_900_000_000_000,
        1_900_000_000_000,
    );
    let raw = serde_json::to_vec(&token).unwrap();
    let out = decryptor.decrypt(&raw).unwrap();

    assert!(!out.application_primary_account_number.as_str().contains(DPAN));
    let dbg_json = serde_json::to_string(&out).unwrap();
    assert!(!dbg_json.contains(DPAN));
}

#[test]
fn samsung_pay_no_pan_in_serialized_output() {
    let priv_key = SecretKey::random(&mut OsRng);
    let merchant_id = "samsung-merchant-test".to_string();
    let mut h = Sha256::new();
    h.update(merchant_id.as_bytes());
    let hash: [u8; 32] = h.finalize().into();
    let chain = CertChain::new(true, 2_000_000_000, hash);
    let cfg = SamsungPayConfig {
        merchant_id: merchant_id.clone(),
        cert_chain: chain,
        recipient_private_key: priv_key.clone(),
        now_unix: 1_700_000_000,
    };
    let tokenizer: Arc<dyn op_mobile_wallets::wallet::VaultTokenizer> =
        Arc::new(Sha256VaultTokenizer);
    let wcfg = WalletConfig {
        apple_pay: None,
        google_pay: None,
        samsung_pay: Some(cfg),
        vault_tokenizer: tokenizer,
    };
    let decryptor = decryptor_for(Wallet::SamsungPay, &wcfg).unwrap();
    let token = sp_ts::build_token(
        &priv_key,
        &merchant_id,
        DPAN,
        "280630",
        "840",
        9999,
        "EMV",
        "QUFBQQ==",
        Some("02"),
        "txn-samsung",
    );
    let raw = serde_json::to_vec(&token).unwrap();
    let out = decryptor.decrypt(&raw).unwrap();

    assert!(!out.application_primary_account_number.as_str().contains(DPAN));
    let dbg_json = serde_json::to_string(&out).unwrap();
    assert!(!dbg_json.contains(DPAN));
}

#[test]
fn identity_tokenizer_still_keeps_dpan_off_string_surface() {
    // Even with the identity tokenizer (which base64-encodes DPAN as
    // the vault id), the DPAN ASCII bytes don't appear as a `String`
    // — they appear inside a base64 blob inside a `VaultRef`. So a
    // text-search for the DPAN must NOT find it.
    let priv_key = SecretKey::random(&mut OsRng);
    let mut hash = [0u8; 32];
    for (i, b) in b"identity-merchant".iter().enumerate() {
        hash[i % 32] ^= *b;
    }
    let chain = CertChain::new(true, 2_000_000_000, hash);
    let cfg = ApplePayConfig {
        merchant_id_hash: hash,
        cert_chain: chain,
        ephemeral_private_key: priv_key.clone(),
        now_unix: 1_700_000_000,
    };
    let tokenizer: Arc<dyn op_mobile_wallets::wallet::VaultTokenizer> =
        Arc::new(IdentityVaultTokenizer);
    let wcfg = WalletConfig {
        apple_pay: Some(cfg),
        google_pay: None,
        samsung_pay: None,
        vault_tokenizer: tokenizer,
    };
    let decryptor = decryptor_for(Wallet::ApplePay, &wcfg).unwrap();
    let token = ap_ts::build_token(
        &priv_key,
        &hash,
        DPAN,
        "271231",
        "840",
        1,
        "EMV",
        "QUFB",
        None,
        "id-txn",
    );
    let raw = serde_json::to_vec(&token).unwrap();
    let out = decryptor.decrypt(&raw).unwrap();
    let json = serde_json::to_string(&out).unwrap();
    assert!(!json.contains(DPAN));
    let _ = ExpirationDate::parse_yymmdd("271231").unwrap();
}
