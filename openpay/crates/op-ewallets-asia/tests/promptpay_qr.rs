//! PromptPay EMVCo MPM QR encoding tests.
//!
//! Includes structural assertions (tag presence, CRC trailer)
//! and the EMVCo standard CRC-16/CCITT-FALSE test vector.

use op_core::{Currency, Money};
use op_ewallets_asia::promptpay::{
    crc_ccitt_false, encode_qr, PromptPayAdapter, PromptPayTarget,
};
use op_ewallets_asia::wallet::{
    AsiaWallet, ChargeIntent, ChargeStatus, PresentmentMode,
};

#[test]
fn crc_emvco_test_vector() {
    // EMVCo §4.7.4 reference vector.
    assert_eq!(crc_ccitt_false(b"123456789"), 0x29B1);
}

#[test]
fn mobile_qr_round_trip() {
    let thb = Currency::try_new(*b"THB", 2).unwrap();
    let intent = ChargeIntent {
        merchant_order_id: "ord-pp-1".into(),
        amount: Money::from_minor(12_345, thb),
        description: "Pad Thai".into(),
        presentment: PresentmentMode::MerchantPresentedQr,
        consumer_hint: None,
        notify_url: None,
    };
    let qr = encode_qr(
        &PromptPayTarget::Mobile("0812345678".into()),
        &intent,
        Some("ACME"),
        Some("BANGKOK"),
    )
    .expect("encode");
    // Payload Format Indicator + dynamic POI.
    assert!(qr.starts_with("000201010212"));
    // Tag 29 (PromptPay account info) present.
    assert!(qr.contains("29"));
    // Thai BoT AID.
    assert!(qr.contains("A000000677010111"));
    // Currency 764 (THB).
    assert!(qr.contains("5303764"));
    // Amount tag 54 with "123.45".
    assert!(qr.contains("123.45"));
    // Country code SG... no, TH.
    assert!(qr.contains("5802TH"));
    // CRC trailer.
    let crc_field = &qr[qr.len() - 8..];
    assert!(crc_field.starts_with("6304"));
    // CRC must verify: re-compute over everything up to and
    // including "6304" and match the four hex digits.
    let crc_body = &qr[..qr.len() - 4];
    let computed = crc_ccitt_false(crc_body.as_bytes());
    let expected = u16::from_str_radix(&qr[qr.len() - 4..], 16).unwrap();
    assert_eq!(computed, expected);
}

#[test]
fn national_id_target_rejects_bad_length() {
    let thb = Currency::try_new(*b"THB", 2).unwrap();
    let intent = ChargeIntent {
        merchant_order_id: "ord".into(),
        amount: Money::from_minor(100, thb),
        description: "x".into(),
        presentment: PresentmentMode::MerchantPresentedQr,
        consumer_hint: None,
        notify_url: None,
    };
    let res = encode_qr(
        &PromptPayTarget::NationalId("123".into()),
        &intent,
        None,
        None,
    );
    assert!(res.is_err());
}

#[test]
fn adapter_rejects_non_thb() {
    let adapter = PromptPayAdapter::new(PromptPayTarget::Mobile("0812345678".into()));
    let intent = ChargeIntent {
        merchant_order_id: "ord".into(),
        amount: Money::from_minor(100, Currency::USD),
        description: "x".into(),
        presentment: PresentmentMode::MerchantPresentedQr,
        consumer_hint: None,
        notify_url: None,
    };
    assert!(adapter.create_charge(&intent).is_err());
}

#[test]
fn adapter_query_returns_unknown() {
    let adapter = PromptPayAdapter::new(PromptPayTarget::Mobile("0812345678".into()));
    let res = adapter.query_charge("ord-1").unwrap();
    assert_eq!(res.status, ChargeStatus::Unknown);
}
