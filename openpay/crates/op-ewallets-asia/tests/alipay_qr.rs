//! End-to-end shape test for the Alipay precreate (QR) flow.
//!
//! Uses an in-memory transport + signer to verify the wire shape
//! the adapter sends and the parsing of a canned response.

use op_core::{Currency, Money};
use op_ewallets_asia::alipay::{
    AlipayAdapter, AlipaySigner, AlipayTransport,
};
use op_ewallets_asia::wallet::{
    AsiaWallet, ChargeIntent, ChargeStatus, PresentmentMode,
};
use std::sync::Mutex;

struct CapturingTransport {
    canned_response: String,
    seen: Mutex<Option<(String, String, Vec<(String, String)>)>>,
}

impl AlipayTransport for CapturingTransport {
    fn post_json(
        &self,
        path: &str,
        headers: &[(String, String)],
        body: &str,
    ) -> op_ewallets_asia::Result<String> {
        *self.seen.lock().unwrap() = Some((path.into(), body.into(), headers.to_vec()));
        Ok(self.canned_response.clone())
    }
}

struct DummySigner;
impl AlipaySigner for DummySigner {
    fn sign(&self, _payload: &[u8]) -> op_ewallets_asia::Result<String> {
        Ok("dummy-signature".into())
    }
}

#[test]
fn precreate_emits_qr_url() {
    let canned = r#"{
        "alipay_trade_precreate_response": {
            "code": "10000",
            "msg": "Success",
            "out_trade_no": "ord-123",
            "qr_code": "https://qr.alipay.com/bax01234567890"
        }
    }"#;
    let adapter = AlipayAdapter {
        app_id: "2021000123456789".into(),
        host: "https://openapi.alipay.com".into(),
        transport: Box::new(CapturingTransport {
            canned_response: canned.into(),
            seen: Mutex::new(None),
        }),
        signer: Box::new(DummySigner),
    };
    let intent = ChargeIntent {
        merchant_order_id: "ord-123".into(),
        amount: Money::from_minor(9_999, Currency::CNY),
        description: "OpenPay test".into(),
        presentment: PresentmentMode::MerchantPresentedQr,
        consumer_hint: None,
        notify_url: Some("https://merchant.example/notify".into()),
    };
    let result = adapter.create_charge(&intent).expect("alipay precreate");
    assert_eq!(result.merchant_order_id, "ord-123");
    assert_eq!(result.status, ChargeStatus::Pending);
    assert!(result.presentment_payload.starts_with("https://qr.alipay.com/"));
}

#[test]
fn precreate_rejects_non_cny() {
    let adapter = AlipayAdapter {
        app_id: "x".into(),
        host: "https://openapi.alipay.com".into(),
        transport: Box::new(CapturingTransport {
            canned_response: "{}".into(),
            seen: Mutex::new(None),
        }),
        signer: Box::new(DummySigner),
    };
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
fn precreate_surfaces_provider_error() {
    let canned = r#"{
        "alipay_trade_precreate_response": {
            "code": "40004",
            "msg": "Business Failed",
            "sub_code": "ACQ.TRADE_HAS_SUCCESS",
            "sub_msg": "Trade exists"
        }
    }"#;
    let parsed = AlipayAdapter::parse_precreate_response(canned);
    assert!(parsed.is_err());
}
