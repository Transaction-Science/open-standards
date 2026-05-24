//! UPI collect + mandate end-to-end shape tests using a stub PSP transport.

use op_core::{Currency, Money};
use op_ewallets_asia::upi::{
    MandateExecution, MandateRecurrence, MandateRequest, UpiAdapter, UpiTransport,
};
use op_ewallets_asia::wallet::{
    AsiaWallet, ChargeIntent, ChargeStatus, PresentmentMode,
};
use std::sync::Mutex;
use time::{Duration, OffsetDateTime};

struct ScriptedTransport {
    responses: Mutex<Vec<String>>,
    calls: Mutex<Vec<(String, String)>>,
}

impl ScriptedTransport {
    fn new(responses: Vec<String>) -> Self {
        Self {
            responses: Mutex::new(responses),
            calls: Mutex::new(Vec::new()),
        }
    }
}

impl UpiTransport for ScriptedTransport {
    fn post_json(&self, path: &str, body: &str) -> op_ewallets_asia::Result<String> {
        self.calls
            .lock()
            .unwrap()
            .push((path.into(), body.into()));
        Ok(self.responses.lock().unwrap().remove(0))
    }
}

#[test]
fn collect_happy_path() {
    let adapter = UpiAdapter {
        merchant_vpa: "acme@hdfcbank".into(),
        merchant_name: "Acme Store".into(),
        mcc: "5411".into(),
        transport: Box::new(ScriptedTransport::new(vec![
            r#"{ "txnId": "AXIS123", "status": "SUCCESS" }"#.into(),
        ])),
    };
    let intent = ChargeIntent {
        merchant_order_id: "ord-upi-1".into(),
        amount: Money::from_minor(50_000, Currency::INR),
        description: "Acme order".into(),
        presentment: PresentmentMode::Browser,
        consumer_hint: Some("alice@upi".into()),
        notify_url: Some("https://merchant.example/notify".into()),
    };
    let res = adapter.create_charge(&intent).expect("upi collect");
    assert_eq!(res.status, ChargeStatus::Succeeded);
    assert_eq!(res.provider_transaction_id, "AXIS123");
}

#[test]
fn intent_uri_encodes_correctly() {
    let adapter = UpiAdapter {
        merchant_vpa: "acme@hdfcbank".into(),
        merchant_name: "Acme Store".into(),
        mcc: "5411".into(),
        transport: Box::new(ScriptedTransport::new(vec![])),
    };
    let intent = ChargeIntent {
        merchant_order_id: "ord-7".into(),
        amount: Money::from_minor(12_345, Currency::INR),
        description: "Acme order".into(),
        presentment: PresentmentMode::Deeplink,
        consumer_hint: None,
        notify_url: None,
    };
    let res = adapter.create_charge(&intent).expect("upi intent");
    assert!(res.presentment_payload.starts_with("upi://pay?"));
    assert!(res.presentment_payload.contains("am=123.45"));
    assert!(res.presentment_payload.contains("cu=INR"));
    assert!(res.presentment_payload.contains("pa=acme%40hdfcbank"));
}

#[test]
fn vpa_validation_rejects_garbage() {
    assert!(UpiAdapter::validate_vpa_syntax("not-a-vpa").is_err());
    assert!(UpiAdapter::validate_vpa_syntax("a@b").is_err()); // too short
    assert!(UpiAdapter::validate_vpa_syntax("alice@upi").is_ok());
}

#[test]
fn mandate_create_then_execute() {
    let transport = ScriptedTransport::new(vec![
        // mandate create
        r#"{ "umn": "UMN-1234", "state": "ACTIVE" }"#.into(),
        // mandate execute
        r#"{ "txnId": "AXIS-EXEC-1", "status": "SUCCESS" }"#.into(),
    ]);
    let adapter = UpiAdapter {
        merchant_vpa: "acme@hdfcbank".into(),
        merchant_name: "Acme Store".into(),
        mcc: "5411".into(),
        transport: Box::new(transport),
    };
    let now = OffsetDateTime::now_utc();
    let create_req = MandateRequest {
        merchant_reference: "mref-1".into(),
        payer_vpa: "alice@upi".into(),
        amount_limit_minor: 100_000,
        recurrence: MandateRecurrence::Monthly,
        valid_from: now,
        valid_until: now + Duration::days(365),
    };
    let created = adapter.create_mandate(&create_req).expect("create");
    assert_eq!(created.umn, "UMN-1234");
    assert_eq!(created.state, "ACTIVE");

    let exec = MandateExecution {
        umn: created.umn.clone(),
        merchant_order_id: "exec-1".into(),
        amount_minor: 50_000,
        nonce_hex: "0123456789abcdef0123456789abcdef".into(),
    };
    let executed = adapter.execute_mandate(&exec).expect("execute");
    assert_eq!(executed.status, ChargeStatus::Succeeded);
    assert_eq!(executed.provider_transaction_id, "AXIS-EXEC-1");
}

#[test]
fn mandate_rejects_bad_nonce() {
    let adapter = UpiAdapter {
        merchant_vpa: "acme@hdfcbank".into(),
        merchant_name: "Acme Store".into(),
        mcc: "5411".into(),
        transport: Box::new(ScriptedTransport::new(vec![])),
    };
    let exec = MandateExecution {
        umn: "UMN-1".into(),
        merchant_order_id: "exec-1".into(),
        amount_minor: 50_000,
        nonce_hex: "too-short".into(),
    };
    assert!(adapter.execute_mandate(&exec).is_err());
}
