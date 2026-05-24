//! WeChat Pay JSAPI happy-path + notify-callback HMAC verification.

use op_core::{Currency, Money};
use op_ewallets_asia::wallet::{
    AsiaWallet, ChargeIntent, ChargeStatus, PresentmentMode,
};
use op_ewallets_asia::wechat::{
    verify_notify_hmac, WeChatAdapter, WeChatSigner, WeChatTransport,
};
use std::sync::Mutex;

struct CannedTransport {
    response: String,
    seen: Mutex<Option<String>>,
}

impl WeChatTransport for CannedTransport {
    fn post_json(
        &self,
        path: &str,
        _headers: &[(String, String)],
        _body: &str,
    ) -> op_ewallets_asia::Result<String> {
        *self.seen.lock().unwrap() = Some(path.into());
        Ok(self.response.clone())
    }
}

struct StubSigner;
impl WeChatSigner for StubSigner {
    fn sign(&self, _canonical: &[u8]) -> op_ewallets_asia::Result<String> {
        Ok("sig".into())
    }
}

#[test]
fn jsapi_happy_path() {
    let canned = r#"{ "prepay_id": "wx20211111000000abcdef" }"#;
    let adapter = WeChatAdapter {
        mch_id: "1900000109".into(),
        app_id: "wxd678efh567hg6787".into(),
        host: "https://api.mch.weixin.qq.com".into(),
        transport: Box::new(CannedTransport {
            response: canned.into(),
            seen: Mutex::new(None),
        }),
        signer: Box::new(StubSigner),
    };
    let intent = ChargeIntent {
        merchant_order_id: "ord-77".into(),
        amount: Money::from_minor(1_500, Currency::CNY),
        description: "OpenPay test".into(),
        presentment: PresentmentMode::InAppJsApi,
        consumer_hint: Some("ozt0n5UqEEFvxOQzm9fHsKVklyAU".into()),
        notify_url: Some("https://merchant.example/notify".into()),
    };
    let result = adapter.create_charge(&intent).expect("wechat jsapi");
    assert_eq!(result.status, ChargeStatus::Pending);
    assert_eq!(result.presentment_payload, "wx20211111000000abcdef");
}

#[test]
fn jsapi_rejects_missing_openid() {
    let adapter = WeChatAdapter {
        mch_id: "x".into(),
        app_id: "x".into(),
        host: "https://api.mch.weixin.qq.com".into(),
        transport: Box::new(CannedTransport {
            response: "{}".into(),
            seen: Mutex::new(None),
        }),
        signer: Box::new(StubSigner),
    };
    let intent = ChargeIntent {
        merchant_order_id: "o".into(),
        amount: Money::from_minor(100, Currency::CNY),
        description: "x".into(),
        presentment: PresentmentMode::InAppJsApi,
        consumer_hint: None,
        notify_url: Some("https://merchant.example/notify".into()),
    };
    assert!(adapter.create_charge(&intent).is_err());
}

#[test]
fn notify_hmac_roundtrip() {
    use hmac::{Hmac, Mac};
    use sha2::Sha256;

    let secret = b"shared-notify-secret";
    let timestamp = "1670000000";
    let nonce = "abc123";
    let body = r#"{"event_type":"TRANSACTION.SUCCESS"}"#;

    // Compute expected the same way the verifier expects.
    let mut mac = <Hmac<Sha256> as Mac>::new_from_slice(secret).unwrap();
    mac.update(timestamp.as_bytes());
    mac.update(b"\n");
    mac.update(nonce.as_bytes());
    mac.update(b"\n");
    mac.update(body.as_bytes());
    mac.update(b"\n");
    let tag = mac.finalize().into_bytes();

    // Standard base64 encode.
    let b64 = base64_encode(&tag);
    assert!(verify_notify_hmac(timestamp, nonce, body, &b64, secret).is_ok());

    // Tamper with the body — should fail.
    let bad = verify_notify_hmac(timestamp, nonce, "{}", &b64, secret);
    assert!(bad.is_err());
}

fn base64_encode(data: &[u8]) -> String {
    const ALPHA: &[u8; 64] =
        b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::new();
    let mut i = 0;
    while i + 3 <= data.len() {
        let n = (u32::from(data[i]) << 16) | (u32::from(data[i + 1]) << 8) | u32::from(data[i + 2]);
        out.push(ALPHA[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHA[((n >> 12) & 0x3F) as usize] as char);
        out.push(ALPHA[((n >> 6) & 0x3F) as usize] as char);
        out.push(ALPHA[(n & 0x3F) as usize] as char);
        i += 3;
    }
    let rem = data.len() - i;
    if rem == 1 {
        let n = u32::from(data[i]) << 16;
        out.push(ALPHA[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHA[((n >> 12) & 0x3F) as usize] as char);
        out.push('=');
        out.push('=');
    } else if rem == 2 {
        let n = (u32::from(data[i]) << 16) | (u32::from(data[i + 1]) << 8);
        out.push(ALPHA[((n >> 18) & 0x3F) as usize] as char);
        out.push(ALPHA[((n >> 12) & 0x3F) as usize] as char);
        out.push(ALPHA[((n >> 6) & 0x3F) as usize] as char);
        out.push('=');
    }
    out
}
