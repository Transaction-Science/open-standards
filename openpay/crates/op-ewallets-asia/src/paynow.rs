//! PayNow (Singapore ABS) — EMVCo Merchant-Presented Mode QR.
//!
//! PayNow is the Singapore banking-industry A2A QR rail standardized
//! by ABS (Association of Banks in Singapore). Like every EMVCo MPM
//! rail it carries the merchant identity in tag `26` (the
//! international-merchant-account container), with sub-tag
//! `00 = SG.PAYNOW` as the global unique identifier, then sub-tag
//! `01` for the proxy type (0 = mobile, 2 = UEN), sub-tag `02` for
//! the proxy value, and sub-tag `03` for the editable-flag.

use op_core::Currency;

use crate::error::Result;
use crate::promptpay::crc_ccitt_false;
use crate::wallet::{
    AsiaWallet, ChargeIntent, ChargeResult, ChargeStatus, PresentmentMode, WalletKind,
};

/// PayNow proxy type as specified by ABS.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum PayNowProxy {
    /// SG mobile number proxy.
    Mobile,
    /// Singapore Unique Entity Number (UEN) proxy. Required for
    /// corporate-PayNow.
    Uen,
}

impl PayNowProxy {
    const fn code(self) -> &'static str {
        match self {
            Self::Mobile => "0",
            Self::Uen => "2",
        }
    }
}

/// PayNow adapter.
#[derive(Clone, Debug)]
pub struct PayNowAdapter {
    /// Which proxy type the merchant registered.
    pub proxy: PayNowProxy,
    /// Proxy value (MSISDN with `+65` prefix, or UEN).
    pub proxy_value: String,
    /// Optional merchant display name (tag 59), max 25 ASCII bytes.
    pub merchant_name: Option<String>,
    /// Optional merchant city (tag 60), max 15 ASCII bytes.
    pub merchant_city: Option<String>,
}

impl PayNowAdapter {
    /// Construct an adapter.
    #[must_use]
    pub const fn new(proxy: PayNowProxy, proxy_value: String) -> Self {
        Self {
            proxy,
            proxy_value,
            merchant_name: None,
            merchant_city: None,
        }
    }
}

impl AsiaWallet for PayNowAdapter {
    fn kind(&self) -> WalletKind {
        WalletKind::PayNow
    }

    fn create_charge(&self, intent: &ChargeIntent) -> Result<ChargeResult> {
        intent.validate_common()?;
        intent.require_currency(Currency::try_new(*b"SGD", 2)?)?;
        if !matches!(intent.presentment, PresentmentMode::MerchantPresentedQr) {
            return Err(crate::Error::Unsupported(
                "PayNow supports merchant-presented QR only",
            ));
        }
        let qr = encode_qr(
            self.proxy,
            &self.proxy_value,
            intent,
            self.merchant_name.as_deref(),
            self.merchant_city.as_deref(),
        )?;
        Ok(ChargeResult {
            merchant_order_id: intent.merchant_order_id.clone(),
            provider_transaction_id: String::new(),
            status: ChargeStatus::Pending,
            presentment_payload: qr,
        })
    }

    fn query_charge(&self, merchant_order_id: &str) -> Result<ChargeResult> {
        Ok(ChargeResult {
            merchant_order_id: merchant_order_id.to_string(),
            provider_transaction_id: String::new(),
            status: ChargeStatus::Unknown,
            presentment_payload: String::new(),
        })
    }
}

/// Encode an EMVCo MPM QR string for PayNow.
///
/// # Errors
/// Returns [`crate::Error::Mpm`] on overrun fields.
pub fn encode_qr(
    proxy: PayNowProxy,
    proxy_value: &str,
    intent: &ChargeIntent,
    merchant_name: Option<&str>,
    merchant_city: Option<&str>,
) -> Result<String> {
    let mut out = String::new();
    push_tlv(&mut out, "00", "01");
    push_tlv(&mut out, "01", "12");

    let mut acc = String::new();
    push_tlv(&mut acc, "00", "SG.PAYNOW");
    push_tlv(&mut acc, "01", proxy.code());
    if proxy_value.is_empty() {
        return Err(crate::Error::Mpm("empty proxy value".into()));
    }
    push_tlv(&mut acc, "02", proxy_value);
    // sub-tag 03: editable (1 = consumer may edit amount, 0 = fixed).
    push_tlv(&mut acc, "03", "0");
    push_tlv(&mut out, "26", &acc);

    push_tlv(&mut out, "52", "0000");
    push_tlv(&mut out, "53", "702"); // SGD numeric
    let major = intent.amount.minor_units / 100;
    let minor = intent.amount.minor_units % 100;
    push_tlv(&mut out, "54", &format!("{major}.{minor:02}"));
    push_tlv(&mut out, "58", "SG");
    if let Some(name) = merchant_name {
        if name.len() > 25 || !name.is_ascii() {
            return Err(crate::Error::Mpm("merchant name overrun".into()));
        }
        push_tlv(&mut out, "59", name);
    }
    if let Some(city) = merchant_city {
        if city.len() > 15 || !city.is_ascii() {
            return Err(crate::Error::Mpm("merchant city overrun".into()));
        }
        push_tlv(&mut out, "60", city);
    }
    out.push_str("6304");
    let crc = crc_ccitt_false(out.as_bytes());
    out.push_str(&format!("{crc:04X}"));
    Ok(out)
}

fn push_tlv(buf: &mut String, tag: &str, value: &str) {
    buf.push_str(tag);
    buf.push_str(&format!("{:02}", value.len()));
    buf.push_str(value);
}
