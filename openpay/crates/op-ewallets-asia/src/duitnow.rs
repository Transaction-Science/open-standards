//! DuitNow (Malaysia PayNet) — EMVCo Merchant-Presented Mode QR.
//!
//! DuitNow is the PayNet-operated Malaysian A2A rail. The DuitNow QR
//! standard (released 2019, revised 2022 to support cross-border
//! BNM-MAS-BI linkages) carries the merchant identifier in EMVCo MPM
//! tag `26`, with sub-tag `00 = MY.COM.PAYNET` and a 14-digit
//! DuitNow ID in sub-tag `01`. Cross-border via the BNM-BI link is
//! signaled by switching sub-tag `00` to the corresponding partner
//! AID; that path is out-of-scope for this domestic adapter.

use op_core::Currency;

use crate::error::Result;
use crate::promptpay::crc_ccitt_false;
use crate::wallet::{
    AsiaWallet, ChargeIntent, ChargeResult, ChargeStatus, PresentmentMode, WalletKind,
};

/// DuitNow adapter.
#[derive(Clone, Debug)]
pub struct DuitNowAdapter {
    /// 14-digit DuitNow merchant ID (assigned by the merchant's
    /// acquiring bank).
    pub duitnow_id: String,
    /// Optional merchant display name (tag 59).
    pub merchant_name: Option<String>,
    /// Optional merchant city (tag 60).
    pub merchant_city: Option<String>,
}

impl DuitNowAdapter {
    /// Construct an adapter.
    #[must_use]
    pub const fn new(duitnow_id: String) -> Self {
        Self {
            duitnow_id,
            merchant_name: None,
            merchant_city: None,
        }
    }
}

impl AsiaWallet for DuitNowAdapter {
    fn kind(&self) -> WalletKind {
        WalletKind::DuitNow
    }

    fn create_charge(&self, intent: &ChargeIntent) -> Result<ChargeResult> {
        intent.validate_common()?;
        intent.require_currency(Currency::try_new(*b"MYR", 2)?)?;
        if !matches!(intent.presentment, PresentmentMode::MerchantPresentedQr) {
            return Err(crate::Error::Unsupported(
                "DuitNow supports merchant-presented QR only",
            ));
        }
        if self.duitnow_id.len() != 14 || !self.duitnow_id.chars().all(|c| c.is_ascii_digit()) {
            return Err(crate::Error::Mpm("duitnow_id must be 14 digits".into()));
        }

        let mut out = String::new();
        push_tlv(&mut out, "00", "01");
        push_tlv(&mut out, "01", "12");
        let mut acc = String::new();
        push_tlv(&mut acc, "00", "MY.COM.PAYNET");
        push_tlv(&mut acc, "01", &self.duitnow_id);
        push_tlv(&mut out, "26", &acc);
        push_tlv(&mut out, "52", "0000");
        push_tlv(&mut out, "53", "458"); // MYR numeric
        let major = intent.amount.minor_units / 100;
        let minor = intent.amount.minor_units % 100;
        push_tlv(&mut out, "54", &format!("{major}.{minor:02}"));
        push_tlv(&mut out, "58", "MY");
        if let Some(name) = &self.merchant_name {
            push_tlv(&mut out, "59", name);
        }
        if let Some(city) = &self.merchant_city {
            push_tlv(&mut out, "60", city);
        }
        out.push_str("6304");
        let crc = crc_ccitt_false(out.as_bytes());
        out.push_str(&format!("{crc:04X}"));

        Ok(ChargeResult {
            merchant_order_id: intent.merchant_order_id.clone(),
            provider_transaction_id: String::new(),
            status: ChargeStatus::Pending,
            presentment_payload: out,
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

fn push_tlv(buf: &mut String, tag: &str, value: &str) {
    buf.push_str(tag);
    buf.push_str(&format!("{:02}", value.len()));
    buf.push_str(value);
}
