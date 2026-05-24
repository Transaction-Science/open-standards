//! QRIS (Indonesia, BI) — EMVCo Merchant-Presented Mode QR.
//!
//! QRIS (Quick Response Code Indonesian Standard) is operated by
//! Bank Indonesia + ASPI and unifies all Indonesian wallet
//! acceptance (GoPay, OVO, DANA, ShopeePay, LinkAja, every domestic
//! issuer) behind a single merchant-presented QR. The merchant
//! account container at tag `26` uses sub-tag
//! `00 = ID.CO.QRIS.WWW` for the BI-assigned NMID (National
//! Merchant ID) with sub-tag `02` holding the 15-digit NMID.

use op_core::Currency;

use crate::error::Result;
use crate::promptpay::crc_ccitt_false;
use crate::wallet::{
    AsiaWallet, ChargeIntent, ChargeResult, ChargeStatus, PresentmentMode, WalletKind,
};

/// QRIS Merchant Criteria Indicator (BI category code).
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum QrisMerchantTier {
    /// UMI — Usaha Mikro (micro).
    Micro,
    /// UKE — Usaha Kecil (small).
    Small,
    /// UME — Usaha Menengah (medium).
    Medium,
    /// UBE — Usaha Besar (large).
    Large,
    /// URE — Usaha Reguler (regular / unclassified).
    Regular,
}

impl QrisMerchantTier {
    const fn code(self) -> &'static str {
        match self {
            Self::Micro => "UMI",
            Self::Small => "UKE",
            Self::Medium => "UME",
            Self::Large => "UBE",
            Self::Regular => "URE",
        }
    }
}

/// QRIS adapter.
#[derive(Clone, Debug)]
pub struct QrisAdapter {
    /// BI-assigned 15-digit National Merchant ID (NMID).
    pub nmid: String,
    /// Tier (drives the dispute/limit ladder BI applies).
    pub tier: QrisMerchantTier,
    /// Merchant display name (tag 59, max 25 bytes).
    pub merchant_name: String,
    /// Merchant city (tag 60, max 15 bytes).
    pub merchant_city: String,
}

impl AsiaWallet for QrisAdapter {
    fn kind(&self) -> WalletKind {
        WalletKind::Qris
    }

    fn create_charge(&self, intent: &ChargeIntent) -> Result<ChargeResult> {
        intent.validate_common()?;
        intent.require_currency(Currency::try_new(*b"IDR", 2)?)?;
        if !matches!(intent.presentment, PresentmentMode::MerchantPresentedQr) {
            return Err(crate::Error::Unsupported(
                "QRIS supports merchant-presented QR only",
            ));
        }
        if self.nmid.len() != 15 || !self.nmid.chars().all(|c| c.is_ascii_digit()) {
            return Err(crate::Error::Mpm("nmid must be 15 digits".into()));
        }
        if self.merchant_name.len() > 25 || !self.merchant_name.is_ascii() {
            return Err(crate::Error::Mpm("merchant_name overrun".into()));
        }
        if self.merchant_city.len() > 15 || !self.merchant_city.is_ascii() {
            return Err(crate::Error::Mpm("merchant_city overrun".into()));
        }

        let mut out = String::new();
        push_tlv(&mut out, "00", "01");
        push_tlv(&mut out, "01", "12");
        let mut acc = String::new();
        push_tlv(&mut acc, "00", "ID.CO.QRIS.WWW");
        push_tlv(&mut acc, "02", &self.nmid);
        push_tlv(&mut acc, "03", self.tier.code());
        push_tlv(&mut out, "26", &acc);
        push_tlv(&mut out, "52", "0000");
        push_tlv(&mut out, "53", "360"); // IDR numeric
        // IDR is technically 2-exponent in our op-core table but
        // major-unit-only at the rail surface (BI rounded IDR to
        // whole units in 2010). We round minor to nearest major.
        let major = intent.amount.minor_units / 100;
        push_tlv(&mut out, "54", &format!("{major}"));
        push_tlv(&mut out, "58", "ID");
        push_tlv(&mut out, "59", &self.merchant_name);
        push_tlv(&mut out, "60", &self.merchant_city);
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
        // QRIS settlement lifecycle arrives via the merchant's
        // acquirer-bank notify callback (which is a non-EMVCo
        // bilateral API per acquirer). Return Unknown here.
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
