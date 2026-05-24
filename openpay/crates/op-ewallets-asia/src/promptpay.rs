//! PromptPay (Thailand, BoT) — EMVCo Merchant-Presented Mode QR.
//!
//! PromptPay encodes the recipient as either a Thai-mobile MSISDN, a
//! Thai national ID, or an e-wallet id, embedded in EMVCo MPM TLV
//! tag `29` (the merchant-account-information container) with
//! sub-tag `00 = A000000677010111` (the Thai BoT AID). The CRC at
//! tag `63` is CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) over
//! every preceding byte including the `6304` length header.
//!
//! ## What this adapter ships
//!
//! - A pure-function encoder ([`encode_qr`]) that produces the EMVCo
//!   MPM TLV string from a [`PromptPayTarget`] + [`ChargeIntent`].
//! - The [`PromptPayAdapter`] type implementing [`AsiaWallet`] by
//!   wrapping the encoder. PromptPay has no central API — the QR
//!   carries every field needed to clear over the Thai
//!   ITMX (Interbank Transaction Management Exchange) rail — so
//!   `query_charge` returns `ChargeStatus::Unknown` and the
//!   operator's notification surface (bank-side webhook) settles
//!   the lifecycle.
//!
//! ## CRC reference
//!
//! EMVCo "Merchant Presented Mode Specification for QR Codes"
//! v1.1 §4.7.4 specifies CRC-16/CCITT-FALSE over the entire payload
//! ending with the literal `6304`, then the four-hex-digit CRC.

use op_core::Currency;

use crate::error::Result;
use crate::wallet::{
    AsiaWallet, ChargeIntent, ChargeResult, ChargeStatus, PresentmentMode, WalletKind,
};

/// PromptPay payee identifier kind.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum PromptPayTarget {
    /// Thai mobile MSISDN, with or without country code. The encoder
    /// normalizes to the 13-digit `0066XXXXXXXXXX` form BoT
    /// specifies for sub-tag 01.
    Mobile(String),
    /// Thai national-ID, 13 digits. Encoded under sub-tag 02.
    NationalId(String),
    /// E-wallet-id (15 digits). Encoded under sub-tag 03.
    EWalletId(String),
}

/// Adapter wrapping the EMVCo MPM encoder.
#[derive(Clone, Debug)]
pub struct PromptPayAdapter {
    /// The merchant's PromptPay target embedded in every QR.
    pub target: PromptPayTarget,
    /// Optional merchant display name (EMVCo MPM tag 59). Capped to
    /// 25 ASCII bytes by the spec.
    pub merchant_name: Option<String>,
    /// Optional merchant city (EMVCo MPM tag 60). Capped to 15
    /// ASCII bytes.
    pub merchant_city: Option<String>,
}

impl PromptPayAdapter {
    /// Construct a new adapter for the given target.
    #[must_use]
    pub const fn new(target: PromptPayTarget) -> Self {
        Self {
            target,
            merchant_name: None,
            merchant_city: None,
        }
    }
}

impl AsiaWallet for PromptPayAdapter {
    fn kind(&self) -> WalletKind {
        WalletKind::PromptPay
    }

    fn create_charge(&self, intent: &ChargeIntent) -> Result<ChargeResult> {
        intent.validate_common()?;
        intent.require_currency(Currency::try_new(*b"THB", 2)?)?;
        if !matches!(intent.presentment, PresentmentMode::MerchantPresentedQr) {
            return Err(crate::Error::Unsupported(
                "PromptPay supports merchant-presented QR only",
            ));
        }
        let qr = encode_qr(
            &self.target,
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
        // PromptPay has no merchant-facing query API — settlement
        // arrives via the merchant's bank-side webhook. We return
        // Unknown so the orchestrator falls back to that channel.
        Ok(ChargeResult {
            merchant_order_id: merchant_order_id.to_string(),
            provider_transaction_id: String::new(),
            status: ChargeStatus::Unknown,
            presentment_payload: String::new(),
        })
    }
}

/// Encode an EMVCo MPM QR string for PromptPay.
///
/// # Errors
/// Returns [`crate::Error::Mpm`] on malformed inputs (target too
/// long, merchant-name overrun, ...).
pub fn encode_qr(
    target: &PromptPayTarget,
    intent: &ChargeIntent,
    merchant_name: Option<&str>,
    merchant_city: Option<&str>,
) -> Result<String> {
    let mut out = String::new();

    // Tag 00: Payload Format Indicator. Always "01" for v1.
    push_tlv(&mut out, "00", "01");
    // Tag 01: Point of Initiation Method. "11" = static QR
    // (reusable), "12" = dynamic (single use). PromptPay charges
    // with a specific amount are dynamic.
    push_tlv(&mut out, "01", "12");

    // Tag 29: Merchant Account Information for PromptPay.
    // Sub-tag 00 = AID, sub-tag 01/02/03 = payee identifier.
    let mut acc = String::new();
    push_tlv(&mut acc, "00", "A000000677010111");
    match target {
        PromptPayTarget::Mobile(m) => {
            let normalized = normalize_msisdn(m)?;
            push_tlv(&mut acc, "01", &normalized);
        }
        PromptPayTarget::NationalId(id) => {
            if id.len() != 13 || !id.chars().all(|c| c.is_ascii_digit()) {
                return Err(crate::Error::Mpm(
                    "national id must be 13 digits".into(),
                ));
            }
            push_tlv(&mut acc, "02", id);
        }
        PromptPayTarget::EWalletId(id) => {
            if id.len() != 15 || !id.chars().all(|c| c.is_ascii_digit()) {
                return Err(crate::Error::Mpm(
                    "e-wallet id must be 15 digits".into(),
                ));
            }
            push_tlv(&mut acc, "03", id);
        }
    }
    push_tlv(&mut out, "29", &acc);

    // Tag 52: Merchant Category Code. "0000" if unspecified.
    push_tlv(&mut out, "52", "0000");
    // Tag 53: Transaction Currency. ISO 4217 numeric. THB = 764.
    push_tlv(&mut out, "53", "764");
    // Tag 54: Transaction Amount, expressed in major units with
    // optional fractional digits. PromptPay amounts always carry
    // two decimal places.
    let major = intent.amount.minor_units / 100;
    let minor = intent.amount.minor_units % 100;
    let amt = format!("{major}.{minor:02}");
    push_tlv(&mut out, "54", &amt);
    // Tag 58: Country Code. ISO 3166-1 alpha-2.
    push_tlv(&mut out, "58", "TH");
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

    // Tag 63: CRC-16/CCITT-FALSE over the entire payload up to and
    // including the literal "6304".
    out.push_str("6304");
    let crc = crc_ccitt_false(out.as_bytes());
    out.push_str(&format!("{crc:04X}"));
    Ok(out)
}

/// Push a TLV chunk: two-digit tag + two-digit length + value.
fn push_tlv(buf: &mut String, tag: &str, value: &str) {
    let len = value.len();
    buf.push_str(tag);
    buf.push_str(&format!("{len:02}"));
    buf.push_str(value);
}

/// Normalize a Thai mobile number into the BoT-required form:
/// strip leading `+`, `00`, or `0`, then prefix with `0066`.
fn normalize_msisdn(m: &str) -> Result<String> {
    let digits: String = m.chars().filter(|c| c.is_ascii_digit()).collect();
    let body = if let Some(rest) = digits.strip_prefix("66") {
        rest.to_string()
    } else if let Some(rest) = digits.strip_prefix('0') {
        rest.to_string()
    } else {
        digits
    };
    if body.is_empty() || body.len() > 11 {
        return Err(crate::Error::Mpm(format!("bad MSISDN: {m}")));
    }
    Ok(format!("0066{body}"))
}

/// CRC-16/CCITT-FALSE: poly 0x1021, init 0xFFFF, no reflection,
/// no final XOR. EMVCo MPM spec §4.7.4.
pub fn crc_ccitt_false(data: &[u8]) -> u16 {
    let mut crc: u16 = 0xFFFF;
    for &b in data {
        crc ^= u16::from(b) << 8;
        for _ in 0..8 {
            if crc & 0x8000 != 0 {
                crc = (crc << 1) ^ 0x1021;
            } else {
                crc <<= 1;
            }
        }
    }
    crc
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::Money;

    #[test]
    fn crc_known_vector() {
        // EMVCo published test vector: CRC of "123456789" = 0x29B1.
        assert_eq!(crc_ccitt_false(b"123456789"), 0x29B1);
    }

    #[test]
    fn mobile_encodes_to_valid_mpm() {
        let adapter = PromptPayAdapter::new(PromptPayTarget::Mobile("0812345678".into()));
        let thb = Currency::try_new(*b"THB", 2).unwrap();
        let intent = ChargeIntent {
            merchant_order_id: "ord-1".into(),
            amount: Money::from_minor(12_345, thb),
            description: "test".into(),
            presentment: PresentmentMode::MerchantPresentedQr,
            consumer_hint: None,
            notify_url: None,
        };
        let res = adapter.create_charge(&intent).unwrap();
        assert!(res.presentment_payload.starts_with("000201"));
        assert!(res.presentment_payload.contains("123.45"));
        // CRC tag at end: "6304" + 4 hex chars.
        let end = &res.presentment_payload[res.presentment_payload.len() - 8..];
        assert!(end.starts_with("6304"));
    }
}
