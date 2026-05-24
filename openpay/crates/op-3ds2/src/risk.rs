//! 3DS Requestor risk-data envelopes.
//!
//! Three flavours of risk evidence travel inside the AReq:
//!
//! - [`AccountInfo`] — what the requestor knows about the cardholder's
//!   account on the merchant side: age, last-change date, password
//!   change history, number of past-six-month orders.
//! - [`BrowserInfo`] — what the browser tells us: user-agent, screen
//!   geometry, language, timezone, accept headers.
//! - [`MerchantRiskIndicator`] — what the merchant knows about the
//!   purchase itself: gift-card amount, shipping mode, delivery
//!   timeframe, address mismatch.
//!
//! All three feed the ACS's risk model. A rich and consistent risk
//! envelope is the single biggest lever for frictionless flow: ACSes
//! suppress the challenge when the risk envelope makes a confident
//! "low risk" picture.

use serde::{Deserialize, Serialize};

/// Account-history envelope, EMVCo 6.2.2.5 `acctInfo`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AccountInfo {
    /// Days since account creation, EMVCo-coded:
    /// `"01"` no account (guest), `"02"` < 30 days, `"03"` 30-60 days,
    /// `"04"` > 60 days.
    pub ch_acc_age_ind: Option<String>,
    /// Date account was created. `YYYYMMDD`.
    pub ch_acc_date: Option<String>,
    /// Days since account last changed. `"01"` < 30 days,
    /// `"02"` 30-60 days, `"03"` > 60 days.
    pub ch_acc_change_ind: Option<String>,
    /// Date of last account change. `YYYYMMDD`.
    pub ch_acc_change: Option<String>,
    /// Days since account password change. Same encoding as
    /// `ch_acc_age_ind`.
    pub ch_acc_pw_change_ind: Option<String>,
    /// Date of last password change. `YYYYMMDD`.
    pub ch_acc_pw_change: Option<String>,
    /// Number of past-six-month orders.
    pub nb_purchase_account: Option<u32>,
    /// Provisioning-attempts past 24 hours.
    pub provision_attempts_day: Option<u32>,
    /// Number of add-card attempts past 24 hours.
    pub txn_activity_day: Option<u32>,
    /// Number of past-year transactions on this account.
    pub txn_activity_year: Option<u32>,
    /// Suspicious activity observed on the account. `"01"` none,
    /// `"02"` suspicious observed.
    pub suspicious_acc_activity: Option<String>,
    /// Days since shipping address first used on this account.
    /// `"01"` first use, `"02"` < 30 days, `"03"` 30-60 days,
    /// `"04"` > 60 days.
    pub ship_address_usage_ind: Option<String>,
    /// Date shipping address was first used. `YYYYMMDD`.
    pub ship_address_usage: Option<String>,
    /// Indicator that the cardholder name on this transaction matches
    /// the name stored on the account. `"01"` match, `"02"` mismatch.
    pub ship_name_indicator: Option<String>,
}

/// Browser-side envelope, EMVCo 6.2.2.4 `browserInformation`.
///
/// Populated by the 3DS-Method-URL JavaScript snippet at checkout
/// load. See [`crate::fingerprint`].
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct BrowserInfo {
    /// `User-Agent` header from the cardholder browser.
    #[serde(rename = "browserUserAgent")]
    pub user_agent: String,
    /// `Accept-Language` header.
    #[serde(rename = "browserLanguage")]
    pub language: String,
    /// `Accept` header.
    #[serde(rename = "browserAcceptHeader")]
    pub accept_header: String,
    /// Screen width in pixels.
    #[serde(rename = "browserScreenWidth")]
    pub screen_width: u32,
    /// Screen height in pixels.
    #[serde(rename = "browserScreenHeight")]
    pub screen_height: u32,
    /// Screen color depth, bits per pixel.
    #[serde(rename = "browserColorDepth")]
    pub color_depth: String,
    /// `Date.getTimezoneOffset()` value: minutes from UTC.
    #[serde(rename = "browserTZ")]
    pub timezone_offset: i32,
    /// `navigator.javaEnabled()` result.
    #[serde(rename = "browserJavaEnabled")]
    pub java_enabled: bool,
    /// `navigator.javascriptEnabled` (introduced 2.2.0).
    #[serde(rename = "browserJavascriptEnabled")]
    pub javascript_enabled: bool,
    /// Cardholder IP as observed by the 3DS Server.
    #[serde(rename = "browserIP", skip_serializing_if = "Option::is_none")]
    pub ip: Option<String>,
}

impl BrowserInfo {
    /// A representative populated [`BrowserInfo`] used by tests and
    /// the fingerprint round-trip examples.
    #[must_use]
    pub fn sample() -> Self {
        Self {
            user_agent: "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605".into(),
            language: "en-US".into(),
            accept_header: "text/html,application/xhtml+xml".into(),
            screen_width: 1920,
            screen_height: 1080,
            color_depth: "24".into(),
            timezone_offset: -420,
            java_enabled: false,
            javascript_enabled: true,
            ip: Some("203.0.113.42".into()),
        }
    }
}

/// `merchantRiskIndicator` envelope, EMVCo 6.2.2.6.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MerchantRiskIndicator {
    /// Hashed (SHA-256, base64) recipient email when the goods are
    /// shipped to a different address than the billing address. The
    /// spec asks for the hash, not the plaintext.
    pub delivery_email_address: Option<String>,
    /// Delivery timeframe.
    pub delivery_timeframe: Option<DeliveryTimeFrame>,
    /// Pre-paid / gift-card amount in transaction currency minor units.
    pub gift_card_amount: Option<u64>,
    /// Currency for `gift_card_amount`.
    pub gift_card_curr: Option<String>,
    /// Number of gift cards.
    pub gift_card_count: Option<u32>,
    /// Pre-order indicator: `"01"` merchandise available,
    /// `"02"` future availability.
    pub pre_order_purchase_ind: Option<String>,
    /// Pre-order date `YYYYMMDD`.
    pub pre_order_date: Option<String>,
    /// Reorder items indicator: `"01"` first time ordered,
    /// `"02"` reordered.
    pub reorder_items_ind: Option<String>,
    /// Shipping indicator: `"01"` ship to billing, `"02"` ship to
    /// verified address, `"03"` ship to different address,
    /// `"04"` store pickup, `"05"` digital goods, `"06"` travel/event
    /// no shipping, `"07"` other.
    pub shipping_indicator: Option<String>,
}

/// Delivery timeframe coding, EMVCo `deliveryTimeframe`.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeliveryTimeFrame {
    /// `"01"` Electronic delivery.
    #[serde(rename = "01")]
    Electronic,
    /// `"02"` Same-day shipping.
    #[serde(rename = "02")]
    SameDay,
    /// `"03"` Overnight shipping.
    #[serde(rename = "03")]
    Overnight,
    /// `"04"` Two-day or more shipping.
    #[serde(rename = "04")]
    TwoOrMore,
}

/// Composite envelope passed by the requestor to the `op-3ds2`
/// codec when building an AReq.
#[derive(Debug, Clone, Default)]
pub struct RequestorRiskData {
    /// What the requestor knows about the cardholder account.
    pub account_info: AccountInfo,
    /// What the browser tells us.
    pub browser_info: BrowserInfo,
    /// What the merchant knows about the purchase itself.
    pub merchant_risk: MerchantRiskIndicator,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn browser_info_round_trip() {
        let b = BrowserInfo::sample();
        let s = serde_json::to_string(&b).unwrap();
        assert!(s.contains("\"browserUserAgent\""));
        assert!(s.contains("\"browserScreenWidth\":1920"));
        let back: BrowserInfo = serde_json::from_str(&s).unwrap();
        assert_eq!(back.user_agent, b.user_agent);
        assert_eq!(back.screen_height, 1080);
    }

    #[test]
    fn delivery_timeframe_encodes_to_two_digit_string() {
        let d = DeliveryTimeFrame::SameDay;
        let s = serde_json::to_string(&d).unwrap();
        assert_eq!(s, "\"02\"");
    }

    #[test]
    fn merchant_risk_indicator_round_trip() {
        let m = MerchantRiskIndicator {
            delivery_timeframe: Some(DeliveryTimeFrame::Electronic),
            gift_card_amount: Some(5000),
            gift_card_curr: Some("840".into()),
            gift_card_count: Some(2),
            shipping_indicator: Some("05".into()),
            ..Default::default()
        };
        let s = serde_json::to_string(&m).unwrap();
        assert!(s.contains("\"giftCardAmount\":5000"));
        let back: MerchantRiskIndicator = serde_json::from_str(&s).unwrap();
        assert_eq!(back.gift_card_count, Some(2));
    }

    #[test]
    fn account_info_round_trip() {
        let a = AccountInfo {
            ch_acc_age_ind: Some("04".into()),
            ch_acc_date: Some("20240101".into()),
            nb_purchase_account: Some(7),
            ..Default::default()
        };
        let s = serde_json::to_string(&a).unwrap();
        let back: AccountInfo = serde_json::from_str(&s).unwrap();
        assert_eq!(back.ch_acc_age_ind.as_deref(), Some("04"));
        assert_eq!(back.nb_purchase_account, Some(7));
    }
}
