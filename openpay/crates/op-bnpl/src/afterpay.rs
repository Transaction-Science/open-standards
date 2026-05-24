//! Afterpay / Clearpay Online API driver.
//!
//! Verified against the public Afterpay Online API reference at
//! <https://developers.afterpay.com/afterpay-online/reference/overview-1>.
//!
//! ## Endpoints used
//!
//! | OpenPay call | HTTP                                          |
//! |--------------|-----------------------------------------------|
//! | `initiate`   | `POST /v2/checkouts`                          |
//! | `authorize`  | `POST /v2/payments/auth`                      |
//! | `capture`    | `POST /v2/payments/{token}/capture`           |
//! | `void`       | `POST /v2/payments/{token}/void`              |
//! | `refund`     | `POST /v2/payments/{token}/refund`            |
//!
//! ## Authentication
//!
//! HTTP Basic with `(merchant_id, secret_key)` — Afterpay's docs name
//! the values "Merchant ID" and "Secret Key".
//!
//! ## Regions
//!
//! Afterpay (US/AU/NZ/CA) and Clearpay (UK/EU) share one API surface
//! but live behind different region-specific base URLs.
//!
//! ## Amounts
//!
//! Afterpay sends amounts as decimal strings with explicit currency
//! (`{"amount":"12.34","currency":"USD"}`). We convert
//! `Money.minor_units` to/from the wire string.

use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use op_core::Money;
use reqwest::Client;
use serde::{Deserialize, Serialize};

use crate::acquirer::{
    AuthorizedCheckout, BnplAcquirer, CapturedCheckout, InitiatedSession, RefundedCheckout,
};
use crate::error::{Error, Result};
use crate::intent::BnplIntent;
use crate::lifecycle::{BnplProvider, InstalmentInterval, InstalmentPlan};

/// Afterpay's regional footprint. Different base URLs and accepted
/// currencies per region.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AfterpayRegion {
    /// United States. USD only.
    Us,
    /// Australia + New Zealand. AUD / NZD.
    AuNz,
    /// United Kingdom (Clearpay brand). GBP.
    Gb,
    /// Canada. CAD.
    Ca,
    /// European Union (Clearpay brand). EUR.
    Eu,
}

impl AfterpayRegion {
    /// Production base URL for this region.
    #[must_use]
    pub const fn base_url(self) -> &'static str {
        match self {
            Self::Us | Self::Ca | Self::AuNz => "https://api.afterpay.com",
            Self::Gb | Self::Eu => "https://api.clearpay.co.uk",
        }
    }

    /// Sandbox base URL for this region.
    #[must_use]
    pub const fn sandbox_url(self) -> &'static str {
        match self {
            Self::Us | Self::Ca | Self::AuNz => "https://api.us-sandbox.afterpay.com",
            Self::Gb | Self::Eu => "https://api-sandbox.clearpay.co.uk",
        }
    }
}

/// Afterpay/Clearpay acquirer.
#[derive(Clone, Debug)]
pub struct AfterpayAcquirer {
    client: Client,
    merchant_id: String,
    secret_key: String,
    region: AfterpayRegion,
    base_url: String,
}

impl AfterpayAcquirer {
    /// Construct against the region's production URL.
    #[must_use]
    pub fn production(
        client: Client,
        merchant_id: impl Into<String>,
        secret_key: impl Into<String>,
        region: AfterpayRegion,
    ) -> Self {
        let base_url = region.base_url().to_owned();
        Self {
            client,
            merchant_id: merchant_id.into(),
            secret_key: secret_key.into(),
            region,
            base_url,
        }
    }

    /// Construct against the region's sandbox URL.
    #[must_use]
    pub fn sandbox(
        client: Client,
        merchant_id: impl Into<String>,
        secret_key: impl Into<String>,
        region: AfterpayRegion,
    ) -> Self {
        let base_url = region.sandbox_url().to_owned();
        Self {
            client,
            merchant_id: merchant_id.into(),
            secret_key: secret_key.into(),
            region,
            base_url,
        }
    }

    /// Construct against an arbitrary base URL (for test mocks).
    #[must_use]
    pub fn with_base_url(
        client: Client,
        merchant_id: impl Into<String>,
        secret_key: impl Into<String>,
        region: AfterpayRegion,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            client,
            merchant_id: merchant_id.into(),
            secret_key: secret_key.into(),
            region,
            base_url: base_url.into(),
        }
    }

    /// Which region this acquirer is bound to.
    #[must_use]
    pub const fn region(&self) -> AfterpayRegion {
        self.region
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url.trim_end_matches('/'), path)
    }

    fn basic_auth(&self) -> String {
        let raw = format!("{}:{}", self.merchant_id, self.secret_key);
        format!("Basic {}", B64.encode(raw.as_bytes()))
    }
}

#[async_trait]
impl BnplAcquirer for AfterpayAcquirer {
    fn provider(&self) -> BnplProvider {
        BnplProvider::AfterpayClearpay
    }

    async fn initiate(&self, intent: &BnplIntent) -> Result<InitiatedSession> {
        intent.validate()?;
        let body = wire::CreateCheckout {
            amount: wire::AmountWire::from_money(intent.amount),
            consumer: wire::Consumer {
                email: intent.consumer.email.clone(),
                given_name: intent.consumer.given_name.clone(),
                surname: intent.consumer.family_name.clone(),
                phone_number: intent.consumer.phone.clone(),
            },
            billing: wire::Contact::from_shipping(&intent.billing),
            shipping: wire::Contact::from_shipping(&intent.shipping),
            items: intent
                .line_items
                .iter()
                .map(|li| wire::Item {
                    name: li.name.clone(),
                    sku: li.sku.clone(),
                    quantity: li.quantity,
                    price: wire::AmountWire::from_money(li.unit_price),
                })
                .collect(),
            merchant: wire::MerchantUrls {
                redirect_confirm_url: intent.redirect_urls.success.clone(),
                redirect_cancel_url: intent.redirect_urls.cancel.clone(),
            },
        };
        let url = self.url("/v2/checkouts");
        let resp = self
            .client
            .post(&url)
            .header(reqwest::header::AUTHORIZATION, self.basic_auth())
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Transport(e.to_string()))?;
        let chk: wire::CheckoutResponse = parse_response(resp).await?;
        Ok(InitiatedSession {
            provider: BnplProvider::AfterpayClearpay,
            provider_ref: chk.token.clone(),
            redirect_url: chk.redirect_checkout_url,
            client_token: None,
            expires_at: None,
        })
    }

    async fn authorize(
        &self,
        session: &InitiatedSession,
        consumer_token: &str,
    ) -> Result<AuthorizedCheckout> {
        let body = wire::AuthBody {
            token: consumer_token.to_owned(),
            merchant_reference: Some(session.provider_ref.clone()),
        };
        let url = self.url("/v2/payments/auth");
        let resp = self
            .client
            .post(&url)
            .header(reqwest::header::AUTHORIZATION, self.basic_auth())
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Transport(e.to_string()))?;
        let payment: wire::PaymentResponse = parse_response(resp).await?;
        let amt = payment.amount.to_money()?;
        Ok(AuthorizedCheckout {
            provider: BnplProvider::AfterpayClearpay,
            provider_ref: payment.id,
            authorized_amount: amt,
            plan: InstalmentPlan::new(
                4,
                Money::from_minor((amt.minor_units + 3) / 4, amt.currency),
                chrono::Utc::now(),
                InstalmentInterval::Biweekly,
            ),
        })
    }

    async fn capture(
        &self,
        auth: &AuthorizedCheckout,
        amount: Option<Money>,
    ) -> Result<CapturedCheckout> {
        let amt = amount.unwrap_or(auth.authorized_amount);
        let body = wire::CaptureBody {
            amount: wire::AmountWire::from_money(amt),
        };
        let url = self.url(&format!(
            "/v2/payments/{}/capture",
            auth.provider_ref
        ));
        let resp = self
            .client
            .post(&url)
            .header(reqwest::header::AUTHORIZATION, self.basic_auth())
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Transport(e.to_string()))?;
        let payment: wire::PaymentResponse = parse_response(resp).await?;
        Ok(CapturedCheckout {
            provider: BnplProvider::AfterpayClearpay,
            provider_ref: payment.id,
            amount: payment.amount.to_money()?,
            settlement_ref: None,
        })
    }

    async fn refund(
        &self,
        captured: &CapturedCheckout,
        amount: Money,
    ) -> Result<RefundedCheckout> {
        let body = wire::CaptureBody {
            amount: wire::AmountWire::from_money(amount),
        };
        let url = self.url(&format!(
            "/v2/payments/{}/refund",
            captured.provider_ref
        ));
        let resp = self
            .client
            .post(&url)
            .header(reqwest::header::AUTHORIZATION, self.basic_auth())
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Transport(e.to_string()))?;
        let refund: wire::RefundResponse = parse_response(resp).await?;
        Ok(RefundedCheckout {
            provider: BnplProvider::AfterpayClearpay,
            provider_ref: captured.provider_ref.clone(),
            refund_ref: refund.refund_id,
            amount,
        })
    }

    async fn void(&self, auth: &AuthorizedCheckout) -> Result<()> {
        let url = self.url(&format!(
            "/v2/payments/{}/void",
            auth.provider_ref
        ));
        let resp = self
            .client
            .post(&url)
            .header(reqwest::header::AUTHORIZATION, self.basic_auth())
            .send()
            .await
            .map_err(|e| Error::Transport(e.to_string()))?;
        if !resp.status().is_success() {
            return Err(map_error(resp).await);
        }
        Ok(())
    }
}

async fn parse_response<T: serde::de::DeserializeOwned>(
    resp: reqwest::Response,
) -> Result<T> {
    if resp.status().is_success() {
        resp.json::<T>()
            .await
            .map_err(|e| Error::Parse(e.to_string()))
    } else {
        Err(map_error(resp).await)
    }
}

async fn map_error(resp: reqwest::Response) -> Error {
    let status = resp.status().as_u16();
    let body = resp.text().await.unwrap_or_default();
    if let Ok(env) = serde_json::from_str::<wire::ErrorEnvelope>(&body) {
        Error::ProviderRejected {
            status,
            code: env.error_code.unwrap_or_else(|| "unknown".into()),
            message: env.message.unwrap_or_default(),
        }
    } else {
        Error::ProviderRejected {
            status,
            code: "unknown".into(),
            message: body,
        }
    }
}

/// Wire-format types.
#[allow(missing_docs)] // internal wire shapes; field semantics documented at provider
pub mod wire {
    use op_core::{Currency, Money};
    use serde::{Deserialize, Serialize};

    use crate::error::Error;

    /// Afterpay amount: decimal string + alpha-3 currency.
    #[derive(Serialize, Deserialize, Debug, Clone)]
    pub struct AmountWire {
        pub amount: String,
        pub currency: String,
    }

    impl AmountWire {
        /// Convert `Money` minor units → decimal string.
        #[must_use]
        pub fn from_money(m: Money) -> Self {
            let exp = u32::from(m.currency.exponent());
            let s = if exp == 0 {
                m.minor_units.to_string()
            } else {
                let divisor = 10_i64.pow(exp);
                let whole = m.minor_units / divisor;
                let frac = (m.minor_units % divisor).abs();
                format!("{whole}.{frac:0width$}", width = exp as usize)
            };
            Self {
                amount: s,
                currency: m.currency.code().to_owned(),
            }
        }

        /// Parse decimal-string + currency back into `Money`.
        ///
        /// # Errors
        /// `Error::Parse` if the decimal string is malformed.
        pub fn to_money(&self) -> crate::Result<Money> {
            let cur = currency_by_code(&self.currency);
            let exp = u32::from(cur.exponent());
            let (whole_part, frac_part) = self
                .amount
                .split_once('.')
                .map_or((self.amount.as_str(), ""), |(w, f)| (w, f));
            let whole: i64 = whole_part
                .parse()
                .map_err(|e: std::num::ParseIntError| Error::Parse(e.to_string()))?;
            let divisor = 10_i64.pow(exp);
            if exp == 0 {
                return Ok(Money::from_minor(whole, cur));
            }
            let mut frac_str = frac_part.to_owned();
            // pad / truncate to exp digits
            while frac_str.len() < exp as usize {
                frac_str.push('0');
            }
            frac_str.truncate(exp as usize);
            let frac: i64 = if frac_str.is_empty() {
                0
            } else {
                frac_str
                    .parse()
                    .map_err(|e: std::num::ParseIntError| Error::Parse(e.to_string()))?
            };
            let sign = if whole < 0 { -1 } else { 1 };
            let minor = whole
                .checked_mul(divisor)
                .ok_or_else(|| Error::Parse("amount overflow".into()))?
                .checked_add(sign * frac)
                .ok_or_else(|| Error::Parse("amount overflow".into()))?;
            Ok(Money::from_minor(minor, cur))
        }
    }

    /// Resolve alpha-3 → Currency with a fallback table.
    fn currency_by_code(code: &str) -> Currency {
        match code {
            "USD" => Currency::USD,
            "EUR" => Currency::EUR,
            "GBP" => Currency::GBP,
            "JPY" => Currency::JPY,
            "INR" => Currency::INR,
            "BRL" => Currency::BRL,
            "CNY" => Currency::CNY,
            _ => {
                if code.len() == 3
                    && code.bytes().all(|b| b.is_ascii_uppercase())
                {
                    let b = code.as_bytes();
                    Currency::try_new([b[0], b[1], b[2]], 2).unwrap_or(Currency::USD)
                } else {
                    Currency::USD
                }
            }
        }
    }

    /// `POST /v2/checkouts` body.
    #[derive(Serialize, Debug, Clone)]
    pub struct CreateCheckout {
        pub amount: AmountWire,
        pub consumer: Consumer,
        pub billing: Contact,
        pub shipping: Contact,
        pub items: Vec<Item>,
        pub merchant: MerchantUrls,
    }

    /// Consumer record.
    #[derive(Serialize, Debug, Clone)]
    pub struct Consumer {
        pub email: String,
        #[serde(rename = "givenNames", skip_serializing_if = "Option::is_none")]
        pub given_name: Option<String>,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub surname: Option<String>,
        #[serde(rename = "phoneNumber", skip_serializing_if = "Option::is_none")]
        pub phone_number: Option<String>,
    }

    /// Address contact (Afterpay's billing/shipping schema).
    #[derive(Serialize, Debug, Clone)]
    pub struct Contact {
        pub name: String,
        pub line1: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub line2: Option<String>,
        pub area1: String,
        pub region: String,
        #[serde(rename = "postcode")]
        pub postal_code: String,
        #[serde(rename = "countryCode")]
        pub country_code: String,
    }

    impl Contact {
        /// Build from our `ShippingInfo`.
        #[must_use]
        pub fn from_shipping(s: &crate::intent::ShippingInfo) -> Self {
            Self {
                name: s.name.clone(),
                line1: s.line1.clone(),
                line2: s.line2.clone(),
                area1: s.city.clone(),
                region: s.region.clone(),
                postal_code: s.postal_code.clone(),
                country_code: s.country.clone(),
            }
        }
    }

    /// Cart item.
    #[derive(Serialize, Debug, Clone)]
    pub struct Item {
        pub name: String,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub sku: Option<String>,
        pub quantity: u32,
        pub price: AmountWire,
    }

    /// Merchant redirect URLs (Afterpay's `merchant` object).
    #[derive(Serialize, Debug, Clone)]
    pub struct MerchantUrls {
        #[serde(rename = "redirectConfirmUrl")]
        pub redirect_confirm_url: String,
        #[serde(rename = "redirectCancelUrl")]
        pub redirect_cancel_url: String,
    }

    /// Response from `/v2/checkouts`.
    #[derive(Deserialize, Debug, Clone)]
    pub struct CheckoutResponse {
        pub token: String,
        #[serde(default, rename = "redirectCheckoutUrl")]
        pub redirect_checkout_url: Option<String>,
    }

    /// `POST /v2/payments/auth` body.
    #[derive(Serialize, Debug, Clone)]
    pub struct AuthBody {
        pub token: String,
        #[serde(rename = "merchantReference", skip_serializing_if = "Option::is_none")]
        pub merchant_reference: Option<String>,
    }

    /// Payment response (auth / capture).
    #[derive(Deserialize, Debug, Clone)]
    pub struct PaymentResponse {
        pub id: String,
        pub amount: AmountWire,
        #[serde(default)]
        pub status: Option<String>,
    }

    /// `POST /v2/payments/{token}/capture|refund` body.
    #[derive(Serialize, Debug, Clone)]
    pub struct CaptureBody {
        pub amount: AmountWire,
    }

    /// Refund response.
    #[derive(Deserialize, Debug, Clone)]
    pub struct RefundResponse {
        #[serde(rename = "refundId", alias = "id")]
        pub refund_id: String,
        pub amount: AmountWire,
    }

    /// Error envelope.
    #[derive(Deserialize, Debug)]
    pub struct ErrorEnvelope {
        #[serde(default, rename = "errorCode")]
        pub error_code: Option<String>,
        #[serde(default)]
        pub message: Option<String>,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::Currency;

    #[test]
    fn region_urls_resolved() {
        assert_eq!(
            AfterpayRegion::Us.base_url(),
            "https://api.afterpay.com"
        );
        assert_eq!(
            AfterpayRegion::Gb.base_url(),
            "https://api.clearpay.co.uk"
        );
        assert!(AfterpayRegion::Us.sandbox_url().contains("sandbox"));
    }

    #[test]
    fn amount_wire_roundtrip_usd() {
        let m = Money::from_minor(12_345, Currency::USD);
        let w = wire::AmountWire::from_money(m);
        assert_eq!(w.amount, "123.45");
        assert_eq!(w.currency, "USD");
        let back = w.to_money().unwrap();
        assert_eq!(back.minor_units, 12_345);
    }

    #[test]
    fn amount_wire_roundtrip_jpy_no_decimals() {
        let m = Money::from_minor(1000, Currency::JPY);
        let w = wire::AmountWire::from_money(m);
        assert_eq!(w.amount, "1000");
        let back = w.to_money().unwrap();
        assert_eq!(back.minor_units, 1000);
    }

    #[test]
    fn amount_wire_parses_short_fraction() {
        let w = wire::AmountWire {
            amount: "12.3".into(),
            currency: "USD".into(),
        };
        let back = w.to_money().unwrap();
        assert_eq!(back.minor_units, 1230);
    }

    #[test]
    fn amount_wire_truncates_overlong_fraction() {
        let w = wire::AmountWire {
            amount: "12.3456".into(),
            currency: "USD".into(),
        };
        let back = w.to_money().unwrap();
        assert_eq!(back.minor_units, 1234);
    }

    #[test]
    fn provider_is_afterpay() {
        let a = AfterpayAcquirer::sandbox(Client::new(), "m", "k", AfterpayRegion::Us);
        assert_eq!(a.provider(), BnplProvider::AfterpayClearpay);
        assert_eq!(a.region(), AfterpayRegion::Us);
    }

    #[test]
    fn basic_auth_header_format() {
        let a = AfterpayAcquirer::sandbox(Client::new(), "mid", "sk", AfterpayRegion::Us);
        // mid:sk → bWlkOnNr
        assert_eq!(a.basic_auth(), "Basic bWlkOnNr");
    }
}
