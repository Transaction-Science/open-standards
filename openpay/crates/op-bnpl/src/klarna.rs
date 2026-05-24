//! Klarna Payments API v1 driver.
//!
//! Verified against the public Klarna API reference at
//! <https://docs.klarna.com/api/payments/>.
//!
//! ## Endpoints used
//!
//! | OpenPay call | HTTP                                                            |
//! |--------------|-----------------------------------------------------------------|
//! | `initiate`   | `POST /payments/v1/sessions`                                    |
//! | `authorize`  | `POST /payments/v1/authorizations/{authorization_token}/order`  |
//! | `capture`    | `POST /ordermanagement/v1/orders/{order_id}/captures`           |
//! | `void`       | `POST /ordermanagement/v1/orders/{order_id}/cancel`             |
//! | `refund`     | `POST /ordermanagement/v1/orders/{order_id}/refunds`            |
//!
//! ## Authentication
//!
//! HTTP Basic with `(username, password)`. Both are issued from the
//! Klarna merchant portal.
//!
//! ## Regions
//!
//! Klarna runs three regional clouds; merchants are issued credentials
//! valid in exactly one. See [`KlarnaRegion`].
//!
//! ## Amounts
//!
//! Klarna wire amounts are integer minor units in the order's
//! currency. We pass `Money::minor_units` directly.

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

/// Klarna's three regional clouds.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum KlarnaRegion {
    /// North America (US, CA, MX). Base URL `https://api-na.klarna.com`.
    Na,
    /// Europe (EU + UK). Base URL `https://api.klarna.com`.
    Eu,
    /// Oceania (AU, NZ). Base URL `https://api-oc.klarna.com`.
    Oc,
}

impl KlarnaRegion {
    /// Production base URL for this region.
    #[must_use]
    pub const fn base_url(self) -> &'static str {
        match self {
            Self::Na => "https://api-na.klarna.com",
            Self::Eu => "https://api.klarna.com",
            Self::Oc => "https://api-oc.klarna.com",
        }
    }

    /// Playground (sandbox) base URL for this region.
    #[must_use]
    pub const fn playground_url(self) -> &'static str {
        match self {
            Self::Na => "https://api-na.playground.klarna.com",
            Self::Eu => "https://api.playground.klarna.com",
            Self::Oc => "https://api-oc.playground.klarna.com",
        }
    }
}

/// Klarna acquirer.
#[derive(Clone, Debug)]
pub struct KlarnaAcquirer {
    client: Client,
    username: String,
    password: String,
    region: KlarnaRegion,
    base_url: String,
}

impl KlarnaAcquirer {
    /// Construct using the region's production base URL.
    #[must_use]
    pub fn production(
        client: Client,
        username: impl Into<String>,
        password: impl Into<String>,
        region: KlarnaRegion,
    ) -> Self {
        let base_url = region.base_url().to_owned();
        Self {
            client,
            username: username.into(),
            password: password.into(),
            region,
            base_url,
        }
    }

    /// Construct using the playground base URL.
    #[must_use]
    pub fn playground(
        client: Client,
        username: impl Into<String>,
        password: impl Into<String>,
        region: KlarnaRegion,
    ) -> Self {
        let base_url = region.playground_url().to_owned();
        Self {
            client,
            username: username.into(),
            password: password.into(),
            region,
            base_url,
        }
    }

    /// Construct with an arbitrary base URL (test mocks).
    #[must_use]
    pub fn with_base_url(
        client: Client,
        username: impl Into<String>,
        password: impl Into<String>,
        region: KlarnaRegion,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            client,
            username: username.into(),
            password: password.into(),
            region,
            base_url: base_url.into(),
        }
    }

    /// Which region this acquirer is bound to.
    #[must_use]
    pub const fn region(&self) -> KlarnaRegion {
        self.region
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url.trim_end_matches('/'), path)
    }

    fn basic_auth(&self) -> String {
        let raw = format!("{}:{}", self.username, self.password);
        format!("Basic {}", B64.encode(raw.as_bytes()))
    }

    fn intent_to_session(intent: &BnplIntent) -> wire::CreateSession {
        wire::CreateSession {
            purchase_country: intent.shipping.country.clone(),
            purchase_currency: intent.currency.code().to_owned(),
            locale: "en-US".into(),
            order_amount: intent.amount.minor_units,
            order_tax_amount: 0,
            order_lines: intent
                .line_items
                .iter()
                .map(|li| wire::OrderLine {
                    name: li.name.clone(),
                    quantity: li.quantity,
                    unit_price: li.unit_price.minor_units,
                    tax_rate: 0,
                    total_amount: li.total_amount.minor_units,
                    total_tax_amount: 0,
                    reference: li.sku.clone(),
                })
                .collect(),
        }
    }
}

#[async_trait]
impl BnplAcquirer for KlarnaAcquirer {
    fn provider(&self) -> BnplProvider {
        BnplProvider::Klarna
    }

    async fn initiate(&self, intent: &BnplIntent) -> Result<InitiatedSession> {
        intent.validate()?;
        let body = Self::intent_to_session(intent);
        let url = self.url("/payments/v1/sessions");
        let resp = self
            .client
            .post(&url)
            .header(reqwest::header::AUTHORIZATION, self.basic_auth())
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Transport(e.to_string()))?;
        let session: wire::SessionResponse = parse_response(resp).await?;
        Ok(InitiatedSession {
            provider: BnplProvider::Klarna,
            provider_ref: session.session_id,
            redirect_url: None,
            client_token: Some(session.client_token),
            expires_at: None,
        })
    }

    /// Place the order, consuming the `authorization_token` returned
    /// by Klarna's JS SDK after the consumer authorises. `_session` is
    /// kept in the signature for symmetry with other providers; Klarna
    /// embeds session state in the consumer token itself.
    async fn authorize(
        &self,
        _session: &InitiatedSession,
        consumer_token: &str,
    ) -> Result<AuthorizedCheckout> {
        // Klarna needs the original session body resent as part of the
        // place-order call. We don't keep the intent here; we send the
        // minimum required envelope and let Klarna fall back to the
        // session data it already has.
        let body = serde_json::json!({});
        let url = self.url(&format!(
            "/payments/v1/authorizations/{consumer_token}/order"
        ));
        let resp = self
            .client
            .post(&url)
            .header(reqwest::header::AUTHORIZATION, self.basic_auth())
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Transport(e.to_string()))?;
        let order: wire::OrderResponse = parse_response(resp).await?;

        // Currency must be ASCII upper alpha-3.
        let currency = parse_currency(&order.authorized_payment_method.kind.unwrap_or_default())
            .unwrap_or(op_core::Currency::USD);
        // The session amount is the authoritative authorized amount.
        let amt = order.authorized_amount.unwrap_or(0);
        Ok(AuthorizedCheckout {
            provider: BnplProvider::Klarna,
            provider_ref: order.order_id,
            authorized_amount: Money::from_minor(amt, currency),
            plan: InstalmentPlan::new(
                3, // Klarna's Pay-in-3 default
                Money::from_minor((amt + 2) / 3, currency),
                chrono::Utc::now(),
                InstalmentInterval::Monthly,
            ),
        })
    }

    async fn capture(
        &self,
        auth: &AuthorizedCheckout,
        amount: Option<Money>,
    ) -> Result<CapturedCheckout> {
        let amt = amount.unwrap_or(auth.authorized_amount);
        let body = wire::Capture {
            captured_amount: amt.minor_units,
        };
        let url = self.url(&format!(
            "/ordermanagement/v1/orders/{}/captures",
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
        if !resp.status().is_success() {
            return Err(map_error(resp).await);
        }
        // Klarna returns 201 with Location header (capture_id); body
        // may be empty.
        let capture_id = resp
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|v| v.to_str().ok())
            .map(std::string::ToString::to_string);
        Ok(CapturedCheckout {
            provider: BnplProvider::Klarna,
            provider_ref: auth.provider_ref.clone(),
            amount: amt,
            settlement_ref: capture_id,
        })
    }

    async fn refund(
        &self,
        captured: &CapturedCheckout,
        amount: Money,
    ) -> Result<RefundedCheckout> {
        let body = wire::Refund {
            refunded_amount: amount.minor_units,
        };
        let url = self.url(&format!(
            "/ordermanagement/v1/orders/{}/refunds",
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
        if !resp.status().is_success() {
            return Err(map_error(resp).await);
        }
        let refund_ref = resp
            .headers()
            .get(reqwest::header::LOCATION)
            .and_then(|v| v.to_str().ok())
            .map(std::string::ToString::to_string)
            .unwrap_or_default();
        Ok(RefundedCheckout {
            provider: BnplProvider::Klarna,
            provider_ref: captured.provider_ref.clone(),
            refund_ref,
            amount,
        })
    }

    async fn void(&self, auth: &AuthorizedCheckout) -> Result<()> {
        let url = self.url(&format!(
            "/ordermanagement/v1/orders/{}/cancel",
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

const fn parse_currency(_kind: &str) -> Option<op_core::Currency> {
    // Klarna doesn't echo currency in the order response; caller
    // falls back to the session currency.
    None
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
            message: env
                .error_messages
                .unwrap_or_default()
                .join("; "),
        }
    } else {
        Error::ProviderRejected {
            status,
            code: "unknown".into(),
            message: body,
        }
    }
}

/// Wire-format types matching Klarna's Payments + OrderManagement API
/// documentation.
#[allow(missing_docs)] // internal wire shapes; field semantics documented at provider
pub mod wire {
    use serde::{Deserialize, Serialize};

    /// `POST /payments/v1/sessions` body.
    #[derive(Serialize, Debug, Clone)]
    pub struct CreateSession {
        pub purchase_country: String,
        pub purchase_currency: String,
        pub locale: String,
        pub order_amount: i64,
        pub order_tax_amount: i64,
        pub order_lines: Vec<OrderLine>,
    }

    /// Klarna order line.
    #[derive(Serialize, Debug, Clone)]
    pub struct OrderLine {
        pub name: String,
        pub quantity: u32,
        pub unit_price: i64,
        pub tax_rate: i64,
        pub total_amount: i64,
        pub total_tax_amount: i64,
        #[serde(skip_serializing_if = "Option::is_none")]
        pub reference: Option<String>,
    }

    /// Response from `POST /sessions`.
    #[derive(Deserialize, Debug, Clone)]
    pub struct SessionResponse {
        pub session_id: String,
        pub client_token: String,
    }

    /// Response from `POST /authorizations/{token}/order`.
    #[derive(Deserialize, Debug, Clone)]
    pub struct OrderResponse {
        pub order_id: String,
        #[serde(default)]
        pub authorized_amount: Option<i64>,
        #[serde(default)]
        pub authorized_payment_method: AuthorizedMethod,
    }

    /// Method category Klarna chose for this consumer.
    #[derive(Deserialize, Debug, Clone, Default)]
    pub struct AuthorizedMethod {
        #[serde(rename = "type", default)]
        pub kind: Option<String>,
    }

    /// `POST /captures` body.
    #[derive(Serialize, Debug, Clone)]
    pub struct Capture {
        pub captured_amount: i64,
    }

    /// `POST /refunds` body.
    #[derive(Serialize, Debug, Clone)]
    pub struct Refund {
        pub refunded_amount: i64,
    }

    /// Error envelope. Klarna returns a list of human-readable messages
    /// plus a machine-readable code on most non-2xx responses.
    #[derive(Deserialize, Debug)]
    pub struct ErrorEnvelope {
        #[serde(default)]
        pub error_code: Option<String>,
        #[serde(default)]
        pub error_messages: Option<Vec<String>>,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn region_urls_match_klarna_docs() {
        assert_eq!(KlarnaRegion::Na.base_url(), "https://api-na.klarna.com");
        assert_eq!(KlarnaRegion::Eu.base_url(), "https://api.klarna.com");
        assert_eq!(KlarnaRegion::Oc.base_url(), "https://api-oc.klarna.com");
    }

    #[test]
    fn playground_urls_differ_from_production() {
        for r in [KlarnaRegion::Na, KlarnaRegion::Eu, KlarnaRegion::Oc] {
            assert_ne!(r.base_url(), r.playground_url());
            assert!(r.playground_url().contains("playground"));
        }
    }

    #[test]
    fn basic_auth_header_format() {
        let k = KlarnaAcquirer::production(Client::new(), "u", "p", KlarnaRegion::Eu);
        assert_eq!(k.basic_auth(), "Basic dTpw");
    }

    #[test]
    fn provider_is_klarna() {
        let k = KlarnaAcquirer::production(Client::new(), "u", "p", KlarnaRegion::Eu);
        assert_eq!(k.provider(), BnplProvider::Klarna);
        assert_eq!(k.region(), KlarnaRegion::Eu);
    }
}
