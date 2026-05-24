//! Affirm Direct API driver.
//!
//! Verified against the public Affirm Direct API reference at
//! <https://docs.affirm.com/developers/reference/charges-api>.
//!
//! ## Endpoints used
//!
//! | OpenPay call | HTTP                                          |
//! |--------------|-----------------------------------------------|
//! | `initiate`   | (no server call — checkout JS object created merchant-side) |
//! | `authorize`  | `POST /api/v2/charges`           (`{checkout_token}`) |
//! | `capture`    | `POST /api/v2/charges/{id}/capture`           |
//! | `void`       | `POST /api/v2/charges/{id}/void`              |
//! | `refund`     | `POST /api/v2/charges/{id}/refund`            |
//! | `fetch`      | `GET  /api/v2/charges/{id}`                   |
//!
//! ## Authentication
//!
//! HTTP Basic with `(public_key, private_key)`. Both are issued from
//! the Affirm merchant dashboard. The private key is server-side only.
//!
//! ## Amounts
//!
//! Affirm wire amounts are integer cents (USD). We pass
//! `Money::minor_units` directly — `op-core` already mandates integer
//! minor units.

use async_trait::async_trait;
use base64::{Engine as _, engine::general_purpose::STANDARD as B64};
use op_core::Money;
use reqwest::Client;

use crate::acquirer::{
    AuthorizedCheckout, BnplAcquirer, CapturedCheckout, InitiatedSession, RefundedCheckout,
};
use crate::error::{Error, Result};
use crate::intent::BnplIntent;
use crate::lifecycle::{BnplProvider, InstalmentInterval, InstalmentPlan};

/// Affirm acquirer.
///
/// `base_url` defaults to the sandbox; operators set production
/// (`https://api.affirm.com`) once they have gone live.
#[derive(Clone, Debug)]
pub struct AffirmAcquirer {
    client: Client,
    public_key: String,
    private_key: String,
    base_url: String,
}

impl AffirmAcquirer {
    /// Affirm sandbox base URL.
    pub const SANDBOX: &'static str = "https://sandbox.affirm.com";
    /// Affirm production base URL.
    pub const PRODUCTION: &'static str = "https://api.affirm.com";

    /// Construct.
    #[must_use]
    pub fn new(
        client: Client,
        public_key: impl Into<String>,
        private_key: impl Into<String>,
        base_url: impl Into<String>,
    ) -> Self {
        Self {
            client,
            public_key: public_key.into(),
            private_key: private_key.into(),
            base_url: base_url.into(),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url.trim_end_matches('/'), path)
    }

    fn basic_auth(&self) -> String {
        let raw = format!("{}:{}", self.public_key, self.private_key);
        format!("Basic {}", B64.encode(raw.as_bytes()))
    }

    /// Look up a charge by id.
    ///
    /// # Errors
    /// `Error::Transport` / `Error::ProviderRejected` / `Error::Parse`.
    pub async fn fetch(&self, charge_id: &str) -> Result<wire::Charge> {
        let url = self.url(&format!("/api/v2/charges/{charge_id}"));
        let resp = self
            .client
            .get(&url)
            .header(reqwest::header::AUTHORIZATION, self.basic_auth())
            .send()
            .await
            .map_err(|e| Error::Transport(e.to_string()))?;
        parse_response(resp).await
    }
}

#[async_trait]
impl BnplAcquirer for AffirmAcquirer {
    fn provider(&self) -> BnplProvider {
        BnplProvider::Affirm
    }

    /// Affirm's flow is client-driven up to consumer acceptance: the
    /// merchant's web page loads `affirm.js`, which opens the modal
    /// using the cart payload. There is no merchant-server step at
    /// session creation; we return a synthetic `InitiatedSession` with
    /// the intent's metadata so the rest of the trait surface is
    /// uniform across providers.
    async fn initiate(&self, intent: &BnplIntent) -> Result<InitiatedSession> {
        intent.validate()?;
        Ok(InitiatedSession {
            provider: BnplProvider::Affirm,
            // Affirm does not issue a server-side session id at this step;
            // the merchant uses the idempotency key as a local correlation
            // handle until the consumer flow completes.
            provider_ref: intent.idempotency_key.as_str().to_owned(),
            redirect_url: None,
            client_token: None,
            expires_at: None,
        })
    }

    /// Consume the `checkout_token` posted back by `affirm.js` after
    /// the consumer accepts the loan. POST it to `/charges` to create
    /// the charge.
    async fn authorize(
        &self,
        _session: &InitiatedSession,
        consumer_token: &str,
    ) -> Result<AuthorizedCheckout> {
        let body = wire::CreateCharge {
            checkout_token: consumer_token,
            order_id: None,
        };
        let url = self.url("/api/v2/charges");
        let resp = self
            .client
            .post(&url)
            .header(reqwest::header::AUTHORIZATION, self.basic_auth())
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Transport(e.to_string()))?;
        let charge: wire::Charge = parse_response(resp).await?;
        let currency = op_core::Currency::USD; // Affirm is USD-only.
        Ok(AuthorizedCheckout {
            provider: BnplProvider::Affirm,
            provider_ref: charge.id.clone(),
            authorized_amount: Money::from_minor(charge.amount, currency),
            plan: derive_plan(charge.amount, currency),
        })
    }

    async fn capture(
        &self,
        auth: &AuthorizedCheckout,
        amount: Option<Money>,
    ) -> Result<CapturedCheckout> {
        let amt = amount.unwrap_or(auth.authorized_amount);
        let body = wire::CaptureBody {
            amount: Some(amt.minor_units),
        };
        let url = self.url(&format!("/api/v2/charges/{}/capture", auth.provider_ref));
        let resp = self
            .client
            .post(&url)
            .header(reqwest::header::AUTHORIZATION, self.basic_auth())
            .json(&body)
            .send()
            .await
            .map_err(|e| Error::Transport(e.to_string()))?;
        let evt: wire::ChargeEvent = parse_response(resp).await?;
        Ok(CapturedCheckout {
            provider: BnplProvider::Affirm,
            provider_ref: auth.provider_ref.clone(),
            amount: Money::from_minor(evt.amount.unwrap_or(amt.minor_units), amt.currency),
            settlement_ref: evt.transaction_id,
        })
    }

    async fn refund(
        &self,
        captured: &CapturedCheckout,
        amount: Money,
    ) -> Result<RefundedCheckout> {
        let body = wire::RefundBody {
            amount: amount.minor_units,
        };
        let url = self.url(&format!(
            "/api/v2/charges/{}/refund",
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
        let evt: wire::ChargeEvent = parse_response(resp).await?;
        Ok(RefundedCheckout {
            provider: BnplProvider::Affirm,
            provider_ref: captured.provider_ref.clone(),
            refund_ref: evt.id.unwrap_or_default(),
            amount: Money::from_minor(evt.amount.unwrap_or(amount.minor_units), amount.currency),
        })
    }

    async fn void(&self, auth: &AuthorizedCheckout) -> Result<()> {
        let url = self.url(&format!("/api/v2/charges/{}/void", auth.provider_ref));
        let resp = self
            .client
            .post(&url)
            .header(reqwest::header::AUTHORIZATION, self.basic_auth())
            .json(&serde_json::json!({}))
            .send()
            .await
            .map_err(|e| Error::Transport(e.to_string()))?;
        let _: wire::ChargeEvent = parse_response(resp).await?;
        Ok(())
    }
}

/// Derive a representative instalment plan from a USD-cents amount.
/// Affirm's actual plan terms (APR, instalment count) are decided
/// per-consumer by their underwriter; this is the merchant-side
/// "happy default" for accounting (Pay-in-4 biweekly, no interest).
fn derive_plan(amount_cents: i64, currency: op_core::Currency) -> InstalmentPlan {
    let per = (amount_cents + 3) / 4; // round-up; provider adjusts last
    InstalmentPlan::new(
        4,
        Money::from_minor(per, currency),
        chrono::Utc::now(),
        InstalmentInterval::Biweekly,
    )
}

/// Parse a reqwest response into either the typed body or a typed error.
async fn parse_response<T: serde::de::DeserializeOwned>(
    resp: reqwest::Response,
) -> Result<T> {
    let status = resp.status();
    if status.is_success() {
        resp.json::<T>()
            .await
            .map_err(|e| Error::Parse(e.to_string()))
    } else {
        let code = status.as_u16();
        let body = resp.text().await.unwrap_or_default();
        if let Ok(env) = serde_json::from_str::<wire::ErrorEnvelope>(&body) {
            Err(Error::ProviderRejected {
                status: code,
                code: env.code.unwrap_or_else(|| "unknown".into()),
                message: env.message.unwrap_or_default(),
            })
        } else {
            Err(Error::ProviderRejected {
                status: code,
                code: "unknown".into(),
                message: body,
            })
        }
    }
}

/// Wire-format types matching Affirm's Direct API documentation.
#[allow(missing_docs)] // internal wire shapes; field semantics documented at provider
pub mod wire {
    use serde::{Deserialize, Serialize};

    /// `POST /charges` request body.
    #[derive(Serialize, Debug, Clone)]
    pub struct CreateCharge<'a> {
        /// Checkout token returned by `affirm.js` after consumer accepts.
        pub checkout_token: &'a str,
        /// Optional merchant order id, forwarded for reconciliation.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub order_id: Option<&'a str>,
    }

    /// Captured charge representation.
    #[derive(Deserialize, Serialize, Debug, Clone)]
    pub struct Charge {
        /// Affirm charge id (`CHARGE_ID` in their docs).
        pub id: String,
        /// Amount in cents.
        pub amount: i64,
        /// Status (`authorized`, `captured`, `voided`, `refunded`,
        /// `partial-refunded`, `disputed`).
        #[serde(default)]
        pub status: String,
    }

    /// `POST /charges/{id}/capture` request body.
    #[derive(Serialize, Debug, Clone)]
    pub struct CaptureBody {
        /// Amount to capture in cents. `None` means full.
        #[serde(skip_serializing_if = "Option::is_none")]
        pub amount: Option<i64>,
    }

    /// `POST /charges/{id}/refund` request body.
    #[derive(Serialize, Debug, Clone)]
    pub struct RefundBody {
        /// Amount to refund in cents.
        pub amount: i64,
    }

    /// Response shape for capture / refund / void — Affirm calls this
    /// the "transaction event".
    #[derive(Deserialize, Serialize, Debug, Clone)]
    pub struct ChargeEvent {
        /// Event id.
        #[serde(default)]
        pub id: Option<String>,
        /// Amount in cents (present on capture / refund).
        #[serde(default)]
        pub amount: Option<i64>,
        /// Bank settlement reference, when known.
        #[serde(default)]
        pub transaction_id: Option<String>,
        /// Event type: `auth`, `capture`, `void`, `refund`.
        #[serde(default, rename = "type")]
        pub kind: Option<String>,
    }

    /// Error envelope returned on non-2xx.
    #[derive(Deserialize, Debug)]
    pub struct ErrorEnvelope {
        #[serde(default)]
        pub code: Option<String>,
        #[serde(default)]
        pub message: Option<String>,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basic_auth_header_format() {
        let a = AffirmAcquirer::new(Client::new(), "pub", "priv", AffirmAcquirer::SANDBOX);
        let h = a.basic_auth();
        assert!(h.starts_with("Basic "));
        // "pub:priv" base64 = "cHViOnByaXY="
        assert_eq!(h, "Basic cHViOnByaXY=");
    }

    #[test]
    fn url_strips_trailing_slash() {
        let a = AffirmAcquirer::new(Client::new(), "p", "k", "https://x.example/");
        assert_eq!(a.url("/api/v2/charges"), "https://x.example/api/v2/charges");
    }

    #[test]
    fn provider_is_affirm() {
        let a = AffirmAcquirer::new(Client::new(), "p", "k", AffirmAcquirer::SANDBOX);
        assert_eq!(a.provider(), BnplProvider::Affirm);
    }

    #[test]
    fn derive_plan_quarters_amount_round_up() {
        let p = derive_plan(10_001, op_core::Currency::USD);
        assert_eq!(p.num_instalments, 4);
        // 10_001 / 4 round-up = 2_501 (provider adjusts last instalment)
        assert_eq!(p.instalment_amount.minor_units, 2_501);
    }
}
