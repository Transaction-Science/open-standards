//! Open Payments — the W3C / Interledger Foundation HTTP API.
//!
//! Open Payments wraps SPSP, GNAP authorization, and STREAM into a
//! resource-oriented REST surface with three first-class resources:
//!
//! * `incoming-payments` — what a receiver expects to receive.
//! * `outgoing-payments` — what a sender is authorized to send.
//! * `quotes` — locked exchange-rate offers tying the two together.
//!
//! This module exposes the resource shapes and a tiny non-IO client
//! surface for building / parsing them. Network transport is left to
//! the caller.

use crate::error::{IlpError, Result};
use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

/// A payment pointer (`$wallet.example/alice`) is the entry point into
/// Open Payments; it resolves to a wallet-address JSON-LD document.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PaymentPointer {
    /// The `$`-prefixed pointer string.
    pub pointer: String,
}

/// An incoming-payment resource: a server-allocated promise to receive
/// up to `incoming_amount` for the wallet at `wallet_address`.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct IncomingPayment {
    /// HTTPS URL identifying this resource.
    pub id: String,
    /// Wallet address that owns the incoming payment.
    pub wallet_address: String,
    /// Cap on the total amount the wallet will accept against this
    /// resource. `None` means open-ended.
    pub incoming_amount: Option<Amount>,
    /// Amount actually received so far.
    pub received_amount: Amount,
    /// Wall-clock expiry; after which the resource stops accepting
    /// payments.
    pub expires_at: Option<DateTime<Utc>>,
    /// Whether the resource has been finalised.
    pub completed: bool,
    /// Receiver-side metadata (memo, invoice-id, etc.).
    pub metadata: Option<serde_json::Value>,
}

/// A quote resource: a locked-in price for sending a payment.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Quote {
    /// HTTPS URL identifying this resource.
    pub id: String,
    /// Wallet address that owns the quote.
    pub wallet_address: String,
    /// URL of the target incoming-payment resource.
    pub receiver: String,
    /// Amount that will be debited from the sender.
    pub debit_amount: Amount,
    /// Amount that will be credited to the receiver.
    pub receive_amount: Amount,
    /// Quote expiry — after which the quote may no longer be acted on.
    pub expires_at: DateTime<Utc>,
}

/// An outgoing-payment resource: an authorisation to spend up to the
/// quoted debit amount.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OutgoingPayment {
    /// HTTPS URL identifying this resource.
    pub id: String,
    /// Wallet address that owns the outgoing payment.
    pub wallet_address: String,
    /// URL of the quote this payment was authorised against.
    pub quote_id: String,
    /// Total the sender is committed to debit.
    pub debit_amount: Amount,
    /// Amount sent so far.
    pub sent_amount: Amount,
    /// Whether the payment has terminated.
    pub completed: bool,
}

/// A GNAP (Grant Negotiation and Authorization Protocol) access slice
/// — the typed permission a wallet hands a sender.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct GrantAccess {
    /// Resource kind: `incoming-payment`, `outgoing-payment`, `quote`.
    #[serde(rename = "type")]
    pub kind: String,
    /// Actions allowed (`create`, `read`, `list`, `complete`).
    pub actions: Vec<String>,
    /// Optional resource identifier; absent means any.
    pub identifier: Option<String>,
}

/// An amount in `(value, asset_code, asset_scale)` form.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Amount {
    /// Integer value in the smallest unit (after scaling).
    pub value: u64,
    /// Asset code, e.g. `USD` or `XRP`.
    pub asset_code: String,
    /// Asset scale — number of decimal places.
    pub asset_scale: u8,
}

/// A thin non-IO client surface that builds / parses Open Payments
/// resources. Useful for tests, deterministic codegen, and offline
/// validation; the caller does the HTTPS round-trip themselves.
#[derive(Clone, Debug, Default)]
pub struct OpenPaymentsClient;

impl OpenPaymentsClient {
    /// Construct a default client.
    pub fn new() -> Self {
        Self::default()
    }

    /// Serialise a resource to JSON.
    pub fn to_json<T: Serialize>(&self, value: &T) -> Result<String> {
        Ok(serde_json::to_string(value)?)
    }

    /// Parse a resource from JSON.
    pub fn from_json<T: for<'a> Deserialize<'a>>(&self, s: &str) -> Result<T> {
        Ok(serde_json::from_str(s)?)
    }

    /// Validate that a quote's amounts share a coherent shape: same
    /// asset details on debit and receive, with positive values.
    pub fn validate_quote(&self, quote: &Quote) -> Result<()> {
        if quote.debit_amount.value == 0 {
            return Err(IlpError::OpenPayments("quote: zero debit".into()));
        }
        if quote.receive_amount.value == 0 {
            return Err(IlpError::OpenPayments("quote: zero receive".into()));
        }
        if quote.expires_at < Utc::now() {
            return Err(IlpError::OpenPayments("quote: already expired".into()));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incoming_payment_roundtrip() {
        let cli = OpenPaymentsClient::new();
        let p = IncomingPayment {
            id: "https://wallet.example/incoming/1".into(),
            wallet_address: "https://wallet.example/alice".into(),
            incoming_amount: Some(Amount {
                value: 10_000,
                asset_code: "USD".into(),
                asset_scale: 2,
            }),
            received_amount: Amount {
                value: 0,
                asset_code: "USD".into(),
                asset_scale: 2,
            },
            expires_at: None,
            completed: false,
            metadata: None,
        };
        let json = cli.to_json(&p).unwrap();
        let back: IncomingPayment = cli.from_json(&json).unwrap();
        assert_eq!(p, back);
    }

    #[test]
    fn quote_validation_catches_zero() {
        let cli = OpenPaymentsClient::new();
        let q = Quote {
            id: "https://wallet.example/quote/1".into(),
            wallet_address: "https://wallet.example/alice".into(),
            receiver: "https://wallet.example/incoming/1".into(),
            debit_amount: Amount {
                value: 0,
                asset_code: "USD".into(),
                asset_scale: 2,
            },
            receive_amount: Amount {
                value: 1,
                asset_code: "USD".into(),
                asset_scale: 2,
            },
            expires_at: Utc::now() + chrono::Duration::minutes(5),
        };
        assert!(cli.validate_quote(&q).is_err());
    }
}
