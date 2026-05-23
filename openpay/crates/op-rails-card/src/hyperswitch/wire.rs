//! Hyperswitch JSON wire types.
//!
//! Field names and shapes verified against the live API reference at
//! <https://api-reference.hyperswitch.io/v1/payments/payments--create>
//! and <https://api-reference.hyperswitch.io/v1/payments/payments--confirm>.
//!
//! We deliberately do NOT model every field. Hyperswitch's request
//! schema has ~80 optional fields; we expose only what `OpenPay` needs
//! and let `serde(deny_unknown_fields)` stay OFF so we don't break when
//! Hyperswitch adds fields.

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Requests
// ---------------------------------------------------------------------------

/// `POST /payments` request body.
///
/// Verified-required fields: `amount`, `currency`. Everything else is
/// optional per the Hyperswitch V1 spec.
#[derive(Debug, Clone, Serialize)]
pub struct CreatePayment<'a> {
    /// Amount in minor units (e.g. 6540 for $65.40 USD).
    pub amount: i64,
    /// ISO 4217 alpha-3 code.
    pub currency: &'a str,
    /// `automatic` | `manual` | `manual_multiple`.
    /// Default per Hyperswitch: `automatic` (auth + capture in one call).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub capture_method: Option<&'a str>,
    /// `three_ds` | `no_three_ds`. Default: `three_ds`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authentication_type: Option<&'a str>,
    /// Merchant-provided idempotency key. 30 characters per spec.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payment_id: Option<String>,
    /// Customer info, optional.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub customer: Option<Customer<'a>>,
    /// If true and `payment_method_data` is present, Hyperswitch will
    /// confirm immediately. We always set this false on create so we
    /// can attach the method in a separate confirm call.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub confirm: Option<bool>,
    /// `card` | `wallet` | `bank_transfer` | ... (Hyperswitch enum).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payment_method: Option<&'a str>,
    /// `apple_pay` | `google_pay` | `credit` | `debit` | ...
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payment_method_type: Option<&'a str>,
    /// Reference to a tokenized payment method in Hyperswitch's vault.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payment_token: Option<&'a str>,
    /// Method-specific data (for now we use this for wallet tokens).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payment_method_data: Option<serde_json::Value>,
    /// Free-form metadata.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub metadata: Option<serde_json::Value>,
    /// Connector-specific metadata. We use this to pass EMV TLV blobs.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub connector_metadata: Option<serde_json::Value>,
}

/// `POST /payments/{id}/confirm` request body.
#[derive(Debug, Clone, Serialize)]
pub struct ConfirmPayment<'a> {
    /// Method type (e.g. `wallet`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payment_method: Option<&'a str>,
    /// Method sub-type (e.g. `apple_pay`).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payment_method_type: Option<&'a str>,
    /// Method data.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payment_method_data: Option<serde_json::Value>,
    /// Vault token reference, if confirming with a stored method.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub payment_token: Option<&'a str>,
}

/// `POST /payments/{id}/capture` request body.
#[derive(Debug, Clone, Serialize)]
pub struct CapturePayment {
    /// Amount to capture, in minor units. Must be ≤ authorized.
    pub amount_to_capture: i64,
}

/// `POST /payments/{id}/cancel` (void) request body.
#[derive(Debug, Clone, Serialize)]
pub struct CancelPayment<'a> {
    /// `requested_by_customer` | `duplicate` | `fraudulent` | other.
    pub cancellation_reason: &'a str,
}

/// `POST /refunds` request body.
#[derive(Debug, Clone, Serialize)]
pub struct CreateRefund<'a> {
    /// PSP payment id to refund against.
    pub payment_id: &'a str,
    /// Amount in minor units.
    pub amount: i64,
    /// Reason text.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reason: Option<&'a str>,
    /// Merchant idempotency key.
    pub merchant_refund_id: String,
}

/// Customer object inline in `CreatePayment.customer`.
#[derive(Debug, Clone, Serialize)]
pub struct Customer<'a> {
    /// Merchant-supplied customer id, 1–64 chars.
    pub id: &'a str,
    /// Optional email.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub email: Option<&'a str>,
    /// Optional name.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<&'a str>,
}

// ---------------------------------------------------------------------------
// Responses
// ---------------------------------------------------------------------------

/// Response body for `POST /payments`, `POST /payments/{id}/confirm`,
/// `POST /payments/{id}/capture`, `POST /payments/{id}/cancel`.
///
/// These four endpoints all return the same `PaymentsResponse` shape
/// per the Hyperswitch V1 spec.
#[derive(Debug, Clone, Deserialize)]
pub struct PaymentsResponse {
    /// PSP payment id (`pay_...`).
    pub payment_id: String,
    /// Status string from the 17-variant Hyperswitch status enum.
    pub status: String,
    /// Requested amount (minor units).
    pub amount: i64,
    /// Amount still capturable (minor units). 0 once fully captured.
    #[serde(default)]
    pub amount_capturable: i64,
    /// Amount actually captured/received (minor units). 0 until capture.
    #[serde(default)]
    pub amount_received: Option<i64>,
    /// Currency alpha-3.
    pub currency: String,
    /// Underlying connector that processed the payment.
    #[serde(default)]
    pub connector: Option<String>,
    /// Connector-side transaction id.
    #[serde(default)]
    pub connector_transaction_id: Option<String>,
    /// 3DS / redirect challenge info (we extract `redirect_to_url`).
    #[serde(default)]
    pub next_action: Option<NextAction>,
    /// PSP error code on failure.
    #[serde(default)]
    pub error_code: Option<String>,
    /// PSP error message on failure.
    #[serde(default)]
    pub error_message: Option<String>,
}

/// `next_action` envelope. We only inspect the URL for redirects;
/// other variants we pass through as raw JSON.
#[derive(Debug, Clone, Deserialize)]
pub struct NextAction {
    /// `redirect_to_url` | `invoke_sdk_client` | ...
    #[serde(rename = "type")]
    pub kind: Option<String>,
    /// Present when `kind == "redirect_to_url"`.
    pub redirect_to_url: Option<String>,
}

/// Response body for `POST /refunds`.
#[derive(Debug, Clone, Deserialize)]
pub struct RefundResponse {
    /// PSP refund id (`ref_...`).
    pub refund_id: String,
    /// Linked payment id.
    pub payment_id: String,
    /// `pending` | `succeeded` | `failed` | `manual_review`.
    pub status: String,
    /// Refund amount in minor units.
    pub amount: i64,
    /// Currency.
    pub currency: String,
}

/// Error body returned by Hyperswitch on 4xx/5xx.
///
/// Hyperswitch's docs show errors come back as:
/// `{ "error": { "type": "...", "message": "...", "code": "..." } }`.
#[derive(Debug, Clone, Deserialize)]
pub struct ErrorEnvelope {
    /// The error object.
    pub error: ErrorBody,
}

/// Inner error fields.
#[derive(Debug, Clone, Deserialize)]
pub struct ErrorBody {
    /// `invalid_request` | `connector_error` | `routing_error` | ...
    #[serde(rename = "type")]
    pub kind: String,
    /// Human-readable message.
    pub message: String,
    /// Machine-readable code (e.g. `IR_05`, `CE_01`).
    pub code: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_payment_minimal_serializes_to_expected_json() {
        let req = CreatePayment {
            amount: 6540,
            currency: "USD",
            capture_method: None,
            authentication_type: None,
            payment_id: None,
            customer: None,
            confirm: None,
            payment_method: None,
            payment_method_type: None,
            payment_token: None,
            payment_method_data: None,
            metadata: None,
            connector_metadata: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        // Verified-shape: matches the canonical curl example from
        // api-reference.hyperswitch.io/v1/payments/payments--create
        assert_eq!(json, serde_json::json!({"amount": 6540, "currency": "USD"}));
    }

    #[test]
    fn create_payment_with_manual_capture_includes_field() {
        let req = CreatePayment {
            amount: 10_00,
            currency: "USD",
            capture_method: Some("manual"),
            authentication_type: Some("no_three_ds"),
            payment_id: Some("pay_test_1234567890123456789012".into()),
            customer: None,
            confirm: Some(false),
            payment_method: None,
            payment_method_type: None,
            payment_token: None,
            payment_method_data: None,
            metadata: None,
            connector_metadata: None,
        };
        let json = serde_json::to_value(&req).unwrap();
        assert_eq!(json["amount"], 1000);
        assert_eq!(json["currency"], "USD");
        assert_eq!(json["capture_method"], "manual");
        assert_eq!(json["authentication_type"], "no_three_ds");
        assert_eq!(json["payment_id"], "pay_test_1234567890123456789012");
        assert_eq!(json["confirm"], false);
        // Optional fields not set should not appear.
        assert!(json.get("metadata").is_none());
        assert!(json.get("customer").is_none());
    }

    #[test]
    fn payments_response_parses_canonical_success_body() {
        // Verbatim from api-reference.hyperswitch.io example response.
        let body = r#"{
            "amount": 6540,
            "amount_capturable": 6540,
            "attempt_count": 1,
            "client_secret": "pay_syxxxxxxxxxxxx_secret_szzzzzzzzzzz",
            "created": "2023-10-26T10:00:00Z",
            "currency": "USD",
            "expires_on": "2023-10-26T10:15:00Z",
            "merchant_id": "merchant_myyyyyyyyyyyy",
            "payment_id": "pay_syxxxxxxxxxxxx",
            "profile_id": "pro_pzzzzzzzzzzz",
            "status": "requires_payment_method"
        }"#;
        let parsed: PaymentsResponse = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.payment_id, "pay_syxxxxxxxxxxxx");
        assert_eq!(parsed.status, "requires_payment_method");
        assert_eq!(parsed.amount, 6540);
        assert_eq!(parsed.amount_capturable, 6540);
        assert_eq!(parsed.currency, "USD");
    }

    #[test]
    fn payments_response_parses_succeeded_with_received_amount() {
        let body = r#"{
            "payment_id": "pay_abc",
            "status": "succeeded",
            "amount": 1000,
            "amount_capturable": 0,
            "amount_received": 1000,
            "currency": "USD",
            "connector": "stripe",
            "connector_transaction_id": "ch_3O123abc"
        }"#;
        let parsed: PaymentsResponse = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.status, "succeeded");
        assert_eq!(parsed.amount_received, Some(1000));
        assert_eq!(parsed.connector.as_deref(), Some("stripe"));
        assert_eq!(
            parsed.connector_transaction_id.as_deref(),
            Some("ch_3O123abc")
        );
    }

    #[test]
    fn payments_response_parses_failure_with_error_fields() {
        let body = r#"{
            "payment_id": "pay_xyz",
            "status": "failed",
            "amount": 1000,
            "amount_capturable": 0,
            "currency": "USD",
            "error_code": "E0001",
            "error_message": "Failed while verifying the card"
        }"#;
        let parsed: PaymentsResponse = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.status, "failed");
        assert_eq!(parsed.error_code.as_deref(), Some("E0001"));
        assert_eq!(
            parsed.error_message.as_deref(),
            Some("Failed while verifying the card")
        );
    }

    #[test]
    fn payments_response_parses_redirect_next_action() {
        let body = r#"{
            "payment_id": "pay_redir",
            "status": "requires_customer_action",
            "amount": 1000,
            "amount_capturable": 1000,
            "currency": "USD",
            "next_action": {
                "type": "redirect_to_url",
                "redirect_to_url": "https://hooks.stripe.com/redirect/abc"
            }
        }"#;
        let parsed: PaymentsResponse = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.status, "requires_customer_action");
        let na = parsed.next_action.expect("next_action must be present");
        assert_eq!(na.kind.as_deref(), Some("redirect_to_url"));
        assert_eq!(
            na.redirect_to_url.as_deref(),
            Some("https://hooks.stripe.com/redirect/abc")
        );
    }

    #[test]
    fn error_envelope_parses_canonical_shape() {
        let body = r#"{
            "error": {
                "type": "invalid_request",
                "message": "Missing required param: amount",
                "code": "IR_05"
            }
        }"#;
        let parsed: ErrorEnvelope = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.error.kind, "invalid_request");
        assert_eq!(parsed.error.code, "IR_05");
        assert_eq!(parsed.error.message, "Missing required param: amount");
    }

    #[test]
    fn refund_response_parses() {
        let body = r#"{
            "refund_id": "ref_test_123",
            "payment_id": "pay_test_456",
            "status": "pending",
            "amount": 1000,
            "currency": "USD"
        }"#;
        let parsed: RefundResponse = serde_json::from_str(body).unwrap();
        assert_eq!(parsed.refund_id, "ref_test_123");
        assert_eq!(parsed.status, "pending");
        assert_eq!(parsed.amount, 1000);
    }

    #[test]
    fn capture_payment_serializes() {
        let req = CapturePayment {
            amount_to_capture: 500,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, r#"{"amount_to_capture":500}"#);
    }

    #[test]
    fn cancel_payment_serializes_reason() {
        let req = CancelPayment {
            cancellation_reason: "requested_by_customer",
        };
        let json = serde_json::to_string(&req).unwrap();
        assert_eq!(json, r#"{"cancellation_reason":"requested_by_customer"}"#);
    }
}
