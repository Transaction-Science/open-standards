//! Cross-rail abstractions: `AsiaWallet`, `WalletKind`, `ChargeIntent`,
//! `ChargeResult`.
//!
//! All APAC e-wallets in this crate normalize to a single shape: a
//! [`ChargeIntent`] in some [`op_core::Currency`] for a
//! [`op_core::Money`] amount, presented to the consumer via a
//! [`PresentmentMode`], and resolved into a [`ChargeResult`] whose
//! [`ChargeStatus`] tracks the rail's lifecycle.

use op_core::{Currency, Money};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// The supported APAC wallet rails.
///
/// `Copy + Eq + Hash` so this works as a routing key in operator
/// orchestrators.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WalletKind {
    /// Ant Group's Alipay (Cross-Border Open API v3). CNY domestic;
    /// settlement currency negotiated cross-border.
    Alipay,
    /// Tencent's WeChat Pay (v3). CNY domestic; cross-border via the
    /// merchant-acquirer agreement.
    WeChatPay,
    /// NPCI UPI 2.0 (India). INR.
    Upi,
    /// Paytm Standard Checkout (India). INR.
    Paytm,
    /// GrabPay — Pay-with-Grab. SGD/MYR/PHP/THB/IDR/VND.
    GrabPay,
    /// GoPay (Gojek, Indonesia). IDR.
    GoPay,
    /// Touch'n Go eWallet (Malaysia). MYR.
    TouchNGo,
    /// PromptPay (Thailand BoT EMVCo MPM). THB.
    PromptPay,
    /// PayNow (Singapore ABS EMVCo MPM). SGD.
    PayNow,
    /// DuitNow (Malaysia PayNet EMVCo MPM). MYR.
    DuitNow,
    /// QRIS (Indonesia BI EMVCo MPM). IDR.
    Qris,
}

impl WalletKind {
    /// Returns true if this rail is fundamentally a QR-presentment
    /// rail (EMVCo MPM family). These rails share a TLV codec and do
    /// not require a hosted-page redirect.
    #[must_use]
    pub const fn is_qr_rail(self) -> bool {
        matches!(
            self,
            Self::PromptPay | Self::PayNow | Self::DuitNow | Self::Qris
        )
    }

    /// Returns true if this rail supports recurring mandates natively
    /// (without falling back to a card-on-file ladder).
    #[must_use]
    pub const fn supports_mandate(self) -> bool {
        matches!(self, Self::Upi | Self::Alipay | Self::WeChatPay)
    }
}

/// How the merchant presents the charge to the consumer.
///
/// Distinct from `WalletKind` because the same wallet can be
/// presented in multiple modes (WeChat JSAPI vs Native vs MicroPay).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PresentmentMode {
    /// In-app JSAPI / mini-program prepay handle. Consumer is
    /// already inside the wallet's app or H5.
    InAppJsApi,
    /// Merchant-presented QR (consumer scans a code the merchant
    /// generated). EMVCo MPM rails default to this.
    MerchantPresentedQr,
    /// Consumer-presented QR (merchant scans the code the consumer's
    /// wallet generated, e.g. WeChat MicroPay).
    ConsumerPresentedQr,
    /// H5 / browser redirect to the wallet's hosted page.
    Browser,
    /// Native-app deeplink (UPI intent, Paytm intent, Grab deeplink).
    Deeplink,
}

/// A normalized charge intent.
///
/// `amount.currency` must match the rail's domestic currency unless
/// the adapter explicitly supports cross-border settlement. Adapters
/// validate this at call-time and surface `Error::InvalidIntent` if
/// the pairing is unsupported.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChargeIntent {
    /// Merchant-side correlation id. Idempotency key for the rail's
    /// notify-callback fan-out.
    pub merchant_order_id: String,
    /// Charge amount.
    pub amount: Money,
    /// Short human-readable description (subject line in Alipay /
    /// WeChat / QRIS; displayed in the consumer's wallet UI).
    pub description: String,
    /// How the charge will be presented to the consumer.
    pub presentment: PresentmentMode,
    /// Optional consumer hint (UPI VPA, WeChat openid, Alipay
    /// buyer-id) the adapter forwards as appropriate.
    pub consumer_hint: Option<String>,
    /// Optional notify URL the rail should hit on terminal-state
    /// transitions. Adapters that require this (Alipay / WeChat /
    /// Paytm) error if `None`.
    pub notify_url: Option<String>,
}

impl ChargeIntent {
    /// Validate the basic pre-flight invariants every rail shares.
    ///
    /// - Order id is non-empty.
    /// - Amount is strictly positive.
    /// - Description is non-empty (rails reject empty `body` /
    ///   `subject`).
    ///
    /// # Errors
    /// Returns [`Error::InvalidIntent`] if any invariant fails.
    pub fn validate_common(&self) -> Result<()> {
        if self.merchant_order_id.is_empty() {
            return Err(Error::InvalidIntent("empty merchant_order_id".into()));
        }
        if self.amount.minor_units <= 0 {
            return Err(Error::InvalidIntent("amount must be positive".into()));
        }
        if self.description.is_empty() {
            return Err(Error::InvalidIntent("empty description".into()));
        }
        Ok(())
    }

    /// Assert that the intent's currency matches an expected
    /// domestic currency.
    ///
    /// # Errors
    /// Returns [`Error::InvalidIntent`] on mismatch.
    pub fn require_currency(&self, expected: Currency) -> Result<()> {
        if self.amount.currency != expected {
            return Err(Error::InvalidIntent(format!(
                "expected {} got {}",
                expected,
                self.amount.currency
            )));
        }
        Ok(())
    }
}

/// Lifecycle status of a charge.
///
/// Rails differ in their intermediate states (UPI exposes a `Deemed`
/// state, WeChat a `USERPAYING` state, ...) but the terminal-state
/// shape is universal. Adapters map their native lifecycle codes
/// into this enum.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChargeStatus {
    /// The intent was created but the consumer has not yet acted.
    Pending,
    /// The wallet has authorized the charge and funds are committed.
    Succeeded,
    /// The consumer cancelled or the wallet declined.
    Failed,
    /// The intent expired before consumer action.
    Expired,
    /// Status is unknown; the orchestrator should poll the rail's
    /// `query` endpoint. UPI in particular emits this on the
    /// initial response when the bank backend is slow.
    Unknown,
}

/// Normalized charge result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChargeResult {
    /// Echo of the merchant's order id.
    pub merchant_order_id: String,
    /// Rail-side transaction id (Alipay `trade_no`, WeChat
    /// `transaction_id`, UPI `txnId`, ...). Empty when the rail
    /// has not yet minted one.
    pub provider_transaction_id: String,
    /// Rail-side lifecycle status.
    pub status: ChargeStatus,
    /// The payload the consumer-facing surface needs to render
    /// (QR code string, JSAPI prepay handle, deeplink URL, ...).
    /// Empty when the rail returned a terminal state without a
    /// presentment artifact.
    pub presentment_payload: String,
}

/// The unified adapter trait. Every APAC rail in this crate
/// implements this so a merchant orchestrator can dispatch through
/// `dyn AsiaWallet` without per-rail pattern-matching.
///
/// Methods are `&self`: adapters are configured at construction time
/// (merchant id, signing-key material, transport) and then immutable.
pub trait AsiaWallet: Send + Sync {
    /// Which rail this adapter handles.
    fn kind(&self) -> WalletKind;

    /// Create a charge against the rail. The returned
    /// [`ChargeResult`] carries the presentment payload (QR string,
    /// JSAPI prepay handle, deeplink) for the consumer-facing surface.
    fn create_charge(&self, intent: &ChargeIntent) -> Result<ChargeResult>;

    /// Query the rail for the current status of a previously-created
    /// charge.
    fn query_charge(&self, merchant_order_id: &str) -> Result<ChargeResult>;
}
