//! Refund + reconciliation semantics — symmetric across every
//! APAC rail in this crate.
//!
//! Refunds in this crate are *intents* not transports. Each rail's
//! adapter exposes a `refund(...)` method that translates a
//! [`RefundIntent`] into the rail-native call:
//!
//! - Alipay → `alipay.trade.refund` (v3 `/v3/alipay/trade/refund`).
//! - WeChat → `/v3/refund/domestic/refunds`.
//! - UPI → PSP `/upi/v2/refund` (NPCI REFUND/REFUND-REVERSAL pair).
//! - Paytm → `/refund/apply`.
//! - GrabPay → `/grabpay/partner/v2/refund`.
//! - GoPay → `/v2/<order_id>/refund`.
//! - TnG → `/v1/payments/refund`.
//! - PromptPay / PayNow / DuitNow / QRIS → out-of-band via the
//!   merchant's acquirer bank (no rail-side refund API exists for
//!   these EMVCo MPM rails; refund is a regular debit in the
//!   reverse direction).
//!
//! ## Why no per-rail refund call here
//!
//! Refund flows are 1:1 with the create-charge flows already in
//! each rail's module. We model the common shape ([`RefundIntent`]
//! / [`RefundResult`]) here so the merchant orchestrator can fan
//! out without per-rail pattern-matching. Per-rail refund
//! transport lives behind operator-supplied transports already
//! injected for charge creation.

use op_core::Money;
use serde::{Deserialize, Serialize};

use crate::wallet::WalletKind;

/// A refund request against a previously-succeeded charge.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefundIntent {
    /// Merchant correlation id for the refund (must be globally
    /// unique across the merchant's refund history, not just per
    /// original charge — adapters use this as the rail-side
    /// idempotency key).
    pub merchant_refund_id: String,
    /// Echo of the original merchant order id.
    pub original_merchant_order_id: String,
    /// Refund amount. Must be ≤ the original charge amount and in
    /// the same currency.
    pub amount: Money,
    /// Operator-facing reason (logged with the rail and forwarded
    /// to the consumer where the rail surfaces it).
    pub reason: String,
}

/// Refund lifecycle status.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum RefundStatus {
    /// Accepted by the rail; not yet settled.
    Pending,
    /// Settled — funds returned to consumer.
    Succeeded,
    /// Rail declined the refund.
    Failed,
}

/// Normalized refund result.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct RefundResult {
    /// Echo of the merchant refund id.
    pub merchant_refund_id: String,
    /// Rail-side refund id (Alipay `out_refund_no`, WeChat
    /// `refund_id`, UPI refund txnId, ...).
    pub provider_refund_id: String,
    /// Rail-side status.
    pub status: RefundStatus,
}

/// One line of a daily reconciliation file.
///
/// Every rail in scope ships some form of T+1 settlement file
/// (Alipay bill-download CSV, WeChat statement, NPCI settlement
/// report, ABS PayNow MT940, ...). The [`ReconciliationLine`]
/// shape normalizes those into a uniform structure the merchant's
/// ledger can ingest.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconciliationLine {
    /// Which rail this line came from.
    pub rail: WalletKind,
    /// Merchant order id (or merchant refund id) the line references.
    pub merchant_id: String,
    /// Rail-side transaction id.
    pub provider_id: String,
    /// Net amount cleared (signed: negative for refunds).
    pub net_amount: Money,
    /// Rail's interpretation of the line.
    pub outcome: ReconciliationOutcome,
}

/// What a reconciliation line attests to.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReconciliationOutcome {
    /// Charge settled to merchant.
    Settled,
    /// Refund debited from merchant settlement.
    Refunded,
    /// Chargeback / consumer dispute deducted from settlement.
    ChargedBack,
    /// Fee line (interchange, scheme fee, FX spread).
    Fee,
    /// Adjustment line the merchant must read and book manually.
    Adjustment,
}

impl ReconciliationLine {
    /// Construct a settled-charge line.
    #[must_use]
    pub const fn settled(
        rail: WalletKind,
        merchant_id: String,
        provider_id: String,
        net_amount: Money,
    ) -> Self {
        Self {
            rail,
            merchant_id,
            provider_id,
            net_amount,
            outcome: ReconciliationOutcome::Settled,
        }
    }

    /// Construct a refund line.
    #[must_use]
    pub const fn refunded(
        rail: WalletKind,
        merchant_id: String,
        provider_id: String,
        net_amount: Money,
    ) -> Self {
        Self {
            rail,
            merchant_id,
            provider_id,
            net_amount,
            outcome: ReconciliationOutcome::Refunded,
        }
    }
}
