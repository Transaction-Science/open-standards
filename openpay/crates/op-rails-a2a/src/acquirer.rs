//! The [`A2aAcquirer`] trait — universal A2A rail interface.
//!
//! Every rail driver (`FedNow`, PIX, RT1, TIPS) implements this trait.
//! The orchestrator holds `Box<dyn A2aAcquirer>` and routes purely on
//! the rail kind in `op_core::PaymentMethod::A2a(_)`.

use op_core::Money;
use serde::{Deserialize, Serialize};

use crate::error::Result;

/// Identifies a participant on an A2A rail.
///
/// Each rail uses different identifier schemes; we union them in one
/// enum because the orchestrator routes on this.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ParticipantId {
    /// US ABA Routing & Transit Number (9 digits). Used by `FedNow`.
    /// Validated via the checksum in [`op_iso20022::bah`].
    Aba(String),
    /// SWIFT BIC, 8 or 11 chars. Used by SEPA SCT Inst (RT1/TIPS).
    Bic(String),
    /// Brazilian ISPB — 8-digit identifier assigned by Bacen. Used by PIX.
    Ispb(String),
    /// Generic free-form participant ID for rails not yet covered.
    Other(String),
}

impl ParticipantId {
    /// Human-readable representation for logging.
    #[must_use]
    pub fn as_str(&self) -> &str {
        match self {
            Self::Aba(s) | Self::Bic(s) | Self::Ispb(s) | Self::Other(s) => s.as_str(),
        }
    }
}

/// A credit-transfer request — the "send money" call.
///
/// This is the abstract form. Each driver translates it to the rail's
/// specific message via `op-iso20022`'s `CreditTransferBuilder<P>`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CreditTransferReq {
    /// UETR — Unique End-to-end Transaction Reference. UUID v4
    /// lowercase, hyphenated. Caller supplies; profile validates.
    pub uetr: String,
    /// `EndToEndId` — caller's own reference, up to 35 chars.
    pub end_to_end_id: String,
    /// Amount and currency.
    pub amount: Money,
    /// Debtor (sender) bank.
    pub debtor_agent: ParticipantId,
    /// Creditor (receiver) bank.
    pub creditor_agent: ParticipantId,
    /// Debtor account identifier (IBAN for SEPA, account number for
    /// `FedNow`, `ChaveDict` or account for PIX).
    pub debtor_account: String,
    /// Creditor account identifier.
    pub creditor_account: String,
    /// Debtor name. SEPA caps at 70 chars; `FedNow` at 140.
    pub debtor_name: String,
    /// Creditor name.
    pub creditor_name: String,
    /// Free-form remittance text. Driver clips to rail max (`FedNow` 140,
    /// SEPA 140, PIX 140).
    pub remittance: Option<String>,
    /// Caller idempotency key. Drivers forward to rail where supported.
    pub idempotency_key: String,
}

/// Request to query the status of a previously sent transfer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StatusQueryReq {
    /// UETR of the original transfer.
    pub uetr: String,
    /// `EndToEndId` of the original transfer.
    pub end_to_end_id: String,
}

/// Rail decision after submitting a transfer or status query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct A2aDecision {
    /// Normalized status.
    pub status: A2aStatus,
    /// Rail-specific status code, preserved for diagnostics.
    pub raw_status: String,
    /// Optional reason code (e.g. SEPA `RC03`, `FedNow` `AC03`).
    pub reason_code: Option<String>,
    /// Optional reason text.
    pub reason_text: Option<String>,
    /// UETR echoed back. Should match the request UETR.
    pub uetr: Option<String>,
    /// Rail-side transaction id, if the rail issues one in addition to
    /// the UETR.
    pub rail_txn_id: Option<String>,
    /// Settled amount (typically equal to requested; some rails apply
    /// FX). None if the rail doesn't report it on this response.
    pub settled_amount: Option<Money>,
}

/// Normalized A2A status. Maps from each rail's native status codes.
///
/// Designed to align with `pacs.002` `TransactionStatus` (ACCP/ACSC/
/// RJCT/PDNG/ACSP) plus the operational states `OpenPay` needs.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum A2aStatus {
    /// `ACSC` — accepted, settlement completed. Funds at the creditor.
    Settled,
    /// `ACCP` — accepted by the rail; settlement in progress / complete
    /// depending on the rail. RT1 may return ACCP even after final
    /// settlement; we use [`Self::Accepted`] as the conservative state.
    Accepted,
    /// `ACSP` — accepted, settlement in progress.
    InProgress,
    /// `PDNG` — pending. Caller should poll status.
    Pending,
    /// `RJCT` — rejected. Funds did not move. Inspect `reason_code`.
    Rejected,
    /// Transient transport failure (timeout, mTLS reset). Caller can
    /// retry. Idempotency keys ensure no duplicate transfers.
    Transient,
    /// Rail is reachable but reports a non-payment failure (auth fail,
    /// quota exhaustion, schema rejection). Inspect `reason_text`.
    OperationalError,
}

impl A2aStatus {
    /// True if funds moved (or are guaranteed to move).
    #[must_use]
    pub const fn funds_moved(self) -> bool {
        matches!(self, Self::Settled | Self::Accepted)
    }

    /// True if the caller can safely retry the same request.
    #[must_use]
    pub const fn is_retryable(self) -> bool {
        matches!(self, Self::Transient)
    }

    /// True if the rail definitively refused.
    #[must_use]
    pub const fn is_failure(self) -> bool {
        matches!(self, Self::Rejected | Self::OperationalError)
    }

    /// True if the caller should poll for a final outcome.
    #[must_use]
    pub const fn needs_polling(self) -> bool {
        matches!(self, Self::Pending | Self::InProgress)
    }
}

/// The generic A2A acquirer interface.
pub trait A2aAcquirer: Send + Sync {
    /// Rail name, e.g. `"fednow"`, `"pix"`, `"sepa-rt1"`.
    fn name(&self) -> &'static str;

    /// Submit a credit-transfer (pacs.008) and synchronously return
    /// the rail's first response (pacs.002 or transport ack).
    ///
    /// For `FedNow` this is the immediate ack from the FRB. For PIX
    /// this is the ICOM response. For RT1 / TIPS this is the
    /// settlement notification.
    fn submit_credit_transfer(&self, req: &CreditTransferReq) -> Result<A2aDecision>;

    /// Query the status of a previously-submitted transfer.
    fn query_status(&self, req: &StatusQueryReq) -> Result<A2aDecision>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_classification_disjoint() {
        for s in [
            A2aStatus::Settled,
            A2aStatus::Accepted,
            A2aStatus::InProgress,
            A2aStatus::Pending,
            A2aStatus::Rejected,
            A2aStatus::Transient,
            A2aStatus::OperationalError,
        ] {
            // No status should be both a failure and a retryable transient.
            assert!(!(s.is_failure() && s.is_retryable()), "{s:?} is both");
            // funds_moved and is_failure must be disjoint.
            assert!(
                !(s.funds_moved() && s.is_failure()),
                "{s:?} both moved and failed"
            );
        }
    }

    #[test]
    fn funds_moved_only_for_settled_and_accepted() {
        assert!(A2aStatus::Settled.funds_moved());
        assert!(A2aStatus::Accepted.funds_moved());
        assert!(!A2aStatus::Pending.funds_moved());
        assert!(!A2aStatus::InProgress.funds_moved());
        assert!(!A2aStatus::Rejected.funds_moved());
        assert!(!A2aStatus::Transient.funds_moved());
    }

    #[test]
    fn needs_polling_only_for_pending_states() {
        assert!(A2aStatus::Pending.needs_polling());
        assert!(A2aStatus::InProgress.needs_polling());
        assert!(!A2aStatus::Settled.needs_polling());
        assert!(!A2aStatus::Rejected.needs_polling());
    }

    #[test]
    fn participant_id_as_str_extracts_inner() {
        assert_eq!(ParticipantId::Aba("021000021".into()).as_str(), "021000021");
        assert_eq!(
            ParticipantId::Bic("DEUTDEFFXXX".into()).as_str(),
            "DEUTDEFFXXX"
        );
        assert_eq!(ParticipantId::Ispb("12345678".into()).as_str(), "12345678");
        assert_eq!(ParticipantId::Other("x".into()).as_str(), "x");
    }

    #[test]
    fn retryable_does_not_overlap_failure() {
        assert!(A2aStatus::Transient.is_retryable());
        assert!(!A2aStatus::Transient.is_failure());
        assert!(A2aStatus::Rejected.is_failure());
        assert!(!A2aStatus::Rejected.is_retryable());
        assert!(A2aStatus::OperationalError.is_failure());
        assert!(!A2aStatus::OperationalError.is_retryable());
    }
}
