//! Payout funding sources.
//!
//! A payout has to be paid for. Two patterns dominate:
//!
//! - **Prefunded balance** — the operator wires funds into the rail
//!   ahead of time (Visa Direct funding account, FBO bank account,
//!   on-chain hot wallet). Payouts debit that balance.
//! - **Pull-based** — the rail / processor debits the operator's
//!   settlement account just-in-time per payout (ACH debit, card
//!   funding via push-to-card with same-card pull).
//!
//! This module models both and exposes the pull instruction so
//! `op-ledger` can record the cash leg.

use op_core::Money;
use serde::{Deserialize, Serialize};

/// Whether a payout is funded from a sitting balance or pulled in
/// just-in-time.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FundingMode {
    /// Funds already sit at the rail in the operator's account.
    Prefunded,
    /// Funds are pulled from the operator's settlement bank at submit
    /// time.
    PullBased,
}

/// The specific funding source a payout draws from.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum FundingSource {
    /// Operator's prefunded balance at the rail. `account_ref` is the
    /// rail-side identifier of that balance (Visa Direct `fundingId`,
    /// FedNow settlement account, hot-wallet address, ...).
    Prefunded {
        /// Rail-side identifier.
        account_ref: String,
    },
    /// Operator's settlement bank account to debit JIT.
    PullDebit {
        /// 9-digit ABA (US) or BIC (intl).
        bank_id: String,
        /// Account number / IBAN.
        account: String,
        /// Standard Entry Class for ACH debits (`"CCD"` is the
        /// corporate-to-corporate code). Ignored for non-ACH pulls.
        sec_code: Option<String>,
    },
}

impl FundingSource {
    /// Funding mode derived from the variant.
    #[must_use]
    pub const fn mode(&self) -> FundingMode {
        match self {
            Self::Prefunded { .. } => FundingMode::Prefunded,
            Self::PullDebit { .. } => FundingMode::PullBased,
        }
    }
}

/// Instruction the orchestrator emits to the funding rail when a
/// payout is pull-funded. Captured here for `op-ledger` to record the
/// matching debit.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct PullDebitInstruction {
    /// Caller-supplied idempotency key, copied from the parent
    /// [`PayoutRequest`](crate::PayoutRequest).
    pub idempotency_key: String,
    /// Amount to pull. Equal to the payout amount plus any rail fee the
    /// operator decides to gross-up.
    pub amount: Money,
    /// Bank identifier (ABA / BIC).
    pub bank_id: String,
    /// Account / IBAN.
    pub account: String,
    /// SEC code for ACH pulls. `None` for non-ACH funding.
    pub sec_code: Option<String>,
}

impl PullDebitInstruction {
    /// Build a pull instruction from a funding source. Returns `None`
    /// when the source is prefunded (no pull needed).
    #[must_use]
    pub fn from_source(
        funding: &FundingSource,
        amount: Money,
        idempotency_key: String,
    ) -> Option<Self> {
        match funding {
            FundingSource::Prefunded { .. } => None,
            FundingSource::PullDebit {
                bank_id,
                account,
                sec_code,
            } => Some(Self {
                idempotency_key,
                amount,
                bank_id: bank_id.clone(),
                account: account.clone(),
                sec_code: sec_code.clone(),
            }),
        }
    }
}
