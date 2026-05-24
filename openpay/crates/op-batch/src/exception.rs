//! Returns, rejects, refunds, reversals — collectively "exceptions".
//!
//! Every batch rail has its own taxonomy:
//!
//! - **NACHA** returns are `R01..R99` codes ([`crate::nacha::ReturnCode`]).
//! - **SEPA** R-transactions are alphanumeric reason codes from the
//!   ISO 20022 ExternalReturnReason / StatusReason code lists
//!   (e.g. `AC01` IncorrectAccountNumber, `MD06` RefundRequestByEndCustomer).
//! - **Wire** rarely produces "returns" — instead, an
//!   *investigation* is raised via `camt.026`/`camt.027`/`camt.029`.
//!
//! `op-batch` normalises all three into one struct so the
//! orchestrator and downstream ledger can treat them uniformly.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::BatchRail;

/// Opaque payment-id used by `op-batch`. Operators map this back to
/// their own ledger's transaction id; we keep it as a string so we
/// don't pin the ledger's id format.
pub type PaymentId = String;

/// A rail-agnostic exception code. We preserve the original
/// rail-specific code verbatim and tag the rail so the orchestrator
/// can route to the right handler.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ExceptionCode {
    /// Source rail.
    pub rail: BatchRail,
    /// Rail-specific code (e.g. `R01`, `AC01`).
    pub code: String,
}

/// One exception event. Produced by [`crate::orchestrator::BatchProcessor::fetch_returns`]
/// or by ad-hoc reconciliation of a bank statement against
/// outbound batches.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Exception {
    /// Which rail raised the exception.
    pub rail: BatchRail,
    /// The operator's id for the original payment.
    pub original_payment_id: PaymentId,
    /// Rail-specific code.
    pub code: String,
    /// Free-text reason (parsed from the addenda / RsnInf field).
    pub reason: String,
    /// When the rail reported the exception (settlement-side date).
    pub raised_at: DateTime<Utc>,
}

/// What the operator should do with an exception. Computed from
/// the code; deterministic, no LLM involved.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ExceptionAction {
    /// Retry the payment on the same or alternate rail.
    Retry,
    /// Stop trying — mark the payment failed permanently.
    Retire,
    /// Escalate to a human operator (compliance, OFAC, mandate
    /// revocation, etc.).
    Escalate,
}

impl Exception {
    /// Suggested action for this exception based on the code.
    ///
    /// The mapping reflects NACHA / EPC operator best-practice:
    /// "soft" reasons (insufficient funds, technical) → Retry;
    /// "hard" reasons (closed account, invalid account) → Retire;
    /// "compliance" reasons (OFAC freeze, unauthorized) → Escalate.
    #[must_use]
    pub fn suggested_action(&self) -> ExceptionAction {
        match self.rail {
            BatchRail::Nacha => map_nacha(&self.code),
            BatchRail::SepaCt | BatchRail::SepaDd => map_sepa(&self.code),
            BatchRail::Fedwire | BatchRail::Swift | BatchRail::Chips => ExceptionAction::Escalate,
            BatchRail::Bacs => map_bacs(&self.code),
        }
    }
}

fn map_nacha(code: &str) -> ExceptionAction {
    match code {
        // Insufficient funds, uncollected funds — try again later.
        "R01" | "R09" => ExceptionAction::Retry,
        // Hard rejects — give up.
        "R02" | "R03" | "R04" | "R11" | "R12" | "R13" | "R14" | "R15" | "R20" => {
            ExceptionAction::Retire
        }
        // Compliance / authorisation issues — human in the loop.
        "R05" | "R07" | "R08" | "R10" | "R16" => ExceptionAction::Escalate,
        _ => ExceptionAction::Escalate,
    }
}

fn map_sepa(code: &str) -> ExceptionAction {
    match code {
        // Insufficient funds, technical.
        "AM04" | "AM05" | "TM01" => ExceptionAction::Retry,
        // Closed / invalid account, mandate issues.
        "AC01" | "AC04" | "AC06" | "MD01" | "MD02" => ExceptionAction::Retire,
        // Customer refund / cancellation requests, fraud.
        "MD06" | "MS02" | "MS03" | "FF01" => ExceptionAction::Escalate,
        _ => ExceptionAction::Escalate,
    }
}

fn map_bacs(code: &str) -> ExceptionAction {
    // Bacs ARUDD/AUDDIS codes use a single character ("0".."Z").
    // Most common: `0` (refer to payer), `1` (instruction cancelled),
    // `2` (payer deceased), `3` (account transferred), `5` (no account),
    // `6` (no instruction), `7` (amount differs), `8` (amount not yet due),
    // `9` (presentation overdue), `B` (account closed).
    match code {
        "0" | "8" => ExceptionAction::Retry,
        "1" | "3" | "5" | "6" | "B" => ExceptionAction::Retire,
        "2" | "7" | "9" => ExceptionAction::Escalate,
        _ => ExceptionAction::Escalate,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ex(rail: BatchRail, code: &str) -> Exception {
        Exception {
            rail,
            original_payment_id: "p1".into(),
            code: code.into(),
            reason: String::new(),
            raised_at: Utc::now(),
        }
    }

    #[test]
    fn nacha_r01_retries() {
        assert_eq!(
            ex(BatchRail::Nacha, "R01").suggested_action(),
            ExceptionAction::Retry
        );
    }

    #[test]
    fn nacha_r02_retires() {
        assert_eq!(
            ex(BatchRail::Nacha, "R02").suggested_action(),
            ExceptionAction::Retire
        );
    }

    #[test]
    fn nacha_r07_escalates() {
        assert_eq!(
            ex(BatchRail::Nacha, "R07").suggested_action(),
            ExceptionAction::Escalate
        );
    }

    #[test]
    fn sepa_ac01_retires() {
        assert_eq!(
            ex(BatchRail::SepaDd, "AC01").suggested_action(),
            ExceptionAction::Retire
        );
    }

    #[test]
    fn sepa_md06_escalates() {
        assert_eq!(
            ex(BatchRail::SepaDd, "MD06").suggested_action(),
            ExceptionAction::Escalate
        );
    }

    #[test]
    fn bacs_5_retires() {
        assert_eq!(
            ex(BatchRail::Bacs, "5").suggested_action(),
            ExceptionAction::Retire
        );
    }

    #[test]
    fn wire_always_escalates() {
        assert_eq!(
            ex(BatchRail::Fedwire, "anything").suggested_action(),
            ExceptionAction::Escalate
        );
    }
}
