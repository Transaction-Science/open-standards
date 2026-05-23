//! Discrepancies and the report that aggregates them.

use op_core::Money;
use op_ledger::{Status, TransactionId};

use crate::statement::StatementLine;

/// One way the ledger and a statement disagree.
///
/// Every variant carries enough context for an operator to act
/// without re-deriving anything: the offending statement line and/or
/// the ledger transaction id, plus what specifically differed.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum Discrepancy {
    /// The statement says money moved, but no ledger transaction
    /// matched it. Classic missed webhook / unbooked settlement.
    UnmatchedStatement {
        /// The orphan bank/PSP line.
        line: StatementLine,
    },

    /// The ledger has a transaction in the window that no statement
    /// line accounts for. Either the rail hasn't reported yet, or we
    /// booked something the bank never saw.
    UnmatchedLedger {
        /// The orphan ledger transaction.
        tx_id: TransactionId,
        /// Its `external_id`, if any, to speed operator lookup.
        external_id: Option<String>,
    },

    /// A line and a transaction matched on reference, but the amounts
    /// differ. Often a fee netted out, or a partial settlement.
    AmountMismatch {
        /// The statement line.
        line: StatementLine,
        /// The matched ledger transaction.
        tx_id: TransactionId,
        /// The ledger transaction's settled magnitude.
        ledger_amount: Money,
    },

    /// Matched on reference and amount, but the ledger's lifecycle
    /// state contradicts the statement (e.g. the bank settled it but
    /// our tx is still `Pending`, or it's `Archived`/voided yet the
    /// bank moved money anyway).
    StatusMismatch {
        /// The statement line.
        line: StatementLine,
        /// The matched ledger transaction.
        tx_id: TransactionId,
        /// The ledger transaction's current status.
        ledger_status: Status,
    },
}

/// A flat, storage-friendly description of one discrepancy.
///
/// The `task_id` is **deterministic** — derived only from the
/// discrepancy's identifying fields, never from a clock or random
/// source — so recording the same unresolved discrepancy twice is
/// idempotent (a store keys on it). This is what lets a nightly
/// reconciliation job run repeatedly without piling up duplicate
/// operator tasks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct TaskDescriptor {
    /// Stable id: `"{kind}:{source_id|tx_id}"`.
    pub task_id: String,
    /// Discrepancy class.
    pub kind: &'static str,
    /// One-line operator-facing summary.
    pub detail: String,
    /// Related ledger transaction, if any.
    pub ledger_tx_id: Option<TransactionId>,
    /// Related statement line source id, if any.
    pub statement_source_id: Option<String>,
}

impl Discrepancy {
    /// Project to a deterministic [`TaskDescriptor`] for persistence.
    #[must_use]
    pub fn task_descriptor(&self) -> TaskDescriptor {
        match self {
            Self::UnmatchedStatement { line } => TaskDescriptor {
                task_id: format!("unmatched_statement:{}", line.source_id),
                kind: "unmatched_statement",
                detail: format!(
                    "statement line {} ({} {}) has no matching ledger tx",
                    line.source_id,
                    line.amount.minor_units,
                    line.amount.currency.code()
                ),
                ledger_tx_id: None,
                statement_source_id: Some(line.source_id.clone()),
            },
            Self::UnmatchedLedger { tx_id, external_id } => TaskDescriptor {
                task_id: format!("unmatched_ledger:{}", tx_id.0),
                kind: "unmatched_ledger",
                detail: format!(
                    "ledger tx {} (external_id {:?}) has no matching statement line",
                    tx_id.0, external_id
                ),
                ledger_tx_id: Some(*tx_id),
                statement_source_id: None,
            },
            Self::AmountMismatch {
                line,
                tx_id,
                ledger_amount,
            } => TaskDescriptor {
                task_id: format!("amount_mismatch:{}", line.source_id),
                kind: "amount_mismatch",
                detail: format!(
                    "line {} says {} {} but ledger tx {} settled {} {}",
                    line.source_id,
                    line.amount.minor_units,
                    line.amount.currency.code(),
                    tx_id.0,
                    ledger_amount.minor_units,
                    ledger_amount.currency.code()
                ),
                ledger_tx_id: Some(*tx_id),
                statement_source_id: Some(line.source_id.clone()),
            },
            Self::StatusMismatch {
                line,
                tx_id,
                ledger_status,
            } => TaskDescriptor {
                task_id: format!("status_mismatch:{}", line.source_id),
                kind: "status_mismatch",
                detail: format!(
                    "line {} settled at the bank but ledger tx {} is {:?}",
                    line.source_id, tx_id.0, ledger_status
                ),
                ledger_tx_id: Some(*tx_id),
                statement_source_id: Some(line.source_id.clone()),
            },
        }
    }
}

/// The outcome of one reconciliation run.
///
/// Serializable so an operator can pipe it straight into whatever
/// ticketing / alerting they run. The graph-materialization path in
/// `op-graph` consumes the same `discrepancies` list.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct ReconciliationReport {
    /// The `[start, end]` unix-second window the caller reconciled.
    pub window: (u64, u64),
    /// Lines that matched a ledger tx exactly (reference + amount +
    /// consistent status).
    pub matched: usize,
    /// Lines matched only by the amount/window heuristic (no shared
    /// reference). Counted as reconciled but flagged because the join
    /// is weaker — an operator may want to spot-check these.
    pub fuzzy_matched: usize,
    /// Everything that didn't cleanly reconcile.
    pub discrepancies: Vec<Discrepancy>,
    /// The full set of `(statement source_id → ledger tx)` matches,
    /// both exact and fuzzy. Used by graph-backed stores to emit
    /// `statement_line --reconciles--> ledger_tx` edges so an
    /// operator can traverse "which bank line settled this tx."
    /// Empty in older serialized reports — the field is
    /// `serde(default)` for backwards compatibility.
    #[serde(default)]
    pub matched_pairs: Vec<crate::matcher::MatchedPair>,
}

impl ReconciliationReport {
    /// True when nothing needs an operator's attention.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.discrepancies.is_empty()
    }

    /// Total lines + transactions that reconciled (exact + fuzzy).
    #[must_use]
    pub fn reconciled(&self) -> usize {
        self.matched + self.fuzzy_matched
    }
}
