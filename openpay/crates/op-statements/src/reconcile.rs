//! Reconciliation against op-ledger style entries.
//!
//! `op-statements` doesn't take a hard dep on `op-ledger` (the
//! crate-level dependency graph stays minimal). Instead we accept a
//! [`LedgerEntry`] — a structural copy of the op-ledger shape — and
//! pair it against [`StatementLine`](crate::StatementLine)s on
//! `external_id` first, then by `(amount, currency, posted_at ± window)`.
//!
//! Output is a [`ReconRecord`] per line classifying it as
//! [`ReconStatus::Matched`], [`ReconStatus::MissingInLedger`], or
//! [`ReconStatus::MissingOnStatement`].

use op_core::Money;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::statement::Statement;

/// Direction of a ledger entry from the merchant's view.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LedgerDirection {
    /// Money credited TO the merchant.
    Credit,
    /// Money debited FROM the merchant.
    Debit,
}

/// A ledger entry shape — minimal structural copy of the relevant
/// fields from [`op_ledger::Entry`] joined with its parent
/// transaction. Caller assembles these from their ledger store.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct LedgerEntry {
    /// Stable ledger transaction id stringified (or external_id).
    pub id: String,
    /// The op-ledger transaction's `external_id` (the strong join
    /// key to the statement line). `None` if the ledger transaction
    /// had no idempotency key.
    pub external_id: Option<String>,
    /// Signed-magnitude.
    pub amount: Money,
    /// Which side.
    pub direction: LedgerDirection,
    /// When the ledger says it happened.
    pub posted_at_unix_secs: u64,
}

/// Match status for one reconciliation record.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReconStatus {
    /// Found in both ledger and statement; amounts agree.
    Matched,
    /// Found on the statement, not in the ledger window.
    MissingInLedger,
    /// Found in the ledger, not on the statement.
    MissingOnStatement,
    /// Both present but amounts differ.
    AmountMismatch,
}

/// One row of reconciliation output.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconRecord {
    /// What happened with this pairing.
    pub status: ReconStatus,
    /// The statement line id, if any.
    pub statement_line_id: Option<String>,
    /// The ledger entry id, if any.
    pub ledger_entry_id: Option<String>,
    /// Amount delta in minor units (`statement.amount - ledger.amount`).
    /// Zero when both sides present and equal.
    pub delta_minor_units: i64,
}

/// Reconciliation engine: walks one [`Statement`] against a slice of
/// [`LedgerEntry`]s, producing [`ReconRecord`]s.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Reconciler {
    /// Permitted timestamp drift in seconds when falling back to
    /// `(amount, currency, posted_at)` matching after the `external_id`
    /// join fails. Default 24h.
    pub timestamp_window_secs: u64,
}

impl Reconciler {
    /// New reconciler with the given window.
    #[must_use]
    pub const fn with_window(timestamp_window_secs: u64) -> Self {
        Self {
            timestamp_window_secs,
        }
    }

    /// Reconcile. Linear two-pass:
    /// 1. external_id direct join.
    /// 2. for remaining lines, amount + window heuristic.
    ///
    /// # Errors
    /// None today; reserved for future structural checks.
    pub fn reconcile(
        &self,
        statement: &Statement,
        ledger: &[LedgerEntry],
    ) -> Result<Vec<ReconRecord>> {
        let mut records = Vec::with_capacity(statement.lines.len() + ledger.len());
        let mut ledger_consumed = vec![false; ledger.len()];

        // Pass 1: external_id join.
        for line in &statement.lines {
            let Some(line_ext) = &line.external_id else {
                continue;
            };
            if let Some((idx, ent)) = ledger
                .iter()
                .enumerate()
                .find(|(i, e)| !ledger_consumed[*i] && e.external_id.as_deref() == Some(line_ext))
            {
                ledger_consumed[idx] = true;
                let delta = line.amount.minor_units - ent.amount.minor_units;
                let status = if delta == 0
                    && line.amount.currency == ent.amount.currency
                {
                    ReconStatus::Matched
                } else {
                    ReconStatus::AmountMismatch
                };
                records.push(ReconRecord {
                    status,
                    statement_line_id: Some(line.id.clone()),
                    ledger_entry_id: Some(ent.id.clone()),
                    delta_minor_units: delta,
                });
            }
        }

        // Pass 2: heuristic match on remaining lines.
        for line in &statement.lines {
            if records.iter().any(|r| r.statement_line_id.as_deref() == Some(&line.id)) {
                continue;
            }
            let candidate = ledger
                .iter()
                .enumerate()
                .find(|(i, e)| {
                    !ledger_consumed[*i]
                        && e.amount == line.amount
                        && time_within(e.posted_at_unix_secs, line.posted_at_unix_secs, self.timestamp_window_secs)
                });
            if let Some((idx, ent)) = candidate {
                ledger_consumed[idx] = true;
                records.push(ReconRecord {
                    status: ReconStatus::Matched,
                    statement_line_id: Some(line.id.clone()),
                    ledger_entry_id: Some(ent.id.clone()),
                    delta_minor_units: 0,
                });
            } else {
                records.push(ReconRecord {
                    status: ReconStatus::MissingInLedger,
                    statement_line_id: Some(line.id.clone()),
                    ledger_entry_id: None,
                    delta_minor_units: line.amount.minor_units,
                });
            }
        }

        // Pass 3: ledger entries we never consumed.
        for (i, ent) in ledger.iter().enumerate() {
            if ledger_consumed[i] {
                continue;
            }
            records.push(ReconRecord {
                status: ReconStatus::MissingOnStatement,
                statement_line_id: None,
                ledger_entry_id: Some(ent.id.clone()),
                delta_minor_units: -ent.amount.minor_units,
            });
        }

        Ok(records)
    }
}

const fn time_within(a: u64, b: u64, window: u64) -> bool {
    let diff = if a > b { a - b } else { b - a };
    diff <= window
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cadence::Period;
    use crate::statement::{Statement, StatementLine, StatementLineKind};
    use op_core::{Currency, Money};

    fn stmt_with(lines: Vec<StatementLine>) -> Statement {
        let mut s = Statement::new(
            "S",
            "M",
            Period::new(0, 86_400).unwrap(),
            Currency::USD,
        )
        .unwrap();
        for l in lines {
            s.push_line(l).unwrap();
        }
        s
    }

    #[test]
    fn external_id_matches() {
        let s = stmt_with(vec![
            StatementLine::new(
                "l1",
                StatementLineKind::GrossCapture,
                Money::from_minor(1_000, Currency::USD),
                100,
            )
            .with_external_id("ord-1"),
        ]);
        let ledger = vec![LedgerEntry {
            id: "tx-1".into(),
            external_id: Some("ord-1".into()),
            amount: Money::from_minor(1_000, Currency::USD),
            direction: LedgerDirection::Credit,
            posted_at_unix_secs: 100,
        }];
        let r = Reconciler::with_window(86_400).reconcile(&s, &ledger).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].status, ReconStatus::Matched);
    }

    #[test]
    fn amount_mismatch_flagged() {
        let s = stmt_with(vec![
            StatementLine::new(
                "l1",
                StatementLineKind::GrossCapture,
                Money::from_minor(1_000, Currency::USD),
                100,
            )
            .with_external_id("ord-1"),
        ]);
        let ledger = vec![LedgerEntry {
            id: "tx-1".into(),
            external_id: Some("ord-1".into()),
            amount: Money::from_minor(999, Currency::USD),
            direction: LedgerDirection::Credit,
            posted_at_unix_secs: 100,
        }];
        let r = Reconciler::with_window(86_400).reconcile(&s, &ledger).unwrap();
        assert_eq!(r[0].status, ReconStatus::AmountMismatch);
        assert_eq!(r[0].delta_minor_units, 1);
    }

    #[test]
    fn missing_in_ledger() {
        let s = stmt_with(vec![StatementLine::new(
            "l1",
            StatementLineKind::GrossCapture,
            Money::from_minor(1_000, Currency::USD),
            100,
        )]);
        let r = Reconciler::with_window(86_400).reconcile(&s, &[]).unwrap();
        assert_eq!(r[0].status, ReconStatus::MissingInLedger);
    }

    #[test]
    fn missing_on_statement() {
        let s = stmt_with(vec![]);
        let ledger = vec![LedgerEntry {
            id: "tx-1".into(),
            external_id: None,
            amount: Money::from_minor(1_000, Currency::USD),
            direction: LedgerDirection::Credit,
            posted_at_unix_secs: 100,
        }];
        let r = Reconciler::with_window(86_400).reconcile(&s, &ledger).unwrap();
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].status, ReconStatus::MissingOnStatement);
    }

    #[test]
    fn window_heuristic_matches_without_external_id() {
        let s = stmt_with(vec![StatementLine::new(
            "l1",
            StatementLineKind::GrossCapture,
            Money::from_minor(1_000, Currency::USD),
            100,
        )]);
        let ledger = vec![LedgerEntry {
            id: "tx-1".into(),
            external_id: None,
            amount: Money::from_minor(1_000, Currency::USD),
            direction: LedgerDirection::Credit,
            posted_at_unix_secs: 200, // 100s diff, well within 1d window
        }];
        let r = Reconciler::with_window(86_400).reconcile(&s, &ledger).unwrap();
        assert_eq!(r[0].status, ReconStatus::Matched);
    }
}
