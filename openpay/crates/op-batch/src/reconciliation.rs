//! Reconciliation hooks.
//!
//! Once a batch is submitted, the bank eventually sends back a
//! statement (`camt.053` end-of-day for ISO 20022 rails, NACHA
//! prenote/return file for ACH, BAI2 for ACH or wire). Our job:
//! match each statement line against an outbound batch entry.
//!
//! ## Scope
//!
//! We expose a **decision** layer only: given a parsed bank
//! statement and an in-memory list of expected outbound entries,
//! emit [`Match`] decisions (`Matched`, `Unmatched`,
//! `AmountMismatch`, `Duplicate`). Persisting decisions lives in
//! `op-reconciliation` / `op-ledger`. We intentionally don't
//! depend on those crates here — that would create a coupling
//! cycle on the value-flow side of OpenPay. Reconciliation in
//! `op-reconciliation` re-uses the same `Match` shape via plain
//! type re-export when it integrates.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::BatchRail;
use crate::error::{Error, Result};

/// What format the inbound bank statement is in.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReconcileSource {
    /// ISO 20022 `camt.053` end-of-day statement.
    Camt053,
    /// NACHA prenote / return file.
    NachaPrenote,
    /// BAI2 — legacy US bank reporting format.
    Bai2,
}

/// One match decision for a single outbound entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Match {
    /// Statement line and expected entry agree on amount + reference.
    Matched {
        /// Operator's payment id.
        payment_id: String,
        /// Statement-line id (e.g. `camt.053` `Ntry/NtryRef`).
        statement_ref: String,
        /// Settled amount (minor units).
        amount_minor: u64,
    },
    /// Expected entry has no corresponding statement line.
    Unmatched {
        /// Operator's payment id.
        payment_id: String,
        /// What we were looking for.
        expected_amount_minor: u64,
    },
    /// Statement line is present but the amount differs.
    AmountMismatch {
        /// Operator's payment id.
        payment_id: String,
        /// Statement-line id.
        statement_ref: String,
        /// What we expected.
        expected_amount_minor: u64,
        /// What the bank actually settled.
        observed_amount_minor: u64,
    },
    /// Two or more statement lines reference the same payment id.
    Duplicate {
        /// Operator's payment id.
        payment_id: String,
        /// All conflicting statement refs.
        statement_refs: Vec<String>,
    },
}

/// An expected outbound entry the operator awaits.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Expected {
    /// Operator's payment id.
    pub payment_id: String,
    /// Amount in minor units.
    pub amount_minor: u64,
    /// Reference field the bank should echo (`EndToEndId` for SEPA,
    /// trace number for NACHA, `:21:` for MT202).
    pub reference: String,
}

/// One parsed statement line.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct StatementLine {
    /// Statement-line id (the bank's `NtryRef` or trace).
    pub statement_ref: String,
    /// The reference the bank echoed back from our outbound entry.
    pub echoed_reference: String,
    /// Settled amount in minor units.
    pub amount_minor: u64,
    /// Settlement date.
    pub value_date: DateTime<Utc>,
}

/// Aggregate report from one reconciliation run.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReconciliationReport {
    /// Rail this report covers.
    pub rail: BatchRail,
    /// Source format.
    pub source: ReconcileSource,
    /// Per-payment match decisions.
    pub matches: Vec<Match>,
}

impl ReconciliationReport {
    /// True if every expected entry matched cleanly.
    #[must_use]
    pub fn is_clean(&self) -> bool {
        self.matches.iter().all(|m| matches!(m, Match::Matched { .. }))
    }
}

/// Reconcile `expected` against `statement_lines`.
///
/// The match key is the echoed reference (`EndToEndId` /
/// trace / `:21:`). Two outbound entries with the same reference
/// is a configuration bug in the originator — we still flag a
/// `Duplicate` rather than silently dropping one.
///
/// # Errors
/// [`Error::Reconciliation`] only when the inputs are structurally
/// degenerate (a statement line with empty reference cannot be
/// matched on anything, so we surface that as an error rather
/// than producing a misleading `Unmatched`).
pub fn reconcile(
    rail: BatchRail,
    source: ReconcileSource,
    expected: &[Expected],
    statement_lines: &[StatementLine],
) -> Result<ReconciliationReport> {
    use std::collections::HashMap;
    let mut by_ref: HashMap<&str, Vec<&StatementLine>> = HashMap::new();
    for line in statement_lines {
        if line.echoed_reference.is_empty() {
            return Err(Error::Reconciliation(
                "statement line has empty reference".into(),
            ));
        }
        by_ref
            .entry(line.echoed_reference.as_str())
            .or_default()
            .push(line);
    }
    let mut matches = Vec::with_capacity(expected.len());
    for exp in expected {
        let lines = by_ref.get(exp.reference.as_str()).cloned();
        match lines.as_deref() {
            None | Some([]) => {
                matches.push(Match::Unmatched {
                    payment_id: exp.payment_id.clone(),
                    expected_amount_minor: exp.amount_minor,
                });
            }
            Some([line]) => {
                if line.amount_minor == exp.amount_minor {
                    matches.push(Match::Matched {
                        payment_id: exp.payment_id.clone(),
                        statement_ref: line.statement_ref.clone(),
                        amount_minor: line.amount_minor,
                    });
                } else {
                    matches.push(Match::AmountMismatch {
                        payment_id: exp.payment_id.clone(),
                        statement_ref: line.statement_ref.clone(),
                        expected_amount_minor: exp.amount_minor,
                        observed_amount_minor: line.amount_minor,
                    });
                }
            }
            Some(many) => {
                matches.push(Match::Duplicate {
                    payment_id: exp.payment_id.clone(),
                    statement_refs: many.iter().map(|l| l.statement_ref.clone()).collect(),
                });
            }
        }
    }
    Ok(ReconciliationReport {
        rail,
        source,
        matches,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn exp(pid: &str, amt: u64, refn: &str) -> Expected {
        Expected {
            payment_id: pid.into(),
            amount_minor: amt,
            reference: refn.into(),
        }
    }

    fn line(refn: &str, amt: u64) -> StatementLine {
        StatementLine {
            statement_ref: format!("NTRY-{refn}"),
            echoed_reference: refn.into(),
            amount_minor: amt,
            value_date: Utc::now(),
        }
    }

    #[test]
    fn matches_when_amount_and_ref_agree() {
        let rep = reconcile(
            BatchRail::SepaCt,
            ReconcileSource::Camt053,
            &[exp("p1", 100, "INV-1")],
            &[line("INV-1", 100)],
        )
        .unwrap();
        assert!(rep.is_clean());
    }

    #[test]
    fn flags_unmatched() {
        let rep = reconcile(
            BatchRail::SepaCt,
            ReconcileSource::Camt053,
            &[exp("p1", 100, "INV-1"), exp("p2", 200, "INV-2")],
            &[line("INV-1", 100)],
        )
        .unwrap();
        assert!(matches!(rep.matches[1], Match::Unmatched { .. }));
    }

    #[test]
    fn flags_amount_mismatch() {
        let rep = reconcile(
            BatchRail::Nacha,
            ReconcileSource::Bai2,
            &[exp("p1", 100, "INV-1")],
            &[line("INV-1", 99)],
        )
        .unwrap();
        assert!(matches!(rep.matches[0], Match::AmountMismatch { .. }));
    }

    #[test]
    fn flags_duplicate() {
        let rep = reconcile(
            BatchRail::Nacha,
            ReconcileSource::NachaPrenote,
            &[exp("p1", 100, "INV-1")],
            &[line("INV-1", 100), line("INV-1", 100)],
        )
        .unwrap();
        assert!(matches!(rep.matches[0], Match::Duplicate { .. }));
    }

    #[test]
    fn rejects_empty_reference() {
        let mut bad = line("INV-1", 100);
        bad.echoed_reference = String::new();
        let err = reconcile(
            BatchRail::Bacs,
            ReconcileSource::Bai2,
            &[exp("p1", 100, "INV-1")],
            &[bad],
        )
        .unwrap_err();
        assert!(matches!(err, Error::Reconciliation(_)));
    }
}
