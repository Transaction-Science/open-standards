//! The reconciler — drives a source against a ledger window.

use op_ledger::Transaction;

use crate::discrepancy::ReconciliationReport;
use crate::error::{Error, Result};
use crate::matcher::match_lines;
use crate::source::ReconciliationSource;

/// Default tier-2 time tolerance: a bank's value date and our
/// effective timestamp routinely differ by up to a day (cut-off
/// times, weekend batching). 24h is a conservative reference default;
/// operators tune it via [`Reconciler::with_fuzzy_tolerance_secs`].
pub const DEFAULT_FUZZY_TOLERANCE_SECS: u64 = 86_400;

/// Reconciles a [`ReconciliationSource`] against a caller-selected
/// slice of ledger transactions.
#[derive(Debug, Clone)]
pub struct Reconciler {
    window: (u64, u64),
    fuzzy_tolerance_secs: u64,
}

impl Reconciler {
    /// Construct for the `[start, end]` unix-second window the caller
    /// intends to reconcile.
    ///
    /// # Errors
    /// [`Error::InvalidWindow`] if `end < start`.
    pub fn new(window_start_unix_secs: u64, window_end_unix_secs: u64) -> Result<Self> {
        if window_end_unix_secs < window_start_unix_secs {
            return Err(Error::InvalidWindow {
                start: window_start_unix_secs,
                end: window_end_unix_secs,
            });
        }
        Ok(Self {
            window: (window_start_unix_secs, window_end_unix_secs),
            fuzzy_tolerance_secs: DEFAULT_FUZZY_TOLERANCE_SECS,
        })
    }

    /// Builder: override the tier-2 (heuristic) time tolerance.
    #[must_use]
    pub fn with_fuzzy_tolerance_secs(mut self, secs: u64) -> Self {
        self.fuzzy_tolerance_secs = secs;
        self
    }

    /// Reconcile `source` against `ledger_txs`.
    ///
    /// `ledger_txs` is the window of transactions the caller pulled
    /// from their store (this crate never imposes a listing API on
    /// `op-ledger` — see the crate docs).
    ///
    /// A per-line parse failure **aborts the run** rather than being
    /// folded into the report: a corrupt statement file must not
    /// silently under-report discrepancies by dropping the lines it
    /// couldn't read.
    ///
    /// # Errors
    /// Propagates the first per-line parse error from the source.
    #[tracing::instrument(
        name = "reconciliation.reconcile",
        skip(self, source, ledger_txs),
        fields(
            window_start = self.window.0,
            window_end = self.window.1,
            tx_count = ledger_txs.len(),
        ),
    )]
    pub fn reconcile(
        &self,
        source: &dyn ReconciliationSource,
        ledger_txs: &[Transaction],
    ) -> Result<ReconciliationReport> {
        let mut lines = Vec::new();
        for line in source.iter_lines() {
            lines.push(line?);
        }

        let outcome = match_lines(&lines, ledger_txs, self.window, self.fuzzy_tolerance_secs);

        Ok(ReconciliationReport {
            window: self.window,
            matched: outcome.matched,
            fuzzy_matched: outcome.fuzzy_matched,
            discrepancies: outcome.discrepancies,
            matched_pairs: outcome.matched_pairs,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::source::ReconciliationSource;
    use crate::statement::{LineDirection, StatementLine};
    use op_core::{Currency, Money};
    use op_ledger::{Entry, Transaction};

    struct VecSource(Vec<StatementLine>);
    impl ReconciliationSource for VecSource {
        fn iter_lines(&self) -> Box<dyn Iterator<Item = Result<StatementLine>> + '_> {
            Box::new(self.0.iter().cloned().map(Ok))
        }
    }

    fn posted_tx(ext: &str, minor: i64, at: u64) -> Transaction {
        let acct_d = op_ledger::AccountId::new();
        let acct_c = op_ledger::AccountId::new();
        Transaction::new_posted(
            op_ledger::LedgerId::new(),
            at,
            vec![
                Entry::debit(acct_d, Money::from_minor(minor, Currency::USD)),
                Entry::credit(acct_c, Money::from_minor(minor, Currency::USD)),
            ],
        )
        .unwrap()
        .with_external_id(ext)
    }

    #[test]
    fn rejects_inverted_window() {
        let err = Reconciler::new(100, 50).unwrap_err();
        assert!(matches!(
            err,
            Error::InvalidWindow {
                start: 100,
                end: 50
            }
        ));
    }

    #[test]
    fn exact_reference_match_is_clean() {
        let tx = posted_tx("ORD-1", 5000, 1_000);
        let line = StatementLine::new(
            "ntry-1",
            Money::from_minor(5000, Currency::USD),
            LineDirection::Credit,
            1_000,
        )
        .with_external_id("ORD-1");
        let r = Reconciler::new(0, 10_000)
            .unwrap()
            .reconcile(&VecSource(vec![line]), &[tx])
            .unwrap();
        assert_eq!(r.matched, 1);
        assert!(r.is_clean());
    }
}
