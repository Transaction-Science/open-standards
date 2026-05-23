//! The two-tier match engine.
//!
//! Tier 1 (strong): join a statement line to a ledger transaction by
//! shared reference (`StatementLine::external_id` ⟷
//! `Transaction::external_id`). On a hit, compare amount and status.
//!
//! Tier 2 (heuristic): lines that didn't join by reference fall back
//! to "same currency, same magnitude, posted within ±tolerance of the
//! transaction's effective time". Counted as reconciled but flagged
//! `fuzzy` so an operator can spot-check.
//!
//! Anything left on either side is an `Unmatched{Statement,Ledger}`.
//!
//! This is deliberately *not* an optimal bipartite assignment. It's
//! deterministic, O(lines + txs) with the index, and easy to reason
//! about during an audit. Optimal matching is future work and is
//! called out as such in the phase doc.

use std::collections::{HashMap, HashSet};

use op_core::Money;
use op_ledger::{Direction, Status, Transaction};

use crate::discrepancy::Discrepancy;
use crate::statement::StatementLine;

/// Outcome of matching one batch of lines against one slice of txs.
pub struct MatchOutcome {
    /// Exact (reference + amount + consistent status) matches.
    pub matched: usize,
    /// Heuristic (amount + window) matches.
    pub fuzzy_matched: usize,
    /// Everything that didn't cleanly reconcile.
    pub discrepancies: Vec<Discrepancy>,
    /// Pairs that did reconcile — `(statement source_id, ledger tx)`.
    /// Both exact and fuzzy matches appear here; the
    /// [`MatchedPair::fuzzy`] flag distinguishes them. Used by
    /// graph-backed reconciliation stores to draw
    /// `statement_line --reconciles--> ledger_tx` edges so an
    /// operator can traverse from a tx to the bank line that
    /// settled it.
    pub matched_pairs: Vec<MatchedPair>,
}

/// One successful match: a statement line that paired with a ledger
/// transaction.
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct MatchedPair {
    /// `source_id` of the [`StatementLine`].
    pub statement_source_id: String,
    /// The ledger transaction it matched.
    pub tx_id: op_ledger::TransactionId,
    /// `true` when the match was tier-2 (amount + window heuristic
    /// without a shared reference); `false` for a tier-1
    /// reference-key match.
    pub fuzzy: bool,
}

/// The settled magnitude of a transaction: the sum of its debit-side
/// entries. By the double-entry invariant this equals the credit-side
/// sum per currency, so the debit side is a representative total.
///
/// Returns `None` for an entry-less transaction (nothing to compare)
/// and assumes a single settlement currency — the common case for the
/// reference implementation. A genuinely multi-currency (FX) tx would
/// need per-currency reconciliation, which is out of scope for v1.
fn tx_settled_amount(tx: &Transaction) -> Option<Money> {
    let mut total: Option<Money> = None;
    for e in &tx.entries {
        if e.direction != Direction::Debit {
            continue;
        }
        match &mut total {
            None => {
                total = Some(Money {
                    minor_units: e.amount.minor_units.abs(),
                    currency: e.amount.currency,
                });
            }
            Some(t) if t.currency == e.amount.currency => {
                t.minor_units += e.amount.minor_units.abs();
            }
            // Mixed-currency tx: not reconcilable by a single line in
            // v1. Bail so it surfaces as UnmatchedLedger rather than a
            // wrong AmountMismatch.
            Some(_) => return None,
        }
    }
    total
}

/// A statement line means money actually moved at the bank, so the
/// only consistent ledger state is `Posted`.
fn status_is_consistent(status: Status) -> bool {
    matches!(status, Status::Posted)
}

/// Run the two-tier match.
///
/// `window` is `[start, end]` unix seconds; a ledger transaction is
/// considered "in scope" (and thus eligible to be flagged
/// `UnmatchedLedger`) only if its `effective_at_unix_secs` falls
/// inside it. `fuzzy_tolerance_secs` is the ± window for tier-2
/// time proximity.
// Matched-pair plumbing pushed `match_lines` past the 100-line
// pedantic ceiling, but the two tiers are already cleanly bracketed
// and breaking them into helpers would shred the legibility this
// algorithm needs in an audit.
#[allow(clippy::too_many_lines)]
#[must_use]
pub fn match_lines(
    lines: &[StatementLine],
    txs: &[Transaction],
    window: (u64, u64),
    fuzzy_tolerance_secs: u64,
) -> MatchOutcome {
    let mut by_external: HashMap<&str, &Transaction> = HashMap::new();
    for tx in txs {
        if let Some(ext) = &tx.external_id {
            by_external.insert(ext.as_str(), tx);
        }
    }

    let mut used: HashSet<op_ledger::TransactionId> = HashSet::new();
    let mut discrepancies = Vec::new();
    let mut matched = 0usize;
    let mut fuzzy_matched = 0usize;
    let mut matched_pairs: Vec<MatchedPair> = Vec::new();
    let mut deferred: Vec<&StatementLine> = Vec::new();

    // ---- Tier 1: reference join ----
    for line in lines {
        let Some(ext) = line.external_id.as_deref() else {
            deferred.push(line);
            continue;
        };
        let Some(tx) = by_external.get(ext) else {
            deferred.push(line);
            continue;
        };
        used.insert(tx.id);

        match tx_settled_amount(tx) {
            Some(ledger_amount)
                if ledger_amount.currency == line.amount.currency
                    && ledger_amount.minor_units == line.amount.minor_units =>
            {
                if status_is_consistent(tx.status) {
                    matched += 1;
                    matched_pairs.push(MatchedPair {
                        statement_source_id: line.source_id.clone(),
                        tx_id: tx.id,
                        fuzzy: false,
                    });
                } else {
                    discrepancies.push(Discrepancy::StatusMismatch {
                        line: line.clone(),
                        tx_id: tx.id,
                        ledger_status: tx.status,
                    });
                }
            }
            Some(ledger_amount) => {
                discrepancies.push(Discrepancy::AmountMismatch {
                    line: line.clone(),
                    tx_id: tx.id,
                    ledger_amount,
                });
            }
            None => {
                // Entry-less or multi-currency tx we can't value:
                // treat the line as unmatched rather than guess.
                discrepancies.push(Discrepancy::UnmatchedStatement { line: line.clone() });
            }
        }
    }

    // ---- Tier 2: amount + window heuristic ----
    for line in deferred {
        let hit = txs.iter().find(|tx| {
            if used.contains(&tx.id) {
                return false;
            }
            let Some(amt) = tx_settled_amount(tx) else {
                return false;
            };
            amt.currency == line.amount.currency
                && amt.minor_units == line.amount.minor_units
                && tx.effective_at_unix_secs.abs_diff(line.posted_at_unix_secs)
                    <= fuzzy_tolerance_secs
        });
        if let Some(tx) = hit {
            used.insert(tx.id);
            fuzzy_matched += 1;
            matched_pairs.push(MatchedPair {
                statement_source_id: line.source_id.clone(),
                tx_id: tx.id,
                fuzzy: true,
            });
        } else {
            discrepancies.push(Discrepancy::UnmatchedStatement { line: line.clone() });
        }
    }

    // ---- Leftover ledger txs in the window ----
    for tx in txs {
        if used.contains(&tx.id) {
            continue;
        }
        let in_window =
            tx.effective_at_unix_secs >= window.0 && tx.effective_at_unix_secs <= window.1;
        if in_window {
            discrepancies.push(Discrepancy::UnmatchedLedger {
                tx_id: tx.id,
                external_id: tx.external_id.clone(),
            });
        }
    }

    MatchOutcome {
        matched,
        fuzzy_matched,
        discrepancies,
        matched_pairs,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::statement::{LineDirection, StatementLine};
    use op_core::{Currency, Money};
    use op_ledger::{AccountId, Entry, LedgerId, Transaction};

    fn line(src: &str, ext: Option<&str>, minor: i64, at: u64) -> StatementLine {
        let l = StatementLine::new(
            src,
            Money::from_minor(minor, Currency::USD),
            LineDirection::Credit,
            at,
        );
        match ext {
            Some(e) => l.with_external_id(e),
            None => l,
        }
    }

    fn posted(ext: &str, minor: i64, at: u64) -> Transaction {
        Transaction::new_posted(
            LedgerId::new(),
            at,
            vec![
                Entry::debit(AccountId::new(), Money::from_minor(minor, Currency::USD)),
                Entry::credit(AccountId::new(), Money::from_minor(minor, Currency::USD)),
            ],
        )
        .unwrap()
        .with_external_id(ext)
    }

    fn pending(ext: &str, minor: i64, at: u64) -> Transaction {
        Transaction::new_pending(
            LedgerId::new(),
            at,
            vec![
                Entry::debit(AccountId::new(), Money::from_minor(minor, Currency::USD)),
                Entry::credit(AccountId::new(), Money::from_minor(minor, Currency::USD)),
            ],
        )
        .unwrap()
        .with_external_id(ext)
    }

    const W: (u64, u64) = (0, 1_000_000);
    const TOL: u64 = 3600;

    #[test]
    fn exact_reference_match() {
        let o = match_lines(
            &[line("n1", Some("ORD-1"), 5000, 100)],
            &[posted("ORD-1", 5000, 100)],
            W,
            TOL,
        );
        assert_eq!(o.matched, 1);
        assert!(o.discrepancies.is_empty());
    }

    #[test]
    fn unmatched_statement_when_no_tx() {
        let o = match_lines(&[line("n1", Some("NOPE"), 1, 1)], &[], W, TOL);
        assert!(matches!(
            o.discrepancies.as_slice(),
            [Discrepancy::UnmatchedStatement { .. }]
        ));
    }

    #[test]
    fn unmatched_ledger_in_window() {
        let o = match_lines(&[], &[posted("ORD-9", 10, 500)], W, TOL);
        assert!(matches!(
            o.discrepancies.as_slice(),
            [Discrepancy::UnmatchedLedger { external_id: Some(e), .. }] if e == "ORD-9"
        ));
    }

    #[test]
    fn out_of_window_ledger_tx_not_flagged() {
        // tx effective beyond the window end → not our concern.
        let o = match_lines(&[], &[posted("ORD-X", 10, 2_000_000)], W, TOL);
        assert!(o.discrepancies.is_empty());
    }

    #[test]
    fn amount_mismatch() {
        let o = match_lines(
            &[line("n1", Some("ORD-2"), 5000, 100)],
            &[posted("ORD-2", 4999, 100)],
            W,
            TOL,
        );
        match o.discrepancies.as_slice() {
            [Discrepancy::AmountMismatch { ledger_amount, .. }] => {
                assert_eq!(ledger_amount.minor_units, 4999);
            }
            other => panic!("expected AmountMismatch, got {other:?}"),
        }
    }

    #[test]
    fn status_mismatch_when_ledger_pending() {
        let o = match_lines(
            &[line("n1", Some("ORD-3"), 5000, 100)],
            &[pending("ORD-3", 5000, 100)],
            W,
            TOL,
        );
        match o.discrepancies.as_slice() {
            [Discrepancy::StatusMismatch { ledger_status, .. }] => {
                assert_eq!(*ledger_status, op_ledger::Status::Pending);
            }
            other => panic!("expected StatusMismatch, got {other:?}"),
        }
        assert_eq!(o.matched, 0);
    }

    #[test]
    fn fuzzy_match_within_tolerance() {
        // No shared reference, but same amount/currency and posted
        // 30 min from the tx's effective time (< 1h tolerance).
        let o = match_lines(
            &[line("n1", None, 7777, 1_000 + 1_800)],
            &[posted("ORD-4", 7777, 1_000)],
            W,
            TOL,
        );
        assert_eq!(o.fuzzy_matched, 1);
        assert!(o.discrepancies.is_empty());
    }

    #[test]
    fn fuzzy_miss_outside_tolerance_is_unmatched_both_sides() {
        // Amount matches but 2h apart (> 1h tol): the line is
        // unmatched AND the tx is left as unmatched-ledger.
        let o = match_lines(
            &[line("n1", None, 7777, 1_000 + 7_200)],
            &[posted("ORD-5", 7777, 1_000)],
            W,
            TOL,
        );
        assert_eq!(o.fuzzy_matched, 0);
        assert_eq!(o.discrepancies.len(), 2);
        assert!(
            o.discrepancies
                .iter()
                .any(|d| matches!(d, Discrepancy::UnmatchedStatement { .. }))
        );
        assert!(
            o.discrepancies
                .iter()
                .any(|d| matches!(d, Discrepancy::UnmatchedLedger { .. }))
        );
    }

    #[test]
    fn multi_currency_tx_cannot_be_valued() {
        // A tx whose debit side spans two currencies isn't reconcilable
        // by a single line in v1: surfaces as UnmatchedStatement, never
        // a wrong AmountMismatch.
        let tx = Transaction::new_posted(
            LedgerId::new(),
            100,
            vec![
                Entry::debit(AccountId::new(), Money::from_minor(100, Currency::USD)),
                Entry::debit(AccountId::new(), Money::from_minor(100, Currency::EUR)),
                Entry::credit(AccountId::new(), Money::from_minor(100, Currency::USD)),
                Entry::credit(AccountId::new(), Money::from_minor(100, Currency::EUR)),
            ],
        )
        .unwrap()
        .with_external_id("FX-1");
        let o = match_lines(&[line("n1", Some("FX-1"), 100, 100)], &[tx], W, TOL);
        assert!(matches!(
            o.discrepancies.as_slice(),
            [Discrepancy::UnmatchedStatement { .. }]
        ));
    }

    #[test]
    fn rerun_is_idempotent() {
        let lines = [line("n1", Some("ORD-6"), 5000, 100)];
        let txs = [pending("ORD-6", 5000, 100)];
        let a = match_lines(&lines, &txs, W, TOL);
        let b = match_lines(&lines, &txs, W, TOL);
        assert_eq!(a.matched, b.matched);
        assert_eq!(a.fuzzy_matched, b.fuzzy_matched);
        assert_eq!(a.discrepancies, b.discrepancies);
    }
}
