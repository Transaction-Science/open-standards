//! Economic-nexus monitor.
//!
//! After the US Supreme Court's *South Dakota v. Wayfair* decision
//! (2018), every US state imposes its own "economic nexus" thresholds.
//! A remote seller becomes obligated to collect that state's sales tax
//! once it crosses the threshold — typically `$100,000 in gross
//! revenue` *or* `200 separate transactions` over a rolling
//! 12-month window. The exact thresholds vary:
//!
//! - **Most states**: $100k OR 200 tx (the South Dakota baseline).
//! - **California, Texas, New York**: $500k revenue, no tx threshold.
//! - **Massachusetts**: $100k revenue, no tx threshold (after a 2023
//!   change).
//! - **Kansas, Oklahoma**: $100k revenue, no tx threshold.
//! - **A handful**: $250k revenue (Alabama, Mississippi).
//!
//! This module tracks per-state running totals and emits
//! [`NexusEvent::Triggered`] the moment a threshold is crossed.
//! Operators wire the event into their compliance pipeline — register
//! with that state's DOR within the window the state requires
//! (usually 30 days).

use chrono::NaiveDate;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::jurisdiction::RegionCode;

/// One US state's economic-nexus threshold.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct NexusThreshold {
    /// Two-letter state code (`CA`, `TX`).
    pub state: RegionCode,
    /// Revenue threshold in USD (whole dollars). `None` means the
    /// state has no revenue threshold (very rare).
    pub revenue_usd: Option<Decimal>,
    /// Transaction-count threshold. `None` means the state has
    /// dropped its transaction threshold (most have, post-2024).
    pub transactions: Option<u32>,
    /// Whether crossing EITHER threshold triggers, or only BOTH.
    /// All states are "either" in current practice; we keep the
    /// field for future-proofing.
    pub either: bool,
}

/// Running total of recorded volume in one state.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct RunningTotal {
    /// Sum of all line amounts (USD whole dollars).
    pub revenue_usd: Decimal,
    /// Count of distinct transactions recorded.
    pub transactions: u32,
    /// Whether we have already emitted a Triggered event for this
    /// state — idempotency guard so we don't fire repeatedly after
    /// the first crossing.
    pub triggered: bool,
}

/// A single recorded transaction relevant to nexus tracking.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct TransactionRecord {
    /// Destination state (US only — the monitor ignores non-US).
    pub state: RegionCode,
    /// Transaction date (used for the rolling-window calculation
    /// when operators wire one — this module tracks lifetime totals;
    /// rolling-window aggregation is a downstream concern).
    pub date: NaiveDate,
    /// Net (pre-tax) revenue, USD whole dollars.
    pub revenue_usd: Decimal,
}

/// Event emitted when a threshold crossing happens.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum NexusEvent {
    /// Seller has just crossed the threshold for this state and
    /// should register to collect within the state-mandated window.
    Triggered {
        /// State whose threshold was crossed.
        state: RegionCode,
        /// Lifetime revenue at crossing.
        revenue_usd: Decimal,
        /// Lifetime transaction count at crossing.
        transactions: u32,
        /// Which dimension crossed (`"revenue"` or `"transactions"`).
        dimension: &'static str,
    },
    /// Seller has approached but not crossed (90%+). Useful for
    /// proactive registration.
    Approaching {
        /// State being approached.
        state: RegionCode,
        /// Percentage of the lower of the two thresholds, in basis
        /// points (`9500` = 95%).
        bp: u32,
    },
}

/// Per-state nexus monitor.
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct NexusMonitor {
    /// Configured thresholds, keyed by state.
    pub thresholds: BTreeMap<RegionCode, NexusThreshold>,
    /// Lifetime running totals per state.
    pub current_volume: BTreeMap<RegionCode, RunningTotal>,
}

impl NexusMonitor {
    /// Construct with no thresholds. Use [`Self::with_default_us_states`]
    /// for the baseline-state set.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Construct with the South Dakota baseline ($100k / 200tx) for
    /// every US state EXCEPT the ones with materially different rules.
    /// Adds the state-specific overrides for CA, TX, NY, MA, KS, OK,
    /// AL, and MS.
    ///
    /// Source: each state's Department of Revenue published thresholds
    /// as of 2026 Q1.
    #[must_use]
    pub fn with_default_us_states() -> Self {
        let mut m = Self::new();

        // South Dakota baseline.
        let baseline = |state: &str| NexusThreshold {
            state: RegionCode::new(state),
            revenue_usd: Some(Decimal::new(100_000, 0)),
            transactions: Some(200),
            either: true,
        };

        let half_mil_rev_only = |state: &str| NexusThreshold {
            state: RegionCode::new(state),
            revenue_usd: Some(Decimal::new(500_000, 0)),
            transactions: None,
            either: true,
        };
        let hundred_k_rev_only = |state: &str| NexusThreshold {
            state: RegionCode::new(state),
            revenue_usd: Some(Decimal::new(100_000, 0)),
            transactions: None,
            either: true,
        };
        let quarter_mil_rev_only = |state: &str| NexusThreshold {
            state: RegionCode::new(state),
            revenue_usd: Some(Decimal::new(250_000, 0)),
            transactions: None,
            either: true,
        };

        // Overrides first.
        for s in ["CA", "TX", "NY"] {
            m.thresholds.insert(RegionCode::new(s), half_mil_rev_only(s));
        }
        for s in ["MA", "KS", "OK"] {
            m.thresholds.insert(RegionCode::new(s), hundred_k_rev_only(s));
        }
        for s in ["AL", "MS"] {
            m.thresholds
                .insert(RegionCode::new(s), quarter_mil_rev_only(s));
        }

        // Every other state with a sales tax — South Dakota baseline.
        // The 5 NOMAD states (NH, OR, MT, AK, DE) have no statewide
        // sales tax and so no economic-nexus threshold either; we
        // omit them. Alaska has local sales taxes only.
        for s in [
            "AZ", "AR", "CO", "CT", "FL", "GA", "HI", "ID", "IL", "IN", "IA", "KY", "LA", "ME",
            "MD", "MI", "MN", "MO", "NE", "NV", "NJ", "NM", "NC", "ND", "OH", "PA", "RI", "SC",
            "SD", "TN", "UT", "VT", "VA", "WA", "WV", "WI", "WY",
        ] {
            m.thresholds.entry(RegionCode::new(s)).or_insert(baseline(s));
        }
        m
    }

    /// Record a transaction. Returns any new [`NexusEvent`]s — at most
    /// one `Triggered` and one `Approaching` per call.
    pub fn record(&mut self, tx: &TransactionRecord) -> Vec<NexusEvent> {
        let mut events = Vec::new();
        let Some(threshold) = self.thresholds.get(&tx.state).cloned() else {
            return events;
        };
        let total = self
            .current_volume
            .entry(tx.state.clone())
            .or_default();
        total.revenue_usd += tx.revenue_usd;
        total.transactions += 1;

        if total.triggered {
            // Already over — don't keep firing.
            return events;
        }

        let revenue_crossed = threshold
            .revenue_usd
            .is_some_and(|t| total.revenue_usd >= t);
        let tx_crossed = threshold
            .transactions
            .is_some_and(|t| total.transactions >= t);

        if revenue_crossed || tx_crossed {
            let dimension = if revenue_crossed {
                "revenue"
            } else {
                "transactions"
            };
            total.triggered = true;
            events.push(NexusEvent::Triggered {
                state: tx.state.clone(),
                revenue_usd: total.revenue_usd,
                transactions: total.transactions,
                dimension,
            });
            return events;
        }

        // Approaching check: 95% of the lower-bound dimension.
        let bp_rev = threshold.revenue_usd.map(|t| {
            ((total.revenue_usd * Decimal::new(10_000, 0)) / t)
                .round()
                .try_into()
                .unwrap_or(0u32)
        });
        let bp_tx = threshold
            .transactions
            .map(|t| (u32::from(total.transactions >= t.saturating_sub(20))) * 9500);
        let bp = match (bp_rev, bp_tx) {
            (Some(a), Some(b)) => a.max(b),
            (Some(a), None) => a,
            (None, Some(b)) => b,
            (None, None) => 0,
        };
        if (9500..10_000).contains(&bp) {
            events.push(NexusEvent::Approaching {
                state: tx.state.clone(),
                bp,
            });
        }

        events
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tx(state: &str, revenue: i64) -> TransactionRecord {
        TransactionRecord {
            state: RegionCode::new(state),
            date: NaiveDate::parse_from_str("2026-06-15", "%Y-%m-%d").unwrap(),
            revenue_usd: Decimal::new(revenue, 0),
        }
    }

    #[test]
    fn south_dakota_baseline_triggers_on_200th_transaction() {
        let mut m = NexusMonitor::with_default_us_states();
        // 199 small transactions.
        for _ in 0..199 {
            let events = m.record(&tx("SD", 10));
            assert!(
                !events
                    .iter()
                    .any(|e| matches!(e, NexusEvent::Triggered { .. }))
            );
        }
        // 200th — must trigger.
        let events = m.record(&tx("SD", 10));
        let trig = events.iter().find_map(|e| match e {
            NexusEvent::Triggered { dimension, .. } => Some(*dimension),
            NexusEvent::Approaching { .. } => None,
        });
        assert_eq!(trig, Some("transactions"));
    }

    #[test]
    fn revenue_only_states_ignore_transaction_count() {
        let mut m = NexusMonitor::with_default_us_states();
        // 500 tiny transactions to CA — no trigger because CA has no
        // transaction threshold and revenue stays low.
        for _ in 0..500 {
            let events = m.record(&tx("CA", 10));
            assert!(
                !events
                    .iter()
                    .any(|e| matches!(e, NexusEvent::Triggered { .. }))
            );
        }
    }

    #[test]
    fn ca_triggers_at_500k_revenue() {
        let mut m = NexusMonitor::with_default_us_states();
        // One big transaction.
        let events = m.record(&tx("CA", 500_000));
        let trig = events
            .iter()
            .any(|e| matches!(e, NexusEvent::Triggered { dimension: "revenue", .. }));
        assert!(trig);
    }

    #[test]
    fn idempotent_after_first_trigger() {
        let mut m = NexusMonitor::with_default_us_states();
        let _ = m.record(&tx("CA", 500_000));
        // A second large transaction should not re-emit Triggered.
        let events = m.record(&tx("CA", 100_000));
        assert!(
            !events
                .iter()
                .any(|e| matches!(e, NexusEvent::Triggered { .. }))
        );
    }

    #[test]
    fn untracked_state_emits_nothing() {
        let mut m = NexusMonitor::new();
        // No thresholds configured.
        let events = m.record(&tx("WA", 1_000_000));
        assert!(events.is_empty());
    }
}
