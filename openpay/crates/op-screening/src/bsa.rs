//! Bank Secrecy Act / FinCEN reporting helpers.
//!
//! Two reports drive every US-bank-equivalent compliance programme:
//!
//! - **CTR (Currency Transaction Report)** — required for any cash
//!   transaction over USD 10,000 in a day. Banks file 31 CFR § 1010.311.
//!   Beyond the bright-line threshold, the report also catches
//!   *structuring* (multiple sub-threshold cash transactions in
//!   aggregate over $10k in 24 hours).
//!
//! - **SAR (Suspicious Activity Report)** — pattern-based. The big four
//!   pattern families are structuring, round-dollar amounts,
//!   rapid-in-rapid-out funds movement, and geographic anomaly. SARs
//!   are filed at the bank's discretion under 31 CFR § 1020.320.
//!
//! Neither helper produces a *filed* report — that's the operator's
//! job. Both produce [`Trigger`]s identifying transactions that need
//! review.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::lists::CountryCode;

/// A single transaction the BSA helpers consume.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Transaction {
    /// Opaque per-tenant transaction id.
    pub id: String,
    /// Customer / account id.
    pub customer_id: String,
    /// When the transaction happened.
    pub at: DateTime<Utc>,
    /// Amount in minor units (cents). Always positive.
    pub amount_cents: u64,
    /// ISO 4217 currency code.
    pub currency: String,
    /// True for physical cash; controls CTR applicability.
    pub is_cash: bool,
    /// Inbound (true) or outbound (false). Used in rapid in/out detection.
    pub inbound: bool,
    /// Optional originating / receiving country.
    pub country: Option<CountryCode>,
}

/// What kind of suspicious activity we flagged.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TriggerKind {
    /// Single cash transaction crossed the CTR threshold.
    CtrThreshold,
    /// Aggregate cash deposits/withdrawals in 24h above the CTR threshold.
    Structuring,
    /// Suspiciously round-dollar amounts.
    RoundDollar,
    /// Rapid inbound followed by outbound (within a short window).
    RapidInOut,
    /// Geographic anomaly relative to the customer's home countries.
    GeographicAnomaly,
}

/// A single reportable pattern hit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Trigger {
    /// Which pattern family.
    pub kind: TriggerKind,
    /// The transactions that participated in the pattern.
    pub transactions: Vec<String>,
    /// Aggregated amount in cents.
    pub aggregate_cents: u64,
    /// Human-readable explanation.
    pub note: String,
}

/// CTR threshold (USD 10,000.00 = 1,000,000 cents).
pub const CTR_THRESHOLD_CENTS: u64 = 1_000_000;

/// Structuring window (24 hours).
const STRUCTURING_WINDOW_SECS: i64 = 24 * 60 * 60;

/// Rapid-in-out window (15 minutes).
const RAPID_IN_OUT_WINDOW_SECS: i64 = 15 * 60;

/// CTR helper.
#[derive(Debug, Default, Clone)]
pub struct CtrHelper;

impl CtrHelper {
    /// Construct.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }

    /// Walk `transactions` and emit triggers.
    #[must_use]
    pub fn analyze(&self, transactions: &[Transaction]) -> Vec<Trigger> {
        let mut out = Vec::new();
        // Sort by time for sliding-window aggregates.
        let mut txs: Vec<&Transaction> = transactions.iter().filter(|t| t.is_cash).collect();
        txs.sort_by_key(|t| t.at);

        // Single-tx threshold breach.
        for t in &txs {
            if t.amount_cents > CTR_THRESHOLD_CENTS {
                out.push(Trigger {
                    kind: TriggerKind::CtrThreshold,
                    transactions: vec![t.id.clone()],
                    aggregate_cents: t.amount_cents,
                    note: format!(
                        "single cash transaction {} over CTR threshold (${:.2})",
                        t.id,
                        (t.amount_cents as f64) / 100.0
                    ),
                });
            }
        }

        // Structuring: per-customer rolling 24h sum of cash.
        let mut by_customer: std::collections::HashMap<&str, Vec<&Transaction>> =
            std::collections::HashMap::new();
        for t in &txs {
            by_customer.entry(t.customer_id.as_str()).or_default().push(t);
        }
        for (cust, txs) in by_customer {
            let n = txs.len();
            for i in 0..n {
                let start = txs[i].at;
                let mut sum = 0u64;
                let mut ids = Vec::new();
                for t in txs.iter().skip(i) {
                    let delta = (t.at - start).num_seconds();
                    if delta > STRUCTURING_WINDOW_SECS {
                        break;
                    }
                    sum = sum.saturating_add(t.amount_cents);
                    ids.push(t.id.clone());
                }
                if sum > CTR_THRESHOLD_CENTS && ids.len() > 1 {
                    out.push(Trigger {
                        kind: TriggerKind::Structuring,
                        transactions: ids,
                        aggregate_cents: sum,
                        note: format!(
                            "{} cash transactions for customer {} aggregating ${:.2} in 24h",
                            n,
                            cust,
                            (sum as f64) / 100.0,
                        ),
                    });
                    // Only emit one structuring trigger per customer per pass.
                    break;
                }
            }
        }

        out
    }
}

/// SAR pattern helper.
#[derive(Debug, Default, Clone)]
pub struct SarHelper {
    /// Per-customer "home countries". Optional gate for geographic anomaly.
    pub customer_home_country: std::collections::HashMap<String, CountryCode>,
}

impl SarHelper {
    /// Construct.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a customer's home country for geographic-anomaly checks.
    pub fn set_home_country(&mut self, customer_id: String, country: CountryCode) {
        self.customer_home_country.insert(customer_id, country);
    }

    /// Walk `transactions` and emit triggers.
    #[must_use]
    pub fn analyze(&self, transactions: &[Transaction]) -> Vec<Trigger> {
        let mut out = Vec::new();
        let mut txs: Vec<&Transaction> = transactions.iter().collect();
        txs.sort_by_key(|t| t.at);

        // Round-dollar: amounts that are exact multiples of $1000 and >= $5000.
        for t in &txs {
            let exact_multiple = t.amount_cents % 100_000 == 0;
            let large = t.amount_cents >= 500_000;
            if exact_multiple && large {
                out.push(Trigger {
                    kind: TriggerKind::RoundDollar,
                    transactions: vec![t.id.clone()],
                    aggregate_cents: t.amount_cents,
                    note: format!(
                        "round-dollar amount ${:.2} on tx {}",
                        (t.amount_cents as f64) / 100.0,
                        t.id
                    ),
                });
            }
        }

        // Rapid in/out: inbound followed by outbound from the same customer
        // within RAPID_IN_OUT_WINDOW_SECS, with the outbound at least 80% of inbound.
        let mut by_customer: std::collections::HashMap<&str, Vec<&Transaction>> =
            std::collections::HashMap::new();
        for t in &txs {
            by_customer.entry(t.customer_id.as_str()).or_default().push(t);
        }
        for (cust, txs) in &by_customer {
            for window in txs.windows(2) {
                let (a, b) = (window[0], window[1]);
                if a.inbound
                    && !b.inbound
                    && (b.at - a.at).num_seconds() <= RAPID_IN_OUT_WINDOW_SECS
                    && b.amount_cents * 100 >= a.amount_cents * 80
                {
                    out.push(Trigger {
                        kind: TriggerKind::RapidInOut,
                        transactions: vec![a.id.clone(), b.id.clone()],
                        aggregate_cents: a.amount_cents + b.amount_cents,
                        note: format!(
                            "rapid in/out for customer {cust}: ${:.2} in, ${:.2} out within {}s",
                            (a.amount_cents as f64) / 100.0,
                            (b.amount_cents as f64) / 100.0,
                            (b.at - a.at).num_seconds(),
                        ),
                    });
                }
            }
        }

        // Geographic anomaly: foreign-country transaction when we have a
        // configured home country that disagrees.
        for t in &txs {
            let Some(home) = self.customer_home_country.get(&t.customer_id) else {
                continue;
            };
            let Some(tx_country) = t.country.as_ref() else {
                continue;
            };
            if tx_country != home {
                out.push(Trigger {
                    kind: TriggerKind::GeographicAnomaly,
                    transactions: vec![t.id.clone()],
                    aggregate_cents: t.amount_cents,
                    note: format!(
                        "tx {} in {} for customer based in {}",
                        t.id, tx_country.0, home.0
                    ),
                });
            }
        }

        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use chrono::TimeZone;

    fn tx(id: &str, cust: &str, secs: i64, amount: u64, cash: bool, inbound: bool) -> Transaction {
        Transaction {
            id: id.to_string(),
            customer_id: cust.to_string(),
            at: Utc.timestamp_opt(1_700_000_000 + secs, 0).single().expect("ts"),
            amount_cents: amount,
            currency: "USD".to_string(),
            is_cash: cash,
            inbound,
            country: None,
        }
    }

    #[test]
    fn ctr_single_threshold() {
        let txs = vec![tx("1", "c1", 0, 1_500_000, true, true)];
        let triggers = CtrHelper::new().analyze(&txs);
        assert!(triggers.iter().any(|t| t.kind == TriggerKind::CtrThreshold));
    }

    #[test]
    fn ctr_structuring_11_deposits() {
        // 11 cash deposits of $1000 each, evenly spaced over 24h, same customer.
        let mut txs = Vec::new();
        for i in 0..11 {
            txs.push(tx(
                &format!("d{i}"),
                "c1",
                i as i64 * 3600, // every hour
                100_000,         // $1000 in cents
                true,
                true,
            ));
        }
        let triggers = CtrHelper::new().analyze(&txs);
        assert!(
            triggers.iter().any(|t| t.kind == TriggerKind::Structuring),
            "expected structuring trigger, got {triggers:#?}",
        );
        let s = triggers
            .iter()
            .find(|t| t.kind == TriggerKind::Structuring)
            .expect("trigger");
        assert!(s.aggregate_cents > CTR_THRESHOLD_CENTS);
    }

    #[test]
    fn sar_round_dollar() {
        let txs = vec![tx("1", "c1", 0, 1_000_000, false, true)]; // $10,000
        let triggers = SarHelper::new().analyze(&txs);
        assert!(triggers.iter().any(|t| t.kind == TriggerKind::RoundDollar));
    }

    #[test]
    fn sar_rapid_in_out() {
        let txs = vec![
            tx("1", "c1", 0, 1_000_000, false, true),
            tx("2", "c1", 60, 950_000, false, false),
        ];
        let triggers = SarHelper::new().analyze(&txs);
        assert!(triggers.iter().any(|t| t.kind == TriggerKind::RapidInOut));
    }

    #[test]
    fn sar_geographic_anomaly() {
        let mut sar = SarHelper::new();
        sar.set_home_country("c1".to_string(), CountryCode("US".to_string()));
        let mut t = tx("1", "c1", 0, 100_000, false, true);
        t.country = Some(CountryCode("KP".to_string()));
        let triggers = sar.analyze(&[t]);
        assert!(triggers.iter().any(|t| t.kind == TriggerKind::GeographicAnomaly));
    }
}
