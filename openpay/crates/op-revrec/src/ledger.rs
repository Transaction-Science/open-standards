//! Deferred-revenue ledger.
//!
//! A revenue-recognition engine without a ledger is just a calculator.
//! The ledger:
//!
//! 1. Records the *contract-inception* posting: cash (or receivable)
//!    debited, deferred revenue credited for the transaction-price
//!    total.
//! 2. As each [`crate::schedule::ScheduleEntry`] comes due, posts a
//!    recognition entry: deferred revenue debited, recognized revenue
//!    credited.
//! 3. On refund / reversal, debits recognized (or deferred, depending
//!    on whether the refund period has been recognized yet) and credits
//!    cash / refund-liability.
//!
//! The trait is async because real GL backends (NetSuite, Workday
//! Financials, QuickBooks Online) are network APIs.
//!
//! [`InMemoryLedger`] is the reference implementation used by the
//! example tests; it is also fine for low-volume operators (it
//! serialises to JSON on demand for backup).
//!
//! ## Tie-in with `op-core::Payment`
//!
//! The ledger surface accepts a `payment_id: Uuid` on every posting so
//! that the caller can correlate to the `Payment<S>` typestate machine
//! in `op-core`. When a `Payment<Captured>` becomes `Payment<Refunded>`
//! via [`op_core::Payment::refund`], the caller is expected to mirror
//! the refund into the ledger by calling [`DeferredRevenueLedger::post_refund`].

use async_trait::async_trait;
use chrono::{DateTime, NaiveDate, Utc};
use op_core::{Currency, Money};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Mutex;
use uuid::Uuid;

use crate::contract::ContractId;
use crate::error::{Error, Result};
use crate::schedule::ScheduleEntry;

/// The kind of posting on the deferred-revenue subledger.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PostingKind {
    /// Initial credit to deferred revenue at contract inception.
    DeferralOpen,
    /// Recognition: debit deferred, credit revenue.
    Recognize,
    /// Refund: debit revenue (if recognized) or debit deferred (if not),
    /// credit cash / refund-liability.
    Refund,
}

/// One immutable posting on the ledger. The ledger is append-only.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Posting {
    /// Monotonic id (UUID v7).
    pub id: Uuid,
    /// Which contract this posting belongs to.
    pub contract_id: ContractId,
    /// Optional correlation to an `op-core::Payment` id.
    pub payment_id: Option<Uuid>,
    /// What kind of posting.
    pub kind: PostingKind,
    /// Amount in minor units. Positive in all kinds; the `kind`
    /// dictates which side of the deferred / recognized accounts moves.
    pub amount_minor: i64,
    /// Currency.
    pub currency: Currency,
    /// Posting date (the recognition date for `Recognize`).
    pub date: NaiveDate,
    /// When the posting was created (audit).
    pub created_at: DateTime<Utc>,
    /// Free-text memo (auditor visible).
    pub memo: String,
}

/// Aggregated balances for one contract.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Balances {
    /// Sum of `DeferralOpen` minus sum of `Recognize` minus sum of
    /// `Refund` (deferred portion). Always non-negative on a healthy
    /// contract.
    pub deferred_minor: i64,
    /// Sum of `Recognize` minus sum of `Refund` (recognized portion).
    /// Refund handling first eats from recognized.
    pub recognized_minor: i64,
}

/// Pluggable deferred-revenue ledger.
///
/// All methods are async to admit network-backed implementations.
#[async_trait]
pub trait DeferredRevenueLedger: Send + Sync {
    /// Open the deferral for a contract.
    ///
    /// # Errors
    /// Implementation-defined.
    async fn open_deferral(
        &self,
        contract_id: ContractId,
        payment_id: Option<Uuid>,
        amount: Money,
        date: NaiveDate,
        memo: &str,
    ) -> Result<Posting>;

    /// Post a recognition from a schedule entry.
    ///
    /// # Errors
    /// Implementation-defined.
    async fn post_recognition(
        &self,
        contract_id: ContractId,
        entry: &ScheduleEntry,
        memo: &str,
    ) -> Result<Posting>;

    /// Post a refund. The implementation must reduce recognized first
    /// (refund of already-booked revenue) and then deferred.
    ///
    /// # Errors
    /// - [`Error::RefundExceedsRecognized`] when the requested refund
    ///   exceeds `recognized + deferred`.
    async fn post_refund(
        &self,
        contract_id: ContractId,
        payment_id: Option<Uuid>,
        amount: Money,
        date: NaiveDate,
        memo: &str,
    ) -> Result<Posting>;

    /// Snapshot of `(deferred, recognized)` balances for a contract.
    ///
    /// # Errors
    /// Implementation-defined.
    async fn balances(&self, contract_id: ContractId) -> Result<Balances>;

    /// All postings for a contract, in insertion order.
    ///
    /// # Errors
    /// Implementation-defined.
    async fn postings(&self, contract_id: ContractId) -> Result<Vec<Posting>>;

    /// Stable name for telemetry (`"in-memory"`, `"netsuite"`, etc.).
    fn name(&self) -> &'static str;
}

/// Reference in-memory ledger. Thread-safe via a single `Mutex`; fine
/// for low-volume operators and exhaustive for unit / integration tests.
#[derive(Default)]
pub struct InMemoryLedger {
    inner: Mutex<HashMap<ContractId, Vec<Posting>>>,
}

impl InMemoryLedger {
    /// Construct an empty ledger.
    #[must_use]
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(HashMap::new()),
        }
    }

    fn compute_balances(postings: &[Posting]) -> Balances {
        let mut deferred: i64 = 0;
        let mut recognized: i64 = 0;
        for p in postings {
            match p.kind {
                PostingKind::DeferralOpen => deferred = deferred.saturating_add(p.amount_minor),
                PostingKind::Recognize => {
                    deferred = deferred.saturating_sub(p.amount_minor);
                    recognized = recognized.saturating_add(p.amount_minor);
                }
                PostingKind::Refund => {
                    // Reduce recognized first, then deferred.
                    let from_recognized = recognized.min(p.amount_minor);
                    recognized = recognized.saturating_sub(from_recognized);
                    let from_deferred = p.amount_minor - from_recognized;
                    deferred = deferred.saturating_sub(from_deferred);
                }
            }
        }
        Balances {
            deferred_minor: deferred,
            recognized_minor: recognized,
        }
    }
}

#[async_trait]
impl DeferredRevenueLedger for InMemoryLedger {
    async fn open_deferral(
        &self,
        contract_id: ContractId,
        payment_id: Option<Uuid>,
        amount: Money,
        date: NaiveDate,
        memo: &str,
    ) -> Result<Posting> {
        let p = Posting {
            id: Uuid::now_v7(),
            contract_id,
            payment_id,
            kind: PostingKind::DeferralOpen,
            amount_minor: amount.minor_units,
            currency: amount.currency,
            date,
            created_at: Utc::now(),
            memo: memo.to_owned(),
        };
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| Error::Invariant(format!("ledger lock poisoned: {e}")))?;
        guard.entry(contract_id).or_default().push(p.clone());
        Ok(p)
    }

    async fn post_recognition(
        &self,
        contract_id: ContractId,
        entry: &ScheduleEntry,
        memo: &str,
    ) -> Result<Posting> {
        let p = Posting {
            id: Uuid::now_v7(),
            contract_id,
            payment_id: None,
            kind: PostingKind::Recognize,
            amount_minor: entry.amount_minor,
            currency: entry.currency,
            date: entry.date,
            created_at: Utc::now(),
            memo: memo.to_owned(),
        };
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| Error::Invariant(format!("ledger lock poisoned: {e}")))?;
        guard.entry(contract_id).or_default().push(p.clone());
        Ok(p)
    }

    async fn post_refund(
        &self,
        contract_id: ContractId,
        payment_id: Option<Uuid>,
        amount: Money,
        date: NaiveDate,
        memo: &str,
    ) -> Result<Posting> {
        let mut guard = self
            .inner
            .lock()
            .map_err(|e| Error::Invariant(format!("ledger lock poisoned: {e}")))?;
        let postings = guard.entry(contract_id).or_default();
        let bal = Self::compute_balances(postings);
        let total_available = bal
            .recognized_minor
            .checked_add(bal.deferred_minor)
            .unwrap_or(i64::MAX);
        if amount.minor_units > total_available {
            return Err(Error::RefundExceedsRecognized {
                refund: amount.minor_units,
                recognized: bal.recognized_minor,
            });
        }
        let p = Posting {
            id: Uuid::now_v7(),
            contract_id,
            payment_id,
            kind: PostingKind::Refund,
            amount_minor: amount.minor_units,
            currency: amount.currency,
            date,
            created_at: Utc::now(),
            memo: memo.to_owned(),
        };
        postings.push(p.clone());
        Ok(p)
    }

    async fn balances(&self, contract_id: ContractId) -> Result<Balances> {
        let guard = self
            .inner
            .lock()
            .map_err(|e| Error::Invariant(format!("ledger lock poisoned: {e}")))?;
        let v = guard.get(&contract_id).cloned().unwrap_or_default();
        Ok(Self::compute_balances(&v))
    }

    async fn postings(&self, contract_id: ContractId) -> Result<Vec<Posting>> {
        let guard = self
            .inner
            .lock()
            .map_err(|e| Error::Invariant(format!("ledger lock poisoned: {e}")))?;
        Ok(guard.get(&contract_id).cloned().unwrap_or_default())
    }

    fn name(&self) -> &'static str {
        "in-memory"
    }
}

/// Multi-currency translation at the recognition date.
///
/// Given a recognition entry in the contract currency and an FX rate
/// to the reporting currency on the recognition date, produces the
/// translated entry. The original `entry` is preserved unchanged; the
/// translation lives alongside it (caller persists both).
///
/// # Errors
/// - [`Error::Money`] on overflow.
pub fn translate(
    entry: &ScheduleEntry,
    to_currency: Currency,
    rate: rust_decimal::Decimal,
) -> Result<ScheduleEntry> {
    use rust_decimal::prelude::ToPrimitive;
    let minor = rust_decimal::Decimal::from(entry.amount_minor) * rate;
    let i = minor
        .round_dp(0)
        .to_i64()
        .ok_or_else(|| Error::Money(op_core::Error::Overflow))?;
    Ok(ScheduleEntry {
        obligation_id: entry.obligation_id.clone(),
        date: entry.date,
        amount_minor: i,
        currency: to_currency,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::contract::ObligationId;

    fn ymd(y: i32, m: u32, d: u32) -> NaiveDate {
        NaiveDate::from_ymd_opt(y, m, d).unwrap_or_default()
    }

    #[tokio::test]
    async fn open_then_recognize_drains_deferred() {
        let l = InMemoryLedger::new();
        let cid = ContractId::new();
        l.open_deferral(
            cid,
            None,
            Money::from_minor(12_000, Currency::USD),
            ymd(2026, 1, 1),
            "open",
        )
        .await
        .expect("open");
        let e = ScheduleEntry {
            obligation_id: ObligationId::new("saas"),
            date: ymd(2026, 1, 31),
            amount_minor: 1_000,
            currency: Currency::USD,
        };
        l.post_recognition(cid, &e, "jan").await.expect("rec");
        let b = l.balances(cid).await.expect("bal");
        assert_eq!(b.deferred_minor, 11_000);
        assert_eq!(b.recognized_minor, 1_000);
    }

    #[tokio::test]
    async fn refund_reduces_recognized_first() {
        let l = InMemoryLedger::new();
        let cid = ContractId::new();
        l.open_deferral(
            cid,
            None,
            Money::from_minor(12_000, Currency::USD),
            ymd(2026, 1, 1),
            "open",
        )
        .await
        .expect("open");
        let e = ScheduleEntry {
            obligation_id: ObligationId::new("saas"),
            date: ymd(2026, 1, 31),
            amount_minor: 3_000,
            currency: Currency::USD,
        };
        l.post_recognition(cid, &e, "jan").await.expect("rec");
        l.post_refund(
            cid,
            None,
            Money::from_minor(2_000, Currency::USD),
            ymd(2026, 2, 1),
            "partial refund",
        )
        .await
        .expect("refund");
        let b = l.balances(cid).await.expect("bal");
        assert_eq!(b.recognized_minor, 1_000);
        assert_eq!(b.deferred_minor, 9_000);
    }

    #[tokio::test]
    async fn refund_exceeding_total_rejected() {
        let l = InMemoryLedger::new();
        let cid = ContractId::new();
        l.open_deferral(
            cid,
            None,
            Money::from_minor(1_000, Currency::USD),
            ymd(2026, 1, 1),
            "open",
        )
        .await
        .expect("open");
        let res = l
            .post_refund(
                cid,
                None,
                Money::from_minor(2_000, Currency::USD),
                ymd(2026, 2, 1),
                "too big",
            )
            .await;
        assert!(matches!(res, Err(Error::RefundExceedsRecognized { .. })));
    }
}
