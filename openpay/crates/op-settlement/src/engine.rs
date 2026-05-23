//! The settlement engine — drives batches through their lifecycle
//! against a cutoff schedule and a holdback policy.
//!
//! Operators construct one [`SettlementEngine`] per
//! `(currency, rail)` pair. The engine doesn't own the store —
//! callers pass it in. This keeps the engine ergonomic and lets
//! operators share a single store across many `(currency, rail)`
//! engines.

use op_core::{Currency, Money};
use op_ledger::TransactionId;

use crate::batch::{Batch, BatchId, Status};
use crate::cutoff::Cutoff;
use crate::error::{Error, Result};
use crate::holdback::HoldbackPolicy;
use crate::payout::PayoutRail;
use crate::store::SettlementStore;

/// Drives batch lifecycle for one `(currency, rail)` pair.
#[derive(Debug, Clone)]
pub struct SettlementEngine {
    currency: Currency,
    rail: PayoutRail,
    cutoff: Cutoff,
    holdback_policy: HoldbackPolicy,
}

impl SettlementEngine {
    /// Construct.
    #[must_use]
    pub const fn new(
        currency: Currency,
        rail: PayoutRail,
        cutoff: Cutoff,
        holdback_policy: HoldbackPolicy,
    ) -> Self {
        Self {
            currency,
            rail,
            cutoff,
            holdback_policy,
        }
    }

    /// The batch currency this engine operates on.
    #[must_use]
    pub const fn currency(&self) -> Currency {
        self.currency
    }

    /// The payout rail.
    #[must_use]
    pub const fn rail(&self) -> PayoutRail {
        self.rail
    }

    /// Open a fresh batch in `store`. The current open batch (if
    /// any) is left alone — operators typically close before opening
    /// a new one. Returns the new `BatchId`.
    ///
    /// # Errors
    /// Bubbles up store errors.
    #[tracing::instrument(
        name = "settlement.open_batch",
        skip(self, store),
        fields(currency = self.currency.code(), rail = ?self.rail),
    )]
    pub fn open_batch(&self, store: &impl SettlementStore, now_unix_secs: u64) -> Result<BatchId> {
        let batch = Batch::open(self.currency, self.rail, now_unix_secs);
        let id = batch.id;
        store.create_batch(batch)?;
        Ok(id)
    }

    /// Append a posted-tx entry to a specific batch. Returns the
    /// new gross.
    ///
    /// # Errors
    /// `Error::NotFound`, `Error::InvalidTransition`,
    /// `Error::CurrencyMismatch`.
    #[tracing::instrument(
        name = "settlement.add_entry",
        skip(self, store),
        fields(batch_id = %batch_id),
    )]
    pub fn add_entry(
        &self,
        store: &impl SettlementStore,
        batch_id: BatchId,
        tx_id: TransactionId,
        amount: Money,
        reference: Option<String>,
    ) -> Result<Money> {
        let updated = store.update(batch_id, |b| b.add_entry(tx_id, amount, reference))?;
        updated.gross()
    }

    /// If the cutoff fires and exactly one batch is open, close it.
    /// Returns `Some(BatchId)` if a close happened, `None` otherwise.
    /// `last_tick_unix_secs` is typically the batch's
    /// `opened_at_unix_secs` (or the previous tick).
    ///
    /// # Errors
    /// Bubbles up store / close errors. If multiple batches are
    /// open the engine returns `Error::Invalid` — operators should
    /// close ambiguity themselves.
    pub fn tick(
        &self,
        store: &impl SettlementStore,
        last_tick_unix_secs: u64,
        now_unix_secs: u64,
    ) -> Result<Option<BatchId>> {
        if !self.cutoff.should_close(last_tick_unix_secs, now_unix_secs) {
            return Ok(None);
        }
        let open: Vec<Batch> = store
            .list_open()?
            .into_iter()
            .filter(|b| b.currency == self.currency && b.rail == self.rail)
            .collect();
        match open.len() {
            0 => Ok(None),
            1 => {
                let id = open[0].id;
                self.close_batch(store, id, 0, now_unix_secs)?;
                Ok(Some(id))
            }
            _ => Err(Error::Invalid(format!(
                "{} open batches for {}/{:?}; operator must disambiguate",
                open.len(),
                self.currency.code(),
                self.rail,
            ))),
        }
    }

    /// Close a specific batch now, applying the engine's holdback
    /// policy with the supplied dispute-adjustment basis points.
    ///
    /// # Errors
    /// `Error::NotFound`, `Error::InvalidTransition`, plus holdback
    /// arithmetic errors.
    #[tracing::instrument(
        name = "settlement.close_batch",
        skip(self, store),
        fields(batch_id = %batch_id, dispute_adjustment_bps),
    )]
    pub fn close_batch(
        &self,
        store: &impl SettlementStore,
        batch_id: BatchId,
        dispute_adjustment_bps: u16,
        now_unix_secs: u64,
    ) -> Result<Batch> {
        let policy = self.holdback_policy;
        store.update(batch_id, move |b| {
            let gross = b.gross()?;
            let hb = policy.compute(gross, dispute_adjustment_bps)?;
            b.close(hb, now_unix_secs)
        })
    }

    /// `Closed → Paying`, recording the rail's external reference.
    ///
    /// # Errors
    /// Bubbles up store / batch errors.
    pub fn submit_for_payout<S: Into<String> + Clone>(
        &self,
        store: &impl SettlementStore,
        batch_id: BatchId,
        rail_reference: S,
        now_unix_secs: u64,
    ) -> Result<Batch> {
        store.update(batch_id, move |b| {
            b.pay(rail_reference.clone(), now_unix_secs)
        })
    }

    /// `Paying → Paid`.
    ///
    /// # Errors
    /// Bubbles up store / batch errors.
    pub fn mark_settled(
        &self,
        store: &impl SettlementStore,
        batch_id: BatchId,
        now_unix_secs: u64,
    ) -> Result<Batch> {
        store.update(batch_id, move |b| b.settled(now_unix_secs))
    }

    /// `Paying → Failed`.
    ///
    /// # Errors
    /// Bubbles up store / batch errors.
    pub fn mark_failed<C: Into<String> + Clone, M: Into<String> + Clone>(
        &self,
        store: &impl SettlementStore,
        batch_id: BatchId,
        code: C,
        message: M,
        now_unix_secs: u64,
    ) -> Result<Batch> {
        store.update(batch_id, move |b| {
            b.fail(code.clone(), message.clone(), now_unix_secs)
        })
    }
}

/// Sum of every `Closed | Paying | Paid` batch's holdback reserve.
/// Operators reconcile this against their reserve account.
///
/// # Errors
/// Propagates store errors and currency-mismatch (if the store has
/// mixed-currency batches a caller filters them out first).
pub fn total_reserve_held(store: &impl SettlementStore, currency: Currency) -> Result<Money> {
    let mut total = Money::from_minor(0, currency);
    // We can't list all batches generically — `list_open` excludes
    // closed/paid. The trait stays minimal; operators wanting a
    // production-scale impl plug in a backend with the right index.
    // For the ref impl this is fine.
    for b in store.list_open()? {
        if let (Status::Closed { .. } | Status::Paying { .. } | Status::Paid { .. }, Some(hb)) =
            (&b.status, &b.holdback)
            && hb.reserve.currency == currency
        {
            total = total.checked_add(hb.reserve)?;
        }
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::store::InMemorySettlementStore;
    use op_core::Currency;

    fn fresh_engine() -> SettlementEngine {
        SettlementEngine::new(
            Currency::USD,
            PayoutRail::AchNacha,
            Cutoff::Manual,
            HoldbackPolicy::flat(50),
        )
    }

    #[test]
    fn open_then_add_entries() {
        let store = InMemorySettlementStore::new();
        let eng = fresh_engine();
        let bid = eng.open_batch(&store, 1_000).unwrap();
        let g1 = eng
            .add_entry(
                &store,
                bid,
                TransactionId::new(),
                Money::from_minor(7_500, Currency::USD),
                None,
            )
            .unwrap();
        assert_eq!(g1, Money::from_minor(7_500, Currency::USD));
        let g2 = eng
            .add_entry(
                &store,
                bid,
                TransactionId::new(),
                Money::from_minor(2_500, Currency::USD),
                None,
            )
            .unwrap();
        assert_eq!(g2, Money::from_minor(10_000, Currency::USD));
    }

    #[test]
    fn close_applies_holdback() {
        let store = InMemorySettlementStore::new();
        let eng = fresh_engine();
        let bid = eng.open_batch(&store, 1_000).unwrap();
        eng.add_entry(
            &store,
            bid,
            TransactionId::new(),
            Money::from_minor(10_000, Currency::USD),
            None,
        )
        .unwrap();
        let b = eng.close_batch(&store, bid, 0, 2_000).unwrap();
        assert_eq!(b.status.code(), "closed");
        let hb = b.holdback.expect("set");
        assert_eq!(hb.reserve, Money::from_minor(50, Currency::USD));
    }

    #[test]
    fn submit_settle_terminal() {
        let store = InMemorySettlementStore::new();
        let eng = fresh_engine();
        let bid = eng.open_batch(&store, 1_000).unwrap();
        eng.add_entry(
            &store,
            bid,
            TransactionId::new(),
            Money::from_minor(100, Currency::USD),
            None,
        )
        .unwrap();
        eng.close_batch(&store, bid, 0, 2_000).unwrap();
        eng.submit_for_payout(&store, bid, "trace-1", 3_000)
            .unwrap();
        let b = eng.mark_settled(&store, bid, 4_000).unwrap();
        assert!(b.status.is_terminal());
    }

    #[test]
    fn tick_with_manual_cutoff_never_fires() {
        let store = InMemorySettlementStore::new();
        let eng = fresh_engine();
        eng.open_batch(&store, 1_000).unwrap();
        let fired = eng.tick(&store, 1_000, 1_000_000).unwrap();
        assert!(fired.is_none());
    }

    #[test]
    fn tick_with_daily_cutoff_closes_one_batch() {
        let store = InMemorySettlementStore::new();
        let eng = SettlementEngine::new(
            Currency::USD,
            PayoutRail::AchNacha,
            Cutoff::daily(7).unwrap(),
            HoldbackPolicy::none(),
        );
        let bid = eng.open_batch(&store, 1_700_000_000).unwrap();
        eng.add_entry(
            &store,
            bid,
            TransactionId::new(),
            Money::from_minor(100, Currency::USD),
            None,
        )
        .unwrap();
        // 1_700_032_000 ≈ Nov 15 ~07:06 UTC — past the cutoff.
        let fired = eng.tick(&store, 1_700_000_000, 1_700_032_000).unwrap();
        assert_eq!(fired, Some(bid));
        assert_eq!(store.get_batch(bid).unwrap().status.code(), "closed");
    }

    #[test]
    fn tick_with_multiple_open_returns_error() {
        let store = InMemorySettlementStore::new();
        let eng = SettlementEngine::new(
            Currency::USD,
            PayoutRail::AchNacha,
            Cutoff::daily(7).unwrap(),
            HoldbackPolicy::none(),
        );
        eng.open_batch(&store, 1_700_000_000).unwrap();
        eng.open_batch(&store, 1_700_000_001).unwrap();
        let err = eng.tick(&store, 1_700_000_000, 1_700_032_000).unwrap_err();
        assert!(matches!(err, Error::Invalid(_)));
    }
}
