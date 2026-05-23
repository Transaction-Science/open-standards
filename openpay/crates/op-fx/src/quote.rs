//! FX quote: a snapshot of a rate at a moment in time.

use op_core::Currency;
use serde::{Deserialize, Serialize};

/// One source→target rate snapshot.
///
/// `rate_ppm` carries six decimal places of precision. A
/// `1.082_500` EUR/USD spot rate is `1_082_500`. Operators
/// modeling spread above mid-market multiply through before
/// constructing the quote — the crate doesn't care whether
/// `rate_ppm` is mid, bid, ask, or a bank's tariff.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Quote {
    /// Source currency.
    pub source_currency: Currency,
    /// Target currency.
    pub target_currency: Currency,
    /// Rate in parts per million. `1.000000 = 1_000_000`.
    pub rate_ppm: u64,
    /// When the provider fetched this quote (unix epoch seconds).
    pub fetched_at_unix_secs: u64,
    /// When the quote stops being usable (unix epoch seconds).
    /// Callers check this against `now` before converting.
    pub valid_until_unix_secs: u64,
    /// Free-form provider tag (`"wise"`, `"oer"`, `"bank-tariff"`,
    /// ...). Operators use it for audit / reconciliation.
    pub source_name: String,
}

impl Quote {
    /// Construct.
    #[must_use]
    pub fn new(
        source_currency: Currency,
        target_currency: Currency,
        rate_ppm: u64,
        fetched_at_unix_secs: u64,
        valid_until_unix_secs: u64,
        source_name: impl Into<String>,
    ) -> Self {
        Self {
            source_currency,
            target_currency,
            rate_ppm,
            fetched_at_unix_secs,
            valid_until_unix_secs,
            source_name: source_name.into(),
        }
    }

    /// True iff this quote is still valid at `now_unix_secs`.
    #[must_use]
    pub const fn is_valid_at(&self, now_unix_secs: u64) -> bool {
        now_unix_secs <= self.valid_until_unix_secs
    }

    /// Construct an "inverse" quote (target → source) by
    /// inverting the rate. Useful when the operator only has a
    /// USD/EUR feed but needs EUR/USD for a particular payout.
    ///
    /// Returns `None` if the rate is zero (which would imply
    /// division-by-zero on inversion).
    #[must_use]
    pub fn inverse(&self) -> Option<Self> {
        if self.rate_ppm == 0 {
            return None;
        }
        // (1 / rate) in ppm = 1_000_000^2 / rate_ppm.
        let inv = (1_000_000_u128 * 1_000_000) / u128::from(self.rate_ppm);
        Some(Self {
            source_currency: self.target_currency,
            target_currency: self.source_currency,
            rate_ppm: u64::try_from(inv).unwrap_or(u64::MAX),
            fetched_at_unix_secs: self.fetched_at_unix_secs,
            valid_until_unix_secs: self.valid_until_unix_secs,
            source_name: self.source_name.clone(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn valid_window_check() {
        let q = Quote::new(Currency::USD, Currency::EUR, 850_000, 1_000, 2_000, "test");
        assert!(q.is_valid_at(1_000));
        assert!(q.is_valid_at(2_000));
        assert!(!q.is_valid_at(2_001));
    }

    #[test]
    fn inverse_round_trips() {
        // EUR/USD = 1.082500 → USD/EUR ≈ 0.923787
        let q = Quote::new(Currency::EUR, Currency::USD, 1_082_500, 0, 10_000, "test");
        let inv = q.inverse().unwrap();
        assert_eq!(inv.source_currency, Currency::USD);
        assert_eq!(inv.target_currency, Currency::EUR);
        // (1e6 * 1e6) / 1_082_500 = 923_787 (floor)
        assert_eq!(inv.rate_ppm, 923_787);
    }

    #[test]
    fn inverse_zero_rate_returns_none() {
        let q = Quote::new(Currency::USD, Currency::EUR, 0, 0, 10_000, "test");
        assert!(q.inverse().is_none());
    }
}
