//! [`QuoteProvider`] trait + two ref impls: `Static` and
//! `Cached`.

use std::collections::HashMap;
use std::sync::Mutex;

use op_core::Currency;

use crate::error::{Error, Result};
use crate::quote::Quote;

/// The pluggable FX feed interface.
///
/// Operators implement this against their preferred source: an
/// HTTP rate API, a bank's tariff CSV, a fixed in-house rate
/// table, a hedged forward-rate engine. The reference stack
/// ships [`StaticQuoteProvider`] (operator-fixed) and
/// [`CachedQuoteProvider`] (TTL wrapper).
pub trait QuoteProvider: Send + Sync {
    /// Fetch a quote for `source → target` at `now_unix_secs`.
    ///
    /// # Errors
    /// [`Error::NoQuote`] if the pair isn't supported.
    fn get_quote(&self, source: Currency, target: Currency, now_unix_secs: u64) -> Result<Quote>;

    /// Provider tag for telemetry / audit. Should be stable across
    /// the process lifetime.
    fn name(&self) -> &'static str;
}

// ============================================================
// StaticQuoteProvider
// ============================================================

/// Quote source backed by a fixed `(source, target) → rate_ppm`
/// table. Useful for:
///
/// - Tests that need deterministic FX behavior.
/// - Operators with bilateral negotiated rates (no live feed
///   needed; rates change once a quarter via a redeploy).
/// - Hedged-payout flows where the operator locks rates in
///   advance.
#[derive(Default)]
pub struct StaticQuoteProvider {
    rates: HashMap<(String, String), u64>,
    validity_secs: u64,
}

impl StaticQuoteProvider {
    /// Construct with the default quote validity (`5 minutes`).
    #[must_use]
    pub fn new() -> Self {
        Self {
            rates: HashMap::new(),
            validity_secs: 300,
        }
    }

    /// Builder: override the validity window.
    #[must_use]
    pub const fn with_validity_secs(mut self, secs: u64) -> Self {
        self.validity_secs = secs;
        self
    }

    /// Builder: insert a `(source, target) → rate_ppm` entry.
    /// Returns `Self` so chains are ergonomic.
    #[must_use]
    pub fn with_rate(mut self, source: Currency, target: Currency, rate_ppm: u64) -> Self {
        self.rates.insert(
            (source.code().to_owned(), target.code().to_owned()),
            rate_ppm,
        );
        self
    }
}

impl QuoteProvider for StaticQuoteProvider {
    fn get_quote(&self, source: Currency, target: Currency, now_unix_secs: u64) -> Result<Quote> {
        let key = (source.code().to_owned(), target.code().to_owned());
        let rate_ppm = self
            .rates
            .get(&key)
            .copied()
            .ok_or_else(|| Error::NoQuote {
                from_currency: source.code().to_owned(),
                to_currency: target.code().to_owned(),
            })?;
        Ok(Quote::new(
            source,
            target,
            rate_ppm,
            now_unix_secs,
            now_unix_secs.saturating_add(self.validity_secs),
            "static",
        ))
    }

    fn name(&self) -> &'static str {
        "static"
    }
}

// ============================================================
// CachedQuoteProvider
// ============================================================

/// Wraps another provider with a TTL cache. The first request for
/// a given pair hits the inner provider; subsequent requests
/// inside the TTL return the cached quote without going through.
///
/// The cache key is `(source, target)`; the TTL is the duration
/// from when the cached quote was fetched, not from when it
/// would expire on the wire.
pub struct CachedQuoteProvider<P: QuoteProvider> {
    inner: P,
    ttl_secs: u64,
    cache: Mutex<HashMap<(String, String), CachedEntry>>,
}

struct CachedEntry {
    quote: Quote,
    cached_at_unix_secs: u64,
}

impl<P: QuoteProvider> CachedQuoteProvider<P> {
    /// Construct with a TTL in seconds.
    #[must_use]
    pub fn new(inner: P, ttl_secs: u64) -> Self {
        Self {
            inner,
            ttl_secs,
            cache: Mutex::new(HashMap::new()),
        }
    }

    /// Diagnostic: cache size.
    ///
    /// # Panics
    /// Only if the internal cache mutex was previously poisoned by
    /// a thread that panicked while holding it.
    pub fn cache_size(&self) -> usize {
        self.cache.lock().expect("poisoned").len()
    }
}

impl<P: QuoteProvider> QuoteProvider for CachedQuoteProvider<P> {
    fn get_quote(&self, source: Currency, target: Currency, now_unix_secs: u64) -> Result<Quote> {
        let key = (source.code().to_owned(), target.code().to_owned());
        {
            let cache = self.cache.lock().expect("poisoned");
            if let Some(entry) = cache.get(&key)
                && now_unix_secs.saturating_sub(entry.cached_at_unix_secs) < self.ttl_secs
                && entry.quote.is_valid_at(now_unix_secs)
            {
                return Ok(entry.quote.clone());
            }
        }
        let q = self.inner.get_quote(source, target, now_unix_secs)?;
        self.cache.lock().expect("poisoned").insert(
            key,
            CachedEntry {
                quote: q.clone(),
                cached_at_unix_secs: now_unix_secs,
            },
        );
        Ok(q)
    }

    fn name(&self) -> &'static str {
        "cached"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn static_provider_returns_configured_rate() {
        let p = StaticQuoteProvider::new().with_rate(Currency::EUR, Currency::USD, 1_082_500);
        let q = p.get_quote(Currency::EUR, Currency::USD, 1_000).unwrap();
        assert_eq!(q.rate_ppm, 1_082_500);
        assert_eq!(q.source_name, "static");
        assert_eq!(q.fetched_at_unix_secs, 1_000);
        assert_eq!(q.valid_until_unix_secs, 1_300);
    }

    #[test]
    fn static_provider_no_quote_for_unconfigured_pair() {
        let p = StaticQuoteProvider::new();
        assert!(matches!(
            p.get_quote(Currency::EUR, Currency::USD, 0),
            Err(Error::NoQuote { .. })
        ));
    }

    #[test]
    fn cached_provider_serves_from_cache_within_ttl() {
        struct CountingProvider {
            count: Mutex<u32>,
        }
        impl QuoteProvider for CountingProvider {
            fn get_quote(&self, source: Currency, target: Currency, now: u64) -> Result<Quote> {
                *self.count.lock().unwrap() += 1;
                Ok(Quote::new(
                    source,
                    target,
                    1_000_000,
                    now,
                    now + 1000,
                    "test",
                ))
            }
            fn name(&self) -> &'static str {
                "counting"
            }
        }
        let inner = CountingProvider {
            count: Mutex::new(0),
        };
        let p = CachedQuoteProvider::new(inner, 60);
        let _ = p.get_quote(Currency::USD, Currency::EUR, 100).unwrap();
        let _ = p.get_quote(Currency::USD, Currency::EUR, 110).unwrap();
        let _ = p.get_quote(Currency::USD, Currency::EUR, 159).unwrap();
        assert_eq!(*p.inner.count.lock().unwrap(), 1);
        // After TTL.
        let _ = p.get_quote(Currency::USD, Currency::EUR, 200).unwrap();
        assert_eq!(*p.inner.count.lock().unwrap(), 2);
    }

    #[test]
    fn cached_provider_bypasses_cache_after_quote_expires() {
        struct ShortLivedProvider;
        impl QuoteProvider for ShortLivedProvider {
            fn get_quote(&self, source: Currency, target: Currency, now: u64) -> Result<Quote> {
                Ok(Quote::new(source, target, 1_000_000, now, now + 5, "short"))
            }
            fn name(&self) -> &'static str {
                "short"
            }
        }
        let p = CachedQuoteProvider::new(ShortLivedProvider, 9_999);
        let q1 = p.get_quote(Currency::USD, Currency::EUR, 100).unwrap();
        assert_eq!(q1.valid_until_unix_secs, 105);
        // 200 is past the inner quote's validity but well inside the cache TTL —
        // cache must refetch rather than serve a stale quote.
        let q2 = p.get_quote(Currency::USD, Currency::EUR, 200).unwrap();
        assert_eq!(q2.valid_until_unix_secs, 205);
    }
}
