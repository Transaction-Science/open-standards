//! Reference [`ReconciliationSource`](crate::ReconciliationSource)
//! implementations and the conversions they share.
//!
//! Two ship in v1:
//!
//! - [`Camt053Source`] — ISO 20022 end-of-day bank statement.
//! - [`WebhookEventSource`] — settlement webhooks from `op-webhook`.
//!
//! `camt.054` (intra-day notification) has the identical `Ntry`
//! shape; a `Camt054Source` is a deliberate follow-on — the
//! `ReconciliationSource` trait exists precisely so it drops in
//! without touching the engine.

mod camt053;
mod camt054;
mod webhook;

pub use camt053::Camt053Source;
pub use camt054::Camt054Source;
pub use webhook::{SETTLEMENT_EVENT_TYPE, WebhookEventSource};

use op_core::{Currency, Money};

use crate::error::{Error, Result};
use crate::statement::LineDirection;

/// Map an ISO 4217 alphabetic code to an [`op_core::Currency`].
///
/// The seven curated constants carry the correct ISO exponent
/// (notably JPY = 0). Anything else falls back to `try_new(code, 2)`:
/// 2 dp is by far the most common ISO 4217 exponent, but it is a
/// **documented assumption** — a source carrying a 0- or 3-dp exotic
/// currency would be mis-scaled. Operators handling those ship a
/// source that constructs `Money` with the right exponent directly.
pub(crate) fn currency_from_code(code: &str) -> Result<Currency> {
    match code {
        "USD" => Ok(Currency::USD),
        "EUR" => Ok(Currency::EUR),
        "BRL" => Ok(Currency::BRL),
        "INR" => Ok(Currency::INR),
        "GBP" => Ok(Currency::GBP),
        "JPY" => Ok(Currency::JPY),
        "CNY" => Ok(Currency::CNY),
        other => {
            let bytes: [u8; 3] = other.as_bytes().try_into().map_err(|_| {
                Error::MalformedLine(format!("currency code {other:?} not 3 chars"))
            })?;
            Currency::try_new(bytes, 2)
                .map_err(|_| Error::MalformedLine(format!("invalid currency code {other:?}")))
        }
    }
}

/// Convert a wire amount (`f64`) plus its currency into exact minor
/// units. CAMT amounts are bounded (max 18 digits) so the `f64`
/// multiply-and-round is exact for every realistic value; the round
/// defends against the classic `0.1`-style binary representation.
pub(crate) fn to_money(value: f64, currency: Currency) -> Money {
    let scale = 10f64.powi(i32::from(currency.exponent()));
    #[allow(clippy::cast_possible_truncation)]
    let minor = (value * scale).round() as i64;
    Money {
        minor_units: minor.abs(),
        currency,
    }
}

/// Parse a CAMT date — either `YYYY-MM-DD` (`Dt`) or an RFC 3339
/// dateTime (`DtTm`) — into unix epoch seconds. A bare date is
/// anchored at 00:00:00 UTC.
pub(crate) fn unix_from_camt_date(s: &str) -> Result<u64> {
    use time::format_description::well_known::Rfc3339;
    use time::{Date, OffsetDateTime};

    if let Ok(dt) = OffsetDateTime::parse(s, &Rfc3339) {
        let ts = dt.unix_timestamp();
        return u64::try_from(ts)
            .map_err(|_| Error::MalformedLine(format!("pre-epoch datetime {s:?}")));
    }
    // `Dt` form: YYYY-MM-DD.
    let fmt = time::macros::format_description!("[year]-[month]-[day]");
    let date = Date::parse(s, &fmt)
        .map_err(|_| Error::MalformedLine(format!("unparseable date {s:?}")))?;
    let midnight = date.with_hms(0, 0, 0).expect("00:00:00 is always valid");
    let ts = midnight.assume_utc().unix_timestamp();
    u64::try_from(ts).map_err(|_| Error::MalformedLine(format!("pre-epoch date {s:?}")))
}

/// Map the CAMT credit/debit flag to our statement-direction enum.
pub(crate) fn direction(is_credit: bool) -> LineDirection {
    if is_credit {
        LineDirection::Credit
    } else {
        LineDirection::Debit
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn currency_curated_and_fallback() {
        assert_eq!(currency_from_code("JPY").unwrap(), Currency::JPY);
        // Unknown but well-formed → 2dp fallback.
        let aud = currency_from_code("AUD").unwrap();
        assert_eq!(aud.code(), "AUD");
        assert_eq!(aud.exponent(), 2);
        assert!(currency_from_code("US").is_err());
    }

    #[test]
    fn money_scaling_respects_exponent() {
        // 12.34 USD → 1234 cents.
        assert_eq!(to_money(12.34, Currency::USD).minor_units, 1234);
        // 5000 JPY (0 dp) → 5000 minor units, not 500000.
        assert_eq!(to_money(5000.0, Currency::JPY).minor_units, 5000);
        // Sign is dropped (carried by direction).
        assert_eq!(to_money(-9.99, Currency::EUR).minor_units, 999);
    }

    #[test]
    fn camt_date_forms() {
        // Bare date → UTC midnight. 2021-01-01 = 1_609_459_200.
        assert_eq!(unix_from_camt_date("2021-01-01").unwrap(), 1_609_459_200);
        // RFC3339 dateTime.
        assert_eq!(
            unix_from_camt_date("2021-01-01T00:00:00Z").unwrap(),
            1_609_459_200
        );
        assert!(unix_from_camt_date("not-a-date").is_err());
    }
}
