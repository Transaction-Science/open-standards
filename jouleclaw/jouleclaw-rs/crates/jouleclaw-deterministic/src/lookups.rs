//! Static-lookup primitives. ISO-3166-1 alpha-2 country codes and
//! ISO-4217 currency codes — both as a small representative table of
//! ~20 entries. The tables are intentionally compact: the doctrine is
//! LUT-over-compute, and consumers who need the full ISO set can plug
//! their own primitive over the same trait.

use jouleclaw_cascade::LawfulPrimitive;
use std::sync::Arc;

pub fn primitives() -> Vec<Arc<dyn LawfulPrimitive>> {
    vec![
        Arc::new(CountryCodeToName),
        Arc::new(CurrencyCodeToName),
        Arc::new(IsoCurrencySymbol),
    ]
}

fn strip_prefix_ci<'a>(q: &'a str, prefix: &str) -> Option<&'a str> {
    let q = q.trim();
    if q.len() < prefix.len() {
        return None;
    }
    let (head, tail) = q.split_at(prefix.len());
    if !head.eq_ignore_ascii_case(prefix) {
        return None;
    }
    let rest = tail.strip_prefix(|c: char| c.is_whitespace())?;
    Some(rest.trim())
}

// ---- countries ----------------------------------------------------------

const COUNTRY_TABLE: &[(&str, &str)] = &[
    ("US", "United States"),
    ("CA", "Canada"),
    ("MX", "Mexico"),
    ("GB", "United Kingdom"),
    ("FR", "France"),
    ("DE", "Germany"),
    ("ES", "Spain"),
    ("IT", "Italy"),
    ("NL", "Netherlands"),
    ("SE", "Sweden"),
    ("NO", "Norway"),
    ("JP", "Japan"),
    ("CN", "China"),
    ("IN", "India"),
    ("KR", "South Korea"),
    ("AU", "Australia"),
    ("NZ", "New Zealand"),
    ("BR", "Brazil"),
    ("AR", "Argentina"),
    ("ZA", "South Africa"),
];

pub struct CountryCodeToName;
impl LawfulPrimitive for CountryCodeToName {
    fn id(&self) -> &str {
        "lawful:lookups:country-name"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_prefix_ci(query, "country")?;
        if rest.is_empty() {
            return None;
        }
        let code = rest.to_ascii_uppercase();
        COUNTRY_TABLE
            .iter()
            .find(|(c, _)| *c == code.as_str())
            .map(|(_, name)| (*name).to_string())
    }
    fn declared_cost_uj(&self) -> u64 {
        90
    }
}

// ---- currencies ---------------------------------------------------------

/// (ISO 4217 alpha code, full name, symbol)
const CURRENCY_TABLE: &[(&str, &str, &str)] = &[
    ("USD", "United States Dollar", "$"),
    ("EUR", "Euro", "\u{20ac}"),
    ("GBP", "Pound Sterling", "\u{a3}"),
    ("JPY", "Japanese Yen", "\u{a5}"),
    ("CNY", "Chinese Yuan", "\u{a5}"),
    ("KRW", "South Korean Won", "\u{20a9}"),
    ("INR", "Indian Rupee", "\u{20b9}"),
    ("CAD", "Canadian Dollar", "$"),
    ("AUD", "Australian Dollar", "$"),
    ("NZD", "New Zealand Dollar", "$"),
    ("CHF", "Swiss Franc", "Fr"),
    ("SEK", "Swedish Krona", "kr"),
    ("NOK", "Norwegian Krone", "kr"),
    ("DKK", "Danish Krone", "kr"),
    ("MXN", "Mexican Peso", "$"),
    ("BRL", "Brazilian Real", "R$"),
    ("ARS", "Argentine Peso", "$"),
    ("ZAR", "South African Rand", "R"),
    ("RUB", "Russian Ruble", "\u{20bd}"),
    ("TRY", "Turkish Lira", "\u{20ba}"),
];

pub struct CurrencyCodeToName;
impl LawfulPrimitive for CurrencyCodeToName {
    fn id(&self) -> &str {
        "lawful:lookups:currency-name"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_prefix_ci(query, "currency")?;
        if rest.is_empty() {
            return None;
        }
        // Guard against "currency symbol …" — that's the other primitive.
        if rest.to_ascii_lowercase().starts_with("symbol") {
            return None;
        }
        let code = rest.to_ascii_uppercase();
        CURRENCY_TABLE
            .iter()
            .find(|(c, _, _)| *c == code.as_str())
            .map(|(_, name, _)| (*name).to_string())
    }
    fn declared_cost_uj(&self) -> u64 {
        90
    }
}

pub struct IsoCurrencySymbol;
impl LawfulPrimitive for IsoCurrencySymbol {
    fn id(&self) -> &str {
        "lawful:lookups:currency-symbol"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_prefix_ci(query, "currency symbol")?;
        if rest.is_empty() {
            return None;
        }
        let code = rest.to_ascii_uppercase();
        CURRENCY_TABLE
            .iter()
            .find(|(c, _, _)| *c == code.as_str())
            .map(|(_, _, sym)| (*sym).to_string())
    }
    fn declared_cost_uj(&self) -> u64 {
        90
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn country_lookup_works() {
        assert_eq!(
            CountryCodeToName.try_resolve("country US").as_deref(),
            Some("United States")
        );
        assert_eq!(
            CountryCodeToName.try_resolve("country gb").as_deref(),
            Some("United Kingdom")
        );
    }

    #[test]
    fn currency_lookup_works() {
        assert_eq!(
            CurrencyCodeToName.try_resolve("currency USD").as_deref(),
            Some("United States Dollar")
        );
        assert_eq!(
            CurrencyCodeToName.try_resolve("currency eur").as_deref(),
            Some("Euro")
        );
    }

    #[test]
    fn currency_symbol_works() {
        assert_eq!(IsoCurrencySymbol.try_resolve("currency symbol USD").as_deref(), Some("$"));
        assert_eq!(IsoCurrencySymbol.try_resolve("currency symbol EUR").as_deref(), Some("\u{20ac}"));
        assert_eq!(IsoCurrencySymbol.try_resolve("currency symbol JPY").as_deref(), Some("\u{a5}"));
    }

    #[test]
    fn currency_does_not_swallow_symbol_queries() {
        // "currency symbol USD" should NOT be answered by CurrencyCodeToName.
        assert!(CurrencyCodeToName.try_resolve("currency symbol USD").is_none());
    }

    #[test]
    fn malformed_returns_none() {
        assert!(CountryCodeToName.try_resolve("country").is_none());
        assert!(CountryCodeToName.try_resolve("country XX").is_none());
        assert!(CurrencyCodeToName.try_resolve("currency XYZ").is_none());
        assert!(IsoCurrencySymbol.try_resolve("currency symbol ZZZ").is_none());
    }

    #[test]
    fn category_count() {
        assert_eq!(primitives().len(), 3);
    }
}
