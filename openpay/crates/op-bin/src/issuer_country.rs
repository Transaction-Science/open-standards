//! ISO 3166-1 alpha-2 country-code wrapper.
//!
//! BIN ranges optionally annotate the issuer's home country.
//! Many ranges (multi-national issuers, co-branded portfolios)
//! are intentionally left unannotated; downstream callers should
//! treat `None` as "unknown / multi-country" rather than as an
//! error.

use core::fmt;

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Two-letter ASCII-uppercase country code per ISO 3166-1
/// alpha-2 (e.g. `"US"`, `"GB"`, `"JP"`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct IssuerCountry([u8; 2]);

impl IssuerCountry {
    /// Parse a two-character ASCII-uppercase code.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidCountryCode`] for any string not matching
    ///   `[A-Z]{2}`.
    pub fn parse(s: &str) -> Result<Self> {
        let bytes = s.as_bytes();
        if bytes.len() != 2 || !bytes.iter().all(|b| b.is_ascii_uppercase()) {
            return Err(Error::InvalidCountryCode(s.to_string()));
        }
        Ok(Self([bytes[0], bytes[1]]))
    }

    /// Construct from a literal at compile time. Panics if the
    /// input is not exactly two ASCII-uppercase bytes — intended
    /// for `const` initializers in static tables.
    pub const fn from_ascii(a: u8, b: u8) -> Self {
        assert!(
            a.is_ascii_uppercase() && b.is_ascii_uppercase(),
            "country code must be two ASCII-uppercase letters",
        );
        Self([a, b])
    }

    /// Wire string.
    pub fn as_str(&self) -> &str {
        // SAFETY-free path: bytes are validated ASCII at construction.
        core::str::from_utf8(&self.0).unwrap_or("??")
    }
}

impl fmt::Display for IssuerCountry {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_us() {
        let c = IssuerCountry::parse("US").expect("valid");
        assert_eq!(c.as_str(), "US");
    }

    #[test]
    fn parse_lowercase_rejected() {
        assert!(matches!(
            IssuerCountry::parse("us"),
            Err(Error::InvalidCountryCode(_))
        ));
    }

    #[test]
    fn parse_long_rejected() {
        assert!(matches!(
            IssuerCountry::parse("USA"),
            Err(Error::InvalidCountryCode(_))
        ));
    }

    #[test]
    fn const_ascii() {
        let c = IssuerCountry::from_ascii(b'U', b'S');
        assert_eq!(c.as_str(), "US");
    }
}
