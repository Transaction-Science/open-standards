//! Money: ISO 4217 currency + exact minor-unit integer amount.
//!
//! ## Why integers
//!
//! Floating point cannot represent `0.10` exactly. A payment system that
//! ever sees `f64` for amounts is wrong. We model money as `i64` minor units
//! (cents for USD, paise for INR, satoshi-equivalent for crypto if added)
//! plus an ISO 4217 currency tag that knows its own decimal places.
//!
//! ## Range
//!
//! `i64` minor units covers ±9.2 × 10^18 minor units. For USD (2 decimals)
//! that is ±$92 quadrillion, comfortably above any single transaction value
//! on any rail. We use signed integers so refunds and adjustments compose
//! arithmetically.

use core::fmt;
use core::ops::{Add, Neg, Sub};

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// An ISO 4217 currency. We carry the alpha-3 code plus the number of
/// minor-unit decimal places (the "exponent" in ISO 4217 terms).
///
/// Only a curated set is hard-coded; arbitrary codes can be constructed
/// via [`Currency::try_new`] for forward-compatibility.
#[derive(Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Currency {
    code: [u8; 3],
    exponent: u8,
}

impl Currency {
    /// United States Dollar (2 decimal places).
    pub const USD: Self = Self {
        code: *b"USD",
        exponent: 2,
    };
    /// Euro (2 decimal places).
    pub const EUR: Self = Self {
        code: *b"EUR",
        exponent: 2,
    };
    /// Brazilian Real (2 decimal places).
    pub const BRL: Self = Self {
        code: *b"BRL",
        exponent: 2,
    };
    /// Indian Rupee (2 decimal places).
    pub const INR: Self = Self {
        code: *b"INR",
        exponent: 2,
    };
    /// British Pound (2 decimal places).
    pub const GBP: Self = Self {
        code: *b"GBP",
        exponent: 2,
    };
    /// Japanese Yen (0 decimal places).
    pub const JPY: Self = Self {
        code: *b"JPY",
        exponent: 0,
    };
    /// Chinese Yuan Renminbi (2 decimal places).
    pub const CNY: Self = Self {
        code: *b"CNY",
        exponent: 2,
    };

    /// Construct a currency from an arbitrary ISO 4217 code and exponent.
    ///
    /// # Errors
    /// Returns [`Error::InvalidCurrency`] if `code` is not three ASCII
    /// uppercase letters, or if `exponent` exceeds 4 (no real ISO 4217
    /// currency has more than 4 decimal places).
    pub const fn try_new(code: [u8; 3], exponent: u8) -> Result<Self> {
        let mut i = 0;
        while i < 3 {
            let c = code[i];
            if c < b'A' || c > b'Z' {
                return Err(Error::InvalidCurrency);
            }
            i += 1;
        }
        if exponent > 4 {
            return Err(Error::InvalidCurrency);
        }
        Ok(Self { code, exponent })
    }

    /// Three-letter ISO 4217 code as a string slice.
    #[must_use]
    pub fn code(&self) -> &str {
        // Safe: constructor guarantees ASCII.
        core::str::from_utf8(&self.code).unwrap_or("???")
    }

    /// Number of decimal places (the ISO 4217 "exponent").
    #[must_use]
    pub const fn exponent(&self) -> u8 {
        self.exponent
    }
}

impl fmt::Debug for Currency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Currency({})", self.code())
    }
}

impl fmt::Display for Currency {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

/// A monetary amount: exact minor-unit integer plus currency.
///
/// Arithmetic between two `Money` values is only defined when the currencies
/// match; mixed-currency operations return [`Error::CurrencyMismatch`].
#[derive(Copy, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Money {
    /// Amount in the currency's minor units (e.g. cents for USD).
    pub minor_units: i64,
    /// The currency this amount is denominated in.
    pub currency: Currency,
}

impl Money {
    /// Construct a money value directly from minor units.
    #[must_use]
    pub const fn from_minor(minor_units: i64, currency: Currency) -> Self {
        Self {
            minor_units,
            currency,
        }
    }

    /// Zero in the given currency.
    #[must_use]
    pub const fn zero(currency: Currency) -> Self {
        Self {
            minor_units: 0,
            currency,
        }
    }

    /// True if this amount is exactly zero.
    #[must_use]
    pub const fn is_zero(&self) -> bool {
        self.minor_units == 0
    }

    /// True if this amount is strictly positive.
    #[must_use]
    pub const fn is_positive(&self) -> bool {
        self.minor_units > 0
    }

    /// Checked addition. Returns [`Error::Overflow`] on i64 overflow,
    /// [`Error::CurrencyMismatch`] on mixed currencies.
    ///
    /// # Errors
    /// See above.
    pub fn checked_add(self, rhs: Self) -> Result<Self> {
        if self.currency != rhs.currency {
            return Err(Error::CurrencyMismatch);
        }
        let minor_units = self
            .minor_units
            .checked_add(rhs.minor_units)
            .ok_or(Error::Overflow)?;
        Ok(Self {
            minor_units,
            currency: self.currency,
        })
    }

    /// Checked subtraction. Same error model as [`Self::checked_add`].
    ///
    /// # Errors
    /// See [`Self::checked_add`].
    pub fn checked_sub(self, rhs: Self) -> Result<Self> {
        if self.currency != rhs.currency {
            return Err(Error::CurrencyMismatch);
        }
        let minor_units = self
            .minor_units
            .checked_sub(rhs.minor_units)
            .ok_or(Error::Overflow)?;
        Ok(Self {
            minor_units,
            currency: self.currency,
        })
    }
}

impl fmt::Debug for Money {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Money({} {})", self.minor_units, self.currency.code())
    }
}

impl fmt::Display for Money {
    /// Renders with the currency's natural decimal placement.
    /// Example: `Money::from_minor(1234, Currency::USD)` -> `"12.34 USD"`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let exp = u32::from(self.currency.exponent);
        if exp == 0 {
            write!(f, "{} {}", self.minor_units, self.currency.code())
        } else {
            let divisor = 10_i64.pow(exp);
            let whole = self.minor_units / divisor;
            let frac = (self.minor_units % divisor).abs();
            write!(
                f,
                "{whole}.{frac:0width$} {cur}",
                width = exp as usize,
                cur = self.currency.code()
            )
        }
    }
}

impl Neg for Money {
    type Output = Result<Self>;
    fn neg(self) -> Result<Self> {
        let minor_units = self.minor_units.checked_neg().ok_or(Error::Overflow)?;
        Ok(Self {
            minor_units,
            currency: self.currency,
        })
    }
}

// Operator sugar for the common case: returns Result, callers must handle.
impl Add for Money {
    type Output = Result<Self>;
    fn add(self, rhs: Self) -> Result<Self> {
        self.checked_add(rhs)
    }
}

impl Sub for Money {
    type Output = Result<Self>;
    fn sub(self, rhs: Self) -> Result<Self> {
        self.checked_sub(rhs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usd_display_two_decimals() {
        let m = Money::from_minor(1234, Currency::USD);
        assert_eq!(format!("{m}"), "12.34 USD");
    }

    #[test]
    fn jpy_display_no_decimals() {
        let m = Money::from_minor(1000, Currency::JPY);
        assert_eq!(format!("{m}"), "1000 JPY");
    }

    #[test]
    fn add_same_currency_succeeds() {
        let a = Money::from_minor(100, Currency::USD);
        let b = Money::from_minor(250, Currency::USD);
        assert_eq!((a + b).unwrap(), Money::from_minor(350, Currency::USD));
    }

    #[test]
    fn add_mixed_currency_fails() {
        let a = Money::from_minor(100, Currency::USD);
        let b = Money::from_minor(100, Currency::EUR);
        assert!(matches!(a + b, Err(Error::CurrencyMismatch)));
    }

    #[test]
    fn overflow_detected() {
        let a = Money::from_minor(i64::MAX, Currency::USD);
        let b = Money::from_minor(1, Currency::USD);
        assert!(matches!(a + b, Err(Error::Overflow)));
    }

    #[test]
    fn negate_works() {
        let a = Money::from_minor(500, Currency::USD);
        assert_eq!((-a).unwrap(), Money::from_minor(-500, Currency::USD));
    }

    #[test]
    fn negate_min_overflows() {
        let a = Money::from_minor(i64::MIN, Currency::USD);
        assert!(matches!(-a, Err(Error::Overflow)));
    }

    #[test]
    fn custom_currency_validates_ascii() {
        assert!(Currency::try_new(*b"AED", 2).is_ok());
        assert!(Currency::try_new(*b"ae3", 2).is_err()); // lowercase + digit
        assert!(Currency::try_new(*b"USD", 5).is_err()); // exponent too large
    }
}
