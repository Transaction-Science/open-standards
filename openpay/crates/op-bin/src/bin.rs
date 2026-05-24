//! BIN value and prefix-range types.
//!
//! ISO/IEC 7812 assigns issuer identification in 6-to-8-digit
//! prefixes. We normalize every input to its **8-digit prefix
//! form** (left-justified, right-padded with zeros for the low
//! end of a range; right-padded with nines for the high end).
//! This lets us collapse 6-, 7-, and 8-digit IINs into a single
//! `u32` keyspace in `0..=99_999_999`.

use core::fmt;

use serde::{Deserialize, Serialize};

use crate::card_type::CardType;
use crate::error::{Error, Result};
use crate::issuer_country::IssuerCountry;
use crate::network::CardNetwork;

/// A validated BIN — 6, 7, or 8 ASCII decimal digits.
///
/// Internally normalized to an 8-digit `u32` prefix (the original
/// length is preserved). Constructors accept strings *or* the
/// leading digits of a PAN; in the latter case the caller is
/// responsible for truncating to `<= 8` digits **before** calling
/// `Bin::parse` (PCI scope).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Bin {
    /// 8-digit prefix form, right-padded with zeros.
    /// Range: `0..=99_999_999`.
    prefix_8: u32,
    /// Original digit length (6, 7, or 8).
    length: u8,
}

impl Bin {
    /// Parse a 6-to-8-digit ASCII numeric string into a `Bin`.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidBinLength`] if `s.len()` is not in `6..=8`.
    /// - [`Error::InvalidBinCharacter`] for any non-`'0'..='9'`.
    pub fn parse(s: &str) -> Result<Self> {
        let n = s.len();
        if !(6..=8).contains(&n) {
            return Err(Error::InvalidBinLength { got: n });
        }
        let mut value: u32 = 0;
        for c in s.chars() {
            let d = c.to_digit(10).ok_or(Error::InvalidBinCharacter(c))?;
            value = value * 10 + d;
        }
        // Left-justify into the 8-digit slot.
        let pad = 8 - n;
        let prefix_8 = value * 10u32.pow(pad as u32);
        Ok(Self {
            prefix_8,
            length: n as u8,
        })
    }

    /// Construct directly from a numeric value, asserting length.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidBinLength`] if `length` is not in `6..=8`.
    pub fn from_u32(value: u32, length: u8) -> Result<Self> {
        if !(6..=8).contains(&length) {
            return Err(Error::InvalidBinLength {
                got: length as usize,
            });
        }
        let pad = 8 - length;
        let prefix_8 = value * 10u32.pow(pad as u32);
        Ok(Self { prefix_8, length })
    }

    /// 8-digit prefix form (right-padded with zeros).
    pub const fn prefix_8(&self) -> u32 {
        self.prefix_8
    }

    /// Original BIN length: 6, 7, or 8.
    pub const fn length(&self) -> u8 {
        self.length
    }

    /// Truncate the leading digits of a PAN to a `Bin`. The caller
    /// must hand us **only** the first `digits` characters; this
    /// helper exists so the BIN-extraction site is grep-able and
    /// PCI-auditable. `digits` is clamped to `6..=8`.
    ///
    /// # Errors
    ///
    /// - As [`Bin::parse`].
    pub fn from_pan_prefix(prefix: &str, digits: usize) -> Result<Self> {
        let d = digits.clamp(6, 8);
        if prefix.len() < d {
            return Err(Error::InvalidBinLength { got: prefix.len() });
        }
        Self::parse(&prefix[..d])
    }
}

impl fmt::Display for Bin {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let pad = 8 - self.length;
        let value = self.prefix_8 / 10u32.pow(pad as u32);
        write!(f, "{:0width$}", value, width = self.length as usize)
    }
}

/// A half-open BIN prefix interval `[low, high)` annotated with
/// the network and bookkeeping flags. Bounds are expressed in
/// **8-digit prefix form**.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BinRange {
    /// Inclusive lower bound (8-digit form).
    pub low: u32,
    /// Exclusive upper bound (8-digit form).
    pub high: u32,
    /// Card network that owns this range.
    pub network: CardNetwork,
    /// Card type — credit / debit / prepaid / charge.
    pub card_type: CardType,
    /// Issuer country (ISO 3166-1 alpha-2). Optional —
    /// many ranges are multi-country.
    pub country: Option<IssuerCountry>,
    /// Whether issuer is subject to Regulation II
    /// debit-interchange caps (Durbin Amendment).
    pub durbin_regulated: bool,
}

impl BinRange {
    /// Build a half-open prefix interval, validating ordering.
    ///
    /// # Errors
    ///
    /// - [`Error::InvalidRange`] if `low >= high`.
    pub fn new(
        low: u32,
        high: u32,
        network: CardNetwork,
        card_type: CardType,
        country: Option<IssuerCountry>,
        durbin_regulated: bool,
    ) -> Result<Self> {
        if low >= high {
            return Err(Error::InvalidRange { low, high });
        }
        Ok(Self {
            low,
            high,
            network,
            card_type,
            country,
            durbin_regulated,
        })
    }

    /// True iff the BIN falls within this half-open interval.
    pub fn contains(&self, bin: &Bin) -> bool {
        let p = bin.prefix_8();
        p >= self.low && p < self.high
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bin_parse_6_digits() {
        let b = Bin::parse("411111").expect("valid");
        assert_eq!(b.length(), 6);
        assert_eq!(b.prefix_8(), 41_111_100);
        assert_eq!(b.to_string(), "411111");
    }

    #[test]
    fn bin_parse_8_digits() {
        let b = Bin::parse("41111111").expect("valid");
        assert_eq!(b.length(), 8);
        assert_eq!(b.prefix_8(), 41_111_111);
    }

    #[test]
    fn bin_rejects_length_5() {
        assert!(matches!(
            Bin::parse("41111"),
            Err(Error::InvalidBinLength { got: 5 })
        ));
    }

    #[test]
    fn bin_rejects_non_digit() {
        assert!(matches!(
            Bin::parse("41A111"),
            Err(Error::InvalidBinCharacter('A'))
        ));
    }

    #[test]
    fn bin_from_pan_prefix_truncates() {
        let b = Bin::from_pan_prefix("4111111111111111", 6).expect("valid");
        assert_eq!(b.length(), 6);
        assert_eq!(b.to_string(), "411111");
    }

    #[test]
    fn range_contains() {
        let r = BinRange::new(
            40_000_000,
            50_000_000,
            CardNetwork::Visa,
            CardType::Credit,
            None,
            false,
        )
        .expect("valid");
        let b = Bin::parse("411111").expect("valid");
        assert!(r.contains(&b));
    }

    #[test]
    fn range_rejects_inverted() {
        assert!(matches!(
            BinRange::new(
                50_000_000,
                40_000_000,
                CardNetwork::Visa,
                CardType::Credit,
                None,
                false,
            ),
            Err(Error::InvalidRange { .. })
        ));
    }
}
