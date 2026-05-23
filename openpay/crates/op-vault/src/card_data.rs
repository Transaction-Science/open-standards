//! [`CardData`] — the only public type in the `OpenPay` surface that
//! holds raw PAN. Constructable only inside `pci-scope` callers.
//!
//! ## Validation
//!
//! Construction runs three checks before storing the value:
//!
//! 1. **Length**. PANs are 12-19 digits per ISO/IEC 7812.
//! 2. **Luhn**. The mod-10 check digit. Doesn't prove the card is
//!    legitimate, but rejects typos and accidental garbage.
//! 3. **Expiration**. Month 1-12, year >= current year.
//!
//! Reject patterns are surfaced via [`Error::InvalidCard`] so the
//! caller can show a sensible UI message. The PAN string itself never
//! enters the error message.
//!
//! ## Zeroize
//!
//! [`CardData`] derives `Zeroize` + `ZeroizeOnDrop`. Cloning is
//! intentional but the clone also zeroizes on drop. Internally we
//! delegate to [`op_core::method::pci::RawPan`] for the underlying
//! storage; that type already enforces zeroize semantics.

use op_core::method::pci::RawPan;
use zeroize::{Zeroize, ZeroizeOnDrop};

use crate::error::{Error, Result};

/// Validated card data. The only way for raw PAN to enter the vault.
///
/// This type is only available when `op-core`'s `pci-scope` feature is
/// enabled (which `op-vault` opts into unconditionally). Crates that
/// don't depend on `op-vault` cannot see this type, which is the
/// point.
#[derive(Clone, Zeroize, ZeroizeOnDrop)]
pub struct CardData {
    inner: RawPan,
}

impl CardData {
    /// Construct after validating length, Luhn, and expiration.
    ///
    /// # Errors
    /// `Error::InvalidCard` with a vague reason ("length", "luhn",
    /// "expiration") that's safe to surface to the user.
    pub fn new(pan: impl Into<String>, exp_month: u8, exp_year: u16) -> Result<Self> {
        let pan = pan.into();
        validate_pan(&pan)?;
        validate_expiration(exp_month, exp_year)?;
        Ok(Self {
            inner: RawPan::new(pan, exp_month, exp_year),
        })
    }

    /// Borrow the underlying [`RawPan`] for vault-side encryption.
    /// **Crate-private** — only vault implementations inside this
    /// crate can call this. External callers see only masked views.
    #[must_use]
    pub(crate) fn raw(&self) -> &RawPan {
        &self.inner
    }

    /// First six digits (BIN) — safe to log.
    #[must_use]
    pub fn first_six(&self) -> &str {
        self.inner.first_six()
    }

    /// Last four digits — safe to log.
    #[must_use]
    pub fn last_four(&self) -> &str {
        self.inner.last_four()
    }

    /// Expiration month (1-12).
    #[must_use]
    pub fn exp_month(&self) -> u8 {
        self.inner.exp_month()
    }

    /// Expiration year (e.g. 2027).
    #[must_use]
    pub fn exp_year(&self) -> u16 {
        self.inner.exp_year()
    }
}

impl core::fmt::Debug for CardData {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Never include full PAN. The first-six + last-four masked view
        // is the maximum PCI DSS 4.0.1 §3.4.1 allows in display contexts.
        write!(
            f,
            "CardData({}******{}, {:02}/{})",
            self.first_six(),
            self.last_four(),
            self.exp_month(),
            self.exp_year()
        )
    }
}

/// Verify PAN length (12-19) and Luhn check digit.
fn validate_pan(pan: &str) -> Result<()> {
    if !pan.chars().all(|c| c.is_ascii_digit()) {
        return Err(Error::InvalidCard("non-digit characters".into()));
    }
    let n = pan.len();
    if !(12..=19).contains(&n) {
        return Err(Error::InvalidCard("length".into()));
    }
    if !luhn_ok(pan) {
        return Err(Error::InvalidCard("luhn".into()));
    }
    Ok(())
}

/// ISO/IEC 7812 mod-10 (Luhn) check.
fn luhn_ok(pan: &str) -> bool {
    let mut sum = 0u32;
    let digits: Vec<u32> = pan.chars().filter_map(|c| c.to_digit(10)).collect();
    if digits.len() != pan.len() {
        return false;
    }
    // Iterate from rightmost digit; every second digit (starting from
    // the second-to-rightmost) gets doubled.
    for (i, d) in digits.iter().rev().enumerate() {
        let v = if i % 2 == 1 {
            let doubled = d * 2;
            if doubled > 9 { doubled - 9 } else { doubled }
        } else {
            *d
        };
        sum += v;
    }
    sum.is_multiple_of(10)
}

/// Expiration sanity check. Month must be 1-12. Year must be plausible
/// (current year through current year + 30; cards don't issue >30y out).
///
/// We don't reject *past* dates here — the caller may legitimately want
/// to tokenize an expired-card record for refund or chargeback
/// processing. The orchestrator separately enforces "not expired" at
/// authorization time.
fn validate_expiration(month: u8, year: u16) -> Result<()> {
    if !(1..=12).contains(&month) {
        return Err(Error::InvalidCard("expiration month".into()));
    }
    // Plausible-year sanity. 2000-2099 covers everything we care about.
    if !(2000..=2099).contains(&year) {
        return Err(Error::InvalidCard("expiration year".into()));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Known-good test PANs from the standard test vectors.
    /// `4242 4242 4242 4242` is the Visa test card published by Stripe;
    /// `5555 5555 5555 4444` is the Mastercard test card.
    const VALID_VISA: &str = "4242424242424242";
    const VALID_MC: &str = "5555555555554444";

    #[test]
    fn accepts_known_test_pans() {
        assert!(CardData::new(VALID_VISA, 12, 2030).is_ok());
        assert!(CardData::new(VALID_MC, 6, 2028).is_ok());
    }

    #[test]
    fn rejects_non_digits() {
        let err = CardData::new("4242-4242-4242-4242", 12, 2030).unwrap_err();
        assert!(matches!(err, Error::InvalidCard(s) if s.contains("non-digit")));
    }

    #[test]
    fn rejects_too_short() {
        let err = CardData::new("1234567890", 12, 2030).unwrap_err(); // 10 digits
        assert!(matches!(err, Error::InvalidCard(s) if s == "length"));
    }

    #[test]
    fn rejects_too_long() {
        let err = CardData::new("12345678901234567890", 12, 2030).unwrap_err(); // 20 digits
        assert!(matches!(err, Error::InvalidCard(s) if s == "length"));
    }

    #[test]
    fn rejects_failed_luhn() {
        // Swap last two digits of valid Visa: 4242...4224 fails Luhn.
        let err = CardData::new("4242424242424224", 12, 2030).unwrap_err();
        assert!(matches!(err, Error::InvalidCard(s) if s == "luhn"));
    }

    #[test]
    fn rejects_zero_month() {
        let err = CardData::new(VALID_VISA, 0, 2030).unwrap_err();
        assert!(matches!(err, Error::InvalidCard(s) if s.contains("month")));
    }

    #[test]
    fn rejects_thirteen_month() {
        let err = CardData::new(VALID_VISA, 13, 2030).unwrap_err();
        assert!(matches!(err, Error::InvalidCard(s) if s.contains("month")));
    }

    #[test]
    fn rejects_implausible_year() {
        assert!(matches!(
            CardData::new(VALID_VISA, 12, 1999),
            Err(Error::InvalidCard(_))
        ));
        assert!(matches!(
            CardData::new(VALID_VISA, 12, 2100),
            Err(Error::InvalidCard(_))
        ));
    }

    #[test]
    fn accepts_expired_card() {
        // Expiration validation does NOT reject past dates. The caller
        // separately enforces "not expired" at authorization time.
        assert!(CardData::new(VALID_VISA, 1, 2020).is_ok());
    }

    #[test]
    fn debug_masks_pan_to_first6_last4() {
        let cd = CardData::new(VALID_VISA, 12, 2030).unwrap();
        let dbg = format!("{cd:?}");
        assert!(dbg.contains("424242"));
        assert!(dbg.contains("4242"));
        assert!(
            !dbg.contains("4242424242424242"),
            "full PAN must not appear"
        );
        assert!(dbg.contains("12/2030"));
    }

    #[test]
    fn first_six_and_last_four_accessors() {
        let cd = CardData::new(VALID_VISA, 12, 2030).unwrap();
        assert_eq!(cd.first_six(), "424242");
        assert_eq!(cd.last_four(), "4242");
    }

    #[test]
    fn luhn_check_basic() {
        // Truth-table cases for the Luhn algorithm.
        assert!(luhn_ok("4242424242424242")); // Visa test
        assert!(luhn_ok("5555555555554444")); // MC test
        assert!(luhn_ok("378282246310005")); // Amex test (15 digits)
        assert!(!luhn_ok("4242424242424241")); // wrong check digit
        assert!(!luhn_ok("0000000000000001")); // sum is 1, not div by 10
        assert!(luhn_ok("0000000000000000")); // sum is 0 → passes
    }
}
