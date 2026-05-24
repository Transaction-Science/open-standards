//! Luhn (ISO/IEC 7812-1 Annex B) check-digit algorithm.
//!
//! The Luhn formula is a mod-10 checksum used by every major card
//! network on the full PAN. This module operates on **ASCII-digit
//! strings** so the caller controls allocation and we never copy
//! into a `String`. It is the only place in `op-bin` that accepts
//! more than 8 digits — calculation requires the whole PAN.
//!
//! ## Algorithm
//!
//! 1. From the **rightmost** digit (the check digit), double every
//!    second digit (positions 2, 4, 6, … counting from the right
//!    starting at 1).
//! 2. If doubling produces a two-digit number, sum its digits
//!    (equivalently, subtract 9).
//! 3. Sum all (possibly transformed) digits.
//! 4. Valid iff `sum mod 10 == 0`.

use crate::error::{Error, Result};

/// Validate a PAN-shaped ASCII-digit string under the Luhn
/// algorithm. Accepts any length `>= 2`; rejects empty strings,
/// single digits, and any non-digit character.
///
/// # Errors
///
/// - [`Error::InvalidBinCharacter`] for non-`'0'..='9'`.
/// - [`Error::LuhnFailed`] on checksum mismatch.
pub fn validate(pan: &str) -> Result<()> {
    if pan.len() < 2 {
        return Err(Error::LuhnFailed);
    }
    let mut sum: u32 = 0;
    let mut alt = false;
    for c in pan.chars().rev() {
        let d = c.to_digit(10).ok_or(Error::InvalidBinCharacter(c))?;
        let v = if alt {
            let doubled = d * 2;
            if doubled > 9 {
                doubled - 9
            } else {
                doubled
            }
        } else {
            d
        };
        sum += v;
        alt = !alt;
    }
    if sum % 10 == 0 {
        Ok(())
    } else {
        Err(Error::LuhnFailed)
    }
}

/// Boolean form of [`validate`]. Returns `false` on any error
/// (non-digit input *or* checksum mismatch).
pub fn is_valid(pan: &str) -> bool {
    validate(pan).is_ok()
}

/// Compute the check digit that would make `partial` Luhn-valid.
/// `partial` is the PAN **without** its final check digit.
///
/// # Errors
///
/// - [`Error::InvalidBinCharacter`] for non-`'0'..='9'`.
pub fn check_digit(partial: &str) -> Result<u8> {
    let mut sum: u32 = 0;
    // Position counting from the right of the **completed** PAN:
    // partial.len() rightmost positions of the completed string
    // start at index 2 (the check digit itself is index 1, which
    // we are computing). So the rightmost digit of `partial`
    // sits at the "doubled" alt position.
    let mut alt = true;
    for c in partial.chars().rev() {
        let d = c.to_digit(10).ok_or(Error::InvalidBinCharacter(c))?;
        let v = if alt {
            let doubled = d * 2;
            if doubled > 9 {
                doubled - 9
            } else {
                doubled
            }
        } else {
            d
        };
        sum += v;
        alt = !alt;
    }
    let cd = (10 - (sum % 10)) % 10;
    Ok(cd as u8)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_valid_visa() {
        // Industry test PAN published in Visa developer docs.
        assert!(is_valid("4111111111111111"));
    }

    #[test]
    fn known_valid_mastercard() {
        assert!(is_valid("5555555555554444"));
    }

    #[test]
    fn known_valid_amex() {
        assert!(is_valid("378282246310005"));
    }

    #[test]
    fn invalid_checksum() {
        assert!(!is_valid("4111111111111112"));
    }

    #[test]
    fn empty_rejected() {
        assert!(!is_valid(""));
    }

    #[test]
    fn non_digit_rejected() {
        assert!(!is_valid("4111-1111-1111-1111"));
    }

    #[test]
    fn check_digit_visa() {
        let cd = check_digit("411111111111111").expect("ok");
        assert_eq!(cd, 1);
    }

    #[test]
    fn check_digit_round_trip() {
        let partial = "53799347395198";
        let cd = check_digit(partial).expect("ok");
        let full = format!("{partial}{cd}");
        assert!(is_valid(&full), "{full} should be Luhn-valid");
    }
}
