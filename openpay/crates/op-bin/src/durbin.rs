//! Durbin Amendment / Federal Reserve Regulation II helpers.
//!
//! The Durbin Amendment (section 1075 of the Dodd-Frank Act,
//! implemented by the Fed's Regulation II) caps debit-interchange
//! fees on cards issued by banks with **>= $10 billion in
//! consolidated assets**. The merchant-side cap as of 2026 is
//! `$0.21 + 5 bps + $0.01 (fraud-adjustment)` per transaction.
//!
//! ## What we can determine from a BIN
//!
//! - **Card type must be debit** (or prepaid that is treated as
//!   debit under Reg II Subpart C). Credit and charge cards are
//!   never regulated under Durbin.
//! - **Issuer asset size**: not derivable from the BIN structure
//!   alone. The reference stack flags each [`BinRange`] with a
//!   `durbin_regulated` boolean populated from operator-supplied
//!   tables; the helpers in this module fold that flag with the
//!   card-type test.
//!
//! ## What this module is NOT
//!
//! Not a fee calculator — Reg II compliance interacts with
//! routing requirements (the merchant's right to route over a
//! second unaffiliated network), exclusivity prohibitions, and
//! the Fed's annual revision of the asset threshold. Use this
//! module only to answer "is this BIN subject to Reg II's
//! interchange cap?", not "what is the cap?".
//!
//! [`BinRange`]: crate::bin::BinRange

use crate::bin::BinRange;
use crate::card_type::CardType;

/// True iff a transaction on the given range is subject to
/// Regulation II's interchange cap.
///
/// Requires both:
/// 1. `card_type` is `Debit` or `Prepaid`, AND
/// 2. `durbin_regulated` is set on the range (issuer >= $10B in
///    assets, populated from operator data).
pub fn is_regulated(range: &BinRange) -> bool {
    matches!(range.card_type, CardType::Debit | CardType::Prepaid) && range.durbin_regulated
}

/// True iff the issuing institution is "exempt" — issuer assets
/// below the Reg II threshold, OR the card is not debit/prepaid.
/// Provided as a counterpart for caller readability.
pub fn is_exempt(range: &BinRange) -> bool {
    !is_regulated(range)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::CardNetwork;

    fn mk(card_type: CardType, durbin: bool) -> BinRange {
        BinRange::new(
            40_000_000,
            50_000_000,
            CardNetwork::Visa,
            card_type,
            None,
            durbin,
        )
        .expect("valid range")
    }

    #[test]
    fn debit_with_durbin_flag_is_regulated() {
        assert!(is_regulated(&mk(CardType::Debit, true)));
    }

    #[test]
    fn debit_without_durbin_flag_is_exempt() {
        assert!(is_exempt(&mk(CardType::Debit, false)));
    }

    #[test]
    fn credit_never_regulated() {
        assert!(!is_regulated(&mk(CardType::Credit, true)));
        assert!(is_exempt(&mk(CardType::Credit, true)));
    }

    #[test]
    fn prepaid_can_be_regulated() {
        assert!(is_regulated(&mk(CardType::Prepaid, true)));
    }
}
