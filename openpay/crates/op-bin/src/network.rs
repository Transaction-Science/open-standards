//! Card-network enum + structural-prefix classifier.
//!
//! [`classify`] applies the public prefix rules drawn from
//! ISO/IEC 7812 IIN assignments and each network's developer
//! documentation. It does **not** consult a commercial BIN feed;
//! it returns the network owning the prefix range. For deeper
//! attributes (card type, country, Durbin), use [`RangeTree`]
//! built from [`crate::network_ranges`].
//!
//! [`RangeTree`]: crate::range_tree::RangeTree

use serde::{Deserialize, Serialize};

use crate::bin::Bin;

/// Card networks supported by `op-bin`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CardNetwork {
    /// Visa.
    Visa,
    /// Mastercard.
    Mastercard,
    /// American Express.
    Amex,
    /// Discover.
    Discover,
    /// JCB (Japan Credit Bureau).
    Jcb,
    /// Diners Club International.
    DinersClub,
    /// China UnionPay.
    UnionPay,
    /// RuPay (India).
    RuPay,
    /// Maestro (Mastercard's international debit brand).
    Maestro,
    /// Elo (Brazil).
    Elo,
    /// Mir (Russia).
    Mir,
    /// Troy (Türkiye).
    Troy,
    /// Unknown / unclassified.
    Unknown,
}

impl CardNetwork {
    /// Lowercase wire string (`"visa"`, `"mastercard"`, …).
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Visa => "visa",
            Self::Mastercard => "mastercard",
            Self::Amex => "amex",
            Self::Discover => "discover",
            Self::Jcb => "jcb",
            Self::DinersClub => "diners-club",
            Self::UnionPay => "unionpay",
            Self::RuPay => "rupay",
            Self::Maestro => "maestro",
            Self::Elo => "elo",
            Self::Mir => "mir",
            Self::Troy => "troy",
            Self::Unknown => "unknown",
        }
    }
}

/// Classify a BIN to its card network using structural prefix
/// rules. This is the **fast path** — no table lookup, just
/// integer prefix arithmetic — and is intentionally
/// conservative: any BIN that doesn't match a documented prefix
/// falls through to [`CardNetwork::Unknown`].
///
/// Full annotation (card type, country, Durbin status) requires
/// a [`RangeTree`](crate::range_tree::RangeTree) built from
/// [`crate::network_ranges`].
pub fn classify(bin: &Bin) -> CardNetwork {
    // Work in the 8-digit prefix domain so all rules compose.
    let p = bin.prefix_8();

    // Read the leading `n` digits of `p`.
    let lead2 = p / 1_000_000; // first 2 digits
    let lead4 = p / 10_000; // first 4 digits
    let lead6 = p / 100; // first 6 digits
    let lead3 = p / 100_000; // first 3 digits
    let lead1 = p / 10_000_000; // first digit

    // --- Visa: leading 4 ---
    if lead1 == 4 {
        return CardNetwork::Visa;
    }

    // --- Amex: 34 or 37 ---
    if lead2 == 34 || lead2 == 37 {
        return CardNetwork::Amex;
    }

    // --- Mastercard:
    //     classic 51-55, plus the 2017 expansion 2221-2720.
    if (51..=55).contains(&lead2) {
        return CardNetwork::Mastercard;
    }
    if (2221..=2720).contains(&lead4) {
        return CardNetwork::Mastercard;
    }

    // --- Discover:
    //     6011, 622126-622925, 644-649, 65.
    if lead4 == 6011 {
        return CardNetwork::Discover;
    }
    if (622_126..=622_925).contains(&lead6) {
        return CardNetwork::Discover;
    }
    if (644..=649).contains(&lead3) {
        return CardNetwork::Discover;
    }
    if lead2 == 65 {
        return CardNetwork::Discover;
    }

    // --- JCB: 3528-3589 ---
    if (3528..=3589).contains(&lead4) {
        return CardNetwork::Jcb;
    }

    // --- Diners Club:
    //     300-305, 309, 36, 38-39. Note: Discover acquired the
    //     Diners US/Canada portfolio in 2008; we still report
    //     the BIN as DinersClub since the network identifier on
    //     the card surface is unchanged.
    if (300..=305).contains(&lead3) || lead3 == 309 {
        return CardNetwork::DinersClub;
    }
    if lead2 == 36 || lead2 == 38 || lead2 == 39 {
        return CardNetwork::DinersClub;
    }

    // --- UnionPay: 62 ---
    if lead2 == 62 {
        return CardNetwork::UnionPay;
    }

    // --- RuPay: 60 (excluding 6011 = Discover), 65 (already
    //     Discover above), 81, 82.
    //     The 65 range is shared at the prefix level; downstream
    //     range-tree lookup distinguishes per sub-range.
    if lead2 == 60 && lead4 != 6011 {
        return CardNetwork::RuPay;
    }
    if lead2 == 81 || lead2 == 82 {
        return CardNetwork::RuPay;
    }

    // --- Mir (Russia): 2200-2204 ---
    if (2200..=2204).contains(&lead4) {
        return CardNetwork::Mir;
    }

    // --- Troy (Türkiye): 9792 ---
    if lead4 == 9792 {
        return CardNetwork::Troy;
    }

    // --- Maestro: 5018, 5020, 5038, 5612, 5893, 6304, 6759,
    //     6761, 6762, 6763. (Partial — Maestro publishes the
    //     classification by sub-range; downstream RangeTree
    //     covers exhaustively.)
    if matches!(
        lead4,
        5018 | 5020 | 5038 | 5612 | 5893 | 6304 | 6759 | 6761 | 6762 | 6763
    ) {
        return CardNetwork::Maestro;
    }

    // --- Elo (Brazil): selected 4-digit prefixes (partial). ---
    if matches!(
        lead4,
        4011 | 4312 | 4389 | 4514 | 5041 | 5066 | 5067 | 5090 | 6277 | 6362 | 6363 | 6504 | 6505
    ) {
        return CardNetwork::Elo;
    }

    CardNetwork::Unknown
}

#[cfg(test)]
mod tests {
    use super::*;

    fn bin(s: &str) -> Bin {
        Bin::parse(s).expect("valid BIN literal")
    }

    #[test]
    fn visa() {
        assert_eq!(classify(&bin("411111")), CardNetwork::Visa);
        assert_eq!(classify(&bin("400000")), CardNetwork::Visa);
        assert_eq!(classify(&bin("499999")), CardNetwork::Visa);
    }

    #[test]
    fn mastercard_classic() {
        assert_eq!(classify(&bin("510000")), CardNetwork::Mastercard);
        assert_eq!(classify(&bin("559999")), CardNetwork::Mastercard);
    }

    #[test]
    fn mastercard_2_series() {
        assert_eq!(classify(&bin("222100")), CardNetwork::Mastercard);
        assert_eq!(classify(&bin("272099")), CardNetwork::Mastercard);
        // Boundary: 2220 is NOT Mastercard.
        assert_ne!(classify(&bin("222000")), CardNetwork::Mastercard);
        // Boundary: 2721 is NOT Mastercard.
        assert_ne!(classify(&bin("272100")), CardNetwork::Mastercard);
    }

    #[test]
    fn amex() {
        assert_eq!(classify(&bin("340000")), CardNetwork::Amex);
        assert_eq!(classify(&bin("370000")), CardNetwork::Amex);
    }

    #[test]
    fn discover_6011() {
        assert_eq!(classify(&bin("601100")), CardNetwork::Discover);
    }

    #[test]
    fn jcb() {
        assert_eq!(classify(&bin("352800")), CardNetwork::Jcb);
        assert_eq!(classify(&bin("358900")), CardNetwork::Jcb);
    }

    #[test]
    fn unionpay() {
        assert_eq!(classify(&bin("621234")), CardNetwork::UnionPay);
    }

    #[test]
    fn diners_36() {
        assert_eq!(classify(&bin("360000")), CardNetwork::DinersClub);
    }

    #[test]
    fn unknown_falls_through() {
        // 7xxxxx is unassigned in the BIN space.
        assert_eq!(classify(&bin("700000")), CardNetwork::Unknown);
    }
}
