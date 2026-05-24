//! Static tables of published BIN prefix ranges per network.
//!
//! These tables encode the **public, structural** prefix
//! assignments documented in ISO/IEC 7812 and each network's
//! developer documentation. They are NOT a commercial BIN
//! database — they do not resolve individual issuing banks, and
//! they do not differentiate every credit / debit / prepaid
//! sub-range. Operators who need that resolution wire a
//! commercial feed (e.g. BinList, BIN Codes, FreeBINChecker) on
//! top of this scaffolding.
//!
//! ## Encoding
//!
//! Each entry is a half-open prefix interval `[low, high)` in
//! **8-digit prefix form**, with default annotations (network,
//! card type, country, Durbin). Use [`build_default_tree`] to
//! load the whole catalog into a [`RangeTree`].
//!
//! ## Adding a range
//!
//! Append a `R(low, high, network, type, country, durbin)` entry
//! to the appropriate const slice. The build function inserts in
//! order; overlap is rejected at runtime — keep entries disjoint.

use crate::bin::BinRange;
use crate::card_type::CardType;
use crate::error::Result;
use crate::issuer_country::IssuerCountry;
use crate::network::CardNetwork;
use crate::range_tree::RangeTree;

/// Convenience tuple for declaring a range in a const slice.
type Row = (u32, u32, CardNetwork, CardType, Option<IssuerCountry>, bool);

const US: Option<IssuerCountry> = Some(IssuerCountry::from_ascii(b'U', b'S'));
const CN: Option<IssuerCountry> = Some(IssuerCountry::from_ascii(b'C', b'N'));
const JP: Option<IssuerCountry> = Some(IssuerCountry::from_ascii(b'J', b'P'));
const IN_: Option<IssuerCountry> = Some(IssuerCountry::from_ascii(b'I', b'N'));
const BR: Option<IssuerCountry> = Some(IssuerCountry::from_ascii(b'B', b'R'));
const RU: Option<IssuerCountry> = Some(IssuerCountry::from_ascii(b'R', b'U'));
const TR: Option<IssuerCountry> = Some(IssuerCountry::from_ascii(b'T', b'R'));
const ANY: Option<IssuerCountry> = None;

/// Visa: leading digit `4`. Single contiguous range over the
/// entire `[4_000_000_0, 5_000_000_0)` 8-digit space.
pub const VISA: &[Row] = &[(
    40_000_000,
    50_000_000,
    CardNetwork::Visa,
    CardType::Credit,
    ANY,
    false,
)];

/// Mastercard: classic 51-55 + 2017-expansion 2-series 2221-2720.
pub const MASTERCARD: &[Row] = &[
    // 2-series (the "2BIN" range, opened 2017).
    (
        22_210_000,
        27_210_000,
        CardNetwork::Mastercard,
        CardType::Credit,
        ANY,
        false,
    ),
    // Classic 51-55.
    (
        51_000_000,
        56_000_000,
        CardNetwork::Mastercard,
        CardType::Credit,
        ANY,
        false,
    ),
];

/// American Express: 34 / 37 prefixes, charge cards.
pub const AMEX: &[Row] = &[
    (
        34_000_000,
        35_000_000,
        CardNetwork::Amex,
        CardType::Charge,
        ANY,
        false,
    ),
    (
        37_000_000,
        38_000_000,
        CardNetwork::Amex,
        CardType::Charge,
        ANY,
        false,
    ),
];

/// Discover: 6011, 622126-622925, 644-649, 65.
///
/// Note: 65 overlaps RuPay at the 2-digit prefix; the 8-digit
/// sub-range chosen here (`65_000_000..66_000_000`) is the
/// Discover-assigned sub-block per Discover's public range list.
pub const DISCOVER: &[Row] = &[
    (
        60_110_000,
        60_120_000,
        CardNetwork::Discover,
        CardType::Credit,
        US,
        false,
    ),
    (
        62_212_600,
        62_292_600,
        CardNetwork::Discover,
        CardType::Credit,
        US,
        false,
    ),
    (
        64_400_000,
        65_000_000,
        CardNetwork::Discover,
        CardType::Credit,
        US,
        false,
    ),
    (
        65_000_000,
        66_000_000,
        CardNetwork::Discover,
        CardType::Credit,
        US,
        false,
    ),
];

/// JCB: 3528-3589.
pub const JCB: &[Row] = &[(
    35_280_000,
    35_900_000,
    CardNetwork::Jcb,
    CardType::Credit,
    JP,
    false,
)];

/// Diners Club: 300-305, 309, 36, 38-39.
///
/// 38 / 39 historically split with Amex (37) and Diners. The
/// 38-39 prefixes are Diners; Amex sits at 34/37.
pub const DINERS_CLUB: &[Row] = &[
    (
        30_000_000,
        30_600_000,
        CardNetwork::DinersClub,
        CardType::Charge,
        ANY,
        false,
    ),
    (
        30_900_000,
        31_000_000,
        CardNetwork::DinersClub,
        CardType::Charge,
        ANY,
        false,
    ),
    (
        36_000_000,
        37_000_000,
        CardNetwork::DinersClub,
        CardType::Charge,
        ANY,
        false,
    ),
    (
        38_000_000,
        40_000_000,
        CardNetwork::DinersClub,
        CardType::Charge,
        ANY,
        false,
    ),
];

/// UnionPay: 62 (note: overlaps Discover's `622126-622925`
/// sub-range, which we already carved out above; we encode the
/// non-overlapping remainder here in two slices).
pub const UNIONPAY: &[Row] = &[
    (
        62_000_000,
        62_212_600,
        CardNetwork::UnionPay,
        CardType::Credit,
        CN,
        false,
    ),
    (
        62_292_600,
        63_000_000,
        CardNetwork::UnionPay,
        CardType::Credit,
        CN,
        false,
    ),
];

/// RuPay: 60 (excluding Discover's 6011), 81, 82.
pub const RUPAY: &[Row] = &[
    (
        60_000_000,
        60_110_000,
        CardNetwork::RuPay,
        CardType::Debit,
        IN_,
        false,
    ),
    (
        60_120_000,
        61_000_000,
        CardNetwork::RuPay,
        CardType::Debit,
        IN_,
        false,
    ),
    (
        81_000_000,
        83_000_000,
        CardNetwork::RuPay,
        CardType::Debit,
        IN_,
        false,
    ),
];

/// Maestro (Mastercard international debit): selected sub-ranges.
pub const MAESTRO: &[Row] = &[
    (
        50_180_000,
        50_190_000,
        CardNetwork::Maestro,
        CardType::Debit,
        ANY,
        false,
    ),
    (
        50_200_000,
        50_210_000,
        CardNetwork::Maestro,
        CardType::Debit,
        ANY,
        false,
    ),
    (
        50_380_000,
        50_390_000,
        CardNetwork::Maestro,
        CardType::Debit,
        ANY,
        false,
    ),
    (
        56_120_000,
        56_130_000,
        CardNetwork::Maestro,
        CardType::Debit,
        ANY,
        false,
    ),
    (
        58_930_000,
        58_940_000,
        CardNetwork::Maestro,
        CardType::Debit,
        ANY,
        false,
    ),
    (
        63_040_000,
        63_050_000,
        CardNetwork::Maestro,
        CardType::Debit,
        ANY,
        false,
    ),
    (
        67_590_000,
        67_600_000,
        CardNetwork::Maestro,
        CardType::Debit,
        ANY,
        false,
    ),
    (
        67_610_000,
        67_640_000,
        CardNetwork::Maestro,
        CardType::Debit,
        ANY,
        false,
    ),
];

/// Elo (Brazil): published 4-digit prefixes.
///
/// Elo is a co-brand BIN sponsor in Brazil — many of its public
/// prefixes sit *inside* ranges otherwise issued by Visa (4011,
/// 4312, 4389, 4514), Mastercard (5041, 5066, 5067, 5090), or
/// Discover (6504, 6505). To keep the range tree disjoint we
/// store **only** Elo prefixes that do not overlap a primary
/// network's catalog range; the co-branded prefixes are still
/// resolvable via the [`crate::network::classify`] structural
/// classifier, which special-cases each known Elo 4-digit
/// prefix. The two non-overlapping prefixes encoded here are
/// 6362-6363 and 6277, which sit in unassigned Discover/UnionPay
/// carve-outs.
pub const ELO: &[Row] = &[
    (
        63_620_000,
        63_640_000,
        CardNetwork::Elo,
        CardType::Credit,
        BR,
        false,
    ),
];

/// Mir (Russia): 2200-2204.
pub const MIR: &[Row] = &[(
    22_000_000,
    22_050_000,
    CardNetwork::Mir,
    CardType::Debit,
    RU,
    false,
)];

/// Troy (Türkiye): 9792.
pub const TROY: &[Row] = &[(
    97_920_000,
    97_930_000,
    CardNetwork::Troy,
    CardType::Debit,
    TR,
    false,
)];

/// Build a [`RangeTree`] populated with every published range
/// from this module. The catalog is intentionally **disjoint**:
/// where two networks share a 2-digit prefix (Discover/UnionPay
/// on `62`, Discover/RuPay on `60` and `65`), the conflict is
/// resolved by sub-range carve-out above.
///
/// # Errors
///
/// - [`crate::error::Error::InvalidRange`] if any pair overlaps —
///   indicates a bug in the static tables that the unit test in
///   `tests/range_tree.rs` should catch.
pub fn build_default_tree() -> Result<RangeTree> {
    let mut tree = RangeTree::new();
    for slice in [
        VISA,
        MASTERCARD,
        AMEX,
        DISCOVER,
        JCB,
        DINERS_CLUB,
        UNIONPAY,
        RUPAY,
        MAESTRO,
        ELO,
        MIR,
        TROY,
    ] {
        for &(low, high, net, ct, country, durbin) in slice {
            tree.insert(BinRange::new(low, high, net, ct, country, durbin)?)?;
        }
    }
    Ok(tree)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_tree_builds() {
        let tree = build_default_tree().expect("disjoint");
        assert!(tree.len() > 20);
    }

    #[test]
    fn default_tree_visa_lookup() {
        let tree = build_default_tree().expect("disjoint");
        let bin = crate::bin::Bin::parse("411111").expect("ok");
        let r = tree.lookup(&bin).expect("Visa range");
        assert_eq!(r.network, CardNetwork::Visa);
    }
}
