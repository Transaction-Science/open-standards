//! Durbin / Regulation II classification — must require both
//! debit (or prepaid) card type AND the issuer-side flag.

use op_bin::bin::BinRange;
use op_bin::durbin;
use op_bin::{CardNetwork, CardType};

fn mk(card_type: CardType, durbin_regulated: bool) -> BinRange {
    BinRange::new(
        40_000_000,
        50_000_000,
        CardNetwork::Visa,
        card_type,
        None,
        durbin_regulated,
    )
    .expect("valid range")
}

#[test]
fn credit_never_regulated_even_with_flag() {
    let r = mk(CardType::Credit, true);
    assert!(!durbin::is_regulated(&r));
    assert!(durbin::is_exempt(&r));
}

#[test]
fn charge_never_regulated() {
    let r = mk(CardType::Charge, true);
    assert!(!durbin::is_regulated(&r));
}

#[test]
fn debit_with_flag_is_regulated() {
    let r = mk(CardType::Debit, true);
    assert!(durbin::is_regulated(&r));
    assert!(!durbin::is_exempt(&r));
}

#[test]
fn debit_without_flag_is_exempt() {
    let r = mk(CardType::Debit, false);
    assert!(!durbin::is_regulated(&r));
    assert!(durbin::is_exempt(&r));
}

#[test]
fn prepaid_with_flag_is_regulated() {
    let r = mk(CardType::Prepaid, true);
    assert!(durbin::is_regulated(&r));
}

#[test]
fn unknown_card_type_is_exempt() {
    let r = mk(CardType::Unknown, true);
    assert!(!durbin::is_regulated(&r));
}
