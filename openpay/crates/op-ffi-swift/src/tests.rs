//! Cross-surface integration tests.
//!
//! Exercises behaviors that span both [`crate::bridge`] (swift-bridge
//! surface) and [`crate::c_api`] (plain C ABI). These verify that the
//! two surfaces don't interfere with each other and that the same
//! underlying Rust types flow through both.

use core::ffi::{CStr, c_char};

use crate::bridge::{RustCardData, RustHeuristicScorer, RustVault};
use crate::c_api::{
    op_card_data_free, op_card_data_new, op_last_error, op_scorer_free, op_scorer_new_heuristic,
    op_string_free, op_vault_free, op_vault_new_ephemeral,
};
use crate::error::FfiError;

const VALID_VISA: &[u8] = b"4242424242424242\0";

unsafe fn take_c_str(p: *mut c_char) -> String {
    assert!(!p.is_null());
    let s = unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_owned();
    unsafe { op_string_free(p) };
    s
}

#[test]
fn bridge_and_c_api_both_construct_card_data_independently() {
    // Bridge surface
    let bridge_card = RustCardData::new("4242424242424242", 12, 2030);
    assert!(bridge_card.is_some());

    // C ABI surface
    unsafe {
        let c_card = op_card_data_new(VALID_VISA.as_ptr() as *const c_char, 12, 2030);
        assert!(!c_card.is_null());
        op_card_data_free(c_card);
    }
}

#[test]
fn bridge_and_c_api_each_have_their_own_thread_local_errors() {
    // The bridge module's last-error and the C ABI's last-error are
    // separate thread-locals. A failure on one surface should not
    // affect the other's reading.

    unsafe {
        // Trigger an error on the C ABI side.
        let card = op_card_data_new(c"bad".as_ptr(), 12, 2030);
        assert!(card.is_null());
        assert_eq!(op_last_error(), FfiError::InvalidInput.as_i32());

        // The bridge's last_error_card should be untouched (still Ok).
        // We can't read it without going through a bridge-side call,
        // but we can verify by triggering a bridge-side success and
        // confirming Ok rather than the previous error.
        let _ = RustCardData::new("4242424242424242", 12, 2030);
        assert_eq!(crate::bridge::last_error_card(), FfiError::Ok.as_i32());
    }
}

#[test]
fn ephemeral_vault_works_on_both_surfaces() {
    let _bridge_vault = RustVault::ephemeral("bridge-side");
    unsafe {
        let c_vault = op_vault_new_ephemeral(c"c-side".as_ptr());
        assert!(!c_vault.is_null());
        op_vault_free(c_vault);
    }
}

#[test]
fn scorer_default_name_matches_across_surfaces() {
    let bridge_scorer = RustHeuristicScorer::default();
    assert_eq!(bridge_scorer.name(), "heuristic-v1");
    unsafe {
        let c_scorer = op_scorer_new_heuristic();
        let name = take_c_str(crate::c_api::op_scorer_name(c_scorer));
        assert_eq!(name, "heuristic-v1");
        op_scorer_free(c_scorer);
    }
}

#[test]
fn c_abi_strings_are_safely_freeable() {
    // Round-trip a sequence of allocations to exercise the string-free
    // path without leaking or double-freeing.
    unsafe {
        for _ in 0..100 {
            let card = op_card_data_new(VALID_VISA.as_ptr() as *const c_char, 12, 2030);
            let f6 = crate::c_api::op_card_data_first_six(card);
            let l4 = crate::c_api::op_card_data_last_four(card);
            op_string_free(f6);
            op_string_free(l4);
            op_card_data_free(card);
        }
    }
}
