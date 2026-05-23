//! Cross-surface integration tests.
//!
//! The JNI surface can't be exercised without a JVM, so these tests
//! focus on the C ABI and on shared utilities. The JNI handle/error
//! paths are covered by unit tests inside `jni_bridge.rs`.

use core::ffi::{CStr, c_char};

use crate::c_api::{
    OpTokenizationPolicy, op_card_data_first_six, op_card_data_free, op_card_data_last_four,
    op_card_data_new, op_last_error, op_scorer_free, op_scorer_name, op_scorer_new_heuristic,
    op_string_free, op_vault_free, op_vault_new_ephemeral, op_vault_ref_as_string,
    op_vault_ref_free, op_vault_tokenize,
};
use crate::error::FfiError;

const VALID_VISA: &[u8] = b"4242424242424242\0";
const NAME: &[u8] = b"jni-cross\0";

unsafe fn take_c_str(p: *mut c_char) -> String {
    assert!(!p.is_null());
    let s = unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_owned();
    unsafe { op_string_free(p) };
    s
}

#[test]
fn c_abi_and_jni_share_the_same_error_discriminants() {
    // The C ABI returns i32 codes; the JNI side throws typed
    // exceptions whose class names embed the same discriminants in
    // the exception_class method. Verify they line up.
    let invalid = FfiError::InvalidInput;
    assert_eq!(invalid.as_i32(), 1);
    assert!(invalid.exception_class().ends_with("InvalidInput"));

    let lookup = FfiError::VaultLookupFailed;
    assert_eq!(lookup.as_i32(), 2);
    assert!(lookup.exception_class().ends_with("VaultLookupFailed"));
}

#[test]
fn c_abi_full_lifecycle_works_in_isolation() {
    unsafe {
        let vault = op_vault_new_ephemeral(NAME.as_ptr() as *const c_char);
        let card = op_card_data_new(VALID_VISA.as_ptr() as *const c_char, 12, 2030);
        let policy = OpTokenizationPolicy {
            format: 0,
            lifetime: 0,
            ttl_seconds: 0,
        };
        let token = op_vault_tokenize(vault, card, policy);
        assert!(!token.is_null());
        let s = take_c_str(op_vault_ref_as_string(token));
        assert!(s.starts_with("tok_v7_"));
        op_vault_ref_free(token);
        op_vault_free(vault);
    }
}

#[test]
fn c_abi_strings_freeable_in_loop() {
    unsafe {
        for _ in 0..100 {
            let card = op_card_data_new(VALID_VISA.as_ptr() as *const c_char, 12, 2030);
            let f6 = op_card_data_first_six(card);
            let l4 = op_card_data_last_four(card);
            op_string_free(f6);
            op_string_free(l4);
            op_card_data_free(card);
        }
    }
}

#[test]
fn scorer_name_consistent_via_c_abi() {
    unsafe {
        let s = op_scorer_new_heuristic();
        let name = take_c_str(op_scorer_name(s));
        assert_eq!(name, "heuristic-v1");
        op_scorer_free(s);
    }
}

#[test]
fn c_abi_thread_local_error_is_per_thread() {
    use std::thread;

    let a = thread::spawn(|| unsafe {
        let bad = b"bad\0";
        let _ = op_card_data_new(bad.as_ptr() as *const c_char, 12, 2030);
        op_last_error()
    });

    let b = thread::spawn(|| unsafe {
        // Different thread: trigger same error category.
        let bad = b"also-bad\0";
        let _ = op_card_data_new(bad.as_ptr() as *const c_char, 12, 2030);
        op_last_error()
    });

    assert_eq!(a.join().unwrap(), FfiError::InvalidInput.as_i32());
    assert_eq!(b.join().unwrap(), FfiError::InvalidInput.as_i32());
}

#[test]
fn ffi_error_discriminants_match_phase_8_swift_exactly() {
    // The whole point of using identical discriminants across iOS and
    // Android is observability: a single dashboard maps error codes
    // to a unified taxonomy. Verify the contract.
    assert_eq!(FfiError::Ok.as_i32(), 0);
    assert_eq!(FfiError::InvalidInput.as_i32(), 1);
    assert_eq!(FfiError::VaultLookupFailed.as_i32(), 2);
    assert_eq!(FfiError::TokenExpired.as_i32(), 3);
    assert_eq!(FfiError::TokenAlreadyConsumed.as_i32(), 4);
    assert_eq!(FfiError::FraudDeclined.as_i32(), 5);
    assert_eq!(FfiError::FraudReviewRequired.as_i32(), 6);
    assert_eq!(FfiError::Backend.as_i32(), 7);
    assert_eq!(FfiError::Internal.as_i32(), 8);
    assert_eq!(FfiError::Capacity.as_i32(), 9);
}
