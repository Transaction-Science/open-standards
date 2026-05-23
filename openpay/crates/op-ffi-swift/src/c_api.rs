//! Plain C ABI surface.
//!
//! Parallels [`crate::bridge`] but uses raw `extern "C"` functions and
//! pointer-based ownership. Use this surface when:
//!
//! - The consumer doesn't want `swift-bridge-build` in their Xcode
//!   workflow.
//! - The consumer wants to call OpenPay from C, Objective-C, or via
//!   Swift's `@_silgen_name` attribute against a hand-rolled wrapper.
//! - You need a clean cdylib boundary for fuzzing or interop testing.
//!
//! ## Ownership protocol
//!
//! Every `op_*_new` / `op_*_create` returns a raw `*mut T` allocated
//! on the heap (`Box::into_raw`). Every such pointer must be freed
//! exactly once via the matching `op_*_free`. Passing a null pointer
//! to `_free` is a no-op. Double-free is undefined behavior, same as
//! any C ABI.
//!
//! Functions that *borrow* take `*const T`; they must not free or
//! mutate. Functions that *consume* (e.g. `op_vault_tokenize` consumes
//! the `RustCardData`) are documented per-function.
//!
//! ## Error reporting
//!
//! C ABI calls that can fail follow one of two conventions:
//!
//! 1. **Pointer-returning calls** return null on failure; the caller
//!    reads `op_last_error_*()` to discriminate.
//! 2. **Status-returning calls** return [`FfiError`] discriminants
//!    directly as `i32`. Zero is success, anything else is the
//!    matching FfiError variant.
//!
//! ## Strings
//!
//! All Rust→C strings are NUL-terminated UTF-8, allocated by Rust,
//! and freed via [`op_string_free`]. C→Rust strings are read-only
//! `*const c_char` and must be valid NUL-terminated UTF-8; the
//! caller retains ownership.

use core::ffi::{CStr, c_char};
use std::ffi::CString;
use std::sync::Arc;

use op_fraud::HeuristicScorer;
use op_vault::{
    CardData, InMemoryVault, TokenFormat, TokenLifetime, TokenizationPolicy, Vault, VaultRef,
};

use crate::error::FfiError;

// ============================================================
// Opaque type aliases. These names mirror the bridge module but
// the C ABI exposes them as forward-declared `struct`s in the
// generated header.
// ============================================================

/// Opaque CardData handle.
pub struct OpCardData(CardData);

/// Opaque VaultRef handle.
pub struct OpVaultRef(VaultRef);

/// Opaque Vault handle.
pub struct OpVault(Arc<dyn Vault>);

/// Opaque HeuristicScorer handle.
pub struct OpScorer(HeuristicScorer);

/// Opaque tokenization policy struct (kept FFI-stable).
#[repr(C)]
#[derive(Copy, Clone)]
pub struct OpTokenizationPolicy {
    /// 0 = Random, 1 = Deterministic.
    pub format: u8,
    /// 0 = Reusable, 1 = SingleUse.
    pub lifetime: u8,
    /// 0 = no TTL, otherwise seconds.
    pub ttl_seconds: u64,
}

// ============================================================
// Thread-local last-error slot for the C ABI surface.
// ============================================================

thread_local! {
    static C_LAST_ERROR: std::cell::Cell<FfiError> = const {
        std::cell::Cell::new(FfiError::Ok)
    };
}

fn set_err(e: FfiError) {
    C_LAST_ERROR.with(|c| c.set(e));
}

/// Read the last error from the most recent C ABI call on this thread.
///
/// Returns the [`FfiError`] discriminant as `i32`. `0` is success.
#[unsafe(no_mangle)]
pub extern "C" fn op_last_error() -> i32 {
    C_LAST_ERROR.with(|c| c.get().as_i32())
}

// ============================================================
// String helpers
// ============================================================

/// Free a Rust-allocated string returned to C. Passing null is a no-op.
///
/// # Safety
/// The pointer must have been returned by a Rust function that
/// documents `op_string_free` as its disposer, or be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_string_free(p: *mut c_char) {
    if p.is_null() {
        return;
    }
    // SAFETY: p was allocated by CString::into_raw in this crate.
    let _ = unsafe { CString::from_raw(p) };
}

/// Build a CString from a Rust String and return its raw pointer.
/// Caller frees via [`op_string_free`].
fn rust_str_to_c(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(cs) => cs.into_raw(),
        Err(_) => {
            // Contains interior NULs — shouldn't happen for our outputs,
            // but defend.
            set_err(FfiError::Internal);
            core::ptr::null_mut()
        }
    }
}

/// Read a C string into a Rust &str. Returns None on null or non-UTF-8.
///
/// # Safety
/// `p` must be either null or a valid NUL-terminated UTF-8 string.
unsafe fn c_str_to_rust<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    // SAFETY: caller guarantees NUL-terminated UTF-8.
    let cs = unsafe { CStr::from_ptr(p) };
    cs.to_str().ok()
}

// ============================================================
// CardData
// ============================================================

/// Construct a [`OpCardData`] from a PAN and expiration. Returns null
/// on invalid input; call [`op_last_error`] for the reason.
///
/// # Safety
/// `pan` must be a valid NUL-terminated UTF-8 C string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_card_data_new(
    pan: *const c_char,
    exp_month: u8,
    exp_year: u16,
) -> *mut OpCardData {
    // SAFETY: caller upholds NUL-terminated UTF-8 invariant.
    let Some(pan_str) = (unsafe { c_str_to_rust(pan) }) else {
        set_err(FfiError::InvalidInput);
        return core::ptr::null_mut();
    };
    match CardData::new(pan_str.to_owned(), exp_month, exp_year) {
        Ok(card) => {
            set_err(FfiError::Ok);
            Box::into_raw(Box::new(OpCardData(card)))
        }
        Err(_) => {
            set_err(FfiError::InvalidInput);
            core::ptr::null_mut()
        }
    }
}

/// Free an [`OpCardData`]. Null is a no-op.
///
/// # Safety
/// Pointer must have come from [`op_card_data_new`] or another
/// crate-documented producer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_card_data_free(p: *mut OpCardData) {
    if p.is_null() {
        return;
    }
    // SAFETY: pointer originated from Box::into_raw.
    let _ = unsafe { Box::from_raw(p) };
}

/// First six digits. Caller frees via [`op_string_free`].
///
/// # Safety
/// `p` must be a valid `OpCardData` pointer (non-null).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_card_data_first_six(p: *const OpCardData) -> *mut c_char {
    if p.is_null() {
        set_err(FfiError::InvalidInput);
        return core::ptr::null_mut();
    }
    // SAFETY: caller guarantees valid pointer.
    let card = unsafe { &*p };
    rust_str_to_c(card.0.first_six().to_owned())
}

/// Last four digits.
///
/// # Safety
/// Same as [`op_card_data_first_six`].
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_card_data_last_four(p: *const OpCardData) -> *mut c_char {
    if p.is_null() {
        set_err(FfiError::InvalidInput);
        return core::ptr::null_mut();
    }
    // SAFETY: caller guarantees valid pointer.
    let card = unsafe { &*p };
    rust_str_to_c(card.0.last_four().to_owned())
}

/// Expiration month.
///
/// # Safety
/// Same as [`op_card_data_first_six`]. Returns 0 on null pointer
/// (which is not a valid month and indicates programmer error).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_card_data_exp_month(p: *const OpCardData) -> u8 {
    if p.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees valid pointer.
    unsafe { (*p).0.exp_month() }
}

/// Expiration year.
///
/// # Safety
/// Same as [`op_card_data_exp_month`]. Returns 0 on null pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_card_data_exp_year(p: *const OpCardData) -> u16 {
    if p.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees valid pointer.
    unsafe { (*p).0.exp_year() }
}

// ============================================================
// VaultRef
// ============================================================

/// Free a [`OpVaultRef`]. Null is a no-op.
///
/// # Safety
/// Pointer must have come from a crate-documented producer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_vault_ref_free(p: *mut OpVaultRef) {
    if p.is_null() {
        return;
    }
    // SAFETY: pointer originated from Box::into_raw.
    let _ = unsafe { Box::from_raw(p) };
}

/// String form of the token. Caller frees via [`op_string_free`].
///
/// # Safety
/// `p` must be a valid [`OpVaultRef`] pointer (non-null).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_vault_ref_as_string(p: *const OpVaultRef) -> *mut c_char {
    if p.is_null() {
        set_err(FfiError::InvalidInput);
        return core::ptr::null_mut();
    }
    // SAFETY: caller guarantees valid pointer.
    let vref = unsafe { &*p };
    rust_str_to_c(vref.0.as_str().to_owned())
}

/// Construct a [`OpVaultRef`] from a string. Used to reconstitute a
/// token from persistent storage (e.g. Core Data, user preferences).
///
/// # Safety
/// `s` must be a valid NUL-terminated UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_vault_ref_from_string(s: *const c_char) -> *mut OpVaultRef {
    // SAFETY: caller upholds NUL-terminated UTF-8 invariant.
    let Some(token) = (unsafe { c_str_to_rust(s) }) else {
        set_err(FfiError::InvalidInput);
        return core::ptr::null_mut();
    };
    set_err(FfiError::Ok);
    Box::into_raw(Box::new(OpVaultRef(VaultRef::new(token.to_owned()))))
}

// ============================================================
// Vault
// ============================================================

/// Construct an ephemeral in-memory vault. Caller frees via
/// [`op_vault_free`].
///
/// # Safety
/// `name` must be a valid NUL-terminated UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_vault_new_ephemeral(name: *const c_char) -> *mut OpVault {
    // SAFETY: caller upholds NUL-terminated UTF-8 invariant.
    let name_str = match unsafe { c_str_to_rust(name) } {
        Some(s) => s.to_owned(),
        None => "default".to_owned(),
    };
    set_err(FfiError::Ok);
    Box::into_raw(Box::new(OpVault(Arc::new(InMemoryVault::ephemeral(
        name_str,
    )))))
}

/// Free a vault. Null is a no-op. Note that this drops the underlying
/// `Arc`; if any cloned references survive (unlikely from C, possible
/// from Rust callers), the vault data persists until all are dropped.
///
/// # Safety
/// Pointer must have come from a crate-documented producer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_vault_free(p: *mut OpVault) {
    if p.is_null() {
        return;
    }
    // SAFETY: pointer originated from Box::into_raw.
    let _ = unsafe { Box::from_raw(p) };
}

/// Tokenize a card. **Consumes** the [`OpCardData`] regardless of
/// outcome — the caller must not free it. Returns null on error; call
/// [`op_last_error`].
///
/// # Safety
/// `vault` must be valid, `card` must be a valid pointer from
/// [`op_card_data_new`] that has not been freed.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_vault_tokenize(
    vault: *const OpVault,
    card: *mut OpCardData,
    policy: OpTokenizationPolicy,
) -> *mut OpVaultRef {
    if vault.is_null() || card.is_null() {
        set_err(FfiError::InvalidInput);
        // Still consume the card pointer if non-null to honor the
        // documented ownership contract.
        if !card.is_null() {
            // SAFETY: card is non-null and originated from Box::into_raw.
            let _ = unsafe { Box::from_raw(card) };
        }
        return core::ptr::null_mut();
    }
    // SAFETY: caller upholds validity invariants.
    let v = unsafe { &*vault };
    // SAFETY: consume the card box.
    let card_box = unsafe { Box::from_raw(card) };
    let p = decode_c_policy(policy);
    match v.0.tokenize(card_box.0, p) {
        Ok(vref) => {
            set_err(FfiError::Ok);
            Box::into_raw(Box::new(OpVaultRef(vref)))
        }
        Err(e) => {
            set_err(FfiError::from(e));
            core::ptr::null_mut()
        }
    }
}

/// Detokenize a token back into card data. Returns null on error.
///
/// # Safety
/// Both pointers must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_vault_detokenize(
    vault: *const OpVault,
    token: *const OpVaultRef,
) -> *mut OpCardData {
    if vault.is_null() || token.is_null() {
        set_err(FfiError::InvalidInput);
        return core::ptr::null_mut();
    }
    // SAFETY: caller upholds validity.
    let v = unsafe { &*vault };
    let t = unsafe { &*token };
    match v.0.detokenize(&t.0) {
        Ok(card) => {
            set_err(FfiError::Ok);
            Box::into_raw(Box::new(OpCardData(card)))
        }
        Err(e) => {
            set_err(FfiError::from(e));
            core::ptr::null_mut()
        }
    }
}

/// Probe existence. Returns 1 for exists, 0 for not, -1 for error.
///
/// # Safety
/// Both pointers must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_vault_exists(vault: *const OpVault, token: *const OpVaultRef) -> i32 {
    if vault.is_null() || token.is_null() {
        set_err(FfiError::InvalidInput);
        return -1;
    }
    // SAFETY: caller upholds validity.
    let v = unsafe { &*vault };
    let t = unsafe { &*token };
    match v.0.exists(&t.0) {
        Ok(true) => {
            set_err(FfiError::Ok);
            1
        }
        Ok(false) => {
            set_err(FfiError::Ok);
            0
        }
        Err(e) => {
            set_err(FfiError::from(e));
            -1
        }
    }
}

/// Delete a token. Returns 1 if removed, 0 if not present, -1 for error.
///
/// # Safety
/// Both pointers must be valid and non-null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_vault_delete(vault: *const OpVault, token: *const OpVaultRef) -> i32 {
    if vault.is_null() || token.is_null() {
        set_err(FfiError::InvalidInput);
        return -1;
    }
    // SAFETY: caller upholds validity.
    let v = unsafe { &*vault };
    let t = unsafe { &*token };
    match v.0.delete(&t.0) {
        Ok(true) => {
            set_err(FfiError::Ok);
            1
        }
        Ok(false) => {
            set_err(FfiError::Ok);
            0
        }
        Err(e) => {
            set_err(FfiError::from(e));
            -1
        }
    }
}

// ============================================================
// Scorer
// ============================================================

/// Construct the default heuristic scorer.
#[unsafe(no_mangle)]
pub extern "C" fn op_scorer_new_heuristic() -> *mut OpScorer {
    set_err(FfiError::Ok);
    Box::into_raw(Box::new(OpScorer(HeuristicScorer::new())))
}

/// Free a scorer.
///
/// # Safety
/// Pointer must have come from a crate-documented producer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_scorer_free(p: *mut OpScorer) {
    if p.is_null() {
        return;
    }
    // SAFETY: pointer originated from Box::into_raw.
    let _ = unsafe { Box::from_raw(p) };
}

/// Scorer name for telemetry. Caller frees via [`op_string_free`].
///
/// # Safety
/// `p` must be a valid scorer pointer (non-null).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_scorer_name(p: *const OpScorer) -> *mut c_char {
    use op_fraud::Scorer;
    if p.is_null() {
        set_err(FfiError::InvalidInput);
        return core::ptr::null_mut();
    }
    // SAFETY: caller guarantees valid pointer.
    let s = unsafe { &*p };
    rust_str_to_c(s.0.name().to_owned())
}

// ============================================================
// Policy decoding
// ============================================================

fn decode_c_policy(p: OpTokenizationPolicy) -> TokenizationPolicy {
    TokenizationPolicy {
        format: match p.format {
            1 => TokenFormat::Deterministic,
            _ => TokenFormat::Random,
        },
        lifetime: match p.lifetime {
            1 => TokenLifetime::SingleUse,
            _ => TokenLifetime::Reusable,
        },
        ttl_seconds: if p.ttl_seconds == 0 {
            None
        } else {
            Some(p.ttl_seconds)
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const VALID_VISA: &[u8] = b"4242424242424242\0";
    const NAME: &[u8] = b"c-test\0";

    fn default_c_policy() -> OpTokenizationPolicy {
        OpTokenizationPolicy {
            format: 0,
            lifetime: 0,
            ttl_seconds: 0,
        }
    }

    /// Helper to read a Rust-returned C string into a Rust `String`,
    /// freeing it as we go. Used in tests only.
    unsafe fn take_c_str(p: *mut c_char) -> String {
        assert!(!p.is_null(), "expected non-null string");
        let s = unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_owned();
        // SAFETY: we just received it from Rust.
        unsafe { op_string_free(p) };
        s
    }

    #[test]
    fn c_abi_card_lifecycle() {
        unsafe {
            let card = op_card_data_new(VALID_VISA.as_ptr() as *const c_char, 12, 2030);
            assert!(!card.is_null());

            let f6 = take_c_str(op_card_data_first_six(card));
            let l4 = take_c_str(op_card_data_last_four(card));
            assert_eq!(f6, "424242");
            assert_eq!(l4, "4242");
            assert_eq!(op_card_data_exp_month(card), 12);
            assert_eq!(op_card_data_exp_year(card), 2030);

            op_card_data_free(card);
        }
    }

    #[test]
    fn c_abi_card_new_with_invalid_pan_returns_null_and_sets_error() {
        unsafe {
            let bad = b"1111111111111111\0";
            let card = op_card_data_new(bad.as_ptr() as *const c_char, 12, 2030);
            assert!(card.is_null());
            assert_eq!(op_last_error(), FfiError::InvalidInput.as_i32());
        }
    }

    #[test]
    fn c_abi_card_new_with_null_pan_returns_null() {
        unsafe {
            let card = op_card_data_new(core::ptr::null(), 12, 2030);
            assert!(card.is_null());
            assert_eq!(op_last_error(), FfiError::InvalidInput.as_i32());
        }
    }

    #[test]
    fn c_abi_vault_round_trip() {
        unsafe {
            let vault = op_vault_new_ephemeral(NAME.as_ptr() as *const c_char);
            assert!(!vault.is_null());

            let card = op_card_data_new(VALID_VISA.as_ptr() as *const c_char, 12, 2030);
            assert!(!card.is_null());

            // tokenize consumes card.
            let token = op_vault_tokenize(vault, card, default_c_policy());
            assert!(!token.is_null());

            let token_str = take_c_str(op_vault_ref_as_string(token));
            assert!(token_str.starts_with("tok_v7_"));

            // detokenize.
            let recovered = op_vault_detokenize(vault, token);
            assert!(!recovered.is_null());
            let l4 = take_c_str(op_card_data_last_four(recovered));
            assert_eq!(l4, "4242");

            op_card_data_free(recovered);
            op_vault_ref_free(token);
            op_vault_free(vault);
        }
    }

    #[test]
    fn c_abi_detokenize_unknown_returns_null_and_sets_lookup_failed() {
        unsafe {
            let vault = op_vault_new_ephemeral(NAME.as_ptr() as *const c_char);
            let bad = b"tok_v7_nope\0";
            let token = op_vault_ref_from_string(bad.as_ptr() as *const c_char);
            let card = op_vault_detokenize(vault, token);
            assert!(card.is_null());
            assert_eq!(op_last_error(), FfiError::VaultLookupFailed.as_i32());
            op_vault_ref_free(token);
            op_vault_free(vault);
        }
    }

    #[test]
    fn c_abi_exists_and_delete() {
        unsafe {
            let vault = op_vault_new_ephemeral(NAME.as_ptr() as *const c_char);
            let card = op_card_data_new(VALID_VISA.as_ptr() as *const c_char, 12, 2030);
            let token = op_vault_tokenize(vault, card, default_c_policy());

            assert_eq!(op_vault_exists(vault, token), 1);
            assert_eq!(op_vault_delete(vault, token), 1);
            assert_eq!(op_vault_exists(vault, token), 0);
            // Idempotent.
            assert_eq!(op_vault_delete(vault, token), 0);

            op_vault_ref_free(token);
            op_vault_free(vault);
        }
    }

    #[test]
    fn c_abi_null_pointer_returns_negative_for_status_calls() {
        unsafe {
            assert_eq!(op_vault_exists(core::ptr::null(), core::ptr::null()), -1);
            assert_eq!(op_vault_delete(core::ptr::null(), core::ptr::null()), -1);
            assert_eq!(op_last_error(), FfiError::InvalidInput.as_i32());
        }
    }

    #[test]
    fn c_abi_scorer_lifecycle() {
        unsafe {
            let s = op_scorer_new_heuristic();
            assert!(!s.is_null());
            let name = take_c_str(op_scorer_name(s));
            assert_eq!(name, "heuristic-v1");
            op_scorer_free(s);
        }
    }

    #[test]
    fn c_abi_string_free_on_null_is_safe() {
        unsafe { op_string_free(core::ptr::null_mut()) };
    }

    #[test]
    fn c_abi_free_on_null_is_safe() {
        unsafe {
            op_card_data_free(core::ptr::null_mut());
            op_vault_ref_free(core::ptr::null_mut());
            op_vault_free(core::ptr::null_mut());
            op_scorer_free(core::ptr::null_mut());
        }
    }

    #[test]
    fn c_abi_tokenize_with_null_card_still_consumes_correctly() {
        unsafe {
            let vault = op_vault_new_ephemeral(NAME.as_ptr() as *const c_char);
            // Pass null card; should return null and set InvalidInput
            // without crashing on the consume path.
            let result = op_vault_tokenize(vault, core::ptr::null_mut(), default_c_policy());
            assert!(result.is_null());
            assert_eq!(op_last_error(), FfiError::InvalidInput.as_i32());
            op_vault_free(vault);
        }
    }

    #[test]
    fn c_abi_policy_decode_single_use_with_ttl() {
        let p = OpTokenizationPolicy {
            format: 0,
            lifetime: 1,
            ttl_seconds: 120,
        };
        let decoded = decode_c_policy(p);
        assert_eq!(decoded.lifetime, TokenLifetime::SingleUse);
        assert_eq!(decoded.ttl_seconds, Some(120));
        assert_eq!(decoded.format, TokenFormat::Random);
    }

    #[test]
    fn c_abi_policy_decode_unknown_format_falls_back_to_random() {
        let p = OpTokenizationPolicy {
            format: 99, // unknown
            lifetime: 0,
            ttl_seconds: 0,
        };
        assert_eq!(decode_c_policy(p).format, TokenFormat::Random);
    }
}
