//! Plain C ABI surface, identical to the one shipped in
//! `op-ffi-swift::c_api`.
//!
//! The function names, types, and ownership rules are deliberately
//! the same. This lets a single hand-rolled NDK wrapper (in C or
//! C++) drive both the iOS and Android builds — important for shops
//! that maintain a unified cross-platform C++ codebase.
//!
//! See `op-ffi-swift/src/c_api.rs` for the full documentation of the
//! ownership protocol. Summary:
//!
//! - `op_*_new` / `op_*_create` returns a `*mut T` allocated with
//!   `Box::into_raw`. Caller frees via `op_*_free`.
//! - `op_*_free` on null is a no-op.
//! - `op_vault_tokenize` **consumes** the `OpCardData` pointer.
//! - Status-returning calls return `i32` (0/+ ok, -1 error).
//! - Pointer-returning calls return null on error.
//! - All strings are NUL-terminated UTF-8, freed via `op_string_free`.
//!
//! The thread-local last-error slot is **separate** from any JNI-
//! surface thread-local. A failure on the JNI side does not affect
//! the C ABI's last-error reading.

use core::ffi::{CStr, c_char};
use std::ffi::CString;
use std::sync::Arc;

use op_fraud::HeuristicScorer;
use op_vault::{
    CardData, InMemoryVault, TokenFormat, TokenLifetime, TokenizationPolicy, Vault, VaultRef,
};

use crate::error::FfiError;

/// Opaque CardData handle.
pub struct OpCardData(pub(crate) CardData);

/// Opaque VaultRef handle.
pub struct OpVaultRef(pub(crate) VaultRef);

/// Opaque Vault handle.
pub struct OpVault(pub(crate) Arc<dyn Vault>);

/// Opaque HeuristicScorer handle.
pub struct OpScorer(pub(crate) HeuristicScorer);

/// FFI-stable tokenization policy struct.
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
// Thread-local last error
// ============================================================

thread_local! {
    static C_LAST_ERROR: std::cell::Cell<FfiError> = const {
        std::cell::Cell::new(FfiError::Ok)
    };
}

pub(crate) fn set_err(e: FfiError) {
    C_LAST_ERROR.with(|c| c.set(e));
}

/// Read the last C-ABI error on this thread.
#[unsafe(no_mangle)]
pub extern "C" fn op_last_error() -> i32 {
    C_LAST_ERROR.with(|c| c.get().as_i32())
}

// ============================================================
// String helpers
// ============================================================

/// Free a Rust-allocated C string. Null is a no-op.
///
/// # Safety
/// `p` must have been returned by a Rust function in this crate or be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_string_free(p: *mut c_char) {
    if p.is_null() {
        return;
    }
    // SAFETY: p originated from CString::into_raw in this crate.
    let _ = unsafe { CString::from_raw(p) };
}

pub(crate) fn rust_str_to_c(s: String) -> *mut c_char {
    match CString::new(s) {
        Ok(cs) => cs.into_raw(),
        Err(_) => {
            set_err(FfiError::Internal);
            core::ptr::null_mut()
        }
    }
}

/// # Safety
/// `p` must be null or a valid NUL-terminated UTF-8 string.
pub(crate) unsafe fn c_str_to_rust<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    // SAFETY: caller upholds NUL-terminated UTF-8.
    let cs = unsafe { CStr::from_ptr(p) };
    cs.to_str().ok()
}

// ============================================================
// CardData
// ============================================================

/// # Safety
/// `pan` must be a valid NUL-terminated UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_card_data_new(
    pan: *const c_char,
    exp_month: u8,
    exp_year: u16,
) -> *mut OpCardData {
    // SAFETY: caller upholds NUL-terminated UTF-8.
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

/// # Safety
/// `p` must have come from `op_card_data_new` or be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_card_data_free(p: *mut OpCardData) {
    if p.is_null() {
        return;
    }
    // SAFETY: p originated from Box::into_raw.
    let _ = unsafe { Box::from_raw(p) };
}

/// # Safety
/// `p` must be a valid non-null `OpCardData` pointer.
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

/// # Safety
/// `p` must be a valid non-null `OpCardData` pointer.
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

/// # Safety
/// `p` must be a valid non-null `OpCardData` pointer.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_card_data_exp_month(p: *const OpCardData) -> u8 {
    if p.is_null() {
        return 0;
    }
    // SAFETY: caller guarantees valid pointer.
    unsafe { (*p).0.exp_month() }
}

/// # Safety
/// `p` must be a valid non-null `OpCardData` pointer.
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

/// # Safety
/// `p` must have come from a crate-documented producer or be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_vault_ref_free(p: *mut OpVaultRef) {
    if p.is_null() {
        return;
    }
    // SAFETY: p originated from Box::into_raw.
    let _ = unsafe { Box::from_raw(p) };
}

/// # Safety
/// `p` must be a valid non-null `OpVaultRef` pointer.
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

/// # Safety
/// `s` must be a valid NUL-terminated UTF-8 string.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_vault_ref_from_string(s: *const c_char) -> *mut OpVaultRef {
    // SAFETY: caller upholds NUL-terminated UTF-8.
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

/// # Safety
/// `name` must be a valid NUL-terminated UTF-8 string or null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_vault_new_ephemeral(name: *const c_char) -> *mut OpVault {
    // SAFETY: caller upholds NUL-terminated UTF-8.
    let name_str = match unsafe { c_str_to_rust(name) } {
        Some(s) => s.to_owned(),
        None => "default".to_owned(),
    };
    set_err(FfiError::Ok);
    Box::into_raw(Box::new(OpVault(Arc::new(InMemoryVault::ephemeral(
        name_str,
    )))))
}

/// # Safety
/// `p` must have come from a crate-documented producer or be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_vault_free(p: *mut OpVault) {
    if p.is_null() {
        return;
    }
    // SAFETY: p originated from Box::into_raw.
    let _ = unsafe { Box::from_raw(p) };
}

/// # Safety
/// `vault` must be a valid non-null pointer; `card` must be a valid
/// non-null pointer from `op_card_data_new` that has not been freed.
/// `card` is consumed regardless of outcome.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_vault_tokenize(
    vault: *const OpVault,
    card: *mut OpCardData,
    policy: OpTokenizationPolicy,
) -> *mut OpVaultRef {
    if vault.is_null() || card.is_null() {
        set_err(FfiError::InvalidInput);
        if !card.is_null() {
            // SAFETY: non-null and from Box::into_raw.
            let _ = unsafe { Box::from_raw(card) };
        }
        return core::ptr::null_mut();
    }
    // SAFETY: caller upholds validity.
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

/// Construct a heuristic fraud scorer. The returned pointer is owned
/// by the caller and must be released with `op_scorer_free`.
#[unsafe(no_mangle)]
pub extern "C" fn op_scorer_new_heuristic() -> *mut OpScorer {
    set_err(FfiError::Ok);
    Box::into_raw(Box::new(OpScorer(HeuristicScorer::new())))
}

/// # Safety
/// `p` must have come from `op_scorer_new_heuristic` or be null.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn op_scorer_free(p: *mut OpScorer) {
    if p.is_null() {
        return;
    }
    // SAFETY: p originated from Box::into_raw.
    let _ = unsafe { Box::from_raw(p) };
}

/// # Safety
/// `p` must be a valid non-null pointer.
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
// Policy decode
// ============================================================

pub(crate) fn decode_c_policy(p: OpTokenizationPolicy) -> TokenizationPolicy {
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

    fn default_policy() -> OpTokenizationPolicy {
        OpTokenizationPolicy {
            format: 0,
            lifetime: 0,
            ttl_seconds: 0,
        }
    }

    unsafe fn take_c_str(p: *mut c_char) -> String {
        assert!(!p.is_null());
        let s = unsafe { CStr::from_ptr(p) }.to_str().unwrap().to_owned();
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
    fn c_abi_invalid_pan_returns_null() {
        unsafe {
            let bad = b"1111111111111111\0";
            let card = op_card_data_new(bad.as_ptr() as *const c_char, 12, 2030);
            assert!(card.is_null());
            assert_eq!(op_last_error(), FfiError::InvalidInput.as_i32());
        }
    }

    #[test]
    fn c_abi_null_pan_returns_null() {
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
            let card = op_card_data_new(VALID_VISA.as_ptr() as *const c_char, 12, 2030);
            let token = op_vault_tokenize(vault, card, default_policy());
            assert!(!token.is_null());
            let s = take_c_str(op_vault_ref_as_string(token));
            assert!(s.starts_with("tok_v7_"));
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
    fn c_abi_detokenize_unknown_collapses_to_lookup_failed() {
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
            let token = op_vault_tokenize(vault, card, default_policy());
            assert_eq!(op_vault_exists(vault, token), 1);
            assert_eq!(op_vault_delete(vault, token), 1);
            assert_eq!(op_vault_exists(vault, token), 0);
            assert_eq!(op_vault_delete(vault, token), 0);
            op_vault_ref_free(token);
            op_vault_free(vault);
        }
    }

    #[test]
    fn c_abi_null_pointer_status_calls() {
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
            let name = take_c_str(op_scorer_name(s));
            assert_eq!(name, "heuristic-v1");
            op_scorer_free(s);
        }
    }

    #[test]
    fn c_abi_string_free_null_safe() {
        unsafe { op_string_free(core::ptr::null_mut()) };
    }

    #[test]
    fn c_abi_free_null_safe() {
        unsafe {
            op_card_data_free(core::ptr::null_mut());
            op_vault_ref_free(core::ptr::null_mut());
            op_vault_free(core::ptr::null_mut());
            op_scorer_free(core::ptr::null_mut());
        }
    }

    #[test]
    fn c_abi_tokenize_null_card_consumes_correctly() {
        unsafe {
            let vault = op_vault_new_ephemeral(NAME.as_ptr() as *const c_char);
            let result = op_vault_tokenize(vault, core::ptr::null_mut(), default_policy());
            assert!(result.is_null());
            assert_eq!(op_last_error(), FfiError::InvalidInput.as_i32());
            op_vault_free(vault);
        }
    }

    #[test]
    fn c_abi_policy_decode_single_use() {
        let p = OpTokenizationPolicy {
            format: 0,
            lifetime: 1,
            ttl_seconds: 120,
        };
        let d = decode_c_policy(p);
        assert_eq!(d.lifetime, TokenLifetime::SingleUse);
        assert_eq!(d.ttl_seconds, Some(120));
    }

    #[test]
    fn c_abi_policy_decode_unknown_format_random() {
        let p = OpTokenizationPolicy {
            format: 99,
            lifetime: 0,
            ttl_seconds: 0,
        };
        assert_eq!(decode_c_policy(p).format, TokenFormat::Random);
    }
}
