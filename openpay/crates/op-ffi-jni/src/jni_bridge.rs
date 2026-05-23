//! JNI surface.
//!
//! Each function in this module is named `Java_dev_openpay_<class>_<method>`
//! per the JNI [resolution rules](https://docs.oracle.com/javase/8/docs/technotes/guides/jni/spec/design.html).
//! The JVM dispatches a `native` declaration in the Kotlin `dev.openpay.<class>`
//! class to the matching Rust function automatically by name. No
//! `RegisterNatives` call is needed.
//!
//! ## Handle protocol
//!
//! Kotlin classes hold a `Long` field (Rust `jlong`, signed 64-bit)
//! that is the pointer to a Rust heap allocation, cast to/from
//! `*mut T` via `Box::into_raw` / `Box::from_raw`. The Kotlin
//! `close()` / `finalize()` methods call back into Rust to free.
//!
//! ## Error reporting
//!
//! When a method fails, the Rust side calls `env.throw_new(class, msg)`
//! with the matching `dev/openpay/OpenPayException$Variant` class.
//! The Kotlin caller then sees a typed `OpenPayException` subclass
//! and pattern-matches via `when`.
//!
//! Per JNI conventions, throwing an exception **does not** abort
//! the native function — it just marks a pending exception. The
//! native function must return immediately after throwing (with a
//! sensible default value); the JVM raises the exception once
//! control returns to Java/Kotlin. Native code that wants to call
//! more JNI functions after throwing must first call
//! `env.exception_clear()`.

use std::sync::Arc;

use jni::JNIEnv;
use jni::objects::{JClass, JString};
use jni::sys::{JNI_FALSE, JNI_TRUE, jboolean, jbyte, jint, jlong, jshort, jstring};

use op_fraud::HeuristicScorer;
use op_vault::{
    CardData, InMemoryVault, TokenFormat, TokenLifetime, TokenizationPolicy, Vault, VaultRef,
};

use crate::error::FfiError;

// ============================================================
// Handle conversion helpers
// ============================================================
//
// A handle is just a non-zero `jlong`. We never expose null
// pointers as valid handles — Kotlin sees `0L` as "uninitialized"
// and refuses to call methods.

/// Convert a Box pointer to a JNI handle.
#[inline]
fn box_to_handle<T>(b: Box<T>) -> jlong {
    Box::into_raw(b) as jlong
}

/// Borrow a handle as `&T`. Returns `None` if the handle is zero.
///
/// # Safety
/// The handle must point to a valid `T` allocated via `box_to_handle`,
/// not yet freed by `handle_drop`.
unsafe fn handle_as_ref<'a, T>(h: jlong) -> Option<&'a T> {
    if h == 0 {
        return None;
    }
    // SAFETY: caller upholds validity invariant. Cast `jlong` -> `*const T`.
    Some(unsafe { &*(h as *const T) })
}

/// Drop the boxed value behind a handle. Zero handles are a no-op.
///
/// # Safety
/// The handle must either be zero or point to a valid `T` from
/// `box_to_handle`. Double-drop is undefined.
unsafe fn handle_drop<T>(h: jlong) {
    if h == 0 {
        return;
    }
    // SAFETY: caller upholds validity invariant.
    let _ = unsafe { Box::from_raw(h as *mut T) };
}

/// Consume a handle (drop the box and return the inner value).
///
/// # Safety
/// Same as `handle_drop`.
unsafe fn handle_take<T>(h: jlong) -> Option<T> {
    if h == 0 {
        return None;
    }
    // SAFETY: caller upholds validity invariant.
    let b: Box<T> = unsafe { Box::from_raw(h as *mut T) };
    Some(*b)
}

// ============================================================
// Exception helpers
// ============================================================

/// Throw a typed `OpenPayException` subclass. Idempotent — if the
/// class can't be found (e.g. running outside a JVM that has the
/// Kotlin classes loaded), falls back to a `RuntimeException`.
fn throw_ffi_error(env: &mut JNIEnv, e: FfiError, msg: &str) {
    let cls = e.exception_class();
    if env.throw_new(cls, msg).is_err() {
        // The targeted class wasn't found. Clear any pending exception
        // and try the JVM's built-in RuntimeException as last resort.
        let _ = env.exception_clear();
        let _ = env.throw_new("java/lang/RuntimeException", msg);
    }
}

// ============================================================
// CardData
// ============================================================

/// `dev.openpay.CardData.nativeNew(pan: String, expMonth: Byte, expYear: Short): Long`
///
/// Validates Luhn + length + expiration. Throws
/// `OpenPayException$InvalidInput` on failure.
///
/// # Safety
/// This is a JNI entry point. The JVM provides valid arguments.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_dev_openpay_CardData_nativeNew(
    mut env: JNIEnv,
    _class: JClass,
    pan: JString,
    exp_month: jbyte,
    exp_year: jshort,
) -> jlong {
    let Ok(pan_str) = env.get_string(&pan) else {
        throw_ffi_error(
            &mut env,
            FfiError::InvalidInput,
            "PAN must be a valid string",
        );
        return 0;
    };
    let pan_rust: String = pan_str.into();

    // JNI gives us signed types where Kotlin's UByte/UShort would be
    // unsigned. Convert with bounds checking; the validation in
    // CardData::new will catch out-of-range anyway, but we surface
    // the right error class here.
    if exp_month < 0 || exp_year < 0 {
        throw_ffi_error(
            &mut env,
            FfiError::InvalidInput,
            "expiration month/year must be non-negative",
        );
        return 0;
    }
    let exp_month_u8 = exp_month as u8;
    let exp_year_u16 = exp_year as u16;

    match CardData::new(pan_rust, exp_month_u8, exp_year_u16) {
        Ok(card) => box_to_handle(Box::new(card)),
        Err(e) => {
            let ffi: FfiError = e.into();
            throw_ffi_error(&mut env, ffi, "invalid card data");
            0
        }
    }
}

/// `dev.openpay.CardData.nativeFree(handle: Long)`
///
/// # Safety
/// Handle must come from `nativeNew` and not yet have been freed.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_dev_openpay_CardData_nativeFree(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    // SAFETY: caller upholds Kotlin-side handle invariant.
    unsafe { handle_drop::<CardData>(handle) };
}

/// `dev.openpay.CardData.nativeFirstSix(handle: Long): String`
///
/// # Safety
/// Handle must be a live `CardData`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_dev_openpay_CardData_nativeFirstSix(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
) -> jstring {
    // SAFETY: caller upholds validity.
    let Some(card) = (unsafe { handle_as_ref::<CardData>(handle) }) else {
        throw_ffi_error(&mut env, FfiError::InvalidInput, "null CardData handle");
        return std::ptr::null_mut();
    };
    match env.new_string(card.first_six()) {
        Ok(s) => s.into_raw(),
        Err(_) => {
            throw_ffi_error(&mut env, FfiError::Internal, "JNI string allocation failed");
            std::ptr::null_mut()
        }
    }
}

/// `dev.openpay.CardData.nativeLastFour(handle: Long): String`
///
/// # Safety
/// Handle must be a live `CardData`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_dev_openpay_CardData_nativeLastFour(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
) -> jstring {
    // SAFETY: caller upholds validity.
    let Some(card) = (unsafe { handle_as_ref::<CardData>(handle) }) else {
        throw_ffi_error(&mut env, FfiError::InvalidInput, "null CardData handle");
        return std::ptr::null_mut();
    };
    match env.new_string(card.last_four()) {
        Ok(s) => s.into_raw(),
        Err(_) => {
            throw_ffi_error(&mut env, FfiError::Internal, "JNI string allocation failed");
            std::ptr::null_mut()
        }
    }
}

/// `dev.openpay.CardData.nativeExpMonth(handle: Long): Byte`
///
/// # Safety
/// Handle must be a live `CardData`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_dev_openpay_CardData_nativeExpMonth(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) -> jbyte {
    // SAFETY: caller upholds validity.
    let Some(card) = (unsafe { handle_as_ref::<CardData>(handle) }) else {
        return 0;
    };
    card.exp_month() as jbyte
}

/// `dev.openpay.CardData.nativeExpYear(handle: Long): Short`
///
/// # Safety
/// Handle must be a live `CardData`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_dev_openpay_CardData_nativeExpYear(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) -> jshort {
    // SAFETY: caller upholds validity.
    let Some(card) = (unsafe { handle_as_ref::<CardData>(handle) }) else {
        return 0;
    };
    card.exp_year() as jshort
}

// ============================================================
// VaultRef
// ============================================================

/// `dev.openpay.VaultRef.nativeFromString(token: String): Long`
///
/// Wrap a token string into a VaultRef handle. No validation — the
/// vault rejects malformed tokens on detokenize.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_dev_openpay_VaultRef_nativeFromString(
    mut env: JNIEnv,
    _class: JClass,
    token: JString,
) -> jlong {
    let Ok(token_str) = env.get_string(&token) else {
        throw_ffi_error(
            &mut env,
            FfiError::InvalidInput,
            "token must be a valid string",
        );
        return 0;
    };
    let token_rust: String = token_str.into();
    box_to_handle(Box::new(VaultRef::new(token_rust)))
}

/// `dev.openpay.VaultRef.nativeFree(handle: Long)`
///
/// # Safety
/// Handle must come from `nativeFromString` or a vault method.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_dev_openpay_VaultRef_nativeFree(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    // SAFETY: caller upholds Kotlin-side invariant.
    unsafe { handle_drop::<VaultRef>(handle) };
}

/// `dev.openpay.VaultRef.nativeAsString(handle: Long): String`
///
/// # Safety
/// Handle must be a live `VaultRef`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_dev_openpay_VaultRef_nativeAsString(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
) -> jstring {
    // SAFETY: caller upholds validity.
    let Some(vref) = (unsafe { handle_as_ref::<VaultRef>(handle) }) else {
        throw_ffi_error(&mut env, FfiError::InvalidInput, "null VaultRef handle");
        return std::ptr::null_mut();
    };
    match env.new_string(vref.as_str()) {
        Ok(s) => s.into_raw(),
        Err(_) => {
            throw_ffi_error(&mut env, FfiError::Internal, "JNI string allocation failed");
            std::ptr::null_mut()
        }
    }
}

// ============================================================
// Vault
// ============================================================

/// Internal handle type for the vault — an `Arc<dyn Vault>` boxed.
type VaultHandle = Arc<dyn Vault>;

/// `dev.openpay.Vault.nativeNewEphemeral(name: String): Long`
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_dev_openpay_Vault_nativeNewEphemeral(
    mut env: JNIEnv,
    _class: JClass,
    name: JString,
) -> jlong {
    let name_str: String = match env.get_string(&name) {
        Ok(s) => s.into(),
        Err(_) => "default".to_owned(),
    };
    let arc: VaultHandle = Arc::new(InMemoryVault::ephemeral(name_str));
    box_to_handle(Box::new(arc))
}

/// `dev.openpay.Vault.nativeFree(handle: Long)`
///
/// # Safety
/// Handle must come from `nativeNewEphemeral`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_dev_openpay_Vault_nativeFree(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    // SAFETY: caller upholds Kotlin-side invariant.
    unsafe { handle_drop::<VaultHandle>(handle) };
}

/// `dev.openpay.Vault.nativeTokenize(vault: Long, card: Long, format: Int, lifetime: Int, ttlSeconds: Long): Long`
///
/// **Consumes** the `card` handle regardless of outcome — Kotlin
/// must set its handle field to 0 immediately after calling.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_dev_openpay_Vault_nativeTokenize(
    mut env: JNIEnv,
    _class: JClass,
    vault: jlong,
    card: jlong,
    format: jint,
    lifetime: jint,
    ttl_seconds: jlong,
) -> jlong {
    // SAFETY: caller upholds vault-handle validity.
    let Some(arc_ref) = (unsafe { handle_as_ref::<VaultHandle>(vault) }) else {
        // Even on error we must consume the card.
        // SAFETY: card handle invariant.
        let _ = unsafe { handle_take::<CardData>(card) };
        throw_ffi_error(&mut env, FfiError::InvalidInput, "null Vault handle");
        return 0;
    };
    let v = arc_ref.clone();
    // SAFETY: card handle invariant.
    let Some(card_data) = (unsafe { handle_take::<CardData>(card) }) else {
        throw_ffi_error(&mut env, FfiError::InvalidInput, "null CardData handle");
        return 0;
    };

    let policy = decode_policy(format, lifetime, ttl_seconds);
    match v.tokenize(card_data, policy) {
        Ok(vref) => box_to_handle(Box::new(vref)),
        Err(e) => {
            let ffi: FfiError = e.into();
            throw_ffi_error(&mut env, ffi, "tokenize failed");
            0
        }
    }
}

/// `dev.openpay.Vault.nativeDetokenize(vault: Long, token: Long): Long`
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_dev_openpay_Vault_nativeDetokenize(
    mut env: JNIEnv,
    _class: JClass,
    vault: jlong,
    token: jlong,
) -> jlong {
    // SAFETY: caller upholds validity.
    let Some(arc_ref) = (unsafe { handle_as_ref::<VaultHandle>(vault) }) else {
        throw_ffi_error(&mut env, FfiError::InvalidInput, "null Vault handle");
        return 0;
    };
    let v = arc_ref.clone();
    // SAFETY: token handle invariant.
    let Some(vref) = (unsafe { handle_as_ref::<VaultRef>(token) }) else {
        throw_ffi_error(&mut env, FfiError::InvalidInput, "null VaultRef handle");
        return 0;
    };
    match v.detokenize(vref) {
        Ok(card) => box_to_handle(Box::new(card)),
        Err(e) => {
            let ffi: FfiError = e.into();
            throw_ffi_error(&mut env, ffi, "detokenize failed");
            0
        }
    }
}

/// `dev.openpay.Vault.nativeExists(vault: Long, token: Long): Boolean`
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_dev_openpay_Vault_nativeExists(
    mut env: JNIEnv,
    _class: JClass,
    vault: jlong,
    token: jlong,
) -> jboolean {
    // SAFETY: caller upholds validity.
    let Some(arc_ref) = (unsafe { handle_as_ref::<VaultHandle>(vault) }) else {
        throw_ffi_error(&mut env, FfiError::InvalidInput, "null Vault handle");
        return JNI_FALSE;
    };
    let v = arc_ref.clone();
    // SAFETY: token handle invariant.
    let Some(vref) = (unsafe { handle_as_ref::<VaultRef>(token) }) else {
        throw_ffi_error(&mut env, FfiError::InvalidInput, "null VaultRef handle");
        return JNI_FALSE;
    };
    match v.exists(vref) {
        Ok(true) => JNI_TRUE,
        Ok(false) => JNI_FALSE,
        Err(e) => {
            let ffi: FfiError = e.into();
            throw_ffi_error(&mut env, ffi, "exists check failed");
            JNI_FALSE
        }
    }
}

/// `dev.openpay.Vault.nativeDelete(vault: Long, token: Long): Boolean`
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_dev_openpay_Vault_nativeDelete(
    mut env: JNIEnv,
    _class: JClass,
    vault: jlong,
    token: jlong,
) -> jboolean {
    // SAFETY: caller upholds validity.
    let Some(arc_ref) = (unsafe { handle_as_ref::<VaultHandle>(vault) }) else {
        throw_ffi_error(&mut env, FfiError::InvalidInput, "null Vault handle");
        return JNI_FALSE;
    };
    let v = arc_ref.clone();
    // SAFETY: token handle invariant.
    let Some(vref) = (unsafe { handle_as_ref::<VaultRef>(token) }) else {
        throw_ffi_error(&mut env, FfiError::InvalidInput, "null VaultRef handle");
        return JNI_FALSE;
    };
    match v.delete(vref) {
        Ok(true) => JNI_TRUE,
        Ok(false) => JNI_FALSE,
        Err(e) => {
            let ffi: FfiError = e.into();
            throw_ffi_error(&mut env, ffi, "delete failed");
            JNI_FALSE
        }
    }
}

// ============================================================
// Scorer
// ============================================================

/// `dev.openpay.HeuristicScorer.nativeNew(): Long`
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_dev_openpay_HeuristicScorer_nativeNew(
    _env: JNIEnv,
    _class: JClass,
) -> jlong {
    box_to_handle(Box::new(HeuristicScorer::new()))
}

/// `dev.openpay.HeuristicScorer.nativeFree(handle: Long)`
///
/// # Safety
/// Handle must come from `nativeNew`.
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_dev_openpay_HeuristicScorer_nativeFree(
    _env: JNIEnv,
    _class: JClass,
    handle: jlong,
) {
    // SAFETY: caller upholds Kotlin-side invariant.
    unsafe { handle_drop::<HeuristicScorer>(handle) };
}

/// `dev.openpay.HeuristicScorer.nativeName(handle: Long): String`
#[unsafe(no_mangle)]
pub unsafe extern "system" fn Java_dev_openpay_HeuristicScorer_nativeName(
    mut env: JNIEnv,
    _class: JClass,
    handle: jlong,
) -> jstring {
    use op_fraud::Scorer;
    // SAFETY: caller upholds validity.
    let Some(scorer) = (unsafe { handle_as_ref::<HeuristicScorer>(handle) }) else {
        throw_ffi_error(
            &mut env,
            FfiError::InvalidInput,
            "null HeuristicScorer handle",
        );
        return std::ptr::null_mut();
    };
    match env.new_string(scorer.name()) {
        Ok(s) => s.into_raw(),
        Err(_) => {
            throw_ffi_error(&mut env, FfiError::Internal, "JNI string allocation failed");
            std::ptr::null_mut()
        }
    }
}

// ============================================================
// Policy decoding
// ============================================================

fn decode_policy(format: jint, lifetime: jint, ttl_seconds: jlong) -> TokenizationPolicy {
    TokenizationPolicy {
        format: match format {
            1 => TokenFormat::Deterministic,
            _ => TokenFormat::Random,
        },
        lifetime: match lifetime {
            1 => TokenLifetime::SingleUse,
            _ => TokenLifetime::Reusable,
        },
        ttl_seconds: if ttl_seconds <= 0 {
            None
        } else {
            Some(ttl_seconds as u64)
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // We can't drive the JNI surface without a live JVM. Instead we
    // test the policy decoder and handle helpers that are pure Rust.

    #[test]
    fn decode_policy_default_is_random_reusable() {
        let p = decode_policy(0, 0, 0);
        assert_eq!(p.format, TokenFormat::Random);
        assert_eq!(p.lifetime, TokenLifetime::Reusable);
        assert_eq!(p.ttl_seconds, None);
    }

    #[test]
    fn decode_policy_single_use_with_ttl() {
        let p = decode_policy(0, 1, 120);
        assert_eq!(p.format, TokenFormat::Random);
        assert_eq!(p.lifetime, TokenLifetime::SingleUse);
        assert_eq!(p.ttl_seconds, Some(120));
    }

    #[test]
    fn decode_policy_deterministic() {
        let p = decode_policy(1, 0, 0);
        assert_eq!(p.format, TokenFormat::Deterministic);
    }

    #[test]
    fn decode_policy_unknown_format_falls_back_to_random() {
        let p = decode_policy(99, 0, 0);
        assert_eq!(p.format, TokenFormat::Random);
    }

    #[test]
    fn decode_policy_unknown_lifetime_falls_back_to_reusable() {
        let p = decode_policy(0, 99, 0);
        assert_eq!(p.lifetime, TokenLifetime::Reusable);
    }

    #[test]
    fn decode_policy_negative_ttl_treated_as_no_ttl() {
        // Kotlin Long can be negative; treat anything <= 0 as no TTL.
        let p = decode_policy(0, 0, -1);
        assert_eq!(p.ttl_seconds, None);
    }

    #[test]
    fn handle_round_trip_via_box() {
        // Verify the box → jlong → box conversion on a non-JNI value.
        let h = box_to_handle(Box::new(42_i32));
        assert_ne!(h, 0);

        // SAFETY: we just allocated this handle and have not freed.
        let n: Option<&i32> = unsafe { handle_as_ref::<i32>(h) };
        assert_eq!(n, Some(&42));

        // SAFETY: same handle, valid, taking ownership.
        let taken: Option<i32> = unsafe { handle_take::<i32>(h) };
        assert_eq!(taken, Some(42));
    }

    #[test]
    fn handle_zero_returns_none_safely() {
        // SAFETY: we test the zero case, which is the safe-by-design
        // path. No allocation occurred.
        let n: Option<&i32> = unsafe { handle_as_ref::<i32>(0) };
        assert_eq!(n, None);

        // SAFETY: zero handle, no allocation.
        let t: Option<i32> = unsafe { handle_take::<i32>(0) };
        assert_eq!(t, None);
    }

    #[test]
    fn handle_drop_zero_is_safe() {
        // SAFETY: zero handle, no allocation to free.
        unsafe { handle_drop::<i32>(0) };
    }

    #[test]
    fn handle_drop_actual_allocation() {
        let h = box_to_handle(Box::new(String::from("test")));
        // SAFETY: allocated above, not yet freed.
        unsafe { handle_drop::<String>(h) };
        // We can't safely read `h` again; the box is dropped.
    }
}
