//! The `swift-bridge` declaration that defines the Swift-facing API.
//!
//! The macro processes this module at build time and emits both a
//! `OpenPay.swift` file and a C header. Swift apps import the
//! generated module; everything inside this file becomes either a
//! shared struct/enum (visible to both languages) or an opaque
//! Rust/Swift type (visible behind a pointer).
//!
//! ## Layered surface
//!
//! - **Money / Currency / PaymentMethod kinds**: shared structs/enums,
//!   transparent on both sides.
//! - **CardData / VaultRef / FraudDecision**: opaque Rust types,
//!   constructed via factory functions, methods called on Swift-held
//!   handles. Swift sees them as classes; drop semantics are wired by
//!   `swift-bridge`'s ownership protocol.
//! - **Vault / Scorer**: opaque Rust types holding `Arc<dyn Vault>`
//!   and `Arc<dyn Scorer>` respectively. Swift constructs an
//!   in-memory vault (for development) or wraps an
//!   operator-provided implementation that delegates to iOS Keychain.
//!
//! ## Why some things are not bridged
//!
//! The card and A2A rails are not exposed here. iOS apps that need to
//! submit a payment do so via a hosted SDK (Stripe SDK, Adyen SDK,
//! etc.) that has its own bridge; OpenPay's role on iOS is to
//! orchestrate vault + fraud + token handoff. The rail crates remain
//! Rust-only.

/// The bridge module. `swift-bridge` parses this at build time.
#[swift_bridge::bridge]
mod ffi {
    // ============================================================
    // Shared enums (visible to both languages, FFI-stable layout)
    // swift-bridge 0.1.59 doesn't accept attributes — including doc
    // comments — on shared enums/structs inside the bridge module.
    // Keep the documentation on the corresponding Rust impl types
    // below the `mod ffi` block.
    // ============================================================

    enum FraudDecisionFfi {
        Approve,
        Review,
        Decline,
        Freeze,
    }

    enum TokenFormatFfi {
        Random,
        Deterministic,
    }

    enum TokenLifetimeFfi {
        Reusable,
        SingleUse,
    }

    // ============================================================
    // Shared structs (transparent on both sides)
    // ============================================================

    #[swift_bridge(swift_repr = "struct")]
    struct TokenizationPolicyFfi {
        format: TokenFormatFfi,
        lifetime: TokenLifetimeFfi,
        ttl_seconds: u64,
    }

    // ============================================================
    // Opaque Rust types exported to Swift
    // ============================================================

    // swift-bridge 0.1.59 infers the receiver type of a `&self` method
    // from the single opaque `type` declared in its OWN extern block.
    // One block per opaque type keeps that inference unambiguous; free
    // functions get their own block with explicit-typed params.

    // ---------- CardData ----------
    extern "Rust" {
        type RustCardData;

        // Construct from PAN + expiration. Validates length, Luhn,
        // and expiration sanity. Returns `nil` on invalid input.
        #[swift_bridge(associated_to = RustCardData)]
        fn new(pan: &str, exp_month: u8, exp_year: u16) -> Option<RustCardData>;

        // First six digits (BIN), safe to display.
        fn first_six(self: &RustCardData) -> String;

        // Last four digits, safe to display.
        fn last_four(self: &RustCardData) -> String;

        // Expiration month (1-12).
        fn exp_month(self: &RustCardData) -> u8;

        // Expiration year (4-digit).
        fn exp_year(self: &RustCardData) -> u16;
    }

    // ---------- VaultRef ----------
    extern "Rust" {
        type RustVaultRef;

        // The opaque token id as a String. Safe to log, store, or
        // transmit — it carries no PAN information.
        fn as_string(self: &RustVaultRef) -> String;
    }

    // ---------- Vault ----------
    extern "Rust" {
        type RustVault;

        // Construct an ephemeral in-memory vault. Useful for
        // development and tests. Production deployments wire up a
        // platform vault and skip this factory entirely.
        #[swift_bridge(associated_to = RustVault)]
        fn ephemeral(name: &str) -> RustVault;

        // Tokenize a card under the given policy. Returns the
        // resulting RustVaultRef or `nil` on error. Takes ownership
        // of `card`; the Swift handle is consumed on this call.
        fn tokenize(
            self: &RustVault,
            card: RustCardData,
            policy: TokenizationPolicyFfi,
        ) -> Option<RustVaultRef>;

        // Detokenize. Returns `nil` on error (token unknown, expired,
        // consumed, or auth failed — collapsed for oracle discipline).
        fn detokenize(self: &RustVault, token: &RustVaultRef) -> Option<RustCardData>;

        // Existence probe. Returns `true` only if a token exists in
        // the vault and is in a usable state.
        fn exists(self: &RustVault, token: &RustVaultRef) -> bool;

        // Delete a token. Returns `true` if a mapping was removed,
        // `false` if the token wasn't present. Idempotent.
        fn delete(self: &RustVault, token: &RustVaultRef) -> bool;
    }

    // ---------- Scorer ----------
    extern "Rust" {
        type RustHeuristicScorer;

        // Construct the default heuristic scorer (no model load).
        #[swift_bridge(associated_to = RustHeuristicScorer)]
        fn default() -> RustHeuristicScorer;

        // Scorer name for telemetry.
        fn name(self: &RustHeuristicScorer) -> String;
    }

    // ---------- last_error ----------
    // Each opaque type stores the last error from a method call that
    // returned None / false. Swift reads this immediately after a
    // failing call to discriminate.
    extern "Rust" {
        fn last_error_vault(v: &RustVault) -> i32;
        fn last_error_card() -> i32;
    }
}

// ============================================================
// Implementation. The types here mirror the names in the bridge
// module above; swift-bridge requires they be defined in the same
// crate.
// ============================================================

use std::cell::Cell;
use std::sync::Arc;

use op_fraud::{HeuristicScorer, Scorer};
use op_vault::{
    CardData, InMemoryVault, TokenFormat, TokenLifetime, TokenizationPolicy, Vault, VaultRef,
};

use crate::error::FfiError;

// Thread-local last-error slots. Each type's failing methods write
// here and Swift reads via `last_error_*`. We use thread-locals so
// concurrent Swift threads each see their own error state.
thread_local! {
    static LAST_ERROR_VAULT: Cell<FfiError> = const { Cell::new(FfiError::Ok) };
    static LAST_ERROR_CARD: Cell<FfiError> = const { Cell::new(FfiError::Ok) };
}

fn set_vault_error(e: FfiError) {
    LAST_ERROR_VAULT.with(|c| c.set(e));
}

fn set_card_error(e: FfiError) {
    LAST_ERROR_CARD.with(|c| c.set(e));
}

// ---------- CardData ----------

/// Opaque CardData wrapper. Bridges to Swift as `RustCardData`.
pub struct RustCardData {
    inner: CardData,
}

impl RustCardData {
    /// Constructor exposed to Swift. Returns `None` on invalid input.
    pub fn new(pan: &str, exp_month: u8, exp_year: u16) -> Option<RustCardData> {
        match CardData::new(pan.to_owned(), exp_month, exp_year) {
            Ok(inner) => {
                set_card_error(FfiError::Ok);
                Some(Self { inner })
            }
            Err(_) => {
                set_card_error(FfiError::InvalidInput);
                None
            }
        }
    }

    /// First six digits.
    pub fn first_six(&self) -> String {
        self.inner.first_six().to_owned()
    }

    /// Last four digits.
    pub fn last_four(&self) -> String {
        self.inner.last_four().to_owned()
    }

    /// Expiration month.
    pub fn exp_month(&self) -> u8 {
        self.inner.exp_month()
    }

    /// Expiration year.
    pub fn exp_year(&self) -> u16 {
        self.inner.exp_year()
    }
}

// ---------- VaultRef ----------

/// Opaque VaultRef wrapper.
pub struct RustVaultRef {
    inner: VaultRef,
}

impl RustVaultRef {
    /// String form of the token.
    pub fn as_string(&self) -> String {
        self.inner.as_str().to_owned()
    }
}

// ---------- Vault ----------

/// Opaque Vault handle. Wraps an `Arc<dyn Vault>` so Swift can hold
/// multiple references without cloning the underlying vault.
pub struct RustVault {
    inner: Arc<dyn Vault>,
}

impl RustVault {
    /// Construct an ephemeral in-memory vault.
    pub fn ephemeral(name: &str) -> RustVault {
        Self {
            inner: Arc::new(InMemoryVault::ephemeral(name.to_owned())),
        }
    }

    /// Tokenize.
    pub fn tokenize(
        &self,
        card: RustCardData,
        policy: ffi::TokenizationPolicyFfi,
    ) -> Option<RustVaultRef> {
        let policy = decode_policy(&policy);
        match self.inner.tokenize(card.inner, policy) {
            Ok(vref) => {
                set_vault_error(FfiError::Ok);
                Some(RustVaultRef { inner: vref })
            }
            Err(e) => {
                set_vault_error(FfiError::from(e));
                None
            }
        }
    }

    /// Detokenize.
    pub fn detokenize(&self, token: &RustVaultRef) -> Option<RustCardData> {
        match self.inner.detokenize(&token.inner) {
            Ok(card) => {
                set_vault_error(FfiError::Ok);
                Some(RustCardData { inner: card })
            }
            Err(e) => {
                set_vault_error(FfiError::from(e));
                None
            }
        }
    }

    /// Exists check.
    pub fn exists(&self, token: &RustVaultRef) -> bool {
        match self.inner.exists(&token.inner) {
            Ok(b) => {
                set_vault_error(FfiError::Ok);
                b
            }
            Err(e) => {
                set_vault_error(FfiError::from(e));
                false
            }
        }
    }

    /// Delete.
    pub fn delete(&self, token: &RustVaultRef) -> bool {
        match self.inner.delete(&token.inner) {
            Ok(b) => {
                set_vault_error(FfiError::Ok);
                b
            }
            Err(e) => {
                set_vault_error(FfiError::from(e));
                false
            }
        }
    }
}

// ---------- Scorer ----------

/// Opaque heuristic-scorer handle.
pub struct RustHeuristicScorer {
    inner: HeuristicScorer,
}

impl RustHeuristicScorer {
    /// Default constructor.
    pub fn default() -> RustHeuristicScorer {
        Self {
            inner: HeuristicScorer::new(),
        }
    }

    /// Name for telemetry.
    pub fn name(&self) -> String {
        self.inner.name().to_owned()
    }
}

// ---------- last_error_* ----------

/// Read the thread-local vault error.
pub fn last_error_vault(_v: &RustVault) -> i32 {
    LAST_ERROR_VAULT.with(|c| c.get().as_i32())
}

/// Read the thread-local CardData error.
pub fn last_error_card() -> i32 {
    LAST_ERROR_CARD.with(|c| c.get().as_i32())
}

// ============================================================
// Helpers for converting shared structs <-> Rust types
// ============================================================

fn decode_policy(p: &ffi::TokenizationPolicyFfi) -> TokenizationPolicy {
    TokenizationPolicy {
        format: match p.format {
            ffi::TokenFormatFfi::Random => TokenFormat::Random,
            ffi::TokenFormatFfi::Deterministic => TokenFormat::Deterministic,
        },
        lifetime: match p.lifetime {
            ffi::TokenLifetimeFfi::Reusable => TokenLifetime::Reusable,
            ffi::TokenLifetimeFfi::SingleUse => TokenLifetime::SingleUse,
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

    const VALID_VISA: &str = "4242424242424242";

    fn default_policy() -> ffi::TokenizationPolicyFfi {
        ffi::TokenizationPolicyFfi {
            format: ffi::TokenFormatFfi::Random,
            lifetime: ffi::TokenLifetimeFfi::Reusable,
            ttl_seconds: 0,
        }
    }

    #[test]
    fn rust_card_data_new_with_valid_pan() {
        let card = RustCardData::new(VALID_VISA, 12, 2030);
        assert!(card.is_some());
        let c = card.unwrap();
        assert_eq!(c.first_six(), "424242");
        assert_eq!(c.last_four(), "4242");
        assert_eq!(c.exp_month(), 12);
        assert_eq!(c.exp_year(), 2030);
    }

    #[test]
    fn rust_card_data_new_with_invalid_pan_returns_none_and_sets_error() {
        let card = RustCardData::new("1111111111111111", 12, 2030);
        assert!(card.is_none());
        assert_eq!(last_error_card(), FfiError::InvalidInput.as_i32());
    }

    #[test]
    fn rust_card_data_new_with_bad_expiration_returns_none() {
        assert!(RustCardData::new(VALID_VISA, 13, 2030).is_none());
        assert!(RustCardData::new(VALID_VISA, 12, 1999).is_none());
    }

    #[test]
    fn ephemeral_vault_round_trip() {
        let vault = RustVault::ephemeral("test");
        let card = RustCardData::new(VALID_VISA, 12, 2030).unwrap();
        let token = vault.tokenize(card, default_policy());
        assert!(token.is_some());

        let token = token.unwrap();
        assert!(token.as_string().starts_with("tok_v7_"));

        let recovered = vault.detokenize(&token);
        assert!(recovered.is_some());
        assert_eq!(recovered.unwrap().last_four(), "4242");

        assert_eq!(last_error_vault(&vault), FfiError::Ok.as_i32());
    }

    #[test]
    fn detokenize_unknown_returns_none_and_sets_lookup_failed() {
        let vault = RustVault::ephemeral("err-test");
        let fake = RustVaultRef {
            inner: VaultRef::new("tok_v7_doesnotexist"),
        };
        let result = vault.detokenize(&fake);
        assert!(result.is_none());
        assert_eq!(
            last_error_vault(&vault),
            FfiError::VaultLookupFailed.as_i32()
        );
    }

    #[test]
    fn detokenize_malformed_token_also_collapses_to_lookup_failed() {
        // Tokens that don't start with our prefix → InvalidToken in
        // the vault → VaultLookupFailed at the FFI per oracle rules.
        let vault = RustVault::ephemeral("malformed-test");
        let bad = RustVaultRef {
            inner: VaultRef::new("4242424242424242"), // looks like a PAN
        };
        let result = vault.detokenize(&bad);
        assert!(result.is_none());
        assert_eq!(
            last_error_vault(&vault),
            FfiError::VaultLookupFailed.as_i32()
        );
    }

    #[test]
    fn single_use_policy_decoded_correctly() {
        let p = ffi::TokenizationPolicyFfi {
            format: ffi::TokenFormatFfi::Random,
            lifetime: ffi::TokenLifetimeFfi::SingleUse,
            ttl_seconds: 60,
        };
        let decoded = decode_policy(&p);
        assert_eq!(decoded.lifetime, TokenLifetime::SingleUse);
        assert_eq!(decoded.ttl_seconds, Some(60));
    }

    #[test]
    fn ttl_seconds_zero_decodes_to_none() {
        let p = ffi::TokenizationPolicyFfi {
            format: ffi::TokenFormatFfi::Random,
            lifetime: ffi::TokenLifetimeFfi::Reusable,
            ttl_seconds: 0,
        };
        assert_eq!(decode_policy(&p).ttl_seconds, None);
    }

    #[test]
    fn deterministic_format_decodes_correctly() {
        let p = ffi::TokenizationPolicyFfi {
            format: ffi::TokenFormatFfi::Deterministic,
            lifetime: ffi::TokenLifetimeFfi::Reusable,
            ttl_seconds: 0,
        };
        assert_eq!(decode_policy(&p).format, TokenFormat::Deterministic);
    }

    #[test]
    fn delete_returns_false_for_unknown() {
        let vault = RustVault::ephemeral("delete-test");
        let fake = RustVaultRef {
            inner: VaultRef::new("tok_v7_nope"),
        };
        assert!(!vault.delete(&fake));
    }

    #[test]
    fn delete_returns_true_after_tokenize() {
        let vault = RustVault::ephemeral("delete-roundtrip");
        let card = RustCardData::new(VALID_VISA, 12, 2030).unwrap();
        let token = vault.tokenize(card, default_policy()).unwrap();
        assert!(vault.delete(&token));
        // Second delete is idempotent and returns false.
        assert!(!vault.delete(&token));
    }

    #[test]
    fn exists_after_tokenize() {
        let vault = RustVault::ephemeral("exists-test");
        let card = RustCardData::new(VALID_VISA, 12, 2030).unwrap();
        let token = vault.tokenize(card, default_policy()).unwrap();
        assert!(vault.exists(&token));
    }

    #[test]
    fn heuristic_scorer_default_has_stable_name() {
        let s = RustHeuristicScorer::default();
        assert_eq!(s.name(), "heuristic-v1");
    }

    #[test]
    fn thread_local_errors_isolate_per_thread() {
        use std::thread;

        // Thread A: trigger an error and read it.
        let h_a = thread::spawn(|| {
            let _ = RustCardData::new("bad", 12, 2030);
            last_error_card()
        });

        // Thread B: trigger a different error.
        let h_b = thread::spawn(|| {
            let _ = RustCardData::new("also-bad", 99, 2030);
            last_error_card()
        });

        // Both threads see InvalidInput in their own slot.
        assert_eq!(h_a.join().unwrap(), FfiError::InvalidInput.as_i32());
        assert_eq!(h_b.join().unwrap(), FfiError::InvalidInput.as_i32());
    }

    #[test]
    fn successful_card_creation_clears_error() {
        // Trigger an error first.
        let _ = RustCardData::new("bad", 12, 2030);
        assert_eq!(last_error_card(), FfiError::InvalidInput.as_i32());
        // Now a successful creation should clear it.
        let _ = RustCardData::new(VALID_VISA, 12, 2030);
        assert_eq!(last_error_card(), FfiError::Ok.as_i32());
    }
}
