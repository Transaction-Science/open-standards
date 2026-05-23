//! Integration tests that run inside a real wasm host (Node.js or
//! a headless browser via `wasm-pack test`).
//!
//! These tests exercise the actual `#[wasm_bindgen]` glue, including
//! the JS object ownership protocol, constructor / getter dispatch,
//! and `Result<T, JsValue>` exception propagation. Run with:
//!
//! ```text
//! # Node.js (default)
//! wasm-pack test --node
//!
//! # Headless Chrome
//! wasm-pack test --headless --chrome
//!
//! # Headless Firefox
//! wasm-pack test --headless --firefox
//! ```
//!
//! ## Note on error inspection
//!
//! When a `Result<T, JsValue>` fails, the `JsValue` is opaque from
//! Rust's perspective — wasm-bindgen doesn't expose a public
//! `TryFromJsValue` for our exported `OpenPayError` class. So we
//! verify failure by checking `result.is_err()` and trust the
//! crate-level `error::tests` to verify the OpenPayError contents
//! before they cross the boundary. JS-side tests (in
//! `js/test.mjs`) cover the consumer-visible error shape.

#![cfg(target_arch = "wasm32")]

use op_wasm::{
    CardData, HeuristicScorer, RustVault, TokenFormat, TokenLifetime, TokenizationPolicy, VaultRef,
};
use wasm_bindgen_test::*;

// By default tests run in Node; uncomment to force browser:
// wasm_bindgen_test_configure!(run_in_browser);

const VALID_VISA: &str = "4242424242424242";
const VALID_MC: &str = "5555555555554444";

#[wasm_bindgen_test]
fn card_data_valid_constructs() {
    let card = CardData::new(VALID_VISA.to_owned(), 12, 2030).unwrap();
    assert_eq!(card.first_six(), "424242");
    assert_eq!(card.last_four(), "4242");
    assert_eq!(card.exp_month(), 12);
    assert_eq!(card.exp_year(), 2030);
}

#[wasm_bindgen_test]
fn card_data_invalid_pan_returns_err() {
    assert!(CardData::new("1111111111111111".to_owned(), 12, 2030).is_err());
}

#[wasm_bindgen_test]
fn card_data_invalid_exp_month_returns_err() {
    assert!(CardData::new(VALID_VISA.to_owned(), 13, 2030).is_err());
    assert!(CardData::new(VALID_VISA.to_owned(), 0, 2030).is_err());
}

#[wasm_bindgen_test]
fn vault_round_trip() {
    let vault = RustVault::new("wasm-test".to_owned());
    assert_eq!(vault.name(), "wasm-test");

    let card = CardData::new(VALID_VISA.to_owned(), 12, 2030).unwrap();
    let token = vault.tokenize(card, None).unwrap();
    assert!(token.as_string().starts_with("tok_v7_"));

    let recovered = vault.detokenize(&token).unwrap();
    assert_eq!(recovered.last_four(), "4242");
}

#[wasm_bindgen_test]
fn vault_unknown_token_returns_err() {
    let vault = RustVault::new("err".to_owned());
    let fake = VaultRef::from_string("tok_v7_doesnotexist".to_owned());
    assert!(vault.detokenize(&fake).is_err());
}

#[wasm_bindgen_test]
fn vault_malformed_token_also_returns_err() {
    let vault = RustVault::new("err".to_owned());
    let bad = VaultRef::from_string("not-a-real-token".to_owned());
    assert!(vault.detokenize(&bad).is_err());
}

#[wasm_bindgen_test]
fn vault_single_use_consumes_on_first_detokenize() {
    let vault = RustVault::new("single".to_owned());
    let card = CardData::new(VALID_VISA.to_owned(), 12, 2030).unwrap();
    let policy = TokenizationPolicy::single_use(Some(120));
    let token = vault.tokenize(card, Some(policy)).unwrap();

    // First detokenize succeeds.
    let recovered = vault.detokenize(&token).unwrap();
    assert_eq!(recovered.last_four(), "4242");

    // Second detokenize fails — single-use already consumed.
    assert!(vault.detokenize(&token).is_err());
}

#[wasm_bindgen_test]
fn vault_reusable_allows_multiple_detokenizes() {
    let vault = RustVault::new("reusable".to_owned());
    let card = CardData::new(VALID_VISA.to_owned(), 12, 2030).unwrap();
    let token = vault.tokenize(card, None).unwrap();
    for _ in 0..5 {
        let r = vault.detokenize(&token).unwrap();
        assert_eq!(r.last_four(), "4242");
    }
}

#[wasm_bindgen_test]
fn vault_exists_and_delete() {
    let vault = RustVault::new("del".to_owned());
    let card = CardData::new(VALID_VISA.to_owned(), 12, 2030).unwrap();
    let token = vault.tokenize(card, None).unwrap();

    assert!(vault.exists(&token).unwrap());
    assert!(vault.delete(&token).unwrap());
    assert!(!vault.exists(&token).unwrap());
    // Idempotent.
    assert!(!vault.delete(&token).unwrap());
}

#[wasm_bindgen_test]
fn vault_distinct_tokens_for_same_pan() {
    let vault = RustVault::new("uniq".to_owned());
    let c1 = CardData::new(VALID_VISA.to_owned(), 12, 2030).unwrap();
    let c2 = CardData::new(VALID_VISA.to_owned(), 12, 2030).unwrap();
    let t1 = vault.tokenize(c1, None).unwrap();
    let t2 = vault.tokenize(c2, None).unwrap();
    assert_ne!(t1.as_string(), t2.as_string());
}

#[wasm_bindgen_test]
fn vault_multiple_cards_coexist() {
    let vault = RustVault::new("multi".to_owned());
    let visa = vault
        .tokenize(
            CardData::new(VALID_VISA.to_owned(), 12, 2030).unwrap(),
            None,
        )
        .unwrap();
    let mc = vault
        .tokenize(CardData::new(VALID_MC.to_owned(), 11, 2028).unwrap(), None)
        .unwrap();

    assert_eq!(vault.detokenize(&visa).unwrap().last_four(), "4242");
    let mc_card = vault.detokenize(&mc).unwrap();
    assert_eq!(mc_card.last_four(), "4444");
    assert_eq!(mc_card.exp_month(), 11);
}

#[wasm_bindgen_test]
fn vault_ref_round_trip() {
    let original = "tok_v7_test123";
    let v = VaultRef::from_string(original.to_owned());
    assert_eq!(v.as_string(), original);
}

#[wasm_bindgen_test]
fn heuristic_scorer_name_stable() {
    let s = HeuristicScorer::new();
    assert_eq!(s.name(), "heuristic-v1");
}

#[wasm_bindgen_test]
fn tokenization_policy_helpers() {
    let single = TokenizationPolicy::single_use(Some(60));
    assert_eq!(single.lifetime(), TokenLifetime::SingleUse);
    assert_eq!(single.ttl_seconds(), 60);

    let cof = TokenizationPolicy::card_on_file();
    assert_eq!(cof.lifetime(), TokenLifetime::Reusable);
    assert_eq!(cof.ttl_seconds(), 0);

    let def = TokenizationPolicy::new();
    assert_eq!(def.format(), TokenFormat::Random);
    assert_eq!(def.lifetime(), TokenLifetime::Reusable);
}

#[wasm_bindgen_test]
fn pan_never_appears_in_token_string() {
    let vault = RustVault::new("opacity".to_owned());
    let card = CardData::new(VALID_VISA.to_owned(), 12, 2030).unwrap();
    let token = vault.tokenize(card, None).unwrap();
    let tok_str = token.as_string();
    assert!(!tok_str.contains(VALID_VISA));
    assert!(!tok_str.contains("424242"));
    assert!(!tok_str.contains("4242"));
}

#[wasm_bindgen_test]
fn tokenize_from_string_convenience_function() {
    let vault = RustVault::new("conv".to_owned());
    let token = op_wasm::vault::tokenize_from_string(&vault, VALID_VISA.to_owned(), 12, 2030, None)
        .unwrap();
    assert!(token.as_string().starts_with("tok_v7_"));

    let recovered = vault.detokenize(&token).unwrap();
    assert_eq!(recovered.last_four(), "4242");
}

#[wasm_bindgen_test]
fn tokenize_from_string_invalid_pan_returns_err() {
    let vault = RustVault::new("conv-err".to_owned());
    let r =
        op_wasm::vault::tokenize_from_string(&vault, "1111111111111111".to_owned(), 12, 2030, None);
    assert!(r.is_err());
}

#[wasm_bindgen_test]
fn deterministic_format_produces_same_token_for_same_pan() {
    let vault = RustVault::new("det".to_owned());
    let mut policy = TokenizationPolicy::new();
    policy.set_format(TokenFormat::Deterministic);

    let c1 = CardData::new(VALID_VISA.to_owned(), 12, 2030).unwrap();
    let c2 = CardData::new(VALID_VISA.to_owned(), 12, 2030).unwrap();
    let t1 = vault.tokenize(c1, Some(policy)).unwrap();
    let t2 = vault.tokenize(c2, Some(policy)).unwrap();
    assert_eq!(t1.as_string(), t2.as_string());
}

#[wasm_bindgen_test]
fn vault_name_is_telemetry_only_does_not_affect_behavior() {
    // Constructing two vaults with the same name doesn't make them
    // share state — they're independent in-memory instances.
    let a = RustVault::new("same".to_owned());
    let b = RustVault::new("same".to_owned());
    let card = CardData::new(VALID_VISA.to_owned(), 12, 2030).unwrap();
    let token = a.tokenize(card, None).unwrap();
    assert!(b.detokenize(&token).is_err());
}
