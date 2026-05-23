//! End-to-end lifecycle test for op-vault.
//!
//! Exercises the full PCI scope-reduction flow:
//! 1. Customer enters card data → CardData (validated, Luhn-checked)
//! 2. Vault tokenizes → VaultRef (opaque, contains no PAN)
//! 3. Orchestrator hands VaultRef around (no PCI exposure)
//! 4. Rail driver, at submit time, detokenizes → recovered CardData
//! 5. PSP receives the PAN, returns auth result
//! 6. Vault retains the token for future charges (or single-use)
//!
//! These tests use the in-memory reference vault. Production deploys
//! plug the same Vault trait into Keychain / Keystore / HSM / KMS.

#![cfg(feature = "in-memory")]

use std::sync::Arc;
use std::thread;

use op_vault::{
    CardData, Error, InMemoryVault, TokenLifetime, TokenizationPolicy, Vault, VaultRef,
};

const VALID_VISA: &str = "4242424242424242";
const VALID_MC: &str = "5555555555554444";

fn card_visa() -> CardData {
    CardData::new(VALID_VISA, 12, 2030).unwrap()
}

#[test]
fn full_lifecycle_tokenize_pass_around_detokenize() {
    // Step 1: customer enters card
    let card = card_visa();
    let last_four = card.last_four().to_owned();
    let first_six = card.first_six().to_owned();

    // Step 2: vault tokenizes
    let vault = Arc::new(InMemoryVault::ephemeral("primary"));
    let token = vault
        .tokenize(card, TokenizationPolicy::card_on_file())
        .unwrap();

    // Step 3: hand the token to the "orchestrator" — just a function
    // that holds the VaultRef and decides routing without touching PAN.
    fn orchestrator_decision(_t: &VaultRef) -> &'static str {
        // The orchestrator has zero ability to read the PAN here.
        // It can route based only on metadata (rail, amount, geo).
        "route_to_card_rail"
    }
    assert_eq!(orchestrator_decision(&token), "route_to_card_rail");

    // Step 4: rail driver detokenizes at submit time
    let recovered = vault.detokenize(&token).unwrap();
    assert_eq!(recovered.last_four(), last_four);
    assert_eq!(recovered.first_six(), first_six);

    // Step 5: token is still valid for the next charge (card-on-file)
    let recovered_again = vault.detokenize(&token).unwrap();
    assert_eq!(recovered_again.last_four(), last_four);
}

#[test]
fn single_use_token_for_3ds_auth() {
    // 3DS authentication completes in <2 minutes. Use a single-use
    // token with a short TTL to bound the risk window.
    let vault = InMemoryVault::ephemeral("3ds");
    let policy = TokenizationPolicy::single_use(120);
    let token = vault.tokenize(card_visa(), policy).unwrap();

    // 3DS handler resolves the token once.
    let _card = vault.detokenize(&token).unwrap();

    // Replay attempt fails closed.
    let err = vault.detokenize(&token).unwrap_err();
    assert!(matches!(err, Error::AlreadyConsumed));
}

#[test]
fn vault_shared_across_threads() {
    // The orchestrator runs worker threads, each holding an Arc<dyn Vault>.
    let vault: Arc<dyn Vault> = Arc::new(InMemoryVault::ephemeral("shared"));

    let token = vault
        .tokenize(card_visa(), TokenizationPolicy::default())
        .unwrap();

    // Spawn workers that each detokenize and verify.
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let v = Arc::clone(&vault);
            let t = token.clone();
            thread::spawn(move || {
                let card = v.detokenize(&t).unwrap();
                card.last_four().to_owned()
            })
        })
        .collect();

    for h in handles {
        assert_eq!(h.join().unwrap(), "4242");
    }
}

#[test]
fn multiple_cards_remain_isolated() {
    let vault = InMemoryVault::ephemeral("multi");
    let c_visa = CardData::new(VALID_VISA, 12, 2030).unwrap();
    let c_mc = CardData::new(VALID_MC, 6, 2028).unwrap();

    let t_visa = vault
        .tokenize(c_visa, TokenizationPolicy::default())
        .unwrap();
    let t_mc = vault.tokenize(c_mc, TokenizationPolicy::default()).unwrap();

    let r_visa = vault.detokenize(&t_visa).unwrap();
    let r_mc = vault.detokenize(&t_mc).unwrap();

    assert_eq!(r_visa.last_four(), "4242");
    assert_eq!(r_mc.last_four(), "4444");
    assert_eq!(r_visa.first_six(), "424242");
    assert_eq!(r_mc.first_six(), "555555");
}

#[test]
fn deleted_token_is_not_recoverable() {
    let vault = InMemoryVault::ephemeral("retention");
    let token = vault
        .tokenize(card_visa(), TokenizationPolicy::default())
        .unwrap();

    // Customer removes their card.
    let removed = vault.delete(&token).unwrap();
    assert!(removed);

    // Subsequent detokenize fails closed.
    let err = vault.detokenize(&token).unwrap_err();
    assert!(matches!(err, Error::NotFound));

    // Idempotent: deleting again is a no-op.
    let removed_again = vault.delete(&token).unwrap();
    assert!(!removed_again);
}

#[test]
fn pan_never_appears_in_token_string() {
    let vault = InMemoryVault::ephemeral("opacity");
    let token = vault
        .tokenize(card_visa(), TokenizationPolicy::default())
        .unwrap();
    let token_str = token.as_str();

    // The full PAN must not appear in the token.
    assert!(!token_str.contains(VALID_VISA));
    // Neither must the BIN or last four (these are NOT PAN, but they
    // can't be derivable from the token either).
    assert!(!token_str.contains("424242"));
    assert!(!token_str.contains("4242"));
}

#[test]
fn debug_format_of_card_data_masks_pan() {
    let card = card_visa();
    let dbg = format!("{card:?}");
    assert!(dbg.contains("424242")); // BIN allowed
    assert!(dbg.contains("4242")); // last4 allowed
    assert!(!dbg.contains("4242424242424242")); // never full PAN
}

#[test]
fn many_tokenize_operations_produce_unique_tokens() {
    // Birthday-paradox-level test: 1000 tokens, all distinct.
    let vault = InMemoryVault::ephemeral("uniq");
    let mut tokens = Vec::with_capacity(1000);
    for _ in 0..1000 {
        tokens.push(
            vault
                .tokenize(card_visa(), TokenizationPolicy::default())
                .unwrap(),
        );
    }
    let mut seen = std::collections::HashSet::new();
    for t in &tokens {
        assert!(seen.insert(t.as_str().to_owned()), "duplicate token: {t:?}");
    }
}

#[test]
fn exists_check_does_not_decrypt() {
    // The orchestrator can probe whether a token is valid before
    // bothering to route a payment, without revealing the PAN.
    let vault = InMemoryVault::ephemeral("probe");
    let token = vault
        .tokenize(card_visa(), TokenizationPolicy::default())
        .unwrap();
    assert!(vault.exists(&token).unwrap());

    let fake = VaultRef::new("tok_v7_definitely_not_a_real_token");
    assert!(!vault.exists(&fake).unwrap());
}

#[test]
fn vault_trait_object_routes_via_dyn() {
    // The orchestrator holds Box<dyn Vault> and doesn't know which
    // implementation it has — same pattern as Phases 4-6.
    fn issue_token(v: &dyn Vault, card: CardData) -> VaultRef {
        v.tokenize(card, TokenizationPolicy::default()).unwrap()
    }
    let vault = InMemoryVault::ephemeral("dyn");
    let token = issue_token(&vault, card_visa());
    assert!(token.as_str().starts_with("tok_v7_"));
}

#[test]
fn policy_helpers_compose_correctly() {
    // The single-use helper produces a token that is consumed once.
    let vault = InMemoryVault::ephemeral("policy");
    let p = TokenizationPolicy::single_use(60);
    assert_eq!(p.lifetime, TokenLifetime::SingleUse);
    let token = vault.tokenize(card_visa(), p).unwrap();
    assert!(vault.detokenize(&token).is_ok());
    assert!(matches!(
        vault.detokenize(&token),
        Err(Error::AlreadyConsumed)
    ));
}
