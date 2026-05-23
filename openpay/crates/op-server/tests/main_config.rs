//! Integration-level coverage for the env-driven boot helpers in
//! [`op_server::config`].
//!
//! These tests exercise `build_state_from_env` and
//! `build_middleware_from_env` via the **public** library surface —
//! the same path `main.rs` takes. The reader closure is backed by a
//! `HashMap`, so no global `std::env` mutation happens and tests run
//! safely in parallel.

use std::collections::HashMap;

use op_core::RailKind;
use op_server::{build_middleware_from_env, build_state_from_env};

fn reader(pairs: &[(&'static str, &'static str)]) -> impl Fn(&str) -> Option<String> {
    let map: HashMap<&'static str, &'static str> = pairs.iter().copied().collect();
    move |k| map.get(k).map(|s| (*s).to_owned())
}

#[test]
fn no_env_returns_in_memory_state_no_middleware() {
    let r = reader(&[]);
    let state = build_state_from_env(&r).expect("build_state");
    let mw = build_middleware_from_env(&r);

    assert!(mw.auth.is_none(), "no auth layer without OP_API_KEYS");
    assert!(
        mw.rate_limit.is_none(),
        "no rate-limit layer without OP_RATE_LIMIT_PER_MINUTE"
    );
    assert!(
        !state.orchestrator.has_driver(RailKind::Crypto, "usdc-base"),
        "no usdc-base adapter without EVM env vars"
    );
}

#[test]
fn api_keys_csv_produces_auth_layer() {
    let r = reader(&[("OP_API_KEYS", "key1,key2,key3")]);
    let mw = build_middleware_from_env(&r);
    assert!(mw.auth.is_some());
    assert!(mw.rate_limit.is_none());
}

#[test]
fn rate_limit_per_minute_parses_u32() {
    let good = reader(&[("OP_RATE_LIMIT_PER_MINUTE", "600")]);
    let mw_good = build_middleware_from_env(&good);
    assert!(mw_good.rate_limit.is_some());

    let bad = reader(&[("OP_RATE_LIMIT_PER_MINUTE", "garbage")]);
    let mw_bad = build_middleware_from_env(&bad);
    assert!(
        mw_bad.rate_limit.is_none(),
        "garbage value should produce None"
    );
}

#[test]
fn evm_env_vars_register_gateway() {
    // Anvil / Hardhat test key #0 — public, well-known, no funds on
    // any mainnet. Safe to embed in tests.
    let r = reader(&[
        ("OP_BASE_RPC_URL", "http://localhost:8545"),
        (
            "OP_USDC_BASE_PRIVATE_KEY",
            "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        ),
    ]);
    let state = build_state_from_env(&r).expect("build_state");
    assert!(
        state.orchestrator.has_driver(RailKind::Crypto, "usdc-base"),
        "expected usdc-base CryptoAdapter registered when both env vars set"
    );
}

#[test]
fn only_rpc_url_skips_evm_with_warning() {
    let r = reader(&[("OP_BASE_RPC_URL", "http://localhost:8545")]);
    let state = build_state_from_env(&r).expect("build_state");
    assert!(!state.orchestrator.has_driver(RailKind::Crypto, "usdc-base"));
}

#[test]
fn only_private_key_skips_evm_with_warning() {
    let r = reader(&[(
        "OP_USDC_BASE_PRIVATE_KEY",
        "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
    )]);
    let state = build_state_from_env(&r).expect("build_state");
    assert!(!state.orchestrator.has_driver(RailKind::Crypto, "usdc-base"));
}

#[test]
fn both_middleware_and_evm_compose() {
    let r = reader(&[
        ("OP_API_KEYS", "alpha,beta"),
        ("OP_RATE_LIMIT_PER_MINUTE", "120"),
        ("OP_BASE_RPC_URL", "http://localhost:8545"),
        (
            "OP_USDC_BASE_PRIVATE_KEY",
            "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        ),
    ]);
    let state = build_state_from_env(&r).expect("build_state");
    let mw = build_middleware_from_env(&r);

    assert!(mw.auth.is_some());
    assert!(mw.rate_limit.is_some());
    assert!(state.orchestrator.has_driver(RailKind::Crypto, "usdc-base"));
}
