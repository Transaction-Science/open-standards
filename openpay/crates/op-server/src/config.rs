//! Env-driven boot configuration.
//!
//! The `op-server` binary is meant to launch with a single command
//! plus a `.env` file (or systemd `EnvironmentFile`):
//!
//! ```bash
//! OP_BIND_ADDR=0.0.0.0:8080 \
//! OP_GRAPH_PATH=/var/lib/openpay/data.graph \
//! OP_API_KEYS=key1,key2,key3 \
//! OP_LOG_FORMAT=json \
//! OP_BASE_RPC_URL=https://mainnet.base.org \
//! OP_USDC_BASE_PRIVATE_KEY=0x... \
//! OP_RATE_LIMIT_PER_MINUTE=600 \
//! op-server
//! ```
//!
//! This module factors the env-parsing logic out of `main.rs` so it
//! can be unit-tested with a mocked env (no global `std::env`
//! mutation, no `serial_test` dance). The two public entry points
//! both take a `reader` closure that maps an env-var name to its
//! value: production passes `|k| std::env::var(k).ok()`; tests pass
//! a closure backed by a `HashMap`.
//!
//! ## Env-var matrix
//!
//! | Var | Default | Effect |
//! |-----|---------|--------|
//! | `OP_GRAPH_PATH` | (unset) | Persist all stores to this `.graph` file. Unset → in-memory (volatile). |
//! | `OP_API_KEYS` | (unset) | Comma-separated list. Unset → no auth (warning logged). |
//! | `OP_RATE_LIMIT_PER_MINUTE` | (unset) | Integer. Unset → no rate-limit. |
//! | `OP_BASE_RPC_URL` + `OP_USDC_BASE_PRIVATE_KEY` | (both unset) | Register a `usdc-base` crypto rail. Only one set → warning, skip. |

use std::collections::HashSet;
use std::sync::Arc;

use op_orchestrator::{CryptoAdapter, Orchestrator};

use crate::auth::ApiKeyAuthLayer;
use crate::rate_limit::RateLimitLayer;
use crate::state::AppState;

/// Bundle of optional middleware layers produced from the env.
///
/// Either field may be `None` when its corresponding env var was
/// unset or malformed. The caller (`main.rs`) decides which `router`
/// constructor to invoke based on what's populated.
#[derive(Default)]
pub struct EnvMiddleware {
    /// `OP_API_KEYS` → API-key auth layer, with `/health` and
    /// `/readiness` bypassed.
    pub auth: Option<ApiKeyAuthLayer>,
    /// `OP_RATE_LIMIT_PER_MINUTE` → token-bucket rate-limit layer.
    pub rate_limit: Option<RateLimitLayer>,
}

/// Build the [`AppState`] from environment variables, reading via
/// the supplied closure.
///
/// Applies:
/// - `OP_GRAPH_PATH` — persistent graph file, in-memory if unset
///   (with a warning).
/// - `OP_BASE_RPC_URL` + `OP_USDC_BASE_PRIVATE_KEY` — registers a
///   live `usdc-base` `CryptoAdapter` on a fresh orchestrator. Both
///   must be set together; if only one is present a warning is
///   logged and the gateway is skipped.
///
/// # Errors
/// - [`op_graph::Error`] if the graph path can't be opened.
pub fn build_state_from_env<F>(reader: F) -> op_graph::Result<AppState>
where
    F: Fn(&str) -> Option<String>,
{
    // 1) Graph backing.
    let mut state = match reader("OP_GRAPH_PATH") {
        Some(path) if !path.is_empty() => {
            tracing::info!(path = %path, "op-server graph: persistent");
            AppState::with_graph_path(&path)?
        }
        _ => {
            tracing::warn!(
                "OP_GRAPH_PATH unset — using in-memory graph; \
                 all state is lost on restart. \
                 Set OP_GRAPH_PATH=/var/lib/openpay/data.graph for persistence."
            );
            AppState::new_in_memory()
        }
    };

    // 2) Optional `usdc-base` crypto rail.
    state = maybe_register_usdc_base(state, &reader);

    Ok(state)
}

/// Build optional middleware layers from environment variables.
///
/// Applies:
/// - `OP_API_KEYS` — comma-separated, builds an
///   [`ApiKeyAuthLayer`] with `/health` + `/readiness` bypass paths.
/// - `OP_RATE_LIMIT_PER_MINUTE` — parsed as `u32`; non-numeric
///   values log a warning and yield `None`.
///
/// Returns `EnvMiddleware { auth: None, rate_limit: None }` when no
/// env vars are set. The caller composes the final router using
/// [`crate::router_with_middleware`] or [`crate::router`] depending
/// on which slots are populated.
#[must_use]
pub fn build_middleware_from_env<F>(reader: F) -> EnvMiddleware
where
    F: Fn(&str) -> Option<String>,
{
    let auth = build_auth_from_env(&reader);
    let rate_limit = build_rate_limit_from_env(&reader);
    EnvMiddleware { auth, rate_limit }
}

fn build_auth_from_env<F>(reader: &F) -> Option<ApiKeyAuthLayer>
where
    F: Fn(&str) -> Option<String>,
{
    let raw = reader("OP_API_KEYS")?;
    let keys: HashSet<String> = raw
        .split(',')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(str::to_owned)
        .collect();
    if keys.is_empty() {
        tracing::warn!("OP_API_KEYS is set but parses to zero keys — server is unauthenticated");
        return None;
    }
    tracing::info!(count = keys.len(), "OP_API_KEYS: auth layer enabled");
    Some(ApiKeyAuthLayer::new(keys).with_bypass_paths(vec!["/health".into(), "/readiness".into()]))
}

fn build_rate_limit_from_env<F>(reader: &F) -> Option<RateLimitLayer>
where
    F: Fn(&str) -> Option<String>,
{
    let raw = reader("OP_RATE_LIMIT_PER_MINUTE")?;
    match raw.parse::<u32>() {
        Ok(n) if n > 0 => {
            tracing::info!(
                per_minute = n,
                "OP_RATE_LIMIT_PER_MINUTE: rate-limit enabled"
            );
            Some(RateLimitLayer::per_minute(n))
        }
        Ok(_) => {
            tracing::warn!("OP_RATE_LIMIT_PER_MINUTE=0 — disabling rate-limit");
            None
        }
        Err(err) => {
            tracing::warn!(
                value = %raw,
                error = %err,
                "OP_RATE_LIMIT_PER_MINUTE: failed to parse as u32 — rate-limit disabled"
            );
            None
        }
    }
}

/// Register a `usdc-base` [`CryptoAdapter`] on a fresh orchestrator
/// when both `OP_BASE_RPC_URL` and `OP_USDC_BASE_PRIVATE_KEY` are
/// set. Single-env-var configurations are explicitly rejected so a
/// half-configured operator doesn't silently launch without the rail.
fn maybe_register_usdc_base<F>(state: AppState, reader: &F) -> AppState
where
    F: Fn(&str) -> Option<String>,
{
    let rpc_url = reader("OP_BASE_RPC_URL").filter(|s| !s.is_empty());
    let priv_key = reader("OP_USDC_BASE_PRIVATE_KEY").filter(|s| !s.is_empty());

    match (rpc_url, priv_key) {
        (Some(rpc), Some(key)) => match build_usdc_base_adapter(&rpc, &key) {
            Ok(adapter) => {
                tracing::info!(
                    "OP_BASE_RPC_URL + OP_USDC_BASE_PRIVATE_KEY: usdc-base rail registered"
                );
                let mut orch = Orchestrator::new();
                orch.register_adapter(Arc::new(adapter));
                state.with_orchestrator(orch)
            }
            Err(err) => {
                tracing::warn!(error = %err, "usdc-base gateway construction failed — skipping");
                state
            }
        },
        (Some(_), None) => {
            tracing::warn!(
                "OP_BASE_RPC_URL is set but OP_USDC_BASE_PRIVATE_KEY is not — skipping usdc-base rail registration"
            );
            state
        }
        (None, Some(_)) => {
            tracing::warn!(
                "OP_USDC_BASE_PRIVATE_KEY is set but OP_BASE_RPC_URL is not — skipping usdc-base rail registration"
            );
            state
        }
        (None, None) => state,
    }
}

/// The concrete gateway type the binary registers under `usdc-base`.
/// Aliased so the binary's signer / gateway plumbing stays in one
/// place; if operators want a different signer (KMS / HSM /
/// Fireblocks) they construct the orchestrator directly and skip
/// [`build_state_from_env`].
type UsdcBaseGateway = op_rails_crypto::EvmJsonRpcGateway<op_rails_crypto::LocalKeyEvmSigner>;

fn build_usdc_base_adapter(
    rpc_url: &str,
    private_key_hex: &str,
) -> op_rails_crypto::Result<CryptoAdapter> {
    use op_rails_crypto::{EvmJsonRpcGateway, LocalKeyEvmSigner, StableToken};

    let signer = LocalKeyEvmSigner::from_hex_private_key(private_key_hex, rpc_url)?;
    let from_address = signer.address();
    let token = StableToken::UsdcBase.token_ref();
    let gateway: UsdcBaseGateway =
        EvmJsonRpcGateway::new("usdc-base", rpc_url, token, from_address, signer)?;
    Ok(CryptoAdapter::new("usdc-base", Arc::new(gateway)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// Build a closure that reads from a `HashMap`. Tests pass this
    /// instead of `std::env::var` so the global env stays untouched.
    fn map_reader(map: HashMap<&'static str, &'static str>) -> impl Fn(&str) -> Option<String> {
        move |k| map.get(k).map(|s| (*s).to_owned())
    }

    #[test]
    fn empty_env_returns_no_middleware() {
        let mw = build_middleware_from_env(map_reader(HashMap::new()));
        assert!(mw.auth.is_none());
        assert!(mw.rate_limit.is_none());
    }

    #[test]
    fn empty_env_returns_in_memory_state() {
        let state = build_state_from_env(map_reader(HashMap::new())).unwrap();
        // No crypto rail registered.
        assert!(
            !state
                .orchestrator
                .has_driver(op_core::RailKind::Crypto, "usdc-base")
        );
    }

    #[test]
    fn api_keys_csv_produces_auth_layer_with_three_keys() {
        let mut map = HashMap::new();
        map.insert("OP_API_KEYS", "alpha,beta,gamma");
        let mw = build_middleware_from_env(map_reader(map));
        assert!(mw.auth.is_some());
        assert!(mw.rate_limit.is_none());
    }

    #[test]
    fn api_keys_strips_whitespace_and_empties() {
        let mut map = HashMap::new();
        map.insert("OP_API_KEYS", " alpha , , beta ,");
        let mw = build_middleware_from_env(map_reader(map));
        assert!(mw.auth.is_some());
    }

    #[test]
    fn api_keys_empty_string_yields_none() {
        let mut map = HashMap::new();
        map.insert("OP_API_KEYS", "");
        let mw = build_middleware_from_env(map_reader(map));
        assert!(mw.auth.is_none());
    }

    #[test]
    fn rate_limit_per_minute_parses_u32() {
        let mut map = HashMap::new();
        map.insert("OP_RATE_LIMIT_PER_MINUTE", "600");
        let mw = build_middleware_from_env(map_reader(map));
        assert!(mw.rate_limit.is_some());
    }

    #[test]
    fn rate_limit_garbage_yields_none() {
        let mut map = HashMap::new();
        map.insert("OP_RATE_LIMIT_PER_MINUTE", "not_a_number");
        let mw = build_middleware_from_env(map_reader(map));
        assert!(mw.rate_limit.is_none());
    }

    #[test]
    fn rate_limit_zero_yields_none() {
        let mut map = HashMap::new();
        map.insert("OP_RATE_LIMIT_PER_MINUTE", "0");
        let mw = build_middleware_from_env(map_reader(map));
        assert!(mw.rate_limit.is_none());
    }

    #[test]
    fn both_middleware_set_returns_both() {
        let mut map = HashMap::new();
        map.insert("OP_API_KEYS", "one,two");
        map.insert("OP_RATE_LIMIT_PER_MINUTE", "120");
        let mw = build_middleware_from_env(map_reader(map));
        assert!(mw.auth.is_some());
        assert!(mw.rate_limit.is_some());
    }

    #[test]
    fn only_rpc_url_skips_gateway() {
        let mut map = HashMap::new();
        map.insert("OP_BASE_RPC_URL", "http://localhost:8545");
        let state = build_state_from_env(map_reader(map)).unwrap();
        assert!(
            !state
                .orchestrator
                .has_driver(op_core::RailKind::Crypto, "usdc-base")
        );
    }

    #[test]
    fn only_private_key_skips_gateway() {
        let mut map = HashMap::new();
        map.insert(
            "OP_USDC_BASE_PRIVATE_KEY",
            "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        );
        let state = build_state_from_env(map_reader(map)).unwrap();
        assert!(
            !state
                .orchestrator
                .has_driver(op_core::RailKind::Crypto, "usdc-base")
        );
    }

    #[test]
    fn evm_env_vars_register_usdc_base_gateway() {
        // Well-known Anvil/Hardhat test key #0; never used on
        // mainnet, no real funds attached.
        let mut map = HashMap::new();
        map.insert("OP_BASE_RPC_URL", "http://localhost:8545");
        map.insert(
            "OP_USDC_BASE_PRIVATE_KEY",
            "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80",
        );
        let state = build_state_from_env(map_reader(map)).unwrap();
        assert!(
            state
                .orchestrator
                .has_driver(op_core::RailKind::Crypto, "usdc-base"),
            "expected usdc-base crypto adapter registered when both env vars set"
        );
    }

    #[test]
    fn evm_invalid_key_skips_gateway_with_warning() {
        let mut map = HashMap::new();
        map.insert("OP_BASE_RPC_URL", "http://localhost:8545");
        // Too short → DriverValidation in LocalKeyEvmSigner.
        map.insert("OP_USDC_BASE_PRIVATE_KEY", "0xdeadbeef");
        let state = build_state_from_env(map_reader(map)).unwrap();
        assert!(
            !state
                .orchestrator
                .has_driver(op_core::RailKind::Crypto, "usdc-base")
        );
    }
}
