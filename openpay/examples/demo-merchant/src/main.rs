//! `demo-merchant` — the smallest end-to-end `OpenPay` payment demo
//! that a developer can run on a laptop in five minutes.
//!
//! The binary:
//!
//! 1. Generates a fresh secp256k1 keypair via `k256` and derives the
//!    20-byte EVM address from it (keccak256 of the uncompressed
//!    public key, last 20 bytes — same rule used by every EVM tool
//!    on Earth).
//! 2. Prints the address and a block-explorer link so the developer
//!    knows where to send funds.
//! 3. Spawns an `axum` HTTP server on the listen address. A single
//!    route, `POST /webhook`, accepts the raw event body that
//!    op-server's webhook fanout posts. If we see a relevant intent
//!    event, we print `PAID` and exit.
//! 4. Independently, runs a polling loop that fires `eth_call` for
//!    `balanceOf(address)` against the configured Base RPC every 30
//!    seconds. The first time the balance crosses zero, we print
//!    `PAID` and exit.
//!
//! Whichever path detects payment first wins. The exit is clean (0)
//! so a calling script can chain it.
//!
//! This crate is not a production component. It intentionally has no
//! durability, no retry, no auth on the webhook port — it's a
//! demo. Real money flowing through here on mainnet is real money
//! sent to a freshly-generated key that the demo controls; the
//! private key is **never persisted**, so anything you send to this
//! address while the demo is running is recoverable only by you (the
//! sender) if you can rewind the on-chain state, which you can't.
//! Use `--testnet` for the first run.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]

use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Duration;

use axum::{
    Router,
    extract::State,
    http::{HeaderMap, StatusCode},
    routing::post,
};
use clap::Parser;
use k256::ecdsa::SigningKey;
use op_rails_crypto::StableToken;
use serde_json::{Value, json};
use sha3::{Digest, Keccak256};
use tokio::sync::oneshot;

/// CLI configuration.
#[derive(Clone, Debug, Parser)]
#[command(
    name = "demo-merchant",
    about = "Local pilot-merchant demo. Generates a fresh wallet, listens for op-server webhooks, polls Base for USDC balance changes, exits on first payment.",
    version
)]
struct Cli {
    /// EVM JSON-RPC URL. Defaults to Base mainnet.
    #[arg(long, default_value = "https://mainnet.base.org")]
    rpc_url: String,
    /// Where the local webhook receiver binds. op-server should be
    /// configured to POST to `http://<this>/webhook`.
    #[arg(long, default_value = "127.0.0.1:9090")]
    listen: SocketAddr,
    /// Display-only: amount the merchant is expecting (in USDC).
    /// Used solely in the "send me $X" instruction line printed at
    /// startup; the demo exits on **any** non-zero balance.
    #[arg(long, default_value_t = 0.10)]
    amount_usdc: f64,
    /// Block-explorer base URL. Used for the address-link printout.
    #[arg(long, default_value = "https://basescan.org")]
    explorer: String,
    /// Convenience: switch RPC + explorer to Base Sepolia for safer
    /// first runs. Equivalent to passing
    /// `--rpc-url https://sepolia.base.org --explorer https://sepolia.basescan.org`.
    #[arg(long, default_value_t = false)]
    testnet: bool,
    /// Polling interval for `balanceOf`, in seconds.
    #[arg(long, default_value_t = 30)]
    poll_secs: u64,
}

/// Cross-task signal used to stop the process cleanly when a
/// payment is observed by either the webhook receiver or the
/// balance-poll loop.
#[derive(Clone)]
struct ExitSignal {
    tx: Arc<tokio::sync::Mutex<Option<oneshot::Sender<&'static str>>>>,
}

impl ExitSignal {
    fn new(tx: oneshot::Sender<&'static str>) -> Self {
        Self {
            tx: Arc::new(tokio::sync::Mutex::new(Some(tx))),
        }
    }

    /// Fire the exit signal. The first caller wins; subsequent
    /// callers are a no-op. Returns true if this call was the one
    /// that fired the signal (so we don't print PAID twice).
    async fn fire(&self, reason: &'static str) -> bool {
        let mut guard = self.tx.lock().await;
        if let Some(tx) = guard.take() {
            let _ = tx.send(reason);
            true
        } else {
            false
        }
    }
}

/// State shared with the axum router.
#[derive(Clone)]
struct AppState {
    wallet_address_lc: String,
    exit: ExitSignal,
}

/// Derive a 20-byte EVM address from a secp256k1 signing key, using
/// the canonical rule:
///
/// ```text
///   addr = keccak256(uncompressed_pubkey[1..])[12..]
/// ```
///
/// The leading byte of the SEC1 uncompressed encoding (0x04) is
/// stripped before hashing — EVM hashes the raw 64-byte X||Y
/// concatenation.
#[must_use]
pub fn evm_address_from_signing_key(sk: &SigningKey) -> [u8; 20] {
    let vk = sk.verifying_key();
    let encoded = vk.to_encoded_point(false);
    let pubkey_bytes = encoded.as_bytes();
    debug_assert_eq!(pubkey_bytes.len(), 65, "SEC1 uncompressed is 65 bytes");
    debug_assert_eq!(pubkey_bytes[0], 0x04, "SEC1 uncompressed prefix");

    let mut hasher = Keccak256::new();
    hasher.update(&pubkey_bytes[1..]);
    let digest = hasher.finalize();
    let mut out = [0u8; 20];
    out.copy_from_slice(&digest[12..]);
    out
}

/// Format a 20-byte address as `0x`-prefixed lowercase hex.
#[must_use]
pub fn format_address(addr: &[u8; 20]) -> String {
    format!("0x{}", hex::encode(addr))
}

/// Build the `eth_call` calldata for ERC-20 `balanceOf(address)`.
/// Selector `0x70a08231` + 32-byte left-padded recipient.
#[must_use]
pub fn encode_erc20_balance_of(addr: &[u8; 20]) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + 32);
    out.extend_from_slice(&[0x70, 0xa0, 0x82, 0x31]);
    out.extend_from_slice(&[0u8; 12]);
    out.extend_from_slice(addr);
    out
}

/// Parse the JSON-RPC `"0x..."` hex result of `eth_call` /
/// `balanceOf` into a `u128`. Returns 0 for the zero-length /
/// all-zeros case rather than erroring; the on-chain encoding is
/// uint256 but USDC's total supply is many orders of magnitude
/// below `u128::MAX`, so we safely narrow.
fn parse_balance_hex(v: &Value) -> Result<u128, String> {
    let s = v
        .as_str()
        .ok_or_else(|| format!("expected hex string, got {v}"))?;
    let stripped = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .ok_or_else(|| format!("missing 0x prefix: {s}"))?;
    if stripped.is_empty() || stripped.chars().all(|c| c == '0') {
        return Ok(0);
    }
    // Strip leading zeros, then parse. uint256 in hex is 64 chars;
    // u128 takes 32. If the high 32 chars are non-zero we cap at
    // `u128::MAX` rather than fail — a balance that large is a sign
    // someone is testing with mainnet WBTC or similar, not USDC.
    let trimmed = stripped.trim_start_matches('0');
    if trimmed.len() > 32 {
        return Ok(u128::MAX);
    }
    u128::from_str_radix(trimmed, 16).map_err(|e| format!("invalid hex: {e}"))
}

/// One `balanceOf` RPC call. Returns the balance in minor units
/// (i.e. 1 USDC = `1_000_000` because USDC has 6 decimals).
async fn fetch_usdc_balance(
    client: &reqwest::Client,
    rpc_url: &str,
    usdc_contract: &str,
    address_hex: &str,
) -> Result<u128, String> {
    let addr_bytes = parse_evm_address_lower(address_hex)?;
    let data = encode_erc20_balance_of(&addr_bytes);
    let body = json!({
        "jsonrpc": "2.0",
        "id": 1,
        "method": "eth_call",
        "params": [
            { "to": usdc_contract, "data": format!("0x{}", hex::encode(data)) },
            "latest",
        ],
    });
    let resp = client
        .post(rpc_url)
        .header("content-type", "application/json")
        .body(body.to_string())
        .send()
        .await
        .map_err(|e| format!("rpc transport: {e}"))?;
    if !resp.status().is_success() {
        return Err(format!("rpc http {}", resp.status()));
    }
    let parsed: Value = resp.json().await.map_err(|e| format!("rpc parse: {e}"))?;
    if let Some(err) = parsed.get("error") {
        return Err(format!("rpc error: {err}"));
    }
    let result = parsed
        .get("result")
        .ok_or_else(|| "rpc missing result".to_string())?;
    parse_balance_hex(result)
}

/// Local mirror of the EVM-address parser in `op-rails-crypto::evm`
/// (private there). 20-byte hex, `0x` prefix, lowercase tolerated.
fn parse_evm_address_lower(s: &str) -> Result<[u8; 20], String> {
    let stripped = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .ok_or_else(|| format!("address missing 0x prefix: {s}"))?;
    if stripped.len() != 40 {
        return Err(format!(
            "address must be 40 hex chars, got {}",
            stripped.len()
        ));
    }
    let raw = hex::decode(stripped).map_err(|e| format!("hex decode: {e}"))?;
    let mut out = [0u8; 20];
    out.copy_from_slice(&raw);
    Ok(out)
}

/// Format a USDC minor-unit balance as a decimal string with 6
/// decimal places. `1_000_000` → `"1.000000"`.
#[must_use]
pub fn format_usdc(minor_units: u128) -> String {
    let whole = minor_units / 1_000_000;
    let frac = minor_units % 1_000_000;
    format!("{whole}.{frac:06}")
}

/// Webhook handler.
///
/// op-server's webhook fanout POSTs the **raw event payload** as the
/// body, with `OpenPay-Event-Type` in the headers. We accept any
/// body (text or JSON), look for an event type that smells like a
/// payment event, and if it carries our wallet address as the
/// destination, fire the exit signal.
///
/// For the demo we deliberately accept fairly broad matches —
/// `intent.approved`, `payment.captured`, `intent.captured` — because
/// the exact event-name taxonomy is operator-defined. The webhook
/// path is **best-effort**; the balance-poll loop is the
/// ground-truth check.
async fn webhook_handler(
    State(state): State<AppState>,
    headers: HeaderMap,
    body: axum::body::Bytes,
) -> StatusCode {
    let event_type = headers
        .get("openpay-event-type")
        .and_then(|h| h.to_str().ok())
        .unwrap_or("<unknown>");

    tracing::info!(event_type, body_bytes = body.len(), "webhook received");
    println!("  webhook: {event_type} ({} bytes)", body.len());

    // Try to JSON-decode and look for our wallet address anywhere in
    // the payload. We don't have a canonical schema here — different
    // event types nest the destination at different paths — so a
    // substring match on the lowercase serialized JSON is the
    // safest "did this event involve our wallet?" heuristic.
    let body_str = String::from_utf8_lossy(&body).to_ascii_lowercase();
    let mentions_our_wallet = body_str.contains(&state.wallet_address_lc);

    let looks_like_payment = matches!(
        event_type,
        "intent.approved"
            | "intent.captured"
            | "payment.captured"
            | "payment.authorized"
            | "settlement.posted"
    );

    if looks_like_payment && mentions_our_wallet {
        let fired = state
            .exit
            .fire("webhook reported payment to our wallet")
            .await;
        if fired {
            println!(
                "PAID: webhook '{event_type}' targets {} — exiting.",
                state.wallet_address_lc
            );
        }
    }

    StatusCode::NO_CONTENT
}

/// Background poll loop: hits `balanceOf` on the configured USDC
/// contract every `poll_secs`, and fires the exit signal the first
/// time the balance is non-zero. Emits a heartbeat log every poll.
async fn poll_loop(
    rpc_url: String,
    usdc_contract: String,
    address_hex: String,
    poll_secs: u64,
    exit: ExitSignal,
) {
    let client = match reqwest::Client::builder()
        .timeout(Duration::from_secs(15))
        .build()
    {
        Ok(c) => c,
        Err(e) => {
            eprintln!("reqwest client init failed: {e}");
            return;
        }
    };

    let mut interval = tokio::time::interval(Duration::from_secs(poll_secs.max(1)));
    interval.set_missed_tick_behavior(tokio::time::MissedTickBehavior::Skip);

    loop {
        interval.tick().await;
        match fetch_usdc_balance(&client, &rpc_url, &usdc_contract, &address_hex).await {
            Ok(0) => {
                println!("  poll: balance = 0 USDC, still waiting...");
            }
            Ok(bal) => {
                let fired = exit.fire("balance-poll observed non-zero balance").await;
                if fired {
                    println!(
                        "PAID: {} USDC arrived at {address_hex} (per balanceOf) — exiting.",
                        format_usdc(bal)
                    );
                }
                return;
            }
            Err(e) => {
                eprintln!("  poll: rpc error: {e}");
            }
        }
    }
}

#[tokio::main(flavor = "multi_thread", worker_threads = 2)]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Default to INFO; honour RUST_LOG if set.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let mut cli = Cli::parse();
    if cli.testnet {
        // Only override if the user left the defaults; respect
        // explicit `--rpc-url` even when `--testnet` is set.
        if cli.rpc_url == "https://mainnet.base.org" {
            cli.rpc_url = "https://sepolia.base.org".into();
        }
        if cli.explorer == "https://basescan.org" {
            cli.explorer = "https://sepolia.basescan.org".into();
        }
    }

    // 1) Generate a fresh keypair and derive the address.
    //    `SigningKey::random` uses `OsRng` internally.
    let sk = SigningKey::random(&mut k256::elliptic_curve::rand_core::OsRng);
    let addr_bytes = evm_address_from_signing_key(&sk);
    let address = format_address(&addr_bytes);

    // 2) Banner.
    let usdc = StableToken::UsdcBase.token_ref();
    println!("OpenPay demo merchant");
    println!("---------------------");
    println!("Merchant wallet: {address}");
    println!("RPC:             {}", cli.rpc_url);
    println!("USDC contract:   {} ({})", usdc.contract, usdc.symbol);
    println!("Explorer:        {}/address/{address}", cli.explorer);
    println!();
    println!(
        "Send EXACTLY {:.2} USDC (or anything non-zero) to the above address",
        cli.amount_usdc
    );
    println!("to trigger the 'paid' notification.");
    println!();
    println!(
        "Listening on http://{} for op-server webhooks at /webhook",
        cli.listen
    );
    println!(
        "Polling {} every {}s for balanceOf updates.",
        cli.rpc_url, cli.poll_secs
    );
    println!();

    // 3) Shared exit signal.
    let (exit_tx, exit_rx) = oneshot::channel::<&'static str>();
    let exit = ExitSignal::new(exit_tx);

    let state = AppState {
        wallet_address_lc: address.clone(),
        exit: exit.clone(),
    };

    let app = Router::new()
        .route("/webhook", post(webhook_handler))
        .with_state(state);

    let listener = tokio::net::TcpListener::bind(cli.listen).await?;

    let server_exit = exit.clone();
    let server_task = tokio::spawn(async move {
        // axum 0.8: serve with graceful shutdown driven by the
        // exit signal. We can't share the oneshot Receiver between
        // tasks, so the server watches a separate channel that
        // fires when the exit signal fires.
        let (shutdown_tx, shutdown_rx) = oneshot::channel::<()>();

        // Bridge: when the exit signal fires, tell the server.
        tokio::spawn(async move {
            // Poll the exit signal by trying to fire a no-op.
            // Simpler: just wait for the global oneshot side via
            // a notification (we don't have one). So we share a
            // notify pattern through a Mutex<Option<_>> on
            // server_exit — but that's already held. Take the
            // pragmatic route: poll every 250ms.
            loop {
                tokio::time::sleep(Duration::from_millis(250)).await;
                if server_exit.tx.lock().await.is_none() {
                    let _ = shutdown_tx.send(());
                    return;
                }
            }
        });

        let _ = axum::serve(listener, app)
            .with_graceful_shutdown(async move {
                let _ = shutdown_rx.await;
            })
            .await;
    });

    // 4) Background poll loop.
    let poll_task = tokio::spawn(poll_loop(
        cli.rpc_url.clone(),
        usdc.contract.clone(),
        address.clone(),
        cli.poll_secs,
        exit.clone(),
    ));

    // 5) Wait for whichever path fires the exit signal.
    let reason = exit_rx.await.unwrap_or("exit channel closed");
    tracing::info!(reason, "shutting down");

    // Best-effort shutdown of both tasks.
    poll_task.abort();
    let _ = tokio::time::timeout(Duration::from_secs(2), server_task).await;

    println!("Exiting cleanly.");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Address-derivation conformance: the canonical
    /// `0x0000000000000000000000000000000000000001` private key
    /// (i.e. scalar 1, the secp256k1 generator) derives to the
    /// well-known address `0x7e5f4552091a69125d5dfcb7b8c2659029395bdf`.
    ///
    /// This pin protects against accidental endianness / encoding
    /// regressions in the keccak / SEC1 path.
    #[test]
    fn known_private_key_derives_canonical_address() {
        // 32-byte big-endian scalar = 1.
        let mut sk_bytes = [0u8; 32];
        sk_bytes[31] = 1;
        let sk = SigningKey::from_bytes((&sk_bytes).into()).expect("valid scalar");
        let addr = evm_address_from_signing_key(&sk);
        let s = format_address(&addr);
        assert_eq!(s, "0x7e5f4552091a69125d5dfcb7b8c2659029395bdf");
    }

    /// Sanity-check the output shape of a randomly-generated wallet.
    /// Must be `0x` + 40 lowercase hex chars.
    #[test]
    fn random_wallet_shape_is_42_chars_lowercase() {
        let sk = SigningKey::random(&mut k256::elliptic_curve::rand_core::OsRng);
        let addr = format_address(&evm_address_from_signing_key(&sk));
        assert_eq!(addr.len(), 42);
        assert!(addr.starts_with("0x"));
        for c in addr.chars().skip(2) {
            assert!(
                c.is_ascii_hexdigit() && !c.is_ascii_uppercase(),
                "non-lowercase-hex char {c} in {addr}"
            );
        }
    }

    #[test]
    fn balance_of_calldata_layout() {
        let mut addr = [0u8; 20];
        addr[19] = 1;
        let data = encode_erc20_balance_of(&addr);
        assert_eq!(data.len(), 36);
        // selector
        assert_eq!(&data[..4], &[0x70, 0xa0, 0x82, 0x31]);
        // 12 bytes of left-pad zero
        for b in &data[4..16] {
            assert_eq!(*b, 0);
        }
        // 19 bytes of zero in the address slot, then the 0x01.
        for b in &data[16..35] {
            assert_eq!(*b, 0);
        }
        assert_eq!(data[35], 0x01);
    }

    #[test]
    fn parse_balance_hex_zero_variants() {
        assert_eq!(parse_balance_hex(&Value::String("0x".into())).unwrap(), 0);
        assert_eq!(parse_balance_hex(&Value::String("0x0".into())).unwrap(), 0);
        assert_eq!(
            parse_balance_hex(&Value::String(
                "0x0000000000000000000000000000000000000000000000000000000000000000".into()
            ))
            .unwrap(),
            0
        );
    }

    #[test]
    fn parse_balance_hex_one_usdc() {
        // 1 USDC = 1_000_000 minor units = 0xf4240.
        let v = Value::String(
            "0x00000000000000000000000000000000000000000000000000000000000f4240".into(),
        );
        assert_eq!(parse_balance_hex(&v).unwrap(), 1_000_000);
    }

    #[test]
    fn parse_balance_hex_overflow_saturates() {
        // 33 non-zero hex chars after leading zeros — past u128.
        let mut s = String::from("0x");
        s.push_str(&"0".repeat(64 - 33));
        s.push_str(&"f".repeat(33));
        assert_eq!(parse_balance_hex(&Value::String(s)).unwrap(), u128::MAX);
    }

    #[test]
    fn format_usdc_pads_decimals() {
        assert_eq!(format_usdc(0), "0.000000");
        assert_eq!(format_usdc(1_000_000), "1.000000");
        assert_eq!(format_usdc(123_456), "0.123456");
        assert_eq!(format_usdc(5_000_001), "5.000001");
    }

    #[test]
    fn parse_evm_address_lower_round_trips() {
        let addr = "0x7e5f4552091a69125d5dfcb7b8c2659029395bdf";
        let bytes = parse_evm_address_lower(addr).unwrap();
        let again = format_address(&bytes);
        assert_eq!(again, addr);
    }
}
