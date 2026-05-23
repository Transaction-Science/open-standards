//! JSON-RPC backed EVM `CryptoGateway`.
//!
//! `EvmJsonRpcGateway` is the production counterpart to
//! `DeterministicCryptoGateway`: it actually broadcasts ERC-20
//! transfers on Ethereum / Base / Polygon / Arbitrum.
//!
//! Architecture: the gateway owns a `reqwest::blocking::Client` and
//! the chain's JSON-RPC URL. It builds the ERC-20 `transfer(...)`
//! calldata, fetches nonce / gas / chain-id via JSON-RPC, then hands
//! the resulting [`UnsignedTx`] to an operator-supplied
//! [`EvmSigner`]. The signer signs and broadcasts; the gateway
//! returns the resulting transaction hash.
//!
//! The gateway is intentionally thin. It does NOT:
//! - Sign anything (operator's signer does).
//! - Track confirmation depth across blocks (operator policy lives
//!   above this layer).
//! - Decode ERC-20 events from the receipt (the `status` field is
//!   the canonical "did the transfer succeed?" signal — events are
//!   for richer indexing, not rail-level decision-making).

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use op_core::CryptoAddress;
use reqwest::blocking::Client;
use serde_json::{Value, json};
use sha3::{Digest, Keccak256};

use crate::error::{Error, Result};
use crate::gateway::{
    CryptoDecision, CryptoGateway, CryptoStatus, CryptoTransferReq, StatusQueryReq,
};
use crate::signer::{EvmSigner, UnsignedTx};
use crate::token::TokenRef;

/// Conservative default per-request HTTP timeout. Public RPCs can
/// stall, but a payment gateway shouldn't block its caller for
/// minutes; failed calls surface as [`Error::Transport`] and the
/// orchestrator's retry layer takes over.
const DEFAULT_TIMEOUT_SECS: u64 = 15;

/// Default gas-limit multiplier on `eth_estimateGas`. Chain dynamics
/// (storage warm/cold, occasional gas-cost EIPs) move the actual
/// required gas around; a 20% buffer absorbs that without bleeding
/// the operator's fee budget.
const GAS_LIMIT_BUFFER_NUM: u64 = 12;
const GAS_LIMIT_BUFFER_DEN: u64 = 10;

/// ERC-20 `transfer(address,uint256)` 4-byte function selector.
/// `keccak256("transfer(address,uint256)")[..4]` = `0xa9059cbb`.
const ERC20_TRANSFER_SELECTOR: [u8; 4] = [0xa9, 0x05, 0x9c, 0xbb];

/// EVM-rail gateway over a JSON-RPC endpoint.
///
/// Construct via [`Self::new`] with the operator's RPC URL, the
/// `(chain, token)` this driver services, the from-address the
/// signer controls, and the signer itself. The `name` is what the
/// orchestrator's policy router uses to key on.
pub struct EvmJsonRpcGateway<S: EvmSigner> {
    name: &'static str,
    rpc_url: String,
    token: TokenRef,
    from_address: String,
    signer: S,
    client: Client,
    /// Monotonic counter for JSON-RPC `id` fields. Helps trace
    /// individual calls in tcpdump / RPC-proxy logs.
    rpc_id: AtomicU64,
}

impl<S: EvmSigner> EvmJsonRpcGateway<S> {
    /// Construct a gateway. `name` is the orchestrator-facing key
    /// (`"usdc-base-prod"` etc.), `rpc_url` is the operator's
    /// JSON-RPC endpoint, `token` is the `(chain, contract,
    /// decimals, symbol)` triple, `from_address` is the signer's
    /// public address, and `signer` is the operator-supplied
    /// [`EvmSigner`].
    ///
    /// `token.chain` must be one of `"ethereum"`, `"base"`,
    /// `"polygon"`, `"arbitrum"`. The constructor does NOT validate
    /// this — gateways for new EVM chains can be added without
    /// modifying this crate, the chain name is just a lookup key
    /// used in `req.to.chain` matching.
    ///
    /// # Errors
    /// Returns [`Error::Transport`] if the reqwest client fails to
    /// initialize (TLS backend failure, malformed timeout).
    pub fn new(
        name: &'static str,
        rpc_url: impl Into<String>,
        token: TokenRef,
        from_address: impl Into<String>,
        signer: S,
    ) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .build()
            .map_err(|e| Error::Transport(format!("reqwest client build failed: {e}")))?;
        Ok(Self {
            name,
            rpc_url: rpc_url.into(),
            token,
            from_address: from_address.into().to_ascii_lowercase(),
            signer,
            client,
            rpc_id: AtomicU64::new(1),
        })
    }

    /// Builder: override the default HTTP timeout. Mostly useful for
    /// tests against in-process mock servers, where the default
    /// 15-second timeout is overkill.
    ///
    /// # Errors
    /// Returns [`Error::Transport`] if the client rebuild fails.
    pub fn with_timeout(mut self, timeout: Duration) -> Result<Self> {
        self.client = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| Error::Transport(format!("reqwest client build failed: {e}")))?;
        Ok(self)
    }

    fn next_rpc_id(&self) -> u64 {
        self.rpc_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Issue a JSON-RPC call and return the `result` field as a
    /// `serde_json::Value`. Transport / parse / RPC errors all map
    /// to [`Error::Transport`] or [`Error::Rejected`] depending on
    /// whether the chain responded at all.
    fn rpc_call(&self, method: &str, params: &Value) -> Result<Value> {
        let body = json!({
            "jsonrpc": "2.0",
            "id": self.next_rpc_id(),
            "method": method,
            "params": params,
        });
        let resp = self
            .client
            .post(&self.rpc_url)
            .header("content-type", "application/json")
            .body(body.to_string())
            .send()
            .map_err(|e| Error::Transport(format!("rpc {method}: {e}")))?;
        if !resp.status().is_success() {
            return Err(Error::Transport(format!(
                "rpc {method}: http {}",
                resp.status()
            )));
        }
        let parsed: Value = resp
            .json()
            .map_err(|e| Error::Transport(format!("rpc {method}: parse: {e}")))?;
        if let Some(err_obj) = parsed.get("error") {
            // RPC-level error: surfaces as Rejected so the
            // orchestrator marks the transfer terminal. (Transport
            // failures would have short-circuited earlier.)
            let code = err_obj
                .get("code")
                .and_then(Value::as_i64)
                .map_or_else(|| "unknown".into(), |c| c.to_string());
            let message = err_obj
                .get("message")
                .and_then(Value::as_str)
                .unwrap_or("rpc error")
                .to_owned();
            return Err(Error::Rejected { code, message });
        }
        parsed
            .get("result")
            .cloned()
            .ok_or_else(|| Error::Transport(format!("rpc {method}: missing result")))
    }

    fn fetch_chain_id(&self) -> Result<u64> {
        let v = self.rpc_call("eth_chainId", &json!([]))?;
        parse_hex_u64(&v).map_err(|reason| Error::Transport(format!("eth_chainId: {reason}")))
    }

    fn fetch_nonce(&self) -> Result<u64> {
        let v = self.rpc_call(
            "eth_getTransactionCount",
            &json!([&self.from_address, "pending"]),
        )?;
        parse_hex_u64(&v)
            .map_err(|reason| Error::Transport(format!("eth_getTransactionCount: {reason}")))
    }

    fn fetch_gas_price(&self) -> Result<u128> {
        let v = self.rpc_call("eth_gasPrice", &json!([]))?;
        parse_hex_u128(&v).map_err(|reason| Error::Transport(format!("eth_gasPrice: {reason}")))
    }

    fn estimate_gas(&self, to_contract: &str, data_hex: &str) -> Result<u64> {
        let v = self.rpc_call(
            "eth_estimateGas",
            &json!([{
                "from": &self.from_address,
                "to": to_contract,
                "data": data_hex,
            }]),
        )?;
        let raw = parse_hex_u64(&v)
            .map_err(|reason| Error::Transport(format!("eth_estimateGas: {reason}")))?;
        // 20% safety margin.
        Ok(raw.saturating_mul(GAS_LIMIT_BUFFER_NUM) / GAS_LIMIT_BUFFER_DEN)
    }

    fn fetch_receipt(&self, tx_hash: &str) -> Result<Option<Value>> {
        let v = self.rpc_call("eth_getTransactionReceipt", &json!([tx_hash]))?;
        if v.is_null() { Ok(None) } else { Ok(Some(v)) }
    }
}

impl<S: EvmSigner> CryptoGateway for EvmJsonRpcGateway<S> {
    fn name(&self) -> &'static str {
        self.name
    }

    fn chain(&self) -> &str {
        &self.token.chain
    }

    fn token(&self) -> &TokenRef {
        &self.token
    }

    fn supports(&self, to: &CryptoAddress) -> bool {
        to.chain == self.token.chain
    }

    fn submit_transfer(&self, req: &CryptoTransferReq) -> Result<CryptoDecision> {
        // 1) Chain match.
        if req.to.chain != self.token.chain {
            return Err(Error::UnsupportedChain(req.to.chain.clone()));
        }
        // 2) Token-contract match (compare case-insensitively;
        // EVM addresses round-trip through mixed-case checksums).
        if !eq_ignore_ascii(&req.token.contract, &self.token.contract) {
            return Err(Error::UnsupportedToken(req.token.symbol.clone()));
        }
        // 3) Validate destination address as EVM hex (20 bytes).
        let dest_bytes =
            parse_evm_address(&req.to.address).map_err(|reason| Error::InvalidAddress {
                chain: req.to.chain.clone(),
                reason,
            })?;
        // 4) Validate amount: must be non-negative (negative minor
        // units are nonsensical for a transfer; the orchestrator
        // catches this earlier but we re-check at the rail boundary).
        let amount = u128::try_from(req.amount.minor_units).map_err(|_| {
            Error::DriverValidation(format!("negative amount: {}", req.amount.minor_units))
        })?;

        // 5) Build calldata.
        let data_bytes = encode_erc20_transfer(&dest_bytes, amount);
        let data_hex = format!("0x{}", hex::encode(&data_bytes));

        // 6) Fetch chain state.
        let chain_id = self.fetch_chain_id()?;
        let nonce = self.fetch_nonce()?;
        let gas_price = self.fetch_gas_price()?;
        let gas_limit = self.estimate_gas(&self.token.contract, &data_hex)?;

        // 7) Hand off to signer.
        let unsigned = UnsignedTx {
            chain_id,
            nonce,
            gas_price,
            gas_limit,
            to: self.token.contract.clone(),
            value: 0,
            data: data_hex,
            from: self.from_address.clone(),
        };
        let tx_hash = self.signer.sign_and_broadcast(unsigned)?;

        Ok(CryptoDecision {
            status: CryptoStatus::Pending,
            tx_hash: Some(tx_hash),
            confirmations: 0,
            settled_amount: None,
            raw_status: Some("submitted".to_owned()),
            reason: None,
        })
    }

    fn query_status(&self, req: &StatusQueryReq) -> Result<CryptoDecision> {
        // The orchestrator may route a query meant for another
        // chain through every gateway in its registry. Reject early.
        if req.chain != self.token.chain {
            return Err(Error::UnsupportedChain(req.chain.clone()));
        }
        let Some(receipt) = self.fetch_receipt(&req.tx_hash)? else {
            // Not mined yet (or never broadcast).
            return Ok(CryptoDecision {
                status: CryptoStatus::Pending,
                tx_hash: Some(req.tx_hash.clone()),
                confirmations: 0,
                settled_amount: None,
                raw_status: Some("pending".to_owned()),
                reason: None,
            });
        };

        let raw_status = receipt.get("status").and_then(Value::as_str).unwrap_or("");
        // EVM JSON-RPC: 0x1 = success, 0x0 = failure (revert).
        // Some L2s and older Geth versions return without the
        // leading zero; normalize.
        let ok = matches!(raw_status, "0x1" | "0x01" | "0x001");
        let failed = matches!(raw_status, "0x0" | "0x00" | "0x000");

        if ok {
            // The driver returns Finalized as soon as the chain
            // mined the tx successfully. Operator-side confirmation
            // depth lives outside this layer; the operator's
            // policy layer can re-query for additional confidence.
            Ok(CryptoDecision {
                status: CryptoStatus::Finalized,
                tx_hash: Some(req.tx_hash.clone()),
                confirmations: 1,
                settled_amount: None,
                raw_status: Some(raw_status.to_owned()),
                reason: None,
            })
        } else if failed {
            let reason = extract_revert_reason(&receipt);
            Ok(CryptoDecision {
                status: CryptoStatus::Rejected,
                tx_hash: Some(req.tx_hash.clone()),
                confirmations: 0,
                settled_amount: None,
                raw_status: Some(raw_status.to_owned()),
                reason,
            })
        } else {
            // Unrecognized status string — surface as Pending so the
            // orchestrator keeps polling rather than declaring
            // either success or failure on garbage.
            Ok(CryptoDecision {
                status: CryptoStatus::Pending,
                tx_hash: Some(req.tx_hash.clone()),
                confirmations: 0,
                settled_amount: None,
                raw_status: Some(raw_status.to_owned()),
                reason: None,
            })
        }
    }
}

/// Encode an ERC-20 `transfer(address,uint256)` call.
///
/// Layout (68 bytes total):
/// - 4 bytes: function selector `0xa9059cbb`.
/// - 32 bytes: left-padded recipient address.
/// - 32 bytes: big-endian uint256 amount.
#[must_use]
pub fn encode_erc20_transfer(recipient_20: &[u8; 20], amount: u128) -> Vec<u8> {
    let mut out = Vec::with_capacity(4 + 32 + 32);
    out.extend_from_slice(&ERC20_TRANSFER_SELECTOR);
    // Address: 12 leading zero bytes + 20 address bytes.
    out.extend_from_slice(&[0u8; 12]);
    out.extend_from_slice(recipient_20);
    // Amount: 16 leading zero bytes + 16 big-endian u128 bytes.
    out.extend_from_slice(&[0u8; 16]);
    out.extend_from_slice(&amount.to_be_bytes());
    out
}

/// Compute the `keccak256("transfer(address,uint256)")` 4-byte
/// selector. Kept as a function (rather than only the const) so
/// tests can independently verify the derivation.
#[must_use]
pub fn erc20_transfer_selector() -> [u8; 4] {
    let mut hasher = Keccak256::new();
    hasher.update(b"transfer(address,uint256)");
    let digest = hasher.finalize();
    let mut sel = [0u8; 4];
    sel.copy_from_slice(&digest[..4]);
    sel
}

/// Parse a `0x`-prefixed hex EVM address into 20 raw bytes.
fn parse_evm_address(s: &str) -> std::result::Result<[u8; 20], String> {
    let stripped = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .ok_or_else(|| format!("address missing 0x prefix: {s}"))?;
    if stripped.len() != 40 {
        return Err(format!(
            "address must be 40 hex chars, got {} ({s})",
            stripped.len()
        ));
    }
    let raw = hex::decode(stripped).map_err(|e| format!("hex decode: {e}"))?;
    let mut out = [0u8; 20];
    out.copy_from_slice(&raw);
    Ok(out)
}

/// Case-insensitive ASCII equality. EVM addresses can appear in
/// lowercase, uppercase, or EIP-55 mixed-case checksums; comparing
/// them at the rail layer must be lenient.
fn eq_ignore_ascii(a: &str, b: &str) -> bool {
    a.eq_ignore_ascii_case(b)
}

/// Parse a JSON-RPC `"0x..."` hex string into a `u64`.
fn parse_hex_u64(v: &Value) -> std::result::Result<u64, String> {
    let s = v
        .as_str()
        .ok_or_else(|| format!("expected hex string, got {v}"))?;
    let stripped = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .ok_or_else(|| format!("missing 0x prefix: {s}"))?;
    if stripped.is_empty() {
        return Ok(0);
    }
    u64::from_str_radix(stripped, 16).map_err(|e| format!("invalid hex u64 {s}: {e}"))
}

/// Parse a JSON-RPC `"0x..."` hex string into a `u128`.
fn parse_hex_u128(v: &Value) -> std::result::Result<u128, String> {
    let s = v
        .as_str()
        .ok_or_else(|| format!("expected hex string, got {v}"))?;
    let stripped = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .ok_or_else(|| format!("missing 0x prefix: {s}"))?;
    if stripped.is_empty() {
        return Ok(0);
    }
    u128::from_str_radix(stripped, 16).map_err(|e| format!("invalid hex u128 {s}: {e}"))
}

/// Best-effort revert-reason extraction from a tx receipt. Geth /
/// Erigon return the reason inline via `revertReason` on some
/// chains; on others the operator has to call `eth_call` against
/// the failed transaction to recover it. We surface whatever the
/// node gives us and leave deeper introspection to operator code.
fn extract_revert_reason(receipt: &Value) -> Option<String> {
    receipt
        .get("revertReason")
        .and_then(Value::as_str)
        .map(str::to_owned)
        .or_else(|| {
            receipt
                .get("error")
                .and_then(Value::as_str)
                .map(str::to_owned)
        })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signer::{EvmSigner, TxHash, UnsignedTx};
    use crate::token::StableToken;
    use httpmock::prelude::*;
    use op_core::{CryptoAddress, Currency, Money};
    use std::sync::Mutex;

    /// Test signer: records the unsigned tx it was handed and
    /// returns a configurable hash.
    struct CapturingSigner {
        captured: Mutex<Option<UnsignedTx>>,
        next_hash: String,
    }

    impl CapturingSigner {
        fn new(next_hash: impl Into<String>) -> Self {
            Self {
                captured: Mutex::new(None),
                next_hash: next_hash.into(),
            }
        }

        fn captured(&self) -> Option<UnsignedTx> {
            self.captured.lock().expect("poisoned").clone()
        }
    }

    impl EvmSigner for CapturingSigner {
        fn sign_and_broadcast(&self, unsigned: UnsignedTx) -> Result<TxHash> {
            *self.captured.lock().expect("poisoned") = Some(unsigned);
            Ok(self.next_hash.clone())
        }
    }

    #[test]
    fn selector_matches_canonical() {
        assert_eq!(erc20_transfer_selector(), ERC20_TRANSFER_SELECTOR);
        assert_eq!(hex::encode(ERC20_TRANSFER_SELECTOR), "a9059cbb");
    }

    #[test]
    fn calldata_layout_is_canonical() {
        // recipient = 0xdeadbeef000000000000000000000000deadbeef (20 bytes)
        let mut recipient = [0u8; 20];
        recipient[..4].copy_from_slice(&[0xde, 0xad, 0xbe, 0xef]);
        recipient[16..20].copy_from_slice(&[0xde, 0xad, 0xbe, 0xef]);

        let amount: u128 = 1_000_000_000;
        let data = encode_erc20_transfer(&recipient, amount);
        let hex_str = hex::encode(&data);

        // Expected:
        //   selector              : a9059cbb
        //   address pad (12 bytes): 000000000000000000000000
        //   address (20 bytes)    : deadbeef000000000000000000000000deadbeef
        //   amount (32 bytes)     : 0...0 + 1_000_000_000 in big-endian hex
        //                           = 0000000000000000000000000000000000000000000000000000000 3b9aca00
        let expected = format!(
            "a9059cbb\
             000000000000000000000000\
             deadbeef000000000000000000000000deadbeef\
             {amount:064x}"
        );
        assert_eq!(hex_str, expected);
        // And: total length is 4 + 32 + 32 = 68 bytes.
        assert_eq!(data.len(), 68);
    }

    #[test]
    fn calldata_amount_max_u128_round_trips() {
        let recipient = [0xab; 20];
        let data = encode_erc20_transfer(&recipient, u128::MAX);
        // Last 16 bytes are 0xff * 16; bytes 36..52 are zero.
        for b in &data[36..52] {
            assert_eq!(*b, 0);
        }
        for b in &data[52..68] {
            assert_eq!(*b, 0xff);
        }
    }

    #[test]
    fn parse_evm_address_accepts_canonical() {
        let bytes = parse_evm_address("0x833589fcd6edb6e08f4c7c32d4f71b54bda02913").unwrap();
        assert_eq!(bytes[0], 0x83);
        assert_eq!(bytes[19], 0x13);
    }

    #[test]
    fn parse_evm_address_rejects_short() {
        let err = parse_evm_address("0x1234").unwrap_err();
        assert!(err.contains("40 hex"));
    }

    #[test]
    fn parse_evm_address_rejects_no_prefix() {
        let err = parse_evm_address("833589fcd6edb6e08f4c7c32d4f71b54bda02913").unwrap_err();
        assert!(err.contains("0x prefix"));
    }

    fn sample_transfer(token: &TokenRef, key: &str, amount: i64) -> CryptoTransferReq {
        CryptoTransferReq {
            token: token.clone(),
            to: CryptoAddress::new(
                token.chain.clone(),
                "0xdeadbeef000000000000000000000000deadbeef",
            ),
            amount: Money::from_minor(amount, Currency::USD),
            idempotency_key: key.into(),
            memo: None,
        }
    }

    /// Spin up an httpmock server that responds to the JSON-RPC
    /// calls a normal `submit_transfer` flow makes, in order.
    fn happy_path_server() -> MockServer {
        let server = MockServer::start();

        // Single mock that pattern-matches on the JSON-RPC method
        // name in the request body. httpmock matches by substring
        // when given `body_contains`, which is sufficient for our
        // four distinct method names.
        server.mock(|when, then| {
            when.method(POST).body_contains("\"eth_chainId\"");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"jsonrpc":"2.0","id":1,"result":"0x2105"}"#); // 8453 = Base
        });
        server.mock(|when, then| {
            when.method(POST)
                .body_contains("\"eth_getTransactionCount\"");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"jsonrpc":"2.0","id":1,"result":"0x2a"}"#); // 42
        });
        server.mock(|when, then| {
            when.method(POST).body_contains("\"eth_gasPrice\"");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"jsonrpc":"2.0","id":1,"result":"0x3b9aca00"}"#); // 1 gwei
        });
        server.mock(|when, then| {
            when.method(POST).body_contains("\"eth_estimateGas\"");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"jsonrpc":"2.0","id":1,"result":"0xea60"}"#); // 60000
        });

        server
    }

    #[test]
    fn happy_path_returns_pending_with_tx_hash() {
        let server = happy_path_server();
        let token = StableToken::UsdcBase.token_ref();
        let signer = CapturingSigner::new(
            "0x1111111111111111111111111111111111111111111111111111111111111111",
        );
        let gw = EvmJsonRpcGateway::new(
            "usdc-base-test",
            server.base_url(),
            token.clone(),
            "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            signer,
        )
        .unwrap();

        let decision = gw
            .submit_transfer(&sample_transfer(&token, "k1", 1_000_000))
            .unwrap();
        assert_eq!(decision.status, CryptoStatus::Pending);
        assert_eq!(
            decision.tx_hash.as_deref(),
            Some("0x1111111111111111111111111111111111111111111111111111111111111111")
        );
        assert_eq!(decision.confirmations, 0);

        // The signer received an UnsignedTx that reflects the chain
        // state we mocked: chain_id 8453, nonce 42, gas_price 1 gwei,
        // gas_limit 60_000 * 1.2 = 72_000.
        let unsigned = gw.signer.captured().expect("signer was called");
        assert_eq!(unsigned.chain_id, 8453);
        assert_eq!(unsigned.nonce, 42);
        assert_eq!(unsigned.gas_price, 1_000_000_000);
        assert_eq!(unsigned.gas_limit, 72_000);
        assert_eq!(unsigned.value, 0);
        assert_eq!(unsigned.to, token.contract);
        // Calldata = selector + recipient + amount.
        assert!(unsigned.data.starts_with("0xa9059cbb"));
        // The decoded calldata's amount field must be 1_000_000.
        let suffix = &unsigned.data[unsigned.data.len() - 64..];
        let amt = u128::from_str_radix(suffix, 16).unwrap();
        assert_eq!(amt, 1_000_000);
    }

    #[test]
    fn rejects_wrong_chain() {
        // No HTTP calls should be made — server set up to fail on
        // any request, to prove we short-circuit before RPC.
        let server = MockServer::start();
        let token = StableToken::UsdcBase.token_ref();
        let signer = CapturingSigner::new("0xabc");
        let gw = EvmJsonRpcGateway::new(
            "usdc-base-test",
            server.base_url(),
            token.clone(),
            "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            signer,
        )
        .unwrap();

        let mut req = sample_transfer(&token, "k1", 100);
        req.to = CryptoAddress::new("polygon", "0xdeadbeef000000000000000000000000deadbeef");
        let err = gw.submit_transfer(&req).unwrap_err();
        assert!(matches!(err, Error::UnsupportedChain(c) if c == "polygon"));
        // Signer was never invoked.
        assert!(gw.signer.captured().is_none());
    }

    #[test]
    fn rejects_wrong_token_contract() {
        let server = MockServer::start();
        let token = StableToken::UsdcBase.token_ref();
        let signer = CapturingSigner::new("0xabc");
        let gw = EvmJsonRpcGateway::new(
            "usdc-base-test",
            server.base_url(),
            token.clone(),
            "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            signer,
        )
        .unwrap();

        let mut req = sample_transfer(&token, "k1", 100);
        req.token = TokenRef::new(
            "base",
            "0x000000000000000000000000000000000000dead",
            6,
            "FAKE",
        );
        let err = gw.submit_transfer(&req).unwrap_err();
        assert!(matches!(err, Error::UnsupportedToken(s) if s == "FAKE"));
    }

    #[test]
    fn query_status_returns_finalized_on_status_0x1() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST)
                .body_contains("\"eth_getTransactionReceipt\"");
            then.status(200)
                .header("content-type", "application/json")
                .body(
                    r#"{"jsonrpc":"2.0","id":1,"result":{
                        "transactionHash":"0xabc",
                        "status":"0x1",
                        "blockNumber":"0x1"
                    }}"#,
                );
        });

        let token = StableToken::UsdcBase.token_ref();
        let gw = EvmJsonRpcGateway::new(
            "usdc-base-test",
            server.base_url(),
            token.clone(),
            "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            CapturingSigner::new("0xabc"),
        )
        .unwrap();

        let d = gw
            .query_status(&StatusQueryReq {
                chain: "base".into(),
                tx_hash: "0xabc".into(),
                idempotency_key: None,
            })
            .unwrap();
        assert_eq!(d.status, CryptoStatus::Finalized);
        assert_eq!(d.confirmations, 1);
        assert_eq!(d.tx_hash.as_deref(), Some("0xabc"));
    }

    #[test]
    fn query_status_returns_rejected_on_status_0x0() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST)
                .body_contains("\"eth_getTransactionReceipt\"");
            then.status(200)
                .header("content-type", "application/json")
                .body(
                    r#"{"jsonrpc":"2.0","id":1,"result":{
                        "transactionHash":"0xdef",
                        "status":"0x0",
                        "revertReason":"ERC20: transfer amount exceeds balance"
                    }}"#,
                );
        });

        let token = StableToken::UsdcBase.token_ref();
        let gw = EvmJsonRpcGateway::new(
            "usdc-base-test",
            server.base_url(),
            token.clone(),
            "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            CapturingSigner::new("0xabc"),
        )
        .unwrap();

        let d = gw
            .query_status(&StatusQueryReq {
                chain: "base".into(),
                tx_hash: "0xdef".into(),
                idempotency_key: None,
            })
            .unwrap();
        assert_eq!(d.status, CryptoStatus::Rejected);
        assert_eq!(d.confirmations, 0);
        assert_eq!(
            d.reason.as_deref(),
            Some("ERC20: transfer amount exceeds balance")
        );
    }

    #[test]
    fn query_status_returns_pending_when_no_receipt() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST)
                .body_contains("\"eth_getTransactionReceipt\"");
            then.status(200)
                .header("content-type", "application/json")
                .body(r#"{"jsonrpc":"2.0","id":1,"result":null}"#);
        });

        let token = StableToken::UsdcBase.token_ref();
        let gw = EvmJsonRpcGateway::new(
            "usdc-base-test",
            server.base_url(),
            token.clone(),
            "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            CapturingSigner::new("0xabc"),
        )
        .unwrap();

        let d = gw
            .query_status(&StatusQueryReq {
                chain: "base".into(),
                tx_hash: "0xpending".into(),
                idempotency_key: None,
            })
            .unwrap();
        assert_eq!(d.status, CryptoStatus::Pending);
    }

    #[test]
    fn rpc_error_surfaces_as_rejected() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).body_contains("\"eth_chainId\"");
            then.status(200)
                .header("content-type", "application/json")
                .body(
                    r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32000,"message":"insufficient funds"}}"#,
                );
        });

        let token = StableToken::UsdcBase.token_ref();
        let gw = EvmJsonRpcGateway::new(
            "usdc-base-test",
            server.base_url(),
            token.clone(),
            "0xaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",
            CapturingSigner::new("0xabc"),
        )
        .unwrap();

        let err = gw
            .submit_transfer(&sample_transfer(&token, "k1", 100))
            .unwrap_err();
        match err {
            Error::Rejected { code, message } => {
                assert_eq!(code, "-32000");
                assert!(message.contains("insufficient"));
            }
            other => panic!("expected Rejected, got {other:?}"),
        }
    }
}
