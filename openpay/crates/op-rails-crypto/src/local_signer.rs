//! In-process hot-wallet [`EvmSigner`].
//!
//! `LocalKeyEvmSigner` holds a raw secp256k1 private key in memory,
//! signs the gateway-supplied [`UnsignedTx`] with k256 (RFC 6979
//! deterministic nonces, recoverable signatures), RLP-encodes the
//! result with the EIP-155 chain-id-aware `v`, and broadcasts it via
//! `eth_sendRawTransaction`.
//!
//! ## Trade-offs
//!
//! - **Hot wallet.** The key sits in process memory. Operationally
//!   fine for a pilot launch where the operator controls the host
//!   end-to-end; replace with `Fireblocks` / AWS KMS / an HSM before
//!   carrying serious balances. The [`EvmSigner`] trait is the swap
//!   point — production deployments wire a different implementation
//!   without touching the gateway.
//!
//! - **Legacy txs (EIP-155), not EIP-1559.** The existing
//!   [`UnsignedTx`] carries a single `gas_price` field, matching the
//!   legacy / type-0 tx envelope. Modern chains (Ethereum mainnet
//!   post-London, all major L2s) still accept legacy txs — the
//!   `gas_price` simply becomes the effective per-gas fee. Migrating
//!   to EIP-1559 type-2 envelopes is a future enhancement that needs
//!   `UnsignedTx` to grow `max_priority_fee_per_gas` /
//!   `max_fee_per_gas` fields, plus `eth_maxPriorityFeePerGas` +
//!   `eth_feeHistory` plumbing in [`crate::evm::EvmJsonRpcGateway`].
//!
//! ## Signing flow (EIP-155 legacy)
//!
//! 1. RLP-encode `[nonce, gas_price, gas_limit, to, value, data,
//!    chain_id, 0, 0]` — the "signing" envelope per EIP-155.
//! 2. `keccak256(rlp)` is the message digest.
//! 3. `SigningKey::sign_prehash_recoverable` produces `(r, s,
//!    recovery_id)`.
//! 4. EIP-155 `v = chain_id * 2 + 35 + recovery_id`.
//! 5. RLP-encode the signed tx `[nonce, gas_price, gas_limit, to,
//!    value, data, v, r, s]` and hex-encode with `0x` prefix.
//! 6. POST as `eth_sendRawTransaction`.
//!
//! No transaction-type byte is prepended (that's the EIP-2718
//! envelope prefix used by EIP-1559 / EIP-2930 txs; legacy txs sit
//! directly in the RLP list).

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

use k256::ecdsa::{RecoveryId, Signature, SigningKey};
use reqwest::blocking::Client;
use rlp::RlpStream;
use serde_json::{Value, json};
use sha3::{Digest, Keccak256};

use crate::error::{Error, Result};
use crate::signer::{EvmSigner, TxHash, UnsignedTx};

/// Conservative HTTP timeout for `eth_sendRawTransaction`. Same
/// rationale as [`crate::evm::EvmJsonRpcGateway`]: failed calls
/// surface as [`Error::Transport`] and the orchestrator's retry
/// layer takes over.
const DEFAULT_TIMEOUT_SECS: u64 = 15;

/// Hot-wallet [`EvmSigner`]: signs locally with a k256 secp256k1
/// private key and broadcasts via the operator's JSON-RPC endpoint.
///
/// Construct via [`Self::from_hex_private_key`] or
/// [`Self::from_env`]. The signing key is held in process memory for
/// the lifetime of the signer; drop the signer (or the gateway that
/// owns it) to zeroize it via `k256`'s underlying `Zeroize` impl.
pub struct LocalKeyEvmSigner {
    signing_key: SigningKey,
    rpc_url: String,
    http: Client,
    /// Monotonic counter for JSON-RPC `id` fields. Matches the
    /// gateway's pattern so tcpdump / RPC-proxy logs line up.
    rpc_id: AtomicU64,
}

impl std::fmt::Debug for LocalKeyEvmSigner {
    /// Redacted: never log the private key, even at debug level.
    /// `http` / `rpc_id` are runtime plumbing not worth surfacing.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LocalKeyEvmSigner")
            .field("signing_key", &"<redacted>")
            .field("rpc_url", &self.rpc_url)
            .field("address", &self.address())
            .finish_non_exhaustive()
    }
}

impl LocalKeyEvmSigner {
    /// Build a signer from a hex-encoded 32-byte private key.
    ///
    /// Accepts an optional `0x` / `0X` prefix. The key must decode to
    /// exactly 32 bytes and lie within the secp256k1 scalar field
    /// (k256 enforces both — the latter rejects the zero key and
    /// anything `>= n`).
    ///
    /// # Errors
    /// Returns [`Error::DriverValidation`] if the hex is malformed,
    /// the byte length is wrong, the scalar is invalid, or the
    /// underlying `reqwest` client fails to build.
    pub fn from_hex_private_key(hex_key: &str, rpc_url: impl Into<String>) -> Result<Self> {
        let stripped = hex_key
            .strip_prefix("0x")
            .or_else(|| hex_key.strip_prefix("0X"))
            .unwrap_or(hex_key);
        if stripped.len() != 64 {
            return Err(Error::DriverValidation(format!(
                "private key must be 64 hex chars (32 bytes), got {}",
                stripped.len()
            )));
        }
        let bytes = hex::decode(stripped)
            .map_err(|e| Error::DriverValidation(format!("private key hex decode: {e}")))?;
        let signing_key = SigningKey::from_slice(&bytes)
            .map_err(|e| Error::DriverValidation(format!("invalid secp256k1 key: {e}")))?;
        let http = Client::builder()
            .timeout(Duration::from_secs(DEFAULT_TIMEOUT_SECS))
            .build()
            .map_err(|e| Error::Transport(format!("reqwest client build failed: {e}")))?;
        Ok(Self {
            signing_key,
            rpc_url: rpc_url.into(),
            http,
            rpc_id: AtomicU64::new(1),
        })
    }

    /// Convenience: load both the RPC URL and the hex private key
    /// from environment variables.
    ///
    /// Typical operator deployment reads `OP_EVM_RPC_URL` and
    /// `OP_EVM_HOT_WALLET_KEY` from a systemd unit's `EnvironmentFile`
    /// or a Docker secret.
    ///
    /// # Errors
    /// Returns [`Error::MissingField`] if either env var is unset,
    /// and any error from [`Self::from_hex_private_key`].
    pub fn from_env(rpc_url_env: &str, key_env: &str) -> Result<Self> {
        let rpc_url = std::env::var(rpc_url_env)
            .map_err(|_| Error::DriverValidation(format!("env var `{rpc_url_env}` is unset")))?;
        let key = std::env::var(key_env)
            .map_err(|_| Error::DriverValidation(format!("env var `{key_env}` is unset")))?;
        Self::from_hex_private_key(&key, rpc_url)
    }

    /// Override the HTTP timeout. Useful in tests against in-process
    /// mock servers where the default 15-second timeout is overkill.
    ///
    /// # Errors
    /// Returns [`Error::Transport`] if the client rebuild fails.
    pub fn with_timeout(mut self, timeout: Duration) -> Result<Self> {
        self.http = Client::builder()
            .timeout(timeout)
            .build()
            .map_err(|e| Error::Transport(format!("reqwest client build failed: {e}")))?;
        Ok(self)
    }

    /// The EVM address (`0x` + 40 lowercase hex chars) derived from
    /// this signer's public key.
    ///
    /// Derivation: take the uncompressed SEC1 encoding of the public
    /// key (65 bytes, leading `0x04` tag), drop the tag,
    /// `keccak256` the remaining 64 bytes, and take the trailing 20.
    #[must_use]
    pub fn address(&self) -> String {
        let verifying_key = self.signing_key.verifying_key();
        let encoded = verifying_key.to_encoded_point(false);
        // SEC1 uncompressed encoding: 0x04 || X(32) || Y(32). Drop
        // the tag byte and keccak the 64-byte (X, Y) tuple.
        let bytes = encoded.as_bytes();
        debug_assert_eq!(bytes.len(), 65);
        debug_assert_eq!(bytes[0], 0x04);
        let mut hasher = Keccak256::new();
        hasher.update(&bytes[1..]);
        let digest = hasher.finalize();
        format!("0x{}", hex::encode(&digest[12..]))
    }

    fn next_rpc_id(&self) -> u64 {
        self.rpc_id.fetch_add(1, Ordering::Relaxed)
    }

    /// Broadcast a hex-encoded signed tx via `eth_sendRawTransaction`
    /// and return the `result` field (the chain's tx hash).
    fn broadcast_raw_tx(&self, raw_tx_hex: &str) -> Result<TxHash> {
        let body = json!({
            "jsonrpc": "2.0",
            "id": self.next_rpc_id(),
            "method": "eth_sendRawTransaction",
            "params": [raw_tx_hex],
        });
        let resp = self
            .http
            .post(&self.rpc_url)
            .header("content-type", "application/json")
            .body(body.to_string())
            .send()
            .map_err(|e| Error::Transport(format!("eth_sendRawTransaction: {e}")))?;
        if !resp.status().is_success() {
            return Err(Error::Transport(format!(
                "eth_sendRawTransaction: http {}",
                resp.status()
            )));
        }
        let parsed: Value = resp
            .json()
            .map_err(|e| Error::Transport(format!("eth_sendRawTransaction: parse: {e}")))?;
        if let Some(err_obj) = parsed.get("error") {
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
            .and_then(Value::as_str)
            .map(str::to_owned)
            .ok_or_else(|| Error::Transport("eth_sendRawTransaction: missing result".into()))
    }
}

impl EvmSigner for LocalKeyEvmSigner {
    fn sign_and_broadcast(&self, unsigned: UnsignedTx) -> Result<TxHash> {
        let to_bytes = parse_hex_bytes(&unsigned.to)
            .map_err(|reason| Error::DriverValidation(format!("invalid `to` address: {reason}")))?;
        if to_bytes.len() != 20 {
            return Err(Error::DriverValidation(format!(
                "`to` must be 20 bytes, got {}",
                to_bytes.len()
            )));
        }
        let data_bytes = parse_hex_bytes(&unsigned.data)
            .map_err(|reason| Error::DriverValidation(format!("invalid `data` hex: {reason}")))?;

        let tx_fields = TxFields {
            nonce: unsigned.nonce,
            gas_price: unsigned.gas_price,
            gas_limit: unsigned.gas_limit,
            to: &to_bytes,
            value: unsigned.value,
            data: &data_bytes,
        };

        // EIP-155 signing envelope:
        //   rlp([nonce, gas_price, gas_limit, to, value, data,
        //        chain_id, 0, 0])
        let signing_rlp = rlp_encode(&tx_fields, TailMode::Signing(unsigned.chain_id));

        let mut hasher = Keccak256::new();
        hasher.update(&signing_rlp);
        let digest = hasher.finalize();

        let (signature, recovery_id) = self
            .signing_key
            .sign_prehash_recoverable(&digest)
            .map_err(|e| Error::DriverValidation(format!("ecdsa sign: {e}")))?;

        let v = eip155_v(unsigned.chain_id, recovery_id);
        let signed_rlp = rlp_encode(
            &tx_fields,
            TailMode::Signed {
                v,
                signature: &signature,
            },
        );
        let raw_tx_hex = format!("0x{}", hex::encode(&signed_rlp));
        self.broadcast_raw_tx(&raw_tx_hex)
    }
}

/// Shared (chain-id-agnostic) fields of a legacy EVM tx. Borrowed
/// fields keep the encoder allocation-free for the two-byte slices.
struct TxFields<'a> {
    nonce: u64,
    gas_price: u128,
    gas_limit: u64,
    to: &'a [u8],
    value: u128,
    data: &'a [u8],
}

/// Picks the last three RLP items of a legacy tx envelope:
/// either the EIP-155 signing tail `(chain_id, 0, 0)` or the signed
/// tail `(v, r, s)`.
#[derive(Copy, Clone)]
enum TailMode<'a> {
    Signing(u64),
    Signed { v: u64, signature: &'a Signature },
}

/// Parse a `0x`-prefixed (or bare) hex string into raw bytes.
fn parse_hex_bytes(s: &str) -> std::result::Result<Vec<u8>, String> {
    let stripped = s
        .strip_prefix("0x")
        .or_else(|| s.strip_prefix("0X"))
        .unwrap_or(s);
    hex::decode(stripped).map_err(|e| format!("hex decode: {e}"))
}

/// Minimal big-endian encoding of a `u128`. EVM RLP rules treat
/// integers as variable-length big-endian byte strings with no
/// leading zeros (zero itself encodes to an empty byte string).
fn minimal_be_u128(v: u128) -> Vec<u8> {
    if v == 0 {
        return Vec::new();
    }
    let full = v.to_be_bytes();
    let start = full.iter().position(|b| *b != 0).unwrap_or(full.len() - 1);
    full[start..].to_vec()
}

/// Minimal big-endian encoding of a `u64`. Same rules as
/// [`minimal_be_u128`].
fn minimal_be_u64(v: u64) -> Vec<u8> {
    if v == 0 {
        return Vec::new();
    }
    let full = v.to_be_bytes();
    let start = full.iter().position(|b| *b != 0).unwrap_or(full.len() - 1);
    full[start..].to_vec()
}

/// RLP-encode a legacy EVM tx, with either the EIP-155 signing tail
/// or the signed `(v, r, s)` tail.
fn rlp_encode(fields: &TxFields<'_>, tail: TailMode<'_>) -> Vec<u8> {
    let mut s = RlpStream::new_list(9);
    s.append(&minimal_be_u64(fields.nonce));
    s.append(&minimal_be_u128(fields.gas_price));
    s.append(&minimal_be_u64(fields.gas_limit));
    s.append(&fields.to);
    s.append(&minimal_be_u128(fields.value));
    s.append(&fields.data);
    match tail {
        TailMode::Signing(chain_id) => {
            s.append(&minimal_be_u64(chain_id));
            // r and s are zero in the signing envelope per EIP-155.
            s.append(&Vec::<u8>::new());
            s.append(&Vec::<u8>::new());
        }
        TailMode::Signed { v, signature } => {
            let (r_bytes, s_bytes) = signature.split_bytes();
            s.append(&minimal_be_u64(v));
            s.append(&strip_leading_zeros(r_bytes.as_slice()));
            s.append(&strip_leading_zeros(s_bytes.as_slice()));
        }
    }
    s.out().to_vec()
}

/// Strip leading zero bytes (RLP wants minimal-length integer
/// encodings). An all-zero input encodes as an empty byte string.
fn strip_leading_zeros(bytes: &[u8]) -> Vec<u8> {
    match bytes.iter().position(|b| *b != 0) {
        Some(start) => bytes[start..].to_vec(),
        None => Vec::new(),
    }
}

/// EIP-155 `v` value: `chain_id * 2 + 35 + recovery_id`. The
/// `recovery_id` is `k256`'s low bit (0 or 1).
fn eip155_v(chain_id: u64, recovery_id: RecoveryId) -> u64 {
    chain_id * 2 + 35 + u64::from(recovery_id.to_byte() & 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use k256::ecdsa::VerifyingKey;

    /// Anvil / Hardhat dev account #0. Well-known across the EVM
    /// ecosystem; deriving its address is the canonical sanity
    /// check for a secp256k1 key-to-address pipeline.
    const ANVIL_ACCOUNT_0_KEY: &str =
        "0xac0974bec39a17e36ba4a6b4d238ff944bacb478cbed5efcae784d7bf4f2ff80";
    const ANVIL_ACCOUNT_0_ADDR: &str = "0xf39fd6e51aad88f6f4ce6ab8827279cfffb92266";

    #[test]
    fn address_derivation_matches_canonical() {
        let signer =
            LocalKeyEvmSigner::from_hex_private_key(ANVIL_ACCOUNT_0_KEY, "http://unused.invalid")
                .expect("constructor");
        assert_eq!(signer.address(), ANVIL_ACCOUNT_0_ADDR);
    }

    #[test]
    fn address_derivation_accepts_unprefixed_hex() {
        let unprefixed = &ANVIL_ACCOUNT_0_KEY[2..];
        let signer = LocalKeyEvmSigner::from_hex_private_key(unprefixed, "http://unused.invalid")
            .expect("constructor");
        assert_eq!(signer.address(), ANVIL_ACCOUNT_0_ADDR);
    }

    #[test]
    fn signature_recovers_to_expected_address() {
        let signer =
            LocalKeyEvmSigner::from_hex_private_key(ANVIL_ACCOUNT_0_KEY, "http://unused.invalid")
                .expect("constructor");

        // Arbitrary 32-byte prehash.
        let prehash: [u8; 32] = [
            0x88, 0xcc, 0xe7, 0x6f, 0x4a, 0x12, 0x9b, 0x2d, 0x1f, 0x42, 0x33, 0x07, 0xaa, 0xbb,
            0xcc, 0xdd, 0xee, 0xff, 0x00, 0x11, 0x22, 0x33, 0x44, 0x55, 0x66, 0x77, 0x88, 0x99,
            0xaa, 0xbb, 0xcc, 0xdd,
        ];
        let (sig, recid) = signer
            .signing_key
            .sign_prehash_recoverable(&prehash)
            .expect("sign");
        let recovered = VerifyingKey::recover_from_prehash(&prehash, &sig, recid).expect("recover");

        // Derive the address from the recovered key the same way
        // `address()` does, and compare.
        let encoded = recovered.to_encoded_point(false);
        let bytes = encoded.as_bytes();
        assert_eq!(bytes[0], 0x04);
        let mut hasher = Keccak256::new();
        hasher.update(&bytes[1..]);
        let digest = hasher.finalize();
        let addr = format!("0x{}", hex::encode(&digest[12..]));
        assert_eq!(addr, ANVIL_ACCOUNT_0_ADDR);
    }

    #[test]
    fn eip155_v_calculation() {
        // chain_id = 1, recovery_id = 0  => v = 2*1 + 35 + 0 = 37
        // chain_id = 1, recovery_id = 1  => v = 2*1 + 35 + 1 = 38
        let recid0 = RecoveryId::from_byte(0).expect("recid 0");
        let recid1 = RecoveryId::from_byte(1).expect("recid 1");
        assert_eq!(eip155_v(1, recid0), 37);
        assert_eq!(eip155_v(1, recid1), 38);

        // Spot-check Base (chain_id = 8453).
        assert_eq!(eip155_v(8453, recid0), 16941); // 2*8453 + 35
        assert_eq!(eip155_v(8453, recid1), 16942);
    }

    #[test]
    fn minimal_be_encodings_drop_leading_zeros() {
        assert_eq!(minimal_be_u64(0), Vec::<u8>::new());
        assert_eq!(minimal_be_u64(1), vec![0x01]);
        assert_eq!(minimal_be_u64(0xff), vec![0xff]);
        assert_eq!(minimal_be_u64(0x0100), vec![0x01, 0x00]);
        assert_eq!(minimal_be_u128(0), Vec::<u8>::new());
        assert_eq!(minimal_be_u128(u128::from(u64::MAX) + 1).len(), 9);
    }

    #[test]
    fn broadcast_returns_tx_hash() {
        let server = MockServer::start();
        let returned_hash = "0xabc123abc123abc123abc123abc123abc123abc123abc123abc123abc123abc1";

        let mock = server.mock(|when, then| {
            when.method(POST)
                .body_contains("\"eth_sendRawTransaction\"")
                .body_contains("\"jsonrpc\":\"2.0\"")
                .body_contains("\"params\":[\"0x");
            then.status(200)
                .header("content-type", "application/json")
                .body(format!(
                    r#"{{"jsonrpc":"2.0","id":1,"result":"{returned_hash}"}}"#
                ));
        });

        let signer =
            LocalKeyEvmSigner::from_hex_private_key(ANVIL_ACCOUNT_0_KEY, server.base_url())
                .expect("constructor")
                .with_timeout(Duration::from_secs(2))
                .expect("timeout");

        // Realistic-looking unsigned tx: ERC-20 transfer calldata on Base.
        let mut data = vec![0xa9, 0x05, 0x9c, 0xbb];
        data.extend_from_slice(&[0u8; 12]);
        data.extend_from_slice(&[0xde; 20]);
        let mut amt_bytes = [0u8; 32];
        amt_bytes[24..].copy_from_slice(&1_000_000u64.to_be_bytes());
        data.extend_from_slice(&amt_bytes);

        let unsigned = UnsignedTx {
            chain_id: 8453,
            nonce: 42,
            gas_price: 1_000_000_000,
            gas_limit: 72_000,
            to: "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913".into(),
            value: 0,
            data: format!("0x{}", hex::encode(&data)),
            from: ANVIL_ACCOUNT_0_ADDR.into(),
        };

        let returned = signer.sign_and_broadcast(unsigned).expect("broadcast");
        assert_eq!(returned, returned_hash);
        mock.assert();
    }

    #[test]
    fn broadcast_surfaces_rpc_error_as_rejected() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST)
                .body_contains("\"eth_sendRawTransaction\"");
            then.status(200)
                .header("content-type", "application/json")
                .body(
                    r#"{"jsonrpc":"2.0","id":1,"error":{"code":-32000,"message":"nonce too low"}}"#,
                );
        });

        let signer =
            LocalKeyEvmSigner::from_hex_private_key(ANVIL_ACCOUNT_0_KEY, server.base_url())
                .expect("constructor")
                .with_timeout(Duration::from_secs(2))
                .expect("timeout");

        let unsigned = UnsignedTx {
            chain_id: 8453,
            nonce: 0,
            gas_price: 1_000_000_000,
            gas_limit: 21_000,
            to: "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913".into(),
            value: 0,
            data: "0x".into(),
            from: ANVIL_ACCOUNT_0_ADDR.into(),
        };

        let err = signer.sign_and_broadcast(unsigned).unwrap_err();
        match err {
            Error::Rejected { code, message } => {
                assert_eq!(code, "-32000");
                assert!(message.contains("nonce too low"));
            }
            other => panic!("expected Rejected, got {other:?}"),
        }
    }

    #[test]
    fn rejects_bad_hex_key() {
        // Empty.
        let err = LocalKeyEvmSigner::from_hex_private_key("", "http://x.invalid").unwrap_err();
        assert!(matches!(err, Error::DriverValidation(msg) if msg.contains("64 hex")));

        // Wrong length.
        let err =
            LocalKeyEvmSigner::from_hex_private_key("0xdeadbeef", "http://x.invalid").unwrap_err();
        assert!(matches!(err, Error::DriverValidation(msg) if msg.contains("64 hex")));

        // Non-hex chars.
        let mut bad = String::from("0x");
        bad.push_str(&"zz".repeat(32));
        let err = LocalKeyEvmSigner::from_hex_private_key(&bad, "http://x.invalid").unwrap_err();
        assert!(matches!(err, Error::DriverValidation(msg) if msg.contains("hex decode")));

        // 32 bytes, but the all-zero scalar isn't a valid private key.
        let zero_key = format!("0x{}", "00".repeat(32));
        let err =
            LocalKeyEvmSigner::from_hex_private_key(&zero_key, "http://x.invalid").unwrap_err();
        assert!(matches!(err, Error::DriverValidation(msg) if msg.contains("secp256k1")));
    }
}
