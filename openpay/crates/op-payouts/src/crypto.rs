//! Crypto payout driver.
//!
//! Covers four substrates:
//!
//! - EVM (Ethereum, Polygon, Base) — ERC-20 transfer of USDC / USDT,
//!   ABI-encoded calldata to the token contract.
//! - Solana — SPL `Token::Transfer` instruction (we expose the
//!   recipient / mint / amount; the operator's signer assembles the
//!   transaction).
//! - Bitcoin — UTXO transfer to a single address; we emit an
//!   intent JSON (amount in satoshis, target address). The operator's
//!   wallet builds the actual transaction.
//! - Lightning — BOLT-11 invoice payment intent.
//!
//! The driver is offline-pure; signing and broadcast happen in
//! `op-rails-crypto` or the operator's KMS/wallet.

use serde::Serialize;
use uuid::Uuid;

use crate::error::{Error, Result};
use crate::payout::{
    BeneficiaryAccount, Payout, PayoutMethod, PayoutRequest, PayoutResult, PayoutStatus,
};

/// USDC contract address on Ethereum mainnet (lowercase, no checksum).
pub const USDC_ETHEREUM: &str = "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48";
/// USDC contract address on Polygon.
pub const USDC_POLYGON: &str = "0x2791bca1f2de4661ed88a30c99a7a9449aa84174";
/// USDC contract address on Base.
pub const USDC_BASE: &str = "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913";
/// USDT contract address on Ethereum mainnet.
pub const USDT_ETHEREUM: &str = "0xdac17f958d2ee523a2206206994597c13d831ec7";

/// Crypto payout driver.
#[derive(Clone, Debug, Default)]
pub struct CryptoPayoutDriver {
    /// Source wallet / signing key handle. Opaque to this crate.
    pub source_wallet: String,
}

#[derive(Debug, Serialize)]
struct EvmIntent<'a> {
    chain: &'a str,
    asset: &'a str,
    token_contract: &'a str,
    from: &'a str,
    to: &'a str,
    /// On-chain amount in the asset's smallest unit (6 decimals for
    /// USDC/USDT, 18 for native ETH — we always carry the asset's
    /// minor units as a string).
    amount_minor: String,
    /// Hex calldata for `transfer(address,uint256)`, ready to set on a
    /// transaction targeting `token_contract`.
    calldata_hex: String,
    idempotency_key: &'a str,
}

#[derive(Debug, Serialize)]
struct SolanaIntent<'a> {
    chain: &'static str,
    asset: &'a str,
    mint: &'a str,
    from: &'a str,
    to: &'a str,
    amount_minor: String,
    idempotency_key: &'a str,
}

#[derive(Debug, Serialize)]
struct BitcoinIntent<'a> {
    chain: &'static str,
    from: &'a str,
    to: &'a str,
    amount_satoshi: i64,
    idempotency_key: &'a str,
}

#[derive(Debug, Serialize)]
struct LightningIntent<'a> {
    chain: &'static str,
    from: &'a str,
    bolt11: &'a str,
    idempotency_key: &'a str,
}

fn token_contract(asset: &str, network: &str) -> Result<&'static str> {
    match (asset, network) {
        ("USDC", "ethereum") => Ok(USDC_ETHEREUM),
        ("USDC", "polygon") => Ok(USDC_POLYGON),
        ("USDC", "base") => Ok(USDC_BASE),
        ("USDT", "ethereum") => Ok(USDT_ETHEREUM),
        _ => Err(Error::DriverValidation(format!(
            "no known token contract for {asset} on {network}"
        ))),
    }
}

/// Encode `transfer(address,uint256)` calldata as hex (no `0x` prefix
/// stripped — we include the prefix).
fn encode_erc20_transfer(to: &str, amount_minor: u128) -> Result<String> {
    let to_clean = to.strip_prefix("0x").unwrap_or(to);
    if to_clean.len() != 40 || !to_clean.chars().all(|c| c.is_ascii_hexdigit()) {
        return Err(Error::InvalidBeneficiary {
            rail: "crypto_evm",
            detail: "EVM address must be 40 hex chars".to_string(),
        });
    }
    // selector for transfer(address,uint256) = 0xa9059cbb
    let mut out = String::from("0xa9059cbb");
    // pad address to 32 bytes
    out.push_str(&"0".repeat(24));
    out.push_str(&to_clean.to_lowercase());
    // pad amount to 32 bytes
    let amt_hex = format!("{amount_minor:064x}");
    out.push_str(&amt_hex);
    Ok(out)
}

impl Payout for CryptoPayoutDriver {
    fn rail(&self) -> &'static str {
        "crypto"
    }

    fn submit(&self, req: &PayoutRequest) -> Result<PayoutResult> {
        if !req.amount.is_positive() {
            return Err(Error::LimitViolation {
                rail: "crypto",
                detail: "amount must be positive".to_string(),
            });
        }
        let (asset, network) = match &req.method {
            PayoutMethod::Crypto { asset, network } => (asset.as_str(), network.as_str()),
            _ => return Err(Error::UnsupportedMethod { rail: "crypto" }),
        };
        let amount_minor = u128::try_from(req.amount.minor_units).map_err(|_| {
            Error::LimitViolation {
                rail: "crypto",
                detail: "amount must be non-negative".to_string(),
            }
        })?;
        let payload = match (asset, network, &req.beneficiary.account) {
            ("USDC" | "USDT", "ethereum" | "polygon" | "base", BeneficiaryAccount::EvmAddress(addr)) => {
                let contract = token_contract(asset, network)?;
                let calldata = encode_erc20_transfer(addr, amount_minor)?;
                let intent = EvmIntent {
                    chain: network,
                    asset,
                    token_contract: contract,
                    from: &self.source_wallet,
                    to: addr,
                    amount_minor: amount_minor.to_string(),
                    calldata_hex: calldata,
                    idempotency_key: &req.idempotency_key,
                };
                serde_json::to_vec(&intent).map_err(|e| Error::DriverValidation(e.to_string()))?
            }
            ("USDC" | "USDT", "solana", BeneficiaryAccount::SolanaAddress(addr)) => {
                if addr.len() < 32 || addr.len() > 44 {
                    return Err(Error::InvalidBeneficiary {
                        rail: "crypto_solana",
                        detail: "Solana address must be 32–44 base58 chars".to_string(),
                    });
                }
                let intent = SolanaIntent {
                    chain: "solana",
                    asset,
                    mint: "operator-configured",
                    from: &self.source_wallet,
                    to: addr,
                    amount_minor: amount_minor.to_string(),
                    idempotency_key: &req.idempotency_key,
                };
                serde_json::to_vec(&intent).map_err(|e| Error::DriverValidation(e.to_string()))?
            }
            ("BTC", "bitcoin", BeneficiaryAccount::BitcoinAddress(addr)) => {
                let intent = BitcoinIntent {
                    chain: "bitcoin",
                    from: &self.source_wallet,
                    to: addr,
                    amount_satoshi: req.amount.minor_units,
                    idempotency_key: &req.idempotency_key,
                };
                serde_json::to_vec(&intent).map_err(|e| Error::DriverValidation(e.to_string()))?
            }
            ("BTC", "lightning", BeneficiaryAccount::LightningInvoice(inv)) => {
                if !inv.to_lowercase().starts_with("ln") {
                    return Err(Error::InvalidBeneficiary {
                        rail: "crypto_lightning",
                        detail: "BOLT-11 invoice must start with 'ln'".to_string(),
                    });
                }
                let intent = LightningIntent {
                    chain: "lightning",
                    from: &self.source_wallet,
                    bolt11: inv,
                    idempotency_key: &req.idempotency_key,
                };
                serde_json::to_vec(&intent).map_err(|e| Error::DriverValidation(e.to_string()))?
            }
            _ => return Err(Error::UnsupportedMethod { rail: "crypto" }),
        };
        Ok(PayoutResult {
            idempotency_key: req.idempotency_key.clone(),
            payout_id: Uuid::now_v7().to_string(),
            status: PayoutStatus::PreparedOffline,
            raw_status: None,
            reason_code: None,
            reason_text: None,
            rail_txn_id: None,
            settled_amount: Some(req.amount),
            wire_payload: Some(payload),
        })
    }

    fn status(&self, _payout_id: &str) -> Result<PayoutResult> {
        Err(Error::DriverValidation(
            "crypto payout status requires on-chain lookup via op-rails-crypto".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::encode_erc20_transfer;

    #[test]
    fn erc20_calldata_shape() {
        let cd = encode_erc20_transfer("0x000000000000000000000000000000000000beef", 1_000_000)
            .expect("valid");
        // 4-byte selector + 32-byte address + 32-byte amount = 68 bytes
        // = 136 hex chars; + "0x" prefix = 138.
        assert_eq!(cd.len(), 2 + 8 + 64 + 64);
        assert!(cd.starts_with("0xa9059cbb"));
    }
}
