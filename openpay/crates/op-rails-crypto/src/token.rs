//! Stablecoin token references.
//!
//! Each `(chain, contract, decimals)` triple identifies a specific
//! token deployment. USDC alone is deployed on 10+ chains with
//! different contract addresses, so the chain is part of the
//! token identity for routing purposes.

use serde::{Deserialize, Serialize};

/// A token reference: chain + contract + decimal precision.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TokenRef {
    /// Lowercase canonical chain identifier (`"solana"`, `"base"`,
    /// `"ethereum"`, `"polygon"`, `"arbitrum"`).
    pub chain: String,
    /// Contract / mint address. EVM hex `0x...`, Solana base58.
    pub contract: String,
    /// Token's decimal precision: 6 for USDC / USDT / EURC,
    /// varies for others. Used to convert between minor-unit
    /// integers and on-chain amounts.
    pub decimals: u8,
    /// Human-readable symbol (`"USDC"`, `"EURC"`, `"PYUSD"`).
    /// Informational only — routing keys off `chain` + `contract`.
    pub symbol: String,
}

impl TokenRef {
    /// Construct.
    #[must_use]
    pub fn new(
        chain: impl Into<String>,
        contract: impl Into<String>,
        decimals: u8,
        symbol: impl Into<String>,
    ) -> Self {
        Self {
            chain: chain.into(),
            contract: contract.into(),
            decimals,
            symbol: symbol.into(),
        }
    }
}

/// Curated list of well-known stablecoins on common chains. Use
/// these constructors to avoid hand-typing contract addresses.
/// Operators with novel chains construct [`TokenRef`] directly.
///
/// Addresses are current as of the time of writing. Drivers
/// **must** double-check against the issuer's official
/// documentation before broadcasting — token contracts have been
/// upgraded / re-deployed in the past, and minting a transfer to a
/// stale contract burns funds.
#[derive(Clone, Copy, Debug)]
pub enum StableToken {
    /// USDC on Solana (mainnet-beta).
    UsdcSolana,
    /// USDC on Base.
    UsdcBase,
    /// USDC on Ethereum mainnet.
    UsdcEthereum,
    /// USDC on Polygon `PoS`.
    UsdcPolygon,
    /// USDC on Arbitrum One.
    UsdcArbitrum,
    /// EURC on Base.
    EurcBase,
    /// PYUSD on Ethereum mainnet.
    PyusdEthereum,
}

impl StableToken {
    /// Convert into a concrete [`TokenRef`] (chain, contract, etc.).
    #[must_use]
    pub fn token_ref(self) -> TokenRef {
        match self {
            Self::UsdcSolana => TokenRef::new(
                "solana",
                "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                6,
                "USDC",
            ),
            Self::UsdcBase => TokenRef::new(
                "base",
                "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913",
                6,
                "USDC",
            ),
            Self::UsdcEthereum => TokenRef::new(
                "ethereum",
                "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
                6,
                "USDC",
            ),
            Self::UsdcPolygon => TokenRef::new(
                "polygon",
                "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359",
                6,
                "USDC",
            ),
            Self::UsdcArbitrum => TokenRef::new(
                "arbitrum",
                "0xaf88d065e77c8cc2239327c5edb3a432268e5831",
                6,
                "USDC",
            ),
            Self::EurcBase => TokenRef::new(
                "base",
                "0x60a3e35cc302bfa44cb288bc5a4f316fdb1adb42",
                6,
                "EURC",
            ),
            Self::PyusdEthereum => TokenRef::new(
                "ethereum",
                "0x6c3ea9036406852006290770bedfcaba0e23a0e8",
                6,
                "PYUSD",
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn usdc_solana_has_canonical_mint() {
        let t = StableToken::UsdcSolana.token_ref();
        assert_eq!(t.chain, "solana");
        assert_eq!(t.symbol, "USDC");
        assert_eq!(t.decimals, 6);
        assert_eq!(t.contract, "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v");
    }

    #[test]
    fn token_refs_distinct_across_chains() {
        let base = StableToken::UsdcBase.token_ref();
        let eth = StableToken::UsdcEthereum.token_ref();
        assert_ne!(base, eth);
        assert_eq!(base.symbol, eth.symbol);
        assert_ne!(base.chain, eth.chain);
    }

    #[test]
    fn evm_addresses_lowercase() {
        for t in [
            StableToken::UsdcBase,
            StableToken::UsdcEthereum,
            StableToken::UsdcPolygon,
            StableToken::UsdcArbitrum,
            StableToken::EurcBase,
            StableToken::PyusdEthereum,
        ] {
            let r = t.token_ref();
            assert!(
                r.contract.starts_with("0x"),
                "{} contract should be EVM hex",
                r.symbol
            );
            assert_eq!(
                r.contract.to_ascii_lowercase(),
                r.contract,
                "{} contract should be lowercase",
                r.symbol
            );
        }
    }
}
