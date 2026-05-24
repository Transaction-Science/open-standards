//! Extended stablecoin catalog.
//!
//! `op-rails-crypto::token::StableToken` covers a curated subset
//! (USDC / EURC / PYUSD on a handful of chains). This module is the
//! comprehensive list: USDC, USDT, PYUSD, DAI, RLUSD, EURC, FDUSD,
//! with multi-chain deployments.
//!
//! Each [`Stablecoin`] variant resolves to a list of
//! [`StablecoinDeployment`] entries (one per chain). Operators
//! pick the deployments they actually settle on; the catalog itself
//! is informational. Contract addresses are best-effort current;
//! verify against the issuer's documentation before broadcasting.

use serde::{Deserialize, Serialize};

/// Chain + address pair, with the chain name in the same
/// lowercase canonical form as `op-rails-crypto::TokenRef`.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ChainAddress {
    /// Canonical chain identifier (`"ethereum"`, `"base"`, ...).
    pub chain: String,
    /// Address (EVM hex `0x...`, Solana base58, etc.).
    pub address: String,
}

impl ChainAddress {
    /// Construct.
    #[must_use]
    pub fn new(chain: impl Into<String>, address: impl Into<String>) -> Self {
        Self {
            chain: chain.into(),
            address: address.into(),
        }
    }
}

/// One specific deployment of a stablecoin: chain + contract +
/// decimals.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StablecoinDeployment {
    /// Stablecoin family (USDC, USDT, ...).
    pub family: Stablecoin,
    /// Where it's deployed.
    pub at: ChainAddress,
    /// Token's smallest-unit precision.
    pub decimals: u8,
}

impl StablecoinDeployment {
    /// Construct.
    #[must_use]
    pub fn new(family: Stablecoin, at: ChainAddress, decimals: u8) -> Self {
        Self {
            family,
            at,
            decimals,
        }
    }
}

/// Stablecoin families covered by this catalog.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Stablecoin {
    /// Circle USD coin.
    Usdc,
    /// Tether USD.
    Usdt,
    /// PayPal USD.
    Pyusd,
    /// MakerDAO DAI.
    Dai,
    /// Ripple USD.
    Rlusd,
    /// Circle euro coin.
    Eurc,
    /// First Digital USD.
    Fdusd,
}

impl Stablecoin {
    /// Human-readable ticker.
    #[must_use]
    pub const fn symbol(self) -> &'static str {
        match self {
            Self::Usdc => "USDC",
            Self::Usdt => "USDT",
            Self::Pyusd => "PYUSD",
            Self::Dai => "DAI",
            Self::Rlusd => "RLUSD",
            Self::Eurc => "EURC",
            Self::Fdusd => "FDUSD",
        }
    }

    /// True iff the stablecoin's peg is denominated in USD. (EURC is
    /// the only euro-peg in this catalog at the moment.)
    #[must_use]
    pub const fn is_usd_peg(self) -> bool {
        !matches!(self, Self::Eurc)
    }

    /// Curated deployments: chain + contract + decimals.
    ///
    /// Coverage rules:
    /// - USDC: native on Ethereum / Base / Polygon PoS / Arbitrum /
    ///   Optimism / Solana / Linea / Polygon zkEVM. Bridged USDC is
    ///   intentionally excluded — operators wanting bridged variants
    ///   construct [`StablecoinDeployment`] directly.
    /// - USDT: dominant on Ethereum / Tron / Solana / Polygon /
    ///   Arbitrum / Optimism. Tron uses a chain name `"tron"`.
    /// - PYUSD: Ethereum + Solana.
    /// - DAI: Ethereum + Polygon + Optimism + Arbitrum + Base.
    /// - RLUSD: Ethereum (XRPL counterpart lives off this catalog).
    /// - EURC: Ethereum + Base + Solana + Avalanche.
    /// - FDUSD: Ethereum + BNB Smart Chain.
    #[must_use]
    pub fn deployments(self) -> Vec<StablecoinDeployment> {
        match self {
            Self::Usdc => vec![
                Self::dep(
                    "ethereum",
                    "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48",
                    6,
                    self,
                ),
                Self::dep("base", "0x833589fcd6edb6e08f4c7c32d4f71b54bda02913", 6, self),
                Self::dep(
                    "polygon",
                    "0x3c499c542cef5e3811e1192ce70d8cc03d5c3359",
                    6,
                    self,
                ),
                Self::dep(
                    "arbitrum",
                    "0xaf88d065e77c8cc2239327c5edb3a432268e5831",
                    6,
                    self,
                ),
                Self::dep(
                    "optimism",
                    "0x0b2c639c533813f4aa9d7837caf62653d097ff85",
                    6,
                    self,
                ),
                Self::dep(
                    "solana",
                    "EPjFWdd5AufqSSqeM2qN1xzybapC8G4wEGGkZwyTDt1v",
                    6,
                    self,
                ),
                Self::dep("linea", "0x176211869ca2b568f2a7d4ee941e073a821ee1ff", 6, self),
                Self::dep(
                    "polygon-zkevm",
                    "0xa8ce8aee21bc2a48a5ef670afcc9274c7bbbc035",
                    6,
                    self,
                ),
            ],
            Self::Usdt => vec![
                Self::dep(
                    "ethereum",
                    "0xdac17f958d2ee523a2206206994597c13d831ec7",
                    6,
                    self,
                ),
                Self::dep("tron", "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t", 6, self),
                Self::dep(
                    "solana",
                    "Es9vMFrzaCERmJfrF4H2FYD4KCoNkY11McCe8BenwNYB",
                    6,
                    self,
                ),
                Self::dep(
                    "polygon",
                    "0xc2132d05d31c914a87c6611c10748aeb04b58e8f",
                    6,
                    self,
                ),
                Self::dep(
                    "arbitrum",
                    "0xfd086bc7cd5c481dcc9c85ebe478a1c0b69fcbb9",
                    6,
                    self,
                ),
                Self::dep(
                    "optimism",
                    "0x94b008aa00579c1307b0ef2c499ad98a8ce58e58",
                    6,
                    self,
                ),
            ],
            Self::Pyusd => vec![
                Self::dep(
                    "ethereum",
                    "0x6c3ea9036406852006290770bedfcaba0e23a0e8",
                    6,
                    self,
                ),
                Self::dep(
                    "solana",
                    "2b1kV6DkPAnxd5ixfnxCpjxmKwqjjaYmCZfHsFu24GXo",
                    6,
                    self,
                ),
            ],
            Self::Dai => vec![
                Self::dep(
                    "ethereum",
                    "0x6b175474e89094c44da98b954eedeac495271d0f",
                    18,
                    self,
                ),
                Self::dep(
                    "polygon",
                    "0x8f3cf7ad23cd3cadbd9735aff958023239c6a063",
                    18,
                    self,
                ),
                Self::dep(
                    "optimism",
                    "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1",
                    18,
                    self,
                ),
                Self::dep(
                    "arbitrum",
                    "0xda10009cbd5d07dd0cecc66161fc93d7c9000da1",
                    18,
                    self,
                ),
                Self::dep("base", "0x50c5725949a6f0c72e6c4a641f24049a917db0cb", 18, self),
            ],
            Self::Rlusd => vec![Self::dep(
                "ethereum",
                "0x8292bb45bf1ee4d140127049757c2e0ff06317ed",
                18,
                self,
            )],
            Self::Eurc => vec![
                Self::dep(
                    "ethereum",
                    "0x1abaea1f7c830bd89acc67ec4af516284b1bc33c",
                    6,
                    self,
                ),
                Self::dep("base", "0x60a3e35cc302bfa44cb288bc5a4f316fdb1adb42", 6, self),
                Self::dep(
                    "solana",
                    "HzwqbKZw8HxMN6bF2yFZNrht3c2iXXzpKcFu7uBEDKtr",
                    6,
                    self,
                ),
                Self::dep(
                    "avalanche",
                    "0xc891eb4cbdeff6e073e859e987815ed1505c2acd",
                    6,
                    self,
                ),
            ],
            Self::Fdusd => vec![
                Self::dep(
                    "ethereum",
                    "0xc5f0f7b66764f6ec8c8dff7ba683102295e16409",
                    6,
                    self,
                ),
                Self::dep("bsc", "0xc5f0f7b66764f6ec8c8dff7ba683102295e16409", 18, self),
            ],
        }
    }

    fn dep(chain: &str, contract: &str, decimals: u8, family: Self) -> StablecoinDeployment {
        StablecoinDeployment::new(family, ChainAddress::new(chain, contract), decimals)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn symbol_round_trips() {
        for s in [
            Stablecoin::Usdc,
            Stablecoin::Usdt,
            Stablecoin::Pyusd,
            Stablecoin::Dai,
            Stablecoin::Rlusd,
            Stablecoin::Eurc,
            Stablecoin::Fdusd,
        ] {
            assert!(!s.symbol().is_empty());
        }
    }

    #[test]
    fn usdc_covers_eight_deployments() {
        let deps = Stablecoin::Usdc.deployments();
        assert_eq!(deps.len(), 8);
        assert!(deps.iter().any(|d| d.at.chain == "solana"));
        assert!(deps.iter().any(|d| d.at.chain == "base"));
        assert!(deps.iter().all(|d| d.decimals == 6));
    }

    #[test]
    fn usd_peg_predicate() {
        assert!(Stablecoin::Usdc.is_usd_peg());
        assert!(Stablecoin::Dai.is_usd_peg());
        assert!(!Stablecoin::Eurc.is_usd_peg());
    }

    #[test]
    fn evm_addresses_lowercase_where_evm() {
        for s in [Stablecoin::Usdc, Stablecoin::Usdt, Stablecoin::Dai] {
            for d in s.deployments() {
                if d.at.address.starts_with("0x") {
                    assert_eq!(
                        d.at.address.to_ascii_lowercase(),
                        d.at.address,
                        "{}@{} should be lowercase",
                        s.symbol(),
                        d.at.chain
                    );
                }
            }
        }
    }

    #[test]
    fn dai_has_18_decimals() {
        for d in Stablecoin::Dai.deployments() {
            assert_eq!(d.decimals, 18);
        }
    }
}
