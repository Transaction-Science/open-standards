//! L2 + L1 catalog: chain id, parent settlement layer, finality
//! model, default public RPC.
//!
//! This is a *catalog*. It is informational — production operators
//! always run their own RPC and verify the chain id, never trusting
//! the bundled defaults. The catalog exists so a `Bring Your Own
//! Chain` operator wiring `op-rails-crypto::EvmJsonRpcGateway` can
//! pick from a known-good list rather than typing 8453 from memory.

use serde::{Deserialize, Serialize};

/// EIP-155 chain id. `u64` because the EIP-155 spec allows arbitrary
/// integers (and L2s have started reaching into the millions).
pub type ChainId = u64;

/// Where a chain ultimately settles its state.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SettlementLayer {
    /// Settles on Ethereum L1 (optimistic / zk rollups, validiums).
    Ethereum,
    /// Standalone L1 (Ethereum, Solana, Bitcoin, Polygon PoS).
    Self_,
    /// Settles on a parent chain that is itself an L2 (rare today
    /// but anticipated in L3 stacks). Field carries the parent's
    /// chain id.
    Parent(ChainId),
}

/// Finality model: how the chain decides a transaction is final.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum FinalityModel {
    /// Probabilistic: more confirmations = more confidence (BTC,
    /// pre-Merge Ethereum PoW).
    Probabilistic,
    /// PoS slot finality (post-Merge Ethereum, ~12.8 min in
    /// practice).
    PosFinality,
    /// Optimistic rollup: ~7-day challenge window for L1-finality;
    /// soft-finality typically a few seconds.
    OptimisticRollup,
    /// ZK rollup: L1-finality on proof inclusion (minutes to hours).
    ZkRollup,
    /// Solana-style BFT: ~32 slots to "finalized" commitment.
    BftFinalized,
}

/// What kind of chain this is.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ChainKind {
    /// EVM L1.
    EvmL1,
    /// EVM optimistic rollup L2.
    EvmOptimisticL2,
    /// EVM zk rollup L2.
    EvmZkL2,
    /// Non-EVM L1 (Solana).
    NonEvmL1,
    /// Non-EVM zk L2 (Starknet).
    NonEvmZkL2,
    /// Bitcoin.
    Bitcoin,
}

/// One chain entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ChainInfo {
    /// Canonical chain identifier (`"ethereum"`, `"base"`,
    /// `"arbitrum"`, ...). Matches `op-rails-crypto::TokenRef.chain`.
    pub name: &'static str,
    /// EIP-155 chain id (or `None` for non-EIP-155 chains like
    /// Solana / Bitcoin).
    pub chain_id: Option<ChainId>,
    /// What family of chain this is.
    pub kind: ChainKind,
    /// Where it settles.
    pub settlement: SettlementLayer,
    /// How it reaches finality.
    pub finality: FinalityModel,
    /// Default public RPC endpoint. **Informational only** — every
    /// production operator runs their own.
    pub default_rpc: &'static str,
}

/// Catalog of L1 + L2 chains relevant to OpenPay's stablecoin rail.
pub struct L2Catalog;

impl L2Catalog {
    /// Static catalog. Operator code typically does `.iter()` then
    /// filters on `kind` / `chain_id`.
    #[must_use]
    pub fn all() -> &'static [ChainInfo] {
        const fn entry(
            name: &'static str,
            chain_id: Option<ChainId>,
            kind: ChainKind,
            settlement: SettlementLayer,
            finality: FinalityModel,
            default_rpc: &'static str,
        ) -> ChainInfo {
            ChainInfo {
                name,
                chain_id,
                kind,
                settlement,
                finality,
                default_rpc,
            }
        }

        static CATALOG: &[ChainInfo] = &[
            entry(
                "ethereum",
                Some(1),
                ChainKind::EvmL1,
                SettlementLayer::Self_,
                FinalityModel::PosFinality,
                "https://eth.llamarpc.com",
            ),
            entry(
                "optimism",
                Some(10),
                ChainKind::EvmOptimisticL2,
                SettlementLayer::Ethereum,
                FinalityModel::OptimisticRollup,
                "https://mainnet.optimism.io",
            ),
            entry(
                "arbitrum",
                Some(42_161),
                ChainKind::EvmOptimisticL2,
                SettlementLayer::Ethereum,
                FinalityModel::OptimisticRollup,
                "https://arb1.arbitrum.io/rpc",
            ),
            entry(
                "base",
                Some(8_453),
                ChainKind::EvmOptimisticL2,
                SettlementLayer::Ethereum,
                FinalityModel::OptimisticRollup,
                "https://mainnet.base.org",
            ),
            entry(
                "zksync-era",
                Some(324),
                ChainKind::EvmZkL2,
                SettlementLayer::Ethereum,
                FinalityModel::ZkRollup,
                "https://mainnet.era.zksync.io",
            ),
            entry(
                "polygon-zkevm",
                Some(1_101),
                ChainKind::EvmZkL2,
                SettlementLayer::Ethereum,
                FinalityModel::ZkRollup,
                "https://zkevm-rpc.com",
            ),
            entry(
                "linea",
                Some(59_144),
                ChainKind::EvmZkL2,
                SettlementLayer::Ethereum,
                FinalityModel::ZkRollup,
                "https://rpc.linea.build",
            ),
            entry(
                "scroll",
                Some(534_352),
                ChainKind::EvmZkL2,
                SettlementLayer::Ethereum,
                FinalityModel::ZkRollup,
                "https://rpc.scroll.io",
            ),
            entry(
                "starknet",
                Some(0x534E_5F4D_4149_4E), // SN_MAIN
                ChainKind::NonEvmZkL2,
                SettlementLayer::Ethereum,
                FinalityModel::ZkRollup,
                "https://starknet-mainnet.public.blastapi.io",
            ),
            entry(
                "polygon",
                Some(137),
                ChainKind::EvmL1,
                SettlementLayer::Self_,
                FinalityModel::PosFinality,
                "https://polygon-rpc.com",
            ),
            entry(
                "solana",
                None,
                ChainKind::NonEvmL1,
                SettlementLayer::Self_,
                FinalityModel::BftFinalized,
                "https://api.mainnet-beta.solana.com",
            ),
            entry(
                "bitcoin",
                None,
                ChainKind::Bitcoin,
                SettlementLayer::Self_,
                FinalityModel::Probabilistic,
                "",
            ),
        ];
        CATALOG
    }

    /// Find by canonical name.
    #[must_use]
    pub fn by_name(name: &str) -> Option<&'static ChainInfo> {
        Self::all().iter().find(|c| c.name == name)
    }

    /// Find by EIP-155 chain id.
    #[must_use]
    pub fn by_chain_id(chain_id: ChainId) -> Option<&'static ChainInfo> {
        Self::all()
            .iter()
            .find(|c| c.chain_id == Some(chain_id))
    }

    /// All entries whose `kind` is in one of the EVM L2 families.
    #[must_use]
    pub fn evm_l2s() -> Vec<&'static ChainInfo> {
        Self::all()
            .iter()
            .filter(|c| matches!(c.kind, ChainKind::EvmOptimisticL2 | ChainKind::EvmZkL2))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn base_lookup_by_chain_id() {
        let c = L2Catalog::by_chain_id(8453).unwrap();
        assert_eq!(c.name, "base");
        assert_eq!(c.settlement, SettlementLayer::Ethereum);
        assert_eq!(c.kind, ChainKind::EvmOptimisticL2);
    }

    #[test]
    fn ethereum_lookup_by_name() {
        let c = L2Catalog::by_name("ethereum").unwrap();
        assert_eq!(c.chain_id, Some(1));
        assert_eq!(c.finality, FinalityModel::PosFinality);
    }

    #[test]
    fn evm_l2s_includes_all_six_evm_l2s() {
        let l2s = L2Catalog::evm_l2s();
        let names: Vec<&str> = l2s.iter().map(|c| c.name).collect();
        for expected in [
            "optimism",
            "arbitrum",
            "base",
            "zksync-era",
            "polygon-zkevm",
            "linea",
            "scroll",
        ] {
            assert!(names.contains(&expected), "missing {expected}");
        }
    }

    #[test]
    fn solana_has_no_eip155_chain_id() {
        let c = L2Catalog::by_name("solana").unwrap();
        assert!(c.chain_id.is_none());
        assert_eq!(c.kind, ChainKind::NonEvmL1);
    }

    #[test]
    fn unknown_returns_none() {
        assert!(L2Catalog::by_name("fnord").is_none());
        assert!(L2Catalog::by_chain_id(999_999_999).is_none());
    }
}
