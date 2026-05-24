//! Circle CCTP v2 — cross-chain USDC burn / mint message format.
//!
//! CCTP v2 is Circle's native cross-chain USDC protocol. Flow:
//! 1. On source chain: `TokenMessenger.depositForBurn(amount,
//!    destinationDomain, mintRecipient, burnToken, destinationCaller,
//!    maxFee, minFinalityThreshold)`. This burns USDC on the source
//!    and emits a `MessageSent` event whose payload includes the
//!    `BurnMessage` body.
//! 2. Off-chain: caller polls Circle's attestation API for an
//!    attestation over the message hash.
//! 3. On destination chain: `MessageTransmitter.receiveMessage(
//!    message, attestation)` mints USDC.
//!
//! This module models the **message body** (the `BurnMessage` v2
//! struct) and the attestation envelope. The `MessageHeader` that
//! wraps it (sourceDomain, destinationDomain, sender, recipient,
//! destinationCaller, version=1) is not modelled here — the
//! MessageTransmitter contract assembles it from `depositForBurn`
//! args, so operators just need to surface the body fields. The
//! types here are sized so they can be encoded byte-for-byte to
//! match Circle's on-chain layout when needed.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// CCTP message version. v2 introduced fast / instant transfers
/// (minFinalityThreshold) and per-transfer `maxFee` hooks.
pub const CCTP_VERSION_V2: u32 = 1;

/// Circle's "domain" enumeration for source / destination chains.
/// Domain IDs are Circle's, not EIP-155.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CctpDomain {
    /// Ethereum mainnet.
    Ethereum,
    /// Avalanche C-Chain.
    Avalanche,
    /// Optimism.
    Optimism,
    /// Arbitrum.
    Arbitrum,
    /// Solana.
    Solana,
    /// Base.
    Base,
    /// Polygon PoS.
    Polygon,
    /// Unknown domain (forward-compat slot for chains added after
    /// this build).
    Other(u32),
}

impl CctpDomain {
    /// Circle's official domain id.
    #[must_use]
    pub const fn id(self) -> u32 {
        match self {
            Self::Ethereum => 0,
            Self::Avalanche => 1,
            Self::Optimism => 2,
            Self::Arbitrum => 3,
            Self::Solana => 5,
            Self::Base => 6,
            Self::Polygon => 7,
            Self::Other(n) => n,
        }
    }

    /// Resolve a Circle domain id to its enum.
    #[must_use]
    pub const fn from_id(id: u32) -> Self {
        match id {
            0 => Self::Ethereum,
            1 => Self::Avalanche,
            2 => Self::Optimism,
            3 => Self::Arbitrum,
            5 => Self::Solana,
            6 => Self::Base,
            7 => Self::Polygon,
            n => Self::Other(n),
        }
    }
}

/// CCTP v2 BurnMessage body.
///
/// Layout matches Circle's on-chain `BurnMessage.format(...)` — used
/// when the operator needs to construct the message bytes off-chain
/// (e.g. to compute the message hash for an attestation lookup
/// before the transaction is mined).
///
/// All addresses are 32-byte (left-padded for EVM, full 32 bytes
/// for Solana).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CctpBurnMessage {
    /// Version (= [`CCTP_VERSION_V2`]).
    pub version: u32,
    /// The token contract being burned (USDC on source). 32 bytes.
    pub burn_token: [u8; 32],
    /// Destination minting recipient. 32 bytes.
    pub mint_recipient: [u8; 32],
    /// Amount burned, in USDC's smallest unit (6 decimals).
    pub amount: u128,
    /// Address that called `depositForBurn` on source. 32 bytes.
    pub message_sender: [u8; 32],
    /// Max attestation fee operator will pay (v2-only). 0 for v1
    /// compatibility transfers.
    pub max_fee: u128,
    /// Fee actually charged by the attestor (set to 0 by the
    /// caller; updated by the destination side at mint time).
    pub fee_executed: u128,
    /// Block height (source-chain) at which burn was emitted, used
    /// to gate the destination's finality-threshold check.
    pub expiration_block: u64,
    /// Hook data: free-form bytes consumed by the destination
    /// caller (e.g. `receiveMessage`-then-call-this-contract).
    pub hook_data: Vec<u8>,
}

impl CctpBurnMessage {
    /// Construct.
    #[must_use]
    pub fn new(
        burn_token: [u8; 32],
        mint_recipient: [u8; 32],
        amount: u128,
        message_sender: [u8; 32],
    ) -> Self {
        Self {
            version: CCTP_VERSION_V2,
            burn_token,
            mint_recipient,
            amount,
            message_sender,
            max_fee: 0,
            fee_executed: 0,
            expiration_block: 0,
            hook_data: Vec::new(),
        }
    }

    /// Encode the body to its on-chain byte layout.
    ///
    /// Layout (228 + len(hook_data) bytes):
    /// - `version`         : 4 bytes, big-endian
    /// - `burn_token`      : 32 bytes
    /// - `mint_recipient`  : 32 bytes
    /// - `amount`          : 32 bytes big-endian (right-aligned u128)
    /// - `message_sender`  : 32 bytes
    /// - `max_fee`         : 32 bytes big-endian
    /// - `fee_executed`    : 32 bytes big-endian
    /// - `expiration_block`: 32 bytes big-endian (right-aligned u64)
    /// - `hook_data`       : variable-length tail
    #[must_use]
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(228 + self.hook_data.len());
        out.extend_from_slice(&self.version.to_be_bytes());
        out.extend_from_slice(&self.burn_token);
        out.extend_from_slice(&self.mint_recipient);
        push_u256(&mut out, self.amount);
        out.extend_from_slice(&self.message_sender);
        push_u256(&mut out, self.max_fee);
        push_u256(&mut out, self.fee_executed);
        let mut block_padded = [0u8; 32];
        block_padded[24..].copy_from_slice(&self.expiration_block.to_be_bytes());
        out.extend_from_slice(&block_padded);
        out.extend_from_slice(&self.hook_data);
        out
    }

    /// Decode a body previously emitted by [`Self::encode`].
    ///
    /// # Errors
    /// Returns [`Error::InvalidLayout`] if `bytes` is shorter than
    /// the fixed-prefix size (228 bytes) or the version is unknown.
    pub fn decode(bytes: &[u8]) -> Result<Self> {
        if bytes.len() < 228 {
            return Err(Error::InvalidLayout(format!(
                "cctp burn-message too short: {} bytes",
                bytes.len()
            )));
        }
        let version = u32::from_be_bytes(slice4(bytes, 0)?);
        if version != CCTP_VERSION_V2 {
            return Err(Error::Unsupported(format!("cctp version {version}")));
        }
        let mut burn_token = [0u8; 32];
        burn_token.copy_from_slice(&bytes[4..36]);
        let mut mint_recipient = [0u8; 32];
        mint_recipient.copy_from_slice(&bytes[36..68]);
        let amount = read_u128_be32(&bytes[68..100])?;
        let mut message_sender = [0u8; 32];
        message_sender.copy_from_slice(&bytes[100..132]);
        let max_fee = read_u128_be32(&bytes[132..164])?;
        let fee_executed = read_u128_be32(&bytes[164..196])?;
        let expiration_block = read_u64_be32(&bytes[196..228])?;
        let hook_data = bytes[228..].to_vec();
        Ok(Self {
            version,
            burn_token,
            mint_recipient,
            amount,
            message_sender,
            max_fee,
            fee_executed,
            expiration_block,
            hook_data,
        })
    }
}

/// Attestation returned by Circle's attestation API.
///
/// Wrapped in its own type so operators don't accidentally treat the
/// raw signature blob as a chain-side signature; the destination
/// `MessageTransmitter` is what verifies it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CctpAttestation {
    /// The opaque signature blob — usually multi-signer
    /// concatenation, length = 65 * N. Operator passes it through to
    /// `MessageTransmitter.receiveMessage`.
    pub signature_bytes: Vec<u8>,
    /// Echo of the message bytes the attestation is over. Operator
    /// double-checks this matches the message they're about to
    /// submit.
    pub message_bytes: Vec<u8>,
}

impl CctpAttestation {
    /// Construct.
    #[must_use]
    pub fn new(signature_bytes: Vec<u8>, message_bytes: Vec<u8>) -> Self {
        Self {
            signature_bytes,
            message_bytes,
        }
    }

    /// Structural sanity check: signature is non-empty and
    /// 65-byte-aligned (each attestor signature is r||s||v=65 bytes).
    ///
    /// # Errors
    /// Returns [`Error::Integrity`] on layout failure.
    pub fn check_layout(&self) -> Result<()> {
        if self.signature_bytes.is_empty() {
            return Err(Error::Integrity("attestation signature empty".into()));
        }
        if !self.signature_bytes.len().is_multiple_of(65) {
            return Err(Error::Integrity(format!(
                "attestation signature length {} not a multiple of 65",
                self.signature_bytes.len()
            )));
        }
        if self.message_bytes.is_empty() {
            return Err(Error::Integrity("attestation message empty".into()));
        }
        Ok(())
    }
}

/// Receipt returned by a successful destination-side mint. Operator
/// records this for reconciliation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CctpMintReceipt {
    /// Destination chain tx hash.
    pub tx_hash: String,
    /// Destination chain name.
    pub destination_chain: String,
    /// USDC minted (smallest unit). Should equal source `amount -
    /// fee_executed`.
    pub minted: u128,
    /// Fee charged by attestors.
    pub fee: u128,
}

fn push_u256(out: &mut Vec<u8>, n: u128) {
    let mut padded = [0u8; 32];
    padded[16..].copy_from_slice(&n.to_be_bytes());
    out.extend_from_slice(&padded);
}

fn slice4(bytes: &[u8], offset: usize) -> Result<[u8; 4]> {
    bytes
        .get(offset..offset + 4)
        .ok_or_else(|| Error::InvalidLayout(format!("slice4 at {offset}")))
        .and_then(|s| {
            let mut out = [0u8; 4];
            out.copy_from_slice(s);
            Ok(out)
        })
}

fn read_u128_be32(bytes: &[u8]) -> Result<u128> {
    if bytes.len() != 32 {
        return Err(Error::InvalidLayout(format!(
            "u128 field must be 32 bytes, got {}",
            bytes.len()
        )));
    }
    // High 16 bytes must be zero (u128 fits in the low 16 bytes).
    for b in &bytes[..16] {
        if *b != 0 {
            return Err(Error::Constraint {
                field: "amount",
                reason: "value exceeds u128".into(),
            });
        }
    }
    let mut buf = [0u8; 16];
    buf.copy_from_slice(&bytes[16..]);
    Ok(u128::from_be_bytes(buf))
}

fn read_u64_be32(bytes: &[u8]) -> Result<u64> {
    if bytes.len() != 32 {
        return Err(Error::InvalidLayout(format!(
            "u64 field must be 32 bytes, got {}",
            bytes.len()
        )));
    }
    for b in &bytes[..24] {
        if *b != 0 {
            return Err(Error::Constraint {
                field: "expiration_block",
                reason: "value exceeds u64".into(),
            });
        }
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&bytes[24..]);
    Ok(u64::from_be_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn domain_ids_match_circle() {
        assert_eq!(CctpDomain::Ethereum.id(), 0);
        assert_eq!(CctpDomain::Solana.id(), 5);
        assert_eq!(CctpDomain::Base.id(), 6);
    }

    #[test]
    fn domain_round_trip() {
        for id in [0u32, 1, 2, 3, 5, 6, 7, 99] {
            assert_eq!(CctpDomain::from_id(id).id(), id);
        }
    }

    #[test]
    fn encode_decode_round_trip() {
        let mut msg = CctpBurnMessage::new([0xab; 32], [0xcd; 32], 1_000_000, [0xef; 32]);
        msg.max_fee = 500;
        msg.expiration_block = 12_345_678;
        msg.hook_data = b"hook!".to_vec();

        let bytes = msg.encode();
        // Fixed prefix is 228 bytes.
        assert_eq!(bytes.len(), 228 + 5);
        let decoded = CctpBurnMessage::decode(&bytes).unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn decode_rejects_short() {
        let err = CctpBurnMessage::decode(&[0u8; 10]).unwrap_err();
        assert!(matches!(err, Error::InvalidLayout(_)));
    }

    #[test]
    fn decode_rejects_unknown_version() {
        let mut bytes = vec![0u8; 228];
        // version = 0xff
        bytes[3] = 0xff;
        let err = CctpBurnMessage::decode(&bytes).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    #[test]
    fn attestation_layout_check() {
        // 65 byte sig.
        let sig = vec![0xaa; 65];
        let msg_bytes = vec![1, 2, 3];
        let att = CctpAttestation::new(sig, msg_bytes);
        assert!(att.check_layout().is_ok());

        // Wrong length.
        let bad = CctpAttestation::new(vec![0xaa; 64], vec![1]);
        assert!(bad.check_layout().is_err());

        // Empty.
        let empty = CctpAttestation::new(vec![], vec![1]);
        assert!(empty.check_layout().is_err());
    }
}
