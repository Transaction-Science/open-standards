//! Prevote, precommit, and signature types.
//!
//! Tendermint distinguishes two vote phases per round: **prevote** (a
//! validator's intent to lock onto a proposed block) and **precommit**
//! (a validator's binding commitment to that block). 2/3+1 voting
//! power across precommits at the same `(height, round, block_id)`
//! forms a commit.

use serde::{Deserialize, Serialize};

use crate::block::Hash;
use crate::validator::ValidatorId;

/// 64-byte ed25519 signature wrapper.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct Signature(#[serde(with = "serde_bytes_array")] pub [u8; 64]);

impl Signature {
    /// Construct from a raw 64-byte signature.
    pub fn from_bytes(bytes: [u8; 64]) -> Self {
        Self(bytes)
    }

    /// Borrow the raw bytes.
    pub fn as_bytes(&self) -> &[u8; 64] {
        &self.0
    }
}

/// Two distinct vote phases per Tendermint round.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VoteType {
    /// Prevote — validator's tentative vote on the proposal.
    Prevote,
    /// Precommit — validator's binding commitment.
    Precommit,
}

/// A signed vote.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Vote {
    /// Prevote or precommit.
    pub kind: VoteType,
    /// Height of the block being voted on.
    pub height: u64,
    /// Round at which this vote was cast.
    pub round: u32,
    /// Block hash being voted for, or `None` for a nil vote.
    pub block_hash: Option<Hash>,
    /// Validator who cast the vote.
    pub validator: ValidatorId,
    /// ed25519 signature over the canonical vote-sign-bytes.
    pub signature: Signature,
}

impl Vote {
    /// Canonical sign-bytes used as the ed25519 message.
    ///
    /// Real Tendermint uses a versioned, Amino-tagged structure; here
    /// we use a fixed-layout Blake3 of `(kind, height, round, hash)`
    /// because the signature semantics are what matters internally.
    pub fn sign_bytes(&self, chain_id: &str) -> [u8; 32] {
        let mut hasher = blake3::Hasher::new();
        hasher.update(chain_id.as_bytes());
        hasher.update(&[match self.kind {
            VoteType::Prevote => 1,
            VoteType::Precommit => 2,
        }]);
        hasher.update(&self.height.to_be_bytes());
        hasher.update(&self.round.to_be_bytes());
        match &self.block_hash {
            Some(h) => {
                hasher.update(&[1]);
                hasher.update(h);
            }
            None => {
                hasher.update(&[0]);
            }
        }
        hasher.update(self.validator.as_bytes());
        *hasher.finalize().as_bytes()
    }
}

mod serde_bytes_array {
    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(bytes: &[u8; 64], s: S) -> Result<S::Ok, S::Error> {
        serde_bytes::Bytes::new(bytes).serialize(s)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<[u8; 64], D::Error> {
        let v: serde_bytes::ByteBuf = serde_bytes::ByteBuf::deserialize(d)?;
        let v = v.into_vec();
        if v.len() != 64 {
            return Err(serde::de::Error::custom("expected 64-byte signature"));
        }
        let mut out = [0u8; 64];
        out.copy_from_slice(&v);
        Ok(out)
    }

    use serde::Serialize;
}
