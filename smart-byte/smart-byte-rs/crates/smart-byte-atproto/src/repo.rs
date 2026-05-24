//! Repository commits and signed commits.
//!
//! An AT Protocol repository is a sequence of *signed commits* over a
//! Merkle Search Tree of records. A commit is a CBOR map with:
//!
//! * `did`     — repo DID,
//! * `version` — commit format version (currently `3`),
//! * `data`    — CID of the MST root,
//! * `prev`    — CID of the previous commit, or `None` for genesis,
//! * `sig`     — Ed25519 signature over the unsigned commit bytes.
//!
//! This module builds and verifies that structure on top of
//! [`Mst`][crate::mst::Mst] and [`CarBlock`][crate::car::CarBlock].

use ed25519_dalek::{Signature, Signer, SigningKey, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;

use crate::car::{CarBlock, CarFile, Cid};
use crate::error::AtprotoError;
use crate::mst::Mst;

/// Current AT Protocol commit format version.
pub const COMMIT_VERSION: u32 = 3;

/// Unsigned commit. This is what gets signed; the signature is then
/// attached in [`SignedCommit`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UnsignedCommit {
    /// Repo DID (e.g. `"did:plc:abc..."`).
    pub did: String,
    /// Commit format version.
    pub version: u32,
    /// CID of the MST root, encoded as raw bytes.
    pub data: ByteBuf,
    /// CID of the previous commit, or `None` for genesis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prev: Option<ByteBuf>,
}

/// Signed commit on the wire.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SignedCommit {
    /// Repo DID.
    pub did: String,
    /// Commit format version.
    pub version: u32,
    /// CID of the MST root.
    pub data: ByteBuf,
    /// CID of the previous commit, or `None` for genesis.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub prev: Option<ByteBuf>,
    /// 64-byte Ed25519 signature over the unsigned-commit CBOR bytes.
    pub sig: ByteBuf,
}

impl SignedCommit {
    /// Strip the signature to recover the unsigned form, for verification.
    pub fn unsigned(&self) -> UnsignedCommit {
        UnsignedCommit {
            did: self.did.clone(),
            version: self.version,
            data: self.data.clone(),
            prev: self.prev.clone(),
        }
    }

    /// CID of this commit (`dag-cbor` over its serialised form).
    pub fn cid(&self) -> Result<Cid, AtprotoError> {
        let bytes = serde_cbor::to_vec(self)?;
        Ok(Cid::dag_cbor(&bytes))
    }

    /// Serialize for storage in a CAR block.
    pub fn to_car_block(&self) -> Result<CarBlock, AtprotoError> {
        let bytes = serde_cbor::to_vec(self)?;
        Ok(CarBlock::dag_cbor(bytes))
    }

    /// Verify the signature against `verifier`.
    pub fn verify(
        &self,
        verifier: &VerifyingKey,
    ) -> Result<(), AtprotoError> {
        let bytes = serde_cbor::to_vec(&self.unsigned())?;
        let sig =
            Signature::from_slice(self.sig.as_ref()).map_err(|e| {
                AtprotoError::Crypto(format!("bad sig bytes: {e}"))
            })?;
        verifier
            .verify(&bytes, &sig)
            .map_err(|e| AtprotoError::Crypto(format!("verify: {e}")))
    }
}

/// In-memory repository.
#[derive(Debug, Clone)]
pub struct Repo {
    /// Repo DID.
    pub did: String,
    /// The MST holding `(collection/rkey, CID)` entries.
    pub mst: Mst,
    /// CID of the last committed signed commit, if any.
    pub head: Option<Cid>,
}

impl Repo {
    /// Construct an empty repo for `did`.
    pub fn new(did: impl Into<String>) -> Self {
        Self {
            did: did.into(),
            mst: Mst::new(),
            head: None,
        }
    }

    /// Insert a record `(key, cid)`. The key must be
    /// `<collection>/<rkey>`.
    pub fn put_record(
        &mut self,
        key: impl Into<String>,
        cid: Cid,
    ) -> Result<(), AtprotoError> {
        let key = key.into();
        Mst::check_key(&key)?;
        self.mst.insert(key, cid);
        Ok(())
    }

    /// Delete a record by key. Returns the previous CID if any.
    pub fn delete_record(&mut self, key: &str) -> Option<Cid> {
        self.mst.delete(key)
    }

    /// Build an unsigned commit pointing at the current MST root.
    pub fn build_unsigned_commit(&self) -> UnsignedCommit {
        UnsignedCommit {
            did: self.did.clone(),
            version: COMMIT_VERSION,
            data: ByteBuf::from(self.mst.root_cid().to_bytes()),
            prev: self
                .head
                .as_ref()
                .map(|c| ByteBuf::from(c.to_bytes())),
        }
    }

    /// Sign the current state into a [`SignedCommit`] using `key`. The
    /// repo's `head` is updated to point at the new commit's CID.
    pub fn sign_commit(
        &mut self,
        key: &SigningKey,
    ) -> Result<SignedCommit, AtprotoError> {
        let unsigned = self.build_unsigned_commit();
        let bytes = serde_cbor::to_vec(&unsigned)?;
        let sig: Signature = key.sign(&bytes);
        let signed = SignedCommit {
            did: unsigned.did,
            version: unsigned.version,
            data: unsigned.data,
            prev: unsigned.prev,
            sig: ByteBuf::from(sig.to_bytes().to_vec()),
        };
        self.head = Some(signed.cid()?);
        Ok(signed)
    }

    /// Export the repo as a CAR file with the signed commit as the root.
    /// `records` is an iterator of `(cid, dag-cbor bytes)` block payloads
    /// to include alongside the commit; the MST root pointer is computed
    /// from the in-memory MST and does not require materialised MST node
    /// blocks for this sketch implementation.
    pub fn to_car(
        &self,
        commit: &SignedCommit,
        records: impl IntoIterator<Item = (Cid, Vec<u8>)>,
    ) -> Result<CarFile, AtprotoError> {
        let commit_block = commit.to_car_block()?;
        let mut car = CarFile::new(vec![commit_block.cid.clone()]);
        car.push(commit_block);
        for (cid, bytes) in records {
            car.push(CarBlock { cid, data: bytes });
        }
        Ok(car)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    #[test]
    fn signed_commit_roundtrip() {
        let mut csprng = OsRng;
        let key = SigningKey::generate(&mut csprng);
        let mut repo = Repo::new("did:plc:testtesttesttesttest");
        repo.put_record("app.bsky.feed.post/abc", Cid::dag_cbor(b"hi"))
            .unwrap();
        let signed = repo.sign_commit(&key).unwrap();
        signed.verify(&key.verifying_key()).unwrap();
    }

    #[test]
    fn commit_chain_updates_head() {
        let mut csprng = OsRng;
        let key = SigningKey::generate(&mut csprng);
        let mut repo = Repo::new("did:plc:testtesttesttesttest");
        repo.put_record("c/1", Cid::dag_cbor(b"v1")).unwrap();
        let first = repo.sign_commit(&key).unwrap();
        let first_cid = first.cid().unwrap();
        repo.put_record("c/2", Cid::dag_cbor(b"v2")).unwrap();
        let second = repo.sign_commit(&key).unwrap();
        let prev_cid_bytes = second.prev.as_ref().unwrap();
        let (prev_cid, _) = Cid::read_from(prev_cid_bytes.as_ref()).unwrap();
        assert_eq!(prev_cid, first_cid);
    }
}
