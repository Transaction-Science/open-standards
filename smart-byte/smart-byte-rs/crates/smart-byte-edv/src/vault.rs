//! `Vault` trait and in-memory reference implementation (DIF EDV v0.10
//! § 6 — HTTP API surface, mapped to a Rust async trait).
//!
//! A vault is a passive ciphertext store: it accepts
//! [`crate::spec::EncryptedDocument`] records, indexes the *blinded*
//! `IndexedEntry` tags, and serves equality queries over those tags. It
//! never sees the plaintext content or attribute values.
//!
//! [`InMemoryVault`] is a minimal reference implementation suitable for
//! tests and embedded deployments. A real production vault would back
//! these operations with a persistent store and enforce ZCAP-LD
//! capability invocations (see [`crate::zcap`]).

use async_trait::async_trait;
use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use crate::error::EdvError;
use crate::index::Query;
use crate::spec::{Config, EncryptedDocument};

/// A reference to a stored document. Used as the return value from
/// [`Vault::insert`] and [`Vault::list`] so callers can fetch the full
/// record by id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DocumentRef {
    /// Document id (URN).
    pub id: String,
    /// Sequence number of the stored revision.
    pub sequence: u64,
}

/// The vault HTTP API, expressed as an async trait.
#[async_trait]
pub trait Vault: Send + Sync {
    /// Read the vault configuration.
    async fn config(&self) -> Result<Config, EdvError>;
    /// Insert a new document. Returns the new [`DocumentRef`].
    async fn insert(
        &self,
        doc: EncryptedDocument,
    ) -> Result<DocumentRef, EdvError>;
    /// Fetch a document by id.
    async fn get(&self, id: &str) -> Result<EncryptedDocument, EdvError>;
    /// Update an existing document. The supplied document's `sequence`
    /// MUST match the currently-stored sequence; the stored sequence is
    /// bumped after a successful update.
    async fn update(
        &self,
        doc: EncryptedDocument,
    ) -> Result<DocumentRef, EdvError>;
    /// Delete a document by id.
    async fn delete(&self, id: &str) -> Result<(), EdvError>;
    /// List every document in the vault.
    async fn list(&self) -> Result<Vec<DocumentRef>, EdvError>;
    /// Evaluate `query` against the vault's encrypted index.
    async fn query(
        &self,
        query: &Query,
    ) -> Result<Vec<EncryptedDocument>, EdvError>;
}

/// A minimal in-memory implementation of [`Vault`].
#[derive(Clone)]
pub struct InMemoryVault {
    cfg: Arc<Config>,
    docs: Arc<RwLock<HashMap<String, EncryptedDocument>>>,
}

impl InMemoryVault {
    /// Create an empty vault bound to `cfg`.
    pub fn new(cfg: Config) -> Self {
        Self {
            cfg: Arc::new(cfg),
            docs: Arc::new(RwLock::new(HashMap::new())),
        }
    }

    /// Count documents currently held.
    pub async fn len(&self) -> usize {
        self.docs.read().await.len()
    }

    /// Convenience: is the vault empty?
    pub async fn is_empty(&self) -> bool {
        self.docs.read().await.is_empty()
    }
}

#[async_trait]
impl Vault for InMemoryVault {
    async fn config(&self) -> Result<Config, EdvError> {
        Ok((*self.cfg).clone())
    }

    async fn insert(
        &self,
        mut doc: EncryptedDocument,
    ) -> Result<DocumentRef, EdvError> {
        let mut docs = self.docs.write().await;
        if docs.contains_key(&doc.id) {
            return Err(EdvError::Internal(format!(
                "document {} already exists",
                doc.id
            )));
        }
        if doc.sequence == 0 {
            doc.sequence = 1;
        }
        let r = DocumentRef {
            id: doc.id.clone(),
            sequence: doc.sequence,
        };
        docs.insert(doc.id.clone(), doc);
        Ok(r)
    }

    async fn get(&self, id: &str) -> Result<EncryptedDocument, EdvError> {
        let docs = self.docs.read().await;
        docs.get(id)
            .cloned()
            .ok_or_else(|| EdvError::NotFound(id.to_string()))
    }

    async fn update(
        &self,
        mut doc: EncryptedDocument,
    ) -> Result<DocumentRef, EdvError> {
        let mut docs = self.docs.write().await;
        let existing = docs
            .get(&doc.id)
            .ok_or_else(|| EdvError::NotFound(doc.id.clone()))?;
        if doc.sequence != existing.sequence {
            return Err(EdvError::Internal(format!(
                "sequence mismatch on {} (have {}, sent {})",
                doc.id, existing.sequence, doc.sequence
            )));
        }
        doc.sequence = existing.sequence + 1;
        let r = DocumentRef {
            id: doc.id.clone(),
            sequence: doc.sequence,
        };
        docs.insert(doc.id.clone(), doc);
        Ok(r)
    }

    async fn delete(&self, id: &str) -> Result<(), EdvError> {
        let mut docs = self.docs.write().await;
        if docs.remove(id).is_none() {
            return Err(EdvError::NotFound(id.to_string()));
        }
        Ok(())
    }

    async fn list(&self) -> Result<Vec<DocumentRef>, EdvError> {
        let docs = self.docs.read().await;
        let mut out: Vec<DocumentRef> = docs
            .values()
            .map(|d| DocumentRef {
                id: d.id.clone(),
                sequence: d.sequence,
            })
            .collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(out)
    }

    async fn query(
        &self,
        query: &Query,
    ) -> Result<Vec<EncryptedDocument>, EdvError> {
        let docs = self.docs.read().await;
        let mut out: Vec<EncryptedDocument> = docs
            .values()
            .filter(|d| query.matches(d))
            .cloned()
            .collect();
        out.sort_by(|a, b| a.id.cmp(&b.id));
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::KeyDescriptor;

    fn cfg() -> Config {
        let kex = KeyDescriptor {
            id: "did:example:alice#kex-1".into(),
            key_type: "JsonWebKey2020".into(),
            controller: None,
        };
        let hmac = KeyDescriptor {
            id: "did:example:alice#hmac-1".into(),
            key_type: "Sha256HmacKey2019".into(),
            controller: None,
        };
        Config::new("urn:uuid:vault", "did:example:alice", kex, hmac)
    }

    fn doc(id: &str) -> EncryptedDocument {
        EncryptedDocument {
            id: id.into(),
            sequence: 0,
            jwe: serde_json::json!({}),
            indexed: Vec::new(),
            stream: None,
        }
    }

    #[tokio::test]
    async fn insert_get_round_trip() {
        let v = InMemoryVault::new(cfg());
        let r = v.insert(doc("urn:a")).await.expect("insert");
        assert_eq!(r.sequence, 1);
        let d = v.get("urn:a").await.expect("get");
        assert_eq!(d.id, "urn:a");
    }

    #[tokio::test]
    async fn duplicate_insert_fails() {
        let v = InMemoryVault::new(cfg());
        v.insert(doc("urn:a")).await.expect("insert");
        let res = v.insert(doc("urn:a")).await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn update_bumps_sequence() {
        let v = InMemoryVault::new(cfg());
        let r1 = v.insert(doc("urn:a")).await.expect("insert");
        let mut d = v.get("urn:a").await.expect("get");
        d.sequence = r1.sequence;
        let r2 = v.update(d).await.expect("update");
        assert_eq!(r2.sequence, r1.sequence + 1);
    }

    #[tokio::test]
    async fn update_with_wrong_sequence_fails() {
        let v = InMemoryVault::new(cfg());
        v.insert(doc("urn:a")).await.expect("insert");
        let mut d = v.get("urn:a").await.expect("get");
        d.sequence = 999;
        let res = v.update(d).await;
        assert!(res.is_err());
    }

    #[tokio::test]
    async fn delete_and_list() {
        let v = InMemoryVault::new(cfg());
        v.insert(doc("urn:a")).await.expect("insert");
        v.insert(doc("urn:b")).await.expect("insert");
        assert_eq!(v.len().await, 2);
        v.delete("urn:a").await.expect("delete");
        assert_eq!(v.len().await, 1);
        let refs = v.list().await.expect("list");
        assert_eq!(refs.len(), 1);
        assert_eq!(refs[0].id, "urn:b");
    }

    #[tokio::test]
    async fn delete_unknown_fails() {
        let v = InMemoryVault::new(cfg());
        let res = v.delete("urn:nope").await;
        assert!(matches!(res, Err(EdvError::NotFound(_))));
    }
}
