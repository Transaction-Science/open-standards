//! Vault replication (DIF EDV v0.10 § 9 — non-normative; modelled here
//! after the implementation pattern shared by Aries cloud-agent and
//! Bedrock's Bluesky-style replication).
//!
//! Replication moves [`EncryptedDocument`] records between two vaults
//! that share a common controller (typically a primary + a hot replica,
//! or a desktop + mobile pair). Because documents are *opaque ciphertext*
//! to the vault, replication does not need to decrypt anything — it only
//! needs to compare document sequences and copy the higher-sequence
//! record across.
//!
//! [`sync`] implements a one-way pull from `source` into `target`: for
//! every document in `source`, if `target` does not have a corresponding
//! record (or has an older `sequence`), the document is copied over.

use async_trait::async_trait;

use crate::error::EdvError;
use crate::spec::EncryptedDocument;
use crate::vault::Vault;

/// Outcome of a single replication pass.
#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ReplicationReport {
    /// Documents inserted into `target` because they were missing.
    pub inserted: Vec<String>,
    /// Documents updated in `target` because `source` had a higher
    /// sequence.
    pub updated: Vec<String>,
    /// Documents skipped because `target` already had an equal-or-newer
    /// sequence.
    pub skipped: Vec<String>,
}

/// One-way pull replication: copy every document from `source` into
/// `target` if `target` is missing it or holds an older revision.
pub async fn sync(
    source: &dyn Vault,
    target: &dyn Vault,
) -> Result<ReplicationReport, EdvError> {
    let mut report = ReplicationReport::default();
    let refs = source.list().await?;
    for r in refs {
        let doc = source.get(&r.id).await?;
        match target.get(&r.id).await {
            Ok(existing) => {
                if doc.sequence > existing.sequence {
                    let mut to_send = doc.clone();
                    to_send.sequence = existing.sequence;
                    target.update(to_send).await?;
                    report.updated.push(r.id);
                } else {
                    report.skipped.push(r.id);
                }
            }
            Err(EdvError::NotFound(_)) => {
                let mut to_send = doc.clone();
                to_send.sequence = 0;
                target.insert(to_send).await?;
                report.inserted.push(r.id);
            }
            Err(other) => return Err(other),
        }
    }
    Ok(report)
}

/// A replication strategy abstraction, in case callers want to wire in
/// a different policy (e.g. push, push+pull, conflict-on-divergent).
#[async_trait]
pub trait Replicator: Send + Sync {
    /// Run a replication pass and return the report.
    async fn replicate(
        &self,
        source: &dyn Vault,
        target: &dyn Vault,
    ) -> Result<ReplicationReport, EdvError>;
}

/// The canonical one-way pull strategy implementing [`sync`].
pub struct PullReplicator;

#[async_trait]
impl Replicator for PullReplicator {
    async fn replicate(
        &self,
        source: &dyn Vault,
        target: &dyn Vault,
    ) -> Result<ReplicationReport, EdvError> {
        sync(source, target).await
    }
}

/// Convenience: convert a list of documents to a Vec for diffing.
pub fn snapshot(docs: &[EncryptedDocument]) -> Vec<(String, u64)> {
    docs.iter()
        .map(|d| (d.id.clone(), d.sequence))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::spec::{Config, EncryptedDocument, KeyDescriptor};
    use crate::vault::InMemoryVault;

    fn cfg() -> Config {
        let k = KeyDescriptor {
            id: "did:example:alice#kex-1".into(),
            key_type: "JsonWebKey2020".into(),
            controller: None,
        };
        let h = KeyDescriptor {
            id: "did:example:alice#hmac-1".into(),
            key_type: "Sha256HmacKey2019".into(),
            controller: None,
        };
        Config::new("urn:uuid:vault", "did:example:alice", k, h)
    }

    fn doc(id: &str) -> EncryptedDocument {
        EncryptedDocument {
            id: id.into(),
            sequence: 0,
            jwe: serde_json::json!({"v": id}),
            indexed: Vec::new(),
            stream: None,
        }
    }

    #[tokio::test]
    async fn pull_inserts_missing() {
        let src = InMemoryVault::new(cfg());
        let dst = InMemoryVault::new(cfg());
        src.insert(doc("urn:a")).await.expect("insert a");
        src.insert(doc("urn:b")).await.expect("insert b");
        let r = sync(&src, &dst).await.expect("sync");
        assert_eq!(r.inserted.len(), 2);
        assert_eq!(dst.len().await, 2);
    }

    #[tokio::test]
    async fn pull_skips_when_target_current() {
        let src = InMemoryVault::new(cfg());
        let dst = InMemoryVault::new(cfg());
        src.insert(doc("urn:a")).await.expect("insert src");
        dst.insert(doc("urn:a")).await.expect("insert dst");
        let r = sync(&src, &dst).await.expect("sync");
        assert!(r.inserted.is_empty());
        assert!(r.updated.is_empty());
        assert_eq!(r.skipped, vec!["urn:a".to_string()]);
    }

    #[tokio::test]
    async fn pull_updates_when_source_newer() {
        let src = InMemoryVault::new(cfg());
        let dst = InMemoryVault::new(cfg());
        src.insert(doc("urn:a")).await.expect("insert src");
        dst.insert(doc("urn:a")).await.expect("insert dst");
        // Bump source twice.
        let mut s = src.get("urn:a").await.expect("get src");
        let cur = s.sequence;
        s.sequence = cur;
        let _ = src.update(s).await.expect("update src");
        let r = sync(&src, &dst).await.expect("sync");
        assert_eq!(r.updated, vec!["urn:a".to_string()]);
    }
}
