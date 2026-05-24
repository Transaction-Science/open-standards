//! Mediator role — an agent that holds messages for a mobile / edge
//! agent and delivers them on connection (Aries RFC 0211 + RFC 0685).
//!
//! A mediator does *not* decrypt messages; it stores opaque packed
//! payloads keyed by recipient DID and serves them on `delivery-request`.

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use smart_byte_did::Did;
use tokio::sync::Mutex;

use crate::error::DidcommError;

/// A stored, opaque (packed) message ready for delivery.
#[derive(Debug, Clone)]
pub struct StoredMessage {
    /// Unique id assigned by the mediator (NOT the DIDComm message id —
    /// that lives inside the encrypted blob).
    pub mediator_id: String,
    /// Packed DIDComm envelope (JWE JSON or compact JWS).
    pub packed: String,
}

/// Pluggable mediator storage backend.
#[async_trait]
pub trait MediatorStorage: Send + Sync {
    /// Queue a packed message for the given recipient.
    async fn enqueue(
        &self,
        recipient: &Did,
        packed: String,
    ) -> Result<String, DidcommError>;
    /// List queued messages for the given recipient (oldest first).
    async fn list(
        &self,
        recipient: &Did,
        limit: usize,
    ) -> Result<Vec<StoredMessage>, DidcommError>;
    /// Acknowledge (remove) messages by mediator id.
    async fn ack(&self, ids: &[String]) -> Result<(), DidcommError>;
    /// Count queued messages for the recipient.
    async fn count(&self, recipient: &Did) -> Result<u64, DidcommError>;
}

/// In-memory storage backend for tests and reference impls.
#[derive(Default)]
pub struct InMemoryStorage {
    queues: Mutex<HashMap<String, Vec<StoredMessage>>>,
}

#[async_trait]
impl MediatorStorage for InMemoryStorage {
    async fn enqueue(
        &self,
        recipient: &Did,
        packed: String,
    ) -> Result<String, DidcommError> {
        let id = uuid::Uuid::new_v4().to_string();
        let mut q = self.queues.lock().await;
        q.entry(recipient.to_string())
            .or_default()
            .push(StoredMessage {
                mediator_id: id.clone(),
                packed,
            });
        Ok(id)
    }

    async fn list(
        &self,
        recipient: &Did,
        limit: usize,
    ) -> Result<Vec<StoredMessage>, DidcommError> {
        let q = self.queues.lock().await;
        Ok(q.get(&recipient.to_string())
            .map(|v| v.iter().take(limit).cloned().collect())
            .unwrap_or_default())
    }

    async fn ack(&self, ids: &[String]) -> Result<(), DidcommError> {
        let mut q = self.queues.lock().await;
        for vec in q.values_mut() {
            vec.retain(|m| !ids.contains(&m.mediator_id));
        }
        Ok(())
    }

    async fn count(&self, recipient: &Did) -> Result<u64, DidcommError> {
        let q = self.queues.lock().await;
        Ok(q.get(&recipient.to_string()).map(|v| v.len() as u64).unwrap_or(0))
    }
}

/// A mediator. Wraps any [`MediatorStorage`] backend.
pub struct Mediator<S: MediatorStorage> {
    /// The mediator's own DID.
    pub did: Did,
    /// The mediator's DIDComm endpoint URL (advertised in `mediate-grant`).
    pub endpoint: String,
    /// Routing keys advertised to mediated clients.
    pub routing_keys: Vec<String>,
    /// Storage backend.
    pub storage: Arc<S>,
}

impl<S: MediatorStorage> Mediator<S> {
    /// Build a mediator with the given storage backend.
    pub fn new(
        did: Did,
        endpoint: impl Into<String>,
        routing_keys: Vec<String>,
        storage: Arc<S>,
    ) -> Self {
        Self {
            did,
            endpoint: endpoint.into(),
            routing_keys,
            storage,
        }
    }
}

/// Convenient default: a mediator with `InMemoryStorage`.
pub type InMemoryMediator = Mediator<InMemoryStorage>;

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn enqueue_then_list_then_ack() {
        let storage = Arc::new(InMemoryStorage::default());
        let did: Did = "did:example:mediator".parse().unwrap();
        let mediator = Mediator::new(
            did,
            "https://mediator.example.com/didcomm",
            vec![],
            storage.clone(),
        );
        let alice: Did = "did:example:alice".parse().unwrap();
        let id1 = mediator
            .storage
            .enqueue(&alice, "blob1".into())
            .await
            .unwrap();
        let _id2 = mediator
            .storage
            .enqueue(&alice, "blob2".into())
            .await
            .unwrap();
        assert_eq!(mediator.storage.count(&alice).await.unwrap(), 2);
        let listed = mediator.storage.list(&alice, 10).await.unwrap();
        assert_eq!(listed.len(), 2);
        mediator.storage.ack(&[id1]).await.unwrap();
        assert_eq!(mediator.storage.count(&alice).await.unwrap(), 1);
    }
}
