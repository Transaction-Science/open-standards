//! Connection abstractions and replay-protection helpers.
//!
//! A [`Connection`] is an asynchronous bidirectional channel between two
//! DIDs. The trait is intentionally minimal — concrete transports
//! (HTTPS, websocket, message-pickup-via-mediator) implement it. For
//! tests and in-process flows the [`InMemoryConnection`] uses an
//! `mpsc` pair.

use std::collections::HashSet;
use std::sync::Arc;

use async_trait::async_trait;
use smart_byte_did::Did;
use tokio::sync::Mutex;
use tokio::sync::mpsc;

use crate::error::DidcommError;
use crate::message::DidcommMessage;

/// An asynchronous DIDComm connection between two DIDs.
#[async_trait]
pub trait Connection: Send + Sync {
    /// Connection id (locally unique).
    fn id(&self) -> &str;
    /// Their DID.
    fn theirs(&self) -> &Did;
    /// Our DID.
    fn ours(&self) -> &Did;
    /// Send a message to the other party.
    async fn send(&self, msg: DidcommMessage) -> Result<(), DidcommError>;
    /// Receive the next message from the other party. Blocks until one
    /// is available or the connection is closed.
    async fn receive(&self) -> Result<DidcommMessage, DidcommError>;
}

/// In-memory connection backed by a pair of `mpsc` channels. Useful for
/// tests and for in-process agent-to-agent flows.
pub struct InMemoryConnection {
    /// Locally unique id.
    pub id: String,
    /// Our DID.
    pub ours: Did,
    /// Their DID.
    pub theirs: Did,
    outbox: mpsc::UnboundedSender<DidcommMessage>,
    inbox: Arc<Mutex<mpsc::UnboundedReceiver<DidcommMessage>>>,
    seen_ids: Arc<Mutex<HashSet<String>>>,
}

impl InMemoryConnection {
    /// Construct a connected pair `(alice_side, bob_side)`.
    pub fn pair(alice: Did, bob: Did) -> (Self, Self) {
        let (atx, brx) = mpsc::unbounded_channel();
        let (btx, arx) = mpsc::unbounded_channel();
        let a = InMemoryConnection {
            id: uuid::Uuid::new_v4().to_string(),
            ours: alice.clone(),
            theirs: bob.clone(),
            outbox: atx,
            inbox: Arc::new(Mutex::new(arx)),
            seen_ids: Arc::new(Mutex::new(HashSet::new())),
        };
        let b = InMemoryConnection {
            id: uuid::Uuid::new_v4().to_string(),
            ours: bob,
            theirs: alice,
            outbox: btx,
            inbox: Arc::new(Mutex::new(brx)),
            seen_ids: Arc::new(Mutex::new(HashSet::new())),
        };
        (a, b)
    }
}

#[async_trait]
impl Connection for InMemoryConnection {
    fn id(&self) -> &str {
        &self.id
    }
    fn theirs(&self) -> &Did {
        &self.theirs
    }
    fn ours(&self) -> &Did {
        &self.ours
    }

    async fn send(&self, msg: DidcommMessage) -> Result<(), DidcommError> {
        self.outbox
            .send(msg)
            .map_err(|e| DidcommError::Internal(format!("send failed: {e}")))
    }

    async fn receive(&self) -> Result<DidcommMessage, DidcommError> {
        let mut rx = self.inbox.lock().await;
        let m = rx
            .recv()
            .await
            .ok_or_else(|| DidcommError::Internal("connection closed".into()))?;
        let mut seen = self.seen_ids.lock().await;
        if !seen.insert(m.id.clone()) {
            return Err(DidcommError::Replay(m.id));
        }
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::protocols::trust_ping::{
        PingBody, TrustPingKind, respond_to,
    };
    use crate::protocol::ProtocolMessage;

    #[tokio::test]
    async fn trust_ping_round_trip() {
        let alice: Did = "did:example:alice".parse().unwrap();
        let bob: Did = "did:example:bob".parse().unwrap();
        let (a, b) = InMemoryConnection::pair(alice.clone(), bob.clone());

        let ping = TrustPingKind::Ping(PingBody {
            response_requested: true,
            comment: None,
        })
        .to_message()
        .from_did(alice)
        .to_dids(vec![bob.clone()]);
        a.send(ping.clone()).await.unwrap();

        let recvd = b.receive().await.unwrap();
        let resp = respond_to(&recvd);
        b.send(resp).await.unwrap();
        let pong = a.receive().await.unwrap();
        match TrustPingKind::from_message(&pong).unwrap() {
            TrustPingKind::PingResponse(_) => {}
            _ => panic!("expected ping-response"),
        }
    }

    #[tokio::test]
    async fn replay_detected() {
        let alice: Did = "did:example:alice".parse().unwrap();
        let bob: Did = "did:example:bob".parse().unwrap();
        let (a, b) = InMemoryConnection::pair(alice, bob);
        let m = TrustPingKind::Ping(PingBody::default()).to_message();
        a.send(m.clone()).await.unwrap();
        b.receive().await.unwrap();
        // Push the same id back into b's inbox.
        a.send(m).await.unwrap();
        let res = b.receive().await;
        assert!(matches!(res, Err(DidcommError::Replay(_))));
    }
}
