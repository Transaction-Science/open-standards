//! Firehose (`com.atproto.sync.subscribeRepos`) consumer.
//!
//! The real AT Protocol firehose is a WebSocket stream of CBOR-framed
//! events. To keep this crate transport-agnostic and unit-testable, the
//! consumer is parameterised over a [`FirehoseTransport`] trait: tests
//! and offline tools can drive it from an in-memory queue; production
//! deployments wire a WebSocket transport on top.

use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;

use crate::error::AtprotoError;

/// A single firehose event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FirehoseEvent {
    /// Sequence number assigned by the upstream relay.
    pub seq: u64,
    /// Repo DID this event pertains to.
    pub repo: String,
    /// ISO-8601 timestamp.
    pub time: String,
    /// Event kind discriminator.
    pub kind: FirehoseKind,
    /// CAR-encoded ops, where applicable. May be empty for tombstone events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub blocks: Option<ByteBuf>,
}

/// Firehose event kind.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub enum FirehoseKind {
    /// A new commit to a repo.
    Commit,
    /// A handle change for a repo.
    Handle,
    /// A migration of a repo to a different PDS.
    Migrate,
    /// A repo was tombstoned (deleted).
    Tombstone,
    /// An informational message.
    Info,
}

/// Transport for the firehose. Implementations yield events one at a
/// time; returning `Ok(None)` indicates end-of-stream.
#[async_trait]
pub trait FirehoseTransport: Send + Sync {
    /// Pull the next event from the upstream relay.
    async fn next_event(&mut self) -> Result<Option<FirehoseEvent>, AtprotoError>;
}

/// In-memory transport, used by tests.
pub struct MemoryTransport {
    queue: Vec<FirehoseEvent>,
    idx: usize,
}

impl MemoryTransport {
    /// Construct a transport that yields `events` in order.
    pub fn new(events: Vec<FirehoseEvent>) -> Self {
        Self {
            queue: events,
            idx: 0,
        }
    }
}

#[async_trait]
impl FirehoseTransport for MemoryTransport {
    async fn next_event(&mut self) -> Result<Option<FirehoseEvent>, AtprotoError> {
        if self.idx >= self.queue.len() {
            return Ok(None);
        }
        let ev = self.queue[self.idx].clone();
        self.idx += 1;
        Ok(Some(ev))
    }
}

/// Generic firehose consumer.
pub struct FirehoseClient<T: FirehoseTransport> {
    transport: T,
    last_seq: u64,
}

impl<T: FirehoseTransport> FirehoseClient<T> {
    /// Construct a client over `transport`, starting from sequence 0.
    pub fn new(transport: T) -> Self {
        Self {
            transport,
            last_seq: 0,
        }
    }

    /// Last sequence number observed.
    pub fn last_seq(&self) -> u64 {
        self.last_seq
    }

    /// Pull the next event, enforcing monotonic sequence numbers.
    pub async fn next(
        &mut self,
    ) -> Result<Option<FirehoseEvent>, AtprotoError> {
        let ev = self.transport.next_event().await?;
        if let Some(ref e) = ev {
            if e.seq < self.last_seq {
                return Err(AtprotoError::Network(format!(
                    "firehose sequence regression: {} < {}",
                    e.seq, self.last_seq
                )));
            }
            self.last_seq = e.seq;
        }
        Ok(ev)
    }

    /// Drain the transport, returning all remaining events.
    pub async fn drain(&mut self) -> Result<Vec<FirehoseEvent>, AtprotoError> {
        let mut out = Vec::new();
        while let Some(ev) = self.next().await? {
            out.push(ev);
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ev(seq: u64) -> FirehoseEvent {
        FirehoseEvent {
            seq,
            repo: "did:plc:test".into(),
            time: "2026-05-24T00:00:00Z".into(),
            kind: FirehoseKind::Commit,
            blocks: None,
        }
    }

    #[tokio::test]
    async fn drain_in_order() {
        let mut client = FirehoseClient::new(MemoryTransport::new(vec![
            ev(1),
            ev(2),
            ev(3),
        ]));
        let got = client.drain().await.unwrap();
        assert_eq!(got.len(), 3);
        assert_eq!(client.last_seq(), 3);
    }

    #[tokio::test]
    async fn sequence_regression_rejected() {
        let mut client = FirehoseClient::new(MemoryTransport::new(vec![
            ev(5),
            ev(3),
        ]));
        let _ = client.next().await.unwrap();
        let err = client.next().await.unwrap_err();
        assert!(matches!(err, AtprotoError::Network(_)));
    }
}
