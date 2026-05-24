//! Operations and the operation log.
//!
//! An [`Op`] describes a single mutation against a [`CrdtDocument`]: set
//! a register value, increment a counter, insert into an RGA list, etc.
//! Op ids are content-addressed: BLAKE3 over the canonical CBOR
//! encoding of the op with `id` zeroed. The op log is append-only and
//! ordered by HLC; identical ops applied twice are no-ops.

use serde::{Deserialize, Serialize};

use crate::document::{DocumentId, Value};
use crate::error::{CrdtError, Result};
use crate::hlc::{HybridLogicalClock, ReplicaId};
use crate::types::{RgaPos, UniqueTag};

/// Content-addressed op identifier (BLAKE3, 32 bytes).
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct OpId(pub [u8; 32]);

/// Path into a [`CrdtDocument`]. Slash-separated, leading `/` optional.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct Path(pub Vec<String>);

impl Path {
    /// Parse a slash-separated string into a `Path`.
    pub fn parse(s: &str) -> Self {
        let segments = s
            .split('/')
            .filter(|seg| !seg.is_empty())
            .map(str::to_owned)
            .collect();
        Path(segments)
    }
    /// Path as a slash-separated string.
    pub fn as_string(&self) -> String {
        format!("/{}", self.0.join("/"))
    }
    /// True iff this is the root path.
    pub fn is_root(&self) -> bool {
        self.0.is_empty()
    }
    /// Borrow the head segment, if any.
    pub fn head(&self) -> Option<&str> {
        self.0.first().map(String::as_str)
    }
    /// Path without its leading segment.
    pub fn tail(&self) -> Path {
        if self.0.is_empty() {
            Path::default()
        } else {
            Path(self.0[1..].to_vec())
        }
    }
}

/// Discriminated union of mutations.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "data")]
pub enum OpKind {
    /// Set a register-typed leaf at `target_path` to `value`.
    RegisterSet { value: Value },
    /// Apply a +/- delta to a counter at `target_path`.
    CounterAdd { delta: i64 },
    /// Add a value to an OR-set at `target_path`. Tag must be globally
    /// unique (constructed from the emitting replica + its HLC + nonce).
    SetAdd { value: Value, tag: UniqueTag },
    /// Remove a value from an OR-set at `target_path`. Tombstones every
    /// add-tag observed by this op.
    SetRemove { value: Value },
    /// Insert into an RGA list at `target_path`. `parent` is `None` to
    /// insert at the head, otherwise the position id of the preceding
    /// element.
    ListInsert {
        parent: Option<RgaPos>,
        value: Value,
        pos: RgaPos,
    },
    /// Delete an RGA list element at `target_path`.
    ListDelete { pos: RgaPos },
    /// Insert a character into a text node at `target_path`.
    TextInsert {
        parent: Option<RgaPos>,
        ch: char,
        pos: RgaPos,
    },
    /// Delete a character at `target_path`.
    TextDelete { pos: RgaPos },
    /// Map insert/replace at `target_path` of value `value` under `key`.
    MapSet { key: String, value: Value },
    /// Map delete at `target_path` of `key`.
    MapDelete { key: String },
}

/// A single operation. Op ids are computed by hashing the canonical CBOR
/// encoding of an op with `id` zeroed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Op {
    pub id: OpId,
    pub document: DocumentId,
    pub target_path: Path,
    pub op: OpKind,
    pub hlc: HybridLogicalClock,
    pub replica: ReplicaId,
}

impl Op {
    /// Construct an op and stamp its content-addressed id.
    pub fn new(
        document: DocumentId,
        target_path: Path,
        op: OpKind,
        hlc: HybridLogicalClock,
        replica: ReplicaId,
    ) -> Result<Self> {
        let mut out = Self {
            id: OpId::default(),
            document,
            target_path,
            op,
            hlc,
            replica,
        };
        out.id = out.compute_id()?;
        Ok(out)
    }

    /// Compute the content-addressed id without stamping it.
    pub fn compute_id(&self) -> Result<OpId> {
        let mut tmp = self.clone();
        tmp.id = OpId::default();
        let bytes = serde_cbor::to_vec(&tmp)?;
        let h = blake3::hash(&bytes);
        Ok(OpId(*h.as_bytes()))
    }

    /// Verify the stamped id matches a freshly-computed one.
    pub fn verify_id(&self) -> Result<()> {
        let computed = self.compute_id()?;
        if computed == self.id {
            Ok(())
        } else {
            Err(CrdtError::OpIntegrity("op id mismatch".into()))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hlc::HybridLogicalClock;

    #[test]
    fn path_round_trip() {
        let p = Path::parse("/users/alice/balance");
        assert_eq!(
            p,
            Path(vec!["users".into(), "alice".into(), "balance".into()])
        );
        assert_eq!(p.as_string(), "/users/alice/balance");
        assert_eq!(p.head(), Some("users"));
        assert_eq!(p.tail().as_string(), "/alice/balance");
    }

    #[test]
    fn op_id_is_content_addressed() {
        let r = ReplicaId::new(1);
        let op = Op::new(
            DocumentId::from_bytes(b"doc"),
            Path::parse("/x"),
            OpKind::CounterAdd { delta: 1 },
            HybridLogicalClock {
                wall: 1,
                logical: 0,
                node: r,
            },
            r,
        )
        .unwrap();
        op.verify_id().unwrap();

        let op2 = Op::new(
            DocumentId::from_bytes(b"doc"),
            Path::parse("/x"),
            OpKind::CounterAdd { delta: 1 },
            HybridLogicalClock {
                wall: 1,
                logical: 0,
                node: r,
            },
            r,
        )
        .unwrap();
        assert_eq!(op.id, op2.id);
    }

    #[test]
    fn different_ops_have_different_ids() {
        let r = ReplicaId::new(1);
        let a = Op::new(
            DocumentId::from_bytes(b"doc"),
            Path::parse("/x"),
            OpKind::CounterAdd { delta: 1 },
            HybridLogicalClock {
                wall: 1,
                logical: 0,
                node: r,
            },
            r,
        )
        .unwrap();
        let b = Op::new(
            DocumentId::from_bytes(b"doc"),
            Path::parse("/x"),
            OpKind::CounterAdd { delta: 2 },
            HybridLogicalClock {
                wall: 1,
                logical: 0,
                node: r,
            },
            r,
        )
        .unwrap();
        assert_ne!(a.id, b.id);
    }
}
