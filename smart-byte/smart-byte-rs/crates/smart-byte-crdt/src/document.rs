//! Tree-structured CRDT documents.
//!
//! A [`CrdtDocument`] is a tree of [`CrdtNode`]s with addressable paths.
//! The root is always a `CrdtNode::Map`. Operations land at a path
//! (`/users/alice/balance`), and the document maintains an append-only
//! op log plus a Hybrid Logical Clock.

use serde::{Deserialize, Serialize};

use crate::error::{CrdtError, Result};
use crate::hlc::{HlcClock, HybridLogicalClock, ReplicaId};
use crate::ops::{Op, OpKind, Path};
use crate::types::{
    CrdtId, LwwMap, LwwRegister, OrSet, PnCounter, RgaList, RgaPos, UniqueTag,
};

/// Globally unique identifier for a CRDT document. Stable; survives
/// merges; survives serialisation.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Default, Serialize, Deserialize)]
#[serde(transparent)]
pub struct DocumentId(pub [u8; 32]);

impl DocumentId {
    /// Derive a deterministic id by hashing arbitrary bytes.
    pub fn from_bytes(bytes: &[u8]) -> Self {
        let h = blake3::hash(bytes);
        Self(*h.as_bytes())
    }
}

/// Polymorphic value held by register / map / set / list nodes.
///
/// CBOR-friendly. The `Json` variant carries an opaque JSON string so
/// callers can embed structured data without dragging in a wider type
/// graph. `Float` carries the IEEE-754 bit pattern as `u64` so equality
/// and hashing are exact; convert with [`Value::float`] / [`Value::as_f64`].
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(tag = "t", content = "v")]
pub enum Value {
    Null,
    Bool(bool),
    Int(i64),
    FloatBits(u64),
    Text(String),
    Bytes(Vec<u8>),
    Json(String),
}

impl Value {
    /// Construct a `Value::FloatBits` from an `f64`.
    pub fn float(f: f64) -> Self {
        Value::FloatBits(f.to_bits())
    }
    /// Recover the `f64` from a `Value::FloatBits`, or `None` for other variants.
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::FloatBits(b) => Some(f64::from_bits(*b)),
            _ => None,
        }
    }
}

/// Recursive CRDT node.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", content = "node")]
pub enum CrdtNode {
    Map(LwwMap<String, CrdtNode>),
    List(RgaList<CrdtNode>),
    Text(RgaList<char>),
    Register(LwwRegister<Value>),
    Counter(PnCounter),
    Set(OrSet<Value>),
}

impl CrdtNode {
    fn kind(&self) -> &'static str {
        match self {
            CrdtNode::Map(_) => "map",
            CrdtNode::List(_) => "list",
            CrdtNode::Text(_) => "text",
            CrdtNode::Register(_) => "register",
            CrdtNode::Counter(_) => "counter",
            CrdtNode::Set(_) => "set",
        }
    }
}

/// A tree-structured CRDT document.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CrdtDocument {
    pub id: DocumentId,
    pub root: CrdtNode,
    pub history: Vec<Op>,
    pub clock: HybridLogicalClock,
}

impl CrdtDocument {
    /// Construct an empty document rooted at an LWW map.
    pub fn new(id: DocumentId, replica: ReplicaId) -> Self {
        Self {
            id,
            root: CrdtNode::Map(LwwMap::new(CrdtId::from_bytes(&id.0))),
            history: Vec::new(),
            clock: HybridLogicalClock {
                wall: 0,
                logical: 0,
                node: replica,
            },
        }
    }

    /// Construct from explicit root.
    pub fn with_root(id: DocumentId, root: CrdtNode, replica: ReplicaId) -> Self {
        Self {
            id,
            root,
            history: Vec::new(),
            clock: HybridLogicalClock {
                wall: 0,
                logical: 0,
                node: replica,
            },
        }
    }

    /// High-level helper: set a [`Value::Int`] at the given path. Will
    /// auto-create intermediate maps and the terminal register.
    pub fn set_at(&mut self, path: &str, value: i64, clock: &mut HlcClock) -> Result<Op> {
        self.write_at(path, Value::Int(value), clock)
    }

    /// High-level helper: set an arbitrary value at the given path.
    pub fn write_at(&mut self, path: &str, value: Value, clock: &mut HlcClock) -> Result<Op> {
        let p = Path::parse(path);
        let hlc = clock.now();
        let op = Op::new(
            self.id,
            p.clone(),
            OpKind::RegisterSet { value },
            hlc,
            clock.node(),
        )?;
        self.apply_op_internal(&op)?;
        self.history.push(op.clone());
        self.clock = hlc;
        Ok(op)
    }

    /// Apply an externally-produced op. Idempotent: if the op id has
    /// already been seen, this is a no-op.
    pub fn apply_op(&mut self, op: &Op) -> Result<bool> {
        op.verify_id()?;
        if self.history.iter().any(|h| h.id == op.id) {
            return Ok(false);
        }
        self.apply_op_internal(op)?;
        self.history.push(op.clone());
        if op.hlc > self.clock {
            self.clock = op.hlc;
        }
        Ok(true)
    }

    /// Borrow a node at the given path.
    pub fn get_node(&self, path: &str) -> Option<&CrdtNode> {
        let p = Path::parse(path);
        Self::traverse(&self.root, &p)
    }

    fn traverse<'a>(node: &'a CrdtNode, path: &Path) -> Option<&'a CrdtNode> {
        if path.is_root() {
            return Some(node);
        }
        let head = path.head()?;
        match node {
            CrdtNode::Map(m) => {
                let child = m.get(&head.to_string())?;
                Self::traverse(child, &path.tail())
            }
            _ => None,
        }
    }

    fn apply_op_internal(&mut self, op: &Op) -> Result<()> {
        Self::apply_recursive(&mut self.root, &op.target_path, op)
    }

    fn apply_recursive(node: &mut CrdtNode, path: &Path, op: &Op) -> Result<()> {
        if path.is_root() {
            return Self::apply_terminal(node, op);
        }
        let head = path.head().ok_or_else(|| {
            CrdtError::InvalidPath(format!(
                "expected non-empty path for op {}",
                op.target_path.as_string()
            ))
        })?;
        let tail = path.tail();

        // Special case: a MapSet / MapDelete at the parent map.
        if tail.is_root()
            && let CrdtNode::Map(map) = node
        {
            match &op.op {
                OpKind::MapSet { key, value } => {
                    let child = CrdtNode::Register(LwwRegister::new(
                        CrdtId::from_bytes(key.as_bytes()),
                        value.clone(),
                        op.hlc,
                        op.replica,
                    ));
                    map.set(key.clone(), child, op.hlc, op.replica);
                    return Ok(());
                }
                OpKind::MapDelete { key } => {
                    map.remove(key, op.hlc, op.replica);
                    return Ok(());
                }
                _ => {}
            }
        }

        match node {
            CrdtNode::Map(map) => {
                // Ensure the child exists; auto-create a map for non-terminal
                // paths, or a register for terminal RegisterSet.
                let key = head.to_string();
                let needs_create = map.get(&key).is_none();
                if needs_create {
                    let new_child = if tail.is_root() {
                        Self::seed_terminal_node(&op.op, &op.hlc, op.replica)
                    } else {
                        CrdtNode::Map(LwwMap::new(CrdtId::from_bytes(key.as_bytes())))
                    };
                    map.set(key.clone(), new_child, op.hlc, op.replica);
                }
                // Borrow mutably via take/put so we can recurse without
                // overlapping borrows.
                let mut entry = map
                    .entries
                    .remove(&key)
                    .ok_or_else(|| CrdtError::InvalidPath(format!("missing key {key}")))?;
                let res = Self::apply_recursive(&mut entry.0, &tail, op);
                map.entries.insert(key, entry);
                res
            }
            other => Err(CrdtError::TypeMismatch {
                path: op.target_path.as_string(),
                expected: "map",
                actual: other.kind(),
            }),
        }
    }

    fn seed_terminal_node(op: &OpKind, hlc: &HybridLogicalClock, replica: ReplicaId) -> CrdtNode {
        match op {
            OpKind::RegisterSet { value } => CrdtNode::Register(LwwRegister::new(
                CrdtId::from_bytes(b"seeded-register"),
                value.clone(),
                *hlc,
                replica,
            )),
            OpKind::CounterAdd { .. } => {
                CrdtNode::Counter(PnCounter::new(CrdtId::from_bytes(b"seeded-counter")))
            }
            OpKind::SetAdd { .. } | OpKind::SetRemove { .. } => {
                CrdtNode::Set(OrSet::new(CrdtId::from_bytes(b"seeded-set")))
            }
            OpKind::ListInsert { .. } | OpKind::ListDelete { .. } => {
                CrdtNode::List(RgaList::new(CrdtId::from_bytes(b"seeded-list")))
            }
            OpKind::TextInsert { .. } | OpKind::TextDelete { .. } => {
                CrdtNode::Text(RgaList::new(CrdtId::from_bytes(b"seeded-text")))
            }
            OpKind::MapSet { .. } | OpKind::MapDelete { .. } => {
                CrdtNode::Map(LwwMap::new(CrdtId::from_bytes(b"seeded-map")))
            }
        }
    }

    fn apply_terminal(node: &mut CrdtNode, op: &Op) -> Result<()> {
        match (&mut *node, &op.op) {
            (CrdtNode::Register(reg), OpKind::RegisterSet { value }) => {
                reg.write(value.clone(), op.hlc, op.replica);
                Ok(())
            }
            // Replace a non-register leaf if the op is RegisterSet.
            (other, OpKind::RegisterSet { value }) if !matches!(other, CrdtNode::Map(_)) => {
                *node = CrdtNode::Register(LwwRegister::new(
                    CrdtId::from_bytes(b"register-replaced"),
                    value.clone(),
                    op.hlc,
                    op.replica,
                ));
                Ok(())
            }
            (CrdtNode::Counter(c), OpKind::CounterAdd { delta }) => {
                if *delta >= 0 {
                    c.increment(op.replica, *delta as u64);
                } else {
                    c.decrement(op.replica, delta.unsigned_abs());
                }
                Ok(())
            }
            (CrdtNode::Set(s), OpKind::SetAdd { value, tag }) => {
                s.add(value.clone(), *tag);
                Ok(())
            }
            (CrdtNode::Set(s), OpKind::SetRemove { value }) => {
                s.remove(value);
                Ok(())
            }
            (CrdtNode::List(l), OpKind::ListInsert { parent, value, pos }) => {
                let wrapped = CrdtNode::Register(LwwRegister::new(
                    CrdtId::from_bytes(b"list-element"),
                    value.clone(),
                    op.hlc,
                    op.replica,
                ));
                l.push_node(*parent, *pos, wrapped);
                Ok(())
            }
            (CrdtNode::List(l), OpKind::ListDelete { pos }) => {
                l.delete(*pos);
                Ok(())
            }
            (CrdtNode::Text(t), OpKind::TextInsert { parent, ch, pos }) => {
                t.push_node(*parent, *pos, *ch);
                Ok(())
            }
            (CrdtNode::Text(t), OpKind::TextDelete { pos }) => {
                t.delete(*pos);
                Ok(())
            }
            (CrdtNode::Map(map), OpKind::MapSet { key, value }) => {
                let child = CrdtNode::Register(LwwRegister::new(
                    CrdtId::from_bytes(key.as_bytes()),
                    value.clone(),
                    op.hlc,
                    op.replica,
                ));
                map.set(key.clone(), child, op.hlc, op.replica);
                Ok(())
            }
            (CrdtNode::Map(map), OpKind::MapDelete { key }) => {
                map.remove(key, op.hlc, op.replica);
                Ok(())
            }
            (node, kind) => Err(CrdtError::TypeMismatch {
                path: op.target_path.as_string(),
                expected: terminal_expected(kind),
                actual: node.kind(),
            }),
        }
    }

    /// Merge another document into this one (Crdt-style). The merged
    /// document's history is replayed; identical op ids are skipped.
    pub fn merge(&mut self, other: &CrdtDocument) -> Result<()> {
        let mut sorted = other.history.clone();
        sorted.sort_by_key(|a| a.hlc);
        for op in &sorted {
            self.apply_op(op)?;
        }
        Ok(())
    }

    /// True iff the document is structurally a map.
    pub fn root_is_map(&self) -> bool {
        matches!(self.root, CrdtNode::Map(_))
    }

    /// Helper: insert into a list at `path`. The list is auto-created.
    pub fn list_insert(
        &mut self,
        path: &str,
        parent: Option<RgaPos>,
        value: Value,
        clock: &mut HlcClock,
        counter: u32,
    ) -> Result<(Op, RgaPos)> {
        let p = Path::parse(path);
        let hlc = clock.now();
        let pos = RgaPos {
            hlc,
            replica: clock.node(),
            counter,
        };
        let op = Op::new(
            self.id,
            p,
            OpKind::ListInsert {
                parent,
                value,
                pos,
            },
            hlc,
            clock.node(),
        )?;
        self.apply_op_internal(&op)?;
        self.history.push(op.clone());
        self.clock = hlc;
        Ok((op, pos))
    }

    /// Helper: insert into a text node at `path`.
    pub fn text_insert(
        &mut self,
        path: &str,
        parent: Option<RgaPos>,
        ch: char,
        clock: &mut HlcClock,
        counter: u32,
    ) -> Result<(Op, RgaPos)> {
        let p = Path::parse(path);
        let hlc = clock.now();
        let pos = RgaPos {
            hlc,
            replica: clock.node(),
            counter,
        };
        let op = Op::new(
            self.id,
            p,
            OpKind::TextInsert { parent, ch, pos },
            hlc,
            clock.node(),
        )?;
        self.apply_op_internal(&op)?;
        self.history.push(op.clone());
        self.clock = hlc;
        Ok((op, pos))
    }

    /// Helper: increment a counter at `path`. The counter is auto-created.
    pub fn counter_add(
        &mut self,
        path: &str,
        delta: i64,
        clock: &mut HlcClock,
    ) -> Result<Op> {
        let p = Path::parse(path);
        let hlc = clock.now();
        let op = Op::new(
            self.id,
            p,
            OpKind::CounterAdd { delta },
            hlc,
            clock.node(),
        )?;
        self.apply_op_internal(&op)?;
        self.history.push(op.clone());
        self.clock = hlc;
        Ok(op)
    }

    /// Helper: add to an OR-set at `path`. The set is auto-created.
    pub fn set_add(
        &mut self,
        path: &str,
        value: Value,
        clock: &mut HlcClock,
        nonce: u64,
    ) -> Result<Op> {
        let p = Path::parse(path);
        let hlc = clock.now();
        let tag = UniqueTag {
            hlc,
            replica: clock.node(),
            nonce,
        };
        let op = Op::new(
            self.id,
            p,
            OpKind::SetAdd { value, tag },
            hlc,
            clock.node(),
        )?;
        self.apply_op_internal(&op)?;
        self.history.push(op.clone());
        self.clock = hlc;
        Ok(op)
    }
}

fn terminal_expected(kind: &OpKind) -> &'static str {
    match kind {
        OpKind::RegisterSet { .. } => "register",
        OpKind::CounterAdd { .. } => "counter",
        OpKind::SetAdd { .. } | OpKind::SetRemove { .. } => "set",
        OpKind::ListInsert { .. } | OpKind::ListDelete { .. } => "list",
        OpKind::TextInsert { .. } | OpKind::TextDelete { .. } => "text",
        OpKind::MapSet { .. } | OpKind::MapDelete { .. } => "map",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rid(n: u128) -> ReplicaId {
        ReplicaId::new(n)
    }

    #[test]
    fn set_at_creates_intermediate_maps() {
        let id = DocumentId::from_bytes(b"doc");
        let r = rid(1);
        let mut doc = CrdtDocument::new(id, r);
        let mut clock = HlcClock::with_manual_wall(r, 1);
        doc.set_at("/users/alice/balance", 100, &mut clock).unwrap();

        let node = doc.get_node("/users/alice/balance").unwrap();
        match node {
            CrdtNode::Register(reg) => assert_eq!(*reg.get(), Value::Int(100)),
            _ => panic!("expected register"),
        }
    }

    #[test]
    fn apply_op_idempotent() {
        let id = DocumentId::from_bytes(b"doc");
        let r = rid(1);
        let mut doc = CrdtDocument::new(id, r);
        let mut clock = HlcClock::with_manual_wall(r, 1);
        let op = doc.set_at("/x", 1, &mut clock).unwrap();

        let applied_again = doc.apply_op(&op).unwrap();
        assert!(!applied_again);
        assert_eq!(doc.history.len(), 1);
    }
}
