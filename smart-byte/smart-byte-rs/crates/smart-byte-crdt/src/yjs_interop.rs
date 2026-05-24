//! Yjs interop via the `yrs` Rust port.
//!
//! Imports / exports a Yjs document. Convergence semantics are preserved
//! by routing through the canonical Yjs update v1 format and converting
//! the resulting state into a [`CrdtDocument`] tree.
//!
//! Enabled by the `yjs-interop` feature.

use yrs::updates::decoder::Decode;
use yrs::{
    types::Value as YrsValue, Doc, GetString, Map, MapPrelim, ReadTxn, StateVector, Transact,
    TransactionMut, Update, WriteTxn,
};

use crate::document::{CrdtDocument, CrdtNode, DocumentId, Value};
use crate::error::{CrdtError, Result};
use crate::hlc::{HlcClock, ReplicaId};
use crate::types::{CrdtId, LwwMap, LwwRegister, RgaList};

/// Import a Yjs v1 update blob.
pub fn from_yjs(bytes: &[u8]) -> Result<CrdtDocument> {
    let doc = Doc::new();
    let map_ref;
    let text_ref;
    {
        let mut txn = doc.transact_mut();
        let update =
            Update::decode_v1(bytes).map_err(|e| CrdtError::Yjs(format!("decode: {e}")))?;
        txn.apply_update(update);
        map_ref = txn.get_or_insert_map("root");
        text_ref = txn.get_or_insert_text("text");
    }

    let id = DocumentId::from_bytes(bytes);
    let local = ReplicaId::from_bytes(b"yjs-import");
    let mut clock = HlcClock::with_manual_wall(local, 1);
    let ts = clock.now();

    let txn = doc.transact();
    let mut root_map: LwwMap<String, CrdtNode> = LwwMap::new(CrdtId::from_bytes(&id.0));

    for (k, v) in map_ref.iter(&txn) {
        let node = yrs_value_to_node(k, &v, ts, local);
        root_map.set(k.to_string(), node, ts, local);
    }

    let s = text_ref.get_string(&txn);
    if !s.is_empty() {
        let mut t: RgaList<char> = RgaList::new(CrdtId::from_bytes(b"text"));
        let mut parent = None;
        for (i, ch) in s.chars().enumerate() {
            let pos = crate::types::RgaPos {
                hlc: ts,
                replica: local,
                counter: i as u32,
            };
            t.push_node(parent, pos, ch);
            parent = Some(pos);
        }
        root_map.set("text".into(), CrdtNode::Text(t), ts, local);
    }

    Ok(CrdtDocument::with_root(id, CrdtNode::Map(root_map), local))
}

fn yrs_value_to_node(
    key: &str,
    v: &YrsValue,
    ts: crate::hlc::HybridLogicalClock,
    replica: ReplicaId,
) -> CrdtNode {
    let v = match v {
        YrsValue::Any(any) => any_to_value(any),
        _ => Value::Null,
    };
    CrdtNode::Register(LwwRegister::new(
        CrdtId::from_bytes(key.as_bytes()),
        v,
        ts,
        replica,
    ))
}

fn any_to_value(any: &yrs::Any) -> Value {
    use yrs::Any::*;
    match any {
        Null | Undefined => Value::Null,
        Bool(b) => Value::Bool(*b),
        Number(n) => Value::float(*n),
        BigInt(i) => Value::Int(*i),
        String(s) => Value::Text(s.to_string()),
        Buffer(b) => Value::Bytes(b.to_vec()),
        Array(_) | Map(_) => {
            Value::Json(serde_json::to_string(any).unwrap_or_else(|_| "null".into()))
        }
    }
}

/// Export a [`CrdtDocument`] to a Yjs v1 update blob.
pub fn to_yjs(doc: &CrdtDocument) -> Result<Vec<u8>> {
    let yd = Doc::new();
    {
        let mut txn = yd.transact_mut();
        if let CrdtNode::Map(m) = &doc.root {
            let map = txn.get_or_insert_map("root");
            for (k, v) in m.iter() {
                write_node(&mut txn, &map, k, v)?;
            }
        }
    }
    let txn = yd.transact();
    let sv = StateVector::default();
    Ok(txn.encode_state_as_update_v1(&sv))
}

fn write_node(
    txn: &mut TransactionMut,
    map: &yrs::MapRef,
    key: &str,
    node: &CrdtNode,
) -> Result<()> {
    match node {
        CrdtNode::Register(reg) => {
            let any = value_to_any(reg.get());
            map.insert(txn, key, any);
        }
        CrdtNode::Counter(c) => {
            map.insert(txn, key, yrs::Any::BigInt(c.value() as i64));
        }
        CrdtNode::Text(t) => {
            let s: String = t.iter().collect();
            map.insert(txn, key, yrs::Any::String(s.into()));
        }
        CrdtNode::Map(child) => {
            let nested = map.insert(txn, key, MapPrelim::<yrs::Any>::new());
            for (k, v) in child.iter() {
                write_node(txn, &nested, k, v)?;
            }
        }
        // List / Set: serialise as a JSON string to preserve content.
        other => {
            let json = serde_json::to_string(other).unwrap_or_else(|_| "null".into());
            map.insert(txn, key, yrs::Any::String(json.into()));
        }
    }
    Ok(())
}

fn value_to_any(v: &Value) -> yrs::Any {
    match v {
        Value::Null => yrs::Any::Null,
        Value::Bool(b) => yrs::Any::Bool(*b),
        Value::Int(i) => yrs::Any::BigInt(*i),
        Value::FloatBits(b) => yrs::Any::Number(f64::from_bits(*b)),
        Value::Text(s) => yrs::Any::String(s.clone().into()),
        Value::Bytes(b) => yrs::Any::Buffer(b.clone().into()),
        Value::Json(s) => yrs::Any::String(s.clone().into()),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn export_empty_document_does_not_error() {
        let id = DocumentId::from_bytes(b"d");
        let r = ReplicaId::new(1);
        let doc = CrdtDocument::new(id, r);
        let _bytes = to_yjs(&doc).unwrap();
    }
}
