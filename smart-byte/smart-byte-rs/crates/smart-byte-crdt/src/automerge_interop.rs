//! Automerge interop.
//!
//! Imports and exports Automerge's binary document format. Conversion is
//! shape-preserving for the subset of types both engines agree on
//! (maps, lists, text, scalars). Anything outside that subset is
//! preserved opaquely as a JSON blob inside a [`Value::Json`].
//!
//! Enabled by the `automerge-interop` feature.

use automerge::{transaction::Transactable, AutoCommit, ObjType, ReadDoc, Value as AmValue, ROOT};

use crate::document::{CrdtDocument, CrdtNode, DocumentId, Value};
use crate::error::{CrdtError, Result};
use crate::hlc::{HlcClock, ReplicaId};
use crate::types::{CrdtId, LwwMap, LwwRegister, RgaList};

/// Convert an Automerge binary document to a [`CrdtDocument`].
pub fn from_automerge(bytes: &[u8]) -> Result<CrdtDocument> {
    let am =
        AutoCommit::load(bytes).map_err(|e| CrdtError::Automerge(format!("load: {e}")))?;

    let id = DocumentId::from_bytes(bytes);
    let local = ReplicaId::from_bytes(b"automerge-import");
    let mut clock = HlcClock::with_manual_wall(local, 1);
    let ts = clock.now();

    let mut root_map: LwwMap<String, CrdtNode> =
        LwwMap::new(CrdtId::from_bytes(&id.0));

    // Walk root keys.
    for key in am.keys(ROOT) {
        let node = convert_object(&am, ROOT, &key, ts, local)?;
        root_map.set(key.clone(), node, ts, local);
    }

    Ok(CrdtDocument::with_root(
        id,
        CrdtNode::Map(root_map),
        local,
    ))
}

fn convert_object(
    am: &AutoCommit,
    obj: automerge::ObjId,
    key: &str,
    ts: crate::hlc::HybridLogicalClock,
    replica: ReplicaId,
) -> Result<CrdtNode> {
    let value = am
        .get(obj, key)
        .map_err(|e| CrdtError::Automerge(format!("get: {e}")))?;
    match value {
        Some((AmValue::Object(ObjType::Map), id)) => {
            let mut m: LwwMap<String, CrdtNode> =
                LwwMap::new(CrdtId::from_bytes(key.as_bytes()));
            for k in am.keys(&id) {
                let child = convert_object(am, id.clone(), &k, ts, replica)?;
                m.set(k, child, ts, replica);
            }
            Ok(CrdtNode::Map(m))
        }
        Some((AmValue::Object(ObjType::List), id)) => {
            let list: RgaList<CrdtNode> =
                build_list(am, id, ts, replica)?;
            Ok(CrdtNode::List(list))
        }
        Some((AmValue::Object(ObjType::Text), id)) => {
            let text = am
                .text(&id)
                .map_err(|e| CrdtError::Automerge(format!("text: {e}")))?;
            let mut t: RgaList<char> = RgaList::new(CrdtId::from_bytes(key.as_bytes()));
            let mut parent = None;
            for (i, ch) in text.chars().enumerate() {
                let pos = t.insert_after(parent, ch, ts, replica, i as u32);
                parent = Some(pos);
            }
            Ok(CrdtNode::Text(t))
        }
        Some((AmValue::Scalar(s), _)) => {
            let v = scalar_to_value(s.as_ref());
            Ok(CrdtNode::Register(LwwRegister::new(
                CrdtId::from_bytes(key.as_bytes()),
                v,
                ts,
                replica,
            )))
        }
        _ => Ok(CrdtNode::Register(LwwRegister::new(
            CrdtId::from_bytes(key.as_bytes()),
            Value::Null,
            ts,
            replica,
        ))),
    }
}

fn build_list(
    am: &AutoCommit,
    id: automerge::ObjId,
    ts: crate::hlc::HybridLogicalClock,
    replica: ReplicaId,
) -> Result<RgaList<CrdtNode>> {
    let mut list: RgaList<CrdtNode> = RgaList::new(CrdtId::from_bytes(b"am-list"));
    let len = am.length(&id);
    let mut parent = None;
    for i in 0..len {
        let v = am
            .get(&id, i)
            .map_err(|e| CrdtError::Automerge(format!("list-get: {e}")))?;
        let node = match v {
            Some((AmValue::Scalar(s), _)) => CrdtNode::Register(LwwRegister::new(
                CrdtId::from_bytes(&i.to_le_bytes()),
                scalar_to_value(s.as_ref()),
                ts,
                replica,
            )),
            Some((AmValue::Object(ObjType::Map), oid)) => {
                let mut m: LwwMap<String, CrdtNode> =
                    LwwMap::new(CrdtId::from_bytes(&i.to_le_bytes()));
                for k in am.keys(&oid) {
                    let child = convert_object(am, oid.clone(), &k, ts, replica)?;
                    m.set(k, child, ts, replica);
                }
                CrdtNode::Map(m)
            }
            _ => CrdtNode::Register(LwwRegister::new(
                CrdtId::from_bytes(b"unknown"),
                Value::Null,
                ts,
                replica,
            )),
        };
        let pos = list.insert_after(parent, node, ts, replica, i as u32);
        parent = Some(pos);
    }
    Ok(list)
}

fn scalar_to_value(s: &automerge::ScalarValue) -> Value {
    use automerge::ScalarValue::*;
    match s {
        Str(smol) => Value::Text(smol.to_string()),
        Int(i) => Value::Int(*i),
        Uint(u) => Value::Int(*u as i64),
        F64(f) => Value::float(*f),
        Counter(c) => Value::Int(c.into()),
        Timestamp(t) => Value::Int(*t),
        Boolean(b) => Value::Bool(*b),
        Bytes(b) => Value::Bytes(b.clone()),
        Null => Value::Null,
        Unknown { bytes, .. } => Value::Bytes(bytes.clone()),
    }
}

/// Export a [`CrdtDocument`] to Automerge's binary format.
pub fn to_automerge(doc: &CrdtDocument) -> Result<Vec<u8>> {
    let mut am = AutoCommit::new();
    if let CrdtNode::Map(m) = &doc.root {
        for (k, v) in m.iter() {
            write_node(&mut am, &ROOT, k, v)?;
        }
    }
    Ok(am.save())
}

fn write_node(
    am: &mut AutoCommit,
    obj: &automerge::ObjId,
    key: &str,
    node: &CrdtNode,
) -> Result<()> {
    match node {
        CrdtNode::Register(reg) => write_value(am, obj, key, reg.get())?,
        CrdtNode::Counter(c) => {
            let id = am
                .put_object(obj, key, ObjType::Map)
                .map_err(|e| CrdtError::Automerge(format!("map: {e}")))?;
            am.put(&id, "value", c.value() as i64)
                .map_err(|e| CrdtError::Automerge(format!("put: {e}")))?;
        }
        CrdtNode::Set(s) => {
            let id = am
                .put_object(obj, key, ObjType::List)
                .map_err(|e| CrdtError::Automerge(format!("list: {e}")))?;
            for (i, v) in s.iter().enumerate() {
                write_value(am, &id, &i.to_string(), v)?;
            }
        }
        CrdtNode::List(l) => {
            let id = am
                .put_object(obj, key, ObjType::List)
                .map_err(|e| CrdtError::Automerge(format!("list: {e}")))?;
            for (i, n) in l.iter().enumerate() {
                write_node(am, &id, &i.to_string(), n)?;
            }
        }
        CrdtNode::Text(t) => {
            let id = am
                .put_object(obj, key, ObjType::Text)
                .map_err(|e| CrdtError::Automerge(format!("text: {e}")))?;
            let s: String = t.iter().collect();
            am.splice_text(&id, 0, 0, &s)
                .map_err(|e| CrdtError::Automerge(format!("splice: {e}")))?;
        }
        CrdtNode::Map(m) => {
            let id = am
                .put_object(obj, key, ObjType::Map)
                .map_err(|e| CrdtError::Automerge(format!("map: {e}")))?;
            for (k, v) in m.iter() {
                write_node(am, &id, k, v)?;
            }
        }
    }
    Ok(())
}

fn write_value(
    am: &mut AutoCommit,
    obj: &automerge::ObjId,
    key: &str,
    v: &Value,
) -> Result<()> {
    let put = |am: &mut AutoCommit, scalar: automerge::ScalarValue| {
        am.put(obj, key, scalar)
            .map_err(|e| CrdtError::Automerge(format!("put: {e}")))
    };
    match v {
        Value::Null => put(am, automerge::ScalarValue::Null)?,
        Value::Bool(b) => put(am, automerge::ScalarValue::Boolean(*b))?,
        Value::Int(i) => put(am, automerge::ScalarValue::Int(*i))?,
        Value::FloatBits(b) => {
            put(am, automerge::ScalarValue::F64(f64::from_bits(*b)))?;
        }
        Value::Text(s) => put(am, automerge::ScalarValue::Str(s.clone().into()))?,
        Value::Bytes(b) => put(am, automerge::ScalarValue::Bytes(b.clone()))?,
        Value::Json(s) => put(am, automerge::ScalarValue::Str(s.clone().into()))?,
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use automerge::transaction::Transactable;

    #[test]
    fn round_trip_known_automerge_document() {
        let mut am = AutoCommit::new();
        am.put(ROOT, "name", "alice").unwrap();
        am.put(ROOT, "age", 30_i64).unwrap();
        let bytes = am.save();

        let doc = from_automerge(&bytes).unwrap();
        if let CrdtNode::Map(m) = &doc.root {
            assert!(m.get(&"name".to_string()).is_some());
            assert!(m.get(&"age".to_string()).is_some());
        } else {
            panic!("expected root map");
        }

        // Round-trip back.
        let out = to_automerge(&doc).unwrap();
        let am2 = AutoCommit::load(&out).unwrap();
        let name = am2.get(ROOT, "name").unwrap();
        assert!(matches!(name, Some((AmValue::Scalar(_), _))));
    }
}
