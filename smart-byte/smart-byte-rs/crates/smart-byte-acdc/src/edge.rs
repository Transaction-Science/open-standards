//! Edge section + graph traversal.
//!
//! An ACDC's `e` field is a JSON object mapping local labels to *edge
//! records*. Per spec §4 an edge record has the shape:
//!
//! ```json
//! { "n": "<target ACDC SAID>", "s": "<schema SAID>", "o": "I2I" }
//! ```
//!
//! where:
//!
//! * `n` — node SAID (the target credential the edge points to);
//! * `s` — schema SAID expected at the target;
//! * `o` — operator (e.g. `I2I` issuer-to-issuer, `NI2I` not-issuer,
//!   `DI2I` delegated). Defaults to `I2I`.
//!
//! Multiple edges form a directed graph among ACDCs. [`EdgeGraph`]
//! stores credentials by SAID and supports breadth-first traversal +
//! cycle detection.

use std::collections::{HashMap, HashSet, VecDeque};

use serde::{Deserialize, Serialize};
use serde_json::Value;
use smart_byte_core::Said;

use crate::acdc::{Acdc, EdgeSection};
use crate::error::{AcdcError, Result};

/// Edge operator vocabulary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeOp {
    /// Issuer-to-Issuer. Default operator.
    I2I,
    /// Not Issuer-to-Issuer (anyone may issue the target).
    Ni2i,
    /// Delegated Issuer-to-Issuer.
    Di2i,
}

impl EdgeOp {
    /// Parse from the canonical text encoding (`"I2I"`, `"NI2I"`,
    /// `"DI2I"`).
    pub fn from_str(s: &str) -> Result<Self> {
        match s {
            "I2I" => Ok(Self::I2I),
            "NI2I" => Ok(Self::Ni2i),
            "DI2I" => Ok(Self::Di2i),
            other => Err(AcdcError::MalformedField {
                field: "o",
                detail: format!("unknown edge operator {other}"),
            }),
        }
    }
}

/// A parsed edge record.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edge {
    /// Local edge label (key in the `e` object).
    pub label: String,
    /// Target ACDC SAID (`n`).
    pub target: Said,
    /// Expected schema SAID at the target (`s`). `None` if the edge
    /// does not constrain the target schema.
    pub schema: Option<Said>,
    /// Operator. Defaults to [`EdgeOp::I2I`].
    pub operator: EdgeOp,
}

impl Edge {
    /// Parse all edges out of an [`EdgeSection`]. Errors if any edge
    /// record is malformed.
    pub fn parse_section(section: &EdgeSection) -> Result<Vec<Self>> {
        let mut out = Vec::with_capacity(section.0.len());
        for (label, v) in &section.0 {
            let obj = v.as_object().ok_or_else(|| AcdcError::MalformedField {
                field: "e",
                detail: format!("edge `{label}` is not an object"),
            })?;
            let target_s = obj
                .get("n")
                .and_then(|v| v.as_str())
                .ok_or(AcdcError::MissingField("e.*.n"))?;
            let target = Said::from_base32(target_s).map_err(|e| AcdcError::MalformedField {
                field: "e.*.n",
                detail: e.to_string(),
            })?;
            let schema = match obj.get("s").and_then(|v| v.as_str()) {
                Some(s) => Some(Said::from_base32(s).map_err(|e| AcdcError::MalformedField {
                    field: "e.*.s",
                    detail: e.to_string(),
                })?),
                None => None,
            };
            let operator = match obj.get("o").and_then(|v| v.as_str()) {
                Some(s) => EdgeOp::from_str(s)?,
                None => EdgeOp::I2I,
            };
            out.push(Self {
                label: label.clone(),
                target,
                schema,
                operator,
            });
        }
        Ok(out)
    }

    /// Build a JSON record suitable for embedding into an
    /// [`EdgeSection`].
    pub fn to_json(&self) -> Value {
        let mut m = serde_json::Map::new();
        m.insert("n".into(), Value::String(self.target.to_base32()));
        if let Some(s) = self.schema {
            m.insert("s".into(), Value::String(s.to_base32()));
        }
        let op = match self.operator {
            EdgeOp::I2I => "I2I",
            EdgeOp::Ni2i => "NI2I",
            EdgeOp::Di2i => "DI2I",
        };
        m.insert("o".into(), Value::String(op.into()));
        Value::Object(m)
    }
}

/// In-memory directed graph of ACDCs keyed by SAID.
#[derive(Clone, Debug, Default)]
pub struct EdgeGraph {
    nodes: HashMap<Said, Acdc>,
}

impl EdgeGraph {
    /// Empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Insert (or replace) an ACDC node.
    pub fn insert(&mut self, acdc: Acdc) -> Result<()> {
        acdc.verify_said()?;
        self.nodes.insert(acdc.d, acdc);
        Ok(())
    }

    /// Number of nodes.
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Is empty.
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Look up an ACDC by SAID.
    pub fn get(&self, said: &Said) -> Option<&Acdc> {
        self.nodes.get(said)
    }

    /// Parsed outgoing edges of an ACDC.
    pub fn edges_of(&self, said: &Said) -> Result<Vec<Edge>> {
        let node = self
            .nodes
            .get(said)
            .ok_or_else(|| AcdcError::EdgeMissing(*said))?;
        Edge::parse_section(&node.e)
    }

    /// Breadth-first traversal beginning at `start`. Visits each
    /// reachable node at most once; errors if an edge points to a SAID
    /// not present in the graph or if a cycle is detected.
    pub fn traverse(&self, start: &Said) -> Result<Vec<Said>> {
        if !self.nodes.contains_key(start) {
            return Err(AcdcError::EdgeMissing(*start));
        }
        let mut order = Vec::new();
        let mut seen: HashSet<Said> = HashSet::new();
        let mut on_path: HashSet<Said> = HashSet::new();
        let mut queue: VecDeque<Said> = VecDeque::new();
        queue.push_back(*start);
        seen.insert(*start);

        while let Some(node_said) = queue.pop_front() {
            if on_path.contains(&node_said) {
                return Err(AcdcError::EdgeCycle(node_said));
            }
            on_path.insert(node_said);
            order.push(node_said);

            let edges = self.edges_of(&node_said)?;
            for edge in edges {
                if edge.target == node_said {
                    return Err(AcdcError::EdgeCycle(node_said));
                }
                if !self.nodes.contains_key(&edge.target) {
                    return Err(AcdcError::EdgeMissing(edge.target));
                }
                if seen.contains(&edge.target) {
                    // Already visited or queued; if it's currently on
                    // the active path that's a cycle.
                    if on_path.contains(&edge.target) {
                        return Err(AcdcError::EdgeCycle(edge.target));
                    }
                    continue;
                }
                seen.insert(edge.target);
                queue.push_back(edge.target);
            }
        }
        Ok(order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acdc::{AcdcBuilder, AttributeSection, SchemaSection};
    use serde_json::json;

    fn mk_acdc(name: &str, edges: Vec<Edge>) -> Acdc {
        let mut s = serde_json::Map::new();
        s.insert("$id".into(), json!(name));
        let mut a = serde_json::Map::new();
        a.insert("name".into(), json!(name));
        let mut edge_section = EdgeSection::default();
        for e in edges {
            edge_section.0.insert(e.label.clone(), e.to_json());
        }
        AcdcBuilder::new()
            .issuer("Bissuer")
            .schema(SchemaSection::Inline(s))
            .attributes(AttributeSection::Inline(a))
            .edges(edge_section)
            .build()
            .expect("build")
    }

    #[test]
    fn parses_and_traverses_chain() {
        let leaf = mk_acdc("leaf", vec![]);
        let mid = mk_acdc(
            "mid",
            vec![Edge {
                label: "down".into(),
                target: leaf.d,
                schema: None,
                operator: EdgeOp::I2I,
            }],
        );
        let root = mk_acdc(
            "root",
            vec![Edge {
                label: "child".into(),
                target: mid.d,
                schema: None,
                operator: EdgeOp::Ni2i,
            }],
        );
        let mut g = EdgeGraph::new();
        let root_id = root.d;
        g.insert(leaf).expect("ins");
        g.insert(mid).expect("ins");
        g.insert(root).expect("ins");
        let order = g.traverse(&root_id).expect("traverse");
        assert_eq!(order.len(), 3);
    }

    #[test]
    fn missing_edge_target_detected() {
        let leaf = mk_acdc(
            "orphan",
            vec![Edge {
                label: "ghost".into(),
                target: Said([0xAB; 32]),
                schema: None,
                operator: EdgeOp::I2I,
            }],
        );
        let leaf_id = leaf.d;
        let mut g = EdgeGraph::new();
        g.insert(leaf).expect("ins");
        assert!(matches!(
            g.traverse(&leaf_id),
            Err(AcdcError::EdgeMissing(_))
        ));
    }
}
