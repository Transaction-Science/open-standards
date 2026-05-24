//! Edge section: link multiple ACDCs into a DAG and traverse.

use serde_json::json;
use smart_byte_acdc::{
    Acdc, AcdcBuilder, AttributeSection, Edge, EdgeGraph, EdgeOp, SchemaSection,
    acdc::EdgeSection,
};
use smart_byte_core::Said;

fn mk(name: &str, edges: Vec<Edge>) -> Acdc {
    let mut s = serde_json::Map::new();
    s.insert("$id".into(), json!(name));
    let mut a = serde_json::Map::new();
    a.insert("name".into(), json!(name));
    let mut section = EdgeSection::default();
    for e in edges {
        section.0.insert(e.label.clone(), e.to_json());
    }
    AcdcBuilder::new()
        .issuer("Bissuer")
        .schema(SchemaSection::Inline(s))
        .attributes(AttributeSection::Inline(a))
        .edges(section)
        .build()
        .expect("build")
}

#[test]
fn three_node_chain_traverses_in_order() {
    let leaf = mk("leaf", vec![]);
    let mid = mk(
        "mid",
        vec![Edge {
            label: "down".into(),
            target: leaf.d,
            schema: None,
            operator: EdgeOp::I2I,
        }],
    );
    let root = mk(
        "root",
        vec![Edge {
            label: "child".into(),
            target: mid.d,
            schema: None,
            operator: EdgeOp::Ni2i,
        }],
    );
    let root_id = root.d;

    let mut g = EdgeGraph::new();
    g.insert(leaf.clone()).expect("ins");
    g.insert(mid.clone()).expect("ins");
    g.insert(root).expect("ins");

    let order = g.traverse(&root_id).expect("traverse");
    assert_eq!(order.len(), 3);
    assert_eq!(order[0], root_id);
    assert_eq!(order[1], mid.d);
    assert_eq!(order[2], leaf.d);
}

#[test]
fn missing_target_caught() {
    let lonely = mk(
        "lonely",
        vec![Edge {
            label: "phantom".into(),
            target: Said([0xCDu8; 32]),
            schema: None,
            operator: EdgeOp::I2I,
        }],
    );
    let id = lonely.d;
    let mut g = EdgeGraph::new();
    g.insert(lonely).expect("ins");
    assert!(g.traverse(&id).is_err());
}

#[test]
fn edge_records_roundtrip() {
    let leaf = mk("leaf", vec![]);
    let parent = mk(
        "parent",
        vec![Edge {
            label: "issued-by".into(),
            target: leaf.d,
            schema: Some(Said::hash(b"some-schema")),
            operator: EdgeOp::Di2i,
        }],
    );
    let edges = Edge::parse_section(&parent.e).expect("parse");
    assert_eq!(edges.len(), 1);
    assert_eq!(edges[0].operator, EdgeOp::Di2i);
    assert!(edges[0].schema.is_some());
}
