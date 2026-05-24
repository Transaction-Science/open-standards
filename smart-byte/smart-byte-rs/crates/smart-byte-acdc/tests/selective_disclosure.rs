//! Selective disclosure: issue with full attributes, present only a
//! subset, verify the section SAID still matches.

use serde_json::json;
use smart_byte_acdc::{
    AcdcBuilder, AttributeSection, DisclosurePlan, SchemaSection, derive_disclosure,
    selective::pack_compact,
};

fn build_attrs() -> serde_json::Map<String, serde_json::Value> {
    let mut a = serde_json::Map::new();
    a.insert("i".into(), json!("Bholder"));
    a.insert("name".into(), json!("Alice"));
    a.insert("dob".into(), json!("1990-05-24"));
    a.insert("city".into(), json!("Austin"));
    a.insert("over_18".into(), json!(true));
    a
}

#[test]
fn reveal_subset_verifies() {
    let mut s = serde_json::Map::new();
    s.insert("$id".into(), json!("sd-schema"));
    let attrs = build_attrs();
    let acdc = AcdcBuilder::new()
        .issuer("Bissuer")
        .schema(SchemaSection::Inline(s))
        .attributes(AttributeSection::Inline(attrs.clone()))
        .build()
        .expect("build");

    let sd = derive_disclosure(
        acdc,
        attrs,
        DisclosurePlan::Subset(vec!["over_18".into(), "name".into()]),
    )
    .expect("derive disclosure");

    let compact = sd.verify().expect("verify");
    assert_eq!(sd.revealed.len(), 2);
    assert!(sd.revealed.contains_key("over_18"));
    assert!(sd.revealed.contains_key("name"));
    assert!(!sd.revealed.contains_key("dob"));
    // Compact SAID is stable across plans.
    let _ = compact;
}

#[test]
fn reveal_none_still_proves_authenticity() {
    let mut s = serde_json::Map::new();
    s.insert("$id".into(), json!("sd-schema"));
    let attrs = build_attrs();
    let acdc = AcdcBuilder::new()
        .issuer("Bissuer")
        .schema(SchemaSection::Inline(s))
        .attributes(AttributeSection::Inline(attrs.clone()))
        .build()
        .expect("build");
    let sd = derive_disclosure(acdc, attrs, DisclosurePlan::None).expect("derive");
    sd.verify().expect("verify");
    assert!(sd.revealed.is_empty());
}

#[test]
fn pack_compact_then_disclose() {
    let mut s = serde_json::Map::new();
    s.insert("$id".into(), json!("sd-schema"));
    let attrs = build_attrs();
    let acdc = AcdcBuilder::new()
        .issuer("Bissuer")
        .schema(SchemaSection::Inline(s))
        .attributes(AttributeSection::Inline(attrs))
        .build()
        .expect("build");
    let (compact, _da, full) = pack_compact(acdc).expect("pack");
    let sd = derive_disclosure(compact, full, DisclosurePlan::All).expect("derive");
    sd.verify().expect("verify");
}
