//! TEL: issuance → revocation lifecycle + chain verification.

use serde_json::json;
use smart_byte_acdc::{
    AcdcBuilder, AttributeSection, CredentialRegistry, InMemoryRegistry, RegistryState,
    SchemaSection, Tel,
};

#[test]
fn iss_then_rev_marks_revoked() {
    let mut registry = InMemoryRegistry::open("Bissuer").expect("open");

    let mut s = serde_json::Map::new();
    s.insert("$id".into(), json!("schema-rev"));
    let mut a = serde_json::Map::new();
    a.insert("i".into(), json!("Bholder"));
    a.insert("credit".into(), json!(1000));

    let acdc = AcdcBuilder::new()
        .issuer("Bissuer")
        .registry(registry.registry_said())
        .schema(SchemaSection::Inline(s))
        .attributes(AttributeSection::Inline(a))
        .build()
        .expect("build");

    let id = acdc.d;
    registry.issue(acdc).expect("issue");
    assert_eq!(registry.state(&id), RegistryState::Active);

    registry.revoke(&id).expect("revoke");
    assert_eq!(registry.state(&id), RegistryState::Revoked);

    registry.tel().verify_chain().expect("chain valid");
}

#[test]
fn standalone_tel_chain_holds() {
    let mut tel = Tel::open("Bissuer").expect("open");
    let cred_a = smart_byte_core::Said::hash(b"cred-a");
    let cred_b = smart_byte_core::Said::hash(b"cred-b");
    tel.issue(cred_a).expect("iss a");
    tel.issue(cred_b).expect("iss b");
    tel.revoke(cred_a).expect("rev a");
    tel.verify_chain().expect("chain");
    assert_eq!(tel.events().len(), 4);
}
