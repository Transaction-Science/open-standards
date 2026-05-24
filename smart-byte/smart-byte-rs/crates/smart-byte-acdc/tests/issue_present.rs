//! End-to-end: issue an ACDC via a registry and present it back via
//! IPEX (`apply` → `grant` → `admit`).

use serde_json::json;
use smart_byte_acdc::{
    AcdcBuilder, AttributeSection, CredentialRegistry, InMemoryRegistry, IpexMessage,
    RegistryState, SchemaSection,
    ipex::verify_exchange,
};

#[test]
fn issue_then_present() {
    let mut registry = InMemoryRegistry::open("Bissuer").expect("open registry");

    let mut schema_body = serde_json::Map::new();
    schema_body.insert("$id".into(), json!("present-schema"));
    schema_body.insert("title".into(), json!("Present"));
    let schema = SchemaSection::Inline(schema_body);

    let mut attrs = serde_json::Map::new();
    attrs.insert("i".into(), json!("Bholder"));
    attrs.insert("name".into(), json!("Alice"));

    let acdc = AcdcBuilder::new()
        .issuer("Bissuer")
        .registry(registry.registry_said())
        .schema(schema)
        .attributes(AttributeSection::Inline(attrs))
        .build()
        .expect("build acdc");

    let cred_said = acdc.d;
    registry.issue(acdc.clone()).expect("issue");
    assert_eq!(registry.state(&cred_said), RegistryState::Active);

    // Holder → issuer: apply for a credential of this schema.
    let schema_said = match &acdc.s {
        SchemaSection::Inline(_) => smart_byte_core::Said::hash(b"present-schema"),
        SchemaSection::Reference(s) => *s,
    };
    let apply = IpexMessage::apply("Bholder", "Bissuer", schema_said).expect("apply");

    // Issuer → holder: grant.
    let grant = IpexMessage::grant("Bissuer", "Bholder", apply.d, acdc).expect("grant");

    // Holder → issuer: admit.
    let admit = IpexMessage::admit("Bholder", "Bissuer", grant.d).expect("admit");

    verify_exchange(&[apply, grant.clone(), admit]).expect("exchange verifies");

    // Verifier (some third party) takes the granted ACDC and asks the
    // registry whether it is still active.
    let presented = grant.e.expect("grant carries acdc");
    presented.verify_said().expect("acdc said valid");
    assert_eq!(registry.state(&presented.d), RegistryState::Active);
}
