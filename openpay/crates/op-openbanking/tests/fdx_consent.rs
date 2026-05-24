//! FDX v6 consent + resource-endpoint integration test.

use op_openbanking::fdx::{FdxConsent, FdxResource, FdxService, FdxVersion};

#[test]
fn v6_endpoint_routing() {
    let svc = FdxService {
        provider_base_url: "https://api.fi.example".into(),
        version: FdxVersion::V6,
    };
    assert_eq!(
        svc.endpoint(FdxResource::Accounts),
        "https://api.fi.example/fdx/v6/accounts"
    );
    assert_eq!(
        svc.endpoint(FdxResource::TaxForms),
        "https://api.fi.example/fdx/v6/tax/forms"
    );
    assert_eq!(
        svc.endpoint(FdxResource::RecurringPayments),
        "https://api.fi.example/fdx/v6/payments/recurring"
    );
}

#[test]
fn consent_serde_round_trips() {
    let c = FdxConsent {
        id: "fdx-consent-1".into(),
        resources: vec![FdxResource::Accounts, FdxResource::Transactions],
        account_ids: vec!["acc-a".into(), "acc-b".into()],
        expires_at: Some(time::OffsetDateTime::UNIX_EPOCH),
    };
    let json = serde_json::to_string(&c).expect("ser");
    let back: FdxConsent = serde_json::from_str(&json).expect("de");
    assert_eq!(c, back);
}

#[test]
fn indefinite_consent_supports_none_expiry() {
    let c = FdxConsent {
        id: "fdx-indef".into(),
        resources: vec![FdxResource::Customers],
        account_ids: vec![],
        expires_at: None,
    };
    let json = serde_json::to_string(&c).expect("ser");
    assert!(json.contains("null"));
}

#[test]
fn v5_still_supported_for_legacy_holders() {
    let svc = FdxService {
        provider_base_url: "https://api.legacy.example".into(),
        version: FdxVersion::V5,
    };
    assert!(
        svc.endpoint(FdxResource::Investments)
            .starts_with("https://api.legacy.example/fdx/v5/")
    );
}
