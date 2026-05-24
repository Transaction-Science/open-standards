//! Acquirer-side idempotency: an `IdempotencyKey` replayed against a
//! lightweight in-test cache returns the previous outcome without
//! making a second network call.
//!
//! Real production wiring uses `op-orchestrator`'s `IdempotencyStore`;
//! this test demonstrates the same semantics at the acquirer boundary.

use std::collections::BTreeMap;
use std::collections::HashMap;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use op_bnpl::{
    AffirmAcquirer, AuthorizedCheckout, BillingInfo, BnplAcquirer, BnplIntent, ConsumerInfo,
    IdempotencyKey, LineItem, RedirectUrls, ShippingInfo,
};
use op_core::{Currency, Money};
use reqwest::Client;
use serde_json::json;
use wiremock::matchers::{method, path};
use wiremock::{Mock, MockServer, Request, ResponseTemplate};

fn intent_with_key(key: &str) -> BnplIntent {
    BnplIntent {
        amount: Money::from_minor(10_000, Currency::USD),
        currency: Currency::USD,
        line_items: vec![LineItem {
            name: "Thing".into(),
            sku: None,
            quantity: 1,
            unit_price: Money::from_minor(10_000, Currency::USD),
            total_amount: Money::from_minor(10_000, Currency::USD),
        }],
        shipping: ShippingInfo {
            name: "A".into(),
            line1: "1".into(),
            line2: None,
            city: "c".into(),
            region: "r".into(),
            postal_code: "p".into(),
            country: "US".into(),
        },
        billing: BillingInfo {
            name: "A".into(),
            line1: "1".into(),
            line2: None,
            city: "c".into(),
            region: "r".into(),
            postal_code: "p".into(),
            country: "US".into(),
        },
        consumer: ConsumerInfo {
            email: "a@b.com".into(),
            phone: None,
            given_name: None,
            family_name: None,
            date_of_birth: None,
        },
        idempotency_key: IdempotencyKey::from(key),
        redirect_urls: RedirectUrls {
            success: "s".into(),
            cancel: "c".into(),
            failure: None,
        },
        metadata: BTreeMap::new(),
    }
}

/// A tiny in-test idempotency cache. Production code uses
/// `op-orchestrator::IdempotencyStore`; we replicate the semantics
/// here to assert no double-network-call.
#[derive(Default)]
struct AuthCache {
    map: Mutex<HashMap<String, AuthorizedCheckout>>,
}

impl AuthCache {
    async fn authorize_idempotent(
        &self,
        acquirer: &AffirmAcquirer,
        intent: &BnplIntent,
        consumer_token: &str,
    ) -> AuthorizedCheckout {
        if let Some(cached) = self.map.lock().unwrap().get(intent.idempotency_key.as_str()) {
            return cached.clone();
        }
        let session = acquirer.initiate(intent).await.unwrap();
        let auth = acquirer.authorize(&session, consumer_token).await.unwrap();
        self.map
            .lock()
            .unwrap()
            .insert(intent.idempotency_key.as_str().to_owned(), auth.clone());
        auth
    }
}

#[tokio::test]
async fn replayed_idempotency_key_returns_cached_outcome_without_network() {
    let server = MockServer::start().await;
    let counter: Arc<AtomicUsize> = Arc::new(AtomicUsize::new(0));
    let c2 = counter.clone();

    Mock::given(method("POST"))
        .and(path("/api/v2/charges"))
        .respond_with(move |_: &Request| {
            c2.fetch_add(1, Ordering::SeqCst);
            ResponseTemplate::new(200).set_body_json(json!({
                "id": "CHG_IDEM_1",
                "amount": 10_000,
                "status": "authorized"
            }))
        })
        .mount(&server)
        .await;

    let acquirer = AffirmAcquirer::new(Client::new(), "pub", "priv", server.uri());
    let cache = AuthCache::default();
    let intent = intent_with_key("idem-replay-1");

    let a1 = cache
        .authorize_idempotent(&acquirer, &intent, "tok-1")
        .await;
    let a2 = cache
        .authorize_idempotent(&acquirer, &intent, "tok-1")
        .await;

    assert_eq!(a1.provider_ref, a2.provider_ref);
    // Network call fired exactly once even though authorize_idempotent
    // was invoked twice.
    assert_eq!(counter.load(Ordering::SeqCst), 1);
}
