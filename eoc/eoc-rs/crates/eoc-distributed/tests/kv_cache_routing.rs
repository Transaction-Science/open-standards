//! KV-cache-aware router keeps a session pinned to its previous worker.

use eoc_distributed::{
    Accelerator, Capability, InMemoryWorker, KvCacheAwareRouter, Load, LocalRequest, Strategy,
    Worker,
};

fn mk(id: &str, micro_j: u32) -> InMemoryWorker {
    InMemoryWorker::new(
        id,
        Capability {
            models: vec!["llama-70b".into()],
            accelerator: Accelerator::Gpu,
            max_concurrency: 16,
            continuous_batching: true,
            paged_kv: true,
            zone: "EU-FR".into(),
        },
        Load {
            micro_joules_per_token: micro_j,
            ..Load::idle()
        },
    )
}

#[test]
fn second_turn_pins_to_same_worker() {
    let mut r = KvCacheAwareRouter::new(Strategy::JouleWeighted);
    let a = mk("a", 200);
    let b = mk("b", 50); // joule-cheaper
    let pool: Vec<&dyn Worker> = vec![&a, &b];

    let first = r
        .pick(
            &pool,
            &LocalRequest {
                session_id: "user-42".into(),
                model: "llama-70b",
                expected_tokens: 64,
            },
        )
        .expect("ok")
        .id()
        .to_string();
    // First turn — fallback picks cheapest.
    assert_eq!(first, "b");

    // Add a brand-new ultra-cheap worker. The locality binding must
    // override the joule-weighted fallback so we still land on "b".
    let c = mk("c", 1);
    let pool2: Vec<&dyn Worker> = vec![&a, &b, &c];
    let second = r
        .pick(
            &pool2,
            &LocalRequest {
                session_id: "user-42".into(),
                model: "llama-70b",
                expected_tokens: 64,
            },
        )
        .expect("ok")
        .id()
        .to_string();
    assert_eq!(second, "b");
}

#[test]
fn forget_clears_binding() {
    let mut r = KvCacheAwareRouter::default();
    let a = mk("a", 100);
    let pool: Vec<&dyn Worker> = vec![&a];
    let _ = r
        .pick(
            &pool,
            &LocalRequest {
                session_id: "s".into(),
                model: "llama-70b",
                expected_tokens: 16,
            },
        )
        .expect("ok");
    assert_eq!(r.len(), 1);
    r.forget("s");
    assert_eq!(r.len(), 0);
}

#[test]
fn stale_binding_falls_back() {
    let mut r = KvCacheAwareRouter::default();
    let a = mk("a", 200);
    let b = mk("b", 50);
    let pool: Vec<&dyn Worker> = vec![&a, &b];
    let first = r
        .pick(
            &pool,
            &LocalRequest {
                session_id: "session-x".into(),
                model: "llama-70b",
                expected_tokens: 32,
            },
        )
        .expect("ok")
        .id()
        .to_string();
    assert_eq!(first, "b");

    // Worker "b" disappears from the pool — fall back to "a".
    let pool2: Vec<&dyn Worker> = vec![&a];
    let second = r
        .pick(
            &pool2,
            &LocalRequest {
                session_id: "session-x".into(),
                model: "llama-70b",
                expected_tokens: 32,
            },
        )
        .expect("ok")
        .id()
        .to_string();
    assert_eq!(second, "a");
}
