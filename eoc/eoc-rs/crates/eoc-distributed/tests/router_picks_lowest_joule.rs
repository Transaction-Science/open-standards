//! Joule-weighted router picks the lowest projected micro-J × tokens.

use eoc_distributed::{
    Accelerator, Capability, InMemoryWorker, Load, Request, Router, Strategy, Worker,
};

fn mk(id: &str, micro_j: u32, in_flight: u32, zone: &str, intensity: f32) -> InMemoryWorker {
    InMemoryWorker::new(
        id,
        Capability {
            models: vec!["llama-70b".into()],
            accelerator: Accelerator::Gpu,
            max_concurrency: 16,
            continuous_batching: true,
            paged_kv: true,
            zone: zone.into(),
        },
        Load {
            in_flight,
            queued_tokens: 0,
            p50_latency_ms: 80,
            p99_latency_ms: 240,
            micro_joules_per_token: micro_j,
            g_co2e_per_kwh: intensity,
        },
    )
}

#[test]
fn joule_router_picks_efficient_worker() {
    let a = mk("us-va", 200, 0, "US-VA", 400.0);
    let b = mk("eu-fr", 50, 0, "EU-FR", 60.0);
    let c = mk("au-nsw", 150, 0, "AU-NSW", 600.0);
    let pool: Vec<&dyn Worker> = vec![&a, &b, &c];

    let r = Router::new(Strategy::JouleWeighted);
    let pick = r
        .pick(
            &pool,
            &Request {
                model: "llama-70b",
                expected_tokens: 256,
            },
        )
        .expect("ok");
    assert_eq!(pick.id(), "eu-fr");
}

#[test]
fn carbon_router_breaks_ties_by_grid_intensity() {
    // Two workers, identical joule efficiency, different grid zones.
    let dirty = mk("dirty", 100, 0, "AU-NSW", 700.0);
    let clean = mk("clean", 100, 0, "EU-FR", 50.0);
    let pool: Vec<&dyn Worker> = vec![&dirty, &clean];

    let r = Router::new(Strategy::CarbonWeighted);
    let pick = r
        .pick(
            &pool,
            &Request {
                model: "llama-70b",
                expected_tokens: 128,
            },
        )
        .expect("ok");
    assert_eq!(pick.id(), "clean");
}

#[test]
fn least_busy_picks_idle() {
    let busy = mk("busy", 100, 12, "EU-FR", 60.0);
    let idle = mk("idle", 200, 0, "US-VA", 400.0);
    let pool: Vec<&dyn Worker> = vec![&busy, &idle];

    let r = Router::new(Strategy::LeastBusy);
    let pick = r
        .pick(
            &pool,
            &Request {
                model: "llama-70b",
                expected_tokens: 64,
            },
        )
        .expect("ok");
    assert_eq!(pick.id(), "idle");
}
