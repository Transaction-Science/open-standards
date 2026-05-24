//! Request → worker placement.
//!
//! The router consumes load snapshots and produces a single worker id.
//! It is deliberately stateless: every call re-reads the snapshot view
//! it was given, so a caller can swap implementations or replay a trace
//! against multiple strategies for ablation.
//!
//! ## Strategies
//!
//! | Variant                | Source of truth                       |
//! |------------------------|----------------------------------------|
//! | [`Strategy::RoundRobin`]   | LiteLLM `simple-shuffle`                |
//! | [`Strategy::LeastBusy`]    | LiteLLM `least-busy` (in-flight + queue) |
//! | [`Strategy::LatencyWeighted`] | LiteLLM `latency-based` (p50)         |
//! | [`Strategy::JouleWeighted`]   | EOC-native: micro-J × tokens, ties → busyness |
//! | [`Strategy::CarbonWeighted`]  | Joules × zone gCO2e/kWh (via [`Load`]) |

use std::sync::atomic::{AtomicUsize, Ordering};

use crate::error::{DistributedError, Result};
use crate::worker::{Load, Worker};

/// Routing strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Strategy {
    /// Round-robin across satisfying workers.
    RoundRobin,
    /// Least-busy (in-flight + queue).
    LeastBusy,
    /// Lowest projected p50 latency.
    LatencyWeighted,
    /// Lowest projected joule cost for a hypothetical N-token request.
    JouleWeighted,
    /// Lowest projected gCO2e for a hypothetical N-token request.
    CarbonWeighted,
}

/// Inputs for one routing decision.
#[derive(Debug, Clone)]
pub struct Request<'a> {
    /// Model the caller wants to invoke. Workers that don't advertise
    /// this model are filtered out before the strategy runs.
    pub model: &'a str,
    /// Expected number of generated tokens. Used by joule- and
    /// carbon-weighted strategies; ignored by the others.
    pub expected_tokens: u32,
}

/// Stateless router. Holds nothing but a round-robin counter.
#[derive(Debug, Default)]
pub struct Router {
    strategy: Strategy,
    rr_counter: AtomicUsize,
}

impl Default for Strategy {
    fn default() -> Self {
        Strategy::JouleWeighted
    }
}

impl Router {
    /// Construct.
    pub fn new(strategy: Strategy) -> Self {
        Self {
            strategy,
            rr_counter: AtomicUsize::new(0),
        }
    }

    /// Configured strategy.
    pub fn strategy(&self) -> Strategy {
        self.strategy
    }

    /// Pick one worker id from `workers` for `req`. Returns
    /// [`DistributedError::NoWorkers`] / [`DistributedError::UnsatisfiedCapability`]
    /// on an empty / filtered candidate set.
    pub fn pick<'w>(&self, workers: &'w [&dyn Worker], req: &Request<'_>) -> Result<&'w dyn Worker> {
        if workers.is_empty() {
            return Err(DistributedError::NoWorkers);
        }
        let mut candidates: Vec<&dyn Worker> = workers
            .iter()
            .copied()
            .filter(|w| w.capability().serves(req.model))
            .collect();
        if candidates.is_empty() {
            return Err(DistributedError::UnsatisfiedCapability(req.model.into()));
        }
        match self.strategy {
            Strategy::RoundRobin => {
                let idx = self.rr_counter.fetch_add(1, Ordering::Relaxed) % candidates.len();
                Ok(candidates.remove(idx))
            }
            Strategy::LeastBusy => Ok(pick_min(&candidates, |w| w.load().busyness())),
            Strategy::LatencyWeighted => {
                Ok(pick_min(&candidates, |w| w.load().p50_latency_ms as f64))
            }
            Strategy::JouleWeighted => Ok(pick_min(&candidates, |w| {
                let l = w.load();
                let proj = l.projected_micro_joules(req.expected_tokens) as f64;
                // Tie-break with busyness to avoid all traffic piling on
                // the single most efficient worker.
                proj + l.busyness() * 1e-6
            })),
            Strategy::CarbonWeighted => Ok(pick_min(&candidates, |w| {
                w.load().projected_g_co2e(req.expected_tokens)
            })),
        }
    }
}

fn pick_min<'w, F>(workers: &[&'w dyn Worker], mut score: F) -> &'w dyn Worker
where
    F: FnMut(&dyn Worker) -> f64,
{
    debug_assert!(!workers.is_empty());
    let mut best: &dyn Worker = workers[0];
    let mut best_score = score(workers[0]);
    for w in &workers[1..] {
        let s = score(*w);
        if s < best_score {
            best_score = s;
            best = *w;
        }
    }
    best
}

/// Helper used by tests and by [`crate::scheduler`] to project the joule
/// score that the [`Strategy::JouleWeighted`] router would have computed.
pub fn joule_score(load: &Load, tokens: u32) -> f64 {
    (load.projected_micro_joules(tokens) as f64) + load.busyness() * 1e-6
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::{Accelerator, Capability, InMemoryWorker, Load};

    fn mk(id: &str, micro_j: u32, in_flight: u32) -> InMemoryWorker {
        InMemoryWorker::new(
            id,
            Capability {
                models: vec!["m".into()],
                accelerator: Accelerator::Gpu,
                max_concurrency: 16,
                continuous_batching: true,
                paged_kv: true,
                zone: "EU-FR".into(),
            },
            Load {
                in_flight,
                queued_tokens: 0,
                p50_latency_ms: 100,
                p99_latency_ms: 250,
                micro_joules_per_token: micro_j,
                g_co2e_per_kwh: 60.0,
            },
        )
    }

    #[test]
    fn empty_pool_errors() {
        let r = Router::new(Strategy::LeastBusy);
        let res = r.pick(
            &[],
            &Request {
                model: "m",
                expected_tokens: 10,
            },
        );
        assert!(matches!(res, Err(DistributedError::NoWorkers)));
    }

    #[test]
    fn unsatisfied_capability_errors() {
        let w = mk("a", 100, 0);
        let pool: Vec<&dyn Worker> = vec![&w];
        let r = Router::new(Strategy::LeastBusy);
        let res = r.pick(
            &pool,
            &Request {
                model: "other",
                expected_tokens: 10,
            },
        );
        assert!(matches!(
            res,
            Err(DistributedError::UnsatisfiedCapability(_))
        ));
    }

    #[test]
    fn joule_weighted_picks_efficient() {
        let a = mk("a", 200, 0);
        let b = mk("b", 50, 0); // 4x more efficient
        let pool: Vec<&dyn Worker> = vec![&a, &b];
        let r = Router::new(Strategy::JouleWeighted);
        let pick = r
            .pick(
                &pool,
                &Request {
                    model: "m",
                    expected_tokens: 32,
                },
            )
            .expect("ok");
        assert_eq!(pick.id(), "b");
    }
}
