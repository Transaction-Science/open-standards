//! SelfModel (spec §4.3).
//!
//! Observes retriever performance, recomputes capability status, and
//! emits a [`SystemCapabilities`] snapshot on demand. This is what
//! makes the planner adaptive: a retriever that's slow today gets
//! flagged DEGRADED, one that's failing gets UNAVAILABLE.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Mutex;

use chrono::Utc;

use jouleclaw_schema::{
    CapabilityStatus, ReasonerCapability, RetrieverCapability, SystemCapabilities,
};

/// One observed retrieval outcome. The orchestrator emits these as
/// it executes; the SelfModel uses them to recompute capability.
#[derive(Debug, Clone, Copy)]
pub struct Observation {
    pub success: bool,
    pub latency_ms: u32,
}

const HISTORY_DEPTH: usize = 100;
const MIN_OBSERVATIONS_FOR_RECOMPUTE: usize = 5;
const UNAVAILABLE_THRESHOLD: f64 = 0.50;
const DEGRADED_SUCCESS_THRESHOLD: f64 = 0.85;
const DEGRADED_LATENCY_MULTIPLIER: u32 = 3;

pub struct SelfModel {
    inner: Mutex<Inner>,
}

struct Inner {
    retrievers: BTreeMap<String, RetrieverCapability>,
    reasoners: BTreeMap<String, ReasonerCapability>,
    history: BTreeMap<String, VecDeque<Observation>>,
}

impl SelfModel {
    pub fn new() -> Self {
        Self {
            inner: Mutex::new(Inner {
                retrievers: BTreeMap::new(),
                reasoners: BTreeMap::new(),
                history: BTreeMap::new(),
            }),
        }
    }

    pub fn register_retriever(&self, capability: RetrieverCapability) {
        let mut g = self.inner.lock().unwrap();
        let id = capability.retriever_id.clone();
        g.retrievers.insert(id.clone(), capability);
        g.history.entry(id).or_default();
    }

    pub fn register_reasoner(&self, capability: ReasonerCapability) {
        let mut g = self.inner.lock().unwrap();
        g.reasoners
            .insert(capability.reasoner_id.clone(), capability);
    }

    /// Record a retrieval observation. Recomputes status when there
    /// are enough samples.
    pub fn observe(&self, retriever_id: &str, obs: Observation) {
        let mut g = self.inner.lock().unwrap();
        let q = g.history.entry(retriever_id.to_string()).or_default();
        if q.len() >= HISTORY_DEPTH {
            q.pop_front();
        }
        q.push_back(obs);

        if q.len() < MIN_OBSERVATIONS_FOR_RECOMPUTE {
            return;
        }
        let observations: Vec<Observation> = q.iter().copied().collect();
        let success_rate = observations.iter().filter(|o| o.success).count() as f64
            / observations.len() as f64;
        let mut latencies: Vec<u32> = observations.iter().map(|o| o.latency_ms).collect();
        latencies.sort_unstable();
        let p99 = percentile_u32(&latencies, 0.99);

        // Drop the history borrow before mutating retrievers.
        let now = Utc::now();
        if let Some(cap) = g.retrievers.get_mut(retriever_id) {
            cap.success_rate_recent = success_rate;
            cap.p99_latency_ms = p99;
            cap.status = if success_rate < UNAVAILABLE_THRESHOLD {
                CapabilityStatus::Unavailable
            } else if success_rate < DEGRADED_SUCCESS_THRESHOLD
                || p99 > cap.typical_latency_ms.saturating_mul(DEGRADED_LATENCY_MULTIPLIER)
            {
                CapabilityStatus::Degraded
            } else {
                CapabilityStatus::Healthy
            };
            if !observations.last().map(|o| o.success).unwrap_or(true) {
                cap.last_failure_at = Some(now);
            }
        }
    }

    /// Produce a fresh [`SystemCapabilities`] snapshot. Cheap;
    /// recomputes overall status from per-retriever statuses.
    pub fn snapshot(&self) -> SystemCapabilities {
        let g = self.inner.lock().unwrap();
        let retrievers: Vec<RetrieverCapability> = g.retrievers.values().cloned().collect();
        let reasoners: Vec<ReasonerCapability> = g.reasoners.values().cloned().collect();
        let overall = if retrievers
            .iter()
            .all(|r| matches!(r.status, CapabilityStatus::Healthy))
        {
            CapabilityStatus::Healthy
        } else if retrievers
            .iter()
            .all(|r| matches!(r.status, CapabilityStatus::Unavailable))
        {
            CapabilityStatus::Unavailable
        } else {
            CapabilityStatus::Degraded
        };
        let degradation_notes: Vec<String> = retrievers
            .iter()
            .filter(|r| !matches!(r.status, CapabilityStatus::Healthy))
            .map(|r| {
                format!(
                    "{}: {:?} (success {:.0}%, p99 {} ms)",
                    r.retriever_id,
                    r.status,
                    r.success_rate_recent * 100.0,
                    r.p99_latency_ms,
                )
            })
            .collect();
        SystemCapabilities {
            schema_version: "5.0".into(),
            snapshot_timestamp: Utc::now(),
            retrievers,
            reasoners,
            overall_status: overall,
            degradation_notes,
            metadata: Default::default(),
        }
    }
}

impl Default for SelfModel {
    fn default() -> Self {
        Self::new()
    }
}

/// Pick a percentile from a sorted ascending slice. `q` in 0.0..=1.0.
fn percentile_u32(sorted: &[u32], q: f64) -> u32 {
    if sorted.is_empty() {
        return 0;
    }
    let idx = (q * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_schema::Modality;

    fn fresh_retriever(id: &str) -> RetrieverCapability {
        RetrieverCapability {
            retriever_id: id.into(),
            status: CapabilityStatus::Healthy,
            domains_covered: vec!["geography".into()],
            modalities_supported: vec![Modality::Text],
            typical_latency_ms: 500,
            p99_latency_ms: 1000,
            success_rate_recent: 1.0,
            last_failure_at: None,
            known_limitations: vec![],
            populates_valid_time: false,
            populates_transaction_time: false,
            populates_granularity: false,
            populates_scope: false,
            populates_certainty: false,
            populates_provenance: true,
            authority_tier: 1,
        }
    }

    #[test]
    fn snapshot_reflects_registered_retrievers() {
        let sm = SelfModel::new();
        sm.register_retriever(fresh_retriever("wikidata"));
        let snap = sm.snapshot();
        assert_eq!(snap.retrievers.len(), 1);
        assert!(matches!(snap.overall_status, CapabilityStatus::Healthy));
    }

    #[test]
    fn enough_failures_marks_unavailable() {
        let sm = SelfModel::new();
        sm.register_retriever(fresh_retriever("wikidata"));
        for _ in 0..10 {
            sm.observe(
                "wikidata",
                Observation {
                    success: false,
                    latency_ms: 200,
                },
            );
        }
        let snap = sm.snapshot();
        assert!(matches!(
            snap.retrievers[0].status,
            CapabilityStatus::Unavailable
        ));
    }

    #[test]
    fn high_p99_marks_degraded() {
        let sm = SelfModel::new();
        sm.register_retriever(fresh_retriever("wikidata"));
        // 10 successes but with terrible latency.
        for _ in 0..10 {
            sm.observe(
                "wikidata",
                Observation {
                    success: true,
                    latency_ms: 8000,
                },
            );
        }
        let snap = sm.snapshot();
        assert!(matches!(
            snap.retrievers[0].status,
            CapabilityStatus::Degraded
        ));
    }

    #[test]
    fn mostly_healthy_stays_healthy() {
        let sm = SelfModel::new();
        sm.register_retriever(fresh_retriever("wikidata"));
        for _ in 0..10 {
            sm.observe(
                "wikidata",
                Observation {
                    success: true,
                    latency_ms: 400,
                },
            );
        }
        let snap = sm.snapshot();
        assert!(matches!(
            snap.retrievers[0].status,
            CapabilityStatus::Healthy
        ));
    }

    #[test]
    fn under_min_observations_no_recompute() {
        let sm = SelfModel::new();
        sm.register_retriever(fresh_retriever("wikidata"));
        // Only 2 failures — below MIN_OBSERVATIONS_FOR_RECOMPUTE.
        sm.observe(
            "wikidata",
            Observation { success: false, latency_ms: 200 },
        );
        sm.observe(
            "wikidata",
            Observation { success: false, latency_ms: 200 },
        );
        let snap = sm.snapshot();
        // Still healthy because recompute hasn't fired yet.
        assert!(matches!(
            snap.retrievers[0].status,
            CapabilityStatus::Healthy
        ));
    }
}
