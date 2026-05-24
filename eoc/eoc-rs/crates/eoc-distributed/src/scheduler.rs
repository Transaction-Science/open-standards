//! Work-stealing scheduler for heterogeneous worker pools.
//!
//! Each [`Worker`] owns a logical inbox of work items. The scheduler
//! tracks per-worker depth and lets idle workers *steal* from busy
//! neighbours. Stealing is asymmetric: a worker with `accelerator =
//! Cpu` may steal from another CPU worker but won't steal a workload
//! tagged GPU-required, mirroring how Ray's resource-typed scheduler
//! enforces affinity.
//!
//! The scheduler doesn't actually execute work — it just decides where
//! work belongs. Execution is the worker's job; the scheduler only
//! moves [`WorkItem`]s between inboxes.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::error::{DistributedError, Result};
use crate::router::joule_score;
use crate::worker::{Accelerator, Worker};

/// A single unit of distributed work.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkItem {
    /// Caller-supplied id (e.g. request UUID).
    pub id: String,
    /// Model name (matches [`crate::worker::Capability::models`]).
    pub model: String,
    /// Expected generated tokens — drives joule-weighted placement.
    pub expected_tokens: u32,
    /// Required accelerator class, if any. `None` means "any".
    pub require: Option<Accelerator>,
    /// Optional KV-cache locality hint — the scheduler hands this to
    /// [`crate::kv_cache_aware`] when set.
    pub locality_hint: Option<String>,
}

/// Per-worker inbox.
#[derive(Debug, Default)]
struct Inbox {
    items: Vec<WorkItem>,
}

/// Work-stealing scheduler.
#[derive(Debug, Default)]
pub struct WorkStealingScheduler {
    inboxes: HashMap<String, Inbox>,
}

impl WorkStealingScheduler {
    /// Construct.
    pub fn new() -> Self {
        Self::default()
    }

    /// Make sure every worker has an inbox.
    pub fn ensure(&mut self, workers: &[&dyn Worker]) {
        for w in workers {
            self.inboxes.entry(w.id().to_string()).or_default();
        }
    }

    /// Submit work to the worker with the lowest joule-weighted score
    /// that can serve `item`. Returns the worker id the item landed in.
    pub fn submit<'w>(&mut self, workers: &[&'w dyn Worker], item: WorkItem) -> Result<&'w str> {
        let candidates: Vec<&'w dyn Worker> = workers
            .iter()
            .copied()
            .filter(|w| w.capability().serves(&item.model))
            .filter(|w| match item.require {
                None => true,
                Some(Accelerator::Mixed) => true,
                Some(req) => w.capability().accelerator == req,
            })
            .collect();
        if candidates.is_empty() {
            return Err(DistributedError::UnsatisfiedCapability(item.model.clone()));
        }
        let pick = candidates
            .iter()
            .min_by(|a, b| {
                joule_score(&a.load(), item.expected_tokens)
                    .partial_cmp(&joule_score(&b.load(), item.expected_tokens))
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .copied()
            .ok_or(DistributedError::NoWorkers)?;
        self.inboxes
            .entry(pick.id().to_string())
            .or_default()
            .items
            .push(item);
        Ok(pick.id())
    }

    /// Depth of `worker`'s inbox.
    pub fn depth(&self, worker_id: &str) -> usize {
        self.inboxes
            .get(worker_id)
            .map(|i| i.items.len())
            .unwrap_or(0)
    }

    /// Total queued work across all workers.
    pub fn total(&self) -> usize {
        self.inboxes.values().map(|i| i.items.len()).sum()
    }

    /// Steal one item from the deepest compatible inbox into `thief`.
    /// Returns the stolen item's id, or `None` if no steal happened.
    pub fn steal(&mut self, workers: &[&dyn Worker], thief: &str) -> Option<String> {
        let thief_cap = workers
            .iter()
            .find(|w| w.id() == thief)
            .map(|w| w.capability().clone())?;
        // Find the deepest inbox other than the thief itself whose head
        // item can run on `thief`.
        let mut victim_id: Option<String> = None;
        let mut victim_depth = 0;
        for (id, inbox) in &self.inboxes {
            if id == thief {
                continue;
            }
            if inbox.items.is_empty() {
                continue;
            }
            // Head item is the candidate steal.
            let head = &inbox.items[0];
            if !thief_cap.serves(&head.model) {
                continue;
            }
            if let Some(req) = head.require {
                if req != Accelerator::Mixed && thief_cap.accelerator != req {
                    continue;
                }
            }
            if inbox.items.len() > victim_depth {
                victim_depth = inbox.items.len();
                victim_id = Some(id.clone());
            }
        }
        let victim = victim_id?;
        let item = self.inboxes.get_mut(&victim)?.items.remove(0);
        let id = item.id.clone();
        self.inboxes
            .entry(thief.to_string())
            .or_default()
            .items
            .push(item);
        Some(id)
    }

    /// Pop the head of `worker`'s inbox.
    pub fn pop(&mut self, worker_id: &str) -> Option<WorkItem> {
        self.inboxes.get_mut(worker_id).and_then(|i| {
            if i.items.is_empty() {
                None
            } else {
                Some(i.items.remove(0))
            }
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::worker::{Capability, InMemoryWorker, Load};

    fn cpu(id: &str, micro_j: u32) -> InMemoryWorker {
        InMemoryWorker::new(
            id,
            Capability {
                models: vec!["m".into()],
                accelerator: Accelerator::Cpu,
                max_concurrency: 8,
                continuous_batching: false,
                paged_kv: false,
                zone: "EU-FR".into(),
            },
            Load {
                micro_joules_per_token: micro_j,
                ..Load::idle()
            },
        )
    }

    #[test]
    fn submit_lands_in_cheapest() {
        let mut s = WorkStealingScheduler::new();
        let a = cpu("a", 200);
        let b = cpu("b", 50);
        let pool: Vec<&dyn Worker> = vec![&a, &b];
        s.ensure(&pool);
        let where_ = s
            .submit(
                &pool,
                WorkItem {
                    id: "r0".into(),
                    model: "m".into(),
                    expected_tokens: 10,
                    require: None,
                    locality_hint: None,
                },
            )
            .expect("ok");
        assert_eq!(where_, "b");
    }
}
