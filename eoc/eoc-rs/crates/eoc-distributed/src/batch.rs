//! Continuous batching scheduler (vLLM / TGI style) and Triton-style
//! dynamic-batching admission.
//!
//! The classic vLLM idea: requests arrive at different times but iterate
//! together. Each "iteration" the scheduler:
//!
//! 1. **Admits** new requests up to the KV-cache budget.
//! 2. **Decodes** one token for every running request.
//! 3. **Evicts** requests that completed.
//!
//! This module provides a deterministic, in-process implementation of
//! that loop suitable for unit tests and as a reference for backends.

use serde::{Deserialize, Serialize};

use crate::error::{DistributedError, Result};

/// A request being tracked by the batcher.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BatchRequest {
    /// Caller-supplied id.
    pub id: String,
    /// Tokens already in the prefill / decode position.
    pub generated_tokens: u32,
    /// Caller's max-new-tokens cap.
    pub max_new_tokens: u32,
    /// Estimated KV slots required for the prefill of this request.
    pub prefill_kv_slots: u32,
}

impl BatchRequest {
    /// Whether the request reached its caller-supplied budget.
    pub fn is_done(&self) -> bool {
        self.generated_tokens >= self.max_new_tokens
    }
}

/// Configuration for the continuous batcher.
#[derive(Debug, Clone, Copy)]
pub struct BatchConfig {
    /// Hard cap on simultaneously-running requests.
    pub max_running: usize,
    /// Total KV-cache slots available (vLLM block budget).
    pub kv_budget: u32,
    /// Triton-style "max_queue_delay" approximated as max tokens we'll
    /// wait before flushing a partial batch.
    pub max_queue_tokens: u32,
}

impl Default for BatchConfig {
    fn default() -> Self {
        Self {
            max_running: 32,
            kv_budget: 4_096,
            max_queue_tokens: 256,
        }
    }
}

/// Continuous batching scheduler.
#[derive(Debug)]
pub struct ContinuousBatcher {
    cfg: BatchConfig,
    waiting: Vec<BatchRequest>,
    running: Vec<BatchRequest>,
    used_kv: u32,
    iteration: u64,
}

impl ContinuousBatcher {
    /// Construct.
    pub fn new(cfg: BatchConfig) -> Self {
        Self {
            cfg,
            waiting: Vec::new(),
            running: Vec::new(),
            used_kv: 0,
            iteration: 0,
        }
    }

    /// Enqueue a request. Returns the new waiting-queue depth.
    pub fn enqueue(&mut self, req: BatchRequest) -> Result<usize> {
        if self.waiting.len() + self.running.len() >= self.cfg.max_running * 4 {
            return Err(DistributedError::QueueFull(
                self.waiting.len() + self.running.len(),
            ));
        }
        self.waiting.push(req);
        Ok(self.waiting.len())
    }

    /// In-flight (running) request count.
    pub fn running_len(&self) -> usize {
        self.running.len()
    }

    /// Waiting request count.
    pub fn waiting_len(&self) -> usize {
        self.waiting.len()
    }

    /// Total KV slots currently allocated.
    pub fn used_kv(&self) -> u32 {
        self.used_kv
    }

    /// Iteration counter (one tick = one decode step).
    pub fn iteration(&self) -> u64 {
        self.iteration
    }

    /// Admit as many waiting requests as the running cap + KV budget
    /// allow. Returns the count admitted this call.
    pub fn admit(&mut self) -> usize {
        let mut admitted = 0;
        while self.running.len() < self.cfg.max_running && !self.waiting.is_empty() {
            let head = &self.waiting[0];
            if self.used_kv + head.prefill_kv_slots > self.cfg.kv_budget {
                break;
            }
            let req = self.waiting.remove(0);
            self.used_kv += req.prefill_kv_slots;
            self.running.push(req);
            admitted += 1;
        }
        admitted
    }

    /// One decode step. Every running request gets one extra token; any
    /// that finished is evicted and its KV slots returned. Returns the
    /// number of completed requests.
    pub fn step(&mut self) -> usize {
        self.iteration += 1;
        let mut completed = 0;
        let mut i = 0;
        while i < self.running.len() {
            self.running[i].generated_tokens += 1;
            if self.running[i].is_done() {
                let done = self.running.remove(i);
                self.used_kv = self.used_kv.saturating_sub(done.prefill_kv_slots);
                completed += 1;
            } else {
                i += 1;
            }
        }
        completed
    }

    /// Whether enough work has piled up that we should force a flush
    /// even before reaching `max_running`. Mirrors Triton's
    /// `max_queue_delay_microseconds` semantics, but in token units.
    pub fn should_flush(&self) -> bool {
        if self.running.is_empty() {
            return false;
        }
        let total: u32 = self.waiting.iter().map(|r| r.prefill_kv_slots).sum();
        total >= self.cfg.max_queue_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn r(id: &str, max_new: u32, slots: u32) -> BatchRequest {
        BatchRequest {
            id: id.into(),
            generated_tokens: 0,
            max_new_tokens: max_new,
            prefill_kv_slots: slots,
        }
    }

    #[test]
    fn admit_respects_kv_budget() {
        let mut b = ContinuousBatcher::new(BatchConfig {
            max_running: 32,
            kv_budget: 100,
            max_queue_tokens: 64,
        });
        b.enqueue(r("a", 4, 60)).expect("ok");
        b.enqueue(r("b", 4, 60)).expect("ok"); // would overflow
        assert_eq!(b.admit(), 1);
        assert_eq!(b.running_len(), 1);
        assert_eq!(b.waiting_len(), 1);
    }

    #[test]
    fn step_advances_and_evicts() {
        let mut b = ContinuousBatcher::new(BatchConfig {
            max_running: 4,
            kv_budget: 1_000,
            max_queue_tokens: 64,
        });
        b.enqueue(r("a", 2, 10)).expect("ok");
        b.admit();
        b.step();
        b.step();
        assert_eq!(b.running_len(), 0);
        assert_eq!(b.used_kv(), 0);
    }
}
