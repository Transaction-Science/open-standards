//! Cost model with hindsight. Replaces the single-scalar `joules`
//! field in `TierEstimate` with a structure that distinguishes:
//!
//! * Prefill cost (compute-bound, proportional to input tokens)
//! * Decode cost (memory-bandwidth-bound, proportional to output)
//! * Substrate (CPU / GPU / NPU / Remote) — same primitives can have
//!   wildly different costs on different hardware
//! * Impedance mismatch `μ` — how badly the primitive set fits the
//!   substrate's native ops
//!
//! Why this matters: in real LLM inference, output tokens cost 3-10×
//! more than input tokens (each one requires a full decode pass).
//! For long-context-short-completion queries the prefill dominates;
//! for short-context-long-completion queries the decode dominates.
//! A single-scalar cost cannot route these correctly.
//!
//! The eLLM observation makes the substrate dimension load-bearing:
//! CPU memory capacity beats GPU at sufficiently long contexts, so
//! the same tier on different substrates has fundamentally different
//! cost shapes.

use std::time::Duration;

// ============================================================
// Substrate — which hardware the tier runs on
// ============================================================

/// What hardware executes this tier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Substrate {
    /// In-process CPU. Memory-rich, compute-modest.
    Cpu,
    /// GPU. Compute-rich, memory-constrained.
    Gpu,
    /// Neural processing unit (mobile, edge).
    Npu,
    /// Remote — cost includes network round-trip.
    Remote,
    /// Pure in-memory (cache hit, no compute).
    InMemory,
}

impl Substrate {
    pub fn name(&self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Gpu => "gpu",
            Self::Npu => "npu",
            Self::Remote => "remote",
            Self::InMemory => "memory",
        }
    }
}

// ============================================================
// Workload shape — what the query asks for
// ============================================================

/// Shape of work a query implies. Used to pick the right cost
/// function from a tier's cost model.
///
/// Derived from query length and expected output length. The router
/// can populate this; tiers consume it to scale their estimates.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct WorkloadShape {
    pub input_tokens: u32,
    pub expected_output_tokens: u32,
}

impl WorkloadShape {
    pub fn short() -> Self {
        Self { input_tokens: 32, expected_output_tokens: 32 }
    }
    pub fn typical() -> Self {
        Self { input_tokens: 256, expected_output_tokens: 256 }
    }
    pub fn long_context() -> Self {
        Self { input_tokens: 32_000, expected_output_tokens: 256 }
    }
    pub fn long_generation() -> Self {
        Self { input_tokens: 256, expected_output_tokens: 4_096 }
    }

    pub fn total_tokens(&self) -> u32 {
        self.input_tokens + self.expected_output_tokens
    }
}

// ============================================================
// Split cost estimate
// ============================================================

/// Cost prediction for a tier on a specific query.
///
/// Splits cost into prefill (the one-time pass over the input) and
/// decode (per-output-token cost). Tracks substrate and impedance
/// mismatch.
///
/// The `joules` accessor gives the legacy single-scalar; new code
/// should use the split fields directly.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct CostEstimate {
    /// Joules paid for the prefill pass.
    pub prefill_joules: f64,
    /// Joules paid per output token during decode.
    pub decode_joules_per_token: f64,
    /// Expected output tokens, for total-cost calculation.
    pub expected_output_tokens: u32,
    /// Wall-clock for prefill.
    pub prefill_latency: Duration,
    /// Wall-clock per output token.
    pub decode_latency_per_token: Duration,
    /// What hardware will execute.
    pub substrate: Substrate,
    /// Impedance mismatch factor. 1.0 = perfect fit; 10× means the
    /// primitive set asks for 10× more than the substrate's native
    /// throughput.
    pub mu: f64,
    /// Tier's lower-bound confidence in this query class.
    pub confidence_floor: f32,
}

impl CostEstimate {
    /// Total joules for the full query.
    pub fn total_joules(&self) -> f64 {
        self.prefill_joules
            + self.decode_joules_per_token * self.expected_output_tokens as f64
    }

    /// Total latency for the full query.
    pub fn total_latency(&self) -> Duration {
        self.prefill_latency
            + self.decode_latency_per_token * self.expected_output_tokens
    }

    /// Equivalent legacy single-scalar `joules`. Use for compatibility
    /// shims; prefer the split fields.
    pub fn joules(&self) -> f64 {
        self.total_joules()
    }

    /// A cost estimate for a pure cache/lookup tier — flat cost, no
    /// decode, in-memory substrate.
    pub fn flat(joules: f64, latency: Duration, confidence_floor: f32) -> Self {
        Self {
            prefill_joules: joules,
            decode_joules_per_token: 0.0,
            expected_output_tokens: 0,
            prefill_latency: latency,
            decode_latency_per_token: Duration::ZERO,
            substrate: Substrate::InMemory,
            mu: 1.0,
            confidence_floor,
        }
    }

    /// A cost estimate for a CPU-bound deterministic primitive
    /// (regex, arithmetic, template fill) — scales linearly with
    /// input size, no decode.
    pub fn deterministic(
        joules_per_char: f64,
        input_chars: usize,
        confidence_floor: f32,
    ) -> Self {
        Self {
            prefill_joules: joules_per_char * input_chars as f64,
            decode_joules_per_token: 0.0,
            expected_output_tokens: 0,
            prefill_latency: Duration::from_nanos(
                (10.0 * input_chars as f64) as u64),
            decode_latency_per_token: Duration::ZERO,
            substrate: Substrate::Cpu,
            mu: 1.0,
            confidence_floor,
        }
    }

    /// A cost estimate for a neural tier — split prefill + decode,
    /// substrate-aware.
    pub fn neural(
        prefill_joules_per_token: f64,
        decode_joules_per_token: f64,
        shape: WorkloadShape,
        substrate: Substrate,
        mu: f64,
        confidence_floor: f32,
    ) -> Self {
        Self {
            prefill_joules: prefill_joules_per_token
                * shape.input_tokens as f64 * mu,
            decode_joules_per_token: decode_joules_per_token * mu,
            expected_output_tokens: shape.expected_output_tokens,
            // ~1 µs per input token for prefill on modern hardware
            prefill_latency: Duration::from_micros(shape.input_tokens as u64),
            // ~10 ms per output token for decode (typical L4)
            decode_latency_per_token: Duration::from_millis(10),
            substrate,
            mu,
            confidence_floor,
        }
    }
}

// ============================================================
// Substrate-specific cost profiles
// ============================================================

/// A multi-substrate cost model. A tier can report different cost
/// shapes for the same underlying work running on different
/// hardware. The router (or a substrate-aware policy) picks which
/// `CostEstimate` to use.
///
/// This is what makes eLLM-style routing operational: the same
/// neural model has different prefill/decode profiles on CPU vs GPU,
/// and which one wins depends on the query shape (long context vs
/// short).
#[derive(Debug, Clone)]
pub struct MultiSubstrateCost {
    pub options: Vec<CostEstimate>,
}

impl MultiSubstrateCost {
    pub fn new() -> Self {
        Self { options: Vec::new() }
    }

    pub fn add(mut self, estimate: CostEstimate) -> Self {
        self.options.push(estimate);
        self
    }

    /// Pick the cheapest substrate for the given workload.
    pub fn cheapest(&self) -> Option<&CostEstimate> {
        self.options.iter().min_by(|a, b| {
            a.total_joules().partial_cmp(&b.total_joules()).unwrap()
        })
    }

    /// Pick the lowest-latency substrate.
    pub fn fastest(&self) -> Option<&CostEstimate> {
        self.options.iter().min_by(|a, b| {
            a.total_latency().cmp(&b.total_latency())
        })
    }

    /// Pick the substrate that fits a specific deadline.
    pub fn fits_deadline(&self, deadline: Duration) -> Option<&CostEstimate> {
        self.options.iter()
            .filter(|c| c.total_latency() <= deadline)
            .min_by(|a, b| {
                a.total_joules().partial_cmp(&b.total_joules()).unwrap()
            })
    }
}

impl Default for MultiSubstrateCost {
    fn default() -> Self { Self::new() }
}

// ============================================================
// Tests
// ============================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flat_cost_has_zero_decode() {
        let c = CostEstimate::flat(1e-9, Duration::from_nanos(10), 0.99);
        assert_eq!(c.total_joules(), 1e-9);
        assert_eq!(c.decode_joules_per_token, 0.0);
    }

    #[test]
    fn deterministic_scales_linearly() {
        let c1 = CostEstimate::deterministic(1e-12, 100, 1.0);
        let c2 = CostEstimate::deterministic(1e-12, 1000, 1.0);
        // 10× the input → 10× the cost.
        let ratio = c2.total_joules() / c1.total_joules();
        assert!((ratio - 10.0).abs() < 1e-9, "ratio = {}", ratio);
    }

    #[test]
    fn neural_cost_split_reflects_workload_shape() {
        // Long context (32K input, 256 output) — prefill dominates.
        let long_ctx = CostEstimate::neural(
            1e-6, 5e-4,           // 1 µJ prefill/tok, 500 µJ decode/tok
            WorkloadShape::long_context(),
            Substrate::Gpu, 1.0, 0.9,
        );
        let prefill_frac = long_ctx.prefill_joules / long_ctx.total_joules();
        assert!(prefill_frac > 0.15,
            "long-context prefill should be ≥15% of total, got {:.1}%",
            prefill_frac * 100.0);

        // Long generation (256 input, 4K output) — decode dominates.
        let long_gen = CostEstimate::neural(
            1e-6, 5e-4,
            WorkloadShape::long_generation(),
            Substrate::Gpu, 1.0, 0.9,
        );
        let decode_total = long_gen.decode_joules_per_token
            * long_gen.expected_output_tokens as f64;
        let decode_frac = decode_total / long_gen.total_joules();
        assert!(decode_frac > 0.99,
            "long-generation decode should dominate, got {:.1}%",
            decode_frac * 100.0);
    }

    #[test]
    fn multi_substrate_picks_cheapest() {
        // Mock: same model, two substrates.
        let m = MultiSubstrateCost::new()
            .add(CostEstimate::neural(
                1e-6, 1e-3, WorkloadShape::typical(),
                Substrate::Gpu, 1.0, 0.9,
            ))
            .add(CostEstimate::neural(
                5e-7, 5e-4, WorkloadShape::typical(),
                Substrate::Cpu, 1.5, 0.9,    // higher μ, lower base cost
            ));

        let cheapest = m.cheapest().unwrap();
        // CPU here is cheaper: 5e-7·1.5·256 + 5e-4·1.5·256
        //   = 192 µJ + 192 mJ ≈ 192 mJ
        // GPU: 1e-6·256 + 1e-3·256 = 256 µJ + 256 mJ ≈ 256 mJ
        assert_eq!(cheapest.substrate, Substrate::Cpu);
    }

    #[test]
    fn ellm_scenario_cpu_wins_at_long_context() {
        // The eLLM argument: at long contexts, CPU's memory capacity
        // beats GPU's compute advantage. Model this with a higher μ
        // for GPU at long contexts (memory pressure forces offload).
        let gpu_long = CostEstimate::neural(
            1e-6, 1e-3, WorkloadShape::long_context(),
            Substrate::Gpu, 3.0, 0.9,   // μ=3 from memory pressure
        );
        let cpu_long = CostEstimate::neural(
            2e-6, 2e-3, WorkloadShape::long_context(),
            Substrate::Cpu, 1.2, 0.9,   // μ=1.2, base 2× higher
        );
        // At 32K context, CPU's lower μ beats GPU's lower base.
        assert!(cpu_long.total_joules() < gpu_long.total_joules(),
            "CPU should win at long context: CPU {:.3e} J vs GPU {:.3e} J",
            cpu_long.total_joules(), gpu_long.total_joules());
    }

    #[test]
    fn legacy_joules_matches_total() {
        let c = CostEstimate::neural(
            1e-6, 5e-4, WorkloadShape::typical(),
            Substrate::Gpu, 1.0, 0.9,
        );
        assert_eq!(c.joules(), c.total_joules());
    }
}
