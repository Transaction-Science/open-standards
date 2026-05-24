//! Sampling strategies.
//!
//! Implements the OTel-defined samplers:
//!
//! - [`AlwaysOnSampler`] — always returns `RecordAndSample`.
//! - [`AlwaysOffSampler`] — always returns `Drop`.
//! - [`TraceIdRatioBased`] — deterministic ratio based on trace_id high bytes.
//! - [`ParentBased`] — defers to the remote-parent decision when present.

use crate::context::{SpanContext, TraceId};

/// Result of a sampling decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SamplingDecision {
    /// Drop the span entirely.
    Drop,
    /// Record the span but do not propagate the sampled flag.
    RecordOnly,
    /// Record and propagate the sampled flag.
    RecordAndSample,
}

impl SamplingDecision {
    /// True if the span should be recorded at all.
    pub fn is_recorded(&self) -> bool {
        matches!(self, SamplingDecision::RecordOnly | SamplingDecision::RecordAndSample)
    }

    /// True if the sampled flag should be propagated.
    pub fn is_sampled(&self) -> bool {
        matches!(self, SamplingDecision::RecordAndSample)
    }
}

/// Sampler trait. Implementations are `Send + Sync`.
pub trait Sampler: Send + Sync + std::fmt::Debug {
    /// Decide whether to sample a span with this trace_id and parent context.
    fn should_sample(
        &self,
        parent: Option<&SpanContext>,
        trace_id: &TraceId,
        name: &str,
    ) -> SamplingDecision;

    /// Human-readable description for diagnostics.
    fn description(&self) -> String;
}

/// Always sample.
#[derive(Debug, Default)]
pub struct AlwaysOnSampler;

impl Sampler for AlwaysOnSampler {
    fn should_sample(
        &self,
        _parent: Option<&SpanContext>,
        _trace_id: &TraceId,
        _name: &str,
    ) -> SamplingDecision {
        SamplingDecision::RecordAndSample
    }

    fn description(&self) -> String {
        "AlwaysOnSampler".to_string()
    }
}

/// Never sample.
#[derive(Debug, Default)]
pub struct AlwaysOffSampler;

impl Sampler for AlwaysOffSampler {
    fn should_sample(
        &self,
        _parent: Option<&SpanContext>,
        _trace_id: &TraceId,
        _name: &str,
    ) -> SamplingDecision {
        SamplingDecision::Drop
    }

    fn description(&self) -> String {
        "AlwaysOffSampler".to_string()
    }
}

/// Deterministic ratio sampler: keeps `ratio` fraction of trace_ids.
///
/// Uses the high 64 bits of the trace_id, interpreted big-endian.
#[derive(Debug)]
pub struct TraceIdRatioBased {
    ratio: f64,
    threshold: u64,
}

impl TraceIdRatioBased {
    /// Build a sampler that keeps `ratio` of traces.
    ///
    /// `ratio` is clamped to `[0.0, 1.0]`.
    pub fn new(ratio: f64) -> Self {
        let clamped = ratio.clamp(0.0, 1.0);
        // Map [0,1] into u64 range. 1.0 -> u64::MAX so every id passes.
        let threshold = if clamped >= 1.0 {
            u64::MAX
        } else if clamped <= 0.0 {
            0
        } else {
            (clamped * (u64::MAX as f64)) as u64
        };
        Self {
            ratio: clamped,
            threshold,
        }
    }

    /// The clamped ratio in `[0.0, 1.0]`.
    pub fn ratio(&self) -> f64 {
        self.ratio
    }

    /// The u64 threshold value used for comparison.
    pub fn threshold(&self) -> u64 {
        self.threshold
    }
}

impl Sampler for TraceIdRatioBased {
    fn should_sample(
        &self,
        _parent: Option<&SpanContext>,
        trace_id: &TraceId,
        _name: &str,
    ) -> SamplingDecision {
        if self.threshold == 0 {
            return SamplingDecision::Drop;
        }
        if self.threshold == u64::MAX {
            return SamplingDecision::RecordAndSample;
        }
        // Big-endian load of the high 8 bytes.
        let hi = u64::from_be_bytes([
            trace_id.0[0],
            trace_id.0[1],
            trace_id.0[2],
            trace_id.0[3],
            trace_id.0[4],
            trace_id.0[5],
            trace_id.0[6],
            trace_id.0[7],
        ]);
        if hi < self.threshold {
            SamplingDecision::RecordAndSample
        } else {
            SamplingDecision::Drop
        }
    }

    fn description(&self) -> String {
        format!("TraceIdRatioBased({:.6})", self.ratio)
    }
}

/// Parent-based sampler: respect the parent's sampled flag if present,
/// otherwise consult a `root` sampler.
#[derive(Debug)]
pub struct ParentBased {
    root: Box<dyn Sampler>,
}

impl ParentBased {
    /// Wrap a root sampler used for spans with no parent.
    pub fn new(root: Box<dyn Sampler>) -> Self {
        Self { root }
    }
}

impl Sampler for ParentBased {
    fn should_sample(
        &self,
        parent: Option<&SpanContext>,
        trace_id: &TraceId,
        name: &str,
    ) -> SamplingDecision {
        match parent {
            Some(p) if p.is_valid() => {
                if p.flags.is_sampled() {
                    SamplingDecision::RecordAndSample
                } else {
                    SamplingDecision::Drop
                }
            }
            _ => self.root.should_sample(parent, trace_id, name),
        }
    }

    fn description(&self) -> String {
        format!("ParentBased({})", self.root.description())
    }
}
