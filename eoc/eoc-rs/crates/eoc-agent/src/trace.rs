//! Structured execution traces.
//!
//! Every loop emits a [`Trace`] consisting of ordered [`Span`]s. A span
//! captures *what kind of step* ran (think / act / observe / reflect),
//! its inputs and outputs as opaque strings, and the joule + token cost
//! attributed to it. Traces are content-addressable via the existing
//! [`eoc_core::Receipt`] machinery — the cascade can memoize entire
//! agent runs, not just individual responses.

use serde::{Deserialize, Serialize};

/// Kind of step.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SpanKind {
    /// Model reasoned (no side-effect).
    Think,
    /// Tool invoked.
    Act,
    /// Tool produced observation.
    Observe,
    /// Model self-critiqued.
    Reflect,
    /// Planner expanded a subgoal.
    Plan,
    /// Worker executed a planned subgoal.
    Execute,
    /// Search node expansion (ToT / LATS).
    Expand,
}

impl SpanKind {
    /// Stable string identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            SpanKind::Think => "think",
            SpanKind::Act => "act",
            SpanKind::Observe => "observe",
            SpanKind::Reflect => "reflect",
            SpanKind::Plan => "plan",
            SpanKind::Execute => "execute",
            SpanKind::Expand => "expand",
        }
    }
}

/// One step in an agent trace.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Span {
    /// Step index within the parent trace.
    pub index: usize,
    /// What kind of step ran.
    pub kind: SpanKind,
    /// Short human-readable label (e.g. tool name).
    pub label: String,
    /// Input payload — typically a prompt or tool-arg JSON.
    pub input: String,
    /// Output payload — typically a completion or tool-result JSON.
    pub output: String,
    /// Tokens charged to this span.
    pub tokens: u64,
    /// Energy charged to this span (micro-joules).
    pub microjoules: u64,
}

impl Span {
    /// Build a new span.
    pub fn new(
        index: usize,
        kind: SpanKind,
        label: impl Into<String>,
        input: impl Into<String>,
        output: impl Into<String>,
    ) -> Self {
        Self {
            index,
            kind,
            label: label.into(),
            input: input.into(),
            output: output.into(),
            tokens: 0,
            microjoules: 0,
        }
    }

    /// Attach a token charge.
    pub fn with_tokens(mut self, tokens: u64) -> Self {
        self.tokens = tokens;
        self
    }

    /// Attach an energy charge.
    pub fn with_microjoules(mut self, microjoules: u64) -> Self {
        self.microjoules = microjoules;
        self
    }
}

/// An ordered execution trace.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Trace {
    /// All spans, in chronological order.
    pub spans: Vec<Span>,
}

impl Trace {
    /// Empty trace.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a span, fixing up its index.
    pub fn push(&mut self, mut span: Span) {
        span.index = self.spans.len();
        self.spans.push(span);
    }

    /// Total tokens across all spans.
    pub fn total_tokens(&self) -> u64 {
        self.spans.iter().map(|s| s.tokens).sum()
    }

    /// Total micro-joules across all spans.
    pub fn total_microjoules(&self) -> u64 {
        self.spans.iter().map(|s| s.microjoules).sum()
    }

    /// Number of spans.
    pub fn len(&self) -> usize {
        self.spans.len()
    }

    /// Empty?
    pub fn is_empty(&self) -> bool {
        self.spans.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trace_accumulates() {
        let mut t = Trace::new();
        t.push(
            Span::new(0, SpanKind::Think, "ponder", "in", "out")
                .with_tokens(10)
                .with_microjoules(200),
        );
        t.push(
            Span::new(0, SpanKind::Act, "calc", "1+1", "2")
                .with_tokens(2)
                .with_microjoules(50),
        );
        assert_eq!(t.len(), 2);
        assert_eq!(t.spans[1].index, 1);
        assert_eq!(t.total_tokens(), 12);
        assert_eq!(t.total_microjoules(), 250);
    }
}
