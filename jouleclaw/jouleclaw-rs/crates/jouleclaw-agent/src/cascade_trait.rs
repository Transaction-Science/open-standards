//! The cascade-dispatch shim that breaks the agent↔runtime cycle.

use jouleclaw_cascade::types::{Answer, AnswerError, Query};

/// The agent's view of "ask the cascade a sub-query." The consumer
/// implements this over their live `Runtime`; the agent never holds a
/// `Runtime` directly, so there is no dependency cycle.
///
/// The bound is `Send` (an agent may move to a worker thread) but **not**
/// `Sync`: dispatch is `&mut self` and runs in the agent's sequential
/// loop, never shared across threads by reference. Requiring `Sync` would
/// reject the canonical adapter — a shim holding a `&mut Runtime` or
/// `Arc<Mutex<Runtime>>` over the live cascade — for no benefit.
pub trait AgentCascade: Send {
    fn dispatch(&mut self, query: &Query) -> Result<Answer, AnswerError>;
}

/// A test double that resolves each sub-query with a canned answer and
/// records the queries it saw. Also usable as a simple programmable
/// stub in integration tests.
pub struct MockCascade {
    /// Per-dispatch answer text, consumed in order. When exhausted,
    /// falls back to `default_answer`.
    pub scripted: std::collections::VecDeque<Result<String, String>>,
    /// Fallback answer when the script is empty.
    pub default_answer: String,
    /// Joules charged per dispatch.
    pub joules_per_dispatch: f64,
    /// Confidence reported per dispatch.
    pub confidence: f32,
    /// Every query text this mock was asked to resolve, in order.
    pub seen: Vec<String>,
}

impl Default for MockCascade {
    fn default() -> Self {
        Self {
            scripted: std::collections::VecDeque::new(),
            default_answer: "ok".to_string(),
            joules_per_dispatch: 1.0,
            confidence: 0.8,
            seen: Vec::new(),
        }
    }
}

impl MockCascade {
    /// Echo each sub-query back as `"answer: <text>"`.
    pub fn echo() -> Self {
        Self::default()
    }

    /// A mock pre-loaded to fail on the Nth (0-based) dispatch.
    pub fn failing_on(n: usize) -> Self {
        let mut scripted = std::collections::VecDeque::new();
        for i in 0..=n {
            if i == n {
                scripted.push_back(Err("sub-dispatch failed".to_string()));
            } else {
                scripted.push_back(Ok(format!("part {i}")));
            }
        }
        Self {
            scripted,
            ..Self::default()
        }
    }
}

impl AgentCascade for MockCascade {
    fn dispatch(&mut self, query: &Query) -> Result<Answer, AnswerError> {
        use jouleclaw_cascade::types::{
            AnswerOutput, ExecutionTrace, QueryInput, TierId,
        };
        use jouleclaw_cascade::verification::VerificationStatus;

        let text = match &query.input {
            QueryInput::Text(t) => t.clone(),
            QueryInput::Multimodal { text, .. } => text.clone(),
            _ => String::new(),
        };
        self.seen.push(text.clone());

        let body = match self.scripted.pop_front() {
            Some(Ok(s)) => s,
            Some(Err(e)) => {
                return Err(AnswerError::TierFailed {
                    tier: TierId::L6Agent,
                    cause: e,
                });
            }
            None => format!("answer: {text}"),
        };

        Ok(Answer {
            output: AnswerOutput::Text(body),
            tier_used: TierId::L1(jouleclaw_cascade::types::L1Primitive::Retrieve),
            joules_spent: self.joules_per_dispatch,
            confidence: self.confidence,
            trace: ExecutionTrace::default(),
            verification: VerificationStatus::Resolved,
        })
    }
}
