//! Core abstractions shared by every loop in this crate.
//!
//! * [`LlmProvider`] — a minimal vendor-neutral text completion port.
//! * [`Tool`] — a side-effectful subroutine the model can call.
//! * [`Budget`] — the combined token / joule / tool-call cap.
//! * [`Agent`] — a trait every loop implements (drives a goal to a verdict).
//! * [`AgentLoop`] — the harness most loops compose: maintains a [`Trace`],
//!   charges every step against a [`Budget`], and meters wall-clock joules
//!   through [`eoc_meter`].
//! * [`StopReason`] — why the loop stopped (success, exhausted, capped, error).

use std::collections::HashMap;
use std::sync::Arc;

use async_trait::async_trait;
use eoc_meter::{JouleCounter, StubCounter};
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::budget::{JouleBudget, TokenBudget};
use crate::error::{AgentError, AgentResult};
use crate::trace::{Span, SpanKind, Trace};

/// A bare-bones vendor-neutral LLM completion port.
///
/// Production deployments wire this to [`eoc_vendor_api`] backends; tests
/// supply scripted implementations.
#[async_trait]
pub trait LlmProvider: Send + Sync {
    /// Run a single completion. Implementations *should* return their
    /// best estimate of `prompt_tokens + completion_tokens` and the
    /// estimated micro-joules attributable to the call.
    async fn complete(&self, prompt: &str) -> AgentResult<LlmReply>;
}

/// Output of one [`LlmProvider::complete`] call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LlmReply {
    /// The completion text.
    pub text: String,
    /// Total tokens (prompt + completion).
    pub tokens: u64,
    /// Energy estimate (micro-joules).
    pub microjoules: u64,
}

impl LlmReply {
    /// Convenience constructor.
    pub fn new(text: impl Into<String>, tokens: u64, microjoules: u64) -> Self {
        Self {
            text: text.into(),
            tokens,
            microjoules,
        }
    }
}

/// A side-effectful primitive the agent can invoke.
#[async_trait]
pub trait Tool: Send + Sync {
    /// Tool name — must be unique within a [`ToolBox`].
    fn name(&self) -> &str;

    /// Run the tool. Arguments and result are opaque JSON.
    async fn run(&self, args: Value) -> AgentResult<Value>;
}

/// A name-indexed collection of [`Tool`]s.
#[derive(Default)]
pub struct ToolBox {
    tools: HashMap<String, Arc<dyn Tool>>,
}

impl ToolBox {
    /// Empty box.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a tool. Last write wins.
    pub fn insert<T: Tool + 'static>(&mut self, tool: T) {
        self.tools.insert(tool.name().to_string(), Arc::new(tool));
    }

    /// Look up by name.
    pub fn get(&self, name: &str) -> Option<Arc<dyn Tool>> {
        self.tools.get(name).cloned()
    }

    /// Number of registered tools.
    pub fn len(&self) -> usize {
        self.tools.len()
    }

    /// Empty?
    pub fn is_empty(&self) -> bool {
        self.tools.is_empty()
    }
}

/// Combined per-loop budget.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Budget {
    /// Token cap.
    pub tokens: TokenBudget,
    /// Joule cap.
    pub joules: JouleBudget,
    /// Maximum tool calls.
    pub max_tool_calls: usize,
    /// Tool calls used so far.
    pub tool_calls_used: usize,
    /// Maximum loop iterations (hard cap on `Think`/`Act`/`Observe` cycles).
    pub max_iterations: usize,
}

impl Budget {
    /// Convenience: an unbounded budget except for a hard iteration cap.
    /// Useful for tests.
    pub fn unlimited(max_iterations: usize) -> Self {
        Self {
            tokens: TokenBudget::unlimited(),
            joules: JouleBudget::unlimited(),
            max_tool_calls: usize::MAX,
            tool_calls_used: 0,
            max_iterations,
        }
    }

    /// Build from explicit caps.
    pub fn new(
        token_cap: u64,
        joule_cap: f64,
        max_tool_calls: usize,
        max_iterations: usize,
    ) -> Self {
        Self {
            tokens: TokenBudget::new(token_cap),
            joules: JouleBudget::from_joules(joule_cap),
            max_tool_calls,
            tool_calls_used: 0,
            max_iterations,
        }
    }

    /// Charge an LLM step. Returns the [`StopReason`] iff a cap is hit.
    pub fn charge_llm(&mut self, tokens: u64, microjoules: u64) -> Option<StopReason> {
        let token_ok = self.tokens.charge(tokens);
        let joule_ok = self.joules.charge(microjoules);
        match (token_ok, joule_ok) {
            (false, _) => Some(StopReason::BudgetExhausted("tokens".into())),
            (_, false) => Some(StopReason::BudgetExhausted("joules".into())),
            _ => None,
        }
    }

    /// Charge a tool call.
    pub fn charge_tool(&mut self, microjoules: u64) -> Option<StopReason> {
        self.tool_calls_used = self.tool_calls_used.saturating_add(1);
        if self.tool_calls_used > self.max_tool_calls {
            return Some(StopReason::BudgetExhausted("tool_calls".into()));
        }
        if !self.joules.charge(microjoules) {
            return Some(StopReason::BudgetExhausted("joules".into()));
        }
        None
    }
}

/// Reason a loop terminated.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StopReason {
    /// The agent returned a final answer.
    Done,
    /// A specific cap exhausted (`"tokens"`, `"joules"`, `"tool_calls"`).
    BudgetExhausted(String),
    /// Hard iteration cap reached without a verdict.
    MaxIterations,
    /// Caller-supplied predicate signalled early stop.
    EarlyStop(String),
    /// Loop aborted by an unrecoverable error.
    Error(String),
}

/// A single step report. Loops yield these via [`Agent::step`] when the
/// caller wants step-by-step control instead of `run_to_completion`.
#[derive(Debug, Clone)]
pub struct AgentStep {
    /// Step index.
    pub index: usize,
    /// Stop reason if the loop has terminated, `None` otherwise.
    pub stop: Option<StopReason>,
    /// Latest output text — for callers that want to stream.
    pub last_output: String,
}

/// The common Agent trait every loop implements.
#[async_trait]
pub trait Agent: Send + Sync {
    /// Drive the agent to completion or a stop condition. Returns the
    /// final output text plus the reason the loop stopped.
    async fn run(&mut self, goal: &str) -> AgentResult<(String, StopReason)>;

    /// Read the accumulated execution trace.
    fn trace(&self) -> &Trace;

    /// Read the current budget state.
    fn budget(&self) -> &Budget;
}

/// Shared loop harness. Most loops embed one of these and call its
/// `charge_llm` / `charge_tool` / `push_span` helpers. Decouples the
/// budget-and-trace bookkeeping from the per-loop control flow.
pub struct AgentLoop {
    /// Budget that gates every step.
    pub budget: Budget,
    /// Trace appended to on every step.
    pub trace: Trace,
    /// Hardware counter for measured energy.
    pub meter: Arc<dyn JouleCounter>,
}

impl AgentLoop {
    /// Build with the default meter for the host.
    pub fn new(budget: Budget) -> Self {
        Self {
            budget,
            trace: Trace::new(),
            meter: Arc::new(StubCounter),
        }
    }

    /// Build with an explicit meter.
    pub fn with_meter(budget: Budget, meter: Arc<dyn JouleCounter>) -> Self {
        Self {
            budget,
            trace: Trace::new(),
            meter,
        }
    }

    /// Take a meter reading. Returns 0 if the counter is unavailable.
    pub fn read_meter(&self) -> u64 {
        self.meter.read_microjoules().unwrap_or(0)
    }

    /// Wrap an async block, charging measured wall-energy from the meter
    /// **plus** an upper-bound `estimated_microjoules` so loops degrade
    /// gracefully on stub-only hosts.
    pub async fn measured<F, Fut, T>(&self, estimated_microjoules: u64, f: F) -> (T, u64)
    where
        F: FnOnce() -> Fut,
        Fut: std::future::Future<Output = T>,
    {
        let before = self.read_meter();
        let out = f().await;
        let after = self.read_meter();
        let measured = after.saturating_sub(before);
        // Use the larger of measured vs estimated as the charge.
        let charged = measured.max(estimated_microjoules);
        (out, charged)
    }

    /// Append a span to the trace.
    pub fn push_span(&mut self, span: Span) {
        self.trace.push(span);
    }

    /// Charge an LLM step against the budget *and* record the span.
    /// Returns the stop reason if the budget is exhausted.
    pub fn charge_llm_step(
        &mut self,
        kind: SpanKind,
        label: impl Into<String>,
        input: impl Into<String>,
        reply: &LlmReply,
    ) -> Option<StopReason> {
        let span = Span::new(self.trace.len(), kind, label, input, reply.text.clone())
            .with_tokens(reply.tokens)
            .with_microjoules(reply.microjoules);
        self.push_span(span);
        self.budget.charge_llm(reply.tokens, reply.microjoules)
    }

    /// Check the hard iteration cap.
    pub fn iteration_check(&self, iter: usize) -> Option<StopReason> {
        if iter >= self.budget.max_iterations {
            Some(StopReason::MaxIterations)
        } else {
            None
        }
    }

    /// Helper: classify a provider error.
    pub fn provider_err(e: impl std::fmt::Display) -> AgentError {
        AgentError::Provider(e.to_string())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn budget_charge_llm_trips_token_cap() {
        let mut b = Budget::new(10, 1.0, 100, 100);
        let stop = b.charge_llm(20, 0);
        assert_eq!(stop, Some(StopReason::BudgetExhausted("tokens".into())));
    }

    #[tokio::test]
    async fn agent_loop_records_spans() {
        let mut loop_ = AgentLoop::new(Budget::unlimited(10));
        let reply = LlmReply::new("hi", 3, 100);
        let stop = loop_.charge_llm_step(SpanKind::Think, "t", "in", &reply);
        assert!(stop.is_none());
        assert_eq!(loop_.trace.len(), 1);
        assert_eq!(loop_.trace.total_tokens(), 3);
    }
}
