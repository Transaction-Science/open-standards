//! Plan-and-Execute: a [`Planner`] decomposes the goal into ordered
//! subgoals, a [`Worker`] runs each one in turn, and the loop returns
//! the worker's final output.
//!
//! Separating planning from execution lets the planner be a larger /
//! more deliberate model than the worker — often resulting in lower
//! total token + joule cost than running one big model end-to-end.

use std::sync::Arc;

use async_trait::async_trait;

use crate::agent::{Agent, AgentLoop, Budget, LlmProvider, StopReason};
use crate::error::{AgentError, AgentResult};
use crate::trace::SpanKind;

/// Decomposes a goal into ordered subgoals.
#[async_trait]
pub trait Planner: Send + Sync {
    /// Produce an ordered list of subgoals.
    async fn plan(&self, goal: &str) -> AgentResult<Vec<String>>;
}

/// Executes one subgoal, optionally given the previous step's output as
/// context.
#[async_trait]
pub trait Worker: Send + Sync {
    /// Execute a subgoal. `prior` is the output of the immediately
    /// preceding step (empty for the first step).
    async fn execute(&self, subgoal: &str, prior: &str) -> AgentResult<String>;
}

/// A planner backed by an [`LlmProvider`]. Expects the model to emit
/// one subgoal per line, optionally prefixed with `1.`, `2.`, `-`, etc.
pub struct LlmPlanner {
    provider: Arc<dyn LlmProvider>,
}

impl LlmPlanner {
    /// Build.
    pub fn new(provider: Arc<dyn LlmProvider>) -> Self {
        Self { provider }
    }
}

#[async_trait]
impl Planner for LlmPlanner {
    async fn plan(&self, goal: &str) -> AgentResult<Vec<String>> {
        let prompt = format!(
            "Decompose this goal into 1-7 numbered subgoals, one per line.\n\nGoal: {goal}"
        );
        let reply = self.provider.complete(&prompt).await?;
        let plan: Vec<String> = reply
            .text
            .lines()
            .map(|l| l.trim_start_matches(|c: char| c.is_ascii_digit() || matches!(c, '.' | '-' | '*' | ' ' | ')')).trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();
        if plan.is_empty() {
            return Err(AgentError::Plan("planner returned no subgoals".into()));
        }
        Ok(plan)
    }
}

/// A worker backed by an [`LlmProvider`].
pub struct LlmWorker {
    provider: Arc<dyn LlmProvider>,
}

impl LlmWorker {
    /// Build.
    pub fn new(provider: Arc<dyn LlmProvider>) -> Self {
        Self { provider }
    }
}

#[async_trait]
impl Worker for LlmWorker {
    async fn execute(&self, subgoal: &str, prior: &str) -> AgentResult<String> {
        let prompt = if prior.is_empty() {
            format!("Subgoal: {subgoal}\n\nProduce the next step's output.")
        } else {
            format!(
                "Prior step output: {prior}\nSubgoal: {subgoal}\n\nProduce the next step's output."
            )
        };
        let reply = self.provider.complete(&prompt).await?;
        Ok(reply.text)
    }
}

/// Plan-and-Execute driver.
pub struct PlanExecuteLoop {
    planner: Arc<dyn Planner>,
    worker: Arc<dyn Worker>,
    inner: AgentLoop,
}

impl PlanExecuteLoop {
    /// Build with explicit planner + worker.
    pub fn new(planner: Arc<dyn Planner>, worker: Arc<dyn Worker>, budget: Budget) -> Self {
        Self {
            planner,
            worker,
            inner: AgentLoop::new(budget),
        }
    }
}

#[async_trait]
impl Agent for PlanExecuteLoop {
    async fn run(&mut self, goal: &str) -> AgentResult<(String, StopReason)> {
        let subgoals = self.planner.plan(goal).await?;
        // Charge a notional cost for the planner step; planners that
        // wrap LlmProvider directly already self-meter via charge_llm_step,
        // but the trait-level abstraction does not.
        self.inner.push_span(crate::trace::Span::new(
            0,
            SpanKind::Plan,
            "plan_execute.plan",
            goal,
            subgoals.join(" | "),
        ));

        let mut prior = String::new();
        for (i, sub) in subgoals.iter().enumerate() {
            if let Some(stop) = self.inner.iteration_check(i) {
                return Ok((prior, stop));
            }
            let out = self.worker.execute(sub, &prior).await?;
            self.inner.push_span(crate::trace::Span::new(
                0,
                SpanKind::Execute,
                "plan_execute.execute",
                sub,
                out.clone(),
            ));
            // Charge tool-call quota for the worker step.
            if let Some(stop) = self.inner.budget.charge_tool(0) {
                return Ok((out, stop));
            }
            prior = out;
        }
        Ok((prior, StopReason::Done))
    }

    fn trace(&self) -> &crate::trace::Trace {
        &self.inner.trace
    }

    fn budget(&self) -> &Budget {
        &self.inner.budget
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct StaticPlanner(Vec<String>);
    #[async_trait]
    impl Planner for StaticPlanner {
        async fn plan(&self, _goal: &str) -> AgentResult<Vec<String>> {
            Ok(self.0.clone())
        }
    }

    struct EchoWorker;
    #[async_trait]
    impl Worker for EchoWorker {
        async fn execute(&self, subgoal: &str, prior: &str) -> AgentResult<String> {
            Ok(format!("{prior}|{subgoal}"))
        }
    }

    #[tokio::test]
    async fn runs_plan_in_order() {
        let planner = Arc::new(StaticPlanner(vec!["a".into(), "b".into(), "c".into()]));
        let worker = Arc::new(EchoWorker);
        let mut pe = PlanExecuteLoop::new(planner, worker, Budget::unlimited(10));
        let (out, reason) = pe.run("ignored").await.unwrap();
        assert_eq!(out, "|a|b|c");
        assert_eq!(reason, StopReason::Done);
    }
}
