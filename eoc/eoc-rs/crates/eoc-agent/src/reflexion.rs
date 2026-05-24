//! Reflexion — self-critique + episodic memory.
//!
//! After each attempt the agent asks an LLM-as-judge whether the answer
//! is correct. On failure the judge's critique is appended to the
//! [`ReflexionMemory`] and the next attempt sees the accumulated lessons
//! prepended to its prompt. Bounded by [`Budget`](crate::agent::Budget).

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::agent::{Agent, AgentLoop, Budget, LlmProvider, StopReason};
use crate::error::AgentResult;
use crate::trace::SpanKind;

/// An append-only log of past critiques.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ReflexionMemory {
    /// One entry per failed attempt.
    pub lessons: Vec<String>,
}

impl ReflexionMemory {
    /// Empty memory.
    pub fn new() -> Self {
        Self::default()
    }

    /// Append a lesson.
    pub fn remember(&mut self, lesson: impl Into<String>) {
        self.lessons.push(lesson.into());
    }

    /// Render as a prompt block (empty string if no lessons yet).
    pub fn as_prompt_block(&self) -> String {
        if self.lessons.is_empty() {
            return String::new();
        }
        let mut s = String::from("Prior critiques to avoid repeating:\n");
        for (i, l) in self.lessons.iter().enumerate() {
            s.push_str(&format!("  {i}. {l}\n"));
        }
        s
    }
}

/// Result of one critique call.
#[derive(Debug, Clone)]
pub struct Critique {
    /// Was the attempt accepted as a final answer?
    pub accepted: bool,
    /// Free-text explanation (becomes a lesson if `!accepted`).
    pub feedback: String,
}

impl Critique {
    /// Parse a critique reply. The judge is expected to begin its reply
    /// with `PASS` (accept) or `FAIL` (reject), followed by free text.
    pub fn parse(text: &str) -> Self {
        let trimmed = text.trim_start();
        let accepted = trimmed
            .lines()
            .next()
            .map(|l| l.trim().eq_ignore_ascii_case("PASS"))
            .unwrap_or(false);
        let feedback = if accepted {
            String::new()
        } else {
            trimmed.to_string()
        };
        Self { accepted, feedback }
    }
}

/// Reflexion driver.
pub struct ReflexionLoop {
    actor: Arc<dyn LlmProvider>,
    judge: Arc<dyn LlmProvider>,
    inner: AgentLoop,
    /// Lessons learned across attempts.
    pub memory: ReflexionMemory,
    /// Maximum self-correction attempts.
    pub max_attempts: usize,
}

impl ReflexionLoop {
    /// Build with separate actor and judge providers.
    pub fn new(
        actor: Arc<dyn LlmProvider>,
        judge: Arc<dyn LlmProvider>,
        budget: Budget,
        max_attempts: usize,
    ) -> Self {
        Self {
            actor,
            judge,
            inner: AgentLoop::new(budget),
            memory: ReflexionMemory::new(),
            max_attempts,
        }
    }
}

#[async_trait]
impl Agent for ReflexionLoop {
    async fn run(&mut self, goal: &str) -> AgentResult<(String, StopReason)> {
        let mut last_attempt = String::new();
        for attempt in 0..self.max_attempts {
            if let Some(stop) = self.inner.iteration_check(attempt) {
                return Ok((last_attempt, stop));
            }

            let actor_prompt =
                format!("{lessons}Task: {goal}\n", lessons = self.memory.as_prompt_block());
            let attempt_reply = self.actor.complete(&actor_prompt).await?;
            last_attempt = attempt_reply.text.clone();
            if let Some(stop) = self.inner.charge_llm_step(
                SpanKind::Think,
                "reflexion.attempt",
                &actor_prompt,
                &attempt_reply,
            ) {
                return Ok((last_attempt, stop));
            }

            let judge_prompt = format!(
                "Goal: {goal}\nAttempt: {att}\n\nReply with PASS or FAIL on the first line, then a brief critique.",
                att = attempt_reply.text
            );
            let judge_reply = self.judge.complete(&judge_prompt).await?;
            if let Some(stop) = self.inner.charge_llm_step(
                SpanKind::Reflect,
                "reflexion.judge",
                &judge_prompt,
                &judge_reply,
            ) {
                return Ok((last_attempt, stop));
            }
            let critique = Critique::parse(&judge_reply.text);
            if critique.accepted {
                return Ok((attempt_reply.text, StopReason::Done));
            }
            self.memory.remember(critique.feedback);
        }
        Ok((last_attempt, StopReason::MaxIterations))
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

    #[test]
    fn critique_parses_pass() {
        let c = Critique::parse("PASS\nlooks fine");
        assert!(c.accepted);
        assert!(c.feedback.is_empty());
    }

    #[test]
    fn critique_parses_fail() {
        let c = Critique::parse("FAIL\nwrong sign");
        assert!(!c.accepted);
        assert!(c.feedback.contains("FAIL"));
    }

    #[test]
    fn memory_prompt_block_empty() {
        let m = ReflexionMemory::new();
        assert!(m.as_prompt_block().is_empty());
    }
}
