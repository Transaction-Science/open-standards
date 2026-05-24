//! Tree of Thoughts — BFS / DFS with a self-scoring expansion.
//!
//! At each node the agent asks the LLM for `k` candidate continuations,
//! then scores them in a second prompt. The frontier is expanded under
//! the search strategy until a `Finish` candidate appears or the
//! [`Budget`](crate::agent::Budget) is exhausted.
//!
//! The branching factor `k` and the depth cap are part of the search
//! state, not the budget — the budget caps *total* steps and energy.

use std::collections::VecDeque;
use std::sync::Arc;

use async_trait::async_trait;

use crate::agent::{Agent, AgentLoop, Budget, LlmProvider, StopReason};
use crate::error::AgentResult;
use crate::trace::SpanKind;

/// Search strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Search {
    /// Breadth-first frontier expansion.
    Bfs,
    /// Depth-first frontier expansion.
    Dfs,
}

/// One node in the thought tree.
#[derive(Debug, Clone)]
pub struct Thought {
    /// Index of this node within the tree.
    pub id: usize,
    /// Parent index (`None` for the root).
    pub parent: Option<usize>,
    /// Text the model emitted at this node.
    pub text: String,
    /// Self-assessed score (higher = more promising).
    pub score: f32,
    /// Depth (root = 0).
    pub depth: usize,
    /// Is this a `Finish` node?
    pub finished: bool,
}

/// ToT driver.
pub struct ToTLoop {
    provider: Arc<dyn LlmProvider>,
    inner: AgentLoop,
    /// Branching factor.
    pub branch: usize,
    /// Max depth.
    pub max_depth: usize,
    /// Search order.
    pub search: Search,
    nodes: Vec<Thought>,
}

impl ToTLoop {
    /// Build.
    pub fn new(
        provider: Arc<dyn LlmProvider>,
        budget: Budget,
        branch: usize,
        max_depth: usize,
        search: Search,
    ) -> Self {
        Self {
            provider,
            inner: AgentLoop::new(budget),
            branch: branch.max(1),
            max_depth: max_depth.max(1),
            search,
            nodes: Vec::new(),
        }
    }

    /// Borrow the tree (read-only).
    pub fn nodes(&self) -> &[Thought] {
        &self.nodes
    }

    fn expand_prompt(&self, goal: &str, parent: Option<&Thought>) -> String {
        match parent {
            None => format!(
                "Goal: {goal}\n\nPropose one next thought. End with `FINISH: answer` if you can solve it now."
            ),
            Some(p) => format!(
                "Goal: {goal}\nCurrent thought: {t}\n\nPropose one continuation. End with `FINISH: answer` if you can solve it now.",
                t = p.text
            ),
        }
    }

    fn score_prompt(text: &str) -> String {
        format!(
            "Rate the promise of this thought on a 0.0-1.0 scale. Reply with just the number.\n\nThought: {text}"
        )
    }

    pub(crate) fn parse_score(text: &str) -> f32 {
        for tok in text.split(|c: char| !(c.is_ascii_digit() || c == '.')) {
            if let Ok(v) = tok.parse::<f32>() {
                return v.clamp(0.0, 1.0);
            }
        }
        0.5
    }

    pub(crate) fn is_finish(text: &str) -> Option<String> {
        for line in text.lines() {
            let t = line.trim();
            if let Some(rest) = t.strip_prefix("FINISH:") {
                return Some(rest.trim().to_string());
            }
        }
        None
    }

    /// Run the search loop directly (instead of via the `Agent` trait)
    /// when the caller wants to inspect the tree.
    pub async fn search(&mut self, goal: &str) -> AgentResult<(String, StopReason)> {
        // Frontier of node-ids waiting to be expanded.
        let mut frontier: VecDeque<usize> = VecDeque::new();
        let mut best_text = String::new();

        // Root.
        self.nodes.push(Thought {
            id: 0,
            parent: None,
            text: String::from("(root)"),
            score: 0.0,
            depth: 0,
            finished: false,
        });
        frontier.push_back(0);

        let mut iter = 0usize;
        while let Some(parent_id) = match self.search {
            Search::Bfs => frontier.pop_front(),
            Search::Dfs => frontier.pop_back(),
        } {
            if let Some(stop) = self.inner.iteration_check(iter) {
                return Ok((best_text, stop));
            }
            iter = iter.saturating_add(1);

            let parent_depth = self.nodes[parent_id].depth;
            if parent_depth >= self.max_depth {
                continue;
            }

            for _ in 0..self.branch {
                let parent_snap = self.nodes[parent_id].clone();
                let prompt = self.expand_prompt(goal, Some(&parent_snap));
                let reply = self.provider.complete(&prompt).await?;
                if let Some(stop) = self.inner.charge_llm_step(
                    SpanKind::Expand,
                    "tot.expand",
                    &prompt,
                    &reply,
                ) {
                    return Ok((best_text, stop));
                }

                let finished_answer = Self::is_finish(&reply.text);
                let score_prompt = Self::score_prompt(&reply.text);
                let score_reply = self.provider.complete(&score_prompt).await?;
                if let Some(stop) = self.inner.charge_llm_step(
                    SpanKind::Think,
                    "tot.score",
                    &score_prompt,
                    &score_reply,
                ) {
                    return Ok((best_text, stop));
                }
                let score = Self::parse_score(&score_reply.text);
                let new_id = self.nodes.len();
                let depth = parent_depth + 1;
                let finished = finished_answer.is_some();
                let text = reply.text.clone();
                self.nodes.push(Thought {
                    id: new_id,
                    parent: Some(parent_id),
                    text: text.clone(),
                    score,
                    depth,
                    finished,
                });
                if let Some(ans) = finished_answer {
                    return Ok((ans, StopReason::Done));
                }
                if score >= self.nodes.iter().map(|n| n.score).fold(0.0_f32, f32::max) {
                    best_text = text;
                }
                if depth < self.max_depth {
                    frontier.push_back(new_id);
                }
            }
        }
        Ok((best_text, StopReason::EarlyStop("frontier_empty".into())))
    }
}

#[async_trait]
impl Agent for ToTLoop {
    async fn run(&mut self, goal: &str) -> AgentResult<(String, StopReason)> {
        self.search(goal).await
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
    fn parse_score_extracts_number() {
        assert!((ToTLoop::parse_score("0.7") - 0.7).abs() < 1e-6);
        assert!((ToTLoop::parse_score("Score: 0.42 thanks") - 0.42).abs() < 1e-6);
        assert!((ToTLoop::parse_score("bogus") - 0.5).abs() < 1e-6);
    }

    #[test]
    fn is_finish_matches() {
        assert_eq!(ToTLoop::is_finish("FINISH: 42"), Some("42".into()));
        assert!(ToTLoop::is_finish("not done yet").is_none());
    }
}
