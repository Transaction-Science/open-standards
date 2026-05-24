//! LATS — Language Agent Tree Search with MCTS / UCT.
//!
//! Reuses the [`Thought`](crate::tot::Thought) node type from
//! [`tot`](crate::tot) and adds UCT-based node selection. The classic
//! UCT formula is
//!
//! ```text
//! UCT(node) = Q(node)/N(node) + c * sqrt(ln(N(parent)) / N(node))
//! ```
//!
//! where `Q` is the cumulative reward, `N` is the visit count, and `c`
//! is the exploration constant.
//!
//! Each MCTS iteration runs *select → expand → simulate → backprop*. The
//! simulator is the LLM acting as a value oracle: it scores a candidate
//! continuation 0-1 and that score is treated as the rollout reward.

use std::sync::Arc;

use async_trait::async_trait;
use serde::{Deserialize, Serialize};

use crate::agent::{Agent, AgentLoop, Budget, LlmProvider, StopReason};
use crate::error::AgentResult;
use crate::trace::SpanKind;
use crate::tot::ToTLoop;

/// LATS hyperparameters.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct LatsConfig {
    /// UCT exploration constant. Defaults to `sqrt(2)`.
    pub c: f32,
    /// Branching factor per expansion.
    pub branch: usize,
    /// Maximum tree depth.
    pub max_depth: usize,
    /// Maximum MCTS iterations (independent of [`Budget::max_iterations`];
    /// the budget caps the *total* number of LLM calls, not the MCTS
    /// loop count).
    pub max_simulations: usize,
}

impl Default for LatsConfig {
    fn default() -> Self {
        Self {
            c: std::f32::consts::SQRT_2,
            branch: 2,
            max_depth: 4,
            max_simulations: 8,
        }
    }
}

#[derive(Debug, Clone)]
struct McNode {
    parent: Option<usize>,
    children: Vec<usize>,
    text: String,
    depth: usize,
    visits: u32,
    reward_sum: f32,
    finished_answer: Option<String>,
}

impl McNode {
    fn ucb(&self, parent_visits: u32, c: f32) -> f32 {
        if self.visits == 0 {
            return f32::INFINITY;
        }
        let mean = self.reward_sum / (self.visits as f32);
        let exploration = c * ((parent_visits.max(1) as f32).ln() / (self.visits as f32)).sqrt();
        mean + exploration
    }
}

/// LATS driver.
pub struct LatsLoop {
    provider: Arc<dyn LlmProvider>,
    inner: AgentLoop,
    cfg: LatsConfig,
    nodes: Vec<McNode>,
}

impl LatsLoop {
    /// Build.
    pub fn new(provider: Arc<dyn LlmProvider>, budget: Budget, cfg: LatsConfig) -> Self {
        Self {
            provider,
            inner: AgentLoop::new(budget),
            cfg,
            nodes: Vec::new(),
        }
    }

    fn select(&self) -> usize {
        // Descend the tree picking the highest UCB child until we hit a
        // node with no children.
        let mut cursor = 0;
        loop {
            let kids = self.nodes[cursor].children.clone();
            if kids.is_empty() {
                return cursor;
            }
            let parent_visits = self.nodes[cursor].visits;
            let mut best = kids[0];
            let mut best_score = f32::NEG_INFINITY;
            for &k in &kids {
                let score = self.nodes[k].ucb(parent_visits, self.cfg.c);
                if score > best_score {
                    best_score = score;
                    best = k;
                }
            }
            cursor = best;
        }
    }

    async fn expand(&mut self, goal: &str, parent_id: usize) -> AgentResult<Option<StopReason>> {
        if self.nodes[parent_id].depth >= self.cfg.max_depth {
            return Ok(None);
        }
        for _ in 0..self.cfg.branch {
            let parent_text = self.nodes[parent_id].text.clone();
            let prompt = format!(
                "Goal: {goal}\nCurrent path: {parent_text}\n\nPropose one continuation. End with `FINISH: answer` if you can solve it now."
            );
            let reply = self.provider.complete(&prompt).await?;
            if let Some(stop) =
                self.inner
                    .charge_llm_step(SpanKind::Expand, "lats.expand", &prompt, &reply)
            {
                return Ok(Some(stop));
            }
            let finished_answer = ToTLoop::is_finish(&reply.text);
            let new_id = self.nodes.len();
            let depth = self.nodes[parent_id].depth + 1;
            let text = reply.text;
            self.nodes.push(McNode {
                parent: Some(parent_id),
                children: Vec::new(),
                text,
                depth,
                visits: 0,
                reward_sum: 0.0,
                finished_answer,
            });
            self.nodes[parent_id].children.push(new_id);
        }
        Ok(None)
    }

    async fn simulate(&mut self, node_id: usize) -> AgentResult<(f32, Option<StopReason>)> {
        let prompt = format!(
            "Rate the likely solution quality of this thought on a 0.0-1.0 scale. Reply with just the number.\n\nThought: {t}",
            t = self.nodes[node_id].text
        );
        let reply = self.provider.complete(&prompt).await?;
        if let Some(stop) =
            self.inner
                .charge_llm_step(SpanKind::Think, "lats.simulate", &prompt, &reply)
        {
            return Ok((0.0, Some(stop)));
        }
        Ok((ToTLoop::parse_score(&reply.text), None))
    }

    fn backprop(&mut self, mut node_id: usize, reward: f32) {
        loop {
            let n = &mut self.nodes[node_id];
            n.visits = n.visits.saturating_add(1);
            n.reward_sum += reward;
            match n.parent {
                Some(p) => node_id = p,
                None => return,
            }
        }
    }

    /// Run the MCTS loop and return the best-scoring leaf answer
    /// (or the highest-mean child of the root if no leaf finished).
    pub async fn search(&mut self, goal: &str) -> AgentResult<(String, StopReason)> {
        // Root.
        self.nodes.push(McNode {
            parent: None,
            children: Vec::new(),
            text: String::from("(root)"),
            depth: 0,
            visits: 0,
            reward_sum: 0.0,
            finished_answer: None,
        });

        let mut best_answer = String::new();
        for sim in 0..self.cfg.max_simulations {
            if let Some(stop) = self.inner.iteration_check(sim) {
                return Ok((best_answer, stop));
            }
            let leaf = self.select();
            if let Some(stop) = self.expand(goal, leaf).await? {
                return Ok((best_answer, stop));
            }
            // Pick the first new child for simulation (or the leaf if no
            // expansion happened because depth was already maxed).
            let target = self.nodes[leaf].children.first().copied().unwrap_or(leaf);
            if let Some(ans) = self.nodes[target].finished_answer.clone() {
                self.backprop(target, 1.0);
                return Ok((ans, StopReason::Done));
            }
            let (reward, maybe_stop) = self.simulate(target).await?;
            if let Some(stop) = maybe_stop {
                return Ok((best_answer, stop));
            }
            self.backprop(target, reward);
            // Track the best-mean leaf so far.
            if let Some(best_id) = self.nodes[0]
                .children
                .iter()
                .max_by(|a, b| {
                    let ma = self.nodes[**a].reward_sum
                        / (self.nodes[**a].visits.max(1) as f32);
                    let mb = self.nodes[**b].reward_sum
                        / (self.nodes[**b].visits.max(1) as f32);
                    ma.partial_cmp(&mb).unwrap_or(std::cmp::Ordering::Equal)
                })
                .copied()
            {
                best_answer = self.nodes[best_id].text.clone();
            }
        }
        Ok((best_answer, StopReason::EarlyStop("max_simulations".into())))
    }
}

#[async_trait]
impl Agent for LatsLoop {
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
    fn ucb_prefers_unvisited() {
        let n = McNode {
            parent: None,
            children: vec![],
            text: String::new(),
            depth: 0,
            visits: 0,
            reward_sum: 0.0,
            finished_answer: None,
        };
        assert!(n.ucb(10, 1.4).is_infinite());
    }

    #[test]
    fn ucb_decreases_with_visits() {
        let n1 = McNode {
            parent: None,
            children: vec![],
            text: String::new(),
            depth: 0,
            visits: 1,
            reward_sum: 0.5,
            finished_answer: None,
        };
        let n10 = McNode {
            parent: None,
            children: vec![],
            text: String::new(),
            depth: 0,
            visits: 10,
            reward_sum: 5.0,
            finished_answer: None,
        };
        // Same mean (0.5) but n10 has higher visit count => less exploration bonus.
        assert!(n1.ucb(20, 1.4) > n10.ucb(20, 1.4));
    }
}
