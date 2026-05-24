//! EOC — agentic loops with per-step energy accounting.
//!
//! Modern agent systems are not single-shot completions: they iterate
//! through *think → act → observe* cycles, sometimes branching, sometimes
//! reflecting, sometimes planning ahead. EOC takes that loop and threads
//! a **joule meter** through every step, so an agent's total cost is the
//! sum of (a) provider-reported token costs, (b) tool-execution wall
//! energy, and (c) any local inference performed between steps.
//!
//! ## Loops shipped
//!
//! | Module                         | Pattern                              |
//! |--------------------------------|--------------------------------------|
//! | [`react`]                      | ReAct (Thought / Action / Observation) |
//! | [`reflexion`]                  | Reflexion (self-critique + episodic memory) |
//! | [`tot`]                        | Tree of Thoughts (BFS / DFS)         |
//! | [`lats`]                       | LATS / MCTS-UCT                      |
//! | [`plan_execute`]               | Plan-and-Execute (planner + worker)  |
//! | [`codeact`]                    | CodeAct (tool calls expressed as code) |
//!
//! ## Budgets
//!
//! Every loop accepts a [`Budget`](agent::Budget) that combines
//! [`TokenBudget`](budget::TokenBudget), [`JouleBudget`](budget::JouleBudget),
//! and a tool-call cap. The first cap to be exhausted produces an
//! [`StopReason::BudgetExhausted`](agent::StopReason) — no loop in this
//! crate is allowed to run unbounded.
//!
//! ## No unsafe, no `unwrap` outside tests.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod agent;
pub mod budget;
pub mod codeact;
pub mod error;
pub mod lats;
pub mod plan_execute;
pub mod react;
pub mod reflexion;
pub mod tot;
pub mod trace;

pub use agent::{Agent, AgentLoop, AgentStep, Budget, StopReason};
pub use budget::{JouleBudget, TokenBudget};
pub use codeact::CodeActLoop;
pub use error::{AgentError, AgentResult};
pub use lats::{LatsConfig, LatsLoop};
pub use plan_execute::{PlanExecuteLoop, Planner, Worker};
pub use react::{ReActLoop, ReActStep};
pub use reflexion::{ReflexionLoop, ReflexionMemory};
pub use tot::{Search, ToTLoop, Thought};
pub use trace::{Span, Trace};
