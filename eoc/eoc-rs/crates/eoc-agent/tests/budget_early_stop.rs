//! Integration test: budgets trigger early stop.

use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use eoc_agent::agent::{Agent, Budget, LlmProvider, LlmReply};
use eoc_agent::budget::{JouleBudget, TokenBudget};
use eoc_agent::error::AgentResult;
use eoc_agent::agent::ToolBox;
use eoc_agent::react::ReActLoop;
use eoc_agent::StopReason;

struct Scripted {
    text: &'static str,
    tokens: u64,
    microjoules: u64,
    calls: Mutex<usize>,
}

#[async_trait]
impl LlmProvider for Scripted {
    async fn complete(&self, _prompt: &str) -> AgentResult<LlmReply> {
        *self.calls.lock().expect("lock") += 1;
        Ok(LlmReply::new(self.text, self.tokens, self.microjoules))
    }
}

#[tokio::test]
async fn token_budget_cuts_react_short() {
    let provider = Arc::new(Scripted {
        text: "Thought: t\nAction: nope[x]",
        tokens: 100,
        microjoules: 0,
        calls: Mutex::new(0),
    });
    let budget = Budget {
        tokens: TokenBudget::new(50),
        joules: JouleBudget::unlimited(),
        max_tool_calls: 100,
        tool_calls_used: 0,
        max_iterations: 100,
    };
    let mut agent = ReActLoop::new(provider.clone(), ToolBox::new(), budget, "sys");
    let (_out, stop) = agent.run("g").await.expect("ok");
    assert_eq!(stop, StopReason::BudgetExhausted("tokens".into()));
    // Exactly one call should have happened before the cap tripped.
    assert_eq!(*provider.calls.lock().expect("lock"), 1);
}

#[tokio::test]
async fn joule_budget_cuts_react_short() {
    let provider = Arc::new(Scripted {
        text: "Thought: t\nAction: nope[x]",
        tokens: 0,
        microjoules: 10_000,
        calls: Mutex::new(0),
    });
    let budget = Budget {
        tokens: TokenBudget::unlimited(),
        joules: JouleBudget::from_microjoules(5_000),
        max_tool_calls: 100,
        tool_calls_used: 0,
        max_iterations: 100,
    };
    let mut agent = ReActLoop::new(provider.clone(), ToolBox::new(), budget, "sys");
    let (_out, stop) = agent.run("g").await.expect("ok");
    assert_eq!(stop, StopReason::BudgetExhausted("joules".into()));
}

#[tokio::test]
async fn iteration_cap_trips_when_no_finish() {
    let provider = Arc::new(Scripted {
        text: "Thought: t\nAction: unknown[x]",
        tokens: 1,
        microjoules: 1,
        calls: Mutex::new(0),
    });
    // Tiny iteration cap.
    let budget = Budget {
        tokens: TokenBudget::unlimited(),
        joules: JouleBudget::unlimited(),
        max_tool_calls: 100,
        tool_calls_used: 0,
        max_iterations: 3,
    };
    let mut agent = ReActLoop::new(provider, ToolBox::new(), budget, "sys");
    let (_out, stop) = agent.run("g").await.expect("ok");
    assert_eq!(stop, StopReason::MaxIterations);
}
