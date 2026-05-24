//! Integration test: ReAct converges with a scripted mock vendor.

use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use eoc_agent::agent::{Agent, Budget, LlmProvider, LlmReply, Tool, ToolBox};
use eoc_agent::error::AgentResult;
use eoc_agent::react::ReActLoop;
use eoc_agent::StopReason;
use serde_json::Value;

struct ScriptedProvider {
    replies: Mutex<Vec<&'static str>>,
}

#[async_trait]
impl LlmProvider for ScriptedProvider {
    async fn complete(&self, _prompt: &str) -> AgentResult<LlmReply> {
        let text = {
            let mut q = self.replies.lock().expect("lock");
            if q.is_empty() {
                "Action: Finish[ran out]".to_string()
            } else {
                q.remove(0).to_string()
            }
        };
        Ok(LlmReply::new(text, 5, 1_000))
    }
}

struct AddTool;
#[async_trait]
impl Tool for AddTool {
    fn name(&self) -> &str {
        "calc"
    }
    async fn run(&self, args: Value) -> AgentResult<Value> {
        let s = args.get("input").and_then(|v| v.as_str()).unwrap_or("0+0");
        // Tiny parser: split on '+', sum integers.
        let mut total = 0i64;
        for tok in s.split('+') {
            total += tok.trim().parse::<i64>().unwrap_or(0);
        }
        Ok(serde_json::json!({ "sum": total }))
    }
}

#[tokio::test]
async fn react_converges() {
    let provider = Arc::new(ScriptedProvider {
        replies: Mutex::new(vec![
            "Thought: I'll add them.\nAction: calc[2+3]",
            "Thought: Now I know the answer.\nAction: Finish[5]",
        ]),
    });
    let mut tools = ToolBox::new();
    tools.insert(AddTool);
    let mut agent =
        ReActLoop::new(provider, tools, Budget::unlimited(10), "you are a math agent");
    let (out, stop) = agent.run("what is 2+3?").await.expect("ok");
    assert_eq!(out, "5");
    assert_eq!(stop, StopReason::Done);
    // Two LLM calls + one tool act => >= 3 spans.
    assert!(agent.trace().len() >= 3, "trace = {:?}", agent.trace().spans);
    assert!(agent.trace().total_tokens() >= 10);
}

#[tokio::test]
async fn react_unknown_tool_recovers() {
    let provider = Arc::new(ScriptedProvider {
        replies: Mutex::new(vec![
            "Thought: try mystery\nAction: mystery[?]",
            "Action: Finish[ok]",
        ]),
    });
    let tools = ToolBox::new();
    let mut agent =
        ReActLoop::new(provider, tools, Budget::unlimited(5), "system");
    let (out, stop) = agent.run("go").await.expect("ok");
    assert_eq!(out, "ok");
    assert_eq!(stop, StopReason::Done);
}
