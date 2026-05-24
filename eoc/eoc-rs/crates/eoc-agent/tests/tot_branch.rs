//! Integration test: Tree of Thoughts BFS expands branches and finishes
//! when a FINISH thought appears.

use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use eoc_agent::agent::{Agent, Budget, LlmProvider, LlmReply};
use eoc_agent::error::AgentResult;
use eoc_agent::tot::{Search, ToTLoop};
use eoc_agent::StopReason;

struct Scripted {
    replies: Mutex<Vec<String>>,
}

#[async_trait]
impl LlmProvider for Scripted {
    async fn complete(&self, _prompt: &str) -> AgentResult<LlmReply> {
        let text = {
            let mut q = self.replies.lock().expect("lock");
            if q.is_empty() {
                "0.0".to_string()
            } else {
                q.remove(0)
            }
        };
        Ok(LlmReply::new(text, 3, 100))
    }
}

#[tokio::test]
async fn tot_finishes_on_finish_thought() {
    // ToT alternates expansion + scoring. Branch=2, max_depth=2.
    // Sequence (BFS):
    //   root -> expand child A => "consider this"  (score "0.4")
    //   root -> expand child B => "FINISH: 7"      (score irrelevant)
    let replies = vec![
        "consider this".to_string(),
        "0.4".to_string(),
        "FINISH: 7".to_string(),
        "0.9".to_string(),
    ];
    let provider = Arc::new(Scripted {
        replies: Mutex::new(replies),
    });
    let mut agent = ToTLoop::new(provider, Budget::unlimited(20), 2, 2, Search::Bfs);
    let (out, stop) = agent.run("solve").await.expect("ok");
    assert_eq!(stop, StopReason::Done);
    assert_eq!(out, "7");
    // 3 nodes: root + 2 children.
    assert_eq!(agent.nodes().len(), 3);
}

#[tokio::test]
async fn tot_dfs_respects_iteration_cap() {
    // Provider always returns a never-finishing thought.
    let provider = Arc::new(Scripted {
        replies: Mutex::new(vec!["keep going".to_string(); 100]),
    });
    let mut agent = ToTLoop::new(provider, Budget::unlimited(2), 2, 4, Search::Dfs);
    let (_out, stop) = agent.run("loop").await.expect("ok");
    assert_eq!(stop, StopReason::MaxIterations);
}
