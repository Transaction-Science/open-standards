//! Integration test: Reflexion turns a failed first attempt into a
//! retry that uses the judge's critique.

use std::sync::Arc;
use std::sync::Mutex;

use async_trait::async_trait;
use eoc_agent::agent::{Agent, Budget, LlmProvider, LlmReply};
use eoc_agent::error::AgentResult;
use eoc_agent::reflexion::ReflexionLoop;
use eoc_agent::StopReason;

struct Scripted {
    replies: Mutex<Vec<&'static str>>,
}

#[async_trait]
impl LlmProvider for Scripted {
    async fn complete(&self, _prompt: &str) -> AgentResult<LlmReply> {
        let text = {
            let mut q = self.replies.lock().expect("lock");
            if q.is_empty() {
                "(empty)".to_string()
            } else {
                q.remove(0).to_string()
            }
        };
        Ok(LlmReply::new(text, 8, 500))
    }
}

#[tokio::test]
async fn reflexion_retries_after_failure() {
    // Actor: first attempt is wrong, second attempt is right.
    let actor = Arc::new(Scripted {
        replies: Mutex::new(vec!["answer: 41", "answer: 42"]),
    });
    // Judge: FAILs the first, PASSes the second.
    let judge = Arc::new(Scripted {
        replies: Mutex::new(vec!["FAIL\noff by one", "PASS\nlooks right"]),
    });
    let mut agent = ReflexionLoop::new(actor, judge, Budget::unlimited(10), 5);
    let (out, stop) = agent.run("what is 6 * 7?").await.expect("ok");
    assert_eq!(out, "answer: 42");
    assert_eq!(stop, StopReason::Done);
    assert_eq!(agent.memory.lessons.len(), 1);
    assert!(agent.memory.lessons[0].contains("off by one"));
}

#[tokio::test]
async fn reflexion_gives_up_after_max_attempts() {
    let actor = Arc::new(Scripted {
        replies: Mutex::new(vec!["bad", "bad", "bad"]),
    });
    let judge = Arc::new(Scripted {
        replies: Mutex::new(vec!["FAIL a", "FAIL b", "FAIL c"]),
    });
    let mut agent = ReflexionLoop::new(actor, judge, Budget::unlimited(10), 2);
    let (_out, stop) = agent.run("goal").await.expect("ok");
    assert_eq!(stop, StopReason::MaxIterations);
    assert_eq!(agent.memory.lessons.len(), 2);
}
