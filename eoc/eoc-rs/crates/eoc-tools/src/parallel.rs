//! Bounded-concurrency parallel tool execution.
//!
//! When a model emits multiple tool calls in a single turn (Anthropic
//! supports up to 64; OpenAI supports an arbitrary count; Gemini in
//! parallel mode similarly), they should run concurrently rather than
//! serially. This module runs them through
//! `futures::stream::FuturesUnordered` with a semaphore-bounded
//! concurrency limit and a per-call timeout.

use std::sync::Arc;
use std::time::Duration;

use futures::StreamExt;
use futures::stream::FuturesUnordered;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio::sync::Semaphore;
use tokio::time::timeout;

use crate::error::ToolError;
use crate::tool::{ToolCallRequest, ToolRegistry};

/// Configuration for [`execute_parallel`].
#[derive(Debug, Clone)]
pub struct ParallelConfig {
    /// Maximum number of tool calls that may run concurrently.
    pub max_concurrency: usize,
    /// Per-call timeout.
    pub per_call_timeout: Duration,
}

impl Default for ParallelConfig {
    fn default() -> Self {
        Self {
            max_concurrency: 8,
            per_call_timeout: Duration::from_secs(30),
        }
    }
}

/// The outcome of a single tool call — kept correlated with the
/// originating [`ToolCallRequest`] so the loop can map results back to
/// `tool_use_id`s before feeding them into the next model turn.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCallResult {
    /// The originating call.
    pub call: ToolCallRequest,
    /// `Ok(output_json)` on success; `Err(string)` on failure
    /// (serialised so the result can be embedded back in the prompt).
    pub outcome: Result<Value, String>,
}

/// Execute a batch of tool calls in parallel.
pub async fn execute_parallel(
    registry: &ToolRegistry,
    calls: Vec<ToolCallRequest>,
    config: ParallelConfig,
) -> Vec<ToolCallResult> {
    let sem = Arc::new(Semaphore::new(config.max_concurrency.max(1)));
    let timeout_dur = config.per_call_timeout;

    let mut futs = FuturesUnordered::new();
    for call in calls {
        let sem = Arc::clone(&sem);
        let call_clone = call.clone();
        futs.push(async move {
            let _permit = match sem.acquire_owned().await {
                Ok(p) => p,
                Err(_) => {
                    return ToolCallResult {
                        call: call_clone,
                        outcome: Err("semaphore closed".to_string()),
                    };
                }
            };
            let res =
                timeout(timeout_dur, registry.dispatch(&call.name, call.args.clone())).await;
            let outcome = match res {
                Ok(Ok(v)) => Ok(v),
                Ok(Err(e)) => Err(e.to_string()),
                Err(_) => Err(ToolError::Timeout(call.name.clone()).to_string()),
            };
            ToolCallResult {
                call: call_clone,
                outcome,
            }
        });
    }

    let mut out = Vec::new();
    while let Some(r) = futs.next().await {
        out.push(r);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ToolSchema;
    use crate::tool::Tool;
    use async_trait::async_trait;
    use serde_json::json;
    use std::time::Instant;

    struct SlowTool {
        schema: ToolSchema,
        delay: Duration,
    }

    #[async_trait]
    impl Tool for SlowTool {
        fn schema(&self) -> &ToolSchema {
            &self.schema
        }
        async fn invoke(&self, args: Value) -> crate::error::ToolResult<Value> {
            tokio::time::sleep(self.delay).await;
            Ok(args)
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn parallel_completes_faster_than_serial() {
        let mut reg = ToolRegistry::new();
        reg.register(SlowTool {
            schema: ToolSchema::new("slow", "", json!({"type": "object"})),
            delay: Duration::from_millis(100),
        });
        let calls: Vec<ToolCallRequest> = (0..5)
            .map(|i| ToolCallRequest {
                id: format!("c{i}"),
                name: "slow".to_string(),
                args: json!({"i": i}),
            })
            .collect();

        let cfg = ParallelConfig {
            max_concurrency: 8,
            per_call_timeout: Duration::from_secs(5),
        };
        let start = Instant::now();
        let results = execute_parallel(&reg, calls, cfg).await;
        let elapsed = start.elapsed();
        assert_eq!(results.len(), 5);
        // 5 × 100ms serial would be 500ms; parallel should finish well
        // under 300ms even on a loaded runner.
        assert!(
            elapsed < Duration::from_millis(300),
            "elapsed {elapsed:?}"
        );
        for r in &results {
            assert!(r.outcome.is_ok());
        }
    }

    #[tokio::test]
    async fn per_call_timeout_fires() {
        let mut reg = ToolRegistry::new();
        reg.register(SlowTool {
            schema: ToolSchema::new("slow", "", json!({"type": "object"})),
            delay: Duration::from_millis(500),
        });
        let calls = vec![ToolCallRequest {
            id: "c0".into(),
            name: "slow".into(),
            args: json!({}),
        }];
        let cfg = ParallelConfig {
            max_concurrency: 1,
            per_call_timeout: Duration::from_millis(50),
        };
        let results = execute_parallel(&reg, calls, cfg).await;
        assert!(results[0].outcome.is_err());
    }
}
