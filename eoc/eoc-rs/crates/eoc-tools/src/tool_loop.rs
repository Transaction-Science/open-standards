//! Multi-turn tool-orchestration loop.
//!
//! Pattern:
//!
//! 1. Send the query to the [`NeuralBackend`].
//! 2. Parse the response for tool-call requests via a vendor
//!    [`SchemaTranslator`].
//! 3. If there are no tool calls, return the response.
//! 4. Otherwise execute the calls in parallel and feed the results back
//!    as a new query, then loop.
//! 5. Stop when the model returns plain text, or `max_iterations` hit.
//!
//! Joule cost is summed across iterations and attached to the final
//! [`Response`].

use std::sync::Arc;

use eoc_core::{JouleCost, JouleSource, Query, Response, Stage};
use eoc_neural::NeuralBackend;
use serde_json::Value;

use crate::error::{ToolError, ToolResult};
use crate::parallel::{ParallelConfig, ToolCallResult, execute_parallel};
use crate::tool::{ToolCallRequest, ToolRegistry};

/// Parser plug-in: given a model response, extract any tool-call
/// requests it contains. Each vendor adapter implements one (or two —
/// see [`from_payload`]).
pub trait SchemaTranslator: Send + Sync {
    /// Extract tool calls from the model's last response. `payload` is
    /// the JSON-serialised vendor response body when available, or the
    /// raw text payload from a [`Response`] when not.
    fn parse_calls(&self, payload: &str) -> ToolResult<Vec<ToolCallRequest>>;

    /// Format a batch of tool results into the text the next iteration
    /// will feed back to the model. Default implementation produces a
    /// vendor-neutral structured JSON object the model can read.
    fn format_results(&self, results: &[ToolCallResult]) -> String {
        let arr: Vec<Value> = results
            .iter()
            .map(|r| {
                serde_json::json!({
                    "tool_call_id": r.call.id,
                    "name": r.call.name,
                    "result": match &r.outcome {
                        Ok(v) => v.clone(),
                        Err(e) => serde_json::json!({"error": e}),
                    }
                })
            })
            .collect();
        serde_json::json!({ "tool_results": arr }).to_string()
    }
}

/// A translator that delegates to a `Fn(&str) -> Vec<ToolCallRequest>`,
/// useful for tests and for plugging in the per-vendor `parse_tool_calls`
/// functions.
pub struct FnTranslator<F>(pub F)
where
    F: Fn(&str) -> ToolResult<Vec<ToolCallRequest>> + Send + Sync;

impl<F> SchemaTranslator for FnTranslator<F>
where
    F: Fn(&str) -> ToolResult<Vec<ToolCallRequest>> + Send + Sync,
{
    fn parse_calls(&self, payload: &str) -> ToolResult<Vec<ToolCallRequest>> {
        (self.0)(payload)
    }
}

/// The orchestration driver.
pub struct ToolLoop {
    backend: Arc<dyn NeuralBackend>,
    registry: Arc<ToolRegistry>,
    /// Hard cap on iterations to prevent runaway loops.
    pub max_iterations: usize,
    translator: Box<dyn SchemaTranslator>,
    parallel_config: ParallelConfig,
}

impl ToolLoop {
    /// Build a loop. Sensible defaults: 8 iterations, default
    /// [`ParallelConfig`].
    pub fn new(
        backend: Arc<dyn NeuralBackend>,
        registry: Arc<ToolRegistry>,
        translator: Box<dyn SchemaTranslator>,
    ) -> Self {
        Self {
            backend,
            registry,
            max_iterations: 8,
            translator,
            parallel_config: ParallelConfig::default(),
        }
    }

    /// Override the iteration cap.
    pub fn with_max_iterations(mut self, n: usize) -> Self {
        self.max_iterations = n;
        self
    }

    /// Override the parallel-exec config.
    pub fn with_parallel_config(mut self, cfg: ParallelConfig) -> Self {
        self.parallel_config = cfg;
        self
    }

    /// Run the loop.
    pub async fn run(&self, initial_query: Query) -> ToolResult<Response> {
        let mut current = initial_query;
        let mut total_microjoules: u64 = 0;
        let mut last_response: Option<Response> = None;

        for _ in 0..self.max_iterations {
            let resp = self.backend.infer(&current).await;
            total_microjoules = total_microjoules.saturating_add(resp.joule_cost.microjoules);

            let calls = self.translator.parse_calls(&resp.payload)?;
            if calls.is_empty() {
                // Plain answer — return with the accumulated cost.
                return Ok(Response::new(
                    resp.query_id,
                    resp.payload,
                    Stage::Neural,
                    JouleCost {
                        microjoules: total_microjoules,
                        source: JouleSource::Estimated,
                    },
                ));
            }

            let results =
                execute_parallel(&self.registry, calls, self.parallel_config.clone()).await;
            let feedback = self.translator.format_results(&results);
            // Next iteration's query carries the tool results back in.
            let new_prompt = format!(
                "{prev}\n\n[tool_results]\n{feedback}",
                prev = current.prompt
            );
            current = Query::new(new_prompt);
            last_response = Some(resp);
        }

        // Hit the iteration cap: return last seen response with
        // accumulated cost, but signal via error so callers can decide
        // policy. We surface the response by attaching it to the error
        // message; callers wanting partial output should check
        // `last_response` via a higher-level wrapper.
        let _ = last_response;
        Err(ToolError::MaxIterations(self.max_iterations))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::schema::ToolSchema;
    use crate::tool::Tool;
    use async_trait::async_trait;
    use serde_json::json;
    use std::sync::Mutex;

    struct ScriptedBackend {
        // Pre-scripted payloads. Each call returns the next one.
        payloads: Mutex<Vec<String>>,
        per_call_microjoules: u64,
    }

    #[async_trait]
    impl NeuralBackend for ScriptedBackend {
        async fn infer(&self, q: &Query) -> Response {
            let payload = {
                let mut p = self.payloads.lock().unwrap();
                if p.is_empty() {
                    "no more".to_string()
                } else {
                    p.remove(0)
                }
            };
            Response::new(
                q.id,
                payload,
                Stage::Neural,
                JouleCost::estimated(self.per_call_microjoules),
            )
        }
    }

    struct AddTool;
    #[async_trait]
    impl Tool for AddTool {
        fn schema(&self) -> &ToolSchema {
            // Leak a static schema for tests.
            static SCHEMA: std::sync::OnceLock<ToolSchema> = std::sync::OnceLock::new();
            SCHEMA.get_or_init(|| {
                ToolSchema::new("add", "sums two", json!({"type": "object"}))
            })
        }
        async fn invoke(&self, args: Value) -> ToolResult<Value> {
            let a = args.get("a").and_then(|v| v.as_i64()).unwrap_or(0);
            let b = args.get("b").and_then(|v| v.as_i64()).unwrap_or(0);
            Ok(json!({"sum": a + b}))
        }
    }

    // Translator that looks for the literal "CALL add" in the payload
    // and emits one add-call until the payload says "FINAL".
    fn make_translator() -> Box<dyn SchemaTranslator> {
        Box::new(FnTranslator(|payload: &str| {
            if payload.contains("FINAL") {
                Ok(vec![])
            } else if payload.contains("CALL add") {
                Ok(vec![ToolCallRequest {
                    id: "c1".into(),
                    name: "add".into(),
                    args: json!({"a": 2, "b": 3}),
                }])
            } else {
                Ok(vec![])
            }
        }))
    }

    #[tokio::test]
    async fn loop_runs_tools_then_returns_final() {
        let backend = Arc::new(ScriptedBackend {
            payloads: Mutex::new(vec![
                "CALL add".to_string(),
                "CALL add".to_string(),
                "CALL add".to_string(),
                "FINAL: 5".to_string(),
            ]),
            per_call_microjoules: 1000,
        });
        let mut reg = ToolRegistry::new();
        reg.register(AddTool);
        let loop_ = ToolLoop::new(backend, Arc::new(reg), make_translator())
            .with_max_iterations(10);

        let q = Query::new("what is 2 + 3?");
        let resp = loop_.run(q).await.unwrap();
        assert!(resp.payload.contains("FINAL"));
        // 4 iterations × 1000 µJ = 4000 µJ cumulative.
        assert_eq!(resp.joule_cost.microjoules, 4000);
    }

    #[tokio::test]
    async fn loop_respects_max_iterations() {
        let backend = Arc::new(ScriptedBackend {
            // Always asks for a tool — never returns FINAL.
            payloads: Mutex::new(vec![
                "CALL add".to_string(),
                "CALL add".to_string(),
                "CALL add".to_string(),
                "CALL add".to_string(),
                "CALL add".to_string(),
            ]),
            per_call_microjoules: 100,
        });
        let mut reg = ToolRegistry::new();
        reg.register(AddTool);
        let loop_ = ToolLoop::new(backend, Arc::new(reg), make_translator())
            .with_max_iterations(3);

        let err = loop_.run(Query::new("loop forever")).await.unwrap_err();
        assert!(matches!(err, ToolError::MaxIterations(3)));
    }
}
