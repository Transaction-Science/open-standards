//! # jouleclaw-mcp
//!
//! JouleClaw's MCP (Model Context Protocol) tool surface.
//!
//! ## Why this crate
//!
//! Standardising on MCP makes JouleClaw plug into the Claude Code /
//! Codex / Goose ecosystem without bespoke glue. But raw MCP measures
//! nothing about energy. This crate adds two things:
//!
//! 1. **Joule-metered tool calls.** Every dispatch routes through a
//!    [`MeteredTool`] wrapper that brackets the call with energy
//!    counter reads and pushes a `jouleclaw-prov::ToolTouch` onto the
//!    current cascade receipt.
//! 2. **The `joule-mcp` transport profile.** A capability-negotiated
//!    extension that allows two JouleClaw-aware endpoints to switch
//!    from JSON-RPC to length-prefixed CBOR for inner loops. Stays
//!    spec-compliant on the wire (any non-JouleClaw MCP client sees
//!    plain JSON-RPC); kicks in only when both sides advertise
//!    `x-jouleclaw/joule-mcp@1` in the handshake.
//!
//! ## What this crate is NOT
//!
//! - Not a re-implementation of MCP. The official Rust SDK is
//!   `modelcontextprotocol/rust-sdk` (`rmcp`). The recommended
//!   composition is to depend on `rmcp` downstream and wrap each
//!   server / client in `MeteredTool` from this crate. We don't
//!   pin `rmcp` here to keep `jouleclaw-mcp` reqwest- and
//!   tokio-feature-free.
//! - Not a transport. The CBOR profile defines envelope shape and
//!   negotiation, not the socket layer.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use async_trait::async_trait;
use jouleclaw_energy::{EnergyCounter, EnergyReading, Provenance};
use jouleclaw_prov::ToolTouch;
use serde::{Deserialize, Serialize};

/// The capability tag two JouleClaw-aware MCP endpoints advertise in
/// their handshake to opt into the binary-transport profile.
pub const JOULE_MCP_CAPABILITY: &str = "x-jouleclaw/joule-mcp@1";

/// Errors a metered MCP call can produce.
#[derive(Debug, thiserror::Error)]
pub enum McpError {
    /// The underlying tool returned an error.
    #[error("tool: {0}")]
    Tool(String),
    /// Failure reading the energy counter — the call still ran, but
    /// the receipt's joules_uj entry is `0` and provenance falls back
    /// to [`Provenance::Estimator`] with explicit drift band.
    #[error("energy counter: {0}")]
    EnergyCounter(String),
    /// Wire encoding failure (JSON or CBOR).
    #[error("encode: {0}")]
    Encode(String),
}

/// One pre-execution + post-execution energy reading pair, used to
/// price a single tool call.
#[derive(Debug, Clone)]
pub struct EnergyBracket {
    /// Counter reading taken just before the call dispatched.
    pub before: EnergyReading,
    /// Counter reading taken just after the call returned.
    pub after: EnergyReading,
    /// Provenance of the counter — load-bearing for the receipt.
    pub provenance: Provenance,
}

impl EnergyBracket {
    /// Microjoules consumed inside the bracket. Saturates on counter
    /// wrap rather than panicking.
    pub fn delta_uj(&self) -> u64 {
        self.after.uj.saturating_sub(self.before.uj)
    }
}

/// A tool wrapped with energy metering and receipt emission.
///
/// Implementations supply the inner tool dispatch; this trait drives
/// the bracket-and-account dance.
#[async_trait]
pub trait MeteredTool: Send + Sync {
    /// Stable tool identifier — `"mcp:filesystem/read"`,
    /// `"mcp:github/issue.create"`, etc. Appears verbatim in the
    /// receipt's `tools_touched` ledger.
    fn tool_id(&self) -> &str;

    /// Dispatch the inner tool. Implementations should be pure
    /// dispatch — no metering, no receipt mutation. This crate's
    /// [`dispatch_metered`] handles those.
    async fn dispatch(&self, request: &[u8]) -> Result<Vec<u8>, McpError>;
}

/// Drive a metered dispatch: pre-read the counter, call the tool,
/// post-read the counter, return the response bytes paired with a
/// [`ToolTouch`] ready to push into a `jouleclaw-prov::ReceiptBuilder`.
///
/// If the counter read fails, the response is still returned but the
/// `ToolTouch.joules_uj = 0` and `energy_provenance = Estimator` so
/// the receipt remains shape-correct (and the worst-counter floor
/// downgrades, alerting the auditor that this call wasn't honestly
/// measured).
pub async fn dispatch_metered<C: EnergyCounter + ?Sized>(
    tool: &dyn MeteredTool,
    counter: &C,
    request: &[u8],
) -> Result<(Vec<u8>, ToolTouch), McpError> {
    let before = counter.read().ok();
    let resp = tool.dispatch(request).await?;
    let after = counter.read().ok();

    let (joules_uj, provenance) = match (before, after) {
        (Some(a), Some(b)) => (b.uj.saturating_sub(a.uj), counter.provenance()),
        // Counter unavailable — receipt records zero cost and flags
        // Estimator so downstream auditors discount the entry.
        _ => (0, Provenance::Estimator),
    };

    let touch = ToolTouch {
        tool_id: tool.tool_id().to_string(),
        joules_uj,
        energy_provenance: provenance,
    };
    Ok((resp, touch))
}

/// Wire encoding for the joule-mcp transport profile.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum WireEncoding {
    /// Plain MCP JSON-RPC over stdio/HTTP/SSE/WebSocket. Always
    /// available; required for non-JouleClaw clients.
    JsonRpc,
    /// Length-prefixed CBOR, advertised under
    /// [`JOULE_MCP_CAPABILITY`]. Roughly 30–50% lower per-call
    /// serialisation tax than JSON-RPC at the cost of opacity to
    /// generic tools.
    Cbor,
}

/// Choose the wire encoding given each side's advertised capabilities.
///
/// Returns [`WireEncoding::Cbor`] iff both endpoints advertise
/// [`JOULE_MCP_CAPABILITY`], otherwise [`WireEncoding::JsonRpc`].
pub fn negotiate(local_caps: &[String], remote_caps: &[String]) -> WireEncoding {
    let cap = JOULE_MCP_CAPABILITY.to_string();
    if local_caps.contains(&cap) && remote_caps.contains(&cap) {
        WireEncoding::Cbor
    } else {
        WireEncoding::JsonRpc
    }
}

/// Encode a value under the negotiated wire encoding.
pub fn encode<T: Serialize>(value: &T, wire: WireEncoding) -> Result<Vec<u8>, McpError> {
    match wire {
        WireEncoding::JsonRpc => {
            serde_json::to_vec(value).map_err(|e| McpError::Encode(e.to_string()))
        }
        WireEncoding::Cbor => {
            serde_cbor::to_vec(value).map_err(|e| McpError::Encode(e.to_string()))
        }
    }
}

/// Decode a value from the negotiated wire encoding.
pub fn decode<T: for<'de> Deserialize<'de>>(bytes: &[u8], wire: WireEncoding) -> Result<T, McpError> {
    match wire {
        WireEncoding::JsonRpc => {
            serde_json::from_slice(bytes).map_err(|e| McpError::Encode(e.to_string()))
        }
        WireEncoding::Cbor => {
            serde_cbor::from_slice(bytes).map_err(|e| McpError::Encode(e.to_string()))
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// MCP Apps + async Tasks
// ─────────────────────────────────────────────────────────────────────

/// A tool's response shape — the MCP Apps extension over plain bytes.
///
/// Existing [`MeteredTool`] consumers return `Vec<u8>` (free-form). The
/// `ToolResponse` envelope lets a tool emit *typed* responses the host
/// can route: plain text, structured JSON, or a `joule-ui` widget tree.
/// The widget tree is carried as `serde_json::Value` so this crate has
/// no path dependency on `joule-ui-core` — the wire shape is the
/// contract, the typed parse is the consumer's responsibility.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case", tag = "kind", content = "value")]
pub enum ToolResponse {
    /// A plain text response.
    Text(String),
    /// A structured response — JSON of any shape the tool agreed on
    /// out-of-band with its caller.
    Structured(serde_json::Value),
    /// A `joule-ui` widget tree (SEP-1865 / A2UI shape). The host
    /// validates and renders; this crate only round-trips the wire form.
    UiWidget(serde_json::Value),
}

/// Opaque identifier for an async [`Task`]. Stable within a store; the
/// wire form is a plain string so it round-trips through any transport.
#[derive(
    Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct TaskId(pub String);

impl std::fmt::Display for TaskId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// The state a task is in.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize,
)]
#[serde(rename_all = "snake_case")]
pub enum TaskStatusKind {
    /// Created, not yet picked up by a worker.
    #[default]
    Pending,
    /// A worker is executing.
    Running,
    /// The worker finished and stored a result.
    Completed,
    /// The worker failed; `error` carries the reason.
    Failed,
}

/// A point-in-time view of a task — what a caller polling
/// [`InMemoryTaskStore::snapshot`] receives.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskSnapshot {
    pub id: TaskId,
    /// The MCP tool that owns this task (e.g. `"mcp:long-running"`).
    pub tool_id: String,
    pub status: TaskStatusKind,
    /// Caller-supplied integration hint — typically a stable reference
    /// to a `jouleclaw-graph` node (`"graph:node-3"`) the consumer
    /// uses to map task completion back into a run. Opaque to this
    /// crate; round-trips through the wire form unchanged.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub slot: Option<String>,
    /// Set on transition to [`TaskStatusKind::Completed`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<ToolResponse>,
    /// Set on transition to [`TaskStatusKind::Failed`].
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

/// Errors from the async-task lifecycle.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum TaskError {
    #[error("unknown task id {0}")]
    Unknown(TaskId),
    #[error("task {0} is not in progress; cannot transition")]
    NotInProgress(TaskId),
}

/// The persistence contract for async tasks. The "call-now / fetch-later"
/// primitive every long-running MCP tool now exposes: the tool
/// `create`s a task and returns its id immediately; a worker
/// `mark_running` then `mark_completed`/`mark_failed`; the caller polls
/// `snapshot` until status is terminal.
pub trait TaskStore: Send {
    /// Create a new task. Returns its id. `slot` is an opaque
    /// integration hint for the caller's run graph.
    fn create(&mut self, tool_id: impl Into<String>, slot: Option<&str>) -> TaskId
    where
        Self: Sized;
    fn mark_running(&mut self, id: &TaskId) -> Result<(), TaskError>;
    fn mark_completed(&mut self, id: &TaskId, result: ToolResponse) -> Result<(), TaskError>;
    fn mark_failed(&mut self, id: &TaskId, reason: impl Into<String>) -> Result<(), TaskError>
    where
        Self: Sized;
    fn snapshot(&self, id: &TaskId) -> Option<TaskSnapshot>;
    fn list(&self) -> Vec<TaskId>;
}

/// In-memory reference [`TaskStore`]. Ids are formatted `task-N` where
/// N is a per-store monotonic counter — deterministic for tests, and
/// the list ordering is stable by insertion.
#[derive(Debug, Default)]
pub struct InMemoryTaskStore {
    tasks: std::collections::BTreeMap<TaskId, TaskSnapshot>,
    counter: u64,
}

impl InMemoryTaskStore {
    pub fn new() -> Self {
        Self::default()
    }
}

impl TaskStore for InMemoryTaskStore {
    fn create(&mut self, tool_id: impl Into<String>, slot: Option<&str>) -> TaskId
    where
        Self: Sized,
    {
        self.counter += 1;
        let id = TaskId(format!("task-{}", self.counter));
        let snap = TaskSnapshot {
            id: id.clone(),
            tool_id: tool_id.into(),
            status: TaskStatusKind::Pending,
            slot: slot.map(String::from),
            result: None,
            error: None,
        };
        self.tasks.insert(id.clone(), snap);
        id
    }

    fn mark_running(&mut self, id: &TaskId) -> Result<(), TaskError> {
        let Some(t) = self.tasks.get_mut(id) else {
            return Err(TaskError::Unknown(id.clone()));
        };
        if !matches!(t.status, TaskStatusKind::Pending) {
            return Err(TaskError::NotInProgress(id.clone()));
        }
        t.status = TaskStatusKind::Running;
        Ok(())
    }

    fn mark_completed(&mut self, id: &TaskId, result: ToolResponse) -> Result<(), TaskError> {
        let Some(t) = self.tasks.get_mut(id) else {
            return Err(TaskError::Unknown(id.clone()));
        };
        if !matches!(t.status, TaskStatusKind::Pending | TaskStatusKind::Running) {
            return Err(TaskError::NotInProgress(id.clone()));
        }
        t.status = TaskStatusKind::Completed;
        t.result = Some(result);
        Ok(())
    }

    fn mark_failed(&mut self, id: &TaskId, reason: impl Into<String>) -> Result<(), TaskError>
    where
        Self: Sized,
    {
        let Some(t) = self.tasks.get_mut(id) else {
            return Err(TaskError::Unknown(id.clone()));
        };
        if !matches!(t.status, TaskStatusKind::Pending | TaskStatusKind::Running) {
            return Err(TaskError::NotInProgress(id.clone()));
        }
        t.status = TaskStatusKind::Failed;
        t.error = Some(reason.into());
        Ok(())
    }

    fn snapshot(&self, id: &TaskId) -> Option<TaskSnapshot> {
        self.tasks.get(id).cloned()
    }

    fn list(&self) -> Vec<TaskId> {
        self.tasks.keys().cloned().collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_energy::{EnergyDomain, EnergyError};

    struct EchoTool;
    #[async_trait]
    impl MeteredTool for EchoTool {
        fn tool_id(&self) -> &str { "test:echo" }
        async fn dispatch(&self, request: &[u8]) -> Result<Vec<u8>, McpError> {
            Ok(request.to_vec())
        }
    }

    /// A counter that emits ascending readings 0, 100, 200, … so the
    /// delta between two consecutive reads is exactly 100 μJ.
    struct StepCounter {
        cell: std::sync::atomic::AtomicU64,
        prov: Provenance,
    }
    impl EnergyCounter for StepCounter {
        fn domain(&self) -> EnergyDomain { EnergyDomain::SocTotal }
        fn provenance(&self) -> Provenance { self.prov }
        fn resolution_uj(&self) -> u64 { 1 }
        fn min_window_ns(&self) -> u64 { 1_000 }
        fn read(&self) -> Result<EnergyReading, EnergyError> {
            let prev = self.cell.fetch_add(100, std::sync::atomic::Ordering::Relaxed);
            Ok(EnergyReading {
                uj: prev,
                timestamp_ns: 0,
                domain: EnergyDomain::SocTotal,
                provenance: self.prov,
            })
        }
    }

    #[test]
    fn negotiate_falls_back_to_jsonrpc() {
        let local = vec![JOULE_MCP_CAPABILITY.to_string()];
        let remote: Vec<String> = vec![];
        assert_eq!(negotiate(&local, &remote), WireEncoding::JsonRpc);
    }

    #[test]
    fn negotiate_picks_cbor_when_both_advertise() {
        let cap = JOULE_MCP_CAPABILITY.to_string();
        assert_eq!(negotiate(&[cap.clone()], &[cap]), WireEncoding::Cbor);
    }

    #[test]
    fn encode_decode_round_trip_jsonrpc() {
        let v = serde_json::json!({"hello": "world"});
        let bytes = encode(&v, WireEncoding::JsonRpc).expect("encode");
        let back: serde_json::Value = decode(&bytes, WireEncoding::JsonRpc).expect("decode");
        assert_eq!(back, v);
    }

    #[test]
    fn encode_decode_round_trip_cbor() {
        let v = serde_json::json!({"hello": "world"});
        let bytes = encode(&v, WireEncoding::Cbor).expect("encode");
        let back: serde_json::Value = decode(&bytes, WireEncoding::Cbor).expect("decode");
        assert_eq!(back, v);
    }

    #[tokio::test]
    async fn dispatch_metered_emits_tool_touch() {
        let tool = EchoTool;
        let counter = StepCounter {
            cell: std::sync::atomic::AtomicU64::new(0),
            prov: Provenance::HwShunt,
        };
        let (resp, touch) = dispatch_metered(&tool, &counter, b"ping").await.expect("dispatch");
        assert_eq!(resp, b"ping");
        assert_eq!(touch.tool_id, "test:echo");
        assert_eq!(touch.joules_uj, 100);
        assert_eq!(touch.energy_provenance, Provenance::HwShunt);
    }

    // ─── MCP Apps + async Tasks ───────────────────────────────────

    #[test]
    fn tool_response_round_trips_through_json() {
        let t = ToolResponse::Text("hello".into());
        let j = serde_json::to_string(&t).unwrap();
        let back: ToolResponse = serde_json::from_str(&j).unwrap();
        assert_eq!(back, t);

        let ui = ToolResponse::UiWidget(serde_json::json!({
            "name": "Card",
            "props": { "title": { "type": "text", "value": "hi" } }
        }));
        let j = serde_json::to_string(&ui).unwrap();
        let back: ToolResponse = serde_json::from_str(&j).unwrap();
        assert_eq!(back, ui);
    }

    #[test]
    fn task_store_create_returns_pending_task() {
        let mut store = InMemoryTaskStore::new();
        let id = store.create("mcp:long-running", Some("graph:node-3"));
        let snap = store.snapshot(&id).expect("snapshot");
        assert_eq!(snap.status, TaskStatusKind::Pending);
        assert_eq!(snap.tool_id, "mcp:long-running");
        assert_eq!(snap.slot.as_deref(), Some("graph:node-3"));
        assert!(snap.result.is_none());
    }

    #[test]
    fn task_store_marks_running_then_completed() {
        let mut store = InMemoryTaskStore::new();
        let id = store.create("mcp:long-running", None);
        store.mark_running(&id).unwrap();
        assert_eq!(
            store.snapshot(&id).unwrap().status,
            TaskStatusKind::Running
        );
        store
            .mark_completed(&id, ToolResponse::Text("done".into()))
            .unwrap();
        let snap = store.snapshot(&id).unwrap();
        assert_eq!(snap.status, TaskStatusKind::Completed);
        match snap.result {
            Some(ToolResponse::Text(t)) => assert_eq!(t, "done"),
            other => panic!("expected Text result, got {other:?}"),
        }
    }

    #[test]
    fn task_store_marks_failed_with_reason() {
        let mut store = InMemoryTaskStore::new();
        let id = store.create("mcp:long-running", None);
        store.mark_failed(&id, "upstream gateway timeout").unwrap();
        let snap = store.snapshot(&id).unwrap();
        assert_eq!(snap.status, TaskStatusKind::Failed);
        assert_eq!(snap.error.as_deref(), Some("upstream gateway timeout"));
    }

    #[test]
    fn task_store_rejects_invalid_transitions() {
        let mut store = InMemoryTaskStore::new();
        let id = store.create("mcp:t", None);
        store.mark_running(&id).unwrap();
        store.mark_completed(&id, ToolResponse::Text("x".into())).unwrap();
        let err = store
            .mark_completed(&id, ToolResponse::Text("y".into()))
            .unwrap_err();
        assert!(matches!(err, TaskError::NotInProgress(_)));
    }

    #[test]
    fn task_store_returns_none_for_unknown_id() {
        let store = InMemoryTaskStore::new();
        assert!(store.snapshot(&TaskId("missing".into())).is_none());
    }

    #[test]
    fn task_id_is_monotonic_within_store() {
        let mut store = InMemoryTaskStore::new();
        let a = store.create("t", None);
        let b = store.create("t", None);
        let c = store.create("t", None);
        assert_ne!(a, b);
        assert_ne!(b, c);
        assert_eq!(store.list().len(), 3);
    }

    #[test]
    fn task_snapshot_round_trips_through_json() {
        let mut store = InMemoryTaskStore::new();
        let id = store.create("mcp:demo", Some("slot-a"));
        store
            .mark_completed(
                &id,
                ToolResponse::UiWidget(serde_json::json!({ "name": "Text" })),
            )
            .unwrap();
        let snap = store.snapshot(&id).unwrap();
        let bytes = serde_json::to_vec(&snap).unwrap();
        let back: TaskSnapshot = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back, snap);
    }
}
