//! Typed facade over Minigraf.
//!
//! Minigraf is an EAV (Entity-Attribute-Value) fact store with a
//! Datalog query language. The OpenPay graph is property-graph
//! shaped — typed vertices, typed directed edges, JSON properties on
//! both — so this facade maps the property-graph model onto EAV
//! facts:
//!
//! ```text
//! Vertex(id, type)                  → [id :_type type]
//! VertexProperty(id, name, value)   → [id name value]
//!
//! Edge(from, type, to)              → mint edge_id; assert
//!                                       [edge_id :_edge/from from]
//!                                       [edge_id :_edge/to to]
//!                                       [edge_id :_edge/type type]
//! EdgeProperty(edge_id, name, val)  → [edge_id name value]
//! ```
//!
//! Internal attributes are prefixed with `_` so user property names
//! can be any non-underscore-leading keyword without collision.
//!
//! ## Persistence
//!
//! [`new_in_memory`](GraphHandle::new_in_memory) gives an ephemeral
//! store (tests, single-process kiosks). [`new_persistent`](GraphHandle::new_persistent)
//! opens a single `.graph` file — the "SQLite of graphs" model
//! Minigraf is shaped around. Closing the handle flushes; reopening
//! the same path recovers every fact and lets the operator pick up
//! exactly where they left off.

use std::sync::Arc;

use minigraf::{BindValue, Minigraf, Value as MgValue};
use serde_json::Value;
use uuid::Uuid;

use crate::error::{Error, Result};

// ============================================================
// Vertex / edge type constants
// ============================================================

/// Ledger vertex types.
pub mod vtypes {
    /// A `ledger_account`.
    pub const LEDGER_ACCOUNT: &str = "ledger_account";
    /// A `ledger_tx`.
    pub const LEDGER_TX: &str = "ledger_tx";
    /// A `ledger_ledger`.
    pub const LEDGER_LEDGER: &str = "ledger_ledger";
    /// A `webhook_event`.
    pub const WEBHOOK_EVENT: &str = "webhook_event";
    /// A `webhook_endpoint`.
    pub const WEBHOOK_ENDPOINT: &str = "webhook_endpoint";
    /// A `webhook_attempt`.
    pub const WEBHOOK_ATTEMPT: &str = "webhook_attempt";
    /// A `statement_line` ingested from a reconciliation source.
    pub const STATEMENT_LINE: &str = "statement_line";
    /// A `reconciliation_task` — one open discrepancy for an operator.
    pub const RECONCILIATION_TASK: &str = "reconciliation_task";
    /// A `rail_attempt` — one rail-driver attempt recorded for
    /// routing-signal purposes. See `crate::rail_telemetry`.
    pub const RAIL_ATTEMPT: &str = "rail_attempt";
    /// A `ledger_checkpoint` — operator-named bookmark into the
    /// bi-temporal history. Holds `name` (String) and `tx_count`
    /// (u64) properties.
    pub const LEDGER_CHECKPOINT: &str = "ledger_checkpoint";
    /// A `refund` — one refund workflow record. Vertex id matches
    /// `RefundId`'s UUID. Properties carry the full `Refund` JSON
    /// in `state`, plus indexed fields `external_id`,
    /// `original_tx_id`, `status_code`.
    pub const REFUND: &str = "refund";
    /// A `dispute` — one chargeback / inquiry / representment
    /// record. Mirror shape of `REFUND`.
    pub const DISPUTE: &str = "dispute";
    /// A `settlement_batch` — one payout-batch lifecycle.
    pub const SETTLEMENT_BATCH: &str = "settlement_batch";
    /// An `idempotency_record` — orchestrator idempotency-key
    /// cache entry. Vertex id is the UUIDv5 over the key string;
    /// properties carry `key`, `body_signature`, `state` (the
    /// cached outcome).
    pub const IDEMPOTENCY_RECORD: &str = "idempotency_record";
    /// A `subscription` — one customer's recurring-billing record.
    /// Vertex id matches `SubscriptionId`'s UUID. Properties carry
    /// full `Subscription` JSON in `state`, plus indexed
    /// `external_id`, `customer_ref`, `status_code`,
    /// `current_period_end`.
    pub const SUBSCRIPTION: &str = "subscription";
}

/// Edge type constants.
pub mod etypes {
    /// `ledger_tx` --debit--> `ledger_account`.
    pub const LEDGER_DEBIT: &str = "ledger_debit";
    /// `ledger_tx` --credit--> `ledger_account`.
    pub const LEDGER_CREDIT: &str = "ledger_credit";
    /// `ledger_tx` --in--> `ledger_ledger`, or
    /// `ledger_account` --in--> `ledger_ledger`.
    pub const LEDGER_IN_LEDGER: &str = "ledger_in_ledger";
    /// `ledger_tx` --reverses--> `ledger_tx` (the original).
    pub const LEDGER_REVERSES: &str = "ledger_reverses";
    /// `webhook_event` --delivers--> `webhook_attempt`.
    pub const WEBHOOK_DELIVERS: &str = "webhook_delivers";
    /// `webhook_attempt` --to--> `webhook_endpoint`.
    pub const WEBHOOK_TO: &str = "webhook_to";
    /// `statement_line` --reconciles--> `ledger_tx` (matched line).
    pub const RECONCILES: &str = "reconciles";
    /// `reconciliation_task` --about--> `statement_line` | `ledger_tx`.
    pub const TASK_ABOUT: &str = "task_about";
    /// `refund` --refunds--> `ledger_tx` (the original transaction).
    pub const REFUND_REFUNDS: &str = "refund_refunds";
    /// `dispute` --disputes--> `ledger_tx`.
    pub const DISPUTE_DISPUTES: &str = "dispute_disputes";
    /// `settlement_batch` --includes--> `ledger_tx` (each posted tx
    /// that's part of the batch).
    pub const BATCH_INCLUDES: &str = "batch_includes";
}

// ============================================================
// Local Vertex / Edge value types
// ============================================================

/// A typed vertex as returned by [`GraphHandle::get_vertex`] and
/// related queries.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Vertex {
    /// Vertex id (matches the caller-supplied id at creation).
    pub id: Uuid,
    /// Vertex type, e.g. [`vtypes::LEDGER_TX`].
    pub t: String,
}

/// A typed directed edge. Edges have their own opaque id so they can
/// carry properties.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Edge {
    /// Internal edge id (used to attach properties).
    pub id: Uuid,
    /// Source vertex id.
    pub from: Uuid,
    /// Edge type, e.g. [`etypes::LEDGER_DEBIT`].
    pub t: String,
    /// Destination vertex id.
    pub to: Uuid,
}

// ============================================================
// GraphHandle
// ============================================================

/// Thread-safe handle to a graph store.
///
/// Internally wraps an `Arc<Minigraf>` so it can be cheaply cloned for
/// multi-threaded callers. The backend is chosen at construction:
/// [`new_in_memory`](Self::new_in_memory) for an ephemeral store, or
/// [`new_persistent`](Self::new_persistent) for a single-file
/// `.graph` database that survives restarts.
#[derive(Clone)]
pub struct GraphHandle {
    inner: Arc<Minigraf>,
}

impl std::fmt::Debug for GraphHandle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // The Minigraf instance itself doesn't surface a useful Debug;
        // logging the tx counter is the most informative thing we can
        // produce without forcing a query.
        f.debug_struct("GraphHandle")
            .field("tx_count", &self.inner.current_tx_count())
            .finish()
    }
}

impl Default for GraphHandle {
    fn default() -> Self {
        Self::new_in_memory()
    }
}

impl GraphHandle {
    /// Construct backed by an in-memory Minigraf instance.
    ///
    /// Drop the handle and the data is gone — appropriate for tests
    /// and for single-process embedded deployments that rebuild graph
    /// state on every restart from a system-of-record store.
    ///
    /// # Panics
    /// Panics only if Minigraf itself fails to initialize an
    /// in-memory database, which it does not under any documented
    /// condition.
    #[must_use]
    pub fn new_in_memory() -> Self {
        Self {
            inner: Arc::new(
                Minigraf::in_memory()
                    .expect("Minigraf::in_memory never fails per documented contract"),
            ),
        }
    }

    /// Construct backed by a single-file `.graph` database at `path`.
    ///
    /// Opens (or creates) the file. Subsequent constructions pointed
    /// at the same path see the previously-written facts — the graph
    /// is the system of record across restarts.
    ///
    /// # Errors
    /// [`Error::Backend`] if Minigraf can't open the path (permission
    /// denied, corrupt file, ...).
    pub fn new_persistent(path: impl AsRef<std::path::Path>) -> Result<Self> {
        let db = Minigraf::open(path).map_err(|e| Error::Backend(e.to_string()))?;
        Ok(Self {
            inner: Arc::new(db),
        })
    }

    /// Borrow the underlying Minigraf instance. **Stability note:**
    /// Minigraf's API is not part of OpenPay's stability surface.
    #[must_use]
    pub fn raw(&self) -> &Minigraf {
        &self.inner
    }

    /// Flush in-memory facts to the durable file and truncate the
    /// WAL sidecar.
    ///
    /// **Scope.** This is Minigraf's `checkpoint` operation —
    /// consolidates the write-ahead log into the main `.graph`
    /// file so reopen is faster and the operator's disk footprint
    /// is what they expect. It does **not** purge retracted facts
    /// from history; Minigraf preserves the full bi-temporal log
    /// by design (time-travel queries depend on it). Operators who
    /// genuinely need historical pruning ship a custom store that
    /// implements the trait surface against a substrate that
    /// supports it.
    ///
    /// In-memory databases: no-op (returns `Ok(())`).
    ///
    /// # Errors
    /// [`Error::Backend`] on I/O failure during checkpoint.
    pub fn compact(&self) -> Result<()> {
        self.inner
            .checkpoint()
            .map_err(|e| Error::Backend(format!("minigraf checkpoint: {e}")))
    }

    /// The current monotonic transaction counter.
    ///
    /// Each successful `transact` / `retract` advances this by
    /// exactly 1, regardless of how many facts the batch carries.
    /// Snapshot it right after a write (`let snap = h.tx_count();`)
    /// to get a stable reference point for later time-travel
    /// queries (`:as-of snap`).
    ///
    /// This is the value [`LedgerHistory`](op_ledger::LedgerHistory)
    /// queries against.
    #[must_use]
    pub fn tx_count(&self) -> u64 {
        self.inner.current_tx_count()
    }

    // --------------------------------------------------------
    // Vertex create / read
    // --------------------------------------------------------

    /// Create a typed vertex with a caller-supplied id.
    pub fn create_vertex(&self, vertex_type: &str, id: Uuid) -> Result<()> {
        let script = format!(
            r#"(transact [[#uuid "{id}" :_type "{vt}"]])"#,
            id = id,
            vt = escape_str(vertex_type),
        );
        self.execute(&script).map(|_| ())
    }

    /// Fetch a vertex by id. Returns `None` if no vertex exists.
    pub fn get_vertex(&self, id: Uuid) -> Result<Option<Vertex>> {
        let script = format!(
            r#"(query [:find ?t :where [#uuid "{id}" :_type ?t]])"#,
            id = id
        );
        let result = self.execute(&script)?;
        let row = result.rows.into_iter().next();
        Ok(row.map(|r| Vertex {
            id,
            t: match r.into_iter().next() {
                Some(MgValue::String(s)) => s,
                _ => String::new(),
            },
        }))
    }

    /// Fetch a vertex and verify its type.
    ///
    /// # Errors
    /// [`Error::VertexNotFound`] / [`Error::VertexTypeMismatch`].
    pub fn get_typed_vertex(&self, id: Uuid, expected_type: &str) -> Result<Vertex> {
        let v = self.get_vertex(id)?.ok_or_else(|| Error::VertexNotFound {
            vertex_type: expected_type.to_owned(),
            id: id.to_string(),
        })?;
        if v.t != expected_type {
            return Err(Error::VertexTypeMismatch {
                id: id.to_string(),
                expected: expected_type.to_owned(),
                actual: v.t,
            });
        }
        Ok(v)
    }

    /// Does a vertex with this id exist?
    pub fn vertex_exists(&self, id: Uuid) -> Result<bool> {
        Ok(self.get_vertex(id)?.is_some())
    }

    // --------------------------------------------------------
    // Edge create / list
    // --------------------------------------------------------

    /// Create a directed, typed edge. The returned [`Edge`] carries
    /// the minted edge id so callers can attach properties.
    pub fn create_edge(&self, from: Uuid, edge_type: &str, to: Uuid) -> Result<Edge> {
        let edge_id = Uuid::new_v4();
        let script = format!(
            r#"(transact [[#uuid "{eid}" :_edge/from #uuid "{from}"]
                          [#uuid "{eid}" :_edge/to #uuid "{to}"]
                          [#uuid "{eid}" :_edge/type "{et}"]])"#,
            eid = edge_id,
            from = from,
            to = to,
            et = escape_str(edge_type),
        );
        self.execute(&script)?;
        Ok(Edge {
            id: edge_id,
            from,
            t: edge_type.to_owned(),
            to,
        })
    }

    /// List outbound edges of a given type from a vertex.
    pub fn out_edges(&self, from: Uuid, edge_type: &str) -> Result<Vec<Edge>> {
        let script = format!(
            r#"(query [:find ?eid ?to
                       :where [?eid :_edge/from #uuid "{from}"]
                              [?eid :_edge/type "{et}"]
                              [?eid :_edge/to ?to]])"#,
            from = from,
            et = escape_str(edge_type),
        );
        let result = self.execute(&script)?;
        let mut edges = Vec::with_capacity(result.rows.len());
        for row in result.rows {
            let mut it = row.into_iter();
            let eid = expect_ref(it.next())?;
            let to = expect_ref(it.next())?;
            edges.push(Edge {
                id: eid,
                from,
                t: edge_type.to_owned(),
                to,
            });
        }
        Ok(edges)
    }

    /// List inbound edges of a given type to a vertex.
    pub fn in_edges(&self, to: Uuid, edge_type: &str) -> Result<Vec<Edge>> {
        let script = format!(
            r#"(query [:find ?eid ?from
                       :where [?eid :_edge/to #uuid "{to}"]
                              [?eid :_edge/type "{et}"]
                              [?eid :_edge/from ?from]])"#,
            to = to,
            et = escape_str(edge_type),
        );
        let result = self.execute(&script)?;
        let mut edges = Vec::with_capacity(result.rows.len());
        for row in result.rows {
            let mut it = row.into_iter();
            let eid = expect_ref(it.next())?;
            let from = expect_ref(it.next())?;
            edges.push(Edge {
                id: eid,
                from,
                t: edge_type.to_owned(),
                to,
            });
        }
        Ok(edges)
    }

    // --------------------------------------------------------
    // Properties
    // --------------------------------------------------------

    /// Set a property on a vertex.
    ///
    /// Property names must be valid Datalog attribute keywords
    /// (alphanumeric, dash, underscore, dot, slash) and must not
    /// start with an underscore (`_` is reserved for internal
    /// schema attributes like `_type`, `_edge/from`).
    ///
    /// Minigraf is bi-temporal: every assert is appended to history.
    /// To give scalar properties "current value" semantics, we
    /// retract the old `(entity, attr)` fact before asserting the
    /// new one. The historical trail is preserved (queryable via
    /// `:as-of`); only the *current* view reflects the latest set.
    pub fn set_vertex_property(&self, vertex_id: Uuid, name: &str, value: Value) -> Result<()> {
        self.set_entity_property(vertex_id, name, value)
    }

    /// Set a property on a directed edge. The edge must already
    /// exist (its id is carried in the `Edge` struct returned by
    /// [`create_edge`](Self::create_edge)). Same retract-then-assert
    /// semantics as [`set_vertex_property`](Self::set_vertex_property).
    pub fn set_edge_property(&self, edge: &Edge, name: &str, value: Value) -> Result<()> {
        self.set_entity_property(edge.id, name, value)
    }

    /// Shared retract-then-assert implementation for scalar
    /// properties on any entity (vertex or edge).
    fn set_entity_property(&self, entity_id: Uuid, name: &str, value: Value) -> Result<()> {
        validate_user_attr(name)?;
        // 1. Find the current value(s) for this (entity, attr) so we
        //    can retract them. There should be at most one for a
        //    scalar, but retracting all defends against any earlier
        //    multi-assert.
        let lookup = format!(
            r#"(query [:find ?v :where [#uuid "{id}" :{attr} ?v]])"#,
            id = entity_id,
            attr = name,
        );
        let current = self.execute(&lookup)?;
        for row in current.rows {
            if let Some(old) = row.into_iter().next() {
                let lit = mg_value_to_literal(&old);
                let script = format!(
                    r#"(retract [[#uuid "{id}" :{attr} {val}]])"#,
                    id = entity_id,
                    attr = name,
                    val = lit,
                );
                self.execute(&script)?;
            }
        }
        // 2. Assert the new value.
        let v = json_to_mg_literal(&value);
        let script = format!(
            r#"(transact [[#uuid "{id}" :{attr} {val}]])"#,
            id = entity_id,
            attr = name,
            val = v,
        );
        self.execute(&script).map(|_| ())
    }

    /// Fetch all user-visible properties for a vertex as a
    /// `serde_json::Map`. Internal `_*` attributes (`_type`,
    /// `_edge/*`) are filtered out.
    pub fn get_vertex_properties(&self, vertex_id: Uuid) -> Result<serde_json::Map<String, Value>> {
        let script = format!(
            r#"(query [:find ?attr ?val :where [#uuid "{id}" ?attr ?val]])"#,
            id = vertex_id
        );
        let result = self.execute(&script)?;
        let mut map = serde_json::Map::new();
        for row in result.rows {
            let mut it = row.into_iter();
            // Minigraf returns keywords with their leading colon
            // (e.g. `Keyword(":name")`); strip it so callers use bare
            // attribute names. Internal `_*` attrs (`:_type`,
            // `:_edge/*`) are filtered out as they're schema, not
            // user-visible properties.
            let attr = match it.next() {
                Some(MgValue::Keyword(k)) => k.trim_start_matches(':').to_owned(),
                Some(MgValue::String(s)) => s,
                _ => continue,
            };
            if attr.starts_with('_') {
                continue;
            }
            if let Some(val) = it.next() {
                map.insert(attr, mg_to_json(val));
            }
        }
        Ok(map)
    }

    /// Fetch all user-visible properties for a specific edge.
    pub fn get_edge_properties(&self, edge: &Edge) -> Result<serde_json::Map<String, Value>> {
        self.get_vertex_properties(edge.id)
    }

    // --------------------------------------------------------
    // Diagnostics
    // --------------------------------------------------------

    /// Number of vertices total (diagnostic only). A vertex is
    /// anything that has a `:_type` fact.
    pub fn vertex_count(&self) -> Result<u64> {
        let result = self.execute("(query [:find ?e :where [?e :_type _]])")?;
        Ok(result.rows.len() as u64)
    }

    /// Number of edges total (diagnostic only). An edge is anything
    /// that has a `:_edge/type` fact.
    pub fn edge_count(&self) -> Result<u64> {
        let result = self.execute("(query [:find ?e :where [?e :_edge/type _]])")?;
        Ok(result.rows.len() as u64)
    }

    // --------------------------------------------------------
    // Time-travel reads (`:as-of N`)
    //
    // Each `_at` method parallels its present-time sibling but
    // appends `:as-of <tx_count>` to the underlying Datalog query.
    // Returns the slice of the graph that was visible at that
    // transaction counter — facts retracted after that point are
    // resurrected; facts asserted after that point are hidden.
    //
    // The Datalog query model: `:as-of N` is a transaction-time
    // filter applied before retraction logic, so the result is the
    // same property-graph view a present-time query would have
    // produced at tx_count = N.
    // --------------------------------------------------------

    /// `out_edges`, scoped to the graph state at `tx_count`.
    pub fn out_edges_at(&self, from: Uuid, edge_type: &str, tx_count: u64) -> Result<Vec<Edge>> {
        let script = format!(
            r#"(query [:find ?eid ?to
                       :where [?eid :_edge/from #uuid "{from}"]
                              [?eid :_edge/type "{et}"]
                              [?eid :_edge/to ?to]
                       :as-of {tx_count}])"#,
            from = from,
            et = escape_str(edge_type),
            tx_count = tx_count,
        );
        let result = self.execute(&script)?;
        let mut edges = Vec::with_capacity(result.rows.len());
        for row in result.rows {
            let mut it = row.into_iter();
            let eid = expect_ref(it.next())?;
            let to = expect_ref(it.next())?;
            edges.push(Edge {
                id: eid,
                from,
                t: edge_type.to_owned(),
                to,
            });
        }
        Ok(edges)
    }

    /// `in_edges`, scoped to the graph state at `tx_count`.
    pub fn in_edges_at(&self, to: Uuid, edge_type: &str, tx_count: u64) -> Result<Vec<Edge>> {
        let script = format!(
            r#"(query [:find ?eid ?from
                       :where [?eid :_edge/to #uuid "{to}"]
                              [?eid :_edge/type "{et}"]
                              [?eid :_edge/from ?from]
                       :as-of {tx_count}])"#,
            to = to,
            et = escape_str(edge_type),
            tx_count = tx_count,
        );
        let result = self.execute(&script)?;
        let mut edges = Vec::with_capacity(result.rows.len());
        for row in result.rows {
            let mut it = row.into_iter();
            let eid = expect_ref(it.next())?;
            let from = expect_ref(it.next())?;
            edges.push(Edge {
                id: eid,
                from,
                t: edge_type.to_owned(),
                to,
            });
        }
        Ok(edges)
    }

    /// `get_vertex_properties`, scoped to `tx_count`. The returned
    /// map reflects the property values in force at that point —
    /// scalars set later are absent, retracted-since values are
    /// resurrected.
    pub fn get_vertex_properties_at(
        &self,
        vertex_id: Uuid,
        tx_count: u64,
    ) -> Result<serde_json::Map<String, Value>> {
        let script = format!(
            r#"(query [:find ?attr ?val
                       :where [#uuid "{id}" ?attr ?val]
                       :as-of {tx_count}])"#,
            id = vertex_id,
            tx_count = tx_count,
        );
        let result = self.execute(&script)?;
        let mut map = serde_json::Map::new();
        for row in result.rows {
            let mut it = row.into_iter();
            let attr = match it.next() {
                Some(MgValue::Keyword(k)) => k.trim_start_matches(':').to_owned(),
                Some(MgValue::String(s)) => s,
                _ => continue,
            };
            if attr.starts_with('_') {
                continue;
            }
            if let Some(val) = it.next() {
                map.insert(attr, mg_to_json(val));
            }
        }
        Ok(map)
    }

    /// `get_edge_properties`, scoped to `tx_count`.
    pub fn get_edge_properties_at(
        &self,
        edge: &Edge,
        tx_count: u64,
    ) -> Result<serde_json::Map<String, Value>> {
        self.get_vertex_properties_at(edge.id, tx_count)
    }

    /// All vertices of a given type.
    pub fn vertices_of_type(&self, vertex_type: &str) -> Result<Vec<Vertex>> {
        let script = format!(
            r#"(query [:find ?e :where [?e :_type "{vt}"]])"#,
            vt = escape_str(vertex_type),
        );
        let result = self.execute(&script)?;
        let mut out = Vec::with_capacity(result.rows.len());
        for row in result.rows {
            let mut it = row.into_iter();
            let id = expect_ref(it.next())?;
            out.push(Vertex {
                id,
                t: vertex_type.to_owned(),
            });
        }
        Ok(out)
    }

    // --------------------------------------------------------
    // Internal helpers
    // --------------------------------------------------------

    fn execute(&self, script: &str) -> Result<QueryRows> {
        let result = self
            .inner
            .execute(script)
            .map_err(|e| Error::Backend(format!("minigraf: {e}")))?;
        Ok(QueryRows::from(result))
    }
}

// ============================================================
// Result adapter + helpers
// ============================================================

/// A minimal projection of `minigraf::QueryResult` that exposes the
/// `rows` as `Vec<Vec<Value>>` regardless of the underlying query
/// shape. Keeps the rest of this file from having to handle the
/// distinction between `Query` and `Transact` result variants.
struct QueryRows {
    rows: Vec<Vec<MgValue>>,
}

impl From<minigraf::QueryResult> for QueryRows {
    fn from(result: minigraf::QueryResult) -> Self {
        match result {
            minigraf::QueryResult::QueryResults { results, .. } => Self { rows: results },
            minigraf::QueryResult::Transacted(_)
            | minigraf::QueryResult::Retracted(_)
            | minigraf::QueryResult::Ok => Self { rows: Vec::new() },
        }
    }
}

fn expect_ref(v: Option<MgValue>) -> Result<Uuid> {
    match v {
        // EntityId is `pub type EntityId = Uuid`, so a `Ref` already
        // carries the Uuid directly.
        Some(MgValue::Ref(eid)) => Ok(eid),
        Some(other) => Err(Error::Backend(format!(
            "expected entity ref, got {other:?}"
        ))),
        None => Err(Error::Backend("query row missing expected column".into())),
    }
}

fn escape_str(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}

/// Allow only attribute names that won't break our minimal Datalog
/// formatting (no whitespace, no quotes), and reject leading `_`
/// which is reserved for schema attributes.
fn validate_user_attr(name: &str) -> Result<()> {
    if name.is_empty()
        || name.starts_with('_')
        || name
            .chars()
            .any(|c| !(c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.' | '/')))
    {
        return Err(Error::InvalidInput(format!(
            "property name {name:?} is not a valid attribute keyword"
        )));
    }
    Ok(())
}

/// Render a `serde_json::Value` as a Datalog literal embedded in a
/// transact form. Nested arrays/objects are stored as their JSON
/// string representation, prefixed with `json:` so reads can detect
/// and re-parse them.
fn json_to_mg_literal(v: &Value) -> String {
    match v {
        Value::Null => "nil".to_owned(),
        Value::Bool(true) => "true".to_owned(),
        Value::Bool(false) => "false".to_owned(),
        Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                i.to_string()
            } else if let Some(f) = n.as_f64() {
                // Datalog floats need a decimal point.
                if f.fract() == 0.0 {
                    format!("{f:.1}")
                } else {
                    f.to_string()
                }
            } else {
                "nil".to_owned()
            }
        }
        Value::String(s) => format!("\"{}\"", escape_str(s)),
        Value::Array(_) | Value::Object(_) => {
            let blob = serde_json::to_string(v).unwrap_or_else(|_| "null".to_owned());
            format!("\"json:{}\"", escape_str(&blob))
        }
    }
}

/// Render a minigraf `Value` as a Datalog literal, for embedding in
/// a `(retract [[entity attr LITERAL]])` form. The inverse of
/// [`json_to_mg_literal`] when crossed through the assert/retract
/// round trip.
fn mg_value_to_literal(v: &MgValue) -> String {
    match v {
        MgValue::Null => "nil".to_owned(),
        MgValue::Boolean(true) => "true".to_owned(),
        MgValue::Boolean(false) => "false".to_owned(),
        MgValue::Integer(i) => i.to_string(),
        MgValue::Float(f) => {
            if f.fract() == 0.0 {
                format!("{f:.1}")
            } else {
                f.to_string()
            }
        }
        MgValue::String(s) => format!("\"{}\"", escape_str(s)),
        // Keywords minigraf returns include the leading `:`; emit
        // them verbatim. Refs use the `#uuid "..."` form.
        MgValue::Keyword(k) => k.clone(),
        MgValue::Ref(u) => format!(r#"#uuid "{u}""#),
    }
}

/// Convert a minigraf `Value` back to a `serde_json::Value`. Strings
/// prefixed with `json:` are re-parsed as their original JSON shape
/// (the inverse of [`json_to_mg_literal`]).
fn mg_to_json(v: MgValue) -> Value {
    match v {
        MgValue::Null => Value::Null,
        MgValue::Boolean(b) => Value::Bool(b),
        MgValue::Integer(i) => Value::Number(i.into()),
        MgValue::Float(f) => serde_json::Number::from_f64(f)
            .map(Value::Number)
            .unwrap_or(Value::Null),
        MgValue::String(s) => {
            if let Some(rest) = s.strip_prefix("json:") {
                serde_json::from_str(rest).unwrap_or(Value::String(s))
            } else {
                Value::String(s)
            }
        }
        MgValue::Keyword(k) => Value::String(k),
        MgValue::Ref(eid) => Value::String(eid.to_string()),
    }
}

// `BindValue` import kept so future parameterized-query helpers can
// be added without touching the import list.
#[allow(dead_code)]
fn _ensure_bindvalue_imported(_: BindValue) {}
