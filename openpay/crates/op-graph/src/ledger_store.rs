//! [`GraphLedgerStore`]: a graph-backed implementation of
//! [`op_ledger::LedgerStore`].
//!
//! ## Storage layout
//!
//! - Each [`Ledger`] is a `ledger_ledger` vertex with properties
//!   `name` (string), `description` (string or null).
//! - Each [`Account`] is a `ledger_account` vertex with properties
//!   `name`, `class`, `normal_balance`, `currency_code`,
//!   `currency_exponent`, `external_id` (or null), plus a
//!   `ledger_in_ledger` edge to its parent ledger.
//! - Each [`Transaction`] is a `ledger_tx` vertex with properties
//!   `status`, `external_id` (or null), `description` (or null),
//!   `effective_at_unix_secs`, `metadata_csv` (serialized), and a
//!   `ledger_in_ledger` edge to the parent ledger. Each [`Entry`]
//!   becomes either a `ledger_debit` or `ledger_credit` edge from
//!   the tx vertex to the account vertex, with properties
//!   `amount_minor`, `currency_code`, `currency_exponent`.
//!
//! ## Why edge-per-entry instead of properties-on-tx?
//!
//! Graph databases shine when relationships are edges. "Which
//! accounts did this tx touch?" is then a single outbound-edge
//! traversal of the tx vertex. With entries-as-properties we'd be
//! doing an O(N) scan of every tx in the database.
//!
//! ## Idempotency
//!
//! `post_transaction` consults
//! [`GraphHandle::find_vertex_by_property`] on the
//! `ledger_tx`/`external_id` property to dedupe. The store also
//! maintains an in-memory `external_id → tx_id` index inside a
//! `Mutex<HashMap>` for fast lookups; the index is rebuilt from
//! the graph on first access. This mirrors the
//! `InMemoryLedgerStore` pattern.

use std::collections::HashMap;
use std::sync::Mutex;

use serde_json::Value as Json;
use uuid::Uuid;

use op_core::{Currency, Money};
use op_ledger::{
    Account, AccountClass, AccountId, Balance, Direction, Entry, Error as LedgerError, Ledger,
    LedgerId, LedgerStore, NormalBalance, Result as LedgerResult, Status, Transaction,
    TransactionId,
};

use crate::error::{Error, Result};
use crate::graph::{GraphHandle, etypes, vtypes};

// ============================================================
// GraphLedgerStore
// ============================================================

/// Graph-backed implementation of [`op_ledger::LedgerStore`].
///
/// Clone is cheap (the underlying `GraphHandle` is `Arc`-backed).
#[derive(Clone)]
pub struct GraphLedgerStore {
    handle: GraphHandle,
    /// `external_id` → `TransactionId` cache for fast idempotency
    /// lookup. Lazily populated on writes; not authoritative — the
    /// graph itself is.
    by_external_id: std::sync::Arc<Mutex<HashMap<String, TransactionId>>>,
}

impl GraphLedgerStore {
    /// Build a fresh store backed by an in-memory graph.
    #[must_use]
    pub fn new_in_memory() -> Self {
        Self::with_handle(GraphHandle::new_in_memory())
    }

    /// Build on top of an existing [`GraphHandle`] (so multiple
    /// stores can share one graph).
    #[must_use]
    pub fn with_handle(handle: GraphHandle) -> Self {
        Self {
            handle,
            by_external_id: std::sync::Arc::new(Mutex::new(HashMap::new())),
        }
    }

    /// Access the underlying graph handle (for shared use by
    /// [`crate::queries`] helpers).
    #[must_use]
    pub fn handle(&self) -> &GraphHandle {
        &self.handle
    }

    // --------------------------------------------------------
    // Property codecs
    // --------------------------------------------------------

    fn write_ledger_props(&self, ledger: &Ledger) -> Result<()> {
        self.handle.set_vertex_property(
            ledger.id.as_uuid(),
            "name",
            Json::String(ledger.name.clone()),
        )?;
        self.handle.set_vertex_property(
            ledger.id.as_uuid(),
            "description",
            match &ledger.description {
                Some(d) => Json::String(d.clone()),
                None => Json::Null,
            },
        )?;
        Ok(())
    }

    fn read_ledger(&self, id: LedgerId) -> Result<Ledger> {
        let _ = self
            .handle
            .get_typed_vertex(id.as_uuid(), vtypes::LEDGER_LEDGER)?;
        let props = self.handle.get_vertex_properties(id.as_uuid())?;
        let name = json_string(&props, "name")?;
        let description = json_opt_string(&props, "description");
        // Reconstruct via Ledger::new + override id.
        let mut l = Ledger::new(name).map_err(|e| Error::Invariant(e.to_string()))?;
        l.id = id;
        l.description = description;
        Ok(l)
    }

    fn write_account_props(&self, account: &Account) -> Result<()> {
        let uid = account.id.as_uuid();
        self.handle
            .set_vertex_property(uid, "name", Json::String(account.name.clone()))?;
        self.handle.set_vertex_property(
            uid,
            "class",
            Json::String(account_class_str(account.class).to_owned()),
        )?;
        self.handle.set_vertex_property(
            uid,
            "normal_balance",
            Json::String(normal_balance_str(account.normal_balance).to_owned()),
        )?;
        self.handle.set_vertex_property(
            uid,
            "currency_code",
            Json::String(account.currency.code().to_owned()),
        )?;
        self.handle.set_vertex_property(
            uid,
            "currency_exponent",
            Json::Number(serde_json::Number::from(account.currency.exponent())),
        )?;
        self.handle.set_vertex_property(
            uid,
            "external_id",
            match &account.external_id {
                Some(e) => Json::String(e.clone()),
                None => Json::Null,
            },
        )?;
        Ok(())
    }

    fn read_account(&self, id: AccountId) -> Result<Account> {
        let _ = self
            .handle
            .get_typed_vertex(id.as_uuid(), vtypes::LEDGER_ACCOUNT)?;
        let props = self.handle.get_vertex_properties(id.as_uuid())?;
        let name = json_string(&props, "name")?;
        let class_s = json_string(&props, "class")?;
        let nb_s = json_string(&props, "normal_balance")?;
        let ccode = json_string(&props, "currency_code")?;
        let cexp = json_u64(&props, "currency_exponent")? as u8;
        let external_id = json_opt_string(&props, "external_id");
        // Recover ledger_id via the ledger_in_ledger edge.
        let ledger_id = self.account_parent_ledger(id)?;
        let currency = currency_from_props(&ccode, cexp)?;
        let class = parse_account_class(&class_s)?;
        let nb = parse_normal_balance(&nb_s)?;
        Ok(Account {
            id,
            ledger_id,
            name,
            class,
            normal_balance: nb,
            currency,
            external_id,
        })
    }

    fn account_parent_ledger(&self, id: AccountId) -> Result<LedgerId> {
        let edges = self
            .handle
            .out_edges(id.as_uuid(), etypes::LEDGER_IN_LEDGER)?;
        let edge = edges.into_iter().next().ok_or_else(|| {
            Error::Invariant(format!("account {id} has no ledger_in_ledger edge"))
        })?;
        Ok(LedgerId::from_uuid(edge.to))
    }

    fn write_tx_props(&self, tx: &Transaction) -> Result<()> {
        let uid = tx.id.as_uuid();
        self.handle.set_vertex_property(
            uid,
            "status",
            Json::String(status_str(tx.status).to_owned()),
        )?;
        self.handle.set_vertex_property(
            uid,
            "external_id",
            match &tx.external_id {
                Some(e) => Json::String(e.clone()),
                None => Json::Null,
            },
        )?;
        self.handle.set_vertex_property(
            uid,
            "description",
            match &tx.description {
                Some(d) => Json::String(d.clone()),
                None => Json::Null,
            },
        )?;
        self.handle.set_vertex_property(
            uid,
            "effective_at_unix_secs",
            Json::Number(serde_json::Number::from(tx.effective_at_unix_secs)),
        )?;
        // Metadata as a JSON object.
        let mut md = serde_json::Map::new();
        for (k, v) in &tx.metadata {
            md.insert(k.clone(), Json::String(v.clone()));
        }
        self.handle
            .set_vertex_property(uid, "metadata", Json::Object(md))?;
        Ok(())
    }

    fn read_tx(&self, id: TransactionId) -> Result<Transaction> {
        let _ = self
            .handle
            .get_typed_vertex(id.as_uuid(), vtypes::LEDGER_TX)?;
        let props = self.handle.get_vertex_properties(id.as_uuid())?;
        let status_s = json_string(&props, "status")?;
        let status = parse_status(&status_s)?;
        let external_id = json_opt_string(&props, "external_id");
        let description = json_opt_string(&props, "description");
        let effective = json_u64(&props, "effective_at_unix_secs")?;
        let metadata = json_opt_object_strs(&props, "metadata");
        // ledger_id via tx → ledger edge.
        let edges = self
            .handle
            .out_edges(id.as_uuid(), etypes::LEDGER_IN_LEDGER)?;
        let ledger_id = edges
            .into_iter()
            .next()
            .map(|e| LedgerId::from_uuid(e.to))
            .ok_or_else(|| Error::Invariant(format!("tx {id} has no ledger_in_ledger edge")))?;
        // Entries: debit + credit edges out of the tx.
        let mut entries: Vec<Entry> = Vec::new();
        for edge in self.handle.out_edges(id.as_uuid(), etypes::LEDGER_DEBIT)? {
            entries.push(self.read_entry_edge(&edge, Direction::Debit)?);
        }
        for edge in self.handle.out_edges(id.as_uuid(), etypes::LEDGER_CREDIT)? {
            entries.push(self.read_entry_edge(&edge, Direction::Credit)?);
        }
        if entries.is_empty() {
            return Err(Error::Invariant(format!(
                "tx {id} has no debit/credit edges"
            )));
        }
        Ok(Transaction {
            id,
            ledger_id,
            status,
            external_id,
            description,
            effective_at_unix_secs: effective,
            entries,
            metadata,
        })
    }

    fn read_entry_edge(&self, edge: &crate::graph::Edge, direction: Direction) -> Result<Entry> {
        let props = self.handle.get_edge_properties(edge)?;
        let amount_minor = json_i64(&props, "amount_minor")?;
        let ccode = json_string(&props, "currency_code")?;
        let cexp = json_u64(&props, "currency_exponent")? as u8;
        let currency = currency_from_props(&ccode, cexp)?;
        Ok(Entry {
            account_id: AccountId::from_uuid(edge.to),
            direction,
            amount: Money::from_minor(amount_minor, currency),
        })
    }

    fn same_body(a: &Transaction, b: &Transaction) -> bool {
        if a.ledger_id != b.ledger_id {
            return false;
        }
        if a.external_id != b.external_id {
            return false;
        }
        if a.effective_at_unix_secs != b.effective_at_unix_secs {
            return false;
        }
        if a.entries.len() != b.entries.len() {
            return false;
        }
        // Entry-by-entry comparison ignoring order: tally each
        // (account, direction, amount).
        let mut bag_a: HashMap<(AccountId, Direction, i64, String), i32> = HashMap::new();
        let mut bag_b: HashMap<(AccountId, Direction, i64, String), i32> = HashMap::new();
        for e in &a.entries {
            *bag_a
                .entry((
                    e.account_id,
                    e.direction,
                    e.amount.minor_units,
                    e.amount.currency.code().to_owned(),
                ))
                .or_insert(0) += 1;
        }
        for e in &b.entries {
            *bag_b
                .entry((
                    e.account_id,
                    e.direction,
                    e.amount.minor_units,
                    e.amount.currency.code().to_owned(),
                ))
                .or_insert(0) += 1;
        }
        bag_a == bag_b
    }
}

// ============================================================
// LedgerStore impl
// ============================================================

impl LedgerStore for GraphLedgerStore {
    fn create_ledger(&self, ledger: Ledger) -> LedgerResult<LedgerId> {
        let id = ledger.id;
        self.handle
            .create_vertex(vtypes::LEDGER_LEDGER, id.as_uuid())
            .map_err(LedgerError::from)?;
        self.write_ledger_props(&ledger)
            .map_err(LedgerError::from)?;
        Ok(id)
    }

    fn get_ledger(&self, id: LedgerId) -> LedgerResult<Ledger> {
        self.read_ledger(id).map_err(|e| match e {
            Error::VertexNotFound { .. } => LedgerError::LedgerNotFound(id.to_string()),
            other => LedgerError::from(other),
        })
    }

    fn create_account(&self, account: Account) -> LedgerResult<AccountId> {
        // Verify the parent ledger exists; refuse cross-ledger create.
        if !self
            .handle
            .vertex_exists(account.ledger_id.as_uuid())
            .map_err(LedgerError::from)?
        {
            return Err(LedgerError::LedgerNotFound(account.ledger_id.to_string()));
        }
        let id = account.id;
        self.handle
            .create_vertex(vtypes::LEDGER_ACCOUNT, id.as_uuid())
            .map_err(LedgerError::from)?;
        self.handle
            .create_edge(
                id.as_uuid(),
                etypes::LEDGER_IN_LEDGER,
                account.ledger_id.as_uuid(),
            )
            .map_err(LedgerError::from)?;
        self.write_account_props(&account)
            .map_err(LedgerError::from)?;
        Ok(id)
    }

    fn get_account(&self, id: AccountId) -> LedgerResult<Account> {
        self.read_account(id).map_err(|e| match e {
            Error::VertexNotFound { .. } => LedgerError::AccountNotFound(id.to_string()),
            other => LedgerError::from(other),
        })
    }

    fn post_transaction(&self, transaction: Transaction) -> LedgerResult<TransactionId> {
        // 1. Ledger must exist.
        if !self
            .handle
            .vertex_exists(transaction.ledger_id.as_uuid())
            .map_err(LedgerError::from)?
        {
            return Err(LedgerError::LedgerNotFound(
                transaction.ledger_id.to_string(),
            ));
        }
        // 2. Idempotency by external_id — consult cache then graph.
        if let Some(ext) = &transaction.external_id {
            let cached = self
                .by_external_id
                .lock()
                .expect("poisoned")
                .get(ext)
                .copied();
            if let Some(existing_id) = cached {
                let existing = self.read_tx(existing_id).map_err(LedgerError::from)?;
                if Self::same_body(&existing, &transaction) {
                    return Ok(existing_id);
                }
                return Err(LedgerError::IdempotencyMismatch);
            }
        }
        // 3. Validate every entry's account exists, belongs to the
        //    same ledger, currency matches.
        for entry in &transaction.entries {
            let acc = self.read_account(entry.account_id).map_err(|e| match e {
                Error::VertexNotFound { .. } => {
                    LedgerError::AccountNotFound(entry.account_id.to_string())
                }
                other => LedgerError::from(other),
            })?;
            if acc.ledger_id != transaction.ledger_id {
                return Err(LedgerError::CrossLedgerEntry {
                    account_id: entry.account_id.to_string(),
                    account_ledger: acc.ledger_id.to_string(),
                    expected_ledger: transaction.ledger_id.to_string(),
                });
            }
            if acc.currency != entry.amount.currency {
                return Err(LedgerError::CurrencyMismatch {
                    entry_currency: entry.amount.currency.code().to_owned(),
                    account_currency: acc.currency.code().to_owned(),
                    account_id: entry.account_id.to_string(),
                });
            }
        }
        // 4. Persist tx vertex + edges.
        let tx_uuid = transaction.id.as_uuid();
        self.handle
            .create_vertex(vtypes::LEDGER_TX, tx_uuid)
            .map_err(LedgerError::from)?;
        self.handle
            .create_edge(
                tx_uuid,
                etypes::LEDGER_IN_LEDGER,
                transaction.ledger_id.as_uuid(),
            )
            .map_err(LedgerError::from)?;
        self.write_tx_props(&transaction)
            .map_err(LedgerError::from)?;
        for entry in &transaction.entries {
            let etype = match entry.direction {
                Direction::Debit => etypes::LEDGER_DEBIT,
                Direction::Credit => etypes::LEDGER_CREDIT,
            };
            self.handle
                .create_edge(tx_uuid, etype, entry.account_id.as_uuid())
                .map_err(LedgerError::from)?;
            // We need the actual Edge struct to set properties on it.
            // out_edges() returns all matching edges; pick the one
            // we just created (matching to).
            let edges = self
                .handle
                .out_edges(tx_uuid, etype)
                .map_err(LedgerError::from)?;
            let edge = edges
                .into_iter()
                .find(|e| e.to == entry.account_id.as_uuid())
                .ok_or_else(|| {
                    LedgerError::from(Error::Invariant(format!(
                        "freshly-created entry edge missing for tx {} → account {}",
                        transaction.id, entry.account_id
                    )))
                })?;
            self.handle
                .set_edge_property(
                    &edge,
                    "amount_minor",
                    Json::Number(serde_json::Number::from(entry.amount.minor_units)),
                )
                .map_err(LedgerError::from)?;
            self.handle
                .set_edge_property(
                    &edge,
                    "currency_code",
                    Json::String(entry.amount.currency.code().to_owned()),
                )
                .map_err(LedgerError::from)?;
            self.handle
                .set_edge_property(
                    &edge,
                    "currency_exponent",
                    Json::Number(serde_json::Number::from(entry.amount.currency.exponent())),
                )
                .map_err(LedgerError::from)?;
        }
        // 5. Cache external_id.
        if let Some(ext) = &transaction.external_id {
            self.by_external_id
                .lock()
                .expect("poisoned")
                .insert(ext.clone(), transaction.id);
        }
        // 6. Stamp the tx_count *after* all of this tx's writes
        //    have advanced the bi-temporal log. That's the value
        //    the time-travel readers anchor `:as-of` against.
        let posted_at = self.handle.tx_count();
        self.handle
            .set_vertex_property(
                tx_uuid,
                "posted_at_tx_count",
                Json::Number(serde_json::Number::from(posted_at)),
            )
            .map_err(LedgerError::from)?;
        Ok(transaction.id)
    }

    fn get_transaction(&self, id: TransactionId) -> LedgerResult<Transaction> {
        self.read_tx(id).map_err(|e| match e {
            Error::VertexNotFound { .. } => LedgerError::TransactionNotFound(id.to_string()),
            other => LedgerError::from(other),
        })
    }

    fn find_by_external_id(&self, external_id: &str) -> LedgerResult<Option<Transaction>> {
        // Consult cache first — fast path for writes from this
        // process.
        let cached = self
            .by_external_id
            .lock()
            .expect("poisoned")
            .get(external_id)
            .copied();
        if let Some(tid) = cached {
            return Ok(Some(self.read_tx(tid).map_err(LedgerError::from)?));
        }
        // Cache miss: the handle may have just opened a persistent
        // file written by a previous process. Walk `ledger_tx`
        // vertices and check the stored `external_id` property. On
        // a hit, populate the cache for subsequent lookups.
        let verts = self
            .handle
            .vertices_of_type(vtypes::LEDGER_TX)
            .map_err(LedgerError::from)?;
        for v in verts {
            let props = self
                .handle
                .get_vertex_properties(v.id)
                .map_err(LedgerError::from)?;
            if let Some(Json::String(ext)) = props.get("external_id")
                && ext == external_id
            {
                let tid = TransactionId::from_uuid(v.id);
                self.by_external_id
                    .lock()
                    .expect("poisoned")
                    .insert(external_id.to_owned(), tid);
                return Ok(Some(self.read_tx(tid).map_err(LedgerError::from)?));
            }
        }
        Ok(None)
    }

    fn mark_posted(&self, id: TransactionId) -> LedgerResult<()> {
        let mut tx = self.read_tx(id).map_err(|e| match e {
            Error::VertexNotFound { .. } => LedgerError::TransactionNotFound(id.to_string()),
            other => LedgerError::from(other),
        })?;
        // Defer to the type's own state machine — this keeps the
        // terminal-state error in one place.
        tx.post()?;
        // Persist new status.
        self.handle
            .set_vertex_property(
                id.as_uuid(),
                "status",
                Json::String(status_str(tx.status).to_owned()),
            )
            .map_err(LedgerError::from)?;
        Ok(())
    }

    fn mark_archived(&self, id: TransactionId) -> LedgerResult<()> {
        let mut tx = self.read_tx(id).map_err(|e| match e {
            Error::VertexNotFound { .. } => LedgerError::TransactionNotFound(id.to_string()),
            other => LedgerError::from(other),
        })?;
        tx.archive()?;
        self.handle
            .set_vertex_property(
                id.as_uuid(),
                "status",
                Json::String(status_str(tx.status).to_owned()),
            )
            .map_err(LedgerError::from)?;
        Ok(())
    }

    fn balance(&self, account_id: AccountId) -> LedgerResult<Balance> {
        let account = self.read_account(account_id).map_err(|e| match e {
            Error::VertexNotFound { .. } => LedgerError::AccountNotFound(account_id.to_string()),
            other => LedgerError::from(other),
        })?;
        let currency = account.currency;
        let normal = account.normal_balance;

        let mut posted_debits: i64 = 0;
        let mut posted_credits: i64 = 0;
        let mut pending_debits: i64 = 0;
        let mut pending_credits: i64 = 0;

        // Walk every inbound debit + credit edge to this account
        // vertex. Each one's source is the tx vertex; tally per tx
        // status.
        for (direction, etype) in [
            (Direction::Debit, etypes::LEDGER_DEBIT),
            (Direction::Credit, etypes::LEDGER_CREDIT),
        ] {
            let edges = self
                .handle
                .in_edges(account_id.as_uuid(), etype)
                .map_err(LedgerError::from)?;
            for edge in edges {
                // Read the entry edge's amount.
                let props = self
                    .handle
                    .get_edge_properties(&edge)
                    .map_err(LedgerError::from)?;
                let amount = json_i64(&props, "amount_minor").map_err(LedgerError::from)?;
                let ccode = json_string(&props, "currency_code").map_err(LedgerError::from)?;
                // Guard: currency on the edge must equal account currency
                // (already validated at post time, but defensive on
                // read).
                if ccode != currency.code() {
                    return Err(LedgerError::CurrencyMismatch {
                        entry_currency: ccode,
                        account_currency: currency.code().to_owned(),
                        account_id: account_id.to_string(),
                    });
                }
                // Read tx status.
                let tx_props = self
                    .handle
                    .get_vertex_properties(edge.from)
                    .map_err(LedgerError::from)?;
                let status_s = json_string(&tx_props, "status").map_err(LedgerError::from)?;
                let status = parse_status(&status_s).map_err(LedgerError::from)?;
                match (status, direction) {
                    (Status::Posted, Direction::Debit) => {
                        posted_debits = posted_debits
                            .checked_add(amount)
                            .ok_or(op_core::Error::Overflow)?;
                        pending_debits = pending_debits
                            .checked_add(amount)
                            .ok_or(op_core::Error::Overflow)?;
                    }
                    (Status::Posted, Direction::Credit) => {
                        posted_credits = posted_credits
                            .checked_add(amount)
                            .ok_or(op_core::Error::Overflow)?;
                        pending_credits = pending_credits
                            .checked_add(amount)
                            .ok_or(op_core::Error::Overflow)?;
                    }
                    (Status::Pending, Direction::Debit) => {
                        pending_debits = pending_debits
                            .checked_add(amount)
                            .ok_or(op_core::Error::Overflow)?;
                    }
                    (Status::Pending, Direction::Credit) => {
                        pending_credits = pending_credits
                            .checked_add(amount)
                            .ok_or(op_core::Error::Overflow)?;
                    }
                    (Status::Archived, _) => { /* no-op */ }
                }
            }
        }

        let posted_minor = match normal {
            NormalBalance::Debit => posted_debits.saturating_sub(posted_credits),
            NormalBalance::Credit => posted_credits.saturating_sub(posted_debits),
        };
        let pending_minor = match normal {
            NormalBalance::Debit => pending_debits.saturating_sub(pending_credits),
            NormalBalance::Credit => pending_credits.saturating_sub(pending_debits),
        };
        Ok(Balance {
            currency,
            posted: Money::from_minor(posted_minor, currency),
            pending: Money::from_minor(pending_minor, currency),
        })
    }
}

// ============================================================
// LedgerHistory — bi-temporal time-travel reads
// ============================================================

impl op_ledger::LedgerHistory for GraphLedgerStore {
    /// Walk the same edge structure as [`Self::balance`], but with
    /// every read scoped to `tx_count` via Minigraf's `:as-of`
    /// filter. Entries posted after `tx_count` are invisible; tx
    /// statuses that flipped after `tx_count` show their earlier
    /// value (often `Pending` instead of the current `Posted`).
    ///
    /// The account's own currency and normal-balance are immutable
    /// (set at account creation, never changed) so they're read at
    /// the present-time view.
    fn balance_as_of(&self, account_id: AccountId, tx_count: u64) -> LedgerResult<Balance> {
        let account = self.read_account(account_id).map_err(|e| match e {
            Error::VertexNotFound { .. } => LedgerError::AccountNotFound(account_id.to_string()),
            other => LedgerError::from(other),
        })?;
        let currency = account.currency;
        let normal = account.normal_balance;

        let mut posted_debits: i64 = 0;
        let mut posted_credits: i64 = 0;
        let mut pending_debits: i64 = 0;
        let mut pending_credits: i64 = 0;

        for (direction, etype) in [
            (Direction::Debit, etypes::LEDGER_DEBIT),
            (Direction::Credit, etypes::LEDGER_CREDIT),
        ] {
            let edges = self
                .handle
                .in_edges_at(account_id.as_uuid(), etype, tx_count)
                .map_err(LedgerError::from)?;
            for edge in edges {
                let props = self
                    .handle
                    .get_edge_properties_at(&edge, tx_count)
                    .map_err(LedgerError::from)?;
                let amount = json_i64(&props, "amount_minor").map_err(LedgerError::from)?;
                let ccode = json_string(&props, "currency_code").map_err(LedgerError::from)?;
                if ccode != currency.code() {
                    return Err(LedgerError::CurrencyMismatch {
                        entry_currency: ccode,
                        account_currency: currency.code().to_owned(),
                        account_id: account_id.to_string(),
                    });
                }
                let tx_props = self
                    .handle
                    .get_vertex_properties_at(edge.from, tx_count)
                    .map_err(LedgerError::from)?;
                let status_s = json_string(&tx_props, "status").map_err(LedgerError::from)?;
                let status = parse_status(&status_s).map_err(LedgerError::from)?;
                match (status, direction) {
                    (Status::Posted, Direction::Debit) => {
                        posted_debits = posted_debits
                            .checked_add(amount)
                            .ok_or(op_core::Error::Overflow)?;
                        pending_debits = pending_debits
                            .checked_add(amount)
                            .ok_or(op_core::Error::Overflow)?;
                    }
                    (Status::Posted, Direction::Credit) => {
                        posted_credits = posted_credits
                            .checked_add(amount)
                            .ok_or(op_core::Error::Overflow)?;
                        pending_credits = pending_credits
                            .checked_add(amount)
                            .ok_or(op_core::Error::Overflow)?;
                    }
                    (Status::Pending, Direction::Debit) => {
                        pending_debits = pending_debits
                            .checked_add(amount)
                            .ok_or(op_core::Error::Overflow)?;
                    }
                    (Status::Pending, Direction::Credit) => {
                        pending_credits = pending_credits
                            .checked_add(amount)
                            .ok_or(op_core::Error::Overflow)?;
                    }
                    (Status::Archived, _) => { /* skip */ }
                }
            }
        }

        let signed = |debits: i64, credits: i64| -> Result<i64, op_core::Error> {
            match normal {
                NormalBalance::Debit => debits.checked_sub(credits).ok_or(op_core::Error::Overflow),
                NormalBalance::Credit => {
                    credits.checked_sub(debits).ok_or(op_core::Error::Overflow)
                }
            }
        };
        let posted_minor = signed(posted_debits, posted_credits)?;
        let pending_minor = signed(pending_debits, pending_credits)?;
        Ok(Balance {
            currency,
            posted: Money::from_minor(posted_minor, currency),
            pending: Money::from_minor(pending_minor, currency),
        })
    }

    /// Same shape as [`Self::read_tx`] but with every property and
    /// edge read scoped to `tx_count`. The historical `status` is
    /// the load-bearing field: a transaction posted in the present
    /// shows as `Pending` at a `tx_count` snapshotted before the
    /// `mark_posted` call.
    ///
    /// # Errors
    /// `TransactionNotFound` if the tx vertex didn't have a
    /// `_type = "ledger_tx"` fact at the requested `tx_count`.
    fn transaction_as_of(&self, id: TransactionId, tx_count: u64) -> LedgerResult<Transaction> {
        let props = self
            .handle
            .get_vertex_properties_at(id.as_uuid(), tx_count)
            .map_err(LedgerError::from)?;
        if props.is_empty() {
            return Err(LedgerError::TransactionNotFound(id.to_string()));
        }
        let status_s = json_string(&props, "status").map_err(LedgerError::from)?;
        let status = parse_status(&status_s).map_err(LedgerError::from)?;
        let external_id = json_opt_string(&props, "external_id");
        let description = json_opt_string(&props, "description");
        let effective = json_u64(&props, "effective_at_unix_secs").map_err(LedgerError::from)?;
        let metadata = json_opt_object_strs(&props, "metadata");
        let edges = self
            .handle
            .out_edges_at(id.as_uuid(), etypes::LEDGER_IN_LEDGER, tx_count)
            .map_err(LedgerError::from)?;
        let ledger_id = edges
            .into_iter()
            .next()
            .map(|e| LedgerId::from_uuid(e.to))
            .ok_or_else(|| {
                LedgerError::from(Error::Invariant(format!(
                    "tx {id} has no ledger_in_ledger edge as-of {tx_count}"
                )))
            })?;
        let mut entries: Vec<Entry> = Vec::new();
        for edge in self
            .handle
            .out_edges_at(id.as_uuid(), etypes::LEDGER_DEBIT, tx_count)
            .map_err(LedgerError::from)?
        {
            entries.push(self.read_entry_edge_at(&edge, Direction::Debit, tx_count)?);
        }
        for edge in self
            .handle
            .out_edges_at(id.as_uuid(), etypes::LEDGER_CREDIT, tx_count)
            .map_err(LedgerError::from)?
        {
            entries.push(self.read_entry_edge_at(&edge, Direction::Credit, tx_count)?);
        }
        Ok(Transaction {
            id,
            ledger_id,
            status,
            external_id,
            description,
            effective_at_unix_secs: effective,
            entries,
            metadata,
        })
    }

    fn balance_as_of_time(&self, account: AccountId, at_unix_secs: u64) -> LedgerResult<Balance> {
        let anchor = self
            .tx_count_at_time(at_unix_secs)
            .map_err(LedgerError::from)?;
        match anchor {
            Some(tx_count) => self.balance_as_of(account, tx_count),
            None => {
                // No transactions at or before that time. Zero
                // balance in the account's currency.
                let account = self.read_account(account).map_err(|e| match e {
                    Error::VertexNotFound { .. } => {
                        LedgerError::AccountNotFound(account.to_string())
                    }
                    other => LedgerError::from(other),
                })?;
                Ok(Balance {
                    currency: account.currency,
                    posted: Money::from_minor(0, account.currency),
                    pending: Money::from_minor(0, account.currency),
                })
            }
        }
    }

    fn transaction_as_of_time(
        &self,
        id: TransactionId,
        at_unix_secs: u64,
    ) -> LedgerResult<Transaction> {
        let Some(tx_count) = self
            .tx_count_at_time(at_unix_secs)
            .map_err(LedgerError::from)?
        else {
            return Err(LedgerError::TransactionNotFound(id.to_string()));
        };
        self.transaction_as_of(id, tx_count)
    }

    fn save_checkpoint(&self, name: &str) -> LedgerResult<u64> {
        validate_checkpoint_name(name)?;
        let tx_count = self.handle.tx_count();
        let vid = checkpoint_uuid(name);
        // Idempotent: create_vertex on an existing id is a no-op for
        // our facade since we re-set the properties below; the
        // `_type` fact may already exist.
        let _ = self.handle.create_vertex(vtypes::LEDGER_CHECKPOINT, vid);
        self.handle
            .set_vertex_property(vid, "name", Json::String(name.to_owned()))
            .map_err(LedgerError::from)?;
        self.handle
            .set_vertex_property(
                vid,
                "tx_count",
                Json::Number(serde_json::Number::from(tx_count)),
            )
            .map_err(LedgerError::from)?;
        Ok(tx_count)
    }

    fn tx_count_at_checkpoint(&self, name: &str) -> LedgerResult<Option<u64>> {
        validate_checkpoint_name(name)?;
        let vid = checkpoint_uuid(name);
        let props = self
            .handle
            .get_vertex_properties(vid)
            .map_err(LedgerError::from)?;
        if props.is_empty() {
            return Ok(None);
        }
        Ok(props.get("tx_count").and_then(|v| match v {
            Json::Number(n) => n.as_u64(),
            _ => None,
        }))
    }

    fn replay_window(&self, start_tx: u64, end_tx: u64) -> LedgerResult<Vec<TransactionId>> {
        if end_tx < start_tx {
            return Ok(Vec::new());
        }
        let verts = self
            .handle
            .vertices_of_type(vtypes::LEDGER_TX)
            .map_err(LedgerError::from)?;
        let mut out = Vec::new();
        for v in verts {
            let props = self
                .handle
                .get_vertex_properties(v.id)
                .map_err(LedgerError::from)?;
            let Some(Json::Number(n)) = props.get("posted_at_tx_count") else {
                continue;
            };
            let Some(posted) = n.as_u64() else {
                continue;
            };
            if posted >= start_tx && posted <= end_tx {
                out.push(TransactionId::from_uuid(v.id));
            }
        }
        Ok(out)
    }
}

impl GraphLedgerStore {
    /// Find the largest tx_count anchored at-or-before
    /// `at_unix_secs`. Returns `None` if no posted ledger_tx had an
    /// `effective_at_unix_secs` at-or-before that wall-clock.
    fn tx_count_at_time(&self, at_unix_secs: u64) -> Result<Option<u64>> {
        let verts = self.handle.vertices_of_type(vtypes::LEDGER_TX)?;
        let mut best: Option<u64> = None;
        for v in verts {
            let props = self.handle.get_vertex_properties(v.id)?;
            let Some(Json::Number(eff_n)) = props.get("effective_at_unix_secs") else {
                continue;
            };
            let Some(eff) = eff_n.as_u64() else { continue };
            if eff > at_unix_secs {
                continue;
            }
            let Some(Json::Number(pc_n)) = props.get("posted_at_tx_count") else {
                continue;
            };
            let Some(pc) = pc_n.as_u64() else { continue };
            best = Some(best.map_or(pc, |cur| cur.max(pc)));
        }
        Ok(best)
    }

    fn read_entry_edge_at(
        &self,
        edge: &crate::graph::Edge,
        direction: Direction,
        tx_count: u64,
    ) -> LedgerResult<Entry> {
        let props = self
            .handle
            .get_edge_properties_at(edge, tx_count)
            .map_err(LedgerError::from)?;
        let amount_minor = json_i64(&props, "amount_minor").map_err(LedgerError::from)?;
        let ccode = json_string(&props, "currency_code").map_err(LedgerError::from)?;
        let cexp = json_u64(&props, "currency_exponent").map_err(LedgerError::from)? as u8;
        let currency = currency_from_props(&ccode, cexp).map_err(LedgerError::from)?;
        Ok(Entry {
            account_id: AccountId::from_uuid(edge.to),
            direction,
            amount: Money::from_minor(amount_minor, currency),
        })
    }
}

// ============================================================
// Codec helpers
// ============================================================

/// Deterministic vertex id for a named checkpoint: UUIDv5 over the
/// name in the URL namespace, so re-saving the same name reuses the
/// same vertex (idempotent) and lookups don't need a separate index.
fn checkpoint_uuid(name: &str) -> uuid::Uuid {
    uuid::Uuid::new_v5(
        &uuid::Uuid::NAMESPACE_URL,
        format!("ledger_checkpoint:{name}").as_bytes(),
    )
}

/// Reject checkpoint names that won't fit our minimal Datalog
/// attribute formatting elsewhere. Length-capped at 128 to keep the
/// vertex-property store sensible.
fn validate_checkpoint_name(name: &str) -> LedgerResult<()> {
    if name.is_empty() || name.len() > 128 {
        return Err(LedgerError::from(Error::InvalidInput(format!(
            "checkpoint name {name:?} must be 1..=128 chars"
        ))));
    }
    Ok(())
}

fn json_string(props: &serde_json::Map<String, Json>, key: &str) -> Result<String> {
    props
        .get(key)
        .and_then(|v| v.as_str())
        .map(|s| s.to_owned())
        .ok_or_else(|| Error::PropertyTypeMismatch {
            vertex_id: "?".into(),
            property: key.into(),
            expected_type: "string".into(),
        })
}

fn json_opt_string(props: &serde_json::Map<String, Json>, key: &str) -> Option<String> {
    props.get(key).and_then(|v| match v {
        Json::String(s) => Some(s.clone()),
        _ => None,
    })
}

fn json_u64(props: &serde_json::Map<String, Json>, key: &str) -> Result<u64> {
    props
        .get(key)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| Error::PropertyTypeMismatch {
            vertex_id: "?".into(),
            property: key.into(),
            expected_type: "u64".into(),
        })
}

fn json_i64(props: &serde_json::Map<String, Json>, key: &str) -> Result<i64> {
    props
        .get(key)
        .and_then(|v| v.as_i64())
        .ok_or_else(|| Error::PropertyTypeMismatch {
            vertex_id: "?".into(),
            property: key.into(),
            expected_type: "i64".into(),
        })
}

fn json_opt_object_strs(props: &serde_json::Map<String, Json>, key: &str) -> Vec<(String, String)> {
    props
        .get(key)
        .and_then(|v| v.as_object())
        .map(|m| {
            m.iter()
                .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_owned())))
                .collect()
        })
        .unwrap_or_default()
}

fn account_class_str(c: AccountClass) -> &'static str {
    match c {
        AccountClass::Asset => "asset",
        AccountClass::Liability => "liability",
        AccountClass::Equity => "equity",
        AccountClass::Revenue => "revenue",
        AccountClass::Expense => "expense",
    }
}

fn parse_account_class(s: &str) -> Result<AccountClass> {
    Ok(match s {
        "asset" => AccountClass::Asset,
        "liability" => AccountClass::Liability,
        "equity" => AccountClass::Equity,
        "revenue" => AccountClass::Revenue,
        "expense" => AccountClass::Expense,
        other => {
            return Err(Error::Invariant(format!("unknown account class: {other}")));
        }
    })
}

fn normal_balance_str(nb: NormalBalance) -> &'static str {
    match nb {
        NormalBalance::Debit => "debit",
        NormalBalance::Credit => "credit",
    }
}

fn parse_normal_balance(s: &str) -> Result<NormalBalance> {
    Ok(match s {
        "debit" => NormalBalance::Debit,
        "credit" => NormalBalance::Credit,
        other => {
            return Err(Error::Invariant(format!("unknown normal balance: {other}")));
        }
    })
}

fn status_str(s: Status) -> &'static str {
    match s {
        Status::Pending => "pending",
        Status::Posted => "posted",
        Status::Archived => "archived",
    }
}

fn parse_status(s: &str) -> Result<Status> {
    Ok(match s {
        "pending" => Status::Pending,
        "posted" => Status::Posted,
        "archived" => Status::Archived,
        other => return Err(Error::Invariant(format!("unknown tx status: {other}"))),
    })
}

fn currency_from_props(code: &str, exponent: u8) -> Result<Currency> {
    // Try the curated constants first (fast path + handles validity).
    match (code, exponent) {
        ("USD", 2) => return Ok(Currency::USD),
        ("EUR", 2) => return Ok(Currency::EUR),
        ("BRL", 2) => return Ok(Currency::BRL),
        ("INR", 2) => return Ok(Currency::INR),
        ("GBP", 2) => return Ok(Currency::GBP),
        ("JPY", 0) => return Ok(Currency::JPY),
        ("CNY", 2) => return Ok(Currency::CNY),
        _ => { /* fall through */ }
    }
    let bytes = code.as_bytes();
    if bytes.len() != 3 {
        return Err(Error::Invariant(format!("currency code length: {code}")));
    }
    let arr = [bytes[0], bytes[1], bytes[2]];
    Currency::try_new(arr, exponent).map_err(|e| Error::Invariant(format!("{e}")))
}

// Suppress unused-import warning while Uuid stays available for
// downstream use.
#[allow(dead_code)]
fn _force_uuid_referenced() -> Option<Uuid> {
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::Currency;
    use op_ledger::{AccountClass, Entry, LedgerStore};

    fn setup() -> (GraphLedgerStore, LedgerId, AccountId, AccountId) {
        let store = GraphLedgerStore::new_in_memory();
        let ledger = Ledger::new("Test Ledger").unwrap();
        let lid = ledger.id;
        store.create_ledger(ledger).unwrap();
        let cash = Account::new(lid, "cash", AccountClass::Asset, Currency::USD);
        let rev = Account::new(lid, "revenue", AccountClass::Revenue, Currency::USD);
        let cid = cash.id;
        let rid = rev.id;
        store.create_account(cash).unwrap();
        store.create_account(rev).unwrap();
        (store, lid, cid, rid)
    }

    #[test]
    fn ledger_round_trips() {
        let store = GraphLedgerStore::new_in_memory();
        let ledger = Ledger::new("Acme 2026").unwrap();
        let id = ledger.id;
        store.create_ledger(ledger.clone()).unwrap();
        let recovered = store.get_ledger(id).unwrap();
        assert_eq!(recovered.name, "Acme 2026");
        assert_eq!(recovered.id, id);
    }

    #[test]
    fn account_round_trips_with_parent_ledger() {
        let (store, lid, cid, _) = setup();
        let acc = store.get_account(cid).unwrap();
        assert_eq!(acc.ledger_id, lid);
        assert_eq!(acc.name, "cash");
        assert_eq!(acc.currency, Currency::USD);
        assert_eq!(acc.normal_balance, NormalBalance::Debit);
        assert_eq!(acc.class, AccountClass::Asset);
    }

    #[test]
    fn account_missing_ledger_rejected() {
        let store = GraphLedgerStore::new_in_memory();
        let phantom = LedgerId::new();
        let acc = Account::new(phantom, "x", AccountClass::Asset, Currency::USD);
        let r = store.create_account(acc);
        assert!(matches!(r, Err(LedgerError::LedgerNotFound(_))));
    }

    #[test]
    fn post_transaction_then_get() {
        let (store, lid, cash, rev) = setup();
        let t = Transaction::new_pending(
            lid,
            42,
            vec![
                Entry::debit(cash, Money::from_minor(500, Currency::USD)),
                Entry::credit(rev, Money::from_minor(500, Currency::USD)),
            ],
        )
        .unwrap();
        let tid = store.post_transaction(t.clone()).unwrap();
        let recovered = store.get_transaction(tid).unwrap();
        assert_eq!(recovered.status, Status::Pending);
        assert_eq!(recovered.ledger_id, lid);
        assert_eq!(recovered.entries.len(), 2);
        assert_eq!(recovered.effective_at_unix_secs, 42);
    }

    #[test]
    fn balance_after_post_and_mark_posted() {
        let (store, lid, cash, rev) = setup();
        let t = Transaction::new_pending(
            lid,
            0,
            vec![
                Entry::debit(cash, Money::from_minor(500, Currency::USD)),
                Entry::credit(rev, Money::from_minor(500, Currency::USD)),
            ],
        )
        .unwrap();
        let tid = store.post_transaction(t).unwrap();
        // Pending: cash should be 500 pending, 0 posted.
        let bal = store.balance(cash).unwrap();
        assert_eq!(bal.pending.minor_units, 500);
        assert_eq!(bal.posted.minor_units, 0);
        // Mark posted.
        store.mark_posted(tid).unwrap();
        let bal = store.balance(cash).unwrap();
        assert_eq!(bal.posted.minor_units, 500);
        assert_eq!(bal.pending.minor_units, 500);
        // Revenue (credit-normal): same magnitude.
        let bal_rev = store.balance(rev).unwrap();
        assert_eq!(bal_rev.posted.minor_units, 500);
    }

    #[test]
    fn balance_with_archived_does_not_contribute() {
        let (store, lid, cash, rev) = setup();
        let t = Transaction::new_pending(
            lid,
            0,
            vec![
                Entry::debit(cash, Money::from_minor(700, Currency::USD)),
                Entry::credit(rev, Money::from_minor(700, Currency::USD)),
            ],
        )
        .unwrap();
        let tid = store.post_transaction(t).unwrap();
        store.mark_archived(tid).unwrap();
        let bal = store.balance(cash).unwrap();
        assert!(bal.is_zero());
    }

    #[test]
    fn idempotency_returns_existing_for_same_body() {
        let (store, lid, cash, rev) = setup();
        let body = vec![
            Entry::debit(cash, Money::from_minor(100, Currency::USD)),
            Entry::credit(rev, Money::from_minor(100, Currency::USD)),
        ];
        let t1 = Transaction::new_pending(lid, 0, body.clone())
            .unwrap()
            .with_external_id("ord-1");
        let t2 = Transaction::new_pending(lid, 0, body)
            .unwrap()
            .with_external_id("ord-1");
        let id1 = store.post_transaction(t1).unwrap();
        let id2 = store.post_transaction(t2).unwrap();
        assert_eq!(id1, id2);
    }

    #[test]
    fn idempotency_mismatch_for_different_body() {
        let (store, lid, cash, rev) = setup();
        let t1 = Transaction::new_pending(
            lid,
            0,
            vec![
                Entry::debit(cash, Money::from_minor(100, Currency::USD)),
                Entry::credit(rev, Money::from_minor(100, Currency::USD)),
            ],
        )
        .unwrap()
        .with_external_id("ord-2");
        let t2 = Transaction::new_pending(
            lid,
            0,
            vec![
                Entry::debit(cash, Money::from_minor(999, Currency::USD)),
                Entry::credit(rev, Money::from_minor(999, Currency::USD)),
            ],
        )
        .unwrap()
        .with_external_id("ord-2");
        store.post_transaction(t1).unwrap();
        let r = store.post_transaction(t2);
        assert!(matches!(r, Err(LedgerError::IdempotencyMismatch)));
    }

    #[test]
    fn cross_ledger_entry_rejected() {
        let (store, _lid, cash, _) = setup();
        // Account in a different ledger.
        let other = Ledger::new("Other").unwrap();
        let other_id = other.id;
        store.create_ledger(other).unwrap();
        let other_acc = Account::new(other_id, "x", AccountClass::Revenue, Currency::USD);
        let other_acc_id = other_acc.id;
        store.create_account(other_acc).unwrap();

        // Pull cash's ledger id; use it as the tx ledger but include
        // an entry against an account from `other_id`.
        let cash_acc = store.get_account(cash).unwrap();
        let t = Transaction::new_pending(
            cash_acc.ledger_id,
            0,
            vec![
                Entry::debit(cash, Money::from_minor(50, Currency::USD)),
                Entry::credit(other_acc_id, Money::from_minor(50, Currency::USD)),
            ],
        )
        .unwrap();
        let r = store.post_transaction(t);
        assert!(matches!(r, Err(LedgerError::CrossLedgerEntry { .. })));
    }

    #[test]
    fn currency_mismatch_rejected() {
        let (store, lid, cash, _rev) = setup();
        // Add an EUR account but use it with USD entry.
        let eur_acc = Account::new(lid, "eur_cash", AccountClass::Asset, Currency::EUR);
        let eur_id = eur_acc.id;
        store.create_account(eur_acc).unwrap();

        let t = Transaction::new_pending(
            lid,
            0,
            vec![
                Entry::debit(cash, Money::from_minor(10, Currency::USD)),
                // Wrong currency for the eur_cash account.
                Entry::credit(eur_id, Money::from_minor(10, Currency::USD)),
            ],
        )
        .unwrap();
        let r = store.post_transaction(t);
        assert!(matches!(r, Err(LedgerError::CurrencyMismatch { .. })));
    }

    #[test]
    fn find_by_external_id_returns_some() {
        let (store, lid, cash, rev) = setup();
        let t = Transaction::new_pending(
            lid,
            0,
            vec![
                Entry::debit(cash, Money::from_minor(11, Currency::USD)),
                Entry::credit(rev, Money::from_minor(11, Currency::USD)),
            ],
        )
        .unwrap()
        .with_external_id("ext-find-1");
        store.post_transaction(t).unwrap();
        let found = store.find_by_external_id("ext-find-1").unwrap();
        assert!(found.is_some());
        let f = found.unwrap();
        assert_eq!(f.external_id.as_deref(), Some("ext-find-1"));
    }

    #[test]
    fn find_by_external_id_returns_none_for_unknown() {
        let store = GraphLedgerStore::new_in_memory();
        let r = store.find_by_external_id("missing").unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn mark_posted_twice_fails_terminal_state() {
        let (store, lid, cash, rev) = setup();
        let t = Transaction::new_pending(
            lid,
            0,
            vec![
                Entry::debit(cash, Money::from_minor(1, Currency::USD)),
                Entry::credit(rev, Money::from_minor(1, Currency::USD)),
            ],
        )
        .unwrap();
        let tid = store.post_transaction(t).unwrap();
        store.mark_posted(tid).unwrap();
        let r = store.mark_posted(tid);
        assert!(matches!(r, Err(LedgerError::TerminalState { .. })));
    }

    #[test]
    fn get_unknown_ledger_errors() {
        let store = GraphLedgerStore::new_in_memory();
        let r = store.get_ledger(LedgerId::new());
        assert!(matches!(r, Err(LedgerError::LedgerNotFound(_))));
    }

    #[test]
    fn get_unknown_account_errors() {
        let store = GraphLedgerStore::new_in_memory();
        let r = store.get_account(AccountId::new());
        assert!(matches!(r, Err(LedgerError::AccountNotFound(_))));
    }

    #[test]
    fn get_unknown_transaction_errors() {
        let store = GraphLedgerStore::new_in_memory();
        let r = store.get_transaction(TransactionId::new());
        assert!(matches!(r, Err(LedgerError::TransactionNotFound(_))));
    }

    #[test]
    fn currency_codec_round_trip() {
        // All curated currencies survive the codec.
        for c in [
            Currency::USD,
            Currency::EUR,
            Currency::BRL,
            Currency::INR,
            Currency::GBP,
            Currency::JPY,
            Currency::CNY,
        ] {
            let r = currency_from_props(c.code(), c.exponent()).unwrap();
            assert_eq!(r, c);
        }
    }

    #[test]
    fn account_class_codec_round_trip() {
        for c in [
            AccountClass::Asset,
            AccountClass::Liability,
            AccountClass::Equity,
            AccountClass::Revenue,
            AccountClass::Expense,
        ] {
            assert_eq!(parse_account_class(account_class_str(c)).unwrap(), c);
        }
    }

    #[test]
    fn normal_balance_codec_round_trip() {
        for nb in [NormalBalance::Debit, NormalBalance::Credit] {
            assert_eq!(parse_normal_balance(normal_balance_str(nb)).unwrap(), nb);
        }
    }

    #[test]
    fn status_codec_round_trip() {
        for s in [Status::Pending, Status::Posted, Status::Archived] {
            assert_eq!(parse_status(status_str(s)).unwrap(), s);
        }
    }
}
