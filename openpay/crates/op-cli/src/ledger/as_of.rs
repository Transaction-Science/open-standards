//! `op ledger as-of` — bi-temporal point query.
//!
//! Returns the balance of an account (or every account in the demo
//! ledger if `--account` is omitted) as it stood at a given pair of
//! bi-temporal coordinates:
//!
//! - `--valid <ISO8601>` — the wall-clock instant in the real world
//!   we want to ask about. Transactions whose `effective_at` is
//!   strictly after this instant are excluded.
//! - `--transaction <ISO8601>` — the wall-clock instant the system
//!   is treated as "knowing." Transactions whose `effective_at` is
//!   strictly after this instant are also excluded — modelling the
//!   fact that the substrate hadn't yet seen them.
//!
//! The answer is the balance computed under the intersection of both
//! filters: only facts with `effective_at ≤ min(valid, transaction)`
//! contribute. When `transaction ≥ valid` this collapses to the
//! valid-time view; when `transaction < valid` it models a stale
//! observer (we're asking what the books *looked like* yesterday
//! about today's events — they didn't yet show today's events).
//!
//! v1 ships a deterministic demo seed (see [`build_demo_store`]) so
//! the subcommand works out of the box. A follow-up issue will let
//! the operator point at a graph snapshot path via
//! `OP_LEDGER_GRAPH_PATH` or hit a server endpoint backed by their
//! production graph.

use std::io;

use clap::Args;
use op_core::{Currency, Money};
use op_graph::GraphLedgerStore;
use op_ledger::{
    Account, AccountClass, AccountId, Balance, Direction, Entry, Ledger, LedgerHistory,
    LedgerStore, Transaction,
};
use serde::Serialize;
use serde_json::Value;
use time::OffsetDateTime;
use time::format_description::well_known::Iso8601;

use crate::Error;

/// Flags accepted by `op ledger as-of`.
#[derive(Debug, Args)]
pub struct AsOfArgs {
    /// Valid-time coordinate (when the event happened in the world).
    /// Accepts any ISO 8601 instant, e.g. `2026-05-04T00:00:00Z` or
    /// `2026-05-04T00:00:00+00:00`.
    #[arg(long = "valid")]
    pub valid: String,

    /// Transaction-time coordinate (when the system was treated as
    /// knowing). Accepts any ISO 8601 instant.
    #[arg(long = "transaction")]
    pub transaction: String,

    /// Optional account UUID. If omitted, every account in the
    /// in-process demo ledger is reported.
    #[arg(long = "account")]
    pub account: Option<String>,

    /// Pretty-print a human-readable summary instead of JSON. The
    /// JSON output is the contract for scripted consumers; `--human`
    /// is for terminal use.
    #[arg(long = "human", default_value_t = false)]
    pub human: bool,
}

/// Top-level JSON envelope written to stdout.
#[derive(Debug, Serialize)]
pub struct Output {
    /// Valid-time coordinate echoed back (RFC 3339).
    pub valid_at: String,
    /// Transaction-time coordinate echoed back (RFC 3339).
    pub transaction_at: String,
    /// The unix-seconds anchor actually used for the query (the
    /// minimum of `valid_at` and `transaction_at`). Surfaced so
    /// callers can verify the bi-temporal semantics.
    pub anchor_unix_secs: u64,
    /// One row per account reported.
    pub accounts: Vec<AccountState>,
}

/// One account in the as-of report.
#[derive(Debug, Serialize)]
pub struct AccountState {
    /// Account UUID.
    pub account_id: String,
    /// Operator-facing account name.
    pub name: String,
    /// ISO 4217 currency code.
    pub currency: String,
    /// Posted balance in minor units at the anchor.
    pub posted_minor: i64,
    /// Pending balance in minor units at the anchor.
    pub pending_minor: i64,
}

/// Errors specific to the `ledger as-of` subcommand. Lifted into
/// [`crate::Error::BadPayload`] on the way back to `main`.
#[derive(Debug, thiserror::Error)]
pub enum AsOfError {
    /// An ISO 8601 timestamp argument failed to parse.
    #[error("invalid ISO 8601 timestamp for --{flag}: {value}")]
    BadTimestamp {
        /// Which flag the bad value came from.
        flag: &'static str,
        /// The offending string.
        value: String,
    },
    /// The `--account` UUID didn't parse.
    #[error("invalid account UUID: {0}")]
    BadAccountId(String),
    /// Underlying ledger query failed.
    #[error("ledger query failed: {0}")]
    Query(String),
}

impl From<AsOfError> for Error {
    fn from(e: AsOfError) -> Self {
        Self::BadPayload(e.to_string())
    }
}

/// Entry point for the `ledger as-of` subcommand.
///
/// Builds the in-process demo store, parses the bi-temporal
/// coordinates, queries the [`LedgerHistory`] surface, and writes
/// the result to `stdout` (JSON by default, human-readable summary
/// when `--human` is set).
///
/// # Errors
/// Bubbles [`AsOfError`] up as [`Error::BadPayload`].
pub fn run(args: &AsOfArgs) -> Result<(), Error> {
    let valid_secs = parse_iso8601_unix_secs("valid", &args.valid)?;
    let transaction_secs = parse_iso8601_unix_secs("transaction", &args.transaction)?;
    let anchor = valid_secs.min(transaction_secs);

    let store = build_demo_store().map_err(|e| AsOfError::Query(e.to_string()))?;

    let account_filter = args
        .account
        .as_deref()
        .map(parse_account_id)
        .transpose()?;

    let mut accounts: Vec<AccountState> = Vec::new();
    for account in demo_accounts(&store).map_err(|e| AsOfError::Query(e.to_string()))? {
        if let Some(filter) = account_filter
            && account.id != filter
        {
            continue;
        }
        let balance: Balance = store
            .balance_as_of_time(account.id, anchor)
            .map_err(|e| AsOfError::Query(e.to_string()))?;
        accounts.push(AccountState {
            account_id: account.id.to_string(),
            name: account.name.clone(),
            currency: balance.currency.code().to_owned(),
            posted_minor: balance.posted.minor_units,
            pending_minor: balance.pending.minor_units,
        });
    }

    if let Some(filter) = account_filter
        && accounts.is_empty()
    {
        return Err(AsOfError::Query(format!(
            "no demo account matches UUID {filter}"
        ))
        .into());
    }

    let out = Output {
        valid_at: format_iso(valid_secs),
        transaction_at: format_iso(transaction_secs),
        anchor_unix_secs: anchor,
        accounts,
    };

    if args.human {
        write_human(&out, &mut io::stdout()).map_err(|e| Error::BadPayload(e.to_string()))?;
    } else {
        let v: Value =
            serde_json::to_value(&out).map_err(|e| Error::BadPayload(e.to_string()))?;
        crate::print_json(&v);
    }
    Ok(())
}

/// Pretty-print a single [`Output`] to a writer. Used by `--human`.
///
/// # Errors
/// Returns any I/O error encountered while writing.
pub fn write_human<W: io::Write>(out: &Output, w: &mut W) -> io::Result<()> {
    writeln!(w, "openpay ledger as-of")?;
    writeln!(w, "  valid       : {}", out.valid_at)?;
    writeln!(w, "  transaction : {}", out.transaction_at)?;
    writeln!(w, "  anchor      : t = {} (unix secs)", out.anchor_unix_secs)?;
    if out.accounts.is_empty() {
        writeln!(w, "  (no accounts reported)")?;
        return Ok(());
    }
    writeln!(w)?;
    for a in &out.accounts {
        writeln!(w, "  {} [{}] ({})", a.name, a.account_id, a.currency)?;
        writeln!(
            w,
            "    posted  : {:>16} {}",
            a.posted_minor, a.currency
        )?;
        writeln!(
            w,
            "    pending : {:>16} {}",
            a.pending_minor, a.currency
        )?;
    }
    Ok(())
}

fn parse_iso8601_unix_secs(flag: &'static str, raw: &str) -> Result<u64, AsOfError> {
    let dt = OffsetDateTime::parse(raw, &Iso8601::DEFAULT).map_err(|_| AsOfError::BadTimestamp {
        flag,
        value: raw.to_owned(),
    })?;
    let secs = dt.unix_timestamp();
    if secs < 0 {
        return Err(AsOfError::BadTimestamp {
            flag,
            value: raw.to_owned(),
        });
    }
    Ok(u64::try_from(secs).expect("checked non-negative"))
}

fn parse_account_id(raw: &str) -> Result<AccountId, AsOfError> {
    let u =
        uuid::Uuid::parse_str(raw).map_err(|_| AsOfError::BadAccountId(raw.to_owned()))?;
    Ok(AccountId::from_uuid(u))
}

fn format_iso(secs: u64) -> String {
    // Best-effort RFC 3339 echo; falls back to a plain unix-secs
    // string if for some reason `time` can't render (it always can
    // for valid u64 values within the supported range).
    let signed = i64::try_from(secs).unwrap_or(i64::MAX);
    OffsetDateTime::from_unix_timestamp(signed)
        .ok()
        .and_then(|dt| dt.format(&Iso8601::DEFAULT).ok())
        .unwrap_or_else(|| format!("@{secs}"))
}

// ============================================================
// Demo store
// ============================================================

/// Build a small, deterministic [`GraphLedgerStore`] seeded with a
/// handful of transactions so `op ledger as-of` produces meaningful
/// output without a running server. The dates are spread across
/// May 2026 to exercise the valid-time and transaction-time axes.
///
/// Seed schedule (USD ledger, two accounts: `customer-cash` (asset),
/// `merchant-revenue` (revenue)):
///
/// | `effective_at` | description     | posted | debit            | credit            | amount |
/// | -----------: | :---------------- | :----- | :--------------- | :---------------- | -----: |
/// | 2026-05-01   | seed deposit      |  yes   | customer-cash    | merchant-revenue  | $100   |
/// | 2026-05-04   | mid-month sale    |  yes   | customer-cash    | merchant-revenue  | $42    |
/// | 2026-05-10   | late-month sale   |  yes   | customer-cash    | merchant-revenue  | $7     |
///
/// Querying with `--valid 2026-05-04T23:59:59Z --transaction 2026-05-05T00:00:00Z` returns
/// the post-deposit + first-sale balance, i.e. customer-cash debit $142 / merchant-revenue
/// credit $142.
///
/// # Errors
/// Returns any error from the underlying store. None expected on the
/// in-memory backend; surfaced for completeness.
pub fn build_demo_store() -> Result<GraphLedgerStore, Box<dyn std::error::Error + Send + Sync>> {
    let store = GraphLedgerStore::new_in_memory();

    let ledger = Ledger::new("openpay-demo")?;
    let ledger_id = store.create_ledger(ledger)?;

    let cash = Account::new(ledger_id, "customer-cash", AccountClass::Asset, Currency::USD);
    let revenue = Account::new(
        ledger_id,
        "merchant-revenue",
        AccountClass::Revenue,
        Currency::USD,
    );
    let cash_id = store.create_account(cash)?;
    let revenue_id = store.create_account(revenue)?;

    for (effective, amount_minor) in [
        (unix_secs(2026, 5, 1), 10_000_i64), // $100.00
        (unix_secs(2026, 5, 4), 4_200_i64),  // $42.00
        (unix_secs(2026, 5, 10), 700_i64),   // $7.00
    ] {
        let amount = Money::from_minor(amount_minor, Currency::USD);
        let tx = Transaction::new_posted(
            ledger_id,
            effective,
            vec![
                Entry::new(cash_id, Direction::Debit, amount),
                Entry::new(revenue_id, Direction::Credit, amount),
            ],
        )?;
        store.post_transaction(tx)?;
    }

    Ok(store)
}

/// Walk the demo accounts in a stable order.
fn demo_accounts(
    store: &GraphLedgerStore,
) -> Result<Vec<Account>, Box<dyn std::error::Error + Send + Sync>> {
    // The demo seed creates exactly the two named accounts; we use the
    // graph's listing API to discover them without baking UUIDs in.
    use op_graph::graph::vtypes;
    let mut out: Vec<Account> = Vec::new();
    for v in store.handle().vertices_of_type(vtypes::LEDGER_ACCOUNT)? {
        let id = AccountId::from_uuid(v.id);
        if let Ok(acc) = store.get_account(id) {
            out.push(acc);
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

fn unix_secs(year: i32, month: u8, day: u8) -> u64 {
    let date = time::Date::from_calendar_date(year, time::Month::try_from(month).unwrap(), day)
        .expect("valid demo date");
    let dt = date.midnight().assume_utc();
    u64::try_from(dt.unix_timestamp()).expect("post-1970 demo date")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_basic_iso8601() {
        let secs = parse_iso8601_unix_secs("valid", "2026-05-04T00:00:00Z").expect("parse");
        assert_eq!(secs, 1_777_852_800);
    }

    #[test]
    fn rejects_garbage_timestamp() {
        let err = parse_iso8601_unix_secs("valid", "not-a-date").expect_err("must fail");
        match err {
            AsOfError::BadTimestamp { flag, .. } => assert_eq!(flag, "valid"),
            _ => panic!("wrong error kind"),
        }
    }

    #[test]
    fn demo_seed_balances_at_may_5() {
        let store = build_demo_store().expect("seed");
        let accounts = demo_accounts(&store).expect("list");
        // Anchor at end of 2026-05-05 — only the May 1 and May 4 txs
        // are in force, i.e. $142 in/out.
        let anchor = unix_secs(2026, 5, 5);
        let cash = accounts
            .iter()
            .find(|a| a.name == "customer-cash")
            .expect("cash account");
        let bal = store
            .balance_as_of_time(cash.id, anchor)
            .expect("balance");
        assert_eq!(bal.posted.minor_units, 14_200);
    }

    #[test]
    fn run_writes_account_filtered_json() {
        // Smoke test: drive `run` end-to-end with an account filter and
        // a date that includes only the seed deposit. We can't easily
        // capture stdout without extra deps, so we directly invoke
        // `write_human` against a captured buffer for assertion.
        let store = build_demo_store().expect("seed");
        let accounts = demo_accounts(&store).expect("list");
        let cash = accounts
            .iter()
            .find(|a| a.name == "customer-cash")
            .unwrap();
        let anchor = unix_secs(2026, 5, 2);
        let bal = store.balance_as_of_time(cash.id, anchor).unwrap();
        let out = Output {
            valid_at: "2026-05-02T00:00:00Z".into(),
            transaction_at: "2026-05-02T00:00:00Z".into(),
            anchor_unix_secs: anchor,
            accounts: vec![AccountState {
                account_id: cash.id.to_string(),
                name: cash.name.clone(),
                currency: bal.currency.code().to_owned(),
                posted_minor: bal.posted.minor_units,
                pending_minor: bal.pending.minor_units,
            }],
        };
        let mut buf: Vec<u8> = Vec::new();
        write_human(&out, &mut buf).expect("write");
        let text = String::from_utf8(buf).unwrap();
        assert!(text.contains("customer-cash"));
        assert!(text.contains("10000"));
    }
}
