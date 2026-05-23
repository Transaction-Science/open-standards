//! # `op-cli` — operator CLI for `OpenPay`
//!
//! A thin command-line front-end to a running [`op_server`]
//! deployment. Operators reach a deployment over plain HTTP and
//! issue read / write requests against the same JSON surface
//! merchants use; this crate is the missing terminal ergonomics
//! layer.
//!
//! The library half exposes:
//!
//! - [`Cli`]: the top-level clap parser (derive style). Embedders
//!   that want to extend the command tree can re-use it.
//! - [`Client`]: a [`reqwest::blocking::Client`] wrapper that
//!   handles base URL + optional `OP_API_KEY` bearer auth and
//!   pretty-prints non-2xx responses.
//! - [`run`]: the entry point the binary calls; takes a parsed
//!   [`Cli`] and either succeeds (printing pretty JSON to stdout)
//!   or returns an [`Error`].
//!
//! No async runtime is pulled in — the blocking `reqwest` client
//! keeps the CLI snappy and shrinks dependency surface.
//!
//! ## Environment
//!
//! - `OP_SERVER_URL` — default `http://127.0.0.1:8080`. Overridden
//!   by `--server`.
//! - `OP_API_KEY` — optional bearer token. Overridden by
//!   `--api-key`.

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![warn(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::missing_errors_doc)]
#![allow(clippy::missing_panics_doc)]

use std::time::Duration;

use clap::{Args, Parser, Subcommand};
use serde_json::{Value, json};

pub mod ledger;

/// Errors the CLI can emit. Surface in `Display` form to the user;
/// `main` maps these to exit code 1.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Network-layer failure reaching the configured server.
    #[error("couldn't reach {url}: {source}")]
    Network {
        /// The endpoint the CLI tried to hit.
        url: String,
        /// Underlying reqwest error.
        source: reqwest::Error,
    },
    /// Server responded but with a non-2xx status. Body is included
    /// verbatim so operators can see the server's error envelope.
    #[error("server returned {status}: {body}")]
    HttpStatus {
        /// HTTP status code.
        status: u16,
        /// Raw response body.
        body: String,
    },
    /// Response wasn't valid JSON (or otherwise couldn't be decoded).
    #[error("invalid response payload: {0}")]
    BadPayload(String),
}

/// Top-level CLI parser. Holds the shared connection options and a
/// single subcommand.
#[derive(Debug, Parser)]
#[command(
    name = "op",
    version,
    about = "OpenPay operator CLI",
    long_about = "Inspect and manage a running op-server deployment from the terminal.\n\nReads OP_SERVER_URL (default http://127.0.0.1:8080) and OP_API_KEY\nfrom the environment unless overridden by --server / --api-key."
)]
pub struct Cli {
    /// Base URL of the op-server deployment (no trailing slash).
    /// Falls back to `OP_SERVER_URL`, then `http://127.0.0.1:8080`.
    #[arg(
        long,
        env = "OP_SERVER_URL",
        default_value = "http://127.0.0.1:8080",
        global = true
    )]
    pub server: String,

    /// Optional bearer token sent as `Authorization: Bearer …`.
    /// Falls back to `OP_API_KEY`.
    #[arg(long, env = "OP_API_KEY", global = true)]
    pub api_key: Option<String>,

    /// The action to perform.
    #[command(subcommand)]
    pub command: Command,
}

/// Top-level subcommand surface.
#[derive(Debug, Subcommand)]
pub enum Command {
    /// `GET /health`. Liveness check.
    Health,
    /// `GET /readiness`. Store-counts readiness report.
    Readiness,
    /// Refund inspection and creation.
    #[command(subcommand)]
    Refund(RefundCommand),
    /// Dispute inspection.
    #[command(subcommand)]
    Dispute(DisputeCommand),
    /// Settlement batch inspection.
    #[command(subcommand)]
    Batch(BatchCommand),
    /// Subscription inspection.
    #[command(subcommand)]
    Subscription(SubscriptionCommand),
    /// Foreign-exchange quote / convert.
    #[command(subcommand)]
    Fx(FxCommand),
    /// Webhook endpoint management.
    #[command(subcommand)]
    Webhooks(WebhooksCommand),
    /// Audit window report.
    #[command(subcommand)]
    Audit(AuditCommand),
    /// Bi-temporal time-travel queries against the ledger substrate.
    #[command(subcommand)]
    Ledger(LedgerCommand),
}

/// `op ledger ...` subcommands.
#[derive(Debug, Subcommand)]
pub enum LedgerCommand {
    /// Point query at a `(valid_time, transaction_time)` coordinate.
    ///
    /// See [`crate::ledger::as_of`] for the bi-temporal semantics.
    AsOf(ledger::AsOfArgs),
}

/// `op refund ...` subcommands.
#[derive(Debug, Subcommand)]
pub enum RefundCommand {
    /// `GET /v1/refunds/{id}`.
    Get {
        /// Refund UUID.
        id: String,
    },
    /// `POST /v1/refunds`.
    Create(RefundCreateArgs),
}

/// Args for `op refund create`. Maps 1:1 onto the
/// `CreateRefundRequest` envelope in `op-server`.
#[derive(Debug, Args)]
pub struct RefundCreateArgs {
    /// Original ledger transaction id (UUID).
    #[arg(long = "tx-id")]
    pub tx_id: String,
    /// Amount in minor units (cents for USD).
    #[arg(long = "amount")]
    pub amount: i64,
    /// ISO 4217 currency code.
    #[arg(long = "currency")]
    pub currency: String,
    /// Refund reason code (`customer_request`, `duplicate_charge`, …).
    #[arg(long = "reason")]
    pub reason: String,
    /// Caller-supplied idempotency key.
    #[arg(long = "external-id")]
    pub external_id: Option<String>,
    /// Unix epoch seconds when the request was made (caller clock).
    #[arg(long = "requested-at-unix-secs")]
    pub requested_at_unix_secs: u64,
}

/// `op dispute ...` subcommands.
#[derive(Debug, Subcommand)]
pub enum DisputeCommand {
    /// `GET /v1/disputes/{id}`.
    Get {
        /// Dispute UUID.
        id: String,
    },
}

/// `op batch ...` subcommands.
#[derive(Debug, Subcommand)]
pub enum BatchCommand {
    /// `GET /v1/settlement/batches/{id}`.
    Get {
        /// Batch UUID.
        id: String,
    },
}

/// `op subscription ...` subcommands.
#[derive(Debug, Subcommand)]
pub enum SubscriptionCommand {
    /// `GET /v1/subscriptions/{id}`.
    Get {
        /// Subscription UUID.
        id: String,
    },
    /// `GET /v1/subscriptions?customer_ref=…`.
    List {
        /// Customer reference to list for.
        #[arg(long = "customer-ref")]
        customer_ref: String,
    },
}

/// `op fx ...` subcommands.
#[derive(Debug, Subcommand)]
pub enum FxCommand {
    /// `GET /v1/fx/quote?from=X&to=Y`.
    Quote {
        /// Source currency.
        #[arg(long)]
        from: String,
        /// Target currency.
        #[arg(long)]
        to: String,
    },
    /// `POST /v1/fx/convert`.
    Convert {
        /// Source currency.
        #[arg(long)]
        from: String,
        /// Target currency.
        #[arg(long)]
        to: String,
        /// Source amount in minor units.
        #[arg(long = "amount-minor")]
        amount_minor: i64,
    },
}

/// `op webhooks ...` subcommands.
#[derive(Debug, Subcommand)]
pub enum WebhooksCommand {
    /// `POST /v1/webhooks/endpoints`.
    Register {
        /// Delivery URL.
        #[arg(long)]
        url: String,
        /// Shared HMAC secret.
        #[arg(long)]
        secret: String,
        /// Event filter (repeatable). `*` matches everything.
        #[arg(long = "event", required = true, num_args = 1..)]
        event: Vec<String>,
    },
}

/// `op audit ...` subcommands.
#[derive(Debug, Subcommand)]
pub enum AuditCommand {
    /// `GET /v1/audit/report?start_tx=…&end_tx=…`.
    Window {
        /// Window start (tx count).
        #[arg(long = "start-tx")]
        start_tx: u64,
        /// Window end (tx count).
        #[arg(long = "end-tx")]
        end_tx: u64,
        /// Wall-clock seconds stamped on the generated report.
        /// Defaults to 0 if the caller doesn't care.
        #[arg(long = "generated-at-unix-secs", default_value_t = 0)]
        generated_at_unix_secs: u64,
    },
}

/// Thin reqwest wrapper that the subcommand dispatch reaches into.
///
/// `Client` owns a blocking [`reqwest::blocking::Client`] with a
/// short connect timeout, the configured base URL, and the optional
/// bearer token. `get` / `post` issue a request, surface 2xx
/// responses as parsed JSON, and turn anything else into an
/// [`Error`].
pub struct Client {
    base: String,
    api_key: Option<String>,
    http: reqwest::blocking::Client,
}

impl Client {
    /// Build a new client. `base` is the server's URL root (e.g.
    /// `http://127.0.0.1:8080`); trailing slashes are trimmed.
    #[must_use]
    pub fn new(base: impl Into<String>, api_key: Option<String>) -> Self {
        let base = base.into().trim_end_matches('/').to_owned();
        let http = reqwest::blocking::Client::builder()
            .connect_timeout(Duration::from_secs(5))
            .timeout(Duration::from_secs(30))
            .build()
            .expect("reqwest client build");
        Self {
            base,
            api_key,
            http,
        }
    }

    /// Resolve a relative path against the configured base.
    fn url(&self, path: &str) -> String {
        if path.starts_with('/') {
            format!("{}{}", self.base, path)
        } else {
            format!("{}/{}", self.base, path)
        }
    }

    /// Send an HTTP request and decode its 2xx response body as JSON.
    fn send(&self, req: reqwest::blocking::RequestBuilder, url: &str) -> Result<Value, Error> {
        let req = if let Some(key) = &self.api_key {
            req.bearer_auth(key)
        } else {
            req
        };
        let res = req.send().map_err(|e| Error::Network {
            url: url.to_owned(),
            source: e,
        })?;
        let status = res.status();
        let text = res
            .text()
            .map_err(|e| Error::BadPayload(format!("read body: {e}")))?;
        if !status.is_success() {
            return Err(Error::HttpStatus {
                status: status.as_u16(),
                body: text,
            });
        }
        if text.is_empty() {
            return Ok(Value::Null);
        }
        serde_json::from_str(&text).map_err(|e| Error::BadPayload(format!("parse json: {e}")))
    }

    /// `GET path`. Returns the parsed JSON body on 2xx.
    pub fn get(&self, path: &str) -> Result<Value, Error> {
        let url = self.url(path);
        self.send(self.http.get(&url), &url)
    }

    /// `POST path` with the given JSON body. Returns the parsed
    /// response body on 2xx.
    pub fn post(&self, path: &str, body: &Value) -> Result<Value, Error> {
        let url = self.url(path);
        self.send(self.http.post(&url).json(body), &url)
    }
}

/// Execute a parsed CLI invocation. Writes pretty-printed JSON to
/// stdout on success; returns the underlying [`Error`] otherwise.
///
/// Most subcommands speak HTTP to a running [`op_server`] deployment.
/// The `ledger` family is the exception — bi-temporal queries run
/// in-process against the local graph substrate, so they bypass
/// [`Client`] entirely and write their own output.
pub fn run(cli: Cli) -> Result<(), Error> {
    if let Command::Ledger(LedgerCommand::AsOf(args)) = cli.command {
        return ledger::run_as_of(&args);
    }
    let client = Client::new(cli.server, cli.api_key);
    let value = dispatch(&client, cli.command)?;
    print_json(&value);
    Ok(())
}

/// Pretty-print a JSON value to stdout. Falls back to a debug
/// rendering if serialization somehow fails (`serde_json` never
/// fails on already-parsed values, but the type signature forces
/// us to handle the result).
pub fn print_json(value: &Value) {
    match serde_json::to_string_pretty(value) {
        Ok(rendered) => println!("{rendered}"),
        Err(_) => println!("{value:?}"),
    }
}

/// Dispatch a single subcommand against `client`. Split out from
/// [`run`] so tests can drive it directly without intercepting
/// stdout.
pub fn dispatch(client: &Client, command: Command) -> Result<Value, Error> {
    match command {
        Command::Health => client.get("/health"),
        Command::Readiness => client.get("/readiness"),
        Command::Refund(RefundCommand::Get { id }) => {
            client.get(&format!("/v1/refunds/{id}"))
        }
        Command::Refund(RefundCommand::Create(a)) => {
            let body = json!({
                "original_tx_id": a.tx_id,
                "amount_minor": a.amount,
                "currency": a.currency,
                "reason": a.reason,
                "external_id": a.external_id,
                "requested_at_unix_secs": a.requested_at_unix_secs,
                "metadata": [],
            });
            client.post("/v1/refunds", &body)
        }
        Command::Dispute(DisputeCommand::Get { id }) => {
            client.get(&format!("/v1/disputes/{id}"))
        }
        Command::Batch(BatchCommand::Get { id }) => {
            client.get(&format!("/v1/settlement/batches/{id}"))
        }
        Command::Subscription(SubscriptionCommand::Get { id }) => {
            client.get(&format!("/v1/subscriptions/{id}"))
        }
        Command::Subscription(SubscriptionCommand::List { customer_ref }) => {
            // reqwest will URL-encode the query value for us.
            client.get(&format!("/v1/subscriptions?customer_ref={customer_ref}"))
        }
        Command::Fx(FxCommand::Quote { from, to }) => {
            client.get(&format!("/v1/fx/quote?from={from}&to={to}"))
        }
        Command::Fx(FxCommand::Convert { from, to, amount_minor }) => {
            let body = json!({
                "from": from,
                "to": to,
                "amount_minor": amount_minor,
            });
            client.post("/v1/fx/convert", &body)
        }
        Command::Webhooks(WebhooksCommand::Register { url, secret, event }) => {
            let body = json!({
                "url": url,
                "secret": secret,
                "event_filters": event,
            });
            client.post("/v1/webhooks/endpoints", &body)
        }
        Command::Audit(AuditCommand::Window { start_tx, end_tx, generated_at_unix_secs }) => {
            client.get(&format!(
                "/v1/audit/report?start_tx={start_tx}&end_tx={end_tx}&generated_at_unix_secs={generated_at_unix_secs}"
            ))
        }
        // Local-only — handled by `run` before dispatch. We never
        // reach this arm in practice; it exists to keep the match
        // exhaustive and clippy happy.
        Command::Ledger(_) => Err(Error::BadPayload(
            "ledger subcommands are dispatched in-process, not over HTTP".into(),
        )),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use clap::CommandFactory;

    /// `clap` itself can validate the command tree at build time.
    #[test]
    fn clap_command_tree_is_well_formed() {
        Cli::command().debug_assert();
    }

    #[test]
    fn parses_health() {
        let cli = Cli::try_parse_from(["op", "health"]).expect("parse");
        assert!(matches!(cli.command, Command::Health));
        assert_eq!(cli.server, "http://127.0.0.1:8080");
        assert!(cli.api_key.is_none());
    }

    #[test]
    fn parses_readiness() {
        let cli = Cli::try_parse_from(["op", "readiness"]).expect("parse");
        assert!(matches!(cli.command, Command::Readiness));
    }

    #[test]
    fn server_flag_overrides_default() {
        let cli = Cli::try_parse_from(["op", "--server", "http://1.2.3.4:9000", "health"])
            .expect("parse");
        assert_eq!(cli.server, "http://1.2.3.4:9000");
    }

    #[test]
    fn api_key_flag_overrides_env() {
        let cli = Cli::try_parse_from(["op", "--api-key", "secret123", "health"]).expect("parse");
        assert_eq!(cli.api_key.as_deref(), Some("secret123"));
    }

    #[test]
    fn parses_refund_get() {
        let cli = Cli::try_parse_from([
            "op",
            "refund",
            "get",
            "00000000-0000-0000-0000-000000000001",
        ])
        .expect("parse");
        match cli.command {
            Command::Refund(RefundCommand::Get { id }) => {
                assert_eq!(id, "00000000-0000-0000-0000-000000000001");
            }
            _ => panic!("wrong subcommand"),
        }
    }

    #[test]
    fn parses_refund_create() {
        let cli = Cli::try_parse_from([
            "op",
            "refund",
            "create",
            "--tx-id",
            "00000000-0000-0000-0000-000000000001",
            "--amount",
            "500",
            "--currency",
            "USD",
            "--reason",
            "customer_request",
            "--external-id",
            "ext-1",
            "--requested-at-unix-secs",
            "1700000000",
        ])
        .expect("parse");
        match cli.command {
            Command::Refund(RefundCommand::Create(a)) => {
                assert_eq!(a.amount, 500);
                assert_eq!(a.currency, "USD");
                assert_eq!(a.reason, "customer_request");
                assert_eq!(a.external_id.as_deref(), Some("ext-1"));
                assert_eq!(a.requested_at_unix_secs, 1_700_000_000);
            }
            _ => panic!("wrong subcommand"),
        }
    }

    #[test]
    fn parses_dispute_get() {
        let cli = Cli::try_parse_from([
            "op",
            "dispute",
            "get",
            "00000000-0000-0000-0000-000000000002",
        ])
        .expect("parse");
        assert!(matches!(
            cli.command,
            Command::Dispute(DisputeCommand::Get { .. })
        ));
    }

    #[test]
    fn parses_batch_get() {
        let cli =
            Cli::try_parse_from(["op", "batch", "get", "00000000-0000-0000-0000-000000000003"])
                .expect("parse");
        assert!(matches!(
            cli.command,
            Command::Batch(BatchCommand::Get { .. })
        ));
    }

    #[test]
    fn parses_subscription_get_and_list() {
        let g = Cli::try_parse_from([
            "op",
            "subscription",
            "get",
            "00000000-0000-0000-0000-000000000004",
        ])
        .expect("parse get");
        assert!(matches!(
            g.command,
            Command::Subscription(SubscriptionCommand::Get { .. })
        ));

        let l = Cli::try_parse_from(["op", "subscription", "list", "--customer-ref", "cust-42"])
            .expect("parse list");
        match l.command {
            Command::Subscription(SubscriptionCommand::List { customer_ref }) => {
                assert_eq!(customer_ref, "cust-42");
            }
            _ => panic!("wrong subcommand"),
        }
    }

    #[test]
    fn parses_fx_quote_and_convert() {
        let q = Cli::try_parse_from(["op", "fx", "quote", "--from", "USD", "--to", "EUR"])
            .expect("parse quote");
        match q.command {
            Command::Fx(FxCommand::Quote { from, to }) => {
                assert_eq!(from, "USD");
                assert_eq!(to, "EUR");
            }
            _ => panic!("wrong subcommand"),
        }

        let c = Cli::try_parse_from([
            "op",
            "fx",
            "convert",
            "--from",
            "USD",
            "--to",
            "EUR",
            "--amount-minor",
            "12345",
        ])
        .expect("parse convert");
        match c.command {
            Command::Fx(FxCommand::Convert {
                from,
                to,
                amount_minor,
            }) => {
                assert_eq!(from, "USD");
                assert_eq!(to, "EUR");
                assert_eq!(amount_minor, 12_345);
            }
            _ => panic!("wrong subcommand"),
        }
    }

    #[test]
    fn parses_webhooks_register_multi_event() {
        let cli = Cli::try_parse_from([
            "op",
            "webhooks",
            "register",
            "--url",
            "https://example.invalid/hook",
            "--secret",
            "topsecret",
            "--event",
            "refund.created",
            "--event",
            "dispute.created",
        ])
        .expect("parse");
        match cli.command {
            Command::Webhooks(WebhooksCommand::Register { url, secret, event }) => {
                assert_eq!(url, "https://example.invalid/hook");
                assert_eq!(secret, "topsecret");
                assert_eq!(event, vec!["refund.created", "dispute.created"]);
            }
            _ => panic!("wrong subcommand"),
        }
    }

    #[test]
    fn parses_audit_window() {
        let cli = Cli::try_parse_from([
            "op",
            "audit",
            "window",
            "--start-tx",
            "10",
            "--end-tx",
            "20",
        ])
        .expect("parse");
        match cli.command {
            Command::Audit(AuditCommand::Window {
                start_tx,
                end_tx,
                generated_at_unix_secs,
            }) => {
                assert_eq!(start_tx, 10);
                assert_eq!(end_tx, 20);
                assert_eq!(generated_at_unix_secs, 0);
            }
            _ => panic!("wrong subcommand"),
        }
    }

    #[test]
    fn client_trims_trailing_slash() {
        let c = Client::new("http://127.0.0.1:8080/", None);
        assert_eq!(c.url("/health"), "http://127.0.0.1:8080/health");
    }

    #[test]
    fn missing_subcommand_fails() {
        let err = Cli::try_parse_from(["op"]).expect_err("must require subcommand");
        // clap returns a `DisplayHelp` / `MissingSubcommand` error
        // when no subcommand is supplied — either way it is not `Ok`.
        let kind = err.kind();
        assert!(
            matches!(
                kind,
                clap::error::ErrorKind::MissingSubcommand
                    | clap::error::ErrorKind::DisplayHelp
                    | clap::error::ErrorKind::DisplayHelpOnMissingArgumentOrSubcommand
            ),
            "unexpected error kind: {kind:?}"
        );
    }
}
