# OpenPay — observability

`op-server` uses `tracing` end-to-end. Every interesting operation
is wrapped in an instrumented span; spans carry structured fields
(amount, currency, driver, idempotency key, batch id, ...). What
you do with that output is a configuration choice.

## Log levels

Set via `RUST_LOG`, parsed by `tracing-subscriber::EnvFilter` in
`crates/op-server/src/main.rs`. Default when unset:

```
RUST_LOG=info,op_server=debug
```

Per-crate overrides work as you'd expect:

```
RUST_LOG=warn,op_server=debug,op_orchestrator=trace,op_webhook=debug
```

## JSON output for ingestion

For journald → loki / Datadog / Honeycomb / OpenTelemetry
collectors, you want newline-delimited JSON. The shipped `main.rs`
uses the default pretty subscriber and **does not** branch on
`OP_LOG_FORMAT`. Patch `main.rs` to read it:

```rust
// crates/op-server/src/main.rs
let want_json = std::env::var("OP_LOG_FORMAT")
    .map(|v| v.eq_ignore_ascii_case("json"))
    .unwrap_or(false);

let base = tracing_subscriber::fmt()
    .with_env_filter(
        EnvFilter::try_from_default_env()
            .unwrap_or_else(|_| EnvFilter::new("info,op_server=debug")),
    );

if want_json {
    base.json().init();
} else {
    base.init();
}
```

(Add `tracing-subscriber` with the `json` feature in
`crates/op-server/Cargo.toml` if it isn't already pulled in.)

With `OP_LOG_FORMAT=json`, every line on stdout is a parseable JSON
object: timestamp, level, target, span path, fields. journald
preserves these; promtail / vector / fluent-bit picks them up
unchanged.

## Spans worth dashboarding

These are the `tracing::instrument` spans the workspace currently
emits. Grep `crates/op-*/src/` for `tracing::instrument` to find
them in source.

| Span | Where | Key fields |
|---|---|---|
| `orchestrator.run` | `op-orchestrator/src/engine.rs` | `idempotency_key`, `amount_minor`, `currency` |
| `orchestrator.resume` | `op-orchestrator/src/engine.rs` | `idempotency_key`, `driver`, `rail` |
| `webhook.dispatch` | `op-webhook/src/dispatcher.rs` | `event_id`, `event_type`, `payload_bytes` |
| `ledger.post_transaction` | `op-ledger/src/store.rs` | `tx_id`, `ledger_id`, `external_id` |
| `settlement.open_batch` | `op-settlement/src/engine.rs` | `currency`, `rail` |
| `settlement.add_entry` | `op-settlement/src/engine.rs` | `batch_id` |
| `settlement.close_batch` | `op-settlement/src/engine.rs` | `batch_id`, `dispute_adjustment_bps` |
| `reconciliation.reconcile` | `op-reconciliation/src/engine.rs` | `window_start`, `window_end`, `tx_count` |

Useful dashboards to build:

- **Orchestrator latency** — p50/p95/p99 duration of `orchestrator.run`, broken out by `currency`. Spikes correlate with rail issues.
- **Webhook delivery health** — rate of `webhook.dispatch` events grouped by `event_type` and outcome. Failures (visible at `WARN` or `ERROR` level inside the span) feed dunning on receiver outages.
- **Ledger throughput** — count of `ledger.post_transaction` per minute. The system-of-record write rate. A sudden drop while `orchestrator.run` keeps rolling means writes are stuck.
- **Settlement cycle time** — gap between `settlement.open_batch` and the matching `settlement.close_batch` (correlate on `batch_id`).
- **Reconciliation lag** — frequency of `reconciliation.reconcile` and the `tx_count` per run.

## Request tracing

`tower_http::trace::TraceLayer` is layered in `main.rs`. Every
HTTP request gets a span with `method`, `uri`, `status`, `latency`.
If you're running behind Caddy, correlate Caddy's access-log
`request_id` field with the request-scope span fields by adding a
header-extracting middleware — Axum's `tower-http::request_id`
will do it in a few lines.

## journald + Loki pipeline

If you're on systemd (the recommended deploy mode in
`openpay.service`):

```bash
journalctl -u openpay -f                # tail live
journalctl -u openpay --since "1h ago"  # last hour
journalctl -u openpay -p err            # errors only
```

To ship to Loki, run `promtail` reading the systemd journal:

```yaml
# /etc/promtail/config.yml (excerpt)
scrape_configs:
  - job_name: openpay
    journal:
      max_age: 12h
      labels:
        job: openpay
    relabel_configs:
      - source_labels: ['__journal__systemd_unit']
        target_label: 'unit'
    pipeline_stages:
      - json:
          expressions:
            level: level
            target: target
            span: 'spans[0].name'
```

With `OP_LOG_FORMAT=json` set, the `json` pipeline stage parses
each line and exposes `level`, `target`, span names as Loki
labels — query by `{job="openpay", level="error"}` etc.

## Datadog / Honeycomb / generic OTLP

For OpenTelemetry export, drop `tracing-opentelemetry` into your
patched `main.rs`. The instrumented spans translate directly into
OTLP spans with attributes; correlation IDs (`idempotency_key`,
`batch_id`, `tx_id`) ride through as span attributes for free.
This is operator-side glue; the workspace deliberately doesn't
ship an opinionated OTLP setup.

## Health and readiness

Two endpoints, both ungated by auth (or behind `with_bypass_paths`
if you turned on `ApiKeyAuthLayer`):

```
GET /health      → 200 always (liveness; "is the process up?")
GET /readiness   → 200 if stores answer; 503 otherwise
```

Wire both into your load balancer / orchestrator. `/health` for
restart decisions, `/readiness` for "should we route traffic
here?" Caddy's `health_uri /health` in `Caddyfile.sample` uses the
first.

## Metrics?

There is no Prometheus / StatsD endpoint shipped today. The
`tracing` event stream is the source of truth — derive metrics
from it via `tracing-subscriber`'s metrics layer, or scrape your
log aggregator. If you need a `/metrics` endpoint badly enough to
add it, the right place is a new route in
`crates/op-server/src/routes.rs` backed by `metrics-exporter-prometheus`
and a few `metrics::counter!` calls sprinkled where the
`tracing::instrument` spans already sit.
