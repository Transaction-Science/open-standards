# OpenPay Observability

The bi-temporal append-only ledger is one of OpenPay's architectural
moats: every state transition is a recorded event with both a
*valid-time* (when the event happened in the real world) and a
*transaction-time* (when the ledger learned of it). Inspecting that
moat requires three things: a CLI for ad-hoc queries, OpenTelemetry
traces over the live `Payment<S>` typestate, and a Grafana dashboard
to watch the system from the operations side.

This directory holds the second and third.

## Wiring up

### 1. OpenTelemetry collector

```yaml
# otel-collector.yaml
receivers:
  otlp:
    protocols:
      grpc:
        endpoint: 0.0.0.0:4317
      http:
        endpoint: 0.0.0.0:4318
exporters:
  prometheus:
    endpoint: 0.0.0.0:9464
service:
  pipelines:
    metrics:
      receivers:  [otlp]
      exporters:  [prometheus]
    traces:
      receivers:  [otlp]
      exporters:  [prometheus]
```

### 2. OpenPay process

Compile `op-orchestrator` with the `telemetry` feature and set
`OPENPAY_OTLP_ENDPOINT=http://localhost:4317` in the environment.
`op_orchestrator::telemetry::init_telemetry(...)` wires the global
subscriber at startup. Every `Payment<S>` state transition emits a
span with these attributes:

| Attribute             | Type    | Example                       |
|-----------------------|---------|-------------------------------|
| `payment.id`          | string  | `pay_019e30baa...`            |
| `payment.state.from`  | string  | `Authorized`                  |
| `payment.state.to`    | string  | `Captured`                    |
| `payment.rail`        | string  | `card` · `a2a` · `crypto`     |
| `payment.amount.minor`| int     | `1299`                        |
| `payment.amount.currency` | string | `USD`                       |

### 3. Prometheus + Grafana

Scrape the collector's prometheus exporter at `:9464`. Import
[`grafana-dashboard.json`](grafana-dashboard.json) into Grafana. The
dashboard ships with three template variables: `$rail`, `$currency`,
and the standard `$time_range`.

## Example bi-temporal query

> What was account `acct_019e3...`'s balance on **May 4** as we knew
> it on **May 5**?

```bash
op ledger as-of \
  --valid 2026-05-04T00:00:00Z \
  --transaction 2026-05-05T00:00:00Z \
  --account acct_019e30baa9507793b467ac644636b3e2 \
  --human
```

Output (with `--human`):

```
account                   acct_019e30baa9507793b467ac644636b3e2
valid-time                2026-05-04T00:00:00Z
transaction-time          2026-05-05T00:00:00Z
balance                   12_450 minor USD
entries-cited             487
```

Strip `--human` to get the same content as structured JSON for
scripting and reconciliation tooling.

## What the dashboard shows

- **State-transition rate** — transitions per second by `(from, to)`
  pair. Spikes on `Authorized → Captured` are the healthy steady state;
  `* → Refunded` rate is the dispute / refund pulse.
- **In-flight by state** — current count of payments in each typestate.
  Long tails in `Authorized` indicate captures that aren't landing.
- **Refund vs capture ratio** — short-window and long-window. Spikes
  beyond a configurable threshold can be wired into the alert manager.
- **Rail split** — card vs A2A vs crypto, by volume and by count.
- **Latency per transition** — p50 / p95 / p99 per state transition,
  bucketed by rail.
- **Error rate per state** — counts of `Err(...)` returns from each
  state-transition function, bucketed by rail and error class.

## Recompute history

The bi-temporal property of the ledger is the real magic: when an old
reconciliation finds a discrepancy in last week's data, recording the
correction does not rewrite history. It appends a new entry with
*valid-time = last week, transaction-time = now*. The CLI query above
will return the correction when asked from "now" forward and the
original (uncorrected) state when asked from before "now". The
dashboard's data source does the same.
