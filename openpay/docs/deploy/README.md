# Deploying OpenPay

Operator-side guide for running `op-server` in production. Read top
to bottom — each section assumes the previous one. The shipped
binary (`crates/op-server/src/main.rs`) is intentionally minimal: it
reads two env vars (`OP_BIND_ADDR`, `RUST_LOG`) and serves the
default in-memory `AppState`. Everything else — persistence, auth,
webhooks, FX, crypto rails — is wired by either editing
`crates/op-server/src/main.rs` or building your own binary around
the exported `op_server::router` / `op_server::router_with_middleware`
functions. The library APIs ship and are stable; the binary is a
reference entry point you are expected to extend.

Companion files in this directory:

| File | Purpose |
|---|---|
| `quickstart.md` | Five-minute clone → build → smoke-test path. |
| `Caddyfile.sample` | TLS-terminating reverse proxy in front of op-server. |
| `openpay.service` | systemd unit. |
| `openpay.env.sample` | Env-file template; covers every var the binary reads (and the ones your custom binary may add). |
| `backup.md` | Single-file `.graph` backup + restore. |
| `monitoring.md` | tracing spans, log format, and what to dashboard. |

---

## 1. Hello, world

The shipped binary listens on `127.0.0.1:8080` and holds state in
memory. No persistence, no auth, no TLS. Suitable for a smoke test.

```bash
cargo build --release -p op-server
OP_BIND_ADDR=127.0.0.1:8080 ./target/release/op-server
```

In another shell:

```bash
curl http://127.0.0.1:8080/health        # → 200 {"status":"ok"}
curl http://127.0.0.1:8080/readiness     # → 200 if stores answer
```

That's it for the default path. Everything below is operator
configuration on top.

---

## 2. Put it behind Caddy with TLS

`op-server` does not terminate TLS. Run Caddy (or nginx, or a load
balancer) in front. See `Caddyfile.sample` in this directory — one
`reverse_proxy 127.0.0.1:8080` block, ACME-issued certs for free.

Bind op-server to loopback so it's only reachable through the
reverse proxy:

```bash
OP_BIND_ADDR=127.0.0.1:8080 ./target/release/op-server
```

Then Caddy:

```bash
sudo caddy run --config /etc/caddy/Caddyfile
```

---

## 3. Persist to a real `.graph` file

The default `main.rs` calls `AppState::new_in_memory()`. To persist
the entire deployment — refunds, disputes, settlement batches,
ledger, reconciliation, webhooks, subscriptions, idempotency cache
— to a single Minigraf-backed file, swap that line:

```rust
// crates/op-server/src/main.rs
let state = AppState::with_graph_path("/var/lib/openpay/data.graph")?;
```

`with_graph_path` opens (or creates) the file and points every
store at it. Reopening the same path on restart recovers every
stored fact. See `crates/op-server/src/state.rs` for the full
docstring.

Or read the path from env in your binary:

```rust
let state = match std::env::var("OP_GRAPH_PATH").ok() {
    Some(path) => AppState::with_graph_path(path)?,
    None => AppState::new_in_memory(),
};
```

`OP_GRAPH_PATH` is **not** read by the shipped `main.rs`. The env
name is a convention this guide adopts; the systemd unit and env
sample assume you've patched `main.rs` accordingly (or shipped your
own binary).

---

## 4. Wire auth keys

`op-server` exports `ApiKeyAuthLayer` (in `crates/op-server/src/auth.rs`)
and `RateLimitLayer` (in `crates/op-server/src/rate_limit.rs`).
Neither is mounted by the default `main.rs`. To turn them on, build
your binary using `router_with_middleware`:

```rust
use std::collections::HashSet;
use op_server::{router_with_middleware, AppState, ApiKeyAuthLayer, RateLimitLayer};

let keys: HashSet<String> = std::env::var("OP_API_KEYS")
    .unwrap_or_default()
    .split(',')
    .filter(|s| !s.is_empty())
    .map(str::to_owned)
    .collect();

let auth = ApiKeyAuthLayer::new(keys)
    .with_bypass_paths(vec!["/health".into(), "/readiness".into()]);
let limit = RateLimitLayer::per_minute(600);

let app = router_with_middleware(state, auth, limit);
```

Treat `OP_API_KEYS` as a deployment convention — same shape as
`OP_GRAPH_PATH` above. The library does not read it; your binary
does.

---

## 5. Wire a real HTTP transport for webhooks

The default emitter (`NoOpEmitter`) drops events. To deliver them,
turn on the `reqwest-transport` feature on `op-webhook` and install
the transport on `AppState`:

```toml
# In your binary's Cargo.toml (or op-server's, if you patch it):
op-webhook = { path = "../op-webhook", features = ["reqwest-transport"] }
```

```rust
use std::sync::Arc;
use op_webhook::ReqwestTransport;

let transport = Arc::new(ReqwestTransport::new()?);
let state = AppState::with_graph_path("/var/lib/openpay/data.graph")?
    .with_webhook_transport(transport);
```

`with_webhook_transport` wires a `WebhookDispatcher` over the
state's `webhooks` store with a Stripe-like exponential backoff
(`ExponentialBackoffPolicy::stripe_like`).

---

## 6. Register a webhook endpoint

Once the dispatcher is wired, register an endpoint to receive
events. Either POST directly, or use the CLI:

```bash
op webhooks register \
    --url https://yourapp.example.com/openpay/events \
    --secret "$(openssl rand -hex 32)" \
    --event-types payment.succeeded,refund.settled
```

(Equivalent: `POST /v1/webhooks/endpoints` — see
`crates/op-server/src/handlers/webhook.rs`.)

The secret is used to sign outbound payloads with HMAC-SHA256. Your
receiver verifies; if it can't, drop the event. See `docs/28-webhooks-progress.md`
for the full delivery / retry / dead-letter story.

---

## 7. Register an FX provider

The default `StaticQuoteProvider::new()` has no rates and rejects
every quote request. For a deployment that calls `/v1/fx/quote`,
either populate the static provider at build time or install a live
one. Static is fine for stable corridors:

```rust
use op_core::Currency;
use op_fx::StaticQuoteProvider;

let fx = std::sync::Arc::new(
    StaticQuoteProvider::new()
        .with_rate(Currency::EUR, Currency::USD, 1_082_500)   // 1.0825
        .with_rate(Currency::USD, Currency::EUR,   923_900)   // 0.9239
);
let state = state.with_fx_provider(fx);
```

`rate_ppm` is parts-per-million: `1_082_500` = 1.0825. See
`docs/29-fx-progress.md` for live-feed integration patterns (Wise,
Open Exchange Rates, an internal hedged feed) — implement
`QuoteProvider` and pass it to `with_fx_provider`.

---

## 8. Register a CryptoGateway

For the stablecoin rail, implement (or pull in) a `CryptoGateway`
for the chain you settle on, wrap it in a `CryptoAdapter`, and
register it on your orchestrator:

```rust
use std::sync::Arc;
use op_orchestrator::{CryptoAdapter, Orchestrator};
use op_rails_crypto::CryptoGateway;

let gateway: Arc<dyn CryptoGateway> = Arc::new(YourSolanaGateway::new(rpc_url));
let adapter = CryptoAdapter::new("solana-usdc", gateway);

let mut orchestrator = Orchestrator::new();
orchestrator.register_adapter(Arc::new(adapter));

let state = AppState::with_graph_path("/var/lib/openpay/data.graph")?
    .with_orchestrator(orchestrator);
```

The driver name (`"solana-usdc"`) is what your `PolicyRouter`
references when choosing a driver. See `docs/25-crypto-rail-progress.md`
for the full crypto-rail design.

---

## Final: smoke test the deployment

With auth on, set the key in your shell:

```bash
export OP_API_KEY="<one of your OP_API_KEYS>"

op health
op readiness

# Create + fetch a refund as a round-trip:
op refund create --payment-id pay_demo --amount-minor 1000 --currency USD --external-id "smoke-$(date +%s)"
op refund get <id-from-previous-output>
```

If `health` returns 200 and the refund round-trips, the deployment
is wired correctly: HTTP → auth → state → graph → response.

---

## Where each knob lives

| Concern | Code path |
|---|---|
| Bind address | `OP_BIND_ADDR` in `crates/op-server/src/main.rs` |
| Log level | `RUST_LOG` env (filtered by `tracing-subscriber::EnvFilter`) |
| Persistence | `AppState::with_graph_path` in `crates/op-server/src/state.rs` |
| Auth | `ApiKeyAuthLayer` in `crates/op-server/src/auth.rs` |
| Rate limit | `RateLimitLayer` in `crates/op-server/src/rate_limit.rs` |
| Webhook transport | `AppState::with_webhook_transport` + `op-webhook` `reqwest-transport` feature |
| FX rates | `AppState::with_fx_provider` |
| Crypto rail | `CryptoAdapter::new(name, gateway)` + `Orchestrator::register_adapter` |
| HTTP routes | `crates/op-server/src/routes.rs` |
| Tracing spans | grep `tracing::instrument` across `crates/op-*/src/` |
