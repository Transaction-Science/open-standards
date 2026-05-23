# OpenPay — five-minute quickstart

The shortest path from `git clone` to a request hitting an
op-server process. No persistence, no TLS, no auth — that's for
the rest of `docs/deploy/`. This is just "does the wheel turn?"

## Prerequisites

- Rust **1.95** (edition 2024). The MSRV is pinned in
  `rust-toolchain.toml` / `Cargo.toml`.
- A working `cargo`. No system deps beyond a C linker.

## The path

```bash
git clone https://github.com/openpay/openpay.git
cd openpay
cargo build --release -p op-server -p op-cli
```

`op-server` is the HTTP daemon; `op-cli` builds an `op` binary you
can use to talk to it. (The CLI is being built in parallel — if
your checkout doesn't have it yet, `cargo build --release -p op-server`
alone is enough; you can hit the API with `curl`.)

In one terminal:

```bash
./target/release/op-server &
# Output: op-server starting, addr=127.0.0.1:8080
```

In another:

```bash
./target/release/op health
# {"status":"ok"}

./target/release/op readiness
# {"status":"ready","stores":{...}}
```

Or with `curl` if the CLI isn't built yet:

```bash
curl http://127.0.0.1:8080/health
curl http://127.0.0.1:8080/readiness
```

## Round-trip a refund

```bash
./target/release/op refund create \
    --payment-id pay_demo \
    --amount-minor 1000 \
    --currency USD \
    --external-id "smoke-$(date +%s)"
# {"id":"rfnd_…","status":"Requested","amount":{...},…}

./target/release/op refund get rfnd_…
# Same record echoed back.
```

The refund lives in the in-memory graph; restarting `op-server`
drops it. To persist across restarts, see
[`README.md` §3 — persisting to a `.graph` file](README.md#3-persist-to-a-real-graph-file).

## Shut it down

```bash
kill %1            # if you backgrounded it with `&`
# Or Ctrl-C in the op-server terminal.
```

The binary installs both SIGINT (Ctrl-C) and SIGTERM handlers (see
`crates/op-server/src/main.rs::shutdown_signal`) and drains
in-flight requests on either signal.

## Next steps

| Want | Read |
|---|---|
| TLS in front of op-server | `README.md` §2 + `Caddyfile.sample` |
| Persistence | `README.md` §3 |
| Auth + rate limit | `README.md` §4 |
| Outbound webhooks | `README.md` §5-6 |
| FX quotes | `README.md` §7 |
| Crypto rail | `README.md` §8 |
| Run under systemd | `openpay.service` |
| Backup the graph file | `backup.md` |
| Wire logs to a backend | `monitoring.md` |
