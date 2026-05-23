# Phase 29 — Multi-currency / FX

**Status**: Draft v0.29
**Date**: 2026-05-22

## Why

OpenPay was single-currency end-to-end: each `Money` carried a
`Currency`, settlement batches enforced same-currency, no
primitive existed for "accept in EUR, pay out in USD." That's a
hard wall for any vendor with global customers — and "global
customers" is most operators today.

Phase 29 adds the FX primitive: a pluggable [`QuoteProvider`]
trait, integer-exact conversion with deterministic rounding, two
reference providers (static + cached), and HTTP endpoints to
quote/convert. Operators wire their FX feed (Wise, Open Exchange
Rates, bank tariff, internal mid-market) behind the trait — the
reference stack stays free of HTTP clients, same architectural
position as `op-webhook` (Phase 13) and `op-rails-crypto` (Phase 25).

Second of three sequenced phases (28 → 29 → 30). Next: 3DS / SCA
resume primitive.

## What shipped

| # | Item | Where |
|--:|---|---|
| 1 | `op-fx` crate — Quote, QuoteProvider, convert, rounding, providers | `crates/op-fx/` |
| 2 | `Quote` type — `(source, target, rate_ppm, fetched_at, valid_until, source_name)` with `inverse()` helper | `quote.rs` |
| 3 | `RoundingMode` — `HalfEven` (default, banker's), `Down`, `Up` | `convert.rs` |
| 4 | `convert(money, quote, mode, now)` — integer-exact via `i128` intermediate; rejects same-currency, mismatched currency, zero rate, expired quotes | `convert.rs` |
| 5 | `QuoteProvider` trait — `get_quote(source, target, now) -> Result<Quote>` + `name()` | `provider.rs` |
| 6 | `StaticQuoteProvider` — operator-fixed `(source, target) → rate_ppm` table with configurable validity window | `provider.rs` |
| 7 | `CachedQuoteProvider<P>` — TTL wrapper that respects both cache-age and quote-validity invariants | `provider.rs` |
| 8 | `AppState.fx: Arc<dyn QuoteProvider>` + `with_fx_provider(...)` builder | `op-server/src/state.rs` |
| 9 | HTTP `GET /v1/fx/quote?from=X&to=Y` and `POST /v1/fx/convert` | `op-server/src/handlers/fx.rs` |
| 10 | `From<op_fx::Error> for ApiError` mapping — `NoQuote → 404`, `QuoteExpired → 409`, `CurrencyMismatch/InvalidRate/SameCurrency → 400`, `Overflow → 500` | `op-server/src/error.rs` |

Workspace at the end of Phase 29:

| Check | Result |
|---|---|
| `cargo build --workspace --all-targets` | **0 errors, 0 warnings** |
| `cargo test --workspace` | **1047 passing, 0 failing** (+20 vs Phase 28) |
| `cargo clippy --workspace --all-targets` | **0 warnings** |

Test-count delta: op-fx 17 (convert 10, provider 4, quote 3) +
op-server 3 (FX endpoints).

## Why integer math (parts-per-million)

FX is the canonical "don't use floats for money" example. A
0.0001 rounding error on `$10M` is $1000 someone cares about.

Rates: **`u64` parts per million** (`1.000000 = 1_000_000`). Six
decimal places is the FX industry standard for spot rates;
carrying more is meaningless given bid/ask spreads.

Conversion:

```text
target_minor = round( source_minor × rate_ppm / 1_000_000 )
```

via `i128` intermediate so an `i64::MAX` minor amount times a
large rate doesn't overflow before the explicit `i64::try_from`
guard at the end. No floats anywhere in the conversion pipeline.

## Rounding semantics

```rust
RoundingMode::HalfEven   // banker's; default; statistically unbiased
RoundingMode::Down       // truncate toward zero; biased toward whoever benefits
RoundingMode::Up         // away from zero; biased toward the other side
```

Half-even (banker's rounding) is the GAAP/IFRS default for
financial systems — eliminates the systematic up-bias of "round
half up." The crate uses it as the `Default` impl.

## Quote validity

Two distinct timing constraints:

1. **Quote's `valid_until_unix_secs`** — when the rate stops being
   usable on the wire. Set by the provider (`StaticQuoteProvider`
   defaults to 5 minutes; live providers come from the feed).
2. **`CachedQuoteProvider` TTL** — how long the cache reuses a
   previously-fetched quote before re-asking the inner provider.

The cache respects both: a cached quote that's still inside its
TTL but past its `valid_until` is refetched, not served. Phase 29
ships an explicit test for this (`cached_provider_bypasses_cache_after_quote_expires`).

`convert(money, &quote, mode, now)` checks `valid_until` and
returns `Error::QuoteExpired` if `now` is past it — operators
can't accidentally settle against a stale rate.

## HTTP surface

```
GET  /v1/fx/quote?from=EUR&to=USD
POST /v1/fx/convert        # body: {from, to, amount_minor, rounding?}
```

Responses are JSON, same error envelope (`{code, message,
details}`) as the rest of op-server.

The reference binary defaults `AppState.fx` to an empty
`StaticQuoteProvider` — all `/v1/fx/quote` requests return `404
not_found` until the operator wires real rates. That's the
explicit "you haven't configured this yet" failure mode.

## Operator wiring

```rust
// Test/demo: fixed rate table.
let provider: Arc<dyn QuoteProvider> = Arc::new(
    StaticQuoteProvider::new()
        .with_rate(Currency::EUR, Currency::USD, 1_082_500)
        .with_rate(Currency::USD, Currency::EUR, 923_787)
);

// Production: operator's live feed, cached for 30 seconds.
let live = Arc::new(MyWiseProvider::new(api_key));
let cached: Arc<dyn QuoteProvider> = Arc::new(CachedQuoteProvider::new(live, 30));

let state = AppState::with_graph_path("/var/lib/openpay/data.graph")?
    .with_fx_provider(cached);
```

The trait surface is one method (`get_quote`) plus a name string —
implementing it against any feed is small.

## What the crate deliberately does NOT do

- **No HTTP client.** `op-fx` has zero networking dependencies.
  Operator-supplied provider impls do the I/O.
- **No spread modeling.** A bank's tariff includes a margin
  above mid-market. Operators bake the margin into the
  `rate_ppm` they hand to the provider; `op-fx` doesn't have a
  separate `spread_ppm` field. Keeps the type small and avoids
  opinions about pricing.
- **No triangulation.** USD/JPY is a direct pair; the crate
  doesn't auto-route via USD/EUR + EUR/JPY when a direct rate is
  missing. Operators who need cross-rates assemble them
  themselves and feed them in as a single quote.
- **No hedging.** Quote validity is an advisory window, not a
  forward lock. Forward contracts are an operator-side concern.

## Settlement integration (deferred)

The settlement batch is still single-currency. Wiring "open a USD
acquiring batch, convert to EUR at close, pay out in EUR" is the
natural next step but adds a `Holdback.payout_currency_amount:
Option<(Money, Quote)>` field and changes the NACHA / pacs.008
generators. Phase 29 ships the conversion primitive; the
settlement-layer integration is intentionally a separate wedge
(operators can compose it externally today by quoting before they
close).

## Honest concerns (carry-forward)

- **No real HTTP-backed provider in the workspace.** Operators
  implement their own — same intentional gap as `op-webhook`'s
  HTTP transport.
- **Cache is unbounded.** `CachedQuoteProvider`'s `HashMap` grows
  with every unique pair seen; no eviction beyond TTL-based skip
  on read. For ≤100 pairs this is fine; operators with a wild
  long-tail of currencies want LRU eviction (drop-in via a
  `HashMap` → `lru::LruCache` swap).
- **No quote-source attestation.** A quote's `source_name` is a
  free-form string; nothing signs it. Operators with stricter
  audit requirements should hash the upstream feed's response and
  carry the digest in `source_name` or extend `Quote` with a
  signature field.
- **No settlement-layer carve-out.** A batch is still one
  currency. Multi-currency settlement is a follow-up that builds
  on top of this primitive.
- **Negative amounts work.** Conversion handles negative
  `minor_units` correctly (refund/dispute reversal flows), but
  operators reading the conversion result for outbound
  transfer-amount construction need their own sign-check on the
  rail side — many rails reject negative-amount messages
  outright.

## Test totals

```
op-fx           17  (convert 10, provider 4, quote 3)
op-server       +3  (quote, convert, 404 on unknown pair)
                                                              ----
                                                              +20 net
```

`cargo test --workspace`: **1047 passing, 0 failing.**
`cargo build --workspace --all-targets`: **0 warnings.**
`cargo clippy --workspace --all-targets`: **0 warnings.**
