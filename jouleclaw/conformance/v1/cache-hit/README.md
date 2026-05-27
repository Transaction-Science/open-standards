# Conformance vector — cache-hit (L0)

Exercises the L0:Cache tier. A conforming runtime MUST resolve a repeat
query at L0 and emit a receipt whose `tier == "L0"` and whose
`tools_touched` is empty.

## What the runtime does

1. Receives `input.txt` for the first time. Resolves it through the
   normal cascade (any tier may answer — doesn't matter for this
   vector). Caches the result keyed by `blake3(normalised_input)`.
2. Receives the **same** input a second time. The L0 cache lookup
   hits and the runtime returns the cached answer without invoking
   any downstream tier.
3. Emits the receipt in `receipt.json` for the second invocation.

## Acceptance bounds

The conforming implementation's receipt for the second invocation
MUST match the canonical receipt on:

- `tier` — exactly `"L0"`
- `input_hash` — exactly the canonical value
- `tools_touched` — exactly empty (`[]`)
- `claims` — exactly empty (`[]`)
- `joules_uj` — within the platform's `Provenance` band:
  - `HwShunt`: declared ± 20%
  - `ModelBased`: declared ± 50%
  - `Estimator`: declared ± 200%

`id` and `closed_at` are per-run and not compared.

## Files

- `input.txt` — the repeated query
- `receipt.json` — the canonical receipt the second invocation must
  reproduce (reference hardware: Apple M5 Max, jouleclaw-rs v0.1.0)

This is the simplest conformance vector. Any runtime that can't pass
this one isn't a JouleClaw runtime — it's just an inference server with
the word "JouleClaw" pasted on top.
