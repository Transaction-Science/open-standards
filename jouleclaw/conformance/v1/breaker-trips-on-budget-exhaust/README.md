# Conformance vector — breaker-trips-on-budget-exhaust

Exercises the thermodynamic circuit breaker. A runtime that escalates
through tiers MUST trip the breaker the instant
`measured J_consumed > J_allocated` and MUST NOT continue dispatch.

## What the runtime does

1. Receives `input.txt` — a query designed not to hit L0/L1/L2 and to
   require an L3 model.
2. The caller sets `budget_uj = 100` (intentionally too low for L3 model
   inference). L3's estimate exceeds the remaining budget.
3. Per the spec, the runtime MUST NOT dispatch L3. It MUST trip the
   breaker and emit a receipt with `tier == "L4"`-style escape *only*
   if L4 is configured AND fits the budget; otherwise the receipt
   records the final state as Refused (output.refused.reason
   "BudgetExhausted").

## Acceptance bounds

The receipt MUST satisfy one of:

- `output.refused.reason == "BudgetExhausted"` AND no L3 tier appears
  in `tools_touched`, OR
- `tier == "L0"` or `"L1"` or `"L2"` AND `joules_uj <= 100`

A runtime that lets an L3 model invocation run past the budget — even
if it eventually returns an answer — is **non-conformant**. The breaker
is the load-bearing safety primitive of this standard.

## Files

- `input.txt` — the query
- `budget.json` — `{ "budget_uj": 100 }`
- `receipt.json` — canonical refusal receipt the implementation must
  reproduce (modulo `id` and `closed_at`)

## Why this vector matters

Without an enforceable circuit breaker, the "energy-first cascade" is a
suggestion, not a guarantee. This vector is the proof that JouleClaw's
honesty contract is observable from the outside.
