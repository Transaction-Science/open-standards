# JouleClaw v1 conformance vectors

A conforming implementation **MUST** round-trip every vector in this
directory and produce a `jouleclaw-prov::Receipt` whose `tier`,
`tools_touched[].tool_id`, and `claims[].content_hash` match the
canonical receipt.

The `joules_uj` field is platform-dependent and **MUST** fall within
the drift band declared in the relevant `.jc.toml` sidecar. The
canonical receipts in this directory are the reference-hardware
baseline (Apple M5 Max running the published `jouleclaw-rs`
reference impl) — implementations on other hardware should match the
shape, not the absolute joule count.

## Coverage (v1.0.0)

| Vector                                | Cascade tier closed | Why it exercises the spec                           |
|---------------------------------------|---------------------|-----------------------------------------------------|
| `cache-hit/`                          | L0                  | Repeat-query content-addressed cache                |
| `lawful-arithmetic/`                  | L1                  | Deterministic primitive — `gcd`, unit conversion    |
| `embed-hybrid-search/`                | L2                  | BM25 + ANN fusion against a fixed local corpus      |
| `fresh-retrieval-with-provenance/`    | L3.5                | Live fetch, Smart Byte envelope, trust tier         |
| `model-local-ssm/`                    | L3                  | Liquid CfC / Mamba forward pass                     |
| `model-local-ternary/`                | L3                  | Prism / BitNet 1.58 ternary forward pass            |
| `model-local-diffusion-image/`        | L3                  | SDXL / SD3 / Flux-dev image gen                     |
| `wire-frontier-rpc/`                  | L4                  | Remote frontier as the explicit escape hatch        |
| `breaker-trips-on-budget-exhaust/`    | (any)               | Thermodynamic kill-switch under runaway loop        |
| `joule-mcp-cbor-negotiation/`         | (any)               | MCP capability handshake elects binary transport    |

Each vector directory contains:

- `input.txt` — the input fed to the runtime
- `pack.jc.toml` — the declared-cost contract for any model touched
- `receipt.json` — the canonical receipt the runtime must reproduce
- `README.md` — what the vector exercises and the acceptance bounds

Vectors are signed at release time. The signature lives in
`SHA256SUMS.sig` next to `SHA256SUMS` — see the JouleClaw release
documentation for verification instructions.

## Adding a vector

1. Land the runtime change in `jouleclaw-rs/` that the vector
   exercises.
2. Generate the canonical receipt by running the reference impl
   against the input.
3. Hand-curate the `pack.jc.toml` drift band so the platform
   variation is sensible (typically 1.5× for `ModelBased` counters,
   1.2× for `HwShunt`, 3× for `Estimator`).
4. Add a `README.md` explaining the vector's intent and the
   acceptance bounds.
5. Refresh `SHA256SUMS` and re-sign.

v1.0.0 stub — vector population follows reference-impl maturity.
