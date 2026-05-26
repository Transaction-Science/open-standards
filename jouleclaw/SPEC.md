# JouleClaw v1 — Energy-Optimised AI Runtime

Status: **Draft Standard**. Reference implementation:
[`jouleclaw-rs/`](jouleclaw-rs/) (Rust workspace, Apache-2.0).
Spec text: CC-BY-4.0.

Keywords **MUST**, **MUST NOT**, **SHOULD**, **MAY** are RFC 2119.

---

## 1. Scope and doctrine

JouleClaw is a runtime + harness + agentic-layer standard for
energy-optimised AI. It applies to:

- text and code generation (LLMs, SSMs, state-space hybrids)
- multimodal understanding and generation (vision, audio, video, 3D)
- diffusion-based image / video synthesis (DDPM / LCM / Euler / Heun /
  DPM++ / CFG++ / Rectified Flow and friends)
- speech recognition (Whisper-class)
- the developer-facing harness that drives the above

The **doctrine** is one sentence: capability per joule, not capability
per parameter.

Three normative consequences follow from the doctrine.

### 1.1 The cascade

A conforming runtime **MUST** dispatch every resolution through an
explicit energy-tiered cascade and **MUST** stop at the first tier
that closes the query. The tiers are:

| Tier | Wire tag | Named tag | Joule class       | What it is                                       |
|------|----------|-----------|-------------------|--------------------------------------------------|
| L0   | `L0`     | Cache     | picojoules        | content-addressed cache hit                      |
| L1   | `L1`     | Lawful    | nanojoules        | deterministic primitive (text / code only)       |
| L2   | `L2`     | Embed     | sub-millijoules   | embedding + hybrid retrieval                     |
| L3   | `L3`     | Model     | joules            | local SSM / ternary / multimodal / diffusion     |
| L4   | `L4`     | Wire      | tens of joules    | remote frontier RPC (escape hatch)               |

L1 (Lawful) is meaningful only for text and code modalities. For
image / audio / video / 3D generation the cascade collapses to
{L0, L2, L3, L4} — there is no deterministic compute path that
produces a high-quality image from "draw me a cat".

A conforming runtime **MUST NOT** invoke a higher-cost tier before
the lower-cost tiers have returned `Unresolvable`. Inference is the
**last resort**, not the entry point.

### 1.2 Recent knowledge over frozen weights

A conforming runtime **MUST** prefer fresh, provenance-stamped
retrieval over synthesis from frozen model weights whenever:

- the cost of the fetch + extract is below the cost of the
  inference, AND
- the retrieved source's trust tier is above the operator's
  configured threshold.

This is the L3.5 stage in the cascade implementation — between L2
(local index) and L3 (local model). The retriever **MUST** emit a
`ClaimProvenance` envelope (see §4) for every fact that contributes
to the resolution.

### 1.3 Honest energy provenance

Every energy reading **MUST** carry a `Provenance` tag declaring
how the value was obtained:

| Provenance     | Meaning                                                |
|----------------|--------------------------------------------------------|
| `HwShunt`      | Real hardware shunt / coulomb counter (RAPL MSR, NVML cumulative-energy counter, Jetson INA3221) |
| `ModelBased`   | Vendor-provided estimate from freq / voltage / utilisation (Apple IOReport, ROCm SMI, NVML `power.draw`) |
| `Estimator`    | JouleClaw static cost model from arch × precision × batch tables |

The thermodynamic circuit breaker **MUST** enforce at the granularity
of the **worst** counter in the request's span. Implementations
**MUST** surface `resolution_uj()` and `min_window_ns()` so callers
can refuse to claim microjoule accuracy on platforms that cannot
deliver it.

Realistic floors:

- Intel / AMD x86 RAPL: ~1 μJ, ~10 ms window. `HwShunt`.
- Jetson INA3221: ~10 mW (integrate for energy). `HwShunt`.
- NVIDIA discrete (NVML cumulative-energy): ~1 mJ, ~50 ms. `HwShunt`.
- Apple Silicon IOReport: ~1 mJ, ~10 ms. **Model-based**, not measured.
  Marketing claims of microjoule precision on Apple Silicon are
  sales talk; the spec is honest about this.
- Consumer AMD GPU, ARM PMU: no usable counter. Use a calibrated
  `Estimator` and surface the wider tolerance band.

---

## 2. The receipt

Every cascade walk **MUST** produce one [`Receipt`][prov]:

```json
{
  "jc_receipt": "1",
  "id": "<uuid v4>",
  "closed_at": "<rfc 3339>",
  "input_hash": "<blake3 hex of normalised input>",
  "tier": "L3",
  "joules_uj": 3500000,
  "energy_provenance": "ModelBased",
  "tools_touched": [
    {
      "tool_id": "model:gemma4-9b-q5_k_m",
      "joules_uj": 3500000,
      "energy_provenance": "ModelBased"
    }
  ],
  "claims": [
    {
      "source": "https://en.wikipedia.org/wiki/Gemma_(model)",
      "content_hash": "<blake3 hex>",
      "fetched_at": "<rfc 3339>",
      "trust_tier": 9
    }
  ],
  "eoc_stage": null
}
```

Receipts are shaped to seal cleanly inside a Smart Byte signed
envelope (see the sibling [Smart Byte standard](../smart-byte/)).
The Smart Byte signature attests to receipt integrity; the receipt
itself is the auditable thermodynamic record.

A receipt's `energy_provenance` is the **worst** counter seen across
all `tools_touched` — not the best. This is normative.

[prov]: jouleclaw-rs/crates/jouleclaw-prov/src/lib.rs

---

## 3. The `.jc.toml` sidecar

A model file on disk (GGUF / safetensors / MLX) tells you tensor
shapes and quant schemes. It does NOT tell you what one forward
pass costs in joules on real hardware. The cascade auction cannot
pick the cheapest backend if every backend lies about its cost.

A `.jc.toml` sidecar travels next to the model file and declares
the energy contract. See [`jouleclaw-pack`][pack] for the schema.
Backends loading a model with a `.jc.toml` are bound to honour the
declared cost within the per-measurement `drift_factor` tolerance;
the runtime trips `CostDrift` and down-weights the backend in
future auctions when measured > declared × drift_factor.

Without a declared-cost contract, "energy-optimised inference" is
marketing. With one, it is an engineering claim a third party can
verify against the published reference-hardware corpus (see §6).

[pack]: jouleclaw-rs/crates/jouleclaw-pack/src/lib.rs

---

## 4. Provenance

Every retrieved claim that contributes to a resolution **MUST**
carry a `ClaimProvenance` envelope:

```json
{
  "source":       "<url, did:plc, doi, ...>",
  "content_hash": "<blake3 hex of bytes as fetched>",
  "fetched_at":   "<rfc 3339>",
  "trust_tier":   9
}
```

`trust_tier` is 0..10. Bootstrap data comes from the Wikipedia
perennial-sources list (machine-readable via the Wikimedia
Enterprise parsed-references endpoint); operators **MAY** extend the
table.

A retriever **MUST NOT** present a synthesised fact as a retrieved
claim. If the runtime falls back to model inference because no
retrieval closed the query, the receipt's `claims` array **MUST**
remain empty for that tier and the resolution carries the
`L3:Model` tag, not `L2:Embed`.

---

## 5. The MCP tool surface

The standard tool dispatch protocol is MCP (Model Context Protocol).
A conforming runtime **SHOULD** speak MCP for interop with the
Claude Code / Codex / Goose ecosystem.

Every MCP tool call **MUST** be wrapped in
`jouleclaw-mcp::dispatch_metered`, which brackets the call with
energy counter reads and pushes a `ToolTouch` entry onto the
running receipt's `tools_touched` ledger.

### 5.1 The `joule-mcp` profile

Two JouleClaw-aware endpoints **MAY** opt into a CBOR transport
profile by advertising the capability tag `x-jouleclaw/joule-mcp@1`
in their handshake. When both sides advertise it, the wire encoding
switches from JSON-RPC to length-prefixed CBOR (~30–50% lower
serialisation tax per call). When either side does not advertise,
the encoding **MUST** fall back to standard MCP JSON-RPC.

Non-JouleClaw MCP clients (Claude Code, Codex, etc.) see plain
JSON-RPC. The CBOR profile is a negotiated extension, never a
replacement.

---

## 6. Conformance

A conforming runtime **MUST** produce a `Receipt` for every published
conformance vector in [`conformance/v1/`](conformance/v1/) that matches
the canonical receipt's `tier`, `tools_touched[].tool_id`, and
`claims[].content_hash`.

The `joules_uj` field in a conforming receipt **MUST** fall within
the platform-specific drift band declared in the `.jc.toml` sidecar
for each tool / model touched. Drift beyond the band is a
non-conformance signal, not a hard failure — the runtime **SHOULD**
emit `DriftAlert` and let the operator decide whether to demote the
backend or pause the deployment.

Conformance vectors are signed at release time. Implementations
self-certify by round-tripping the public vectors and publishing
their receipts.

---

## 7. Wire format versioning

- The cascade tier wire tags (`L0`–`L4`) are stable across versions.
  A future major version **MAY** add tiers (e.g. `L5:Quantum`) but
  **MUST NOT** renumber existing ones.
- The `jc_receipt` field on a receipt is the schema version. This
  document defines `"1"`. Unknown major → reject. Unknown minor →
  accept by ignoring unknown fields.
- The `jc_pack` field on a `.jc.toml` sidecar follows the same rule.

---

## 8. What this spec deliberately does not say

- **How to build the model.** That's the model-author's concern.
  JouleClaw measures and dispatches; it doesn't train.
- **Which retrieval API to use.** Brave / Tavily / Exa / Serper all
  plug in through the `SearchProvider` trait. Pick whichever your
  deployment permits.
- **How to ship receipts.** The receipt is produced. Transport
  (HTTP / lockstep / message queue) is the deployer's choice. Smart
  Byte's signed-envelope replication is the recommended substrate.
- **A specific signing key topology.** Bring your own. Smart Byte's
  KERI-based AID rotation is the recommended path.

---

## 9. References

- [JouleClaw Charter](../CHARTER.md) — stewardship pattern (shared
  across all five open standards)
- [Smart Byte open standard](../smart-byte/) — signed envelopes
- [EOC open standard](../eoc/) — the four-stage memoising cascade
  pattern JouleClaw embodies
- [WAI open standard](../wai/) — media transport + capability
  dispatch
- [OpenPay open standard](../openpay/) — the typestate-enforced
  lifecycle pattern adopted by JouleClaw's cascade
- W3C PROV-O — provenance vocabulary
- RFC 9449 — DPoP (sender-constrained tokens, recommended for
  ClaimProvenance envelopes that carry a fetch credential)
- Landauer, R. (1961). "Irreversibility and Heat Generation in the
  Computing Process." The thermodynamic floor under everything in
  this spec.
