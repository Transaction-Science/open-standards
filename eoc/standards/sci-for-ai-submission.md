# Mapping EOC to the SCI-for-AI Accounting Framework

*Draft submission for filing to [github.com/Green-Software-Foundation/sci](https://github.com/Green-Software-Foundation/sci). Licensed CC-BY-4.0.*

*Status: working draft v0.1. Authors: Transaction Science (steward of the EOC specification). Target SCI revision: SCI-for-AI v1.0 (ratified December 2025).*

---

## a. Summary

**EOC (Energy-Optimized Compute)** is a vendor-neutral specification for resolving AI inference at the lowest-cost stage that suffices for the task, with every resolution returning a content-addressed receipt carrying a falsifiable joule cost. **SCI-for-AI** is the Green Software Foundation's accounting framework, ratified December 2025 and aligned to ISO/IEC 21031, that expresses the carbon impact of AI workloads as `SCI = (E × I + M) per R`. **This document maps EOC's receipt schema onto SCI-for-AI's four-component formula** so that a deployment using EOC can produce SCI-conformant reports by direct field extraction — and, in the other direction, an SCI-for-AI reporter can ingest EOC receipts as a structured per-query data source. Together they let a deployment report not just *how much energy was used* but *which stage answered the query*, which is the accounting signal most useful for engineering action.

## b. Background — SCI-for-AI

The Green Software Foundation's **Software Carbon Intensity (SCI)** specification defines a rate-based metric for the carbon impact of a software application. SCI-for-AI, ratified by the GSF Standards Working Group in December 2025, extends the base SCI specification to handle the structural peculiarities of AI inference and training workloads — model warm-up amortization, batch vs. per-query attribution, embodied carbon of accelerators, and the share of compute that is wasted on speculative decoding or aborted generations.

The headline formula is:

```
SCI = (E × I + M) per R
```

where:

- **E** — energy consumed by the software, in **kWh**. For AI workloads, this includes inference compute, the host CPU's share of that compute, network transfer of model weights, and (when amortized over the relevant accounting window) training energy.
- **I** — **location-based** or **market-based** carbon intensity of the electricity that supplied **E**, in **gCO₂eq / kWh**. SCI-for-AI requires location-based reporting and permits market-based reporting as a supplementary figure.
- **M** — embodied carbon of the hardware allocated to the work, amortized over the hardware's expected useful life and the fraction of that life spent on **R**, in **gCO₂eq**. For accelerators (GPUs, TPUs, custom silicon) M is typically the dominant contribution at low utilization.
- **R** — the **functional unit** of the application. SCI-for-AI explicitly permits R = *one inference*, R = *one correct answer*, R = *one user-session*, R = *one model-card-equivalent benchmark score*, or any other deployment-defined unit — provided the unit is stated and reproducible.

SCI-for-AI is aligned with ISO/IEC 21031:2024, which standardizes the SCI methodology under the umbrella of ISO's software-sustainability work. Ratification was announced by the GSF Standards WG on 12 December 2025; the canonical specification text is hosted at `https://github.com/Green-Software-Foundation/sci`.

SCI-for-AI is deliberately **silent about routing**. It tells you how to *report* an inference workload's carbon impact. It does not tell you how to *reduce* it. A reporter is free to compute SCI for an inference pipeline that always uses a 70-billion-parameter generative model, or for one that resolves 95% of queries from a cache and only invokes the generative model on the remaining 5%. Both pipelines produce a valid SCI number — and the second pipeline produces a much smaller one. EOC supplies the routing rule that produces that second pipeline.

## c. Background — EOC

**EOC-1** specifies a **four-stage cascade** through which an AI query is routed:

1. **Cache** — content-addressed exact-match retrieval. A query whose canonical BLAKE3 hash matches a stored answer hash is resolved here. Median cost on commodity hardware: ~10 μJ.
2. **KV** — approximate-match retrieval via dense vector similarity (cosine over sentence-embedding space). Resolves when the nearest stored query is within a deployment-configured similarity threshold. Median cost: ~1 mJ.
3. **Graph** — structured retrieval against a typed knowledge graph using **DCY** (Deterministic Cypher, EOC-4). Resolves when the query decomposes into a graph pattern whose answer is materialized in the graph. Median cost: ~10 mJ.
4. **Neural** — generative model fallback. Always available; always last. Median cost varies by model size; a 4-billion-parameter quantized model on a consumer GPU produces ~1 J per short answer.

Each stage carries a **resolution predicate** that determines whether the stage answers or passes the query to the next stage. The first stage whose predicate returns true wins. The cascade is monotonically increasing in median energy cost — so the cheapest stage that suffices always answers.

Every resolution emits an **EOC receipt** — a signed, content-addressed record carrying:

- `query.hash` — BLAKE3 hash of the canonicalized query.
- `query.id` — deployment-assigned identifier (UUID v7 or similar).
- `resolved_at_stage` — integer in {1, 2, 3, 4} naming the cascade stage that answered.
- `joule_cost.microjoules.measured` — integer microjoules drawn from hardware energy counters (Intel RAPL, NVIDIA NVML, ARM PMU, or analogous).
- `joule_cost.microjoules.estimated` — integer microjoules from a calibrated estimator, used when hardware counters are unavailable.
- `joule_cost.method` — one of `rapl`, `nvml`, `pmu`, `estimator-v1`, `estimator-v2`.
- `answer.hash` — BLAKE3 hash of the canonicalized answer.
- `wall_clock_ns` — wall-clock duration of the resolution in nanoseconds.
- `signature` — ed25519 over the canonical CBOR serialization of the receipt body.

The receipt is the unit of accounting. It is content-addressed, independently verifiable, and carries the joule cost in its smallest reasonably-falsifiable unit (microjoules), so downstream reporters can aggregate without precision loss. The full receipt schema is published at `eoc.transaction.science/sci-mapping/v1.schema.json` (target URL — to be served on registration of this extension).

## d. Mapping

The mapping from SCI-for-AI components to EOC receipt fields is direct. Every SCI-for-AI component is extractable from one or more EOC receipts by a fixed unit conversion, with no semantic gap.

| SCI-for-AI component | EOC receipt field | Unit conversion | Notes |
|---|---|---|---|
| **E** (kWh) | `joule_cost.microjoules.measured` + `joule_cost.microjoules.estimated` | `E_kWh = (measured + estimated) / 3.6e12` | Sum of both fields; the estimator field is zero when hardware counters supplied the full reading. 3.6 × 10¹² μJ = 1 kWh. |
| **I** (gCO₂eq/kWh) | *not in receipt — supplied by deployment* | identity | I is a property of the grid that powered the host, not of the query. Reporter supplies I per ISO 21031 location-based methodology. |
| **M** (gCO₂eq) | *not in receipt — supplied by deployment* | identity | M is amortized hardware embodied carbon. Reporter supplies the per-second amortization rate; multiply by `wall_clock_ns / 1e9`. |
| **R** (functional unit) | `query.id` | identity | One receipt = one resolved query = one R unit, under the default SCI-for-AI choice R = *one inference*. Deployments choosing R = *one correct answer* must additionally bind a correctness label to the receipt (see §e). |
| *(extension)* **resolved_at_stage** | `resolved_at_stage` | identity | EOC-extension field. Not part of base SCI-for-AI but exposed to enable per-stage carbon accounting. |
| *(extension)* **measurement_method** | `joule_cost.method` | identity | EOC-extension field. Lets a downstream reporter weight measured vs. estimated readings appropriately. |

The two extension fields — `resolved_at_stage` and `measurement_method` — are the substantive contribution of this mapping. SCI-for-AI alone aggregates all of an application's inference compute into a single E. EOC's receipts let a reporter decompose E by cascade stage, producing per-stage SCI numbers:

```
SCI_cache    = (E_cache    × I + M_cache)    per R_cache
SCI_kv       = (E_kv       × I + M_kv)       per R_kv
SCI_graph    = (E_graph    × I + M_graph)    per R_graph
SCI_neural   = (E_neural   × I + M_neural)   per R_neural
SCI_total    = (E_total    × I + M_total)    per R_total
```

A deployment can then answer the engineering question "which stage is responsible for our SCI number?" — typically the neural stage, by an order of magnitude, even when it answers fewer than 10% of queries.

## e. Worked example

Consider a query *"what is the boiling point of water at sea level?"* routed through an EOC cascade deployed on a host with Intel RAPL counters available.

**Receipt** (CBOR-decoded to JSON for readability):

```json
{
  "query": {
    "hash": "b3:b1946ac92492d2347c6235b4d2611184",
    "id": "01914a2b-3c4d-7e8f-9a0b-1c2d3e4f5a6b",
    "canonical": "what is the boiling point of water at sea level?"
  },
  "resolved_at_stage": 1,
  "joule_cost": {
    "microjoules": {
      "measured": 14,
      "estimated": 0
    },
    "method": "rapl"
  },
  "answer": {
    "hash": "b3:7c5e9b2a8f1d4e6c3a0b9d8e7f6a5c4b",
    "canonical": "100 °C"
  },
  "wall_clock_ns": 38421,
  "signature": "ed25519:3a8c...redacted"
}
```

The query hit the cache stage. Total energy consumed: **14 microjoules**, measured directly from RAPL counters.

**SCI calculation**, with deployment-supplied I and M:

- **E** = (14 + 0) μJ = 14 μJ = 14 × 10⁻⁶ J ÷ 3.6 × 10⁶ J/kWh = **3.89 × 10⁻¹² kWh**.
- **I** = 380 gCO₂eq/kWh (US grid average, location-based, deployment-supplied).
- **M** = (host accelerator amortization rate) × (wall_clock_ns / 10⁹) = 1.2 × 10⁻⁹ gCO₂eq/s × 38.421 × 10⁻⁶ s = **4.6 × 10⁻¹⁴ gCO₂eq**.
- **R** = 1 inference.

```
SCI = (3.89e-12 kWh × 380 gCO₂eq/kWh + 4.6e-14 gCO₂eq) per 1 inference
    = (1.48e-9 + 4.6e-14) gCO₂eq per inference
    ≈ 1.48 × 10⁻⁹ gCO₂eq per inference
```

By comparison, the same query routed to a 70-billion-parameter generative model on the same hardware (resolved_at_stage = 4) would have measured roughly 8 J — six orders of magnitude greater energy — producing an SCI of approximately **8.4 × 10⁻⁴ gCO₂eq per inference**.

The factor-of-500,000 difference is what the cascade saves. SCI-for-AI gives the *reporting unit* that makes the saving visible; EOC gives the *routing rule* that captures it.

## f. Why this mapping matters

SCI-for-AI is a strong accounting standard. It is not, by design, a routing standard. A deployment can use it to report carbon impact without committing to any particular implementation choice — which is exactly the property a reporting standard should have.

EOC is a strong routing standard. It is not, by design, an accounting standard. The receipt schema captures the joule cost of each resolution, but the framing — "joules per task against an open evaluation corpus" — is engineering-internal. It does not, on its own, plug into the carbon-accounting ecosystem that SCI-for-AI anchors.

The two together produce a strictly richer reporting unit than either alone:

- **SCI-for-AI alone**: "this deployment emitted X gCO₂eq per inference."
- **EOC alone**: "this query cost N joules and was answered at stage K."
- **SCI-for-AI + EOC**: "this deployment emitted X gCO₂eq per inference, of which 99.4% came from 4.1% of queries that fell through to the neural stage; the remaining 95.9% of queries cost a combined Y gCO₂eq, three orders of magnitude less."

The combined unit is actionable in a way that neither alone is. It tells the engineer which stage to invest in (almost always: improve the cache and KV stages so fewer queries fall through). It tells the reporter what counterfactual to publish ("if we had no cascade, our SCI would have been 200× higher"). It tells the regulator what to ask for (per-stage SCI, not just aggregate).

This is the reporting unit that ISO/IEC 21031's *useful work* clause anticipates but does not specify. The mapping operationalizes it.

## g. Proposal

We propose to register EOC's receipt schema as a **named extension** to the SCI-for-AI specification, with:

- **URN**: `urn:gsf:sci-ai:ext:eoc:v1`
- **Canonical schema URL**: `https://eoc.transaction.science/sci-mapping/v1.schema.json`
- **Mapping document**: this file, lodged at `https://eoc.transaction.science/standards/sci-for-ai-submission.md` and mirrored at `github.com/Green-Software-Foundation/sci/extensions/eoc/`.
- **Conformance**: a deployment claiming "SCI-for-AI conformant via EOC extension" must (1) emit an EOC receipt for every resolved query, (2) make the receipts available to the reporter, and (3) supply I and M per ISO 21031 location-based methodology. The reporter is free to be a separate process from the resolver.
- **Versioning**: the extension URN carries a monotonic version suffix. v1 is this mapping; v2 will be issued if the EOC receipt schema changes in a non-backward-compatible way.
- **License**: this mapping document is **CC-BY-4.0**, consistent with the EOC specification and the GSF SCI specification.
- **Status**: the present text is a working draft. Filing to `github.com/Green-Software-Foundation/sci` and the conformance test suite are follow-up issues.

The extension imposes no obligation on deployments that do not use EOC and does not alter the base SCI-for-AI calculation. It is purely additive — a structured per-query data source that an SCI reporter may consume.

## h. References

- **GSF Software Carbon Intensity specification**: <https://github.com/Green-Software-Foundation/sci> (canonical text); <https://sci.greensoftware.foundation> (rendered).
- **SCI-for-AI ratification announcement**: Green Software Foundation Standards Working Group, December 2025.
- **ISO/IEC 21031:2024** — Information technology — Software measurement — Software Carbon Intensity (SCI) specification. <https://www.iso.org/standard/86612.html>.
- **EOC-1 specification** (substrate architecture, four-stage cascade): `eoc/spec/eoc1_v0_2.docx` and addendum `eoc/spec/eoc1_v0_2_addendum_A.docx`.
- **EOC-2 specification** (wire protocol, signed envelopes carrying receipts): `eoc/spec/eoc2_wire.docx`.
- **EOC-4 specification** (DCY query language for the graph stage): `eoc/spec/eoc4_dcy.docx`.
- **EOC reference implementation** (Rust): `eoc-rs/` (forthcoming; tracked in EOC roadmap).
- **Cookbook companion** (HuggingFace AI Energy Score integration): `eoc/cookbook/eoc-cascade-with-hf-energy-score.md`.
- **HuggingFace AI Energy Score**: <https://huggingface.co/spaces/AIEnergyScore/Leaderboard>.

---

*This document is the draft submission. The actual filing to `github.com/Green-Software-Foundation/sci` will be tracked as a follow-up issue against the GSF SCI repository, accompanied by a conformance test suite, a JSON-Schema definition at the canonical URL above, and a worked example reproducible from the cookbook companion.*
