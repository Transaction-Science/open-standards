# WAI codec glossary — schema

Canonical, machine-readable catalog of media codecs surveyed for WAI capability registration. Lives in [`codecs.json`](codecs.json) alongside this schema.

The glossary is reference material — it is **not** the registry of WAI capabilities (that lives in [`../SPEC.md §5`](../SPEC.md)). Listing a codec here means it exists and is documented; listing a codec in the spec means a conforming sink MAY or MUST implement it.

## Provenance

Originally extracted from the inline JavaScript array in `wai-info-page/wai-glossary.html` (the user's purpose-built reference page) into typed form, then ported to JSON for language-neutral consumption.

Mirrored as a typed Astro module at [`wai-transaction-science-web/src/data/codecs.ts`](https://wai.transaction.science/glossary) — the site renders the JSON statically at build time so search and filters work without JavaScript.

## Top-level shape

```json
{
  "schema":  "https://github.com/Transaction-Science/open-standards/blob/main/wai/glossary/SCHEMA.md",
  "entries": [ Codec, ... ]
}
```

## `Codec` entry

```ts
{
  name:     string;     // canonical short name, e.g. "AV1", "HEVC", "JPEG-XL"
  year:     number;     // year of first published spec or first release
  org:      string;     // standards body, vendor, or research group that owns lineage
  modality: "image" | "video" | "audio" | "speech";
  era:      "classical" | "neural";
  status:   "success" | "niche" | "superseded" | "failed" | "academic" | "emerging";
  lineage:  "industry" | "standards" | "academic" | "opensource";
  desc:     string;     // 1–3 sentence factual summary — what it is, who made it,
                        // its design choice, its current footprint. Self-referential
                        // only (no "better than X", no competitor framing).
}
```

### Field values

- **`status`** is descriptive, not normative:
  - `success` — the codec carries non-trivial production traffic today.
  - `niche` — alive in specific contexts (icons, pipelines, debug tools, particular hardware).
  - `superseded` — readable everywhere, written nowhere modern.
  - `failed` — shipped, did not achieve adoption.
  - `academic` — published only in research; no production deployment.
  - `emerging` — recent enough that adoption trajectory is undetermined.
- **`lineage`** captures origin culture (affects how the codec evolves):
  - `industry` — vendor-driven (a company shipping a product).
  - `standards` — formal body (ISO, ITU-T, IETF, IEEE, SMPTE).
  - `academic` — research lab.
  - `opensource` — community-led from inception.
- **`era`** marks the discontinuity: `classical` = handcrafted DCT/wavelet/predictive, `neural` = end-to-end learned bitstream.

## Contributing

Two operations:

1. **Add a codec** — append a new entry to `entries[]`. Keep entries roughly chronological within their modality block.
2. **Correct a description** — edit `desc`. Aim for factual density over flair: spec year, the spec body that owns it, the architectural choice that defines it, its current deployment footprint. No comparisons; no opinions about quality unless backed by a citation in the description.

A patch that touches only `codecs.json` requires no test run. A patch that adds a new field or status enum value should:

1. Update `SCHEMA.md` first (define the field).
2. Update the typed module at `wai-transaction-science-web/src/data/codecs.ts` and the glossary page that consumes it (`wai-transaction-science-web/src/pages/glossary.astro`).

## Relationship to the WAI spec

A codec listed here becomes a WAI capability (e.g., `wai.image.jxl`) only when the spec registers it in `SPEC.md §5`. Registration carries normative requirements: a canonical reference library, a payload format declaration, and conformance via bit-exact decode-equivalence. The glossary documents the full landscape; the spec selects from it.
