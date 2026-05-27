# JouleClaw roadmap

Where the standard is going after v0.1.0 (currently committed to
`main`). The reference implementation in [`jouleclaw-rs/`](jouleclaw-rs/)
is 30 crates, ~170k LOC, and all tests pass.

## Deferred work, by phase

### Near-term (v0.2)

- **`gemma4` port.** Pattern-lang's `gemma4` tier registers itself
  against `joule-l1`'s lawful lexicon and `pattern-core`'s synthesizer
  for prompt routing. With Phase 6's `jouleclaw-cascade::lawful`
  trait surface now landed, the port is unblocked — but the
  `pattern-core` synthesizer routing path needs the consumer (or a
  thin `pattern-core`-shaped consumer crate inside JouleClaw) to plug
  through `LawfulRegistry`. Tracked.
- **`jouleclaw-edge-server`.** Pattern-lang's `joule-edge/src/server.rs`
  was an HTTP wrapper around the demo runtime. The library form
  (`jouleclaw-edge`) shipped; the server form is a thin axum wrapper
  that can land any time. Out-of-scope for v0.1 because the demo
  binary covers the same surface for development.
- **Conformance vector population.** `conformance/v1/` ships the
  three structural vectors (cache-hit, breaker-trip,
  joule-mcp-CBOR-negotiation). The remaining seven vectors named in
  `conformance/v1/README.md`'s coverage matrix
  (lawful-arithmetic, embed-hybrid-search, fresh-retrieval-with-
  provenance, model-local-ssm, model-local-ternary, model-local-
  diffusion-image, wire-frontier-rpc) await the canonical receipts
  that fall out of running the reference impl with the standard
  reference-hardware (Apple M5 Max baseline). Each is a few hundred
  lines of JSON + a `.jc.toml` sidecar.
- **`SHA256SUMS` + release signature.** The v1.0.0 conformance
  vector corpus signs at release time. Tracked alongside the v1.0.0
  release tag.

### Medium-term (v0.3)

- **Split `jouleclaw-omni`.** The ~105k-LOC monolith ported wholesale
  from `efficient-genai` in Phase 5 preserves the donor's internal
  layout (`core / hal / inference / modalities / orchestrator /
  quality / runtime / server / tensor / weight_store`). Splitting
  into separate crates is multi-week surgery because the internal
  cross-references (`crate::hal::*`, `crate::tensor::*`,
  `crate::core::*`) need to be rewired into proper Cargo path deps.
  The plan when this lands:
  - `jouleclaw-hal` — `Device / Kernel / Memory / Capabilities`
    trait surface
  - `jouleclaw-tensor` — the tensor module
  - `jouleclaw-diffusion` — the 40+ sampler catalog
  - `jouleclaw-musicgen`, `jouleclaw-whisper`, `jouleclaw-gaussian3d`,
    `jouleclaw-video`, `jouleclaw-fusion` — per-engine crates
  - `jouleclaw-modality-{text,image,audio,video,3d}` — modality
    surfaces
  - `jouleclaw-omni-server` — axum server, optional
  - `jouleclaw-omni` becomes a meta-crate that re-exports the above
    behind a `legacy` feature for downstream consumers who don't
    want to re-find all the names.

  Until then, `jouleclaw-omni` ships as one crate with feature gates
  (`cuda`, `metal`, `rocm`, `vulkan`, `safetensors`, `jit`, `simd`,
  `rayon`, `server`).

- **Standalone GPU-backend crates.** `jouleclaw-omni::hal::*` already
  has the CUDA / Metal / ROCm / Vulkan paths feature-gated. Extracting
  them as standalone `jouleclaw-backend-cuda`, `-rocm`, `-vulkan`
  crates is a sub-task of the omni split above and lands in the same
  v0.3 work. Until then, downstream consumers enable the feature on
  `jouleclaw-omni` and the backend is available behind
  `jouleclaw_omni::hal::cuda::*` etc.

### Long-term (v1.0)

- **Multi-machine cascade.** The current cascade assumes a single
  box. v1.0 will define how a JouleClaw deployment routes work across
  a fleet — content-addressed cache sharing, per-node energy budgets,
  cross-node receipt aggregation. Out-of-scope for v0.x.
- **Hardware-attested measurement.** Some platforms (Apple Secure
  Enclave, NVIDIA Hopper Confidential Compute) can attest energy
  measurements cryptographically. Receipt schema v2 will carry the
  attestation envelope as an optional field.
- **A second reference implementation.** A second language (Go or
  Zig) reference impl that round-trips the same conformance vectors
  is the long-term proof that the standard is implementable cleanly.

## Not on the roadmap

- **Training framework.** JouleClaw is a runtime + harness. Training
  is the model author's concern.
- **A specific signing key topology.** Bring your own. Smart Byte's
  KERI-based AID rotation is the recommended path; the standard
  doesn't require it.
- **A specific retrieval API.** Brave / Tavily / Exa / Serper all
  plug in through `jouleclaw-fresh::SearchProvider`. The standard
  doesn't endorse one.
