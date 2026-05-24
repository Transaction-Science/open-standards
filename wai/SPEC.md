# WAI v1.0 — Web AI Media Transport & Execution

Status: **Draft Standard**. Reference implementation: `wai-rs/` (Rust
lib + cdylib + staticlib, C FFI). Two paths open every conforming
file:

- the **neural condition** — the sink regenerates the media from a
  compact conditioning payload against a *shared ambient prior* (a
  model). The model is a **requirement of existence**, never
  transported and never hash-pinned, exactly as `.mp3` does not ship
  its decoder.
- the **zeroth condition** — a *menu of registered SOTA standard
  codecs*. Always satisfiable on any device that ships ffmpeg or
  equivalent. WAI does **not** define new codecs; it dispatches to the
  field's best-in-class lossy+lossless libraries (AVIF, JPEG-XL, PNG,
  Opus, FLAC, AV1, zstd, XZ).

The capability a file requires is *named*, not supplied. Sinks that
have the named capability use it; sinks that don't fall back to a
declared `fallback` capability.

Keywords **MUST**, **MUST NOT**, **SHOULD**, **MAY** are RFC 2119.

---

## 1. Scope and model

WAI is a **container + capability dispatch standard**. It is media-
general (image, audio, video, text), royalty-free, and built around the
principle that the standard's value is the envelope and the dispatch
rules — not in re-implementing existing SOTA codecs.

A WAI file declares (in its manifest):
- what media it carries,
- what *capability* a sink must have to decode it,
- a fallback capability the sink can use if it lacks the primary,
- and the conditioning bytes for that capability.

Conforming sinks MUST implement the container envelope (§2), the
manifest schema (§3), the capability-dispatch algorithm (§4), and at
least the **mandatory floor** capabilities (§5). Other capabilities
(neural and additional zeroth) are OPTIONAL but RECOMMENDED.

---

## 2. Container

```
+--------+----------------+-------------------+----------------+-----------+
| "WAI1" | u32  man_len   | manifest (JSON)   | u32 payload_len| payload   |
| 4 B    | little-endian  | man_len bytes     | little-endian  | bytes     |
+--------+----------------+-------------------+----------------+-----------+
```

- Bytes 0..4 MUST be ASCII `WAI1`. A reader MUST reject any file whose
  first 4 bytes differ.
- `man_len` and `payload_len` are unsigned 32-bit little-endian.
- The manifest is UTF-8 JSON, serialized with no insignificant
  whitespace, preserving key insertion order. Readers MUST NOT depend
  on key order for *semantics*; encoders SHOULD preserve order so files
  round-trip byte-identically.
- The payload is opaque codec bytes whose interpretation is fully
  determined by the manifest's `model_requirement.capability`.

All WAI multi-byte integers are little-endian unless stated otherwise.

---

## 3. Manifest schema

```json
{
  "wai": "1.0",
  "media": "image" | "audio" | "video" | "text",
  "intent": "replicate" | "create" | "improve" | "...",
  "model_requirement": { "capability": "<capability-string>",
                         "fallback":   "<capability-string>" | null },
  "conditioning": { "kind": "<codec-id>" },
  "target":   { ... }
}
```

- `wai` — spec version. This document defines `"1.0"`. Unknown major →
  reject. Unknown minor → accept by ignoring unknown fields.
- `media` — media class; informational. The capability is authoritative.
- `intent` — generative intent. `replicate` (reconstruct the source) is
  the only intent with normative reconstruction semantics in v1.0.
- `model_requirement.capability` — the capability string a sink MUST
  resolve. See §5 for the registered set.
- `model_requirement.fallback` — a second capability the sink MAY use
  if it lacks the primary. `null` means the file is inert without the
  primary.
- `conditioning.kind` — short codec id matching the capability. Helps
  multiplexers route bytes without parsing the full capability string.
- `target` — codec-specific reconstruction parameters (dimensions,
  sample rate, duration, frame count). Informational; the payload is
  self-describing per the codec's own format.

---

## 4. Capability dispatch

A conforming sink MUST implement this algorithm:

1. Read the manifest. If `wai` major version is unknown → reject.
2. Let `cap = model_requirement.capability`.
3. If the sink advertises `cap`, decode `payload` via the codec
   registered for `cap` (§5) and return.
4. Else, let `fb = model_requirement.fallback`.
5. If `fb != null` and the sink advertises `fb`, decode via `fb`'s
   codec and return.
6. Else the file is **inert at this sink** — return a clear "missing
   capability" error to the calling application.

A WAI **payload always corresponds to exactly one capability**. v1.0
does not support multi-codec multiplexed payloads.

---

## 5. Registered capabilities (the zeroth menu)

WAI v1.0 registers the following capabilities. Each names a SOTA
royalty-free standard codec. The "payload format" column is the
on-the-wire payload bytes; in every case it is the canonical bytes
that codec normally emits (a JXL file, an Opus stream, etc.) — WAI
does not wrap or re-frame them.

### Mandatory floor

Every conforming sink MUST implement:

| capability | payload | role |
|---|---|---|
| `wai.image.png` | PNG (RFC 2083) | universal lossless image |
| `wai.audio.flac` | FLAC (xiph.org/flac) | universal lossless audio |
| `wai.text.zstd` | zstd (RFC 8478) | universal compressed bytes |

These three guarantee that any sink with the standard system codec
libraries can open *some* file in every WAI-supported media class.

### Recommended modern set

Every conforming sink SHOULD implement:

| capability | payload | role |
|---|---|---|
| `wai.image.jxl` | JPEG-XL (ISO/IEC 18181) | lossless + lossy, modern SOTA |
| `wai.image.avif` | AVIF (ISO/IEC 23000-22) | lossy SOTA (AV1-based) |
| `wai.image.jpeg` | JPEG (ITU-T T.81) | legacy compatibility |
| `wai.audio.opus` | Opus (RFC 6716) | lossy SOTA |
| `wai.video.av1` | AV1 (AOMedia) | lossy SOTA |
| `wai.video.av1.lossless` | AV1 (AOMedia), `quantizer=0` | mathematically lossless YUV (lossy in RGB due to color conversion) |
| `wai.text.xz` | XZ/LZMA2 | maximum classical text ratio |

### Neural capabilities (OPTIONAL, sink advertises if its ML runtime supports them)

Each names a model the sink MAY have installed. Payload is a compact
conditioning representation specific to the capability (latent tokens,
prompt embeddings, etc.). The standard does NOT specify the byte
layout of these payloads — the capability owner does.

Neural capabilities are picked on the **best-quality-per-bit at each
bitrate band** principle — not on whatever model is famous or small.
2026 SOTA per medium, with the wire-byte win each one delivers:

| capability | model | win vs zeroth | source |
|---|---|---|---|
| `wai.neural.encodec32` | Meta EnCodec at 32 kHz, 4 codebooks | ~3 kbps acceptable music | `facebook/encodec_32khz` |
| `wai.neural.dac` | Descript Audio Codec at 44.1 kHz | 6 kbps near-transparent music | `descript/dac_44khz` |
| `wai.neural.mimi` | Kyutai Mimi, 12.5 Hz frame rate | ~1.1 kbps real-time speech | `kyutai/mimi` |
| `wai.neural.wavtokenizer` | WavTokenizer single-codebook VQ | 0.9 kbps beats DAC @ 9 kbps on UTMOS | `novateur/WavTokenizer-large-*` |
| `wai.neural.bmshj2018` | bmshj2018-factorized + rANS (byte-exact CompressAI port) | ~80× vs raw RGB at q=3 (~0.26 bpp, 29 dB PSNR); 15-20% smaller than the zstd-packed variant | CompressAI |
| `wai.neural.video_bmshj2018` | per-frame bmshj2018 | 1000×+ vs raw RGB, browser-decodable on WebGPU | CompressAI |
| `wai.neural.glc` | Generative Latent Coding (future) | <0.05 bpp where JPEG-XL collapses | research code |
| `wai.neural.dcvc_rt` | DCVC-RT (future, native-sink only) | AV1 quality at 21% less bitrate, 112 fps 1080p | requires NVIDIA CUDA + custom kernels |

**Sink-architecture constraint:** DCVC-RT and similar inter-frame neural
video codecs require NVIDIA CUDA and custom CUDA kernels at decode time
— they don't deploy to browser sinks. WAI declares the capability for
native-sink deployers, but browser sinks should declare only
`wai.neural.video_bmshj2018` (or any future browser-decodable neural
video codec) for the `video` media class.

**TAESD is intentionally absent.** TAESD is a distilled SD VAE decoder
— it inverts a generative pipeline's encoder, not a general image
codec. Outside the SD pipeline it underperforms JPEG-XL at most
bitrates while costing ~5 MB of weights and ONNX compute. Don't ship
codecs for the brand; ship them for the bit budget.

**Image/video scope is honest:** zeroth codecs (JPEG-XL, AV1) are
already SOTA-adjacent in the bitrate bands where most media lives.
Neural image/video capabilities are registered for the bands where
zeroth codecs collapse (ultra-low bpp images, bandwidth-constrained
streaming) — not as defaults.

Each capability owns its payload format. Decoder implementations
(Python source-side, ONNX-export, browser onnxruntime-web) live in
`tools/wai_<capability>_*.py` and `wai-web/demo/`. New neural
capabilities are added by registration; the WAI envelope is unchanged.

---

## 6. Conformance

- A **conforming sink** MUST: implement the container (§2), capability
  dispatch (§4), and the mandatory floor (§5).
- A **conforming encoder** MUST emit containers (§2) that any
  conforming sink can read; the payload MUST be the canonical byte
  stream for the registered capability.
- Conformance is verified by **bit-exact decode-equivalence with the
  reference library** named in §5 for each registered capability. The
  WAI Rust reference impl in `wai-rs/` is built on those very
  libraries and ships a test suite that exercises every registered
  capability through the full WAI envelope (`cargo test --lib`).

---

## 6.1. Multi-rendition envelope (v1.1 extension)

The v1.0 single-payload envelope is sufficient when one source produces
one rendition for one capability. v1.1 adds an OPTIONAL multi-payload
form for cases where the source emits the SAME content in multiple
renditions and lets the sink pick — typically because the sink's
compute budget, bandwidth budget, or runtime preference (browser-
native vs WebGPU vs WebNN vs native ML) differs at decode time.

The `model_requirement.fallback` field in v1.0 is **declarative** — it
names a capability the sink CAN look for. v1.1 makes fallback (and
multi-rendition selection generally) **executable** by carrying every
declared capability's payload in one envelope.

### Self-contained-ecosystem note

In a controlled-deployment ecosystem (every sink ships from one
authority and therefore has a known capability set) the multi-rendition
form is NOT a "graceful fallback for missing capability" — every sink
has every required capability. Renditions exist for **policy selection**:
high-quality on a desktop sink with WebGPU; low-quality on a phone sink;
quality-vs-latency by bandwidth headroom; native vs WebNN by which
runtime is initialized. The sink picks by deployer-defined policy, not
by which capability happens to be installed.

### v1.1 wire format

```
+--------+----------------+-------------------+----------------+----...---+
| "WAI2" | u32  man_len   | manifest (JSON)   | rendition table| payloads |
+--------+----------------+-------------------+----------------+----...---+
```

- Magic is `WAI2` (distinct from v1.0's `WAI1`; sinks dispatch by magic).
- Manifest is JSON with a single `renditions` array of self-contained
  rendition entries, in deployer-preferred order:
  ```json
  {
    "wai":    "1.1",
    "media":  "audio",
    "intent": "replicate",
    "renditions": [
      { "capability": "wai.neural.encodec32", "kind": "encodec_tokens" },
      { "capability": "wai.audio.opus",       "kind": "opus"           }
    ],
    "target": { "sr": 48000, "dur": 5.0 }
  }
  ```
  Each entry has its own `capability` (the dispatch key) and `kind`
  (the codec id matching the bytes in the corresponding payload).
- Rendition table: `u16 n_renditions | n × (u32 offset, u32 length)`,
  offsets relative to the start of the payloads block. The table's
  i-th entry corresponds to `renditions[i]` in the manifest.
- Payloads block: the rendition byte ranges, packed contiguously.
- `n_renditions` MUST equal the manifest's `renditions.length`.

### Selection algorithm

A v1.1-aware sink MUST:

1. Read the manifest, get the ordered list of `model_requirement`
   entries.
2. Walk the list in order; pick the first entry whose capability the
   sink either advertises OR whose declared backend the sink prefers
   under its deployer policy.
3. Read the corresponding rendition's bytes from the payload block.
4. Dispatch as in v1.0 §4.

A v1.0 sink that encounters a `WAI2` envelope MUST refuse it cleanly
(unknown major). No silent fallback to the first payload — that would
hide a major-version mismatch.

### v1.0 compatibility

v1.0 envelopes (`WAI1`) remain valid and are unchanged. Encoders
SHOULD emit `WAI1` when only one rendition is present (simpler form,
wider sink support). `WAI2` is for sources that genuinely benefit from
shipping multiple renditions per file.

---

## 7. Versioning

- The `wai` manifest field is `MAJOR.MINOR`. A reader MUST reject an
  unknown MAJOR. A reader MUST accept an unknown MINOR by ignoring
  fields it does not understand, provided MAJOR matches.
- The container format (§2) and the manifest schema (§3) are frozen
  for all of `1.x`.
- New capabilities (§5) MAY be added in any MINOR revision. Existing
  capabilities MUST NOT be redefined — once a capability string is
  registered for a codec, its payload format is fixed forever.
- Removing or repurposing an existing capability requires a new MAJOR.

---

## Appendix A — Reference implementation

`wai-rs/` (Rust). Builds as `lib`, `cdylib`, `staticlib` so the C ABI
in `wai-rs/src/ffi.rs` drops in as `libwai.dylib` / `.so` / `.dll` and
can be called from any language. The crate wraps:

- **image**: `image` crate (PNG, JPEG, AVIF via `ravif`), `jpegxl-rs`
  (libjxl 0.11.x).
- **audio**: `opus` (libopus 1.x), `claxon` + `flac-bound` (libFLAC).
- **video**: `rav1e` (AV1 encode) + `dav1d` (AV1 decode).
- **text**: `zstd` (libzstd), `xz2` (liblzma).

Run `cargo test --lib` (with `RUSTFLAGS="-L /opt/homebrew/opt/flac/lib"`
on macOS, or via the committed `.cargo/config.toml`).

## Appendix B — Why a *capability menu*, not a custom codec

Earlier drafts of WAI defined custom transform + entropy stages for
each medium. That direction was dropped: re-implementing JPEG / MPEG /
Opus poorly is strictly worse than calling the field's mature
libraries. WAI's value is the envelope + capability dispatch + the
neural-shared-prior model, all of which are absent from existing
standards. The zeroth condition's purpose is **availability**, not
codec novelty.

## Appendix C — License

Apache-2.0. The standard and reference implementation are open;
royalty-free implementation is a precondition for adoption (avoiding
the H.264/H.265 royalty trap; following the VP9/AV1/AVIF/JPEG-XL open-
codec precedent).
