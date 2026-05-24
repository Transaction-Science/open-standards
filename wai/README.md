# WAI — Web AI Media Transport & Execution

A container + capability-dispatch standard for media. **WAI does not
re-implement codecs**; it dispatches to SOTA standard libraries (AVIF,
JPEG-XL, PNG, JPEG, Opus, FLAC, AV1, zstd, XZ) and adds an envelope
that lets the *neural-shared-prior* path coexist with the model-free
floor.

Two paths open every conforming file:

- **Neural condition** — the sink regenerates the media against a
  shared ambient prior (a model). The model is a *requirement of
  existence*, never shipped (an `.mp3` doesn't ship its decoder
  either).
- **Zeroth condition** — a *menu of registered SOTA codecs*. Always
  satisfiable on any device that ships ffmpeg or equivalent.

The capability a file requires is *named*, not supplied.

## What's here

| Path | Contents |
|------|----------|
| [`SPEC.md`](SPEC.md) | WAI v1.0 spec — envelope, manifest, capability menu |
| [`wai-rs/`](wai-rs/) | Rust reference implementation (lib + cdylib + staticlib + C FFI) |
| [`glossary/`](glossary/) | Reference catalog of media codecs surveyed for capability registration. [`codecs.json`](glossary/codecs.json) is the canonical 342-entry dataset; [`SCHEMA.md`](glossary/SCHEMA.md) defines the shape and contribution rules. Rendered at [wai.transaction.science/glossary](https://wai.transaction.science/glossary). |
| [`corpus/`](corpus/) | Standard test material (Kodak/CLIC/Tecnick/SIPI images, Xiph Derf + UVG video, SQAM-class audio, enwik8 + Silesia text). Not committed; `CORPUS.md` documents reacquisition. |

## Quickstart

```bash
cd wai-rs
cargo test --lib    # 11 tests — every registered capability + envelope round-trip
cargo build --release --bin wai
cargo build --release --lib    # produces libwai.dylib/.so/.dll + libwai.a
```

On macOS, libflac lives under the Homebrew "flac" formula prefix that
Cargo doesn't probe automatically. The committed `.cargo/config.toml`
adds the path; if you build outside the repo set
`RUSTFLAGS="-L /opt/homebrew/opt/flac/lib"`.

## SDK consumption

`wai-rs` exposes a stable C ABI in [`src/ffi.rs`](wai-rs/src/ffi.rs) —
the cdylib drops in as `libwai.dylib`/`.so`/`.dll`. Bindings for any
language (Python ctypes, Node N-API, Swift, JNI, Go cgo, etc.) call
into:

- `wai_image_{png,jpeg,avif,jxl}_{encode,decode}`
- `wai_audio_{opus,flac}_{encode,decode}`
- `wai_text_{zstd,xz}_{encode,decode}`
- `wai_envelope_{pack,unpack}` — wrap/unwrap the WAI container
- `wai_buffer_free`, `wai_last_error` — memory + error handling

## Registered capabilities (the zeroth menu)

| Capability | Codec | Lossless? | Mandatory? |
|---|---|---|---|
| `wai.image.png` | PNG | yes | **mandatory floor** |
| `wai.image.jxl` | JPEG-XL | both | recommended |
| `wai.image.avif` | AVIF (AV1-based) | lossy | recommended |
| `wai.image.jpeg` | JPEG | lossy | recommended (legacy compat) |
| `wai.audio.flac` | FLAC | yes | **mandatory floor** |
| `wai.audio.opus` | Opus | lossy | recommended |
| `wai.video.av1` | AV1 | lossy | recommended |
| `wai.video.av1.lossless` | AV1 (`quantizer=0`) | YUV-lossless | recommended |
| `wai.text.zstd` | zstd | yes | **mandatory floor** |
| `wai.text.xz` | XZ/LZMA2 | yes | recommended |

Neural capabilities (`wai.neural.*`) are OPTIONAL and sink-advertised.

## License

Apache-2.0 — see [`LICENSE`](LICENSE).
