//! `eoc-quantize` — quantization primitives + format converters.
//!
//! Energy-Optimized Compute lives or dies on bits-per-weight. This
//! crate is the reference implementation of the quantization primitives
//! used across the EOC stack: int2/int4/int8 weight quantization, fp8
//! and NF4 for low-rank adapters, calibration helpers, KV-cache
//! quantization, and read sketches for the dominant on-disk formats
//! (GGUF v3, safetensors).
//!
//! It also exposes an `energy_tradeoff` estimator that converts a
//! chosen scheme into an estimated joules-per-token figure — the bridge
//! between this crate and the wider EOC meter / cascade machinery.
//!
//! ## INGEST
//!
//! The implementations here are pure-Rust sketches inspired by the
//! public literature:
//!
//! * **int8** — symmetric and asymmetric PTQ
//!   (Jacob et al., 2018; Krishnamoorthi 2018).
//! * **int4** — group-quantized weights with per-group scale + zero
//!   point as used by GPTQ (Frantar et al., 2022) and AWQ
//!   (Lin et al., 2023).
//! * **int2** — sign-magnitude / ternary weights as in BitNet
//!   (Wang et al., 2023).
//! * **fp8** — E4M3 and E5M2 (OFP8 / Nvidia Hopper).
//! * **nf4** — NormalFloat4 (Dettmers et al., 2023, QLoRA).
//! * **GGUF v3** — llama.cpp on-disk format header sketch.
//! * **safetensors** — Hugging Face zero-copy tensor container.
//! * **SmoothQuant** (Xiao et al., 2022) — bias migration between
//!   activations and weights, exposed as a helper.
//! * **LLM.int8()** (Dettmers et al., 2022) — outlier-aware mixed
//!   precision (sketched in `int8::mixed_precision`).
//!
//! Nothing here calls a vendor SDK; everything is byte-shuffling and
//! arithmetic in `alloc`.

#![forbid(unsafe_code)]
#![deny(missing_docs)]

pub mod calibration;
pub mod energy_tradeoff;
pub mod error;
pub mod fp8;
pub mod gguf;
pub mod int2;
pub mod int4;
pub mod int8;
pub mod kv_cache;
pub mod nf4;
pub mod safetensors;
pub mod scheme;

pub use error::QuantError;
pub use scheme::{Numeric, QuantizationScheme};
