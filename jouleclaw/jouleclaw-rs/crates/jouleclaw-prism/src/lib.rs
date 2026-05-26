//! Prism — 1-bit and ternary matrix kernels for ultra-low-energy inference.
//!
//! The PrismML thesis (Caltech): a model whose weights are constrained to
//! {-1, 0, +1} (ternary) or {-1, +1} (1-bit) replaces every multiply in
//! its matmuls with a conditional add. Per the PrismML public claims, this
//! delivers ~14× smaller footprint, ~8× faster execution, ~5× lower
//! energy versus full-precision counterparts. The math is exact when the
//! weights have been quantization-aware-trained; the runtime savings come
//! from doing arithmetic that better matches what silicon is good at.
//!
//! This crate provides:
//!
//! - [`ternary::TernaryMatrix`] — packed {-1, 0, +1} weights with a
//!   per-row f32 scale. 2 bits per weight (4 trits per byte) — looser
//!   than upstream TQ1_0 (1.5 bpw) but simpler to verify correctness on.
//!   R28.1 will add the tighter 5-trits-per-byte packing.
//!
//! - [`bit::BitMatrix`] — packed {-1, +1} weights with a per-row f32
//!   scale. 1 bit per weight, 8 weights per byte.
//!
//! - [`tier::PrismTier`] — joule cascade tier shell. The dispatch surface
//!   is in place and reports honest cost/coord; full model integration
//!   (load Bonsai-class weights, run forward pass through transformer
//!   layers, sample tokens) is R28.1 — the kernels here are the floor it
//!   builds on.
//!
//! All kernels are deterministic, bit-reproducible, and pure-Rust — no
//! SIMD intrinsics, no `unsafe`. They establish correctness first;
//! hardware-specific acceleration belongs in `joule-backend-*` crates.

pub mod bit;
pub mod forward;
pub mod model;
pub mod ternary;
pub mod tier;

pub use bit::BitMatrix;
pub use forward::{rmsnorm, silu, softmax, TernaryBlock, TernaryDecoder};
pub use model::{synthetic_model, ModelConfig};
pub use ternary::TernaryMatrix;
pub use tier::PrismTier;
