//! LMM — Large Multimodal Model adapters.
//!
//! The Liquid AI LFM2-VL line (and similar small vision-language models)
//! consume text + images in a single forward pass: image patches are
//! linearly projected into the same embedding space as text tokens,
//! concatenated into a unified sequence, and run through one transformer
//! stack. The cascade sees this as a `Tokens` interface (vision tokens
//! and text tokens are indistinguishable to the router), so the I axis
//! of the synthesis coord stays at 4 values — no axis cardinality blow-up.
//!
//! What R31 enables:
//!
//! - `QueryInput::{Image, Audio, Multimodal}` variants in `jouleclaw-cascade`,
//!   so callers can express multimodal queries directly. The cascade and
//!   history layers hash and persist these variants alongside Text.
//! - [`preprocess`] — pure-Rust image preprocessing: bilinear resize,
//!   per-channel mean/std normalization, and ViT-style patchification.
//!   Operates on already-decoded RGB float tensors; raw byte → RGB
//!   decoding is R31.1.
//! - [`tier::LmmTier`] — joule cascade tier shell. Coordinate declared,
//!   honest floor cost reported, full forward pass + tokenizer +
//!   sampling is R31.1.
//!
//! Coordinate:
//!
//!   Z = Z2_3         (statistical inference at small-medium scale)
//!   E = Reactive
//!   T = L1_Measure
//!   I = Tokens       (image patches are tokens once projected)
//!   V = Statistical
//!   R = Facts
//!   P = { MlpForward, AttentionGrouped, Embed, Sample }

pub mod preprocess;
pub mod tier;
pub mod vision;
pub mod vl_real;
pub mod vlm;

pub use vision::{VisionError, VisionTower};
pub use vl_real::{LfmVl, VlError};

pub use preprocess::{
    bilinear_resize, normalize, patchify, ImageError, PreprocessConfig, RgbImage,
};
pub use tier::LmmTier;
pub use vlm::{
    audio_to_tokens, build_multimodal_tokens, generate_multimodal, image_to_tokens,
    AUDIO_TOKENS_PER_CLIP, VISION_TOKENS_PER_IMAGE,
};
