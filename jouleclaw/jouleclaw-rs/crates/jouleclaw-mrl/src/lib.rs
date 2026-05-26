//! MRL — Matryoshka Representation Learning adapters.
//!
//! Kusupati, Bhatt, Rege et al., "Matryoshka Representation Learning"
//! (NeurIPS 2022, arXiv:2205.13147). MRL trains a single embedding so
//! that every prefix is itself a valid embedding at a smaller dimension.
//! A 2048-d model can be sliced to {2048, 1024, 512, 256, 128, 64, 32,
//! 16, 8} dims at inference time with no retraining and a quality drop
//! that's gentle for tasks like retrieval. The paper reports up to
//! 14× retrieval speedup with ≤2% quality loss on ImageNet.
//!
//! What MRL changes for joule:
//!
//! - **Computation savings** are modest at the embed step (you mostly
//!   still compute the full forward pass, then drop the tail).
//! - **Retrieval savings are the real win.** Nearest-neighbor search
//!   over a corpus of N vectors at dim D is O(N·D). Picking D=64 vs
//!   D=512 cuts the search cost 8× linearly. This is what the cascade
//!   routes against.
//!
//! Scope at R30.0 (this revision):
//!
//! - [`embedder::Embedder`] — abstract trait for an underlying model.
//! - [`embedder::IdentityEmbedder`] — a deterministic mock for tests.
//! - [`matryoshka::MatryoshkaEmbedder`] — wraps any `Embedder`, exposes
//!   a sorted dim ladder, supports `embed_at_dim`, and carries a quality
//!   model + retrieval cost model.
//! - [`picker::DimPicker`] — chooses the smallest dim meeting a quality
//!   floor under a per-query retrieval budget.
//! - [`tier::MrlTier`] — joule cascade tier shell. Coordinate declared,
//!   honest floor cost reported, full corpus + nearest-neighbor lookup
//!   integration is R30.1.

pub mod embedder;
pub mod gguf_embedder;
pub mod matryoshka;
pub mod picker;
pub mod retrieval;
pub mod tier;

pub use embedder::{Embedder, EmbedderError, IdentityEmbedder};
pub use gguf_embedder::{GgufTextEmbedder, EmbedError, PoolingKind};
pub use matryoshka::{MatryoshkaEmbedder, QualityModel};
pub use picker::{DimPicker, PickError};
pub use retrieval::{Corpus, CorpusDoc, RetrievalError, RetrievalHit};
pub use tier::MrlTier;
