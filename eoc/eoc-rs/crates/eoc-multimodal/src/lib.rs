//! EOC multi-modal extensions.
//!
//! The base EOC cascade is text-only: a [`Query`](eoc_core::Query) carries a
//! prompt string and the four-stage cascade resolves it. To be a complete
//! substrate EOC must also ingest multi-modal queries (text + image + audio +
//! video frames), produce multi-modal embeddings, transcribe audio, synthesise
//! speech, and route each query to the model best suited to the modalities it
//! actually contains.
//!
//! This crate is that surface. It defines:
//!
//! * [`modality`] — the [`MultimodalQuery`] / [`QueryPart`] / [`Modality`] type
//!   backbone plus reference wrappers ([`ImageRef`], [`AudioRef`],
//!   [`VideoRef`]).
//! * [`vision`] — vision-language backends (`gpt-4o`, `claude-3.5-sonnet`,
//!   `gemini-2.0-pro`) and vision embedders (Cohere multimodal, CLIP, SigLIP,
//!   ColPali).
//! * [`audio`] — Whisper (API + local), Meta MMS, and TTS backends
//!   (OpenAI tts-1, Bark, MMS-TTS).
//! * [`video`] — keyframe extraction (pure-Rust subset by default, optional
//!   `ffmpeg` feature for the full pipeline).
//! * [`router`] — [`ModalityRouter`], which inspects a query's modalities and
//!   picks a text-only, vision-language, audio-language, or unified backend,
//!   emitting a [`Response`](eoc_core::Response) with joule attribution.
//!
//! ## Feature flags
//!
//! | flag      | enables                                                 | wasm |
//! |-----------|---------------------------------------------------------|------|
//! | (default) | vendor APIs, image preprocessing, router, frame sampling | yes  |
//! | `local`   | local CLIP / SigLIP / ColPali / Whisper / Bark / MMS    | no   |
//! | `ffmpeg`  | full video decode via `ffmpeg-next`                     | no   |
//!
//! Heavy native deps (`ort`, `ffmpeg-next`) sit behind features so the crate
//! compiles to `wasm32-unknown-unknown` with the default feature set.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod audio;
pub mod error;
pub mod modality;
pub mod router;
pub mod video;
pub mod vision;

pub use error::{MultimodalError, MultimodalResult};
pub use modality::{
    AudioRef, ImageRef, Modality, MultimodalQuery, QueryPart, VideoRef,
};
pub use router::ModalityRouter;
