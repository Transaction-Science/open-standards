//! Vision-language backends + vision embedders.
//!
//! Two distinct surfaces live in here:
//!
//! * **Vision-language inference** ([`openai_vision`], [`anthropic_vision`],
//!   [`google_vision`]): wrap the chat endpoint of each vendor with the
//!   right content-block schema so a [`crate::MultimodalQuery`] with image
//!   parts produces a coherent answer.
//! * **Vision embedders** ([`clip`]): produce fixed-dimension vectors from
//!   images so the [`eoc_kv`]-style cache can hit on visual similarity.

pub mod anthropic_vision;
pub mod clip;
pub mod google_vision;
pub mod openai_vision;
pub mod preprocess;

pub use anthropic_vision::AnthropicVisionBackend;
pub use clip::{CohereVisionEmbedder, VisionEmbedder};
pub use google_vision::GoogleVisionBackend;
pub use openai_vision::OpenAiVisionBackend;
