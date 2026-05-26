//! Modality-specific handlers for different media types.
//!
//! Each modality (text, image, video, audio, 3D) has unique characteristics
//! that demand specialized handling while sharing common infrastructure.

pub mod text;
pub mod image;
pub mod video;
pub mod audio;
pub mod three_d;

use crate::core::Modality;
use alloc::boxed::Box;

/// Common trait for all modality handlers.
pub trait ModalityHandler: Send + Sync {
    /// Get the modality type.
    fn modality(&self) -> Modality;

    /// Optimal chunk size for streaming given available memory.
    fn optimal_chunk_size(&self, available_memory: usize) -> usize;

    /// Whether this modality supports streaming output.
    fn supports_streaming(&self) -> bool;

    /// Prefetch pattern for this modality.
    fn prefetch_pattern(&self) -> PrefetchPattern;

    /// Cache strategy for this modality.
    fn cache_strategy(&self) -> CacheStrategy;
}

/// Prefetch patterns for different modalities.
#[derive(Debug, Clone, Copy)]
pub enum PrefetchPattern {
    /// Sequential access (text tokens, video frames)
    Sequential,
    /// Random access (image tiles)
    Random,
    /// Locality-based (3D views from similar angles)
    Spatial,
    /// Temporal locality (recent frames more likely)
    Temporal,
}

/// Cache strategies for different modalities.
#[derive(Debug, Clone, Copy)]
pub enum CacheStrategy {
    /// Keep everything in cache (small data)
    KeepAll,
    /// LRU eviction
    Lru,
    /// Keep recent N items
    Recent(usize),
    /// Custom eviction based on access patterns
    Adaptive,
}

/// Input for any modality.
#[derive(Debug)]
pub enum ModalityInput {
    /// Text input.
    Text(text::TextInput),
    /// Image input.
    Image(image::ImageInput),
    /// Video input.
    Video(video::VideoInput),
    /// Audio input.
    Audio(audio::AudioInput),
    /// 3D input.
    ThreeD(three_d::ThreeDInput),
}

impl ModalityInput {
    /// Get the modality type.
    pub fn modality(&self) -> Modality {
        match self {
            Self::Text(_) => Modality::Text,
            Self::Image(_) => Modality::Image,
            Self::Video(_) => Modality::Video,
            Self::Audio(_) => Modality::Audio,
            Self::ThreeD(_) => Modality::ThreeD,
        }
    }
}

/// Output for any modality.
#[derive(Debug)]
pub enum ModalityOutput {
    /// Text output.
    Text(text::TextOutput),
    /// Image output.
    Image(image::ImageOutput),
    /// Video output.
    Video(video::VideoOutput),
    /// Audio output.
    Audio(audio::AudioOutput),
    /// 3D output.
    ThreeD(three_d::ThreeDOutput),
}

impl ModalityOutput {
    /// Get the modality type.
    pub fn modality(&self) -> Modality {
        match self {
            Self::Text(_) => Modality::Text,
            Self::Image(_) => Modality::Image,
            Self::Video(_) => Modality::Video,
            Self::Audio(_) => Modality::Audio,
            Self::ThreeD(_) => Modality::ThreeD,
        }
    }
}

/// Registry of modality handlers.
#[derive(Default)]
pub struct ModalityRegistry {
    handlers: dashmap::DashMap<Modality, Box<dyn ModalityHandler>>,
}

impl ModalityRegistry {
    /// Create a new registry with default handlers.
    pub fn new() -> Self {
        let registry = Self::default();
        registry.register(Box::new(text::TextHandler::new()));
        registry.register(Box::new(image::ImageHandler::new()));
        registry.register(Box::new(video::VideoHandler::new()));
        registry.register(Box::new(audio::AudioHandler::new()));
        registry.register(Box::new(three_d::ThreeDHandler::new()));
        registry
    }

    /// Register a modality handler.
    pub fn register(&self, handler: Box<dyn ModalityHandler>) {
        let modality = handler.modality();
        self.handlers.insert(modality, handler);
    }

    /// Get a handler for a modality.
    pub fn get(&self, modality: Modality) -> Option<impl core::ops::Deref<Target = Box<dyn ModalityHandler>> + '_> {
        self.handlers.get(&modality)
    }

}
