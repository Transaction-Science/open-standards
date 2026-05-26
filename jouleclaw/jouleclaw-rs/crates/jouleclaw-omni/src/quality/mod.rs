//! Output quality validation for generated content.
//!
//! Provides quantitative, objective metrics to detect poorly implemented
//! algorithms that produce meaningless output (noise, silence, repetition).
//!
//! ## Modality-specific metrics
//!
//! - **Image**: entropy, sharpness, saturation, uniformity detection
//! - **Audio**: RMS energy, crest factor, silence/clipping detection
//! - **Text**: unique token ratio, repetition detection, OOV count
//! - **Video**: per-frame image metrics + temporal coherence
//!
//! ## Usage
//!
//! ```rust,ignore
//! use efficient_genai::quality::{image, audio, text};
//!
//! // Image quality check
//! let pixel_data: Vec<f32> = tensor.to_f32_vec().unwrap();
//! let report = image::image_report(&pixel_data, 3, 512, 512);
//! if !report.is_valid || report.is_uniform {
//!     eprintln!("Image quality warning: {}", report.summary());
//! }
//!
//! // Audio quality check
//! let samples: Vec<f32> = vec![/* PCM data */];
//! let report = audio::audio_report(&samples, 16000);
//! if report.is_silence {
//!     eprintln!("Audio is silent!");
//! }
//!
//! // Text quality check
//! let tokens: Vec<u32> = vec![/* token IDs */];
//! let report = text::text_report(&tokens, 32000, None);
//! if report.is_degenerate {
//!     eprintln!("Text output is degenerate: {}", report.summary());
//! }
//! ```

/// Core statistics (mean, std, histogram, entropy, percentile).
pub mod stats;

/// Image quality metrics.
pub mod image;

/// Audio quality metrics.
pub mod audio;

/// Text generation quality metrics.
pub mod text;

/// Unified quality report and serialization.
pub mod report;

// Re-exports for convenience
pub use audio::AudioReport;
pub use image::ImageReport;
pub use report::{QualityReport, report_to_json, temporal_coherence};
pub use stats::TensorStats;
pub use text::TextReport;
