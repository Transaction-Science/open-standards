//! The [`Embedder`] trait — uniform surface across vendors and local backends.

use async_trait::async_trait;
use eoc_core::JouleCost;

use crate::error::EmbeddingResult;

/// A text-embedding backend.
///
/// Implementations encode one or more strings into fixed-dimension
/// floating-point vectors. All implementations are `Send + Sync` so the
/// EOC runtime can share a single embedder across stages.
#[async_trait]
pub trait Embedder: Send + Sync {
    /// Embed a batch of texts into vectors of length [`Self::dimensions`].
    async fn embed(&self, texts: &[&str]) -> EmbeddingResult<Vec<Vec<f32>>>;

    /// Dimensionality of returned vectors.
    fn dimensions(&self) -> usize;

    /// Canonical model identifier (e.g. `"text-embedding-3-small"`).
    fn model_name(&self) -> &str;

    /// Estimated energy cost for embedding `text_len_chars` characters.
    ///
    /// This is a coarse estimate based on per-model energy profiles in
    /// [`crate::joule_estimator`]. Real measurements come from
    /// [`eoc_meter`] when hardware counters are present.
    fn joule_estimate(&self, text_len_chars: usize) -> JouleCost;
}
