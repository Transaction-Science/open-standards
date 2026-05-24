//! The multi-modal type backbone.
//!
//! [`MultimodalQuery`] is the input to the [`crate::router::ModalityRouter`].
//! It carries an ordered list of [`QueryPart`]s — each of which is either a
//! text fragment, an image reference, an audio reference, or a video
//! reference. The router inspects the set of [`Modality`]s present in the
//! query to pick a backend.
//!
//! References ([`ImageRef`], [`AudioRef`], [`VideoRef`]) are deliberately
//! enums so the caller can pass any of:
//!
//! * a URL the vendor fetches itself;
//! * an in-memory byte buffer with content-type;
//! * a base64-encoded string (vendor-ready);
//! * a local file path (resolved at send time).

use std::collections::BTreeMap;
use std::path::PathBuf;

use base64::Engine as _;
use base64::engine::general_purpose::STANDARD as B64;
use eoc_core::QueryId;
use serde::{Deserialize, Serialize};

use crate::error::{MultimodalError, MultimodalResult};

/// A single media modality.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Modality {
    /// Plain text.
    Text,
    /// A still image.
    Image,
    /// An audio clip.
    Audio,
    /// A video clip (decomposable into frames + audio track).
    Video,
}

/// A reference to an image payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ImageRef {
    /// Vendor-fetchable URL.
    Url(String),
    /// In-memory bytes with explicit content-type (e.g. `image/png`).
    Bytes {
        /// MIME type of the payload.
        content_type: String,
        /// Raw image bytes.
        bytes: Vec<u8>,
    },
    /// Pre-encoded base64 string (vendor-ready).
    Base64(String),
    /// Path on the local filesystem (resolved lazily).
    File(PathBuf),
}

/// A reference to an audio payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AudioRef {
    /// Vendor-fetchable URL.
    Url(String),
    /// In-memory bytes with explicit content-type (e.g. `audio/wav`).
    Bytes {
        /// MIME type of the payload.
        content_type: String,
        /// Raw audio bytes.
        bytes: Vec<u8>,
    },
    /// Pre-encoded base64 string.
    Base64(String),
    /// Path on the local filesystem.
    File(PathBuf),
}

/// A reference to a video payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum VideoRef {
    /// Vendor-fetchable URL.
    Url(String),
    /// In-memory bytes with explicit content-type (e.g. `video/mp4`).
    Bytes {
        /// MIME type of the payload.
        content_type: String,
        /// Raw video bytes.
        bytes: Vec<u8>,
    },
    /// Pre-encoded base64 string.
    Base64(String),
    /// Path on the local filesystem.
    File(PathBuf),
}

/// One part of a multi-modal query.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum QueryPart {
    /// A text fragment.
    Text(String),
    /// An image attachment.
    Image(ImageRef),
    /// An audio attachment.
    Audio(AudioRef),
    /// A video attachment.
    Video(VideoRef),
}

impl QueryPart {
    /// The [`Modality`] this part represents.
    pub fn modality(&self) -> Modality {
        match self {
            QueryPart::Text(_) => Modality::Text,
            QueryPart::Image(_) => Modality::Image,
            QueryPart::Audio(_) => Modality::Audio,
            QueryPart::Video(_) => Modality::Video,
        }
    }
}

/// A multi-modal query — an ordered list of parts plus metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalQuery {
    /// Content-addressed identifier (derived from the joined textual parts).
    pub id: QueryId,
    /// Ordered query parts.
    pub parts: Vec<QueryPart>,
    /// Free-form metadata (tenant, request-id, model hint, etc.).
    pub metadata: BTreeMap<String, String>,
}

impl MultimodalQuery {
    /// Build a query from an ordered list of parts. The id is derived from a
    /// canonical serialisation of the textual parts so cache hits work
    /// deterministically across runs.
    pub fn new(parts: Vec<QueryPart>) -> Self {
        let textual = parts
            .iter()
            .filter_map(|p| match p {
                QueryPart::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        Self {
            id: QueryId::from_prompt(&textual),
            parts,
            metadata: BTreeMap::new(),
        }
    }

    /// Convenience: a text-only multi-modal query.
    pub fn text(prompt: impl Into<String>) -> Self {
        Self::new(vec![QueryPart::Text(prompt.into())])
    }

    /// Attach a metadata pair.
    pub fn with_meta(mut self, key: impl Into<String>, value: impl Into<String>) -> Self {
        self.metadata.insert(key.into(), value.into());
        self
    }

    /// The set of modalities present in this query.
    pub fn modalities(&self) -> Vec<Modality> {
        let mut seen: Vec<Modality> = Vec::with_capacity(4);
        for p in &self.parts {
            let m = p.modality();
            if !seen.contains(&m) {
                seen.push(m);
            }
        }
        seen
    }

    /// Return the concatenated text of all [`QueryPart::Text`] parts.
    pub fn text_prompt(&self) -> String {
        self.parts
            .iter()
            .filter_map(|p| match p {
                QueryPart::Text(t) => Some(t.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n")
    }
}

impl ImageRef {
    /// Resolve this reference to raw bytes + content-type.
    ///
    /// `Url` variants are *not* fetched here — vendors that accept URLs
    /// directly should be given the URL; vendors that require bytes should
    /// fetch via [`fetch_url`].
    pub fn to_bytes(&self) -> MultimodalResult<(String, Vec<u8>)> {
        match self {
            ImageRef::Url(_) => Err(MultimodalError::Decode(
                "ImageRef::Url must be fetched explicitly via fetch_url()".to_string(),
            )),
            ImageRef::Bytes { content_type, bytes } => {
                Ok((content_type.clone(), bytes.clone()))
            }
            ImageRef::Base64(s) => {
                let bytes = B64.decode(s.as_bytes())?;
                let ct = mime_guess::from_path("image.bin")
                    .first()
                    .map(|m| m.to_string())
                    .unwrap_or_else(|| "application/octet-stream".to_string());
                Ok((ct, bytes))
            }
            ImageRef::File(path) => {
                let bytes = std::fs::read(path)?;
                let ct = mime_guess::from_path(path)
                    .first()
                    .map(|m| m.to_string())
                    .unwrap_or_else(|| "image/png".to_string());
                Ok((ct, bytes))
            }
        }
    }

    /// Encode as base64 — useful for vendors that take `image_base64` /
    /// inline `data` fields.
    pub fn to_base64(&self) -> MultimodalResult<(String, String)> {
        if let ImageRef::Base64(s) = self {
            // Best-effort content-type when we only have the b64 string.
            return Ok(("image/png".to_string(), s.clone()));
        }
        let (ct, bytes) = self.to_bytes()?;
        Ok((ct, B64.encode(bytes)))
    }
}

impl AudioRef {
    /// Resolve to raw bytes + content-type.
    pub fn to_bytes(&self) -> MultimodalResult<(String, Vec<u8>)> {
        match self {
            AudioRef::Url(_) => Err(MultimodalError::Decode(
                "AudioRef::Url must be fetched explicitly".to_string(),
            )),
            AudioRef::Bytes { content_type, bytes } => Ok((content_type.clone(), bytes.clone())),
            AudioRef::Base64(s) => {
                let bytes = B64.decode(s.as_bytes())?;
                Ok(("audio/wav".to_string(), bytes))
            }
            AudioRef::File(path) => {
                let bytes = std::fs::read(path)?;
                let ct = mime_guess::from_path(path)
                    .first()
                    .map(|m| m.to_string())
                    .unwrap_or_else(|| "audio/wav".to_string());
                Ok((ct, bytes))
            }
        }
    }
}

impl VideoRef {
    /// Resolve to raw bytes + content-type.
    pub fn to_bytes(&self) -> MultimodalResult<(String, Vec<u8>)> {
        match self {
            VideoRef::Url(_) => Err(MultimodalError::Decode(
                "VideoRef::Url must be fetched explicitly".to_string(),
            )),
            VideoRef::Bytes { content_type, bytes } => Ok((content_type.clone(), bytes.clone())),
            VideoRef::Base64(s) => {
                let bytes = B64.decode(s.as_bytes())?;
                Ok(("video/mp4".to_string(), bytes))
            }
            VideoRef::File(path) => {
                let bytes = std::fs::read(path)?;
                let ct = mime_guess::from_path(path)
                    .first()
                    .map(|m| m.to_string())
                    .unwrap_or_else(|| "video/mp4".to_string());
                Ok((ct, bytes))
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn modalities_dedup_in_order() {
        let q = MultimodalQuery::new(vec![
            QueryPart::Text("a".to_string()),
            QueryPart::Image(ImageRef::Url("u".to_string())),
            QueryPart::Text("b".to_string()),
            QueryPart::Audio(AudioRef::Url("u".to_string())),
        ]);
        assert_eq!(
            q.modalities(),
            vec![Modality::Text, Modality::Image, Modality::Audio]
        );
        assert_eq!(q.text_prompt(), "a\nb");
    }

    #[test]
    fn text_constructor_round_trip() {
        let q = MultimodalQuery::text("hello world");
        assert_eq!(q.modalities(), vec![Modality::Text]);
        assert_eq!(q.text_prompt(), "hello world");
        // Id should match the eoc-core derivation directly.
        assert_eq!(q.id, QueryId::from_prompt("hello world"));
    }

    #[test]
    fn base64_round_trip_image() {
        let img = ImageRef::Bytes {
            content_type: "image/png".to_string(),
            bytes: vec![0xDE, 0xAD, 0xBE, 0xEF],
        };
        let (ct, b64) = img.to_base64().expect("encode");
        assert_eq!(ct, "image/png");
        // 4 bytes b64-encoded → 8 chars (with padding) "3q2+7w==".
        assert_eq!(b64, "3q2+7w==");
    }
}
