//! Modality routing.
//!
//! Routing is the bridge between the multi-modal surface and the
//! single-model [`NeuralBackend`] trait the cascade expects. Given a
//! [`MultimodalQuery`] the router inspects the set of [`Modality`]s
//! present and picks a backend:
//!
//! * **text only** → `text_backend`
//! * **text + image** → `vision_backend`
//! * **text + audio** → `audio_backend`
//! * **image + audio** (or anything richer, e.g. with video) → `unified_backend`
//!   if configured, otherwise `vision_backend`
//!
//! Each backend is just a [`NeuralBackend`]; the router synthesises a
//! text [`Query`] from the textual parts of the multi-modal query and
//! lets the backend's own implementation pick up image / audio data from
//! query metadata. Vendors that need richer in-band image / audio passing
//! provide their own `infer_multimodal()` method on the concrete backend
//! type — the router here is the simple "pick a backend" layer.

use eoc_core::{Query, Response};
use eoc_neural::NeuralBackend;

use crate::error::MultimodalResult;
use crate::modality::{Modality, MultimodalQuery};

/// Which backend the router picked.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RouteChoice {
    /// The text-only backend (default for `Modality::Text`).
    Text,
    /// The vision-language backend (`text + image`).
    Vision,
    /// The audio-language backend (`text + audio`).
    Audio,
    /// The unified backend (e.g. Gemini, GPT-4o) for queries with both
    /// image *and* audio, or anything richer (video).
    Unified,
}

/// Routes a [`MultimodalQuery`] to the modality-appropriate
/// [`NeuralBackend`] and returns a [`Response`].
pub struct ModalityRouter {
    /// Vision-language backend (image-bearing queries).
    pub vision_backend: Box<dyn NeuralBackend>,
    /// Audio-language backend (audio-bearing queries).
    pub audio_backend: Box<dyn NeuralBackend>,
    /// Text-only backend (fallback for text-only queries).
    pub text_backend: Box<dyn NeuralBackend>,
    /// Unified-modality backend (e.g. Gemini) for image+audio mixed queries.
    /// Falls back to [`Self::vision_backend`] when `None`.
    pub unified_backend: Option<Box<dyn NeuralBackend>>,
}

impl ModalityRouter {
    /// Construct with the three required backends. `unified_backend`
    /// defaults to `None` (routes fall back to vision).
    pub fn new(
        text_backend: Box<dyn NeuralBackend>,
        vision_backend: Box<dyn NeuralBackend>,
        audio_backend: Box<dyn NeuralBackend>,
    ) -> Self {
        Self {
            text_backend,
            vision_backend,
            audio_backend,
            unified_backend: None,
        }
    }

    /// Attach a unified-modality backend.
    pub fn with_unified(mut self, unified: Box<dyn NeuralBackend>) -> Self {
        self.unified_backend = Some(unified);
        self
    }

    /// Decide which backend handles this query.
    pub fn choose(&self, q: &MultimodalQuery) -> RouteChoice {
        let modalities = q.modalities();
        let has_image =
            modalities.contains(&Modality::Image) || modalities.contains(&Modality::Video);
        let has_audio = modalities.contains(&Modality::Audio);

        match (has_image, has_audio) {
            (true, true) => RouteChoice::Unified,
            (true, false) => RouteChoice::Vision,
            (false, true) => RouteChoice::Audio,
            (false, false) => RouteChoice::Text,
        }
    }

    /// Route the query and return a [`Response`]. Joule attribution and
    /// stage tagging come from whichever backend handled the call.
    pub async fn route(&self, q: &MultimodalQuery) -> MultimodalResult<Response> {
        let backend = match self.choose(q) {
            RouteChoice::Text => &self.text_backend,
            RouteChoice::Vision => &self.vision_backend,
            RouteChoice::Audio => &self.audio_backend,
            RouteChoice::Unified => self
                .unified_backend
                .as_ref()
                .unwrap_or(&self.vision_backend),
        };
        // Build a text-only `Query` that preserves the query id derived
        // from the multi-modal payload; backends that need image / audio
        // data should be invoked through their concrete API (e.g.
        // [`crate::vision::OpenAiVisionBackend::infer_multimodal`]). The
        // router here is the simple "which backend?" dispatcher.
        let mut tq = Query::new(q.text_prompt());
        tq.id = q.id;
        tq.metadata = q.metadata.clone();
        Ok(backend.infer(&tq).await)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use async_trait::async_trait;
    use eoc_core::{JouleCost, Stage};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicU8, Ordering};

    #[derive(Default)]
    struct LabelBackend {
        label: &'static str,
        hits: Arc<AtomicU8>,
    }

    #[async_trait]
    impl NeuralBackend for LabelBackend {
        async fn infer(&self, q: &Query) -> Response {
            self.hits.fetch_add(1, Ordering::SeqCst);
            Response::new(
                q.id,
                self.label.to_string(),
                Stage::Neural,
                JouleCost::estimated(1),
            )
        }
    }

    fn router(unified: bool) -> (ModalityRouter, Arc<AtomicU8>, Arc<AtomicU8>, Arc<AtomicU8>, Arc<AtomicU8>) {
        let (t, v, a, u) = (
            Arc::new(AtomicU8::new(0)),
            Arc::new(AtomicU8::new(0)),
            Arc::new(AtomicU8::new(0)),
            Arc::new(AtomicU8::new(0)),
        );
        let mut router = ModalityRouter::new(
            Box::new(LabelBackend { label: "text", hits: t.clone() }),
            Box::new(LabelBackend { label: "vision", hits: v.clone() }),
            Box::new(LabelBackend { label: "audio", hits: a.clone() }),
        );
        if unified {
            router = router.with_unified(Box::new(LabelBackend {
                label: "unified",
                hits: u.clone(),
            }));
        }
        (router, t, v, a, u)
    }

    #[tokio::test]
    async fn text_only_routes_to_text() {
        let (r, t, _, _, _) = router(false);
        let q = MultimodalQuery::text("hi");
        let resp = r.route(&q).await.expect("ok");
        assert_eq!(resp.payload, "text");
        assert_eq!(t.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn image_query_routes_to_vision() {
        let (r, _, v, _, _) = router(false);
        let q = MultimodalQuery::new(vec![
            crate::modality::QueryPart::Text("describe".to_string()),
            crate::modality::QueryPart::Image(crate::modality::ImageRef::Url(
                "u".to_string(),
            )),
        ]);
        let resp = r.route(&q).await.expect("ok");
        assert_eq!(resp.payload, "vision");
        assert_eq!(v.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn audio_query_routes_to_audio() {
        let (r, _, _, a, _) = router(false);
        let q = MultimodalQuery::new(vec![crate::modality::QueryPart::Audio(
            crate::modality::AudioRef::Url("u".to_string()),
        )]);
        let resp = r.route(&q).await.expect("ok");
        assert_eq!(resp.payload, "audio");
        assert_eq!(a.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn image_plus_audio_routes_to_unified_when_present() {
        let (r, _, _, _, u) = router(true);
        let q = MultimodalQuery::new(vec![
            crate::modality::QueryPart::Image(crate::modality::ImageRef::Url("u".to_string())),
            crate::modality::QueryPart::Audio(crate::modality::AudioRef::Url("u".to_string())),
        ]);
        let resp = r.route(&q).await.expect("ok");
        assert_eq!(resp.payload, "unified");
        assert_eq!(u.load(Ordering::SeqCst), 1);
    }

    #[tokio::test]
    async fn image_plus_audio_falls_back_to_vision_when_no_unified() {
        let (r, _, v, _, _) = router(false);
        let q = MultimodalQuery::new(vec![
            crate::modality::QueryPart::Image(crate::modality::ImageRef::Url("u".to_string())),
            crate::modality::QueryPart::Audio(crate::modality::AudioRef::Url("u".to_string())),
        ]);
        let resp = r.route(&q).await.expect("ok");
        assert_eq!(resp.payload, "vision");
        assert_eq!(v.load(Ordering::SeqCst), 1);
    }
}
