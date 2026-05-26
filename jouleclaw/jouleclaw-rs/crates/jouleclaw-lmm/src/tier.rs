//! `LmmTier` — joule cascade tier shell for vision-language models.
//!
//! R31.0: cascade integration point + preprocessing. The tier inspects
//! `QueryInput` to decide whether it can attempt a query — accepting
//! `Multimodal`, `Image`, `Audio`, and pure `Text`. Refuses everything
//! with a structured reason until R31.1 wires the model.

use std::time::Duration;

use jouleclaw_cascade::*;
use jouleclaw_prism::TernaryDecoder;

use crate::vlm;

/// Cascade tier for vision-language models. R31.0 declared dims;
/// R31.1.0 wires an actual forward pass via a `prism::TernaryDecoder`
/// shared text backbone. With a decoder loaded, multimodal/image/text
/// queries all generate Text answers (random-byte continuations until
/// trained weights land in R31.1.2).
pub struct LmmTier {
    pub model_id: u32,
    /// Number of vision tokens the model produces per image (e.g. 256 for a
    /// 14-patch grid). Used for cost estimation; 0 means "no image support".
    pub vision_tokens_per_image: usize,
    /// Number of decoder layers — drives the per-token forward cost.
    pub num_layers: usize,
    /// Hidden dim. ~1024 for an LFM2-VL-450M, ~2560 for the 1.6B class.
    pub hidden_dim: usize,
    /// Optional text backbone (R31.1.0). When present, `try_answer`
    /// runs the multimodal forward pass and returns Text.
    pub decoder: Option<TernaryDecoder>,
    /// Optional real LFM2-VL backbone (R31.2): SigLIP ViT + LFM2
    /// projector + LFM2 text stack from real GGUF weights. When set,
    /// `try_answer` routes image/multimodal queries through
    /// `LfmVl::generate` instead of the byte-hash stand-in.
    pub vl: Option<crate::vl_real::LfmVl>,
    /// New tokens generated beyond the prompt sequence.
    pub max_new_tokens: usize,
}

impl LmmTier {
    pub fn empty(model_id: u32) -> Self {
        Self {
            model_id,
            vision_tokens_per_image: 0,
            num_layers: 0,
            hidden_dim: 0,
            decoder: None,
            vl: None,
            max_new_tokens: 16,
        }
    }

    /// Load a real LFM2-VL backbone (SigLIP ViT + LFM2 projector +
    /// LFM2 text stack) from GGUF and wrap it as a cascade tier.
    /// Image / multimodal queries dispatch via `LfmVl::generate`;
    /// pure-text queries currently refuse (text-only LFM2 lives in
    /// the `prism` PrismTier).
    pub fn from_lfm_vl_gguf<P: AsRef<std::path::Path>, Q: AsRef<std::path::Path>>(
        model_id: u32,
        text_path: P,
        mmproj_path: Q,
    ) -> Result<Self, crate::vl_real::VlError> {
        let vl = crate::vl_real::LfmVl::from_gguf(text_path, mmproj_path)?;
        let hidden = vl.d_model();
        // LFM2.5-VL-450M: 16 layers, 64 image tokens per image (after
        // 2×2 pixel-unshuffle merge of 256 patches). The constants are
        // model-family conventions, not asserted from the GGUF
        // (the loader could expose them later).
        Ok(Self {
            model_id,
            vision_tokens_per_image: 64,
            num_layers: 16,
            hidden_dim: hidden,
            decoder: None,
            vl: Some(vl),
            max_new_tokens: 8,
        })
    }

    pub fn with_dims(
        mut self,
        vision_tokens_per_image: usize,
        num_layers: usize,
        hidden_dim: usize,
    ) -> Self {
        self.vision_tokens_per_image = vision_tokens_per_image;
        self.num_layers = num_layers;
        self.hidden_dim = hidden_dim;
        self
    }

    /// Wire a text-decoder backbone for the multimodal forward pass.
    /// Images and audio are tokenized via the byte-hash stand-ins in
    /// [`crate::vlm`]; real image decoders are R31.1.1.
    pub fn from_decoder(model_id: u32, decoder: TernaryDecoder) -> Self {
        let n_layers = decoder.n_layers;
        let hidden = decoder.d_model;
        Self {
            model_id,
            vision_tokens_per_image: vlm::VISION_TOKENS_PER_IMAGE,
            num_layers: n_layers,
            hidden_dim: hidden,
            decoder: Some(decoder),
            vl: None,
            max_new_tokens: 8,
        }
    }

    pub fn with_max_new_tokens(mut self, n: usize) -> Self {
        self.max_new_tokens = n;
        self
    }

    /// True if this tier should consider answering the given query.
    /// Pure-Text queries are also accepted (an LMM can answer text-only).
    fn can_attempt(input: &QueryInput) -> bool {
        matches!(
            input,
            QueryInput::Text(_)
                | QueryInput::Image(_)
                | QueryInput::Audio(_)
                | QueryInput::Multimodal { .. }
        )
    }

    /// Static cost estimate for a single forward token. Cost ~ layers ×
    /// hidden^2 × 10 pJ per FMA, plus a 100 nJ dispatch floor.
    fn cost_per_decoded_token(&self) -> f64 {
        const FMA_PJ: f64 = 10.0;
        const DISPATCH_FLOOR_NJ: f64 = 100.0;
        let ops = (self.num_layers as f64) * (self.hidden_dim as f64) * (self.hidden_dim as f64);
        ops * FMA_PJ * 1e-12 + DISPATCH_FLOOR_NJ * 1e-9
    }

    /// Cost of one forward pass over the input sequence (encode-only,
    /// before token generation). The sequence length is dominated by
    /// vision tokens for image-heavy queries.
    fn cost_for_input(&self, q: &QueryInput) -> f64 {
        let n_images = match q {
            QueryInput::Image(_) => 1,
            QueryInput::Multimodal { images, .. } => images.len(),
            _ => 0,
        };
        let text_tokens = match q {
            QueryInput::Text(s) => (s.len() / 4).max(1), // rough bytes/token
            QueryInput::Multimodal { text, .. } => (text.len() / 4).max(1),
            _ => 1,
        };
        let seq_len = (n_images * self.vision_tokens_per_image + text_tokens) as f64;
        seq_len * self.cost_per_decoded_token()
    }
}

impl Tier for LmmTier {
    fn id(&self) -> TierId {
        TierId::L3(L3ModelId(self.model_id))
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        if !Self::can_attempt(&q.input) {
            return None;
        }
        let base = self.cost_for_input(&q.input);
        Some(TierEstimate {
            joules: if base > 0.0 { base } else { 100e-9 },
            latency: Duration::from_millis(self.num_layers.max(1) as u64),
            // Placeholder — R31.1 will calibrate.
            confidence_floor: 0.5,
        })
    }

    fn try_answer(&mut self, q: &Query, _budget: f64) -> Result<Answer, AnswerError> {
        if !Self::can_attempt(&q.input) {
            return Ok(refused(self.id(), 0.0, RefusalReason::Inapplicable));
        }
        let cost = self.cost_for_input(&q.input);

        // R31.2 hot path: real LFM2-VL backbone is loaded → SigLIP
        // ViT + LFM2 projector + LFM2 text stack produce a real
        // caption. Image-bearing queries route here; pure-text falls
        // through to the synthetic decoder path or refuses.
        if let Some(vl) = &self.vl {
            let (image_bytes, prompt): (Option<&[u8]>, &str) = match &q.input {
                QueryInput::Image(b) => (Some(b.as_slice()), ""),
                QueryInput::Multimodal { text, images, .. } => {
                    (images.first().map(|v| v.as_slice()), text.as_str())
                }
                QueryInput::Text(_) => (None, ""),
                _ => return Ok(refused(self.id(), 0.0, RefusalReason::Inapplicable)),
            };
            if let Some(bytes) = image_bytes {
                match vl.generate(bytes, prompt, self.max_new_tokens) {
                    Ok((caption, joules)) => {
                        return Ok(Answer {
                            output: AnswerOutput::Text(caption),
                            tier_used: self.id(),
                            joules_spent: joules,
                            confidence: 0.5,
                            trace: hit_trace(self.id(), joules),
                            verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
                        });
                    }
                    Err(e) => {
                        let reason = RefusalReason::TierSpecific(format!(
                            "LfmVl::generate failed: {}", e));
                        return Ok(refused(self.id(), cost, reason));
                    }
                }
            }
            // Pure-text with a VL backbone is out of scope; refuse so
            // the router falls through to PrismTier (text-only LFM2).
            let reason = RefusalReason::TierSpecific(
                "LmmTier with LFM2-VL backbone only handles image-bearing \
                 queries; use PrismTier for text-only LFM2".into());
            return Ok(refused(self.id(), 0.0, reason));
        }

        // R31.1.0 hot path: backbone is loaded → generate Text.
        if let Some(decoder) = &self.decoder {
            let (text, images, audio): (&str, &[Vec<u8>], &[Vec<u8>]) = match &q.input {
                QueryInput::Text(s) => (s.as_str(), &[], &[]),
                QueryInput::Image(b) => ("", std::slice::from_ref(b), &[]),
                QueryInput::Audio(b) => ("", &[], std::slice::from_ref(b)),
                QueryInput::Multimodal { text, images, audio } => {
                    (text.as_str(), images.as_slice(), audio.as_slice())
                }
                _ => return Ok(refused(self.id(), 0.0, RefusalReason::Inapplicable)),
            };
            let out_text = vlm::generate_multimodal(
                decoder, text, images, audio, self.max_new_tokens,
            );
            return Ok(Answer {
                output: AnswerOutput::Text(out_text),
                tier_used: self.id(),
                joules_spent: cost,
                confidence: 0.5,
                trace: hit_trace(self.id(), cost),
                verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
            });
        }

        // R31.0 path: no backbone loaded.
        let modality = match &q.input {
            QueryInput::Text(_) => "text",
            QueryInput::Image(_) => "image",
            QueryInput::Audio(_) => "audio",
            QueryInput::Multimodal { .. } => "multimodal",
            _ => "unknown",
        };
        let reason = RefusalReason::TierSpecific(format!(
            "LmmTier received {} input; preprocessing + tier shell ready, \
             load a backbone via from_decoder to enable inference",
            modality
        ));
        Ok(refused(self.id(), cost, reason))
    }

    fn coord(&self) -> Option<jouleclaw_cascade::coord::Coord> {
        use jouleclaw_cascade::coord::{
            Coord, Encoding, Entity, Interface, NamedPrimitive, PrimitiveSet, Thermo,
            Verify, Zone,
        };
        Some(
            Coord::new(
                Zone::Z2_3,
                Entity::Reactive,
                Thermo::L1_Measure,
                Interface::Tokens,
                Verify::Statistical,
                Encoding::Facts,
            )
            .with_primitives(PrimitiveSet::of(&[
                NamedPrimitive::AttentionGrouped,
                NamedPrimitive::MlpForward,
                NamedPrimitive::Embed,
                NamedPrimitive::Sample,
            ])),
        )
    }
}

fn hit_trace(tier: TierId, joules: f64) -> ExecutionTrace {
    let mut t = ExecutionTrace::default();
    t.attempts.push(TraceEntry {
        tier,
        outcome: TraceOutcome::Hit,
        joules,
    });
    t
}

fn refused(tier: TierId, joules: f64, reason: RefusalReason) -> Answer {
    let mut trace = ExecutionTrace::default();
    trace.attempts.push(TraceEntry {
        tier,
        outcome: TraceOutcome::Refused(reason.clone()),
        joules,
    });
    Answer {
        output: AnswerOutput::Refused(reason),
        tier_used: tier,
        joules_spent: joules,
        confidence: 0.0,
        trace,
        verification: jouleclaw_cascade::verification::VerificationStatus::Resolved,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn q_text(s: &str) -> Query {
        Query {
            input: QueryInput::Text(s.to_string()),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    fn q_image(bytes: Vec<u8>) -> Query {
        Query {
            input: QueryInput::Image(bytes),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    fn q_multimodal(text: &str, images: usize) -> Query {
        Query {
            input: QueryInput::Multimodal {
                text: text.to_string(),
                images: vec![vec![0u8; 100]; images],
                audio: vec![],
            },
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    #[test]
    fn empty_tier_declares_tokens_interface_and_l3_id() {
        let tier = LmmTier::empty(11);
        assert_eq!(tier.id(), TierId::L3(L3ModelId(11)));
        let coord = tier.coord().expect("coord");
        assert!(matches!(coord.interface, jouleclaw_cascade::coord::Interface::Tokens));
        assert!(matches!(coord.zone, jouleclaw_cascade::coord::Zone::Z2_3));
        assert!(coord.primitives.named.iter().any(|p|
            matches!(p, jouleclaw_cascade::coord::NamedPrimitive::AttentionGrouped)
        ));
    }

    #[test]
    fn tier_accepts_text_image_audio_and_multimodal() {
        let tier = LmmTier::empty(0).with_dims(64, 8, 256);
        assert!(tier.estimate_cost(&q_text("hi")).is_some());
        assert!(tier.estimate_cost(&q_image(vec![0u8; 10])).is_some());
        assert!(tier.estimate_cost(&q_multimodal("describe", 2)).is_some());
        // Structured / Binary are not multimodal — refuse.
        let q_struct = Query {
            input: QueryInput::Structured(vec![0u8; 4]),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        assert!(tier.estimate_cost(&q_struct).is_none());
    }

    #[test]
    fn multimodal_cost_grows_with_image_count() {
        let tier = LmmTier::empty(0).with_dims(256, 12, 1024);
        let c1 = tier.estimate_cost(&q_multimodal("a", 1)).unwrap().joules;
        let c4 = tier.estimate_cost(&q_multimodal("a", 4)).unwrap().joules;
        assert!(c4 > c1 * 3.0, "4 images should be ~4× cost of 1: c1={} c4={}", c1, c4);
    }

    #[test]
    fn empty_tier_refuses_with_structured_reason() {
        let mut tier = LmmTier::empty(0).with_dims(256, 12, 1024);
        let ans = tier.try_answer(&q_multimodal("describe this", 1), 1.0).unwrap();
        match ans.output {
            AnswerOutput::Refused(RefusalReason::TierSpecific(msg)) => {
                assert!(
                    msg.contains("from_decoder"),
                    "refusal should point users at from_decoder: {}",
                    msg
                );
                assert!(msg.contains("multimodal"), "expected modality tag: {}", msg);
            }
            other => panic!("expected refusal, got {:?}", other),
        }
    }

    #[test]
    fn decoder_loaded_tier_hits_on_multimodal_query() {
        use jouleclaw_prism::{synthetic_model, ModelConfig};
        let decoder = synthetic_model(ModelConfig::tiny_byte(), 0xABCDEF).unwrap();
        let mut tier = LmmTier::from_decoder(99, decoder).with_max_new_tokens(4);
        let ans = tier.try_answer(&q_multimodal("describe", 1), 1.0).unwrap();
        match ans.output {
            AnswerOutput::Text(s) => {
                // Random weights → arbitrary continuation, but must be Text.
                let _ = s;
            }
            other => panic!("expected Text, got {:?}", other),
        }
        assert_eq!(ans.tier_used, TierId::L3(L3ModelId(99)));
        assert!(ans.joules_spent > 0.0);
    }

    #[test]
    fn decoder_loaded_tier_hits_on_image_only_query() {
        use jouleclaw_prism::{synthetic_model, ModelConfig};
        let decoder = synthetic_model(ModelConfig::tiny_byte(), 0xCAFE).unwrap();
        let mut tier = LmmTier::from_decoder(0, decoder).with_max_new_tokens(2);
        let ans = tier.try_answer(&q_image(vec![1u8, 2, 3, 4, 5, 6, 7, 8]), 1.0).unwrap();
        match ans.output {
            AnswerOutput::Text(_) => {}
            other => panic!("image query should produce Text, got {:?}", other),
        }
    }

    #[test]
    fn cascade_can_register_lmm_tier() {
        let mut cascade = Cascade::new();
        cascade.register(Box::new(LmmTier::empty(0).with_dims(64, 4, 128)));
        let mut rt = Runtime::new_without_l0(cascade);
        let _ = rt.answer(q_multimodal("what's in this image?", 1));
    }
}
