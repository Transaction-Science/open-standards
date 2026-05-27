//! [`jouleclaw_cascade::tier::Tier`] adapter that dispatches into an
//! [`LlmBackend`].
//!
//! The tier accepts text queries directly (`QueryInput::Text`) and a
//! richer structured shape (`QueryInput::Structured`) carrying canonical
//! JSON of the form:
//!
//! ```json
//! {
//!   "prompt":     "...",         // required, string
//!   "system":     "...",         // optional, string
//!   "max_tokens": 256,             // optional, number ≥ 1
//!   "stop":       ["</s>"],        // optional, array of strings
//!   "temperature": 0.3,            // optional, number
//!   "grammar":    "..."          // optional, string (opaque)
//! }
//! ```
//!
//! Other [`QueryInput`] variants (Binary, Image, Audio, Multimodal) are
//! out of scope here — a multimodal cheap-LLM tier would live in
//! `jouleclaw-lmm`.

use std::marker::PhantomData;
use std::time::Duration;

use jouleclaw_cascade::tier::{Tier, TierEstimate};
use jouleclaw_cascade::types::{
    Answer, AnswerError, AnswerOutput, ExecutionTrace, L3ModelId, Query, QueryInput,
    RefusalReason, TierId,
};
use jouleclaw_cascade::verification::VerificationStatus;
use jouleclaw_energy::Provenance;

use crate::backend::{FinishReason, LlmBackend, LlmError, LlmRequest, LlmResponse};

/// Default model id used when the caller does not supply one. The L3
/// surface is wire-stable at the coarse `L3` tag; the inner `L3ModelId(0)`
/// is a placeholder slot the runtime treats as "default backend".
pub const DEFAULT_MODEL_ID: L3ModelId = L3ModelId(0);

/// Default latency the tier reports in [`TierEstimate`]. Donor cites
/// ~3 s for the cheapest hosted LLMs; we surface that as the estimator's
/// best guess.
pub const DEFAULT_LATENCY: Duration = Duration::from_secs(3);

/// Default confidence floor reported by [`Tier::estimate_cost`]. Half —
/// the cheap LLM is the model class's least-trusted dispatch and the
/// runtime's quality gate should be allowed to skip it for high-floor
/// queries.
pub const DEFAULT_CONFIDENCE_FLOOR: f32 = 0.5;

/// Default `max_tokens` cap when the structured payload doesn't carry
/// one. Matches the donor's 512.
const DEFAULT_MAX_TOKENS: u32 = 512;

/// Default temperature for built-from-text requests. Matches the donor.
const DEFAULT_TEMPERATURE: f32 = 0.3;

/// The L3 cheap-LLM tier, parameterised by an [`LlmBackend`].
///
/// Construct with [`LlmCheapTier::new`] (default model id and latency) or
/// the builder-style [`LlmCheapTier::with_model_id`] /
/// [`LlmCheapTier::with_latency`] / [`LlmCheapTier::with_confidence_floor`]
/// methods.
pub struct LlmCheapTier<B: LlmBackend> {
    backend: B,
    model_id: L3ModelId,
    latency: Duration,
    confidence_floor: f32,
    /// Default token cap when the caller's structured payload doesn't
    /// supply one. Plain-text queries always use this.
    default_max_tokens: u32,
    /// Default sampling temperature when the structured payload doesn't
    /// supply one.
    default_temperature: f32,
    _marker: PhantomData<()>,
}

impl<B: LlmBackend> LlmCheapTier<B> {
    /// Construct a tier with default model id, latency, confidence floor,
    /// and token / temperature defaults.
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            model_id: DEFAULT_MODEL_ID,
            latency: DEFAULT_LATENCY,
            confidence_floor: DEFAULT_CONFIDENCE_FLOOR,
            default_max_tokens: DEFAULT_MAX_TOKENS,
            default_temperature: DEFAULT_TEMPERATURE,
            _marker: PhantomData,
        }
    }

    /// Override the inner `L3ModelId` slot.
    pub fn with_model_id(mut self, id: L3ModelId) -> Self {
        self.model_id = id;
        self
    }

    /// Override the latency reported in [`TierEstimate`].
    pub fn with_latency(mut self, latency: Duration) -> Self {
        self.latency = latency;
        self
    }

    /// Override the confidence floor reported in [`TierEstimate`].
    /// Clamped to `[0.0, 1.0]`.
    pub fn with_confidence_floor(mut self, floor: f32) -> Self {
        self.confidence_floor = floor.clamp(0.0, 1.0);
        self
    }

    /// Override the default `max_tokens` cap applied when the caller's
    /// structured payload doesn't carry one (and for all plain-text
    /// queries).
    pub fn with_default_max_tokens(mut self, max_tokens: u32) -> Self {
        self.default_max_tokens = max_tokens.max(1);
        self
    }

    /// Override the default sampling temperature applied when the
    /// structured payload doesn't carry one.
    pub fn with_default_temperature(mut self, temperature: f32) -> Self {
        self.default_temperature = temperature;
        self
    }

    /// Borrow the wrapped backend.
    pub fn backend(&self) -> &B {
        &self.backend
    }

    /// The [`jouleclaw_energy::Provenance`] this tier will tag a future
    /// answer with given a backend response. Exposed so callers can pre-
    /// flight provenance decisions without dispatching.
    ///
    /// * `Some(_)` energy from the backend → [`Provenance::ModelBased`]
    ///   (the backend has at least a vendor model — we cannot promise
    ///   HwShunt from this layer without inspecting the meter)
    /// * `None` energy from the backend → [`Provenance::Estimator`]
    pub fn report_provenance(response: &LlmResponse) -> Provenance {
        match response.energy_joules {
            Some(_) => Provenance::ModelBased,
            None => Provenance::Estimator,
        }
    }
}

impl<B: LlmBackend> Tier for LlmCheapTier<B> {
    fn id(&self) -> TierId {
        TierId::L3(self.model_id)
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        // Applicable to plain-text and structured payloads. Other variants
        // (Binary / Image / Audio / Multimodal) belong to other tiers.
        let applicable = match &q.input {
            QueryInput::Text(_) => true,
            QueryInput::Structured(bytes) => structured_has_prompt(bytes),
            _ => false,
        };
        if !applicable {
            return None;
        }
        Some(TierEstimate {
            joules: self.backend.typical_joules_per_call(),
            latency: self.latency,
            confidence_floor: self.confidence_floor,
        })
    }

    fn try_answer(
        &mut self,
        q: &Query,
        _budget_remaining: f64,
    ) -> Result<Answer, AnswerError> {
        // Build the request from the query.
        let request = match build_request(
            q,
            self.default_max_tokens,
            self.default_temperature,
        ) {
            Ok(r) => r,
            Err(BuildError::Inapplicable) => {
                return Ok(refused(
                    self.id(),
                    RefusalReason::Inapplicable,
                    0.0,
                ));
            }
            Err(BuildError::Malformed(msg)) => {
                return Ok(refused(
                    self.id(),
                    RefusalReason::TierSpecific(format!("malformed structured input: {msg}")),
                    0.0,
                ));
            }
        };

        // Dispatch.
        let typical = self.backend.typical_joules_per_call();
        match self.backend.complete(&request) {
            Ok(response) => {
                let joules = response.energy_joules.unwrap_or(typical);
                let (output, confidence) = match &response.finish_reason {
                    FinishReason::Stop => (
                        AnswerOutput::Text(response.text.clone()),
                        0.8_f32,
                    ),
                    FinishReason::Length => (
                        AnswerOutput::Text(response.text.clone()),
                        0.6_f32,
                    ),
                    FinishReason::ContentFilter => (
                        AnswerOutput::Refused(RefusalReason::TierSpecific(
                            "content filter".to_string(),
                        )),
                        0.0_f32,
                    ),
                    FinishReason::Error(msg) => (
                        AnswerOutput::Refused(RefusalReason::TierSpecific(format!(
                            "backend error: {msg}"
                        ))),
                        0.0_f32,
                    ),
                };
                Ok(Answer {
                    output,
                    tier_used: self.id(),
                    joules_spent: joules,
                    confidence,
                    trace: ExecutionTrace::default(),
                    verification: VerificationStatus::Resolved,
                })
            }
            Err(e) => {
                // Backend errors are reported as a refusal so the cascade
                // keeps walking. The donor mirrors this behaviour: an LLM
                // failure does not abort the cascade.
                Ok(refused(
                    self.id(),
                    RefusalReason::TierSpecific(format!("backend: {e}")),
                    typical,
                ))
            }
        }
    }
}

/// Error from request construction. Internal — callers see either
/// `Refused(Inapplicable)` or `Refused(TierSpecific("malformed …"))`.
enum BuildError {
    /// The query variant is not one we handle (Binary / Image / …).
    Inapplicable,
    /// The structured payload didn't parse as our expected shape.
    Malformed(String),
}

impl From<LlmError> for BuildError {
    fn from(e: LlmError) -> Self {
        Self::Malformed(e.to_string())
    }
}

/// Cheap structural check on a `QueryInput::Structured` payload — does
/// it have a non-empty `"prompt"` field? Used by `estimate_cost` so we
/// don't claim applicability for unrelated JSON that happens to land in
/// `Structured`.
fn structured_has_prompt(bytes: &[u8]) -> bool {
    let v: serde_json::Value = match serde_json::from_slice(bytes) {
        Ok(v) => v,
        Err(_) => return false,
    };
    matches!(v.get("prompt").and_then(|p| p.as_str()), Some(s) if !s.is_empty())
}

/// Build the request from the query, applying defaults for missing
/// structured fields.
fn build_request(
    q: &Query,
    default_max_tokens: u32,
    default_temperature: f32,
) -> Result<LlmRequest, BuildError> {
    match &q.input {
        QueryInput::Text(prompt) => {
            if prompt.is_empty() {
                return Err(BuildError::Inapplicable);
            }
            Ok(LlmRequest {
                prompt: prompt.clone(),
                max_tokens: default_max_tokens,
                temperature: default_temperature,
                stop: Vec::new(),
                system: None,
                grammar: None,
            })
        }
        QueryInput::Structured(bytes) => {
            let v: serde_json::Value = serde_json::from_slice(bytes)
                .map_err(|e| BuildError::Malformed(format!("json: {e}")))?;
            let prompt = v
                .get("prompt")
                .and_then(|p| p.as_str())
                .ok_or_else(|| BuildError::Malformed("missing 'prompt' (string)".into()))?;
            if prompt.is_empty() {
                return Err(BuildError::Malformed("'prompt' is empty".into()));
            }
            let max_tokens = v
                .get("max_tokens")
                .and_then(|m| m.as_u64())
                .map(|n| n.min(u32::MAX as u64) as u32)
                .unwrap_or(default_max_tokens)
                .max(1);
            let temperature = v
                .get("temperature")
                .and_then(|t| t.as_f64())
                .map(|f| f as f32)
                .unwrap_or(default_temperature);
            let stop = v
                .get("stop")
                .and_then(|s| s.as_array())
                .map(|arr| {
                    arr.iter()
                        .filter_map(|x| x.as_str().map(str::to_string))
                        .collect()
                })
                .unwrap_or_default();
            let system = v
                .get("system")
                .and_then(|s| s.as_str())
                .map(str::to_string);
            let grammar = v
                .get("grammar")
                .and_then(|g| g.as_str())
                .map(str::to_string);
            Ok(LlmRequest {
                prompt: prompt.to_string(),
                max_tokens,
                temperature,
                stop,
                system,
                grammar,
            })
        }
        _ => Err(BuildError::Inapplicable),
    }
}

/// Build a `Refused` Answer for this tier.
fn refused(tier: TierId, reason: RefusalReason, joules_spent: f64) -> Answer {
    Answer {
        output: AnswerOutput::Refused(reason),
        tier_used: tier,
        joules_spent,
        confidence: 0.0,
        trace: ExecutionTrace::default(),
        verification: VerificationStatus::Resolved,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::{EchoBackend, FinishReason, LlmRequest, LlmResponse};
    use jouleclaw_cascade::tier::{Cascade, Runtime};
    use jouleclaw_cascade::types::{
        ContextRef, JouleBudget, QualityFloor, Query, QueryInput,
    };

    // ── helpers ─────────────────────────────────────────────────────

    fn text_query(s: &str) -> Query {
        Query {
            input: QueryInput::Text(s.to_string()),
            // L3 is the model class; the donor cost is ~2 J/call, above
            // `standard()`'s 1 J hard limit, so we use `expensive()` (100 J)
            // for the test bench.
            budget: JouleBudget::expensive(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    fn structured_query(json: serde_json::Value) -> Query {
        let bytes = serde_json::to_vec(&json).expect("test json serialises");
        Query {
            input: QueryInput::Structured(bytes),
            budget: JouleBudget::expensive(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    /// Backend that returns a fixed response. Lets tests drive every
    /// FinishReason branch.
    struct FixedBackend {
        response: LlmResponse,
    }

    impl FixedBackend {
        fn new(response: LlmResponse) -> Self {
            Self { response }
        }
    }

    impl LlmBackend for FixedBackend {
        fn model_name(&self) -> &str {
            "fixed"
        }
        fn complete(&self, _req: &LlmRequest) -> Result<LlmResponse, LlmError> {
            Ok(self.response.clone())
        }
        fn typical_joules_per_call(&self) -> f64 {
            1.5
        }
    }

    /// Backend that always errors.
    struct ErrBackend;
    impl LlmBackend for ErrBackend {
        fn model_name(&self) -> &str {
            "err"
        }
        fn complete(&self, _req: &LlmRequest) -> Result<LlmResponse, LlmError> {
            Err(LlmError::Unavailable("offline".into()))
        }
        fn typical_joules_per_call(&self) -> f64 {
            2.5
        }
    }

    /// Backend that records the request it received.
    struct RecordingBackend {
        last: std::sync::Mutex<Option<LlmRequest>>,
    }
    impl RecordingBackend {
        fn new() -> Self {
            Self {
                last: std::sync::Mutex::new(None),
            }
        }
        fn last(&self) -> Option<LlmRequest> {
            self.last.lock().ok().and_then(|g| g.clone())
        }
    }
    impl LlmBackend for RecordingBackend {
        fn model_name(&self) -> &str {
            "recording"
        }
        fn complete(&self, req: &LlmRequest) -> Result<LlmResponse, LlmError> {
            if let Ok(mut g) = self.last.lock() {
                *g = Some(req.clone());
            }
            Ok(LlmResponse {
                text: "ok".into(),
                finish_reason: FinishReason::Stop,
                input_tokens: 0,
                output_tokens: 0,
                energy_joules: Some(0.123),
            })
        }
    }

    // ── id / estimate ───────────────────────────────────────────────

    #[test]
    fn id_is_l3_with_configured_model_id() {
        let t = LlmCheapTier::new(EchoBackend::new()).with_model_id(L3ModelId(7));
        assert_eq!(t.id(), TierId::L3(L3ModelId(7)));
        assert_eq!(t.id().wire_tag(), "L3");
        assert_eq!(t.id().name(), "Model");
    }

    #[test]
    fn estimate_text_returns_some() {
        let t = LlmCheapTier::new(EchoBackend::new());
        let q = text_query("what is rust?");
        let est = t.estimate_cost(&q).expect("text → estimate");
        assert!((est.joules - crate::DEFAULT_TYPICAL_JOULES).abs() < f64::EPSILON);
        assert_eq!(est.latency, DEFAULT_LATENCY);
        assert!((est.confidence_floor - DEFAULT_CONFIDENCE_FLOOR).abs() < f32::EPSILON);
    }

    #[test]
    fn estimate_structured_with_prompt_returns_some() {
        let t = LlmCheapTier::new(EchoBackend::new());
        let q = structured_query(serde_json::json!({"prompt": "hi"}));
        assert!(t.estimate_cost(&q).is_some());
    }

    #[test]
    fn estimate_structured_without_prompt_returns_none() {
        let t = LlmCheapTier::new(EchoBackend::new());
        let q = structured_query(serde_json::json!({"foo": "bar"}));
        assert!(t.estimate_cost(&q).is_none());
    }

    #[test]
    fn estimate_binary_returns_none() {
        let t = LlmCheapTier::new(EchoBackend::new());
        let q = Query {
            input: QueryInput::Binary(vec![1, 2, 3]),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        assert!(t.estimate_cost(&q).is_none());
    }

    // ── try_answer ──────────────────────────────────────────────────

    #[test]
    fn try_answer_text_stop_finish_reason() {
        let mut t = LlmCheapTier::new(EchoBackend::new());
        let q = text_query("rust?");
        let ans = t.try_answer(&q, 10.0).expect("echo never fails");
        match ans.output {
            AnswerOutput::Text(s) => assert_eq!(s, "echo: rust?"),
            other => panic!("expected Text, got {other:?}"),
        }
        // Stop → 0.8 confidence.
        assert!((ans.confidence - 0.8).abs() < f32::EPSILON);
        // EchoBackend reports no energy → fallback to typical.
        assert!((ans.joules_spent - crate::DEFAULT_TYPICAL_JOULES).abs() < f64::EPSILON);
    }

    #[test]
    fn try_answer_length_finish_reason_lowers_confidence() {
        let backend = FixedBackend::new(LlmResponse {
            text: "truncated".into(),
            finish_reason: FinishReason::Length,
            input_tokens: 0,
            output_tokens: 0,
            energy_joules: Some(0.9),
        });
        let mut t = LlmCheapTier::new(backend);
        let q = text_query("anything");
        let ans = t.try_answer(&q, 10.0).expect("answer");
        match ans.output {
            AnswerOutput::Text(s) => assert_eq!(s, "truncated"),
            other => panic!("expected Text, got {other:?}"),
        }
        assert!((ans.confidence - 0.6).abs() < f32::EPSILON);
        assert!((ans.joules_spent - 0.9).abs() < f64::EPSILON);
    }

    #[test]
    fn try_answer_content_filter_refuses() {
        let backend = FixedBackend::new(LlmResponse {
            text: String::new(),
            finish_reason: FinishReason::ContentFilter,
            input_tokens: 0,
            output_tokens: 0,
            energy_joules: Some(0.05),
        });
        let mut t = LlmCheapTier::new(backend);
        let q = text_query("anything");
        let ans = t.try_answer(&q, 10.0).expect("answer");
        match ans.output {
            AnswerOutput::Refused(RefusalReason::TierSpecific(s)) => {
                assert!(s.contains("content filter"));
            }
            other => panic!("expected Refused(TierSpecific(content filter)), got {other:?}"),
        }
        assert_eq!(ans.confidence, 0.0);
    }

    #[test]
    fn try_answer_error_finish_reason_refuses() {
        let backend = FixedBackend::new(LlmResponse {
            text: String::new(),
            finish_reason: FinishReason::Error("nan loss".into()),
            input_tokens: 0,
            output_tokens: 0,
            energy_joules: None,
        });
        let mut t = LlmCheapTier::new(backend);
        let q = text_query("anything");
        let ans = t.try_answer(&q, 10.0).expect("answer");
        match ans.output {
            AnswerOutput::Refused(RefusalReason::TierSpecific(s)) => {
                assert!(s.contains("backend error"));
                assert!(s.contains("nan loss"));
            }
            other => panic!("expected Refused with error message, got {other:?}"),
        }
    }

    #[test]
    fn try_answer_backend_unavailable_refuses_with_typical_cost() {
        let mut t = LlmCheapTier::new(ErrBackend);
        let q = text_query("hi");
        let ans = t.try_answer(&q, 10.0).expect("answer");
        match &ans.output {
            AnswerOutput::Refused(RefusalReason::TierSpecific(s)) => {
                assert!(s.contains("backend"));
                assert!(s.contains("offline"));
            }
            other => panic!("expected Refused(TierSpecific(backend …)), got {other:?}"),
        }
        // We charge the typical cost on a hard backend error, so the
        // cascade reflects the wasted joules.
        assert!((ans.joules_spent - 2.5).abs() < f64::EPSILON);
    }

    #[test]
    fn try_answer_empty_text_refuses_inapplicable() {
        let mut t = LlmCheapTier::new(EchoBackend::new());
        let q = text_query("");
        let ans = t.try_answer(&q, 10.0).expect("answer");
        assert!(matches!(
            ans.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable)
        ));
    }

    #[test]
    fn try_answer_non_text_refuses_inapplicable() {
        let mut t = LlmCheapTier::new(EchoBackend::new());
        let q = Query {
            input: QueryInput::Image(vec![0, 1, 2]),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        let ans = t.try_answer(&q, 10.0).expect("answer");
        assert!(matches!(
            ans.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable)
        ));
    }

    #[test]
    fn try_answer_structured_full_payload_propagates_fields() {
        let backend = RecordingBackend::new();
        let mut t = LlmCheapTier::new(backend);
        let q = structured_query(serde_json::json!({
            "prompt": "describe rust",
            "system": "be concise",
            "max_tokens": 128,
            "temperature": 0.7,
            "stop": ["</s>", "\n\n"],
            "grammar": "root ::= .*"
        }));
        let ans = t.try_answer(&q, 10.0).expect("answer");
        assert!(matches!(ans.output, AnswerOutput::Text(_)));
        let last = t.backend().last().expect("recorded");
        assert_eq!(last.prompt, "describe rust");
        assert_eq!(last.system.as_deref(), Some("be concise"));
        assert_eq!(last.max_tokens, 128);
        assert!((last.temperature - 0.7).abs() < f32::EPSILON);
        assert_eq!(last.stop, vec!["</s>".to_string(), "\n\n".to_string()]);
        assert_eq!(last.grammar.as_deref(), Some("root ::= .*"));
    }

    #[test]
    fn try_answer_structured_missing_prompt_refuses_malformed() {
        let mut t = LlmCheapTier::new(EchoBackend::new());
        let q = structured_query(serde_json::json!({"foo": "bar"}));
        let ans = t.try_answer(&q, 10.0).expect("answer");
        match ans.output {
            AnswerOutput::Refused(RefusalReason::TierSpecific(s)) => {
                assert!(s.contains("malformed structured input"));
            }
            other => panic!("expected malformed refusal, got {other:?}"),
        }
    }

    #[test]
    fn try_answer_structured_empty_prompt_refuses_malformed() {
        let mut t = LlmCheapTier::new(EchoBackend::new());
        let q = structured_query(serde_json::json!({"prompt": ""}));
        let ans = t.try_answer(&q, 10.0).expect("answer");
        assert!(matches!(
            ans.output,
            AnswerOutput::Refused(RefusalReason::TierSpecific(_))
        ));
    }

    // ── builder / overrides ─────────────────────────────────────────

    #[test]
    fn builder_overrides_apply() {
        let t = LlmCheapTier::new(EchoBackend::new())
            .with_model_id(L3ModelId(42))
            .with_latency(Duration::from_millis(500))
            .with_confidence_floor(0.9)
            .with_default_max_tokens(64)
            .with_default_temperature(0.0);
        assert_eq!(t.id(), TierId::L3(L3ModelId(42)));
        let est = t.estimate_cost(&text_query("hi")).expect("estimate");
        assert_eq!(est.latency, Duration::from_millis(500));
        assert!((est.confidence_floor - 0.9).abs() < f32::EPSILON);
    }

    #[test]
    fn confidence_floor_clamped_to_unit_range() {
        let t = LlmCheapTier::new(EchoBackend::new()).with_confidence_floor(5.0);
        let est = t.estimate_cost(&text_query("hi")).expect("estimate");
        assert!((est.confidence_floor - 1.0).abs() < f32::EPSILON);
    }

    #[test]
    fn report_provenance_model_based_when_energy_reported() {
        let resp = LlmResponse {
            text: "x".into(),
            finish_reason: FinishReason::Stop,
            input_tokens: 0,
            output_tokens: 0,
            energy_joules: Some(0.1),
        };
        assert_eq!(
            LlmCheapTier::<EchoBackend>::report_provenance(&resp),
            Provenance::ModelBased,
        );
    }

    #[test]
    fn report_provenance_estimator_when_no_energy() {
        let resp = LlmResponse {
            text: "x".into(),
            finish_reason: FinishReason::Stop,
            input_tokens: 0,
            output_tokens: 0,
            energy_joules: None,
        };
        assert_eq!(
            LlmCheapTier::<EchoBackend>::report_provenance(&resp),
            Provenance::Estimator,
        );
    }

    // ── end-to-end via Cascade + Runtime ────────────────────────────

    #[test]
    fn end_to_end_via_cascade_runtime_text() {
        let mut cascade = Cascade::new();
        cascade.register(Box::new(LlmCheapTier::new(EchoBackend::new())));
        let mut rt = Runtime::new_without_l0(cascade);

        let q = text_query("ping");
        let ans = rt.answer(q).expect("runtime answer");
        assert_eq!(ans.tier_used, TierId::L3(DEFAULT_MODEL_ID));
        match ans.output {
            AnswerOutput::Text(s) => assert_eq!(s, "echo: ping"),
            other => panic!("expected Text, got {other:?}"),
        }
        assert!((ans.confidence - 0.8).abs() < f32::EPSILON);
        assert!(ans.joules_spent > 0.0);
    }

    #[test]
    fn end_to_end_via_cascade_runtime_structured() {
        let mut cascade = Cascade::new();
        cascade.register(Box::new(LlmCheapTier::new(EchoBackend::new())));
        let mut rt = Runtime::new_without_l0(cascade);

        let q = structured_query(serde_json::json!({
            "prompt": "structured-ping",
            "max_tokens": 256,
        }));
        let ans = rt.answer(q).expect("runtime answer");
        match ans.output {
            AnswerOutput::Text(s) => assert_eq!(s, "echo: structured-ping"),
            other => panic!("expected Text, got {other:?}"),
        }
        assert_eq!(ans.tier_used, TierId::L3(DEFAULT_MODEL_ID));
    }
}
