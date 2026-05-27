//! [`jouleclaw_cascade::Tier`] implementation for the L1.5 SSM reader.
//!
//! The tier owns a [`ReadingComprehender`] (the deterministic
//! [`ExtractiveReader`] by default) and consumes a structured envelope
//! containing a `question` and a slice of `passages`. On hit it returns
//! the reader's verbatim answer as [`AnswerOutput::Text`]; below the
//! confidence floor it refuses with [`RefusalReason::LowConfidence`].

use std::time::Duration;

use jouleclaw_cascade::tier::{Tier, TierEstimate};
use jouleclaw_cascade::types::{
    Answer, AnswerError, AnswerOutput, ExecutionTrace, Query, QueryInput, RefusalReason, TierId,
};
use jouleclaw_cascade::verification::VerificationStatus;

use crate::reader::{ExtractiveReader, Passage, ReadingComprehender};

/// Flat dispatch energy. ~20 mJ — class-typical for a small local SSM
/// (Mamba-3 ~1B params at int8) reading 3–5 short passages. Higher than
/// the L0.75 router's ~100 µJ because the tier does full reading, not
/// just classification. The default [`ExtractiveReader`] is far cheaper
/// (zero energy, deterministic CPU), but we report the SSM-class figure
/// so production swaps stay honest to the cascade's calibration ledger.
pub const SSM_READER_JOULES: f64 = 20_000e-6;

/// Flat dispatch latency. ~10 ms.
pub const SSM_READER_LATENCY: Duration = Duration::from_millis(10);

/// Confidence floor reported by `estimate_cost`. Tier-supplied answers
/// below this floor refuse with [`RefusalReason::LowConfidence`] so the
/// runtime continues to L2 federation.
pub const SSM_READER_CONFIDENCE_FLOOR: f32 = 0.7;

/// Errors surfaced by the SSM-reader tier. The envelope-parsing errors
/// are kept as a typed enum so downstream consumers can wrap them.
#[derive(Debug, thiserror::Error)]
pub enum SsmReaderError {
    /// The structured envelope did not parse as JSON.
    #[error("envelope parse failed: {0}")]
    InvalidEnvelope(String),

    /// The envelope parsed but did not contain the required fields.
    #[error("envelope missing field: {0}")]
    MissingField(&'static str),

    /// The reader's backend failed.
    #[error("reader backend failed: {0}")]
    Reader(String),
}

/// The L1.5 SSM-reader tier.
///
/// Construct with [`SsmReaderTier::new`] for the deterministic
/// [`ExtractiveReader`] default, or [`SsmReaderTier::with_reader`] to
/// plug in a Mamba-3 / Liquid backend.
pub struct SsmReaderTier {
    reader: Box<dyn ReadingComprehender>,
}

impl SsmReaderTier {
    /// Construct a tier with the deterministic v0.1 [`ExtractiveReader`].
    pub fn new() -> Self {
        Self {
            reader: Box::new(ExtractiveReader::new()),
        }
    }

    /// Construct a tier with a caller-supplied [`ReadingComprehender`].
    pub fn with_reader(reader: Box<dyn ReadingComprehender>) -> Self {
        Self { reader }
    }

    /// Borrow the reader (for diagnostics — e.g. `name()`).
    pub fn reader(&self) -> &dyn ReadingComprehender {
        &*self.reader
    }
}

impl Default for SsmReaderTier {
    fn default() -> Self {
        Self::new()
    }
}

impl Tier for SsmReaderTier {
    fn id(&self) -> TierId {
        TierId::L1_5SsmReader
    }

    fn estimate_cost(&self, q: &Query) -> Option<TierEstimate> {
        // The L1.5 tier is only applicable when a structured envelope
        // carries both a non-empty `question` and at least one passage.
        // Anything else (raw text, binary, image, audio, multimodal) is
        // out-of-scope; the runtime moves on cleanly.
        match &q.input {
            QueryInput::Structured(bytes) => match parse_envelope(bytes) {
                Ok((question, passages))
                    if !question.trim().is_empty() && !passages.is_empty() =>
                {
                    Some(TierEstimate {
                        joules: SSM_READER_JOULES,
                        latency: SSM_READER_LATENCY,
                        confidence_floor: SSM_READER_CONFIDENCE_FLOOR,
                    })
                }
                _ => None,
            },
            _ => None,
        }
    }

    fn try_answer(
        &mut self,
        q: &Query,
        _budget_remaining: f64,
    ) -> Result<Answer, AnswerError> {
        let bytes = match &q.input {
            QueryInput::Structured(b) => b.as_slice(),
            _ => return Ok(refused(RefusalReason::Inapplicable, 0.0)),
        };

        let (question, passages) =
            parse_envelope(bytes).map_err(|e| AnswerError::TierFailed {
                tier: TierId::L1_5SsmReader,
                cause: format!("envelope: {e}"),
            })?;

        if question.trim().is_empty() || passages.is_empty() {
            return Ok(refused(RefusalReason::Inapplicable, 0.0));
        }

        let reading =
            self.reader
                .read_and_answer(&question, &passages)
                .map_err(|e| AnswerError::TierFailed {
                    tier: TierId::L1_5SsmReader,
                    cause: format!("reader: {e}"),
                })?;

        // Below the confidence floor: refuse with LowConfidence so the
        // runtime continues to L2. The donor always returned the answer
        // text even below threshold (for speculative drafting); the
        // JouleClaw cascade enforces quality floors via Refused.
        if reading.confidence < SSM_READER_CONFIDENCE_FLOOR {
            return Ok(Answer {
                output: AnswerOutput::Refused(RefusalReason::low_confidence(reading.confidence)),
                tier_used: TierId::L1_5SsmReader,
                joules_spent: SSM_READER_JOULES,
                confidence: reading.confidence,
                trace: ExecutionTrace::default(),
                verification: VerificationStatus::Resolved,
            });
        }

        Ok(Answer {
            output: AnswerOutput::Text(reading.answer),
            tier_used: TierId::L1_5SsmReader,
            joules_spent: SSM_READER_JOULES,
            confidence: reading.confidence,
            trace: ExecutionTrace::default(),
            verification: VerificationStatus::Resolved,
        })
    }
}

fn refused(reason: RefusalReason, joules: f64) -> Answer {
    Answer {
        output: AnswerOutput::Refused(reason),
        tier_used: TierId::L1_5SsmReader,
        joules_spent: joules,
        confidence: 0.0,
        trace: ExecutionTrace::default(),
        verification: VerificationStatus::Resolved,
    }
}

/// Parse the JSON envelope that wraps L1.5 inputs into a question +
/// passage list. Exposed so callers building queries by hand can share
/// the same parse path with the tier.
///
/// Envelope shape:
///
/// ```json
/// {
///   "question": "...",
///   "passages": [
///     { "text": "...", "source": "..." },
///     { "text": "..." }
///   ]
/// }
/// ```
///
/// `source` is optional per passage. The bytes MUST be UTF-8 JSON.
pub fn parse_envelope(bytes: &[u8]) -> Result<(String, Vec<Passage>), SsmReaderError> {
    let value: serde_json::Value = serde_json::from_slice(bytes)
        .map_err(|e| SsmReaderError::InvalidEnvelope(e.to_string()))?;

    let question = value
        .get("question")
        .and_then(|v| v.as_str())
        .ok_or(SsmReaderError::MissingField("question"))?
        .to_string();

    let raw_passages = value
        .get("passages")
        .and_then(|v| v.as_array())
        .ok_or(SsmReaderError::MissingField("passages"))?;

    let mut passages = Vec::with_capacity(raw_passages.len());
    for p in raw_passages {
        let text = p
            .get("text")
            .and_then(|v| v.as_str())
            .ok_or(SsmReaderError::MissingField("passages[].text"))?
            .to_string();
        let source = p
            .get("source")
            .and_then(|v| v.as_str())
            .map(|s| s.to_string());
        passages.push(Passage { text, source });
    }

    Ok((question, passages))
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_cascade::types::{ContextRef, JouleBudget, QualityFloor, Query, QueryInput};
    use serde_json::json;

    use crate::reader::{Reading, ReaderError};

    fn envelope(question: &str, passages: &[(&str, Option<&str>)]) -> Vec<u8> {
        let v = json!({
            "question": question,
            "passages": passages
                .iter()
                .map(|(t, s)| match s {
                    Some(src) => json!({"text": t, "source": src}),
                    None => json!({"text": t}),
                })
                .collect::<Vec<_>>(),
        });
        serde_json::to_vec(&v).expect("envelope ser")
    }

    fn structured_query(bytes: Vec<u8>) -> Query {
        Query {
            input: QueryInput::Structured(bytes),
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        }
    }

    #[test]
    fn id_is_l1_5_ssm_reader() {
        let t = SsmReaderTier::new();
        assert_eq!(t.id(), TierId::L1_5SsmReader);
        assert_eq!(t.id().wire_tag(), "L1.5");
        assert_eq!(t.id().name(), "SsmReader");
    }

    #[test]
    fn estimate_with_question_and_passages_returns_some() {
        let t = SsmReaderTier::new();
        let bytes = envelope(
            "What is the capital of France?",
            &[("Paris is the capital of France.", None)],
        );
        let q = structured_query(bytes);
        let est = t.estimate_cost(&q).expect("envelope → estimate");
        assert_eq!(est.joules, SSM_READER_JOULES);
        assert_eq!(est.latency, SSM_READER_LATENCY);
        assert!((est.confidence_floor - SSM_READER_CONFIDENCE_FLOOR).abs() < f32::EPSILON);
    }

    #[test]
    fn estimate_empty_passages_returns_none() {
        let t = SsmReaderTier::new();
        let bytes = envelope("What is the capital of France?", &[]);
        let q = structured_query(bytes);
        assert!(t.estimate_cost(&q).is_none());
    }

    #[test]
    fn estimate_empty_question_returns_none() {
        let t = SsmReaderTier::new();
        let bytes = envelope("   ", &[("Paris is the capital of France.", None)]);
        let q = structured_query(bytes);
        assert!(t.estimate_cost(&q).is_none());
    }

    #[test]
    fn estimate_text_input_returns_none() {
        // L1.5 only takes structured envelopes — plain text is for the
        // L0.75 router, not the reader.
        let t = SsmReaderTier::new();
        let q = Query {
            input: QueryInput::Text("hello".to_string()),
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        assert!(t.estimate_cost(&q).is_none());
    }

    #[test]
    fn estimate_non_text_inputs_return_none() {
        let t = SsmReaderTier::new();
        for input in [
            QueryInput::Binary(vec![1, 2, 3]),
            QueryInput::Image(vec![0xff, 0xd8]),
            QueryInput::Audio(vec![0x52, 0x49]),
        ] {
            let q = Query {
                input,
                budget: JouleBudget::cheap(),
                quality: QualityFloor::any(),
                context: ContextRef::fresh(),
                deadline: None,
            };
            assert!(t.estimate_cost(&q).is_none());
        }
    }

    #[test]
    fn try_answer_extractive_default_reads_passages() {
        let mut t = SsmReaderTier::new();
        let bytes = envelope(
            "What is the capital of France?",
            &[
                ("Berlin is in Germany.", Some("wp:Berlin")),
                (
                    "Paris is the capital of France. France borders Spain.",
                    Some("wp:Paris"),
                ),
            ],
        );
        let q = structured_query(bytes);
        let ans = t.try_answer(&q, 1.0).expect("reads");
        match ans.output {
            AnswerOutput::Text(text) => {
                assert!(text.contains("Paris"), "got {text}");
            }
            other => panic!("expected Text, got {other:?}"),
        }
        assert_eq!(ans.tier_used, TierId::L1_5SsmReader);
        assert_eq!(ans.joules_spent, SSM_READER_JOULES);
        assert!(ans.confidence >= SSM_READER_CONFIDENCE_FLOOR);
    }

    #[test]
    fn try_answer_low_confidence_refuses() {
        let mut t = SsmReaderTier::new();
        // Question shares only one stop-word-free token ("blue") with
        // the passage; the extractive reader's confidence will sit
        // below the 0.7 floor.
        let bytes = envelope(
            "What is the velocity of an unladen swallow?",
            &[("Birds fly through blue skies.", None)],
        );
        let q = structured_query(bytes);
        let ans = t.try_answer(&q, 1.0).expect("refuses cleanly");
        assert!(matches!(
            ans.output,
            AnswerOutput::Refused(RefusalReason::LowConfidence(_))
        ));
        assert!(ans.confidence < SSM_READER_CONFIDENCE_FLOOR);
        assert_eq!(ans.tier_used, TierId::L1_5SsmReader);
        assert_eq!(ans.joules_spent, SSM_READER_JOULES);
    }

    #[test]
    fn try_answer_non_structured_refuses_inapplicable() {
        let mut t = SsmReaderTier::new();
        let q = Query {
            input: QueryInput::Text("plain text".to_string()),
            budget: JouleBudget::cheap(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        let ans = t.try_answer(&q, 1.0).expect("refuses");
        assert!(matches!(
            ans.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable)
        ));
        assert_eq!(ans.confidence, 0.0);
    }

    #[test]
    fn try_answer_empty_passages_refuses_inapplicable() {
        let mut t = SsmReaderTier::new();
        let bytes = envelope("question?", &[]);
        let q = structured_query(bytes);
        let ans = t.try_answer(&q, 1.0).expect("refuses");
        assert!(matches!(
            ans.output,
            AnswerOutput::Refused(RefusalReason::Inapplicable)
        ));
    }

    #[test]
    fn try_answer_malformed_envelope_fails() {
        let mut t = SsmReaderTier::new();
        let q = structured_query(b"not json".to_vec());
        let err = t.try_answer(&q, 1.0).expect_err("invalid json");
        match err {
            AnswerError::TierFailed { tier, cause } => {
                assert_eq!(tier, TierId::L1_5SsmReader);
                assert!(cause.contains("envelope"));
            }
            other => panic!("expected TierFailed, got {other:?}"),
        }
    }

    #[test]
    fn parse_envelope_round_trip() {
        let bytes = envelope(
            "what?",
            &[("a.", Some("src-a")), ("b.", None)],
        );
        let (question, passages) = parse_envelope(&bytes).expect("parse");
        assert_eq!(question, "what?");
        assert_eq!(passages.len(), 2);
        assert_eq!(passages[0].text, "a.");
        assert_eq!(passages[0].source.as_deref(), Some("src-a"));
        assert_eq!(passages[1].text, "b.");
        assert!(passages[1].source.is_none());
    }

    #[test]
    fn parse_envelope_missing_question_errors() {
        let bytes = serde_json::to_vec(&json!({"passages": []})).expect("ser");
        match parse_envelope(&bytes) {
            Err(SsmReaderError::MissingField("question")) => {}
            other => panic!("expected MissingField(question), got {other:?}"),
        }
    }

    #[test]
    fn parse_envelope_missing_passages_errors() {
        let bytes = serde_json::to_vec(&json!({"question": "what?"})).expect("ser");
        match parse_envelope(&bytes) {
            Err(SsmReaderError::MissingField("passages")) => {}
            other => panic!("expected MissingField(passages), got {other:?}"),
        }
    }

    // ── End-to-end via cascade runtime ──────────────────────────────

    #[test]
    fn end_to_end_via_cascade_runtime() {
        use jouleclaw_cascade::tier::{Cascade, Runtime};

        let mut cascade = Cascade::new();
        cascade.register(Box::new(SsmReaderTier::new()));
        let mut rt = Runtime::new_without_l0(cascade);

        let bytes = envelope(
            "What is the capital of France?",
            &[(
                "Paris is the capital of France. France borders Spain.",
                Some("wp:Paris"),
            )],
        );
        // L1.5 is class-typical ~20 mJ; the cheap() budget caps at 1 mJ
        // so we use standard() (1 J hard) here. This documents that
        // L1.5 is *not* a cheap-budget tier.
        let q = Query {
            input: QueryInput::Structured(bytes),
            budget: JouleBudget::standard(),
            quality: QualityFloor::any(),
            context: ContextRef::fresh(),
            deadline: None,
        };
        let ans = rt.answer(q).expect("runtime answer");
        assert_eq!(ans.tier_used, TierId::L1_5SsmReader);
        match ans.output {
            AnswerOutput::Text(text) => assert!(text.contains("Paris"), "got {text}"),
            other => panic!("expected Text, got {other:?}"),
        }
    }

    // ── Custom-reader swap ──────────────────────────────────────────

    /// Test-only deterministic reader that always answers `"42"` at high
    /// confidence.
    struct AlwaysFortyTwo;
    impl ReadingComprehender for AlwaysFortyTwo {
        fn read_and_answer(
            &self,
            _question: &str,
            _passages: &[Passage],
        ) -> Result<Reading, ReaderError> {
            Ok(Reading {
                answer: "42".to_string(),
                confidence: 0.99,
                used_passages: vec![0],
            })
        }
        fn name(&self) -> &'static str {
            "always-42"
        }
    }

    #[test]
    fn with_reader_swaps_backend() {
        let mut t = SsmReaderTier::with_reader(Box::new(AlwaysFortyTwo));
        assert_eq!(t.reader().name(), "always-42");
        let bytes = envelope("anything?", &[("filler.", None)]);
        let q = structured_query(bytes);
        let ans = t.try_answer(&q, 1.0).expect("reads");
        match ans.output {
            AnswerOutput::Text(text) => assert_eq!(text, "42"),
            other => panic!("expected Text, got {other:?}"),
        }
        assert!((ans.confidence - 0.99).abs() < 1e-6);
    }

    /// Test-only reader that always fails at the backend boundary.
    struct AlwaysBackendError;
    impl ReadingComprehender for AlwaysBackendError {
        fn read_and_answer(
            &self,
            _question: &str,
            _passages: &[Passage],
        ) -> Result<Reading, ReaderError> {
            Err(ReaderError::Backend("simulated".into()))
        }
    }

    #[test]
    fn reader_backend_error_surfaces_as_tier_failed() {
        let mut t = SsmReaderTier::with_reader(Box::new(AlwaysBackendError));
        let bytes = envelope("question?", &[("filler.", None)]);
        let q = structured_query(bytes);
        let err = t.try_answer(&q, 1.0).expect_err("backend fails");
        match err {
            AnswerError::TierFailed { tier, cause } => {
                assert_eq!(tier, TierId::L1_5SsmReader);
                assert!(cause.contains("reader"));
                assert!(cause.contains("simulated"));
            }
            other => panic!("expected TierFailed, got {other:?}"),
        }
    }

    #[test]
    fn reader_default_is_extractive_v0() {
        let t = SsmReaderTier::new();
        assert_eq!(t.reader().name(), "extractive-v0");
    }
}
