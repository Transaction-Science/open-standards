//! Reading-comprehension trait and the deterministic v0.1 default
//! ([`ExtractiveReader`]).
//!
//! Production deployments plug in a Mamba-3 / Liquid SSM backend by
//! implementing [`ReadingComprehender`]. The default reader is a pure
//! deterministic extractive QA: it picks the sentence in the passage set
//! with the largest question-token overlap and returns it verbatim. Zero
//! energy, sub-microsecond, fully reproducible.

use serde::{Deserialize, Serialize};

// ─── Public types ─────────────────────────────────────────────────────

/// A retrieved passage handed to the reader.
///
/// The L1.5 tier consumes a slice of `Passage`s wrapped in a JSON
/// envelope; see the crate-level docs for the wire shape. `source` is
/// advisory metadata for receipts and is not used by the default
/// extractive reader.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Passage {
    /// The passage body.
    pub text: String,
    /// Optional opaque source tag (e.g., `"wp:Paris"`,
    /// `"hist:2026-05-27"`). Carried into receipts; not used for
    /// scoring.
    #[serde(default)]
    pub source: Option<String>,
}

impl Passage {
    /// Construct a passage with text only and no source tag.
    pub fn text<S: Into<String>>(text: S) -> Self {
        Self {
            text: text.into(),
            source: None,
        }
    }

    /// Construct a passage with text and source tag.
    pub fn with_source<S: Into<String>, T: Into<String>>(text: S, source: T) -> Self {
        Self {
            text: text.into(),
            source: Some(source.into()),
        }
    }
}

/// The structured reading result the SSM reader emits per query.
///
/// Returned both as the typed value to embedded callers (see
/// [`ReadingComprehender::read_and_answer`]) and surfaced as the tier's
/// `Answer.output` text. The fields are intentionally minimal: receipts
/// derive provenance from the surrounding tier metadata, not from this
/// struct.
#[derive(Debug, Clone, PartialEq)]
pub struct Reading {
    /// The extracted answer text.
    pub answer: String,
    /// Reader confidence in this answer, in `[0, 1]`.
    pub confidence: f32,
    /// Indices into the input `passages` slice that the reader actually
    /// used to produce the answer. Empty means no passage contributed
    /// (the reader fell through to a stock answer).
    pub used_passages: Vec<usize>,
}

/// Errors surfaced by the reader implementation. The default
/// [`ExtractiveReader`] is infallible; this enum exists so production
/// SSM backends can surface tokeniser / weights / OOM failures cleanly.
#[derive(Debug, thiserror::Error)]
pub enum ReaderError {
    /// The reader could not tokenise or otherwise prepare its input.
    #[error("reader input preparation failed: {0}")]
    InvalidInput(String),

    /// The reader's backend (model, runtime, allocator) failed.
    #[error("reader backend failed: {0}")]
    Backend(String),
}

// ─── ReadingComprehender trait ────────────────────────────────────────

/// Pluggable reading-comprehension backend. Implementations may be
/// deterministic (the default [`ExtractiveReader`]) or model-based (a
/// downstream Mamba-3 / Liquid SSM backend).
///
/// Contract:
/// - `read_and_answer` is pure: same `question` and same `passages` →
///   same [`Reading`], no I/O.
/// - `read_and_answer` budgets itself: the L1.5 tier reports a flat
///   ~20 mJ estimate, so the implementation must not exceed that in
///   practice (production SSM weights ≤ ~1B params at int8).
/// - `read_and_answer` MUST set `confidence` to `0.0` when it cannot
///   answer, *not* return an error. Errors are reserved for backend
///   failures (tokeniser, weights, OOM); low-confidence answers are a
///   normal outcome.
pub trait ReadingComprehender: Send + Sync {
    /// Read `passages` with `question` in mind and return the best
    /// extracted answer.
    fn read_and_answer(
        &self,
        question: &str,
        passages: &[Passage],
    ) -> Result<Reading, ReaderError>;

    /// Stable, human-readable name of the reader (for diagnostics).
    fn name(&self) -> &'static str {
        "unnamed"
    }
}

// ─── Default ExtractiveReader ─────────────────────────────────────────

/// Deterministic, zero-energy v0.1 reading-comprehension reference impl.
///
/// Algorithm:
///
/// 1. Tokenise the question into lowercase ASCII words; drop a small
///    stop-word set so the score is not dominated by `"the"`, `"is"`,
///    etc.
/// 2. Split each passage into sentences on `.`, `?`, and `!`.
/// 3. For every sentence, count distinct question-token matches. The
///    sentence with the highest match count wins; ties resolve to the
///    earliest passage, then the earliest sentence.
/// 4. Confidence = `matches / question_tokens`, clamped to `[0, 1]`.
///
/// The reader is *deterministic* (same input → same output across runs
/// and machines) and *zero-energy* (pure ASCII work). It is **not** a
/// production QA system — it is a reference implementation and a
/// fallback. Production deployments should swap in a real SSM via
/// [`ReadingComprehender`].
#[derive(Debug, Clone, Default)]
pub struct ExtractiveReader {
    _private: (),
}

impl ExtractiveReader {
    /// Construct a fresh extractive reader. No state to initialise.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

const STOPWORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "of", "in", "on", "at",
    "to", "for", "with", "by", "from", "and", "or", "but", "if", "as", "it", "its", "this",
    "that", "these", "those", "do", "does", "did", "what", "who", "where", "when", "which",
    "how", "why",
];

fn is_stopword(token: &str) -> bool {
    STOPWORDS.iter().any(|s| *s == token)
}

/// Tokenise a string into lowercase ASCII word tokens, dropping
/// stop-words. Pure, deterministic.
fn tokenize(text: &str) -> Vec<String> {
    text.split(|c: char| !c.is_ascii_alphanumeric())
        .filter(|s| !s.is_empty())
        .map(|s| s.to_ascii_lowercase())
        .filter(|s| !is_stopword(s))
        .collect()
}

/// Split a passage into sentence-shaped fragments. Sentences are
/// trimmed; empty fragments are dropped.
fn split_sentences(text: &str) -> Vec<&str> {
    text.split(|c: char| c == '.' || c == '?' || c == '!')
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .collect()
}

impl ReadingComprehender for ExtractiveReader {
    fn name(&self) -> &'static str {
        "extractive-v0"
    }

    fn read_and_answer(
        &self,
        question: &str,
        passages: &[Passage],
    ) -> Result<Reading, ReaderError> {
        let q_tokens = tokenize(question);
        let q_total = q_tokens.len().max(1) as f32;

        let mut best_score: usize = 0;
        let mut best_answer: Option<String> = None;
        let mut best_passage: Option<usize> = None;

        for (idx, passage) in passages.iter().enumerate() {
            for sentence in split_sentences(&passage.text) {
                let s_tokens = tokenize(sentence);
                if s_tokens.is_empty() {
                    continue;
                }
                let mut matches = 0usize;
                for q in &q_tokens {
                    if s_tokens.iter().any(|s| s == q) {
                        matches += 1;
                    }
                }
                if matches > best_score {
                    best_score = matches;
                    best_answer = Some(sentence.to_string());
                    best_passage = Some(idx);
                }
            }
        }

        match (best_answer, best_passage) {
            (Some(ans), Some(idx)) => {
                let raw = best_score as f32 / q_total;
                let confidence = raw.clamp(0.0, 1.0);
                Ok(Reading {
                    answer: ans,
                    confidence,
                    used_passages: vec![idx],
                })
            }
            _ => Ok(Reading {
                answer: String::new(),
                confidence: 0.0,
                used_passages: Vec::new(),
            }),
        }
    }
}

// ─── Tests ────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn passage_constructors_set_fields() {
        let p = Passage::text("hello world");
        assert_eq!(p.text, "hello world");
        assert!(p.source.is_none());

        let p = Passage::with_source("hello world", "wp:Hello");
        assert_eq!(p.text, "hello world");
        assert_eq!(p.source.as_deref(), Some("wp:Hello"));
    }

    #[test]
    fn tokenize_drops_stopwords_and_lowercases() {
        let t = tokenize("What IS the Capital of France?");
        // 'what', 'is', 'the', 'of' are stopwords.
        assert_eq!(t, vec!["capital", "france"]);
    }

    #[test]
    fn split_sentences_handles_punctuation() {
        let s = split_sentences("Alice ran. Bob walked? Carol stood!");
        assert_eq!(s, vec!["Alice ran", "Bob walked", "Carol stood"]);
    }

    #[test]
    fn extractive_reader_picks_overlapping_sentence() {
        let r = ExtractiveReader::new();
        let passages = vec![
            Passage::text(
                "France is in western Europe. Paris is the capital of France. \
                 The Seine flows through it.",
            ),
        ];
        let reading = r
            .read_and_answer("What is the capital of France?", &passages)
            .expect("reads");
        assert!(reading.answer.contains("Paris"));
        assert!(reading.confidence > 0.0);
        assert_eq!(reading.used_passages, vec![0]);
    }

    #[test]
    fn extractive_reader_zero_passages_returns_empty() {
        let r = ExtractiveReader::new();
        let reading = r.read_and_answer("anything?", &[]).expect("reads");
        assert!(reading.answer.is_empty());
        assert_eq!(reading.confidence, 0.0);
        assert!(reading.used_passages.is_empty());
    }

    #[test]
    fn extractive_reader_is_deterministic() {
        let r = ExtractiveReader::new();
        let passages = vec![
            Passage::text("Cats purr. Dogs bark. Birds sing."),
        ];
        let a = r.read_and_answer("Do cats purr?", &passages).expect("a");
        let b = r.read_and_answer("Do cats purr?", &passages).expect("b");
        assert_eq!(a, b);
    }

    #[test]
    fn extractive_reader_no_overlap_returns_low_confidence() {
        let r = ExtractiveReader::new();
        let passages = vec![Passage::text("The sky is blue. Grass is green.")];
        let reading = r
            .read_and_answer("xenobiology?", &passages)
            .expect("reads");
        // No tokens overlap; confidence stays at 0.0 and used_passages
        // is empty (no candidate sentence was selected).
        assert_eq!(reading.confidence, 0.0);
        assert!(reading.used_passages.is_empty());
    }

    #[test]
    fn extractive_reader_picks_earliest_on_tie() {
        let r = ExtractiveReader::new();
        // Both sentences match exactly one token: "Paris".
        let passages = vec![
            Passage::text("Paris is here. Paris is there."),
        ];
        let reading = r
            .read_and_answer("Where is Paris?", &passages)
            .expect("reads");
        assert!(reading.answer.starts_with("Paris is here"));
    }

    #[test]
    fn extractive_reader_uses_best_passage_index() {
        let r = ExtractiveReader::new();
        let passages = vec![
            Passage::with_source("Berlin is in Germany.", "wp:Berlin"),
            Passage::with_source(
                "Paris is the capital of France. France borders Spain.",
                "wp:Paris",
            ),
            Passage::with_source("Rome is in Italy.", "wp:Rome"),
        ];
        let reading = r
            .read_and_answer("What is the capital of France?", &passages)
            .expect("reads");
        assert!(reading.answer.contains("Paris"));
        assert_eq!(reading.used_passages, vec![1]);
    }

    #[test]
    fn extractive_reader_name_is_extractive_v0() {
        let r = ExtractiveReader::new();
        assert_eq!(r.name(), "extractive-v0");
    }

    #[test]
    fn reader_error_display_is_human_readable() {
        let e = ReaderError::InvalidInput("bad bytes".into());
        let msg = format!("{e}");
        assert!(msg.contains("bad bytes"));
        let e = ReaderError::Backend("OOM".into());
        assert!(format!("{e}").contains("OOM"));
    }
}
