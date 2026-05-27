//! Intent classifier trait and the deterministic v0.1 default
//! ([`KeywordClassifier`]).
//!
//! Production deployments plug in a Liquid / Mamba / Hyena backend by
//! implementing [`IntentClassifier`]. The default classifier is keyword-
//! and entropy-based: zero-energy CPU work, fully deterministic, suitable
//! for conformance vectors and tests.

use std::collections::HashMap;

use jouleclaw_cascade::types::TierId;
use serde::{Deserialize, Serialize};

// ─── Intent labels ────────────────────────────────────────────────────

/// The five intent categories emitted by the SSM router.
///
/// The taxonomy mirrors the donor (`verity-cascade::layers::l075_ssm_router`)
/// and is intentionally small: each intent corresponds to a cluster of
/// downstream tiers, not a single one.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Intent {
    /// Conversational / greeting / acknowledgement. Cheapest path —
    /// route to the cache tier, or skip the cascade entirely.
    Conversational,
    /// Deterministic computation: tool dispatch, formula, regex, hash,
    /// unit conversion. Route to L0.5 (tool-compute) and L0.25 (formula).
    Computation,
    /// Factual lookup: short query, low entropy, "what is" / "who is".
    /// Route to L0.1 (LUT), L1.25 (graph-RAG), L2 (federation).
    Factual,
    /// Structural / relational query: "compare", "vs", "between".
    /// Route to L0.25 / L1.375 (formula passes).
    Structural,
    /// Open reasoning: high entropy, complex multi-step. Route to L3
    /// (model) and L4 (wire) as a last resort.
    Reasoning,
}

impl Intent {
    /// Stable wire tag for the intent. Receipts and conformance vectors
    /// emit this string.
    pub fn wire_tag(&self) -> &'static str {
        match self {
            Self::Conversational => "conversational",
            Self::Computation => "computation",
            Self::Factual => "factual",
            Self::Structural => "structural",
            Self::Reasoning => "reasoning",
        }
    }

    /// The downstream tiers the cascade should *prefer* given this
    /// intent. The list is advice, not enforcement — the runtime still
    /// walks the cascade in cost order; the route hint lets coordinate-
    /// aware routers reorder.
    pub fn routed_to(&self) -> Vec<TierId> {
        match self {
            Self::Conversational => vec![TierId::L0, TierId::L0_1FactLut],
            Self::Computation => vec![TierId::L0_5ToolCompute, TierId::L0_25FormulaFirst],
            Self::Factual => vec![
                TierId::L0_1FactLut,
                TierId::L1_25GraphRag,
                TierId::L2(jouleclaw_cascade::types::L2ModelId(0)),
            ],
            Self::Structural => vec![TierId::L0_25FormulaFirst, TierId::L1_375StructContrast],
            Self::Reasoning => vec![
                TierId::L1_5SsmReader,
                TierId::L3(jouleclaw_cascade::types::L3ModelId(0)),
                TierId::L4(jouleclaw_cascade::types::L4ModelId(0)),
            ],
        }
    }
}

// ─── Route hint ───────────────────────────────────────────────────────

/// The structured route hint the SSM router publishes per query.
///
/// Returned both as the typed value to embedded callers (see
/// [`IntentClassifier::classify`]) and as JSON inside the tier's
/// `Answer.output`. The JSON shape is the wire-stable contract:
///
/// ```json
/// {
///   "intent": "factual",
///   "routed_to": ["L0.1", "L1.25", "L2"],
///   "confidence": 0.83
/// }
/// ```
#[derive(Debug, Clone, PartialEq)]
pub struct RouteHint {
    /// The classified intent.
    pub intent: Intent,
    /// Tiers the router suggests in preferred dispatch order.
    pub routed_to: Vec<TierId>,
    /// Classifier confidence in this routing decision, in `[0, 1]`.
    pub confidence: f32,
}

// ─── IntentClassifier trait ───────────────────────────────────────────

/// Pluggable intent classifier. Implementations may be deterministic
/// (the default [`KeywordClassifier`]) or model-based (a downstream
/// Liquid / Mamba / Hyena backend).
///
/// Contract:
/// - `classify` is pure: same input → same output, no I/O.
/// - `classify` budgets itself: the L0.75 tier reports a flat 100 µJ
///   estimate, so the implementation must not exceed that in practice
///   (production SSM weights ≤ ~64M params).
/// - `classify` always returns a [`RouteHint`]; it never refuses. The
///   tier above this trait decides whether to surface the route or to
///   refuse (e.g., for non-text queries).
pub trait IntentClassifier: Send + Sync {
    /// Classify the text into an intent + downstream route advice.
    fn classify(&self, text: &str) -> RouteHint;

    /// Stable, human-readable name of the classifier (for diagnostics).
    fn name(&self) -> &'static str {
        "unnamed"
    }
}

// ─── Default KeywordClassifier ────────────────────────────────────────

/// Deterministic, hash-stable v0.1 intent classifier.
///
/// Combines keyword cues with Shannon entropy and a blake3-stable
/// tie-break. Zero energy, sub-microsecond, fully reproducible: the same
/// query produces the same intent and confidence on every machine, every
/// run. Suitable for conformance vectors.
#[derive(Debug, Clone, Default)]
pub struct KeywordClassifier {
    _private: (),
}

impl KeywordClassifier {
    /// Construct a fresh classifier. No state to initialise.
    pub fn new() -> Self {
        Self { _private: () }
    }
}

impl IntentClassifier for KeywordClassifier {
    fn name(&self) -> &'static str {
        "keyword-v0"
    }

    fn classify(&self, text: &str) -> RouteHint {
        let lower = text.to_lowercase();
        let words: Vec<&str> = lower.split_whitespace().collect();
        let entropy = query_entropy(&lower);

        // 1. Conversational — short query, low entropy, greeting word.
        if words.len() <= 3 && entropy.combined < 2.5 {
            const GREETINGS: &[&str] = &[
                "hello", "hi", "hey", "thanks", "thank", "bye", "ok", "okay", "yes", "no",
            ];
            if words.iter().any(|w| GREETINGS.contains(w)) {
                return RouteHint {
                    intent: Intent::Conversational,
                    routed_to: Intent::Conversational.routed_to(),
                    confidence: 0.92,
                };
            }
        }

        // 2. Computation — tool-shaped keyword present.
        const TOOL_SIGNALS: &[&str] = &[
            "convert", "calculate", "compute", "base64", "sha256", "sha512", "md5", "uuid",
            "regex", "json", "csv", "hex", "binary", "encode", "decode", "hash", "percent",
            "percentage",
        ];
        if words.iter().any(|w| TOOL_SIGNALS.contains(w)) {
            return RouteHint {
                intent: Intent::Computation,
                routed_to: Intent::Computation.routed_to(),
                confidence: 0.95,
            };
        }

        // 3. Structural — comparison / relational cues. Short queries
        // only; a structural keyword inside a long reasoning prompt is
        // not a routing signal.
        const STRUCTURAL_SIGNALS: &[&str] =
            &["compare", "vs", "versus", "between", "difference", "relate", "relation"];
        let has_structural_kw = words.iter().any(|w| STRUCTURAL_SIGNALS.contains(w));
        let short_and_pair = entropy.combined < 4.0 && words.contains(&"and") && words.len() <= 6;
        if (has_structural_kw && words.len() <= 8) || short_and_pair {
            return RouteHint {
                intent: Intent::Structural,
                routed_to: Intent::Structural.routed_to(),
                confidence: 0.78,
            };
        }

        // 4. Factual lookup — short query, "what/who/where/when" cue.
        const FACTUAL_LEADS: &[&str] = &["what", "who", "where", "when", "which", "how"];
        if !words.is_empty()
            && FACTUAL_LEADS.contains(&words[0])
            && words.len() <= 10
            && entropy.combined < 4.5
        {
            return RouteHint {
                intent: Intent::Factual,
                routed_to: Intent::Factual.routed_to(),
                confidence: 0.84,
            };
        }

        // 5. Entropy fallback.
        if entropy.combined > 4.5 || words.len() > 12 {
            return RouteHint {
                intent: Intent::Reasoning,
                routed_to: Intent::Reasoning.routed_to(),
                confidence: 0.7,
            };
        }
        if entropy.combined < 3.0 && words.len() <= 5 {
            return RouteHint {
                intent: Intent::Factual,
                routed_to: Intent::Factual.routed_to(),
                confidence: 0.6,
            };
        }

        // 6. Hash-stable tie-break. When no keyword cue fires and entropy
        // is medium, fold the blake3 digest into a discrete bucket so
        // identical queries always pick the same intent.
        let bucket = stable_bucket(text);
        let intent = match bucket {
            0 => Intent::Factual,
            1 => Intent::Structural,
            _ => Intent::Reasoning,
        };
        RouteHint {
            intent,
            routed_to: intent.routed_to(),
            confidence: 0.55,
        }
    }
}

/// Stable per-query bucket in `0..=2`. The donor's keyword routing has
/// no tie-break, which causes near-equal queries to oscillate between
/// intents under tiny perturbations. We fold the JouleClaw-standard
/// blake3 digest into a discrete bucket so the classifier is bit-stable.
fn stable_bucket(text: &str) -> u8 {
    // We deliberately avoid pulling in the `blake3` crate here — the
    // jouleclaw-energy / -prov crates already use it, but this layer is
    // optional. A simple FNV-1a 64-bit digest is more than enough to
    // randomise three buckets and stays deterministic across platforms.
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for b in text.as_bytes() {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100_0000_01b3);
    }
    (h % 3) as u8
}

// ─── Entropy helpers ──────────────────────────────────────────────────

/// Query entropy measurements at multiple levels. Ported wholesale from
/// the donor.
#[derive(Debug, Clone, Copy)]
pub struct QueryEntropy {
    /// Shannon entropy of character distribution (bits).
    pub char_entropy: f64,
    /// Shannon entropy of word distribution (bits).
    pub word_entropy: f64,
    /// Combined entropy signal: 60% char + 40% word, in bits.
    pub combined: f64,
    /// Vocabulary richness: `unique_words / total_words` in `[0, 1]`.
    pub vocab_richness: f64,
}

/// Compute Shannon entropy of the query at character and word levels.
///
/// `H(X) = -Σ p(x) log₂ p(x)` measures the average information content
/// per symbol. Low entropy = predictable = lookup. High entropy =
/// unpredictable = needs computation. O(n) — a few hundred nanoseconds,
/// ~0 J.
pub fn query_entropy(query: &str) -> QueryEntropy {
    // Character-level entropy.
    let char_entropy = {
        let mut counts: HashMap<char, u32> = HashMap::new();
        let mut total = 0u32;
        for c in query.chars() {
            if c.is_ascii_whitespace() {
                continue;
            }
            *counts.entry(c.to_ascii_lowercase()).or_default() += 1;
            total += 1;
        }
        if total == 0 {
            return QueryEntropy {
                char_entropy: 0.0,
                word_entropy: 0.0,
                combined: 0.0,
                vocab_richness: 0.0,
            };
        }
        let t = total as f64;
        counts
            .values()
            .map(|&c| {
                let p = c as f64 / t;
                -p * p.log2()
            })
            .sum::<f64>()
    };

    // Word-level entropy.
    let words: Vec<&str> = query.split_whitespace().collect();
    let word_entropy = if words.is_empty() {
        0.0
    } else {
        let mut counts: HashMap<&str, u32> = HashMap::new();
        for &w in &words {
            *counts.entry(w).or_default() += 1;
        }
        let total = words.len() as f64;
        counts
            .values()
            .map(|&c| {
                let p = c as f64 / total;
                -p * p.log2()
            })
            .sum::<f64>()
    };

    // Vocabulary richness.
    let unique_words = {
        let mut seen: HashMap<&str, bool> = HashMap::new();
        for &w in &words {
            seen.entry(w).or_insert(true);
        }
        seen.len()
    };
    let vocab_richness = if words.is_empty() {
        0.0
    } else {
        unique_words as f64 / words.len() as f64
    };

    // 60% character + 40% word — character captures structural complexity,
    // word captures semantic diversity.
    let combined = 0.6 * char_entropy + 0.4 * word_entropy;

    QueryEntropy {
        char_entropy,
        word_entropy,
        combined,
        vocab_richness,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn intent_wire_tags_are_stable() {
        assert_eq!(Intent::Conversational.wire_tag(), "conversational");
        assert_eq!(Intent::Computation.wire_tag(), "computation");
        assert_eq!(Intent::Factual.wire_tag(), "factual");
        assert_eq!(Intent::Structural.wire_tag(), "structural");
        assert_eq!(Intent::Reasoning.wire_tag(), "reasoning");
    }

    #[test]
    fn intent_routed_to_is_nonempty() {
        for i in [
            Intent::Conversational,
            Intent::Computation,
            Intent::Factual,
            Intent::Structural,
            Intent::Reasoning,
        ] {
            assert!(!i.routed_to().is_empty(), "{:?} routes empty", i);
        }
    }

    #[test]
    fn classifier_detects_greeting() {
        let c = KeywordClassifier::new();
        let r = c.classify("hello");
        assert_eq!(r.intent, Intent::Conversational);
        assert!(r.confidence > 0.9);
    }

    #[test]
    fn classifier_detects_tool() {
        let c = KeywordClassifier::new();
        let r = c.classify("convert 5 miles to km");
        assert_eq!(r.intent, Intent::Computation);
        assert!(r.confidence > 0.9);
        assert!(r.routed_to.contains(&TierId::L0_5ToolCompute));
    }

    #[test]
    fn classifier_detects_structural() {
        let c = KeywordClassifier::new();
        let r = c.classify("compare fire and water");
        assert_eq!(r.intent, Intent::Structural);
    }

    #[test]
    fn classifier_detects_factual() {
        let c = KeywordClassifier::new();
        let r = c.classify("what is the capital of France");
        assert_eq!(r.intent, Intent::Factual);
    }

    #[test]
    fn classifier_detects_reasoning_high_entropy() {
        let c = KeywordClassifier::new();
        let r = c.classify(
            "explain the mechanistic relationship between quantum entanglement \
             and information theoretic entropy bounds in distributed computing systems",
        );
        assert_eq!(r.intent, Intent::Reasoning);
    }

    #[test]
    fn classifier_is_deterministic() {
        let c = KeywordClassifier::new();
        let r1 = c.classify("midweight ambiguous medium length sentence here today");
        let r2 = c.classify("midweight ambiguous medium length sentence here today");
        assert_eq!(r1.intent, r2.intent);
        assert!((r1.confidence - r2.confidence).abs() < f32::EPSILON);
    }

    #[test]
    fn entropy_hello_is_low() {
        let e = query_entropy("hello");
        assert!(e.combined < 3.0, "got {}", e.combined);
    }

    #[test]
    fn entropy_complex_is_high() {
        let e = query_entropy(
            "explain the mechanistic relationship between quantum entanglement \
             and information theoretic entropy bounds in distributed computing systems",
        );
        assert!(e.combined > 4.0, "got {}", e.combined);
    }

    #[test]
    fn entropy_empty_is_zero() {
        let e = query_entropy("");
        assert_eq!(e.char_entropy, 0.0);
        assert_eq!(e.word_entropy, 0.0);
        assert_eq!(e.combined, 0.0);
        assert_eq!(e.vocab_richness, 0.0);
    }

    #[test]
    fn vocab_richness_repetitive_is_low() {
        let e = query_entropy("the the the the the");
        assert!(e.vocab_richness < 0.3, "got {}", e.vocab_richness);
    }

    #[test]
    fn vocab_richness_diverse_is_high() {
        let e = query_entropy("every word here is completely unique and different");
        assert!(e.vocab_richness > 0.9, "got {}", e.vocab_richness);
    }

    #[test]
    fn stable_bucket_is_deterministic() {
        assert_eq!(stable_bucket("hello world"), stable_bucket("hello world"));
        // Different inputs may yield the same bucket; we only test stability.
        let a = stable_bucket("a");
        let b = stable_bucket("a");
        assert_eq!(a, b);
    }

    #[test]
    fn classifier_name_is_keyword_v0() {
        let c = KeywordClassifier::new();
        assert_eq!(c.name(), "keyword-v0");
    }
}
