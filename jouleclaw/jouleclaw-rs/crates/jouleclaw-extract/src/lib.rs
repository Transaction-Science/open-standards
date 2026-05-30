//! L1 fact extraction — the deterministic stage every production memory
//! framework runs before the model.
//!
//! Mem0, Zep, Hindsight, and friends all stack a fact-extraction pass
//! between &ldquo;raw text in&rdquo; and &ldquo;structured tags stored.&rdquo; The standard
//! approach is to call a small LLM per capture — Mem0's `gpt-4o-mini`,
//! Hindsight's structured-output prompts. That works, but it makes every
//! captured thought a per-token charge against the model tier. JouleClaw's
//! prior is **inference is the last resort**, so this crate inverts the
//! sequence: an honest deterministic L1 extractor runs first; an L3 LLM
//! extractor (the [`Extractor`] trait) is the fallback the consumer
//! provides when L1 produces nothing useful.
//!
//! ## Honest scope (v1)
//!
//! The reference [`HeuristicExtractor`] pulls four kinds of high-precision
//! tags from raw text:
//!
//! - **Hashtags** — `#topic` tokens.
//! - **Mentions** — `@person` tokens.
//! - **Quoted strings** — content inside `"…"` or `'…'`, preserved verbatim.
//! - **Key:value pairs** — `key: value` markdown-style annotations.
//!
//! These are the four patterns where regex is right. Anything that
//! requires named-entity recognition (&ldquo;Sarah is considering leaving&rdquo;
//! &mdash; pulling `Sarah` as a person) requires NER or an LLM and is
//! **not** done here. A consumer-supplied [`Extractor`] implementation
//! plugs that in at L3 when needed.
//!
//! ## Composing with `jouleclaw-memory`
//!
//! The output [`Extraction`] folds directly into the
//! `BTreeMap<String, String>` shape `jouleclaw-memory` uses for fact
//! metadata via [`merge_into_metadata`]. So the capture path becomes:
//!
//! ```text
//! text → Extractor::extract → Extraction → merge_into_metadata
//!     → MemoryStore::capture (content-address is now keyed on the tags)
//! ```
//!
//! &mdash; without any per-fact model call.

#![forbid(unsafe_code)]

use jouleclaw_memory::MemoryFact;
use regex::Regex;
use std::collections::BTreeMap;
use std::sync::OnceLock;

// ─────────────────────────────────────────────────────────────────────
// Extraction shape
// ─────────────────────────────────────────────────────────────────────

/// The structured output of a fact-extraction pass. Each list is
/// deduplicated and stable-ordered (first occurrence wins).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Extraction {
    /// `#topic` tokens, without the leading `#`.
    pub hashtags: Vec<String>,
    /// `@person` tokens, without the leading `@`.
    pub mentions: Vec<String>,
    /// Substrings appearing inside `"…"` or `'…'`. Verbatim, no quotes.
    pub quoted: Vec<String>,
    /// `key: value` pairs from markdown-style annotations. Keys are
    /// lowercased; values are trimmed.
    pub key_values: Vec<(String, String)>,
}

impl Extraction {
    /// True when no tags of any kind were extracted — the signal the
    /// caller uses to decide whether to escalate to an L3 [`Extractor`].
    pub fn is_empty(&self) -> bool {
        self.hashtags.is_empty()
            && self.mentions.is_empty()
            && self.quoted.is_empty()
            && self.key_values.is_empty()
    }
}

// ─────────────────────────────────────────────────────────────────────
// Trait + heuristic reference impl
// ─────────────────────────────────────────────────────────────────────

/// Pulls structured tags from raw text. The reference
/// [`HeuristicExtractor`] is deterministic and L1-cheap; a consumer
/// supplies a wider implementation (e.g. an LLM-backed extractor) when
/// the heuristic returns nothing useful.
pub trait Extractor: Send + Sync {
    fn extract(&self, text: &str) -> Extraction;
}

/// Deterministic L1 extractor: regex over hashtags, mentions, quoted
/// strings, and key:value annotations. No NER, no LLM.
#[derive(Debug, Default, Clone, Copy)]
pub struct HeuristicExtractor;

fn hashtag_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?:^|\s)#([A-Za-z][A-Za-z0-9_\-]*)").unwrap())
}

fn mention_re() -> &'static Regex {
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r"(?:^|\s)@([A-Za-z][A-Za-z0-9_\-]*)").unwrap())
}

fn quoted_re() -> &'static Regex {
    // Capture either "..." or '...'. Non-greedy. Single-line.
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| Regex::new(r#""([^"\n]+)"|'([^'\n]+)'"#).unwrap())
}

fn kv_re() -> &'static Regex {
    // Line- or fragment-anchored "key: value" where key is a bare word.
    // Value runs to end-of-line / sentence terminator / comma.
    static R: OnceLock<Regex> = OnceLock::new();
    R.get_or_init(|| {
        Regex::new(r"(?:^|[\n;])\s*([A-Za-z][A-Za-z0-9_\-]*)\s*:\s*([^\n;]+)").unwrap()
    })
}

/// Append `value` to `out` unless it's already there (case-sensitive).
fn push_unique(out: &mut Vec<String>, value: String) {
    if !out.iter().any(|v| v == &value) {
        out.push(value);
    }
}

impl Extractor for HeuristicExtractor {
    fn extract(&self, text: &str) -> Extraction {
        let mut e = Extraction::default();

        for cap in hashtag_re().captures_iter(text) {
            if let Some(m) = cap.get(1) {
                push_unique(&mut e.hashtags, m.as_str().to_string());
            }
        }
        for cap in mention_re().captures_iter(text) {
            if let Some(m) = cap.get(1) {
                push_unique(&mut e.mentions, m.as_str().to_string());
            }
        }
        for cap in quoted_re().captures_iter(text) {
            let m = cap.get(1).or_else(|| cap.get(2));
            if let Some(m) = m {
                let s = m.as_str().trim().to_string();
                if !s.is_empty() {
                    push_unique(&mut e.quoted, s);
                }
            }
        }
        for cap in kv_re().captures_iter(text) {
            let (Some(k), Some(v)) = (cap.get(1), cap.get(2)) else {
                continue;
            };
            let key = k.as_str().to_lowercase();
            let val = v.as_str().trim().to_string();
            if val.is_empty() {
                continue;
            }
            // First key wins (don't shadow earlier values from earlier
            // fragments) — keep the extractor's behaviour stable.
            if !e.key_values.iter().any(|(kk, _)| kk == &key) {
                e.key_values.push((key, val));
            }
        }

        e
    }
}

// ─────────────────────────────────────────────────────────────────────
// Metadata merge
// ─────────────────────────────────────────────────────────────────────

/// Fold an [`Extraction`] into a `metadata` map suitable for
/// `jouleclaw_memory::CaptureOptions::metadata`. Stable keys:
///
/// - `hashtags`  — comma-joined hashtag list (no `#`).
/// - `mentions`  — comma-joined mention list (no `@`).
/// - `quoted`    — comma-joined quoted strings (joined with `, `).
/// - one entry per `key:value` pair, namespaced under `kv.<key>`.
///
/// Empty fields are skipped — they would only bloat the content-address.
pub fn merge_into_metadata(extraction: &Extraction, metadata: &mut BTreeMap<String, String>) {
    if !extraction.hashtags.is_empty() {
        metadata.insert("hashtags".to_string(), extraction.hashtags.join(","));
    }
    if !extraction.mentions.is_empty() {
        metadata.insert("mentions".to_string(), extraction.mentions.join(","));
    }
    if !extraction.quoted.is_empty() {
        metadata.insert("quoted".to_string(), extraction.quoted.join(", "));
    }
    for (k, v) in &extraction.key_values {
        metadata.insert(format!("kv.{k}"), v.clone());
    }
}

/// Convenience: extract via `extractor`, fold into a fresh metadata map.
pub fn extract_to_metadata<E: Extractor>(extractor: &E, text: &str) -> BTreeMap<String, String> {
    let mut m = BTreeMap::new();
    merge_into_metadata(&extractor.extract(text), &mut m);
    m
}

/// Convenience: derive the metadata that *would* be attached to a fact
/// extracted from `text`. Used by tests and by the consolidation path to
/// re-derive tags from a stored fact's text without re-extracting on
/// every capture.
pub fn metadata_for_fact<E: Extractor>(extractor: &E, fact: &MemoryFact) -> BTreeMap<String, String> {
    extract_to_metadata(extractor, &fact.text)
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_memory::{CaptureOptions, InMemoryStore, MemoryStore, MemoryType};

    #[test]
    fn extracts_hashtags_and_mentions() {
        let e = HeuristicExtractor.extract(
            "Talked to @sarah today about #career-change and #consulting — promising.",
        );
        assert_eq!(e.mentions, vec!["sarah"]);
        assert_eq!(e.hashtags, vec!["career-change", "consulting"]);
    }

    #[test]
    fn extracts_quoted_strings_double_and_single() {
        // Note: a single-quoted span ends at the *first* apostrophe, so the
        // input doesn't put apostrophes inside `'...'`.
        let e = HeuristicExtractor.extract(
            r#"She said "I'm tired of the reorg" and later 'time to leave'."#,
        );
        assert_eq!(e.quoted, vec!["I'm tired of the reorg", "time to leave"]);
    }

    #[test]
    fn extracts_key_value_annotations() {
        let e = HeuristicExtractor.extract(
            "Decision: ship the new tier; owner: dcharlot; status: in-progress",
        );
        assert_eq!(
            e.key_values,
            vec![
                ("decision".to_string(), "ship the new tier".to_string()),
                ("owner".to_string(), "dcharlot".to_string()),
                ("status".to_string(), "in-progress".to_string()),
            ]
        );
    }

    #[test]
    fn deduplicates_repeated_tags() {
        let e = HeuristicExtractor.extract("#focus #focus and @nate again @nate");
        assert_eq!(e.hashtags, vec!["focus"]);
        assert_eq!(e.mentions, vec!["nate"]);
    }

    #[test]
    fn empty_means_nothing_was_pulled() {
        let e = HeuristicExtractor.extract("nothing tagged in this sentence");
        assert!(e.is_empty(), "expected empty extraction, got {e:?}");
    }

    #[test]
    fn merge_into_metadata_uses_stable_keys() {
        // `kv` anchors only at line-start or `;` to avoid catching sentence
        // colons; the input uses an explicit `;` separator.
        let e = HeuristicExtractor.extract(
            "Talked to @sarah about #career-change; decision: revisit in Q3",
        );
        let mut m = BTreeMap::new();
        merge_into_metadata(&e, &mut m);
        assert_eq!(m.get("mentions").map(String::as_str), Some("sarah"));
        assert_eq!(m.get("hashtags").map(String::as_str), Some("career-change"));
        assert_eq!(m.get("kv.decision").map(String::as_str), Some("revisit in Q3"));
        assert!(!m.contains_key("quoted"), "no quotes in this input");
    }

    #[test]
    fn empty_extraction_does_not_pollute_metadata() {
        let e = Extraction::default();
        let mut m = BTreeMap::new();
        merge_into_metadata(&e, &mut m);
        assert!(m.is_empty());
    }

    #[test]
    fn composes_with_jouleclaw_memory_capture() {
        // The full L1 path: extract tags → fold into metadata → capture.
        // No model call; the tags become part of the content-address.
        let extractor = HeuristicExtractor;
        let text = "Talked to @sarah about #career-change; owner: dcharlot";
        let metadata = extract_to_metadata(&extractor, text);

        let mut store = InMemoryStore::new();
        let fact = store.capture(
            text,
            CaptureOptions {
                kind: Some(MemoryType::Episodic),
                metadata,
                ..Default::default()
            },
            1_700_000_000,
        );
        assert_eq!(fact.metadata.get("mentions").map(String::as_str), Some("sarah"));
        assert_eq!(
            fact.metadata.get("hashtags").map(String::as_str),
            Some("career-change")
        );
        assert_eq!(
            fact.metadata.get("kv.owner").map(String::as_str),
            Some("dcharlot")
        );

        // Recall by the extracted topic finds it.
        let hits = store.recall("career change", Default::default());
        assert!(!hits.is_empty());
    }

    #[test]
    fn hashtag_must_start_with_letter() {
        // Numeric prefixes are not hashtags — avoid pulling `#1` etc.
        let e = HeuristicExtractor.extract("ranked #1 last week, then #recovered");
        assert_eq!(e.hashtags, vec!["recovered"]);
    }

    #[test]
    fn metadata_for_fact_recovers_tags_from_stored_text() {
        let mut store = InMemoryStore::new();
        let fact = store.capture(
            "@sarah and @nate discussed #ARL",
            CaptureOptions::default(),
            1,
        );
        let m = metadata_for_fact(&HeuristicExtractor, &fact);
        assert_eq!(m.get("mentions").map(String::as_str), Some("sarah,nate"));
        assert_eq!(m.get("hashtags").map(String::as_str), Some("ARL"));
    }
}
