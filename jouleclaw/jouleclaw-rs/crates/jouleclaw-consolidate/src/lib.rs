//! Consolidation — the CoALA episodic→semantic transform, run on a
//! schedule the consumer drives.
//!
//! CoALA (Sumers et al. 2023) names **consolidation** as the load-bearing
//! mechanism that makes episodic + semantic productive: episodic memories
//! accumulate one-by-one, then on a periodic pass they are *clustered,
//! summarised, and lifted* into semantic facts that compress the
//! accumulation. The same transform feeds *procedural* memory when a
//! cluster's pattern is regular enough to compile into a deterministic
//! procedure — that path goes through [`jouleclaw-skill`] /
//! [`jouleclaw-promote`], not this crate.
//!
//! ## Honest scope (v1)
//!
//! This crate ships the **deterministic, L1-cheap** half of consolidation:
//!
//! - A [`Consolidator`] trait — the contract a consolidator implements.
//! - [`TagClusterConsolidator`], a deterministic reference: group recent
//!   episodic facts by shared metadata key:value pairs (e.g.
//!   `person:Sarah`, `topic:career`); when ≥ `threshold` facts share a
//!   key:value, emit one **semantic fact** that aggregates them.
//! - [`run_consolidation_pass`] — the scheduler entry: walk the recent
//!   episodic window, run the consolidator, capture the new semantic
//!   facts back into the store. Idempotent on content-address so a
//!   repeated pass over an unchanged window does not duplicate.
//!
//! LLM-driven prose-level abstraction ("Sarah is exploring leaving her
//! current job to start her own firm") is the obvious next step; it is
//! exactly the [`Consolidator`] trait's intended L3 extension point and is
//! **not** implemented here. The deterministic reference does what
//! deterministic code can honestly do: aggregate by tag, name the
//! cluster, point at the constituents.

#![forbid(unsafe_code)]

use jouleclaw_memory::{
    CaptureOptions, MemoryFact, MemoryStore, MemoryType, TrustTier,
};
use std::collections::BTreeMap;

// ─────────────────────────────────────────────────────────────────────
// Consolidator trait + emitted facts
// ─────────────────────────────────────────────────────────────────────

/// A semantic fact a consolidator wants to capture, in
/// [`CaptureOptions`]-shaped form so it can flow directly through
/// [`MemoryStore::capture`].
#[derive(Debug, Clone)]
pub struct EmittedFact {
    /// The semantic fact's text. Deterministic reference emits an
    /// aggregate; an LLM consolidator would emit prose.
    pub text: String,
    /// Capture options — `kind` defaults to [`MemoryType::Semantic`].
    pub options: CaptureOptions,
}

/// The consolidator contract — single method, single direction (no
/// mutation of the inputs).
pub trait Consolidator: Send + Sync {
    /// Examine a window of episodic facts and emit zero or more
    /// semantic facts. Implementations MUST be pure (no external
    /// side-effects); the caller decides whether to capture them.
    fn consolidate(&self, episodic: &[MemoryFact]) -> Vec<EmittedFact>;
}

// ─────────────────────────────────────────────────────────────────────
// Deterministic reference: tag clustering
// ─────────────────────────────────────────────────────────────────────

/// Reference consolidator: cluster episodic facts by shared metadata
/// `key=value`; for each cluster with at least `threshold` members, emit
/// one semantic fact aggregating the constituents.
///
/// Excludes high-cardinality keys that would generate one cluster per
/// fact (e.g. `created_at`-style timestamps). The default exclude list is
/// empty; the caller filters by what their tag schema produces.
pub struct TagClusterConsolidator {
    /// Minimum cluster size to consolidate. Default 3.
    pub threshold: usize,
    /// Metadata keys to ignore when forming clusters. Default empty.
    pub exclude_keys: Vec<String>,
}

impl Default for TagClusterConsolidator {
    fn default() -> Self {
        Self {
            threshold: 3,
            exclude_keys: Vec::new(),
        }
    }
}

impl TagClusterConsolidator {
    pub fn with_threshold(mut self, threshold: usize) -> Self {
        self.threshold = threshold.max(2);
        self
    }
    pub fn exclude(mut self, key: impl Into<String>) -> Self {
        self.exclude_keys.push(key.into());
        self
    }

    fn is_excluded(&self, key: &str) -> bool {
        self.exclude_keys.iter().any(|k| k == key)
    }
}

impl Consolidator for TagClusterConsolidator {
    fn consolidate(&self, episodic: &[MemoryFact]) -> Vec<EmittedFact> {
        // Cluster key: "key=value" string for stable grouping.
        let mut clusters: BTreeMap<String, Vec<&MemoryFact>> = BTreeMap::new();
        for f in episodic {
            // Only episodic facts participate in consolidation.
            if f.kind != MemoryType::Episodic {
                continue;
            }
            for (k, v) in &f.metadata {
                if self.is_excluded(k) {
                    continue;
                }
                let key = format!("{k}={v}");
                clusters.entry(key).or_default().push(f);
            }
        }
        let mut out = Vec::new();
        for (cluster_key, facts) in clusters {
            if facts.len() < self.threshold {
                continue;
            }
            // Stable ordering by id for deterministic output.
            let mut ids: Vec<String> = facts.iter().map(|f| f.id.clone()).collect();
            ids.sort();
            // Aggregate metadata: keep the cluster's defining key=value
            // plus a `consolidated_from` count and a sample id range.
            let mut meta = BTreeMap::new();
            if let Some((k, v)) = cluster_key.split_once('=') {
                meta.insert(k.to_string(), v.to_string());
            }
            meta.insert("consolidated_from".to_string(), ids.len().to_string());
            // First + last id (sorted) so the cluster is auditable but
            // the metadata stays bounded — content-address must not blow
            // up with cluster size.
            if let (Some(first), Some(last)) = (ids.first(), ids.last()) {
                meta.insert("first_episode".to_string(), first.clone());
                if first != last {
                    meta.insert("last_episode".to_string(), last.clone());
                }
            }
            // Lowest trust across the cluster — the consolidated semantic
            // fact can't claim more trust than its weakest constituent.
            let min_trust = facts
                .iter()
                .map(|f| f.source_trust.0)
                .min()
                .unwrap_or(TrustTier::DEFAULT.0);
            let text = format!(
                "cluster:{cluster_key} (n={n}) — consolidated from {n} episodic memories",
                n = ids.len()
            );
            out.push(EmittedFact {
                text,
                options: CaptureOptions {
                    kind: Some(MemoryType::Semantic),
                    metadata: meta,
                    source_trust: Some(TrustTier(min_trust)),
                    valid_from: None,
                    valid_to: None,
                },
            });
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────
// Scheduler / runner
// ─────────────────────────────────────────────────────────────────────

/// Options for [`run_consolidation_pass`].
#[derive(Debug, Clone)]
pub struct ConsolidationOptions {
    /// How many recent facts to examine. Default 200.
    pub window: usize,
}

impl Default for ConsolidationOptions {
    fn default() -> Self {
        Self { window: 200 }
    }
}

/// What a consolidation pass produced.
#[derive(Debug, Clone, Default)]
pub struct ConsolidationReport {
    /// The episodic-fact window that was examined.
    pub window: usize,
    /// Newly-captured semantic facts (deduplicated by content-address —
    /// re-running an unchanged pass returns zero new captures).
    pub new_semantic: Vec<MemoryFact>,
}

/// Run one consolidation pass.
///
/// 1. Pull the `window` most-recent facts from `store`.
/// 2. Filter to [`MemoryType::Episodic`] (consolidation only flows
///    episodic→semantic by construction).
/// 3. Run `consolidator` over the filtered window.
/// 4. Capture each emitted fact back into the store; the store's
///    content-address idempotency makes the pass safe to repeat.
///
/// `now_secs` is the wall-clock seed for `created_at` on emitted facts —
/// the caller owns the clock so this stays testable.
pub fn run_consolidation_pass<S, C>(
    store: &mut S,
    consolidator: &C,
    opts: ConsolidationOptions,
    now_secs: u64,
) -> ConsolidationReport
where
    S: MemoryStore + ?Sized,
    C: Consolidator + ?Sized,
{
    let recent = store.recent(opts.window);
    let episodic: Vec<MemoryFact> = recent
        .into_iter()
        .filter(|f| f.kind == MemoryType::Episodic)
        .collect();
    let emitted = consolidator.consolidate(&episodic);
    let mut new_semantic = Vec::with_capacity(emitted.len());
    let before = store.len();
    for em in emitted {
        let prior = store.len();
        let fact = store.capture(&em.text, em.options, now_secs);
        if store.len() > prior {
            new_semantic.push(fact);
        }
    }
    // Defensive: if `store.len()` shrank (impossible with the in-memory
    // store but allowed by the trait), report what we saw rather than
    // panic.
    let _ = before;
    ConsolidationReport {
        window: opts.window,
        new_semantic,
    }
}

// ─────────────────────────────────────────────────────────────────────
// LLM-backed consolidator
// ─────────────────────────────────────────────────────────────────────

/// LLM-backed consolidator — wraps a
/// [`jouleclaw_llm_cheap::LlmBackend`] to emit prose-level semantic
/// facts from a window of episodic facts.
///
/// Honest scope (v1):
///
/// - **One LLM call per pass**, not per fact. Cheaper; relies on the
///   model holding the whole window in context. Window-size discipline
///   is the consumer's job — see [`ConsolidationOptions::window`].
/// - The reply MUST be a JSON object
///   `{"facts": [{"text": "...", "key": "person", "value": "Sarah"}]}`.
///   On parse failure, the consolidator emits **zero** facts and the
///   pass is a no-op (honest: a model that won't conform is a failed
///   abstractor, not a passing one — silently emitting hallucinated
///   semantic memory would corrupt the store).
/// - Emitted facts get `TrustTier::Estimator` and inherit the cluster
///   `key=value` as a tag (so they can be de-duplicated by the
///   content-addressing store the same way the deterministic
///   consolidator's emissions are).
/// - Joule spend is **not** carried back through the
///   [`Consolidator`] trait (which returns `Vec<EmittedFact>` and is
///   intentionally pure). The consumer reads
///   `last_call_joules_uj()` after `consolidate()` if it wants the
///   number.
pub struct LlmConsolidator {
    backend: std::sync::Arc<dyn jouleclaw_llm_cheap::LlmBackend>,
    max_tokens: u32,
    temperature: f32,
    last_joules_uj: std::sync::Mutex<u64>,
}

impl LlmConsolidator {
    /// Build a new LLM consolidator. Default 512 token cap, temperature 0.0.
    pub fn new(backend: std::sync::Arc<dyn jouleclaw_llm_cheap::LlmBackend>) -> Self {
        Self {
            backend,
            max_tokens: 512,
            temperature: 0.0,
            last_joules_uj: std::sync::Mutex::new(0),
        }
    }

    pub fn with_max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens.max(1);
        self
    }

    pub fn with_temperature(mut self, t: f32) -> Self {
        self.temperature = t;
        self
    }

    /// Microjoules spent on the most recent `consolidate()` call.
    /// Returns 0 before any call has run.
    pub fn last_call_joules_uj(&self) -> u64 {
        *self.last_joules_uj.lock().expect("mutex")
    }

    fn render_prompt(&self, episodic: &[MemoryFact]) -> String {
        let mut p = String::new();
        p.push_str("Summarise the recurring patterns in these EPISODIC facts as SEMANTIC facts.\n");
        p.push_str(
            "Reply with ONLY a JSON object: \
             {\"facts\": [{\"text\": \"<one sentence>\", \"key\": \"<tag>\", \"value\": \"<tag value>\"}]}.\n",
        );
        p.push_str(
            "Only emit a semantic fact if at least 3 episodic facts share the same key:value. \
             Do not invent facts that the episodic record does not support.\n\n",
        );
        p.push_str("EPISODIC:\n");
        for f in episodic.iter().filter(|f| f.kind == MemoryType::Episodic) {
            p.push_str(&format!("- {}", f.text));
            if !f.metadata.is_empty() {
                p.push_str(" [");
                let pairs: Vec<String> = f
                    .metadata
                    .iter()
                    .map(|(k, v)| format!("{k}={v}"))
                    .collect();
                p.push_str(&pairs.join(", "));
                p.push(']');
            }
            p.push('\n');
        }
        p
    }
}

impl Consolidator for LlmConsolidator {
    fn consolidate(&self, episodic: &[MemoryFact]) -> Vec<EmittedFact> {
        if episodic.is_empty() {
            *self.last_joules_uj.lock().expect("mutex") = 0;
            return Vec::new();
        }

        let prompt = self.render_prompt(episodic);
        let mut req = jouleclaw_llm_cheap::LlmRequest::from_prompt(prompt, self.max_tokens);
        req.temperature = self.temperature;

        let resp = match self.backend.complete(&req) {
            Ok(r) => r,
            Err(_) => {
                // Backend failure — honest no-op. Don't fabricate
                // semantic memory.
                let est_uj = (self.backend.typical_joules_per_call() * 1_000_000.0) as u64;
                *self.last_joules_uj.lock().expect("mutex") = est_uj;
                return Vec::new();
            }
        };

        let joules_uj = match resp.energy_joules {
            Some(j) => (j * 1_000_000.0) as u64,
            None => (self.backend.typical_joules_per_call() * 1_000_000.0) as u64,
        };
        *self.last_joules_uj.lock().expect("mutex") = joules_uj;

        let parsed: serde_json::Value = match serde_json::from_str(resp.text.trim()) {
            Ok(v) => v,
            Err(_) => return Vec::new(),
        };
        let arr = match parsed.get("facts").and_then(|v| v.as_array()) {
            Some(a) => a,
            None => return Vec::new(),
        };
        let mut out = Vec::with_capacity(arr.len());
        for f in arr {
            let text = match f.get("text").and_then(|v| v.as_str()) {
                Some(s) if !s.is_empty() => s.to_string(),
                _ => continue,
            };
            let mut metadata = BTreeMap::new();
            if let (Some(k), Some(v)) = (
                f.get("key").and_then(|x| x.as_str()),
                f.get("value").and_then(|x| x.as_str()),
            ) {
                metadata.insert(k.to_string(), v.to_string());
            }
            metadata.insert("source".into(), "llm-consolidator".into());
            // Lowest trust across the episodic window — a semantic
            // abstraction can't claim more trust than its weakest
            // constituent. Mirrors TagClusterConsolidator.
            let min_trust = episodic
                .iter()
                .map(|f| f.source_trust.0)
                .min()
                .unwrap_or(TrustTier::DEFAULT.0);
            let options = CaptureOptions {
                kind: Some(MemoryType::Semantic),
                metadata,
                source_trust: Some(TrustTier(min_trust)),
                valid_from: None,
                valid_to: None,
            };
            out.push(EmittedFact { text, options });
        }
        out
    }
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_memory::InMemoryStore;

    fn meta(pairs: &[(&str, &str)]) -> BTreeMap<String, String> {
        pairs
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    fn capture_episodic(store: &mut InMemoryStore, text: &str, m: BTreeMap<String, String>, t: u64) {
        store.capture(
            text,
            CaptureOptions {
                kind: Some(MemoryType::Episodic),
                metadata: m,
                ..Default::default()
            },
            t,
        );
    }

    #[test]
    fn cluster_below_threshold_emits_nothing() {
        let c = TagClusterConsolidator::default(); // threshold 3
        let mut s = InMemoryStore::new();
        capture_episodic(&mut s, "a", meta(&[("person", "Sarah")]), 1);
        capture_episodic(&mut s, "b", meta(&[("person", "Sarah")]), 2);
        let report = run_consolidation_pass(&mut s, &c, Default::default(), 100);
        assert_eq!(report.new_semantic.len(), 0);
    }

    #[test]
    fn cluster_at_threshold_emits_one_semantic_fact() {
        let c = TagClusterConsolidator::default(); // threshold 3
        let mut s = InMemoryStore::new();
        capture_episodic(&mut s, "thought 1", meta(&[("person", "Sarah")]), 1);
        capture_episodic(&mut s, "thought 2", meta(&[("person", "Sarah")]), 2);
        capture_episodic(&mut s, "thought 3", meta(&[("person", "Sarah")]), 3);
        let report = run_consolidation_pass(&mut s, &c, Default::default(), 100);
        assert_eq!(report.new_semantic.len(), 1);
        let semantic = &report.new_semantic[0];
        assert_eq!(semantic.kind, MemoryType::Semantic);
        assert!(semantic.text.contains("person=Sarah"));
        assert_eq!(
            semantic.metadata.get("consolidated_from").map(String::as_str),
            Some("3")
        );
        assert_eq!(
            semantic.metadata.get("person").map(String::as_str),
            Some("Sarah")
        );
    }

    #[test]
    fn separate_tag_values_form_separate_clusters() {
        let c = TagClusterConsolidator::default(); // threshold 3
        let mut s = InMemoryStore::new();
        for i in 0..3 {
            capture_episodic(&mut s, &format!("S{i}"), meta(&[("person", "Sarah")]), i as u64);
        }
        for i in 0..3 {
            capture_episodic(&mut s, &format!("N{i}"), meta(&[("person", "Nate")]), 10 + i as u64);
        }
        let report = run_consolidation_pass(&mut s, &c, Default::default(), 100);
        assert_eq!(report.new_semantic.len(), 2);
    }

    #[test]
    fn excluded_keys_do_not_cluster() {
        let c = TagClusterConsolidator::default().exclude("session_id");
        let mut s = InMemoryStore::new();
        // Five facts all in the same session — would cluster on session
        // alone but we excluded that key, so only the `topic` cluster
        // emits.
        for i in 0..5 {
            capture_episodic(
                &mut s,
                &format!("note {i}"),
                meta(&[("session_id", "abc"), ("topic", "career")]),
                i,
            );
        }
        let report = run_consolidation_pass(&mut s, &c, Default::default(), 100);
        assert_eq!(report.new_semantic.len(), 1);
        assert!(report.new_semantic[0]
            .text
            .contains("topic=career"));
    }

    #[test]
    fn semantic_facts_in_window_are_ignored() {
        let c = TagClusterConsolidator::default().with_threshold(2);
        let mut s = InMemoryStore::new();
        capture_episodic(&mut s, "ep1", meta(&[("topic", "x")]), 1);
        capture_episodic(&mut s, "ep2", meta(&[("topic", "x")]), 2);
        // A pre-existing semantic fact with the same tag must not
        // re-consolidate.
        s.capture(
            "previous semantic",
            CaptureOptions {
                kind: Some(MemoryType::Semantic),
                metadata: meta(&[("topic", "x")]),
                ..Default::default()
            },
            3,
        );
        let report = run_consolidation_pass(&mut s, &c, Default::default(), 100);
        assert_eq!(report.new_semantic.len(), 1);
        assert_eq!(
            report.new_semantic[0].metadata.get("consolidated_from").map(String::as_str),
            Some("2"),
            "only the two episodic facts cluster — the pre-existing semantic is ignored"
        );
    }

    #[test]
    fn repeat_pass_over_unchanged_window_is_idempotent() {
        let c = TagClusterConsolidator::default().with_threshold(2);
        let mut s = InMemoryStore::new();
        capture_episodic(&mut s, "ep1", meta(&[("person", "Sarah")]), 1);
        capture_episodic(&mut s, "ep2", meta(&[("person", "Sarah")]), 2);
        let first = run_consolidation_pass(&mut s, &c, Default::default(), 100);
        assert_eq!(first.new_semantic.len(), 1);
        let len_after = s.len();
        // Second pass — same window, same consolidator → no new captures
        // (content-address idempotency).
        let second = run_consolidation_pass(&mut s, &c, Default::default(), 200);
        assert_eq!(second.new_semantic.len(), 0);
        assert_eq!(s.len(), len_after);
    }

    #[test]
    fn min_trust_propagates_to_consolidated_fact() {
        let c = TagClusterConsolidator::default().with_threshold(2);
        let mut s = InMemoryStore::new();
        // Two episodic facts with the same tag, mixed trust.
        s.capture(
            "primary",
            CaptureOptions {
                kind: Some(MemoryType::Episodic),
                metadata: meta(&[("topic", "career")]),
                source_trust: Some(TrustTier(8)),
                ..Default::default()
            },
            1,
        );
        s.capture(
            "hearsay",
            CaptureOptions {
                kind: Some(MemoryType::Episodic),
                metadata: meta(&[("topic", "career")]),
                source_trust: Some(TrustTier(2)),
                ..Default::default()
            },
            2,
        );
        let report = run_consolidation_pass(&mut s, &c, Default::default(), 100);
        assert_eq!(report.new_semantic.len(), 1);
        // Consolidated trust = floor across constituents.
        assert_eq!(report.new_semantic[0].source_trust, TrustTier(2));
    }

    #[test]
    fn high_cardinality_keys_can_be_excluded() {
        // A unique-id key would create one cluster per fact (size 1) and
        // emit nothing — verify the threshold check protects us.
        let c = TagClusterConsolidator::default().with_threshold(3);
        let mut s = InMemoryStore::new();
        for i in 0..3 {
            capture_episodic(&mut s, &format!("f{i}"), meta(&[("uid", &format!("u{i}"))]), i);
        }
        let report = run_consolidation_pass(&mut s, &c, Default::default(), 100);
        assert_eq!(report.new_semantic.len(), 0);
    }

    // ─── LlmConsolidator ─────────────────────────────────────────────

    struct ScriptedBackend(String);
    impl jouleclaw_llm_cheap::LlmBackend for ScriptedBackend {
        fn model_name(&self) -> &str {
            "test:scripted"
        }
        fn complete(
            &self,
            _req: &jouleclaw_llm_cheap::LlmRequest,
        ) -> Result<jouleclaw_llm_cheap::LlmResponse, jouleclaw_llm_cheap::LlmError> {
            Ok(jouleclaw_llm_cheap::LlmResponse {
                text: self.0.clone(),
                finish_reason: jouleclaw_llm_cheap::FinishReason::Stop,
                input_tokens: 0,
                output_tokens: self.0.len() as u32,
                energy_joules: Some(0.002),
            })
        }
    }

    struct ErrBackend;
    impl jouleclaw_llm_cheap::LlmBackend for ErrBackend {
        fn model_name(&self) -> &str {
            "test:err"
        }
        fn complete(
            &self,
            _req: &jouleclaw_llm_cheap::LlmRequest,
        ) -> Result<jouleclaw_llm_cheap::LlmResponse, jouleclaw_llm_cheap::LlmError> {
            Err(jouleclaw_llm_cheap::LlmError::Unavailable("test".into()))
        }
    }

    fn make_episodic(text: &str, m: BTreeMap<String, String>, t: u64) -> MemoryFact {
        let id = MemoryFact::content_address(text, &m, MemoryType::Episodic, None, None);
        MemoryFact {
            id,
            text: text.into(),
            metadata: m,
            kind: MemoryType::Episodic,
            created_at: t,
            source_trust: TrustTier::DEFAULT,
            valid_from: None,
            valid_to: None,
        }
    }

    fn episodic_window() -> Vec<MemoryFact> {
        vec![
            make_episodic("Sarah mentioned her side project", meta(&[("person", "Sarah")]), 1),
            make_episodic("Sarah talked about her career arc", meta(&[("person", "Sarah")]), 2),
            make_episodic("Sarah is considering a startup", meta(&[("person", "Sarah")]), 3),
        ]
    }

    #[test]
    fn llm_consolidator_emits_facts_from_json_reply() {
        let c = LlmConsolidator::new(std::sync::Arc::new(ScriptedBackend(
            r#"{"facts": [
                {"text": "Sarah is exploring leaving her job", "key": "person", "value": "Sarah"}
            ]}"#
            .into(),
        )));
        let emitted = c.consolidate(&episodic_window());
        assert_eq!(emitted.len(), 1);
        assert_eq!(
            emitted[0].text,
            "Sarah is exploring leaving her job"
        );
        assert_eq!(emitted[0].options.kind, Some(MemoryType::Semantic));
        assert_eq!(
            emitted[0].options.metadata.get("person"),
            Some(&"Sarah".to_string())
        );
        assert_eq!(
            emitted[0].options.metadata.get("source"),
            Some(&"llm-consolidator".to_string())
        );
        assert!(c.last_call_joules_uj() > 0);
    }

    #[test]
    fn llm_consolidator_empty_window_is_noop() {
        let c = LlmConsolidator::new(std::sync::Arc::new(ScriptedBackend(
            r#"{"facts": [{"text": "hallucinated", "key": "x", "value": "y"}]}"#.into(),
        )));
        let emitted = c.consolidate(&[]);
        assert!(emitted.is_empty());
        assert_eq!(c.last_call_joules_uj(), 0, "no call made");
    }

    #[test]
    fn llm_consolidator_unparseable_reply_emits_nothing() {
        let c = LlmConsolidator::new(std::sync::Arc::new(ScriptedBackend(
            "garbage not json".into(),
        )));
        let emitted = c.consolidate(&episodic_window());
        assert!(emitted.is_empty(), "honest no-op on bad reply");
    }

    #[test]
    fn llm_consolidator_backend_error_emits_nothing_but_logs_cost() {
        let c = LlmConsolidator::new(std::sync::Arc::new(ErrBackend));
        let emitted = c.consolidate(&episodic_window());
        assert!(emitted.is_empty());
        assert!(c.last_call_joules_uj() > 0, "estimated cost still logged");
    }

    #[test]
    fn llm_consolidator_drops_empty_text_facts() {
        let c = LlmConsolidator::new(std::sync::Arc::new(ScriptedBackend(
            r#"{"facts": [
                {"text": "", "key": "person", "value": "Sarah"},
                {"text": "actual fact", "key": "person", "value": "Sarah"}
            ]}"#
            .into(),
        )));
        let emitted = c.consolidate(&episodic_window());
        assert_eq!(emitted.len(), 1);
        assert_eq!(emitted[0].text, "actual fact");
    }

    #[test]
    fn llm_consolidator_runs_through_consolidation_pass() {
        let c = LlmConsolidator::new(std::sync::Arc::new(ScriptedBackend(
            r#"{"facts": [{"text": "Sarah is exploring leaving her job", "key": "person", "value": "Sarah"}]}"#
                .into(),
        )));
        let mut s = InMemoryStore::new();
        capture_episodic(&mut s, "ep1", meta(&[("person", "Sarah")]), 1);
        capture_episodic(&mut s, "ep2", meta(&[("person", "Sarah")]), 2);
        capture_episodic(&mut s, "ep3", meta(&[("person", "Sarah")]), 3);
        let report = run_consolidation_pass(&mut s, &c, Default::default(), 100);
        assert_eq!(report.new_semantic.len(), 1);
        assert_eq!(report.new_semantic[0].kind, MemoryType::Semantic);
    }
}
