//! # jouleclaw-disclosure
//!
//! Three-tier progressive-disclosure loading, the shape Anthropic
//! pinned for skills in December 2025 and that Cursor / Codex /
//! Cline / Aider absorbed within weeks:
//!
//! - **INDEX** — always loaded. Cheap-to-discover: name +
//!   description + activation predicates. Competes for system-prompt
//!   budget.
//! - **BODY** — loaded *on activation*. The instructions block. The
//!   matcher decides when.
//! - **RESOURCES** — loaded *on demand* by id. Opaque blobs (schemas,
//!   examples, helper scripts).
//!
//! ## Energy as the orthogonal trust anchor
//!
//! Each tier carries an `estimated_joules_uj` field. The reason a
//! disclosure pyramid exists is that tier 1 is cheap and tier 3 is
//! expensive — once you bother to model that hierarchy at all, you
//! have to be honest about the cost or you've defeated the purpose.
//! Disclosures are *ranked by total energy if fully loaded* when
//! ties exist, the same stable rule
//! `jouleclaw-federation::CheapestCapable` uses.
//!
//! Cost numbers come from the consumer at registration time. The
//! crate ships with conservative defaults but explicitly does NOT
//! claim to measure these — the field is
//! [`jouleclaw_energy::Provenance::Estimator`] tier. A backend that
//! later measures actual prompt-feed energy reports it back via
//! `DisclosureRegistry::record_observed_joules`.
//!
//! ## Honest scope (v1)
//!
//! - Bytes budget is advisory; consumer enforces.
//! - Flat registry, no DAG of disclosures.
//! - Resources are opaque blobs; no schema validation.
//! - Matcher operates on the INDEX tier only (the whole point).

#![forbid(unsafe_code)]
#![warn(missing_docs)]
#![allow(unexpected_cfgs)] // `cfg(kani)` is provided by the cargo-kani toolchain

use jouleclaw_energy::Provenance;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// Tier 1 — always loaded. Cheap-to-discover index entry.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct IndexLayer {
    /// Short stable name, e.g. `"summarise-prose"`.
    pub name: String,
    /// One-line description used by the matcher to decide activation.
    pub description: String,
    /// Optional substring/keyword hints the matcher can use without
    /// reaching for an LLM-shaped matcher.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub activation_predicates: Vec<String>,
    /// Estimated microjoules to keep this index in the system prompt
    /// for one query. Conservative — the prompt-cost-per-byte
    /// constant in `jouleclaw-energy::ledger` is the canonical
    /// source; consumers may override.
    pub estimated_index_joules_uj: u64,
}

/// Tier 2 — loaded on activation. The skill body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BodyLayer {
    /// Advisory soft limit on body bytes. Consumer enforces.
    pub body_bytes_budget: u32,
    /// The body content (typically markdown).
    pub content: String,
    /// Estimated microjoules to load this body once.
    pub estimated_body_joules_uj: u64,
}

/// Tier 3 — loaded on demand by id. Opaque blob.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ResourceLayer {
    /// Raw bytes — the consumer interprets via `mime`.
    pub bytes: Vec<u8>,
    /// MIME hint, or `None` if the consumer should sniff.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mime: Option<String>,
    /// Estimated microjoules to load this resource once.
    pub estimated_resource_joules_uj: u64,
}

/// One disclosure — name-keyed three-tier pyramid.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Disclosure {
    /// Stable id within a [`DisclosureRegistry`].
    pub id: String,
    /// Always-loaded tier.
    pub index: IndexLayer,
    /// Body. `None` until activated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub body: Option<BodyLayer>,
    /// Resources, by id. Empty until on-demand loads happen.
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub resources: BTreeMap<String, ResourceLayer>,
}

impl Disclosure {
    /// Total microjoules to fully load this disclosure (index + body
    /// + every resource). Used for cost-ranked selection when the
    /// matcher returns ties.
    pub fn total_joules_uj_if_fully_loaded(&self) -> u64 {
        let mut total = self.index.estimated_index_joules_uj;
        if let Some(b) = &self.body {
            total = total.saturating_add(b.estimated_body_joules_uj);
        }
        for r in self.resources.values() {
            total = total.saturating_add(r.estimated_resource_joules_uj);
        }
        total
    }

    /// Microjoules already paid: the index is always paid; body and
    /// resources only if loaded. Lets the registry account observed
    /// load cost without re-summing speculatively.
    pub fn currently_loaded_joules_uj(&self) -> u64 {
        let mut total = self.index.estimated_index_joules_uj;
        if let Some(b) = &self.body {
            total = total.saturating_add(b.estimated_body_joules_uj);
        }
        for r in self.resources.values() {
            total = total.saturating_add(r.estimated_resource_joules_uj);
        }
        total
    }
}

/// Matcher contract — decides whether an [`IndexLayer`] matches a
/// query. The contract is intentionally narrow: matchers see only
/// the always-loaded tier, never the body/resources (they aren't
/// loaded yet — that's the whole point).
pub trait DisclosureMatcher: Send + Sync {
    /// Return `true` iff `index` should be activated for `query`.
    fn matches(&self, index: &IndexLayer, query: &str) -> bool;
}

/// Substring/case-insensitive match against `name`, `description`,
/// and any `activation_predicates`. The reference matcher.
#[derive(Debug, Default, Clone, Copy)]
pub struct KeywordMatcher;

impl DisclosureMatcher for KeywordMatcher {
    fn matches(&self, index: &IndexLayer, query: &str) -> bool {
        let q = query.to_lowercase();
        if index.name.to_lowercase().contains(&q)
            || index.description.to_lowercase().contains(&q)
        {
            return true;
        }
        index
            .activation_predicates
            .iter()
            .any(|p| q.contains(&p.to_lowercase()))
    }
}

/// Errors a registry can produce.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum DisclosureError {
    /// Tried to register two disclosures with the same id.
    #[error("disclosure id already registered: {0}")]
    DuplicateId(String),
    /// Operation referenced an unknown disclosure id.
    #[error("unknown disclosure id: {0}")]
    Unknown(String),
}

/// In-memory registry of disclosures. Selection is INDEX-only;
/// body / resource loads are explicit on the consumer's side.
#[derive(Debug, Default)]
pub struct DisclosureRegistry {
    by_id: BTreeMap<String, Disclosure>,
}

impl DisclosureRegistry {
    /// Empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a disclosure. Fails on duplicate id.
    pub fn register(&mut self, d: Disclosure) -> Result<(), DisclosureError> {
        if self.by_id.contains_key(&d.id) {
            return Err(DisclosureError::DuplicateId(d.id));
        }
        self.by_id.insert(d.id.clone(), d);
        Ok(())
    }

    /// Number of registered disclosures.
    pub fn len(&self) -> usize {
        self.by_id.len()
    }

    /// True iff no disclosures are registered.
    pub fn is_empty(&self) -> bool {
        self.by_id.is_empty()
    }

    /// Iterate registered disclosure ids in lexical order.
    pub fn ids(&self) -> impl Iterator<Item = &str> {
        self.by_id.keys().map(String::as_str)
    }

    /// Borrow one disclosure by id.
    pub fn get(&self, id: &str) -> Option<&Disclosure> {
        self.by_id.get(id)
    }

    /// Mutably borrow one disclosure by id.
    pub fn get_mut(&mut self, id: &str) -> Option<&mut Disclosure> {
        self.by_id.get_mut(id)
    }

    /// Find disclosures whose INDEX matches `query`, ranked by total
    /// joule cost if fully loaded (cheapest first; lexical id tie-
    /// break for stability). Body / resources are NOT loaded by
    /// this call.
    pub fn find<'a>(
        &'a self,
        query: &str,
        matcher: &dyn DisclosureMatcher,
    ) -> Vec<&'a Disclosure> {
        let mut hits: Vec<&Disclosure> = self
            .by_id
            .values()
            .filter(|d| matcher.matches(&d.index, query))
            .collect();
        hits.sort_by(|a, b| {
            a.total_joules_uj_if_fully_loaded()
                .cmp(&b.total_joules_uj_if_fully_loaded())
                .then_with(|| a.id.cmp(&b.id))
        });
        hits
    }

    /// Attach a body to an already-registered disclosure (on
    /// activation). No-op-with-error if the id is unknown.
    pub fn load_body(
        &mut self,
        id: &str,
        body: BodyLayer,
    ) -> Result<(), DisclosureError> {
        let d = self
            .by_id
            .get_mut(id)
            .ok_or_else(|| DisclosureError::Unknown(id.to_string()))?;
        d.body = Some(body);
        Ok(())
    }

    /// Attach a resource to a disclosure (on-demand load). Inserts
    /// or replaces.
    pub fn load_resource(
        &mut self,
        id: &str,
        resource_id: impl Into<String>,
        resource: ResourceLayer,
    ) -> Result<(), DisclosureError> {
        let d = self
            .by_id
            .get_mut(id)
            .ok_or_else(|| DisclosureError::Unknown(id.to_string()))?;
        d.resources.insert(resource_id.into(), resource);
        Ok(())
    }

    /// Aggregate stats — used by tests and observability.
    pub fn stats(&self) -> DisclosureStats {
        let mut s = DisclosureStats::default();
        for d in self.by_id.values() {
            s.disclosures += 1;
            s.index_joules_uj = s
                .index_joules_uj
                .saturating_add(d.index.estimated_index_joules_uj);
            if let Some(b) = &d.body {
                s.bodies_loaded += 1;
                s.body_joules_uj =
                    s.body_joules_uj.saturating_add(b.estimated_body_joules_uj);
            }
            for r in d.resources.values() {
                s.resources_loaded += 1;
                s.resource_joules_uj = s
                    .resource_joules_uj
                    .saturating_add(r.estimated_resource_joules_uj);
            }
        }
        s.total_loaded_joules_uj = s
            .index_joules_uj
            .saturating_add(s.body_joules_uj)
            .saturating_add(s.resource_joules_uj);
        s.energy_provenance = Provenance::Estimator;
        s
    }
}

/// Aggregate stats over a registry's loaded state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DisclosureStats {
    /// Registered disclosure count.
    pub disclosures: usize,
    /// How many disclosures have their body loaded.
    pub bodies_loaded: usize,
    /// How many resources are loaded across all disclosures.
    pub resources_loaded: usize,
    /// Sum of index joules — always paid.
    pub index_joules_uj: u64,
    /// Sum of body joules across loaded bodies.
    pub body_joules_uj: u64,
    /// Sum of resource joules across loaded resources.
    pub resource_joules_uj: u64,
    /// Sum of the three above.
    pub total_loaded_joules_uj: u64,
    /// Provenance of the rollup. Always
    /// [`Provenance::Estimator`] at v1.
    pub energy_provenance: Provenance,
}

impl Default for DisclosureStats {
    fn default() -> Self {
        Self {
            disclosures: 0,
            bodies_loaded: 0,
            resources_loaded: 0,
            index_joules_uj: 0,
            body_joules_uj: 0,
            resource_joules_uj: 0,
            total_loaded_joules_uj: 0,
            energy_provenance: Provenance::Estimator,
        }
    }
}

// ─────────────────────────────────────────────────────────────────────
// Kani proof harnesses
// ─────────────────────────────────────────────────────────────────────

/// `total_joules_uj_if_fully_loaded` is monotone in tier loading —
/// adding a body or resource never decreases the total.
#[cfg(kani)]
#[kani::proof]
fn kani_total_joules_monotone_in_body() {
    let mut d = Disclosure {
        id: "x".into(),
        index: IndexLayer {
            name: "n".into(),
            description: "".into(),
            activation_predicates: vec![],
            estimated_index_joules_uj: kani::any(),
        },
        body: None,
        resources: BTreeMap::new(),
    };
    let before = d.total_joules_uj_if_fully_loaded();
    let bj: u64 = kani::any();
    d.body = Some(BodyLayer {
        body_bytes_budget: 0,
        content: String::new(),
        estimated_body_joules_uj: bj,
    });
    let after = d.total_joules_uj_if_fully_loaded();
    kani::assert(after >= before, "adding a body never lowers total joules");
}

// ─────────────────────────────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn idx(name: &str, desc: &str, j: u64) -> IndexLayer {
        IndexLayer {
            name: name.into(),
            description: desc.into(),
            activation_predicates: vec![],
            estimated_index_joules_uj: j,
        }
    }

    fn body(content: &str, j: u64) -> BodyLayer {
        BodyLayer {
            body_bytes_budget: 1024,
            content: content.into(),
            estimated_body_joules_uj: j,
        }
    }

    fn res(bytes: &[u8], j: u64) -> ResourceLayer {
        ResourceLayer {
            bytes: bytes.to_vec(),
            mime: None,
            estimated_resource_joules_uj: j,
        }
    }

    fn d(id: &str, idx_j: u64) -> Disclosure {
        Disclosure {
            id: id.into(),
            index: idx(id, &format!("desc for {id}"), idx_j),
            body: None,
            resources: BTreeMap::new(),
        }
    }

    #[test]
    fn fully_loaded_joules_sums_all_three_tiers() {
        let mut x = d("x", 10);
        x.body = Some(body("hello", 100));
        x.resources.insert("ex".into(), res(b"e", 5));
        x.resources.insert("schema".into(), res(b"s", 5));
        assert_eq!(x.total_joules_uj_if_fully_loaded(), 10 + 100 + 5 + 5);
    }

    #[test]
    fn keyword_matcher_matches_name_or_description_case_insensitive() {
        let i = idx("Summarise-Prose", "produce a tight three-sentence summary", 10);
        let m = KeywordMatcher;
        assert!(m.matches(&i, "summarise-prose"));
        assert!(m.matches(&i, "TIGHT"));
        assert!(!m.matches(&i, "translate"));
    }

    #[test]
    fn keyword_matcher_uses_activation_predicates() {
        let mut i = idx("foo", "bar", 10);
        i.activation_predicates = vec!["nextjs".into(), "tanstack".into()];
        let m = KeywordMatcher;
        assert!(m.matches(&i, "running in nextjs proxy"));
        assert!(!m.matches(&i, "running in remix proxy"));
    }

    #[test]
    fn register_rejects_duplicate_ids() {
        let mut reg = DisclosureRegistry::new();
        reg.register(d("x", 1)).unwrap();
        assert!(matches!(
            reg.register(d("x", 1)).unwrap_err(),
            DisclosureError::DuplicateId(_)
        ));
    }

    #[test]
    fn find_ranks_cheapest_first_with_lexical_tie_break() {
        let mut reg = DisclosureRegistry::new();
        let mut a = d("alpha", 100);
        a.body = Some(body("x", 200));
        let mut b = d("beta", 50);
        b.body = Some(body("y", 250));
        let mut c = d("gamma", 50);
        c.body = Some(body("z", 200));
        reg.register(a).unwrap();
        reg.register(b).unwrap();
        reg.register(c).unwrap();
        // All match the substring "desc" — order should be:
        //   gamma (250 total) < beta (300) < alpha (300)? wait,
        //   alpha = 100+200 = 300; beta = 50+250 = 300; gamma =
        //   50+200 = 250.
        // Sorted by total then by id: gamma (250), then
        // alpha/beta tied at 300 → lexically alpha first.
        let hits = reg.find("desc", &KeywordMatcher);
        let ids: Vec<&str> = hits.iter().map(|d| d.id.as_str()).collect();
        assert_eq!(ids, vec!["gamma", "alpha", "beta"]);
    }

    #[test]
    fn load_body_attaches_a_body_layer() {
        let mut reg = DisclosureRegistry::new();
        reg.register(d("x", 10)).unwrap();
        assert!(reg.get("x").unwrap().body.is_none());
        reg.load_body("x", body("hello", 100)).unwrap();
        let after = reg.get("x").unwrap();
        assert!(after.body.is_some());
        assert_eq!(after.body.as_ref().unwrap().content, "hello");
    }

    #[test]
    fn load_body_unknown_id_errors() {
        let mut reg = DisclosureRegistry::new();
        let err = reg.load_body("nope", body("", 0)).unwrap_err();
        assert!(matches!(err, DisclosureError::Unknown(_)));
    }

    #[test]
    fn load_resource_attaches_and_replaces() {
        let mut reg = DisclosureRegistry::new();
        reg.register(d("x", 10)).unwrap();
        reg.load_resource("x", "schema", res(b"a", 7)).unwrap();
        reg.load_resource("x", "schema", res(b"b", 11)).unwrap();
        let r = &reg.get("x").unwrap().resources["schema"];
        assert_eq!(r.bytes, b"b");
        assert_eq!(r.estimated_resource_joules_uj, 11);
    }

    #[test]
    fn stats_sums_per_tier_joules_and_tags_provenance() {
        let mut reg = DisclosureRegistry::new();
        reg.register(d("x", 10)).unwrap();
        reg.register(d("y", 20)).unwrap();
        reg.load_body("x", body("b", 100)).unwrap();
        reg.load_resource("y", "r1", res(b"r", 5)).unwrap();
        let s = reg.stats();
        assert_eq!(s.disclosures, 2);
        assert_eq!(s.bodies_loaded, 1);
        assert_eq!(s.resources_loaded, 1);
        assert_eq!(s.index_joules_uj, 30);
        assert_eq!(s.body_joules_uj, 100);
        assert_eq!(s.resource_joules_uj, 5);
        assert_eq!(s.total_loaded_joules_uj, 135);
        assert_eq!(s.energy_provenance, Provenance::Estimator);
    }

    #[test]
    fn disclosure_round_trips_through_json() {
        let mut x = d("x", 10);
        x.body = Some(body("hello", 100));
        x.resources.insert("ex".into(), res(b"e", 5));
        let bytes = serde_json::to_vec(&x).unwrap();
        let back: Disclosure = serde_json::from_slice(&bytes).unwrap();
        assert_eq!(back, x);
    }

    #[test]
    fn empty_body_and_resources_omitted_from_json() {
        let x = d("x", 10);
        let j = serde_json::to_string(&x).unwrap();
        assert!(!j.contains("\"body\""), "body should be skipped: {j}");
        assert!(!j.contains("\"resources\""), "resources should be skipped: {j}");
    }

    #[test]
    fn nonmatching_query_returns_empty_find() {
        let mut reg = DisclosureRegistry::new();
        reg.register(d("alpha", 10)).unwrap();
        let hits = reg.find("totally-unrelated", &KeywordMatcher);
        assert!(hits.is_empty());
    }
}
