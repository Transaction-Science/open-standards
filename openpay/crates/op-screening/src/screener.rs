//! Top-level screening orchestration.
//!
//! [`Screener`] is the type operators actually wire into their payment
//! flow. It owns the index, the config (threshold, max results, list
//! filter, optional DOB / address gating), and an [`crate::audit::AuditLog`]
//! that every screen call appends to. The result classification
//! ([`ScreenDecision::Clear`] / [`Hit`](ScreenDecision::Hit) /
//! [`AmbiguousNeedsReview`](ScreenDecision::AmbiguousNeedsReview))
//! drives the caller's downstream action.

use std::collections::HashSet;
use std::sync::Mutex;

use chrono::NaiveDate;
use serde::{Deserialize, Serialize};

use crate::audit::AuditLog;
use crate::error::Result;
use crate::lists::{Address, Identification, SanctionsList};
use crate::matching::{MatchScore, screen};
use crate::storage::SanctionsIndex;

/// Tunable knobs for [`Screener`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenerConfig {
    /// Combined-score threshold below which a candidate is dropped.
    /// Default in production deployments: `0.85`.
    pub threshold: f32,
    /// Hard cap on the number of hits returned to the caller.
    pub max_results: usize,
    /// If true and the request carries a DOB, hits whose entity DOB
    /// disagrees by more than ±1 year are demoted to
    /// [`ScreenDecision::AmbiguousNeedsReview`].
    pub screen_dob: bool,
    /// If true, address country mismatch demotes the result similarly.
    pub screen_address: bool,
    /// If non-empty, only consider hits whose `source_list` is in the set.
    pub watchlist_filter: HashSet<SanctionsList>,
    /// Above-`threshold` but below `hit_threshold` lands in
    /// `AmbiguousNeedsReview`. Defaults to `0.95`.
    pub hit_threshold: f32,
}

impl Default for ScreenerConfig {
    fn default() -> Self {
        Self {
            threshold: 0.85,
            max_results: 25,
            screen_dob: false,
            screen_address: false,
            watchlist_filter: HashSet::new(),
            hit_threshold: 0.95,
        }
    }
}

/// What the operator wants to screen.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ScreenRequest {
    /// Name to screen. Required.
    pub name: String,
    /// Date of birth, if known. Used when `screen_dob` is on.
    pub dob: Option<NaiveDate>,
    /// Address, if known. Used when `screen_address` is on.
    pub address: Option<Address>,
    /// Other identifications to cross-screen (passport, national ID).
    pub additional_ids: Vec<Identification>,
}

/// Three-way result.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ScreenDecision {
    /// No hits at or above threshold.
    Clear,
    /// At least one hit at or above `hit_threshold`.
    Hit,
    /// Hits between `threshold` and `hit_threshold`. Human review required.
    AmbiguousNeedsReview,
}

/// The output of [`Screener::screen`].
#[derive(Debug, Clone)]
pub struct ScreenResult {
    /// All hits that survived filtering, sorted desc by score.
    pub hits: Vec<MatchScore>,
    /// Top-line decision.
    pub decision: ScreenDecision,
    /// Audit-log row id that recorded this call.
    pub audit_id: String,
}

/// Stateful screening engine.
///
/// Wrap the audit log in a `Mutex` because every screen call is a
/// write into a chained, signed log; we need exclusive access for
/// the duration of the append. The index itself is read-only.
pub struct Screener {
    index: SanctionsIndex,
    config: ScreenerConfig,
    audit_log: Mutex<AuditLog>,
}

impl Screener {
    /// Build a new screener.
    #[must_use]
    pub fn new(index: SanctionsIndex, config: ScreenerConfig, audit_log: AuditLog) -> Self {
        Self {
            index,
            config,
            audit_log: Mutex::new(audit_log),
        }
    }

    /// Borrow the audit log (e.g. for verification).
    pub fn with_audit_log<R>(&self, f: impl FnOnce(&AuditLog) -> R) -> R {
        let log = self.audit_log.lock().expect("audit-log mutex poisoned");
        f(&log)
    }

    /// Active configuration.
    #[must_use]
    pub fn config(&self) -> &ScreenerConfig {
        &self.config
    }

    /// How many entities the index covers.
    #[must_use]
    pub fn index_len(&self) -> usize {
        self.index.len()
    }

    /// Screen a single request.
    ///
    /// Always appends to the audit log, even on `Clear` — operators
    /// rely on the negative-evidence trail when regulators inspect.
    pub async fn screen(&self, request: &ScreenRequest) -> Result<ScreenResult> {
        let raw_hits = screen(&request.name, &self.index, self.config.threshold);

        // Optional list filter.
        let filtered: Vec<MatchScore> = if self.config.watchlist_filter.is_empty() {
            raw_hits
        } else {
            raw_hits
                .into_iter()
                .filter(|h| {
                    self.config
                        .watchlist_filter
                        .contains(&h.entity.source_list)
                })
                .collect()
        };

        // Optional DOB / address discrimination. We don't drop hits —
        // a mismatched DOB on a perfect-name match is still ambiguous;
        // we just demote the per-hit confidence so the overall decision
        // gates into AmbiguousNeedsReview.
        let mut hits = filtered;
        if self.config.screen_dob {
            if let Some(req_dob) = request.dob {
                for h in &mut hits {
                    if let Some(ent_dob) = h.entity.dob {
                        let delta = (ent_dob - req_dob).num_days().abs();
                        if delta > 366 {
                            // ±1 year. Scale the score down by 20%.
                            h.score *= 0.8;
                        }
                    }
                }
            }
        }
        if self.config.screen_address {
            if let Some(addr) = request.address.as_ref() {
                for h in &mut hits {
                    let any_country_match = h
                        .entity
                        .addresses
                        .iter()
                        .any(|a| a.country == addr.country);
                    if !any_country_match && !h.entity.addresses.is_empty() {
                        h.score *= 0.9;
                    }
                }
            }
        }

        hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        hits.truncate(self.config.max_results);

        let decision = if hits.is_empty() {
            ScreenDecision::Clear
        } else if hits[0].score >= self.config.hit_threshold {
            ScreenDecision::Hit
        } else {
            ScreenDecision::AmbiguousNeedsReview
        };

        let audit_id = {
            let mut log = self.audit_log.lock().expect("audit mutex poisoned");
            log.record(
                &request.name,
                &hits,
                match decision {
                    ScreenDecision::Clear => "clear",
                    ScreenDecision::Hit => "hit",
                    ScreenDecision::AmbiguousNeedsReview => "review",
                },
            )?
        };

        Ok(ScreenResult { hits, decision, audit_id })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::audit::AuditLog;
    use crate::lists::{EntityType, SanctionedEntity};
    use chrono::Utc;

    fn ent(id: &str, name: &str) -> SanctionedEntity {
        SanctionedEntity {
            id: id.to_string(),
            name: name.to_string(),
            name_aliases: vec![],
            entity_type: EntityType::Individual,
            dob: None,
            place_of_birth: None,
            addresses: vec![],
            nationalities: vec![],
            identifications: vec![],
            programs: vec![],
            last_updated: Utc::now(),
            source_list: SanctionsList::OfacSdn,
        }
    }

    #[tokio::test]
    async fn clear_when_no_match() {
        let idx = SanctionsIndex::build(vec![ent("1", "John Smith")]);
        let s = Screener::new(idx, ScreenerConfig::default(), AuditLog::new());
        let r = s
            .screen(&ScreenRequest {
                name: "Totally Unrelated Person".into(),
                dob: None,
                address: None,
                additional_ids: vec![],
            })
            .await
            .expect("ok");
        assert_eq!(r.decision, ScreenDecision::Clear);
        assert!(r.hits.is_empty());
    }

    #[tokio::test]
    async fn exact_match_is_hit() {
        let idx = SanctionsIndex::build(vec![ent("1", "John Smith")]);
        let s = Screener::new(idx, ScreenerConfig::default(), AuditLog::new());
        let r = s
            .screen(&ScreenRequest {
                name: "John Smith".into(),
                dob: None,
                address: None,
                additional_ids: vec![],
            })
            .await
            .expect("ok");
        assert_eq!(r.decision, ScreenDecision::Hit);
        assert_eq!(r.hits.len(), 1);
    }
}
