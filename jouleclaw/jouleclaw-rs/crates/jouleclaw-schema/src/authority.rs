//! Authority scoring (spec §5.4 — deferred to v1, reconstructed here).
//!
//! The v6 spec carries `AuthorityRecord` forward from v1 unchanged but
//! does not redefine its shape. This module reconstructs it from
//! how it's referenced throughout v6:
//!
//! - `KnowledgeAxes.source_authority_tier: int 1..=4`
//!   - 1 = primary peer-reviewed
//!   - 2 = secondary authoritative
//!   - 3 = tertiary aggregator
//!   - 4 = community / unverified
//! - `ValidAnswerModel.authority_violations(plan, items, authority)`
//!   takes a list of authority records and checks them against
//!   `plan.constraints.minimum_authority_tier`.
//! - Section 5.4 describes the scorer as "multidimensional"; we model
//!   that as a per-source record carrying the tier plus
//!   sub-dimensions that justify it.
//!
//! When the v1 spec surfaces, fields here may need adjustment; the
//! `tier` + `source_id` core is stable.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::common::Metadata;

/// Authority tier ladder. Matches the integers used by
/// [`crate::knowledge_axes::KnowledgeAxes::source_authority_tier`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum AuthorityTier {
    /// 1 — peer-reviewed primary source, official record, structured
    /// authority (e.g. Wikidata canonical statement, PubMed indexed
    /// study, government registry).
    Primary,
    /// 2 — authoritative secondary (e.g. major reference work,
    /// recognized encyclopedia article, vetted curated dataset).
    Secondary,
    /// 3 — tertiary aggregator (e.g. mainstream news write-up of a
    /// primary finding).
    Tertiary,
    /// 4 — community or unverified content (e.g. forum post, social
    /// media, blog).
    Community,
}

impl AuthorityTier {
    pub fn as_u8(self) -> u8 {
        match self {
            Self::Primary => 1,
            Self::Secondary => 2,
            Self::Tertiary => 3,
            Self::Community => 4,
        }
    }

    pub fn from_u8(n: u8) -> Option<Self> {
        match n {
            1 => Some(Self::Primary),
            2 => Some(Self::Secondary),
            3 => Some(Self::Tertiary),
            4 => Some(Self::Community),
            _ => None,
        }
    }

    /// True iff `self` meets the minimum required (lower number = higher).
    pub fn meets(self, minimum: AuthorityTier) -> bool {
        self.as_u8() <= minimum.as_u8()
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AuthorityRecord {
    pub schema_version: String,
    pub record_id: Uuid,
    /// Matches `RetrievedItem.source_id`.
    pub source_id: String,
    pub tier: AuthorityTier,
    /// Sub-dimensions that justify the tier. Names follow the spec's
    /// "multidimensional" framing: domain reputation, peer review
    /// status, recency of last verification, citation count.
    /// Values in 0.0..=1.0 where higher = better.
    #[serde(default)]
    pub dimensions: std::collections::BTreeMap<String, f64>,
    /// When this record was assigned its current tier.
    pub assessed_at: DateTime<Utc>,
    /// Free-text rationale supporting the tier assignment.
    #[serde(default)]
    pub rationale: Option<String>,
    #[serde(default)]
    pub metadata: Metadata,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tier_ladder_ordering() {
        assert!(AuthorityTier::Primary.meets(AuthorityTier::Primary));
        assert!(AuthorityTier::Primary.meets(AuthorityTier::Tertiary));
        assert!(!AuthorityTier::Community.meets(AuthorityTier::Primary));
        assert!(AuthorityTier::Secondary.meets(AuthorityTier::Tertiary));
    }

    #[test]
    fn u8_roundtrip() {
        for n in 1..=4u8 {
            let t = AuthorityTier::from_u8(n).unwrap();
            assert_eq!(t.as_u8(), n);
        }
        assert!(AuthorityTier::from_u8(0).is_none());
        assert!(AuthorityTier::from_u8(5).is_none());
    }

    #[test]
    fn roundtrips_through_json() {
        let r = AuthorityRecord {
            schema_version: "2.0".into(),
            record_id: Uuid::new_v4(),
            source_id: "wikidata".into(),
            tier: AuthorityTier::Primary,
            dimensions: [("peer_review".to_string(), 1.0)].into(),
            assessed_at: Utc::now(),
            rationale: Some("canonical structured KB".into()),
            metadata: Default::default(),
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: AuthorityRecord = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }
}
