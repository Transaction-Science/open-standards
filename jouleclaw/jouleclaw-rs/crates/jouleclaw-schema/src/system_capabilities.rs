//! The self-model (spec §3.6, §4.3).
//!
//! A queryable description of what the system can currently do. Read
//! by the planner before plan production, written by the orchestrator
//! after observing performance.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::common::Metadata;
use crate::query_plan::Modality;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum CapabilityStatus {
    Healthy,
    Degraded,
    Unavailable,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RetrieverCapability {
    pub retriever_id: String,
    pub status: CapabilityStatus,
    /// Domain tags this retriever serves.
    pub domains_covered: Vec<String>,
    pub modalities_supported: Vec<Modality>,
    pub typical_latency_ms: u32,
    pub p99_latency_ms: u32,
    /// Observed success rate over recent calls (0.0..=1.0).
    pub success_rate_recent: f64,
    #[serde(default)]
    pub last_failure_at: Option<DateTime<Utc>>,
    #[serde(default)]
    pub known_limitations: Vec<String>,

    // v5 additions (spec §3.6): which knowledge axes this retriever
    // can populate, and with what authority.
    #[serde(default)]
    pub populates_valid_time: bool,
    #[serde(default)]
    pub populates_transaction_time: bool,
    #[serde(default)]
    pub populates_granularity: bool,
    #[serde(default)]
    pub populates_scope: bool,
    #[serde(default)]
    pub populates_certainty: bool,
    #[serde(default = "default_populates_provenance")]
    pub populates_provenance: bool,
    pub authority_tier: u8,
}

fn default_populates_provenance() -> bool {
    true
}

/// v3.1 addition (spec §3.6, §5.7): energy + capability profile per
/// reasoner architecture family so the planner can prefer the family
/// appropriate for the query class.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ReasonerCapability {
    pub reasoner_id: String,
    /// `"transformer" | "ssm" | "moe" | "jepa" | "hybrid" |
    /// "encoder_only"`.
    pub architecture_family: String,
    /// e.g. `"state-spaces/mamba-3-7b"`,
    /// `"deepseek-ai/DeepSeek-V4-Flash"`.
    pub model_identifier: String,
    pub optimal_query_classes: Vec<String>,
    /// Measured joules per query, keyed by query class name.
    pub typical_joules_per_query: BTreeMap<String, f64>,
    /// SSM/Transformer crossover length when present (e.g. Mamba-3
    /// becomes preferable above 8K context).
    #[serde(default)]
    pub crossover_context_length: Option<u32>,
    /// MoE: fraction of total params active per token (0.0..=1.0).
    #[serde(default)]
    pub activation_ratio: Option<f64>,
    pub total_parameters: u64,
    /// Equals `total_parameters` for dense models.
    pub active_parameters: u64,
    pub status: CapabilityStatus,
    pub license: String,
    #[serde(default)]
    pub metadata: Metadata,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct SystemCapabilities {
    pub schema_version: String,
    pub snapshot_timestamp: DateTime<Utc>,
    pub retrievers: Vec<RetrieverCapability>,
    #[serde(default)]
    pub reasoners: Vec<ReasonerCapability>,
    pub overall_status: CapabilityStatus,
    #[serde(default)]
    pub degradation_notes: Vec<String>,
    #[serde(default)]
    pub metadata: Metadata,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_through_json() {
        let s = SystemCapabilities {
            schema_version: "5.0".into(),
            snapshot_timestamp: Utc::now(),
            retrievers: vec![RetrieverCapability {
                retriever_id: "wikidata".into(),
                status: CapabilityStatus::Healthy,
                domains_covered: vec!["geography".into(), "people".into()],
                modalities_supported: vec![Modality::Text, Modality::Structured],
                typical_latency_ms: 800,
                p99_latency_ms: 7000,
                success_rate_recent: 0.97,
                last_failure_at: None,
                known_limitations: vec![],
                populates_valid_time: true,
                populates_transaction_time: true,
                populates_granularity: false,
                populates_scope: true,
                populates_certainty: false,
                populates_provenance: true,
                authority_tier: 1,
            }],
            reasoners: vec![],
            overall_status: CapabilityStatus::Healthy,
            degradation_notes: vec![],
            metadata: Default::default(),
        };
        let json = serde_json::to_string(&s).unwrap();
        let back: SystemCapabilities = serde_json::from_str(&json).unwrap();
        assert_eq!(s, back);
    }

    #[test]
    fn populates_provenance_defaults_true() {
        let json = serde_json::json!({
            "retriever_id": "x",
            "status": "healthy",
            "domains_covered": [],
            "modalities_supported": [],
            "typical_latency_ms": 1,
            "p99_latency_ms": 1,
            "success_rate_recent": 1.0,
            "authority_tier": 3
        });
        let r: RetrieverCapability = serde_json::from_value(json).unwrap();
        assert!(r.populates_provenance);
    }
}
