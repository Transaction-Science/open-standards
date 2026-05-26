//! Anytime interruption protocol (spec §3.7, §8.2).
//!
//! Every long-running component produces an [`AnytimeResult`] that
//! can be queried for its current-best answer. The orchestrator
//! exposes the same shape so the user can interrupt the system at any
//! point and receive what's available with an honest status flag.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::common::Metadata;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum CompletionStatus {
    NotStarted,
    InProgress,
    Complete,
    Interrupted,
    Failed,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AnytimeResult {
    pub schema_version: String,
    pub task_id: Uuid,
    pub status: CompletionStatus,
    /// Component-specific best-so-far. Opaque to the orchestrator;
    /// each component documents its own shape here.
    #[serde(default)]
    pub current_best: Option<serde_json::Value>,
    /// 0.0..=1.0.
    pub completion_fraction: f64,
    pub elapsed_ms: u64,
    #[serde(default)]
    pub estimated_remaining_ms: Option<u64>,
    /// Whether the component can be interrupted right now without
    /// data loss.
    pub interrupt_safe: bool,
    #[serde(default)]
    pub metadata: Metadata,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrips_through_json() {
        let a = AnytimeResult {
            schema_version: "2.0".into(),
            task_id: Uuid::new_v4(),
            status: CompletionStatus::InProgress,
            current_best: Some(serde_json::json!({"items": 3})),
            completion_fraction: 0.5,
            elapsed_ms: 800,
            estimated_remaining_ms: Some(800),
            interrupt_safe: true,
            metadata: Default::default(),
        };
        let json = serde_json::to_string(&a).unwrap();
        let back: AnytimeResult = serde_json::from_str(&json).unwrap();
        assert_eq!(a, back);
    }
}
