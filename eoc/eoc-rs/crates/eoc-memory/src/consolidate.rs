//! Dream-cycle consolidation: episodic → semantic.
//!
//! Inspired by hippocampus → neocortex replay (Walker & Stickgold
//! 2004), we periodically *consolidate* episodic events into
//! semantic triples. The caller supplies an extractor closure that
//! turns each episode into zero-or-more triples; this module
//! orchestrates the batch and reports what was learned.

use serde::{Deserialize, Serialize};

use crate::episodic::{Episode, EpisodicLog};
use crate::error::{MemoryError, MemoryResult};
use crate::semantic::{SemanticGraph, Triple};

/// Configuration for one consolidation pass.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ConsolidateConfig {
    /// Half-open `[t0, t1)` time window in ms to consolidate over.
    pub window_ms: (u64, u64),
    /// Maximum number of triples to emit (safety cap).
    pub max_triples: usize,
}

impl ConsolidateConfig {
    /// Construct a config with validation.
    pub fn new(window_ms: (u64, u64), max_triples: usize) -> MemoryResult<Self> {
        if window_ms.0 > window_ms.1 {
            return Err(MemoryError::Config("window_ms.0 > window_ms.1".into()));
        }
        if max_triples == 0 {
            return Err(MemoryError::Config("max_triples must be > 0".into()));
        }
        Ok(Self {
            window_ms,
            max_triples,
        })
    }
}

/// Report from one dream-cycle pass.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct ConsolidationReport {
    /// Episodes inspected.
    pub episodes_scanned: usize,
    /// Triples newly asserted (not already present).
    pub triples_asserted: usize,
    /// Triples skipped because the graph already contained them.
    pub triples_duplicate: usize,
}

/// Run one dream-cycle: for every episode in `[t0, t1)`, the
/// `extractor` produces candidate triples; new triples are asserted
/// into the graph, duplicates are counted.
pub fn consolidate<F>(
    log: &EpisodicLog,
    graph: &mut SemanticGraph,
    cfg: &ConsolidateConfig,
    mut extractor: F,
) -> MemoryResult<ConsolidationReport>
where
    F: FnMut(&Episode) -> Vec<Triple>,
{
    let mut report = ConsolidationReport::default();
    let episodes = log.range(cfg.window_ms.0, cfg.window_ms.1);
    report.episodes_scanned = episodes.len();

    for ep in episodes {
        for candidate in extractor(ep) {
            if report.triples_asserted >= cfg.max_triples {
                return Ok(report);
            }
            if graph.contains(&candidate) {
                report.triples_duplicate += 1;
            } else {
                graph.assert(candidate);
                report.triples_asserted += 1;
            }
        }
    }
    Ok(report)
}
