//! Synthetic-identity heuristics.
//!
//! A synthetic identity is fabricated personal information stitched
//! together from real and made-up parts. The classic structural
//! signatures, all visible in a fraud graph:
//!
//! 1. **Freshness** — the identity first appeared very recently.
//! 2. **Thin edges** — few connections, but each used heavily
//!    (high tx_count / degree ratio).
//! 3. **Cross-kind oddity** — email + phone + address all introduced
//!    in the same payment, not built up over time.
//! 4. **Address recycling** — many "different" identities share an
//!    address (mail-drop).
//!
//! We score each vertex independently in `[0, 1]`. The score is a
//! *feature*, not a verdict — the orchestrator combines it with other
//! signals.

use std::collections::HashSet;

use crate::entity::EntityKind;
use crate::graph::{FraudGraph, VertexId};

/// Configuration for [`SyntheticIdentityScorer`].
#[derive(Debug, Clone, Copy)]
pub struct SyntheticIdentityScorer {
    /// Vertex is "fresh" if `now_unix - first_seen` is less than this.
    pub freshness_window_secs: i64,
    /// Address that fans out to more than this many distinct accounts
    /// is flagged as a mail-drop signal.
    pub mail_drop_threshold: u32,
    /// Vertex whose `tx_count / degree` exceeds this is flagged as
    /// over-used relative to its sparsity.
    pub heavy_use_ratio: f32,
}

impl Default for SyntheticIdentityScorer {
    fn default() -> Self {
        Self {
            freshness_window_secs: 60 * 60 * 24 * 14, // 14 days
            mail_drop_threshold: 4,
            heavy_use_ratio: 10.0,
        }
    }
}

impl SyntheticIdentityScorer {
    /// Score a single vertex against `g` at observation time `now_unix`.
    pub fn score(&self, g: &FraudGraph, v: VertexId, now_unix: i64) -> f32 {
        let meta = match g.meta(v) {
            Ok(m) => m,
            Err(_) => return 0.0,
        };
        let nbrs = g.neighbours(v).unwrap_or(&[]);
        let degree = nbrs.len() as u32;

        let mut score = 0.0_f32;

        // Freshness.
        let age = now_unix.saturating_sub(meta.first_seen);
        if age < self.freshness_window_secs {
            // Linear ramp from 1.0 at 0s old → 0.0 at the window edge.
            let frac = 1.0 - (age as f32 / self.freshness_window_secs as f32);
            score += 0.35 * frac.clamp(0.0, 1.0);
        }

        // Heavy use relative to degree.
        if degree > 0 {
            let ratio = meta.tx_count as f32 / degree as f32;
            if ratio > self.heavy_use_ratio {
                let frac = (ratio / (self.heavy_use_ratio * 4.0)).clamp(0.0, 1.0);
                score += 0.25 * frac;
            }
        } else if meta.tx_count > 0 {
            score += 0.25;
        }

        // Mail-drop: address fans out to many distinct Account neighbours.
        if meta.entity.key.kind == EntityKind::Address {
            let mut distinct_accounts: HashSet<VertexId> = HashSet::new();
            for e in nbrs {
                let other = match e.other(v) {
                    Some(o) => o,
                    None => continue,
                };
                if let Ok(om) = g.meta(other) {
                    if om.entity.key.kind == EntityKind::Account {
                        distinct_accounts.insert(other);
                    }
                }
            }
            if (distinct_accounts.len() as u32) >= self.mail_drop_threshold {
                score += 0.4;
            }
        }

        score.clamp(0.0, 1.0)
    }
}
