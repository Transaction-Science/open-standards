//! Response — the output of the cascade.

use serde::{Deserialize, Serialize};

use crate::{JouleCost, QueryId, Receipt, Stage};

/// A response produced by exactly one stage of the cascade.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Response {
    /// The query this answers.
    pub query_id: QueryId,
    /// Response payload (string today; binary/multimodal later).
    pub payload: String,
    /// Which stage resolved the query.
    pub stage: Stage,
    /// Energy cost attributed to this resolution.
    pub joule_cost: JouleCost,
    /// Content-addressed receipt over `(query_id, stage, joule_cost, payload)`.
    pub receipt: Receipt,
}

impl Response {
    /// Build a `Response`, computing the receipt over the fields.
    pub fn new(query_id: QueryId, payload: String, stage: Stage, joule_cost: JouleCost) -> Self {
        let receipt = Receipt::compute(&query_id, &stage, &joule_cost, &payload);
        Self {
            query_id,
            payload,
            stage,
            joule_cost,
            receipt,
        }
    }
}
