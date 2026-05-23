//! Content-addressed receipt for a resolved response.
//!
//! The receipt is the BLAKE3 hash of the CBOR-encoded tuple
//! `(query_id, stage, joule_cost, payload)`. It is the proof that a
//! particular stage produced a particular answer at a particular energy cost.

use serde::{Deserialize, Serialize};

use crate::{JouleCost, QueryId, Stage};

/// A 32-byte content-addressed receipt.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct Receipt(pub [u8; 32]);

impl Receipt {
    /// Compute the receipt over the canonical fields.
    pub fn compute(
        query_id: &QueryId,
        stage: &Stage,
        joule_cost: &JouleCost,
        payload: &str,
    ) -> Self {
        // CBOR provides a canonical byte sequence; BLAKE3 is the content hash.
        let bytes = serde_cbor::to_vec(&(query_id, stage, joule_cost, payload))
            .expect("receipt fields are always serializable");
        Self(*blake3::hash(&bytes).as_bytes())
    }

    /// Hex encoding.
    pub fn to_hex(&self) -> String {
        let mut out = String::with_capacity(64);
        for b in &self.0 {
            out.push_str(&format!("{b:02x}"));
        }
        out
    }

    /// Verify a receipt against a candidate tuple.
    pub fn verify(
        &self,
        query_id: &QueryId,
        stage: &Stage,
        joule_cost: &JouleCost,
        payload: &str,
    ) -> bool {
        &Self::compute(query_id, stage, joule_cost, payload) == self
    }
}

impl std::fmt::Display for Receipt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.to_hex())
    }
}

#[cfg(test)]
mod tests {
    use crate::{JouleCost, JouleSource, Query, Response, Stage};

    #[test]
    fn response_roundtrip() {
        let q = Query::new("what is 2+2");
        let r = Response::new(
            q.id,
            "4".to_string(),
            Stage::Cache,
            JouleCost {
                microjoules: 12,
                source: JouleSource::Measured,
            },
        );
        let bytes = serde_cbor::to_vec(&r).unwrap();
        let r2: Response = serde_cbor::from_slice(&bytes).unwrap();
        assert_eq!(r.query_id, r2.query_id);
        assert_eq!(r.stage, r2.stage);
        assert_eq!(r.payload, r2.payload);
        assert_eq!(r.joule_cost, r2.joule_cost);
        assert_eq!(r.receipt, r2.receipt);
        assert!(r2
            .receipt
            .verify(&r2.query_id, &r2.stage, &r2.joule_cost, &r2.payload));
    }

    #[test]
    fn receipt_detects_tampering() {
        let q = Query::new("hello");
        let r = Response::new(q.id, "hi".to_string(), Stage::Cache, JouleCost::zero());
        // Tamper with the payload — receipt should reject.
        assert!(!r.receipt.verify(&r.query_id, &r.stage, &r.joule_cost, "HI"));
        // Tamper with the stage — receipt should reject.
        assert!(!r
            .receipt
            .verify(&r.query_id, &Stage::Neural, &r.joule_cost, &r.payload));
    }

    #[test]
    fn query_id_is_content_addressed() {
        let a = Query::new("hello");
        let b = Query::new("hello");
        let c = Query::new("HELLO");
        assert_eq!(a.id, b.id);
        assert_ne!(a.id, c.id);
    }
}
