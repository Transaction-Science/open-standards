//! Evidence the merchant submits to defend a chargeback.
//!
//! Card networks (Visa, Mastercard) define structured evidence
//! categories: shipping proof, customer correspondence, IP / device
//! match, etc. The actual submission is rail-specific; this crate
//! only tracks *references* to evidence — operators store the
//! actual files in their own document system and pass the URLs /
//! object keys here.

use serde::{Deserialize, Serialize};

/// Pointer to a piece of evidence the operator has gathered. The
/// crate doesn't validate the URL or fetch the content — it's a
/// reference for the merchant's UI and for the rail submission
/// step.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EvidenceRef {
    /// What kind of evidence this is. Short normalized string —
    /// `"shipping_tracking"`, `"customer_email"`, `"receipt"`,
    /// `"refund_policy"`, etc. Operators define their taxonomy;
    /// network-side submission maps it to the rail's categories.
    pub kind: String,
    /// Where the file lives. Typically an HTTPS URL or an object
    /// store key (`s3://bucket/key`). We don't constrain the
    /// format.
    pub url: String,
    /// Optional free-form description for the operator's UI.
    pub note: Option<String>,
    /// When the operator attached this evidence (unix epoch seconds).
    pub attached_at_unix_secs: u64,
}

impl EvidenceRef {
    /// Construct a reference with required fields.
    #[must_use]
    pub fn new(
        kind: impl Into<String>,
        url: impl Into<String>,
        attached_at_unix_secs: u64,
    ) -> Self {
        Self {
            kind: kind.into(),
            url: url.into(),
            note: None,
            attached_at_unix_secs,
        }
    }

    /// Builder: add a free-form note.
    #[must_use]
    pub fn with_note(mut self, note: impl Into<String>) -> Self {
        self.note = Some(note.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_chains() {
        let e = EvidenceRef::new("shipping_tracking", "https://tracking/abc", 1_700_000_000)
            .with_note("USPS delivery confirmation");
        assert_eq!(e.kind, "shipping_tracking");
        assert!(e.note.is_some());
    }

    #[test]
    fn round_trips_via_json() {
        let e = EvidenceRef::new("receipt", "s3://bucket/k", 0);
        let s = serde_json::to_string(&e).unwrap();
        let back: EvidenceRef = serde_json::from_str(&s).unwrap();
        assert_eq!(e, back);
    }
}
