//! Evidence-package builder.
//!
//! The [`EvidencePackage`] is a typed bundle of artifacts a merchant
//! ships with a representment. The builder pattern enforces that
//! every required-evidence entry from
//! [`crate::reason_codes::ReasonCodeCatalog`] is satisfied before
//! the package can be sealed.
//!
//! No I/O is performed here. Evidence items carry a small content
//! summary + opaque bytes; operators decide how to persist /
//! transport the underlying blob (S3, GCS, on-prem disk, etc.).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;

use crate::error::{Error, Result};
use crate::reason_codes::{EvidenceRequirement, ReasonCode, ReasonCodeCatalog};

/// Conservative cap on per-item payload size. Networks have their
/// own limits (Visa caps representments at ~2MB per file, 19 files
/// total); this lets the builder refuse oversized blobs before they
/// hit the wire.
pub const MAX_ITEM_BYTES: usize = 2 * 1024 * 1024;

/// A single piece of evidence.
///
/// `payload` is treated as opaque bytes — JSON for AVS / CVV
/// results, a PDF for receipts, a CSV for delivery confirmations,
/// etc. Operators are responsible for serializing it consistently.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidenceItem {
    /// Which requirement class this item satisfies.
    pub kind: EvidenceRequirement,
    /// Short human description (filename, "AVS=Y; CVV=M", etc.).
    pub label: String,
    /// MIME type of the payload.
    pub mime_type: String,
    /// Opaque bytes the operator must persist + transport.
    pub payload: Vec<u8>,
    /// When the underlying artifact was captured.
    pub captured_at: OffsetDateTime,
}

impl EvidenceItem {
    /// Construct an evidence item with validation of the size cap.
    ///
    /// # Errors
    /// Returns [`Error::InvalidEvidence`] when `payload` exceeds
    /// [`MAX_ITEM_BYTES`].
    pub fn new(
        kind: EvidenceRequirement,
        label: impl Into<String>,
        mime_type: impl Into<String>,
        payload: Vec<u8>,
        captured_at: OffsetDateTime,
    ) -> Result<Self> {
        if payload.len() > MAX_ITEM_BYTES {
            return Err(Error::InvalidEvidence("payload exceeds MAX_ITEM_BYTES"));
        }
        Ok(Self {
            kind,
            label: label.into(),
            mime_type: mime_type.into(),
            payload,
            captured_at,
        })
    }
}

/// A sealed, network-ready bundle of evidence.
///
/// Operators construct one via [`EvidencePackageBuilder`]; this
/// type is purposefully read-only once built so the wire-layer
/// adapter can hash / sign it without worrying about in-flight
/// mutation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EvidencePackage {
    /// Reason code the package is being filed against.
    pub reason: ReasonCode,
    /// Items, keyed by requirement so duplicate-kind submissions
    /// overwrite rather than ambiguously stack.
    items: BTreeMap<EvidenceRequirement, Vec<EvidenceItem>>,
    /// When the package was sealed.
    pub sealed_at: OffsetDateTime,
}

impl EvidencePackage {
    /// All items, in deterministic order (sorted by requirement).
    #[must_use]
    pub fn items(&self) -> impl Iterator<Item = &EvidenceItem> {
        self.items.values().flat_map(|v| v.iter())
    }

    /// Items submitted for a given requirement kind.
    #[must_use]
    pub fn items_for(&self, kind: EvidenceRequirement) -> &[EvidenceItem] {
        self.items
            .get(&kind)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }

    /// Requirement kinds present in the package.
    #[must_use]
    pub fn satisfied(&self) -> Vec<EvidenceRequirement> {
        self.items.keys().copied().collect()
    }

    /// Total payload size across all items, in bytes.
    #[must_use]
    pub fn total_bytes(&self) -> usize {
        self.items
            .values()
            .flat_map(|v| v.iter())
            .map(|i| i.payload.len())
            .sum()
    }
}

/// Builder enforcing the required-evidence contract before sealing.
#[derive(Debug)]
pub struct EvidencePackageBuilder {
    reason: ReasonCode,
    items: BTreeMap<EvidenceRequirement, Vec<EvidenceItem>>,
}

impl EvidencePackageBuilder {
    /// Start a new builder for the given reason code.
    #[must_use]
    pub fn new(reason: ReasonCode) -> Self {
        Self {
            reason,
            items: BTreeMap::new(),
        }
    }

    /// Add an evidence item. Multiple items per requirement are
    /// permitted (e.g., two delivery confirmations for a split
    /// shipment).
    #[must_use]
    pub fn add(mut self, item: EvidenceItem) -> Self {
        self.items.entry(item.kind).or_default().push(item);
        self
    }

    /// Required-evidence list for this builder's reason code.
    #[must_use]
    pub fn required(&self) -> &'static [EvidenceRequirement] {
        ReasonCodeCatalog::required_evidence(self.reason)
    }

    /// Requirements not yet satisfied by [`Self::add`] calls.
    #[must_use]
    pub fn missing(&self) -> Vec<EvidenceRequirement> {
        self.required()
            .iter()
            .copied()
            .filter(|r| !self.items.contains_key(r))
            .collect()
    }

    /// Seal the builder into an immutable [`EvidencePackage`].
    ///
    /// # Errors
    ///
    /// Returns [`Error::MissingEvidence`] with the first
    /// unsatisfied requirement's static name when the package is
    /// incomplete.
    pub fn seal(self, sealed_at: OffsetDateTime) -> Result<EvidencePackage> {
        if let Some(missing) = self.missing().first() {
            return Err(Error::MissingEvidence(requirement_static_name(*missing)));
        }
        if self.items.is_empty() {
            return Err(Error::InvalidEvidence("package has no items"));
        }
        Ok(EvidencePackage {
            reason: self.reason,
            items: self.items,
            sealed_at,
        })
    }
}

const fn requirement_static_name(r: EvidenceRequirement) -> &'static str {
    match r {
        EvidenceRequirement::Receipt => "Receipt",
        EvidenceRequirement::AvsResult => "AvsResult",
        EvidenceRequirement::CvvResult => "CvvResult",
        EvidenceRequirement::ThreeDsAuthValue => "ThreeDsAuthValue",
        EvidenceRequirement::ProofOfDelivery => "ProofOfDelivery",
        EvidenceRequirement::CustomerCommunications => "CustomerCommunications",
        EvidenceRequirement::CheckoutIp => "CheckoutIp",
        EvidenceRequirement::DeviceFingerprint => "DeviceFingerprint",
        EvidenceRequirement::TermsOfService => "TermsOfService",
        EvidenceRequirement::QualifyingHistory => "QualifyingHistory",
        EvidenceRequirement::SubscriptionCancellation => "SubscriptionCancellation",
        EvidenceRequirement::RefundReceipt => "RefundReceipt",
        EvidenceRequirement::AuthorizationLog => "AuthorizationLog",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::VisaReasonCode;

    fn t() -> OffsetDateTime {
        OffsetDateTime::from_unix_timestamp(1_700_000_000).expect("valid")
    }

    #[test]
    fn seal_fails_when_required_missing() {
        let b = EvidencePackageBuilder::new(ReasonCode::Visa(VisaReasonCode::F1040)).add(
            EvidenceItem::new(
                EvidenceRequirement::Receipt,
                "invoice.pdf",
                "application/pdf",
                vec![0; 32],
                t(),
            )
            .expect("ok"),
        );
        let err = b.seal(t()).expect_err("incomplete must fail");
        assert!(matches!(err, Error::MissingEvidence(_)));
    }

    #[test]
    fn oversized_payload_refused() {
        let big = vec![0u8; MAX_ITEM_BYTES + 1];
        let err = EvidenceItem::new(
            EvidenceRequirement::Receipt,
            "huge",
            "application/octet-stream",
            big,
            t(),
        )
        .expect_err("size cap");
        assert!(matches!(err, Error::InvalidEvidence(_)));
    }
}
