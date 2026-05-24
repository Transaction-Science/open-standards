//! Win-rate scoring heuristic.
//!
//! Operators want a fast triage signal: "given this reason code and
//! this evidence package, is it worth filing the representment?"
//! The numbers here are heuristic — not from any single PSP — but
//! they're built from the public consensus on win rates across the
//! card networks. Operators should override the table with their
//! own historicals once they have signal.
//!
//! This module deliberately keeps the math arithmetic (no ML) so
//! the deterministic-contract doctrine holds: same inputs, same
//! score, every time, with no model file to version.

use serde::{Deserialize, Serialize};

use crate::evidence::EvidencePackage;
use crate::network::VisaReasonCode;
use crate::reason_codes::{EvidenceRequirement, ReasonCode, ReasonCodeCatalog};

/// Coarse triage band attached to a [`WinScore`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WinScoreBand {
    /// Very poor outlook — recommend accepting the loss.
    Reject,
    /// Marginal — file only if amount justifies the fee risk.
    Marginal,
    /// Reasonable — file.
    Likely,
    /// Strong — file with high confidence.
    Strong,
}

/// Per-dispute win-rate estimate, expressed as a probability and a
/// triage band.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WinScore {
    /// Estimated win probability, `0.0..=1.0`.
    pub probability: f32,
    /// Coarse band derived from the probability.
    pub band: WinScoreBand,
    /// Count of required-evidence items satisfied by the package.
    pub satisfied_required: usize,
    /// Count of required-evidence items the package is missing.
    pub missing_required: usize,
}

impl WinScore {
    /// Compute the win-rate estimate for `(reason, package)`.
    ///
    /// Heuristic:
    ///
    /// - Start from a baseline per reason-code class (fraud /
    ///   processing / consumer / authorization).
    /// - Multiply by an "evidence completeness" factor: the
    ///   fraction of required-evidence items present.
    /// - Add a small bonus for the high-value items (3DS auth
    ///   value, qualifying history) when present.
    /// - Clamp to `[0.0, 0.95]`.
    #[must_use]
    pub fn evaluate(reason: ReasonCode, package: &EvidencePackage) -> Self {
        let baseline = baseline(reason);
        let required = ReasonCodeCatalog::required_evidence(reason);
        let present = package.satisfied();
        let satisfied = required.iter().filter(|r| present.contains(r)).count();
        let missing = required.len().saturating_sub(satisfied);

        let completeness = if required.is_empty() {
            1.0
        } else {
            // f32 conversion: counts are small (single digits) so the
            // precision loss here is irrelevant.
            #[allow(clippy::cast_precision_loss)]
            let s = satisfied as f32;
            #[allow(clippy::cast_precision_loss)]
            let r = required.len() as f32;
            s / r
        };

        let mut score = baseline * completeness;

        if present.contains(&EvidenceRequirement::ThreeDsAuthValue) {
            score += 0.10;
        }
        if present.contains(&EvidenceRequirement::QualifyingHistory) {
            score += 0.10;
        }

        if score < 0.0 {
            score = 0.0;
        }
        if score > 0.95 {
            score = 0.95;
        }

        Self {
            probability: score,
            band: band_for(score),
            satisfied_required: satisfied,
            missing_required: missing,
        }
    }
}

fn band_for(p: f32) -> WinScoreBand {
    if p < 0.25 {
        WinScoreBand::Reject
    } else if p < 0.50 {
        WinScoreBand::Marginal
    } else if p < 0.75 {
        WinScoreBand::Likely
    } else {
        WinScoreBand::Strong
    }
}

fn baseline(reason: ReasonCode) -> f32 {
    match reason {
        // Visa
        ReasonCode::Visa(VisaReasonCode::F1040) => 0.55, // CE3.0 raises this
        ReasonCode::Visa(VisaReasonCode::F1010 | VisaReasonCode::F1020) => 0.15, // EMV liability lost
        ReasonCode::Visa(VisaReasonCode::F1030 | VisaReasonCode::F1050) => 0.30,
        ReasonCode::Visa(
            VisaReasonCode::A1110 | VisaReasonCode::A1120 | VisaReasonCode::A1130,
        ) => 0.20,
        ReasonCode::Visa(
            VisaReasonCode::P1210
            | VisaReasonCode::P1220
            | VisaReasonCode::P1230
            | VisaReasonCode::P1240
            | VisaReasonCode::P1250
            | VisaReasonCode::P1260
            | VisaReasonCode::P1270,
        ) => 0.45,
        ReasonCode::Visa(_) => 0.40,
        // Mastercard
        ReasonCode::Mastercard(_) => 0.40,
        // Amex tends to side with the cardholder; baseline lower.
        ReasonCode::Amex(_) => 0.30,
        // Discover similar to Visa.
        ReasonCode::Discover(_) => 0.40,
        // PayPal seller-protection wins are common for INR + tracked
        // shipping; baseline middling.
        ReasonCode::PayPal(_) => 0.45,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::evidence::{EvidenceItem, EvidencePackageBuilder};
    use crate::reason_codes::EvidenceRequirement;
    use time::OffsetDateTime;

    fn t() -> OffsetDateTime {
        OffsetDateTime::from_unix_timestamp(1_700_000_000).expect("ok")
    }

    fn item(kind: EvidenceRequirement) -> EvidenceItem {
        EvidenceItem::new(kind, "x", "text/plain", b"x".to_vec(), t()).expect("ok")
    }

    #[test]
    fn missing_evidence_drops_band() {
        // A 10.4 with only the receipt should be Reject/Marginal.
        let mut b =
            EvidencePackageBuilder::new(ReasonCode::Visa(VisaReasonCode::F1040));
        for req in [
            EvidenceRequirement::Receipt,
            EvidenceRequirement::AvsResult,
            EvidenceRequirement::CvvResult,
            EvidenceRequirement::ThreeDsAuthValue,
            EvidenceRequirement::ProofOfDelivery,
            EvidenceRequirement::CheckoutIp,
            EvidenceRequirement::DeviceFingerprint,
            EvidenceRequirement::QualifyingHistory,
        ] {
            b = b.add(item(req));
        }
        let pkg = b.seal(t()).expect("seal");
        let score = WinScore::evaluate(ReasonCode::Visa(VisaReasonCode::F1040), &pkg);
        assert!(matches!(
            score.band,
            WinScoreBand::Likely | WinScoreBand::Strong
        ));
        assert_eq!(score.missing_required, 0);
    }
}
