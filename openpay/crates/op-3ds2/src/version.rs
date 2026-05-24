//! 3-D Secure protocol versions and per-version field capability matrix.
//!
//! The 2.x family is not one wire format: every minor revision added
//! or removed fields from the message catalogue and changed the
//! required/optional/forbidden classification for several. This module
//! captures the differences as a single `FieldRule` lookup so the
//! `message` codec can validate AReq/CReq payloads against the
//! negotiated version *before* they hit the DS.
//!
//! Versions supported:
//!
//! - **2.1.0** — initial production release. Browser-flow only;
//!   no decoupled, no 3RI, no acctType for fraud scoring.
//! - **2.2.0** — added decoupled authentication, 3RI, whitelisting
//!   ("trusted beneficiary"), and the `purchaseInstalData` recurring
//!   fields. PSD2 RTS-baseline version.
//! - **2.3.0** — added the SPC (Secure Payment Confirmation) browser
//!   API, expanded device-info envelope, and the PSD3-aligned
//!   delegated-authentication identifier set.

use serde::{Deserialize, Serialize};

/// The three EMVCo 3DS protocol versions OpenPay speaks.
///
/// Each variant carries the canonical version string as published in
/// the EMVCo message catalogue.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ProtocolVersion {
    /// 3DS 2.1.0 — initial production release.
    V2_1,
    /// 3DS 2.2.0 — adds decoupled, 3RI, whitelisting. PSD2 baseline.
    V2_2,
    /// 3DS 2.3.0 — adds SPC, expanded device-info, PSD3-aligned
    /// delegated-authentication identifiers.
    V2_3,
}

impl ProtocolVersion {
    /// Canonical wire string (`"2.1.0"`, `"2.2.0"`, `"2.3.0"`).
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::V2_1 => "2.1.0",
            Self::V2_2 => "2.2.0",
            Self::V2_3 => "2.3.0",
        }
    }

    /// Parse a version string from the DS version-check response.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        match s.trim() {
            "2.1.0" => Some(Self::V2_1),
            "2.2.0" => Some(Self::V2_2),
            "2.3.0" => Some(Self::V2_3),
            _ => None,
        }
    }

    /// Highest-version-first ordering, used by version negotiation.
    #[must_use]
    pub const fn preference_order() -> [Self; 3] {
        [Self::V2_3, Self::V2_2, Self::V2_1]
    }

    /// True if this version supports decoupled authentication (2.2.0+).
    #[must_use]
    pub const fn supports_decoupled(self) -> bool {
        matches!(self, Self::V2_2 | Self::V2_3)
    }

    /// True if this version supports the 3RI subsequent-transaction
    /// flow (2.2.0+).
    #[must_use]
    pub const fn supports_threeri(self) -> bool {
        matches!(self, Self::V2_2 | Self::V2_3)
    }

    /// True if this version supports the SPC browser API (2.3.0).
    #[must_use]
    pub const fn supports_spc(self) -> bool {
        matches!(self, Self::V2_3)
    }

    /// True if this version supports trusted-beneficiary
    /// ("whitelisting") exemption signalling (2.2.0+).
    #[must_use]
    pub const fn supports_trusted_beneficiary(self) -> bool {
        matches!(self, Self::V2_2 | Self::V2_3)
    }

    /// True if this version supports the data-only flow (2.2.0+).
    #[must_use]
    pub const fn supports_data_only(self) -> bool {
        matches!(self, Self::V2_2 | Self::V2_3)
    }
}

/// Per-field, per-version constraint.
///
/// Each [`ProtocolVersion`] has a different required/optional/forbidden
/// classification for the same field, depending on the
/// `messageCategory`, `deviceChannel`, and `acctType`. This enum is
/// the lookup result returned by [`field_rule`].
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum FieldRule {
    /// Must be present; missing it yields
    /// [`crate::Error::MissingField`].
    Required,
    /// May be present.
    Optional,
    /// Must not be present; including it yields
    /// [`crate::Error::ForbiddenField`].
    Forbidden,
    /// Conditional on another field; the codec layer enforces the
    /// specific predicate.
    Conditional,
}

/// Look up the rule for a field in the AReq/CReq message catalogue.
///
/// `field` is the camelCase JSON key as published by EMVCo
/// (e.g. `"threeDSRequestorChallengeInd"`).
#[must_use]
pub const fn field_rule(field: &str, version: ProtocolVersion) -> FieldRule {
    // We pattern-match on the byte length first to keep this `const`
    // friendly, then on the string. The set is finite and small (~40
    // discriminating fields across the three versions).
    match (version, field.as_bytes()) {
        // threeDSServerTransID — required in all versions.
        (_, b"threeDSServerTransID") => FieldRule::Required,
        // messageVersion — required in all.
        (_, b"messageVersion") => FieldRule::Required,
        // messageType — required in all.
        (_, b"messageType") => FieldRule::Required,
        // acctNumber — required in AReq for all versions.
        (_, b"acctNumber") => FieldRule::Required,
        // deviceChannel — required for AReq.
        (_, b"deviceChannel") => FieldRule::Required,
        // messageCategory — required.
        (_, b"messageCategory") => FieldRule::Required,
        // browserInfo — required when deviceChannel is "02" (browser);
        // the codec layer enforces the conditional.
        (_, b"browserInfo") => FieldRule::Conditional,
        // sdkAppID — required when deviceChannel is "01" (app).
        (_, b"sdkAppID") => FieldRule::Conditional,
        (_, b"sdkEphemPubKey") => FieldRule::Conditional,
        (_, b"sdkReferenceNumber") => FieldRule::Conditional,
        (_, b"sdkTransID") => FieldRule::Conditional,
        (_, b"sdkMaxTimeout") => FieldRule::Conditional,
        // 3RI is 2.2.0+; forbidden in 2.1.0.
        (ProtocolVersion::V2_1, b"threeRIInd") => FieldRule::Forbidden,
        (ProtocolVersion::V2_2 | ProtocolVersion::V2_3, b"threeRIInd") => FieldRule::Optional,
        // Decoupled is 2.2.0+; forbidden in 2.1.0.
        (ProtocolVersion::V2_1, b"threeDSReqAuthMethod") => FieldRule::Forbidden,
        (ProtocolVersion::V2_2 | ProtocolVersion::V2_3, b"threeDSReqAuthMethod") => {
            FieldRule::Optional
        }
        (ProtocolVersion::V2_1, b"decoupledAuthInd") => FieldRule::Forbidden,
        (ProtocolVersion::V2_2 | ProtocolVersion::V2_3, b"decoupledAuthInd") => {
            FieldRule::Optional
        }
        (ProtocolVersion::V2_1, b"decoupledAuthMaxTime") => FieldRule::Forbidden,
        (ProtocolVersion::V2_2 | ProtocolVersion::V2_3, b"decoupledAuthMaxTime") => {
            FieldRule::Optional
        }
        // Whitelisting / trusted-beneficiary is 2.2.0+.
        (ProtocolVersion::V2_1, b"whiteListStatus") => FieldRule::Forbidden,
        (ProtocolVersion::V2_2 | ProtocolVersion::V2_3, b"whiteListStatus") => FieldRule::Optional,
        // SPC is 2.3.0 only.
        (ProtocolVersion::V2_1 | ProtocolVersion::V2_2, b"spcIncomp") => FieldRule::Forbidden,
        (ProtocolVersion::V2_3, b"spcIncomp") => FieldRule::Optional,
        // Delegated authentication — 2.3.0 expanded set.
        (ProtocolVersion::V2_1 | ProtocolVersion::V2_2, b"delegatedAuthData") => {
            FieldRule::Forbidden
        }
        (ProtocolVersion::V2_3, b"delegatedAuthData") => FieldRule::Optional,
        // Anything else is optional by default; the codec layer is
        // permissive on unknown extensions so scheme-specific addenda
        // (`extensions[]`) pass through.
        _ => FieldRule::Optional,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn version_strings_round_trip() {
        for v in ProtocolVersion::preference_order() {
            assert_eq!(ProtocolVersion::parse(v.as_str()), Some(v));
        }
    }

    #[test]
    fn unknown_version_parses_to_none() {
        assert_eq!(ProtocolVersion::parse("2.4.0"), None);
        assert_eq!(ProtocolVersion::parse(""), None);
    }

    #[test]
    fn preference_order_is_high_to_low() {
        let p = ProtocolVersion::preference_order();
        assert_eq!(p[0], ProtocolVersion::V2_3);
        assert_eq!(p[2], ProtocolVersion::V2_1);
    }

    #[test]
    fn capabilities_match_spec() {
        assert!(!ProtocolVersion::V2_1.supports_decoupled());
        assert!(ProtocolVersion::V2_2.supports_decoupled());
        assert!(ProtocolVersion::V2_3.supports_decoupled());

        assert!(!ProtocolVersion::V2_1.supports_threeri());
        assert!(ProtocolVersion::V2_2.supports_threeri());

        assert!(!ProtocolVersion::V2_2.supports_spc());
        assert!(ProtocolVersion::V2_3.supports_spc());
    }

    #[test]
    fn threeri_forbidden_in_2_1_optional_in_higher() {
        assert_eq!(
            field_rule("threeRIInd", ProtocolVersion::V2_1),
            FieldRule::Forbidden
        );
        assert_eq!(
            field_rule("threeRIInd", ProtocolVersion::V2_2),
            FieldRule::Optional
        );
        assert_eq!(
            field_rule("threeRIInd", ProtocolVersion::V2_3),
            FieldRule::Optional
        );
    }

    #[test]
    fn spc_only_in_2_3() {
        assert_eq!(
            field_rule("spcIncomp", ProtocolVersion::V2_2),
            FieldRule::Forbidden
        );
        assert_eq!(
            field_rule("spcIncomp", ProtocolVersion::V2_3),
            FieldRule::Optional
        );
    }

    #[test]
    fn core_required_fields_are_required_everywhere() {
        for v in ProtocolVersion::preference_order() {
            assert_eq!(field_rule("threeDSServerTransID", v), FieldRule::Required);
            assert_eq!(field_rule("messageVersion", v), FieldRule::Required);
            assert_eq!(field_rule("acctNumber", v), FieldRule::Required);
        }
    }
}
