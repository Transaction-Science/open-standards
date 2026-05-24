//! EUDI Wallet Architecture and Reference Framework (ARF) profile.
//!
//! The EU Digital Identity Wallet ARF mandates a specific subset of
//! OID4VC for member-state-issued credentials (PID, mDL, …):
//!
//! * **Credential formats**: `dc+sd-jwt` (SD-JWT VC) for IETF-style
//!   credentials and `mso_mdoc` (ISO/IEC 18013-5 mdoc) for the
//!   Personal Identification Data (PID) and mobile Driving Licence.
//! * **Signing algorithms**: `ES256` (P-256) is the baseline; `ES384`
//!   and `ES512` are optional.
//! * **DPoP**: mandatory at the token + credential endpoints.
//! * **Status mechanism**: Token Status List (IETF) plus optional
//!   `validUntil` claim.
//! * **Wallet attestation**: required (key attestation header `kat`
//!   per ARF 1.x).
//! * **PID/mDL doctypes**: `eu.europa.ec.eudi.pid.1` and
//!   `org.iso.18013.5.1.mDL`.
//!
//! This module supplies constants + a profile evaluator that combines
//! HAIP checks with ARF-specific additions.

use serde::{Deserialize, Serialize};

use crate::error::OidcError;
use crate::haip;
use crate::issuer::{AuthorizationServerMetadata, CredentialConfiguration};

/// ARF SD-JWT VC PID doctype.
pub const EUDI_PID_VCT: &str = "https://eu.europa.ec.eudi/pid/1";
/// ARF SD-JWT VC mDL doctype.
pub const EUDI_MDL_VCT: &str = "https://eu.europa.ec.eudi/mdl/1";
/// ARF mdoc PID doctype.
pub const EUDI_PID_DOCTYPE: &str = "eu.europa.ec.eudi.pid.1";
/// ISO/IEC 18013-5 mDL doctype.
pub const EUDI_MDL_DOCTYPE: &str = "org.iso.18013.5.1.mDL";

/// One additional ARF-specific check (beyond HAIP).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EudiCheck {
    /// vct or doctype is a recognised EUDI ARF identifier.
    EudiDoctypeRecognised,
    /// `ES256` is in the signing alg set.
    Es256Supported,
    /// DPoP signing algorithms include `ES256`.
    DpopEs256,
}

impl EudiCheck {
    /// Wire identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::EudiDoctypeRecognised => "eudi_doctype_recognised",
            Self::Es256Supported => "es256_supported",
            Self::DpopEs256 => "dpop_es256",
        }
    }
}

/// EUDI evaluation report. Builds on top of [`crate::haip::HaipReport`]
/// so callers can see both layers in one go.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EudiReport {
    /// Underlying HAIP report.
    pub haip: haip::HaipReport,
    /// EUDI-specific checks that passed.
    pub passed: Vec<String>,
    /// EUDI-specific checks that failed.
    pub failed: Vec<(String, String)>,
}

impl EudiReport {
    /// True if no check failed at either layer.
    pub fn is_ok(&self) -> bool {
        self.haip.is_ok() && self.failed.is_empty()
    }

    /// Collapse to a single error, or `Ok(())` if both layers passed.
    pub fn into_result(self) -> Result<(), OidcError> {
        let haip_clone = self.haip.clone();
        if let Err(e) = haip_clone.into_result() {
            return Err(OidcError::Eudi(format!("haip: {e}")));
        }
        if self.failed.is_empty() {
            Ok(())
        } else {
            let msg = self
                .failed
                .iter()
                .map(|(c, r)| format!("{c}: {r}"))
                .collect::<Vec<_>>()
                .join("; ");
            Err(OidcError::Eudi(msg))
        }
    }
}

/// Evaluate a credential configuration against the EUDI ARF profile.
pub fn evaluate(
    config: &CredentialConfiguration,
    as_metadata: &AuthorizationServerMetadata,
) -> EudiReport {
    let haip = haip::evaluate(config, as_metadata);
    let mut passed = Vec::new();
    let mut failed: Vec<(String, String)> = Vec::new();

    let recognised = match config.format.as_str() {
        "dc+sd-jwt" | "vc+sd-jwt" => config
            .vct
            .as_deref()
            .map(|v| v == EUDI_PID_VCT || v == EUDI_MDL_VCT)
            .unwrap_or(false),
        "mso_mdoc" => config
            .doctype
            .as_deref()
            .map(|d| d == EUDI_PID_DOCTYPE || d == EUDI_MDL_DOCTYPE)
            .unwrap_or(false),
        _ => false,
    };
    if recognised {
        passed.push(EudiCheck::EudiDoctypeRecognised.as_str().into());
    } else {
        failed.push((
            EudiCheck::EudiDoctypeRecognised.as_str().into(),
            "credential is neither EUDI PID nor mDL".into(),
        ));
    }

    if config
        .credential_signing_alg_values_supported
        .iter()
        .any(|a| a == "ES256")
    {
        passed.push(EudiCheck::Es256Supported.as_str().into());
    } else {
        failed.push((
            EudiCheck::Es256Supported.as_str().into(),
            "ES256 must be in credential_signing_alg_values_supported".into(),
        ));
    }

    if as_metadata
        .dpop_signing_alg_values_supported
        .iter()
        .any(|a| a == "ES256")
    {
        passed.push(EudiCheck::DpopEs256.as_str().into());
    } else {
        failed.push((
            EudiCheck::DpopEs256.as_str().into(),
            "DPoP must offer ES256 per EUDI ARF".into(),
        ));
    }

    EudiReport {
        haip,
        passed,
        failed,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn good_pid() -> CredentialConfiguration {
        CredentialConfiguration {
            format: "dc+sd-jwt".into(),
            scope: None,
            credential_signing_alg_values_supported: vec!["ES256".into()],
            proof_types_supported: HashMap::new(),
            vct: Some(EUDI_PID_VCT.into()),
            doctype: None,
            credential_definition: None,
        }
    }

    fn good_mdl_mdoc() -> CredentialConfiguration {
        CredentialConfiguration {
            format: "mso_mdoc".into(),
            scope: None,
            credential_signing_alg_values_supported: vec!["ES256".into()],
            proof_types_supported: HashMap::new(),
            vct: None,
            doctype: Some(EUDI_MDL_DOCTYPE.into()),
            credential_definition: None,
        }
    }

    fn good_as() -> AuthorizationServerMetadata {
        AuthorizationServerMetadata {
            issuer: "https://issuer.example.com".into(),
            token_endpoint: "https://issuer.example.com/token".into(),
            authorization_endpoint: None,
            pushed_authorization_request_endpoint: None,
            grant_types_supported: vec![
                AuthorizationServerMetadata::PREAUTH_GRANT.into(),
            ],
            dpop_signing_alg_values_supported: vec!["ES256".into()],
            code_challenge_methods_supported: vec!["S256".into()],
            pre_authorized_grant_anonymous_access_supported: true,
        }
    }

    #[test]
    fn pid_profile_passes() {
        let r = evaluate(&good_pid(), &good_as());
        assert!(r.is_ok(), "haip={:?} eudi={:?}", r.haip.failed, r.failed);
    }

    #[test]
    fn mdl_mdoc_passes() {
        let r = evaluate(&good_mdl_mdoc(), &good_as());
        assert!(r.is_ok(), "haip={:?} eudi={:?}", r.haip.failed, r.failed);
    }

    #[test]
    fn unrecognised_vct_fails() {
        let mut c = good_pid();
        c.vct = Some("https://other.example.com/x".into());
        let r = evaluate(&c, &good_as());
        assert!(!r.is_ok());
    }

    #[test]
    fn missing_es256_fails() {
        let mut c = good_pid();
        c.credential_signing_alg_values_supported = vec!["EdDSA".into()];
        let r = evaluate(&c, &good_as());
        assert!(!r.is_ok());
    }
}
