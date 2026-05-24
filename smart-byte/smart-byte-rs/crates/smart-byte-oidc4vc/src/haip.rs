//! High-Assurance Interoperability Profile (HAIP) for OID4VC.
//!
//! HAIP is the OIDF "tightened" interoperability profile that constrains
//! the open set of choices in OID4VCI / OID4VP to a small, profiled
//! subset suitable for high-assurance deployments (government IDs,
//! qualified electronic signatures, …):
//!
//! * Credential formats: `dc+sd-jwt` (SD-JWT VC) and `mso_mdoc`.
//! * Issuer signing: `ES256` (P-256 ECDSA) is mandatory; `EdDSA` is
//!   recommended.
//! * `code_challenge_method = S256` for authorization-code flow.
//! * DPoP is mandatory for confidential clients.
//! * Status mechanism: Token Status List (IETF) for SD-JWT VC and mdoc
//!   `IssuerSigned` status for mdoc.
//! * Wallet attestation MAY be required at the credential endpoint.

use std::collections::HashSet;

use serde::{Deserialize, Serialize};

use crate::error::OidcError;
use crate::issuer::{AuthorizationServerMetadata, CredentialConfiguration};

/// Credential formats permitted by HAIP.
pub const HAIP_FORMAT_SD_JWT_VC: &str = "dc+sd-jwt";
/// mdoc CBOR format.
pub const HAIP_FORMAT_MSO_MDOC: &str = "mso_mdoc";

/// Signing algorithms allowed by HAIP.
pub const HAIP_ALG_ES256: &str = "ES256";
/// EdDSA signing algorithm.
pub const HAIP_ALG_EDDSA: &str = "EdDSA";

/// One profile-level check.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum HaipCheck {
    /// Format is in the allowed set.
    FormatAllowed,
    /// Signing alg includes a HAIP-allowed value.
    AlgAllowed,
    /// AS metadata declares `S256` PKCE.
    PkceS256,
    /// AS metadata declares at least one DPoP alg.
    DpopRequired,
}

impl HaipCheck {
    /// Wire identifier for diagnostics.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::FormatAllowed => "format_allowed",
            Self::AlgAllowed => "alg_allowed",
            Self::PkceS256 => "pkce_s256",
            Self::DpopRequired => "dpop_required",
        }
    }
}

/// Result of profile evaluation. `passed.len() + failed.len()` equals
/// the number of checks attempted.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HaipReport {
    /// Checks that passed.
    pub passed: Vec<String>,
    /// Checks that failed (with reason).
    pub failed: Vec<(String, String)>,
}

impl HaipReport {
    /// True if no check failed.
    pub fn is_ok(&self) -> bool {
        self.failed.is_empty()
    }

    /// Return Err if any check failed, with concatenated reasons.
    pub fn into_result(self) -> Result<(), OidcError> {
        if self.is_ok() {
            Ok(())
        } else {
            let msg = self
                .failed
                .iter()
                .map(|(c, r)| format!("{c}: {r}"))
                .collect::<Vec<_>>()
                .join("; ");
            Err(OidcError::Haip(msg))
        }
    }
}

/// Evaluate a credential configuration + AS metadata against HAIP.
pub fn evaluate(
    config: &CredentialConfiguration,
    as_metadata: &AuthorizationServerMetadata,
) -> HaipReport {
    let mut report = HaipReport::default();
    let allowed_formats: HashSet<&str> =
        [HAIP_FORMAT_SD_JWT_VC, HAIP_FORMAT_MSO_MDOC]
            .into_iter()
            .collect();
    let allowed_algs: HashSet<&str> =
        [HAIP_ALG_ES256, HAIP_ALG_EDDSA].into_iter().collect();

    if allowed_formats.contains(config.format.as_str()) {
        report.passed.push(HaipCheck::FormatAllowed.as_str().into());
    } else {
        report.failed.push((
            HaipCheck::FormatAllowed.as_str().into(),
            format!("format {} not permitted by HAIP", config.format),
        ));
    }

    let alg_ok = config
        .credential_signing_alg_values_supported
        .iter()
        .any(|a| allowed_algs.contains(a.as_str()));
    if alg_ok {
        report.passed.push(HaipCheck::AlgAllowed.as_str().into());
    } else {
        report.failed.push((
            HaipCheck::AlgAllowed.as_str().into(),
            "no HAIP-permitted credential_signing_alg present".into(),
        ));
    }

    if as_metadata
        .code_challenge_methods_supported
        .iter()
        .any(|m| m == "S256")
    {
        report.passed.push(HaipCheck::PkceS256.as_str().into());
    } else if as_metadata.supports_authorization_code() {
        report.failed.push((
            HaipCheck::PkceS256.as_str().into(),
            "authorization_code grant requires S256 PKCE".into(),
        ));
    } else {
        // pre-auth-only deployments don't need PKCE.
        report.passed.push(HaipCheck::PkceS256.as_str().into());
    }

    if !as_metadata.dpop_signing_alg_values_supported.is_empty() {
        report.passed.push(HaipCheck::DpopRequired.as_str().into());
    } else {
        report.failed.push((
            HaipCheck::DpopRequired.as_str().into(),
            "DPoP signing algs missing from AS metadata".into(),
        ));
    }

    report
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::issuer::AuthorizationServerMetadata;
    use std::collections::HashMap;

    fn good_config() -> CredentialConfiguration {
        CredentialConfiguration {
            format: HAIP_FORMAT_SD_JWT_VC.into(),
            scope: None,
            credential_signing_alg_values_supported: vec![
                HAIP_ALG_ES256.into(),
            ],
            proof_types_supported: HashMap::new(),
            vct: Some("https://example.com/X".into()),
            doctype: None,
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
    fn ok_profile_passes() {
        let r = evaluate(&good_config(), &good_as());
        assert!(r.is_ok(), "{:?}", r.failed);
    }

    #[test]
    fn rejects_non_haip_format() {
        let mut c = good_config();
        c.format = "jwt_vc_json".into();
        let r = evaluate(&c, &good_as());
        assert!(!r.is_ok());
        assert!(r.failed.iter().any(|(k, _)| k == "format_allowed"));
    }

    #[test]
    fn rejects_missing_alg() {
        let mut c = good_config();
        c.credential_signing_alg_values_supported = vec!["RS256".into()];
        let r = evaluate(&c, &good_as());
        assert!(!r.is_ok());
        assert!(r.failed.iter().any(|(k, _)| k == "alg_allowed"));
    }

    #[test]
    fn rejects_missing_dpop() {
        let mut a = good_as();
        a.dpop_signing_alg_values_supported.clear();
        let r = evaluate(&good_config(), &a);
        assert!(!r.is_ok());
    }
}
