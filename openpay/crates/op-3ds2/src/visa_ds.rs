//! Visa Directory Server adapter.
//!
//! Production deployments wire this adapter against Visa's Directory
//! Server endpoints (URLs and mTLS certificates issued by Visa's
//! developer portal). This module ships a stub that returns canned
//! responses suitable for local testing and conformance fixtures.
//!
//! Visa-specific notes
//! -------------------
//! - ARes `eci` for Visa: `"05"` cardholder fully authenticated,
//!   `"06"` attempted (issuer doesn't participate).
//! - Cryptogram envelope is the CAVV (Cardholder Authentication
//!   Verification Value), 28 bytes base64-encoded.
//! - `acsReferenceNumber` format `3DS_LOA_ACS_VISA_<6digits>_<5digits>`.

use crate::auth_response::eci_for;
use crate::directory_server::{DirectoryServer, VersionCheckResponse, async_trait};
use crate::error::Result;
use crate::message::{ARes, AReq, RRes, RReq};
use crate::version::ProtocolVersion;

/// Visa DS stub adapter.
#[derive(Debug, Clone)]
pub struct VisaDs {
    /// Optional canned `acsReferenceNumber` override (tests).
    pub acs_reference_number: String,
    /// Stub `transStatus` for every AReq. Defaults to `"Y"` (authenticated).
    pub stub_trans_status: String,
    /// Stub CAVV for every ARes. Base64.
    pub stub_cavv: String,
}

impl Default for VisaDs {
    fn default() -> Self {
        Self {
            acs_reference_number: "3DS_LOA_ACS_VISA_020300_00001".into(),
            stub_trans_status: "Y".into(),
            stub_cavv: "AAABBkgkkRIBARQAAAAGCSRBgkk=".into(),
        }
    }
}

#[async_trait]
impl DirectoryServer for VisaDs {
    async fn version_check(&self, _card_range_pan: &str) -> Result<VersionCheckResponse> {
        Ok(VersionCheckResponse {
            supported_versions: vec![ProtocolVersion::V2_2, ProtocolVersion::V2_3],
            acs_reference_number: self.acs_reference_number.clone(),
            three_ds_method_url: Some(
                "https://3ds.visa.example/method".into(),
            ),
        })
    }

    async fn auth_request(&self, areq: &AReq) -> Result<ARes> {
        Ok(ARes {
            message_version: areq.message_version.clone(),
            three_ds_server_trans_id: areq.three_ds_server_trans_id.clone(),
            ds_trans_id: "ds-visa-00000001".into(),
            acs_trans_id: "acs-visa-00000001".into(),
            trans_status: self.stub_trans_status.clone(),
            acs_reference_number: self.acs_reference_number.clone(),
            acs_operator_id: "visa-ds".into(),
            acs_url: None,
            acs_challenge_mandated: None,
            authentication_value: Some(self.stub_cavv.clone()),
            eci: Some(
                eci_for(crate::directory_server::DsRoute::Visa, &self.stub_trans_status)
                    .to_owned(),
            ),
            acs_signed_content: None,
            trans_status_reason: None,
            extensions: vec![],
        })
    }

    async fn results_request(&self, rreq: &RReq) -> Result<RRes> {
        Ok(RRes {
            message_version: rreq.message_version.clone(),
            three_ds_server_trans_id: rreq.three_ds_server_trans_id.clone(),
            results_status: "01".into(),
        })
    }
}
