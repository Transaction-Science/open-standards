//! American Express SafeKey Directory Server adapter.
//!
//! Amex-specific notes
//! -------------------
//! - ARes `eci` for Amex: `"05"` cardholder fully authenticated,
//!   `"06"` attempted.
//! - Cryptogram envelope is the AEVV (Amex Verification Value),
//!   base64-encoded.

use crate::auth_response::eci_for;
use crate::directory_server::{DirectoryServer, VersionCheckResponse, async_trait};
use crate::error::Result;
use crate::message::{ARes, AReq, RRes, RReq};
use crate::version::ProtocolVersion;

/// Amex SafeKey DS stub adapter.
#[derive(Debug, Clone)]
pub struct AmexDs {
    /// `acsReferenceNumber` placeholder.
    pub acs_reference_number: String,
    /// Stub `transStatus`.
    pub stub_trans_status: String,
    /// Stub AEVV.
    pub stub_aevv: String,
}

impl Default for AmexDs {
    fn default() -> Self {
        Self {
            acs_reference_number: "3DS_LOA_ACS_AEXP_020300_00001".into(),
            stub_trans_status: "Y".into(),
            stub_aevv: "AEVVAAAABBkgkkR=".into(),
        }
    }
}

#[async_trait]
impl DirectoryServer for AmexDs {
    async fn version_check(&self, _card_range_pan: &str) -> Result<VersionCheckResponse> {
        Ok(VersionCheckResponse {
            supported_versions: vec![ProtocolVersion::V2_2, ProtocolVersion::V2_3],
            acs_reference_number: self.acs_reference_number.clone(),
            three_ds_method_url: Some("https://3ds.amex.example/method".into()),
        })
    }

    async fn auth_request(&self, areq: &AReq) -> Result<ARes> {
        Ok(ARes {
            message_version: areq.message_version.clone(),
            three_ds_server_trans_id: areq.three_ds_server_trans_id.clone(),
            ds_trans_id: "ds-amex-00000001".into(),
            acs_trans_id: "acs-amex-00000001".into(),
            trans_status: self.stub_trans_status.clone(),
            acs_reference_number: self.acs_reference_number.clone(),
            acs_operator_id: "amex-ds".into(),
            acs_url: None,
            acs_challenge_mandated: None,
            authentication_value: Some(self.stub_aevv.clone()),
            eci: Some(
                eci_for(crate::directory_server::DsRoute::Amex, &self.stub_trans_status)
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
