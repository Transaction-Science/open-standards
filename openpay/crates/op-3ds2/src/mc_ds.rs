//! Mastercard Identity Check Directory Server adapter.
//!
//! Mastercard-specific notes
//! -------------------------
//! - ARes `eci` for Mastercard: `"02"` cardholder fully authenticated,
//!   `"01"` attempted (issuer doesn't participate),
//!   `"00"` not authenticated.
//! - Cryptogram envelope is the AAV (Accountholder Authentication
//!   Value), base64-encoded; previously called UCAF.
//! - Identity Check supports the Decoupled flow natively in 2.2.0+.

use crate::auth_response::eci_for;
use crate::directory_server::{DirectoryServer, VersionCheckResponse, async_trait};
use crate::error::Result;
use crate::message::{ARes, AReq, RRes, RReq};
use crate::version::ProtocolVersion;

/// Mastercard DS stub adapter.
#[derive(Debug, Clone)]
pub struct MastercardDs {
    /// `acsReferenceNumber` to populate in stub responses.
    pub acs_reference_number: String,
    /// Stub `transStatus`.
    pub stub_trans_status: String,
    /// Stub AAV. Base64.
    pub stub_aav: String,
}

impl Default for MastercardDs {
    fn default() -> Self {
        Self {
            acs_reference_number: "3DS_LOA_ACS_MCDS_020300_00001".into(),
            stub_trans_status: "Y".into(),
            stub_aav: "jJ81HADVRtXfCBATEp01CJUAAAA=".into(),
        }
    }
}

#[async_trait]
impl DirectoryServer for MastercardDs {
    async fn version_check(&self, _card_range_pan: &str) -> Result<VersionCheckResponse> {
        Ok(VersionCheckResponse {
            supported_versions: vec![
                ProtocolVersion::V2_1,
                ProtocolVersion::V2_2,
                ProtocolVersion::V2_3,
            ],
            acs_reference_number: self.acs_reference_number.clone(),
            three_ds_method_url: Some("https://3ds.mc.example/method".into()),
        })
    }

    async fn auth_request(&self, areq: &AReq) -> Result<ARes> {
        Ok(ARes {
            message_version: areq.message_version.clone(),
            three_ds_server_trans_id: areq.three_ds_server_trans_id.clone(),
            ds_trans_id: "ds-mc-00000001".into(),
            acs_trans_id: "acs-mc-00000001".into(),
            trans_status: self.stub_trans_status.clone(),
            acs_reference_number: self.acs_reference_number.clone(),
            acs_operator_id: "mc-ds".into(),
            acs_url: None,
            acs_challenge_mandated: None,
            authentication_value: Some(self.stub_aav.clone()),
            eci: Some(
                eci_for(
                    crate::directory_server::DsRoute::Mastercard,
                    &self.stub_trans_status,
                )
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
