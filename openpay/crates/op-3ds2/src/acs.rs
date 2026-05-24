//! Access Control Server (ACS) — issuer-side primitives.
//!
//! Most OpenPay operators speak to the ACS through the DS (as
//! merchants). But some operators *are* issuers — neobanks,
//! processors offering issuer-processing, embedded-finance providers.
//! For them the ACS lives inside their stack and must be able to
//! emit valid ARes / CRes payloads and verify cardholders via
//! one of the SCA factors:
//!
//! - Static OTP (legacy; PSD2-deprecated for new flows).
//! - Out-of-band push to a banking app, confirmed by biometric.
//! - WebAuthn / FIDO2 challenge on a registered authenticator.
//!
//! The [`AcsServer`] struct emits ARes/CRes payloads from an
//! [`AcsConfig`]. The verification primitives are deliberately small
//! and pluggable — they capture what an ACS *decides* (approve /
//! decline / challenge / decoupled), not the UX of how the cardholder
//! confirms.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use crate::auth_response::TransactionStatus;
use crate::message::{ARes, AReq, CRes};

/// Authentication method an ACS chose for a given AReq.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum AcsAuthMethod {
    /// Issued an OTP over SMS / app.
    StaticOtp,
    /// Push to the cardholder's banking app, confirmed via biometric.
    OobBiometric,
    /// WebAuthn / FIDO2 challenge.
    Fido,
    /// No authentication — frictionless approval (issuer trusted the
    /// envelope).
    Frictionless,
    /// No authentication — issuer declined.
    Declined,
}

impl AcsAuthMethod {
    /// `authenticationMethod` (RReq) field value per EMVCo.
    #[must_use]
    pub const fn as_wire(self) -> &'static str {
        match self {
            // EMVCo 6.2.4.5 authenticationMethod: 01 static passcode,
            // 02 SMS OTP, 04 OOB authentication, 05 cardholder app
            // OOB, 06 hybrid, 07 issuer-initiated.
            Self::StaticOtp => "01",
            Self::OobBiometric => "04",
            Self::Fido => "06",
            Self::Frictionless | Self::Declined => "07",
        }
    }
}

/// ACS configuration. One per ACS instance the operator runs.
#[derive(Debug, Clone)]
pub struct AcsConfig {
    /// Operator id assigned by the DS at onboarding.
    pub acs_operator_id: String,
    /// ACS reference number assigned by EMVCo at LoA certification.
    pub acs_reference_number: String,
    /// Public URL the 3DS Server should POST CReq to.
    pub acs_url: String,
    /// Public URL used for decoupled polling.
    pub acs_decoupled_url: Option<String>,
}

/// Minimal issuer-side ACS that produces ARes / CRes payloads.
#[derive(Debug, Clone)]
pub struct AcsServer {
    /// Static configuration the operator wired in.
    pub config: AcsConfig,
}

impl AcsServer {
    /// Construct.
    #[must_use]
    pub const fn new(config: AcsConfig) -> Self {
        Self { config }
    }

    /// Decide and emit the ARes for an incoming AReq.
    ///
    /// `decision` is the issuer's chosen outcome; the ACS does *not*
    /// look at the cardholder credentials here — that's a separate
    /// upstream call. This method is purely about packaging the
    /// decision into the wire format.
    #[must_use]
    pub fn build_ares(
        &self,
        areq: &AReq,
        decision: TransactionStatus,
        cavv: Option<String>,
        eci: Option<String>,
    ) -> ARes {
        ARes {
            message_version: areq.message_version.clone(),
            three_ds_server_trans_id: areq.three_ds_server_trans_id.clone(),
            ds_trans_id: Uuid::new_v4().to_string(),
            acs_trans_id: Uuid::new_v4().to_string(),
            trans_status: decision.as_letter().to_owned(),
            acs_reference_number: self.config.acs_reference_number.clone(),
            acs_operator_id: self.config.acs_operator_id.clone(),
            acs_url: match decision {
                TransactionStatus::ChallengeRequired
                | TransactionStatus::ChallengeRequiredDecoupled => {
                    Some(self.config.acs_url.clone())
                }
                _ => None,
            },
            acs_challenge_mandated: match decision {
                TransactionStatus::ChallengeRequired => Some("Y".into()),
                _ => None,
            },
            authentication_value: cavv,
            eci,
            acs_signed_content: None,
            trans_status_reason: None,
            extensions: vec![],
        }
    }

    /// Build the final CRes for a settled challenge.
    #[must_use]
    pub fn build_final_cres(
        &self,
        acs_trans_id: &str,
        three_ds_server_trans_id: &str,
        decision: TransactionStatus,
    ) -> CRes {
        CRes {
            message_version: "2.2.0".into(),
            three_ds_server_trans_id: three_ds_server_trans_id.to_owned(),
            acs_trans_id: acs_trans_id.to_owned(),
            acs_counter_a_to_s: Some("002".into()),
            acs_html: None,
            challenge_completion_ind: Some("Y".into()),
            trans_status: Some(decision.as_letter().to_owned()),
            trans_status_reason: None,
            oob_app_url: None,
            oob_app_label: None,
            acs_decoupled_url: self.config.acs_decoupled_url.clone(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::message::{DeviceChannel, MessageCategory};
    use crate::risk::BrowserInfo;

    fn config() -> AcsConfig {
        AcsConfig {
            acs_operator_id: "openpay-acs".into(),
            acs_reference_number: "3DS_LOA_ACS_OPNP_020300_00001".into(),
            acs_url: "https://acs.openpay.example/challenge".into(),
            acs_decoupled_url: Some("https://acs.openpay.example/poll".into()),
        }
    }

    fn areq() -> AReq {
        AReq {
            message_version: "2.2.0".into(),
            three_ds_server_trans_id: Uuid::new_v4().to_string(),
            three_ds_server_ref_number: "3DS_LOA_SER_OPNP_020100_00001".into(),
            three_ds_server_url: "https://3ds.openpay.example/callback".into(),
            three_ds_requestor_id: "openpay_req_id".into(),
            three_ds_requestor_name: "OpenPay Merchant".into(),
            three_ds_requestor_challenge_ind: "01".into(),
            message_category: MessageCategory::Payment,
            device_channel: DeviceChannel::Browser,
            acct_number: "4111111111111111".into(),
            acct_type: None,
            merchant_name: "OpenPay".into(),
            mcc: "5411".into(),
            acquirer_bin: "400000".into(),
            acquirer_merchant_id: "MID-001".into(),
            merchant_country_code: "840".into(),
            purchase_currency: "840".into(),
            purchase_amount: "1234".into(),
            purchase_exponent: 2,
            purchase_date: "20260524093000".into(),
            browser_info: Some(BrowserInfo::sample()),
            sdk_app_id: None,
            sdk_ephem_pub_key: None,
            sdk_reference_number: None,
            sdk_trans_id: None,
            sdk_max_timeout: None,
            three_ri_ind: None,
            three_ds_req_auth_method: None,
            decoupled_auth_ind: None,
            decoupled_auth_max_time: None,
            white_list_status: None,
            spc_incomp: None,
            delegated_auth_data: None,
            merchant_risk_indicator: None,
            acct_info: None,
            extensions: vec![],
        }
    }

    #[test]
    fn frictionless_ares_has_cavv_and_no_acs_url() {
        let server = AcsServer::new(config());
        let r = server.build_ares(
            &areq(),
            TransactionStatus::Authenticated,
            Some("CAVV==".into()),
            Some("05".into()),
        );
        assert_eq!(r.trans_status, "Y");
        assert_eq!(r.authentication_value.as_deref(), Some("CAVV=="));
        assert!(r.acs_url.is_none());
    }

    #[test]
    fn challenge_ares_has_acs_url_and_mandated_flag() {
        let server = AcsServer::new(config());
        let r = server.build_ares(&areq(), TransactionStatus::ChallengeRequired, None, None);
        assert_eq!(r.trans_status, "C");
        assert!(r.acs_url.is_some());
        assert_eq!(r.acs_challenge_mandated.as_deref(), Some("Y"));
    }

    #[test]
    fn build_final_cres_signals_completion() {
        let server = AcsServer::new(config());
        let c = server.build_final_cres("acs-1", "tid-1", TransactionStatus::Authenticated);
        assert_eq!(c.challenge_completion_ind.as_deref(), Some("Y"));
        assert_eq!(c.trans_status.as_deref(), Some("Y"));
    }

    #[test]
    fn auth_method_wire_values() {
        assert_eq!(AcsAuthMethod::StaticOtp.as_wire(), "01");
        assert_eq!(AcsAuthMethod::OobBiometric.as_wire(), "04");
        assert_eq!(AcsAuthMethod::Fido.as_wire(), "06");
    }
}
