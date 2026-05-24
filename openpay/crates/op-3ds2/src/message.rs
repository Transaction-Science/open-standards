//! 3-D Secure 2.x message catalogue.
//!
//! Implements the AReq / ARes / CReq / CRes / RReq / RRes / Erro
//! payloads that travel between the 3DS Requestor, 3DS Server,
//! Directory Server (DS) and Access Control Server (ACS).
//!
//! Field names follow the EMVCo 3-D Secure Protocol and Core
//! Functions Specification, sections 6 (Message Detail) and 7
//! (Data Element Detail). Fields are camelCase in the JSON wire
//! format and we use `#[serde(rename_all = "camelCase")]` plus
//! per-field `rename` overrides for the handful of names that don't
//! map cleanly (e.g. `acctID`, `acctNumber`, `acsURL`).
//!
//! ## Validation
//!
//! [`AReq::validate`] runs the per-version rule lookup from
//! [`crate::version::field_rule`] and additionally enforces the
//! device-channel conditionals: `browserInfo` is required for browser
//! flows, `sdkAppID`/`sdkEphemPubKey`/`sdkTransID` are required for
//! app flows, and 3RI flows must populate `threeRIInd`.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::risk::{AccountInfo, BrowserInfo, MerchantRiskIndicator};
use crate::version::{FieldRule, ProtocolVersion, field_rule};

/// All 3DS 2.x messages, tagged by `messageType`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "messageType")]
pub enum ThreeDsMessage {
    /// Authentication Request (3DS Server → DS → ACS).
    AReq(Box<AReq>),
    /// Authentication Response (ACS → DS → 3DS Server).
    ARes(Box<ARes>),
    /// Challenge Request (3DS Server → ACS, via cardholder browser/app).
    CReq(Box<CReq>),
    /// Challenge Response (ACS → 3DS Server, via cardholder browser/app).
    CRes(Box<CRes>),
    /// Results Request (ACS → DS → 3DS Server, terminal-state report).
    RReq(Box<RReq>),
    /// Results Response (3DS Server → DS → ACS, acknowledgement).
    RRes(Box<RRes>),
    /// Erro — protocol-level error message.
    #[serde(rename = "Erro")]
    ErrorMessage(Box<ErrorMessage>),
}

/// `deviceChannel` discriminator from the EMVCo spec.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeviceChannel {
    /// `"01"` — app-based 3DS, payload comes from an SDK on a mobile
    /// device.
    #[serde(rename = "01")]
    App,
    /// `"02"` — browser-based 3DS, payload comes from a web checkout.
    #[serde(rename = "02")]
    Browser,
    /// `"03"` — 3RI (3DS Requestor Initiated), no cardholder present.
    #[serde(rename = "03")]
    ThreeRi,
}

/// `messageCategory` discriminator.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum MessageCategory {
    /// `"01"` — Payment Authentication (PA). The cardholder is paying.
    #[serde(rename = "01")]
    Payment,
    /// `"02"` — Non-Payment Authentication (NPA). Account-binding,
    /// add-card, no money movement.
    #[serde(rename = "02")]
    NonPayment,
}

/// **AReq** — Authentication Request.
///
/// 130+ fields per the EMVCo catalogue. We carry the load-bearing
/// subset by name and stash the rest in [`AReq::extensions`] so the
/// codec is forward-compatible with scheme-specific addenda.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct AReq {
    /// Spec version this message targets. Wire field: `messageVersion`.
    pub message_version: String,

    /// Server-side transaction id. UUIDv4. Required.
    #[serde(rename = "threeDSServerTransID")]
    pub three_ds_server_trans_id: String,

    /// 3DS Server reference number (assigned by EMVCo on registration).
    #[serde(rename = "threeDSServerRefNumber")]
    pub three_ds_server_ref_number: String,

    /// Operator URL the DS should use for the RReq callback.
    #[serde(rename = "threeDSServerURL")]
    pub three_ds_server_url: String,

    /// Requestor (merchant) id assigned by the DS.
    #[serde(rename = "threeDSRequestorID")]
    pub three_ds_requestor_id: String,

    /// Requestor name assigned by the DS.
    #[serde(rename = "threeDSRequestorName")]
    pub three_ds_requestor_name: String,

    /// `"01"` (no preference), `"02"` (no challenge requested),
    /// `"03"` (challenge requested),
    /// `"04"` (challenge mandated for regulatory reasons),
    /// `"05"` (no challenge requested, TRA exemption),
    /// `"06"` (no challenge requested, data-share only),
    /// `"07"` (no challenge requested, SCP exemption),
    /// `"08"` (no challenge requested, low-value exemption),
    /// `"09"` (no challenge requested, trusted-beneficiary).
    #[serde(rename = "threeDSRequestorChallengeInd")]
    pub three_ds_requestor_challenge_ind: String,

    /// PA / NPA category. Wire: `messageCategory`.
    pub message_category: MessageCategory,

    /// Where the user is. Wire: `deviceChannel`.
    pub device_channel: DeviceChannel,

    /// PAN of the card being authenticated. Wire: `acctNumber`.
    pub acct_number: String,

    /// Cardholder type: `"01"` Not a credit account, `"02"` Credit,
    /// `"03"` Debit.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acct_type: Option<String>,

    /// Merchant name as it appears on the cardholder's statement.
    pub merchant_name: String,

    /// MCC (ISO 18245 four-digit numeric).
    pub mcc: String,

    /// Acquirer BIN (assigned by the scheme).
    #[serde(rename = "acquirerBIN")]
    pub acquirer_bin: String,

    /// Acquirer merchant id assigned by the acquirer.
    pub acquirer_merchant_id: String,

    /// Three-digit numeric ISO 3166-1.
    pub merchant_country_code: String,

    /// Three-digit numeric ISO 4217.
    pub purchase_currency: String,

    /// Whole-number amount in the currency's minor units, as a string.
    /// The 3DS spec carries it as a numeric string (max 48 digits).
    pub purchase_amount: String,

    /// Number of decimal places the currency has. ISO 4217 exponent.
    pub purchase_exponent: u8,

    /// Date of the purchase in `YYYYMMDDHHMMSS` UTC format.
    pub purchase_date: String,

    /// Browser data envelope. Required when `deviceChannel == Browser`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub browser_info: Option<BrowserInfo>,

    /// SDK app id. Required when `deviceChannel == App`.
    #[serde(rename = "sdkAppID", skip_serializing_if = "Option::is_none")]
    pub sdk_app_id: Option<String>,

    /// SDK ephemeral public key (JWK). Required when
    /// `deviceChannel == App`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sdk_ephem_pub_key: Option<String>,

    /// SDK reference number assigned to the SDK build by EMVCo.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sdk_reference_number: Option<String>,

    /// SDK transaction id. UUIDv4. Required when
    /// `deviceChannel == App`.
    #[serde(rename = "sdkTransID", skip_serializing_if = "Option::is_none")]
    pub sdk_trans_id: Option<String>,

    /// Max time (minutes) the SDK is willing to wait for the ARes.
    /// Required when `deviceChannel == App`. Minimum value 5.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub sdk_max_timeout: Option<u32>,

    /// 3RI indicator. Required when `deviceChannel == ThreeRi`.
    /// `"01"` Recurring, `"02"` Installment, `"03"` Add card,
    /// `"04"` Maintain card, etc.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub three_ri_ind: Option<String>,

    /// Authentication method used by the requestor to authenticate the
    /// cardholder before this transaction. `"01"` No 3DSR auth,
    /// `"02"` Login w/ requestor's own credentials, `"03"` Federated id,
    /// `"04"` Issuer credentials, `"05"` Third-party authentication,
    /// `"06"` FIDO authenticator.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub three_ds_req_auth_method: Option<String>,

    /// Decoupled authentication indicator. `"Y"` requested,
    /// `"N"` not requested. 2.2.0+.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoupled_auth_ind: Option<String>,

    /// Max minutes the requestor is willing to wait for the decoupled
    /// outcome. 2.2.0+.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub decoupled_auth_max_time: Option<u32>,

    /// Whitelisting / trusted-beneficiary status. `"01"`-`"04"`. 2.2.0+.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub white_list_status: Option<String>,

    /// SPC (Secure Payment Confirmation) incompatibility indicator.
    /// 2.3.0 only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub spc_incomp: Option<String>,

    /// Delegated authentication evidence. 2.3.0 only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub delegated_auth_data: Option<String>,

    /// Merchant risk indicator envelope.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub merchant_risk_indicator: Option<MerchantRiskIndicator>,

    /// Account information envelope.
    #[serde(rename = "acctInfo", skip_serializing_if = "Option::is_none")]
    pub acct_info: Option<AccountInfo>,

    /// Forward-compatible bag for scheme extensions (Visa CIT/MIT
    /// indicators, Mastercard Identity Check addenda, etc.).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extensions: Vec<MessageExtension>,
}

impl AReq {
    /// Validate the message against the per-version field rules and
    /// device-channel conditionals. Returns the first violation.
    pub fn validate(&self, version: ProtocolVersion) -> Result<()> {
        // Version-wide constants. (Spec required everywhere.)
        if self.three_ds_server_trans_id.is_empty() {
            return Err(Error::MissingField {
                version,
                field: "threeDSServerTransID",
            });
        }
        if self.message_version.is_empty() {
            return Err(Error::MissingField {
                version,
                field: "messageVersion",
            });
        }
        if self.acct_number.is_empty() {
            return Err(Error::MissingField {
                version,
                field: "acctNumber",
            });
        }

        // Device-channel conditionals.
        match self.device_channel {
            DeviceChannel::Browser => {
                if self.browser_info.is_none() {
                    return Err(Error::MissingField {
                        version,
                        field: "browserInfo",
                    });
                }
            }
            DeviceChannel::App => {
                if self.sdk_app_id.is_none() {
                    return Err(Error::MissingField {
                        version,
                        field: "sdkAppID",
                    });
                }
                if self.sdk_ephem_pub_key.is_none() {
                    return Err(Error::MissingField {
                        version,
                        field: "sdkEphemPubKey",
                    });
                }
                if self.sdk_trans_id.is_none() {
                    return Err(Error::MissingField {
                        version,
                        field: "sdkTransID",
                    });
                }
                if self.sdk_max_timeout.is_none_or(|t| t < 5) {
                    return Err(Error::MissingField {
                        version,
                        field: "sdkMaxTimeout",
                    });
                }
            }
            DeviceChannel::ThreeRi => {
                if self.three_ri_ind.is_none() {
                    return Err(Error::MissingField {
                        version,
                        field: "threeRIInd",
                    });
                }
                // 3RI is forbidden in 2.1.0.
                if matches!(field_rule("threeRIInd", version), FieldRule::Forbidden) {
                    return Err(Error::ForbiddenField {
                        version,
                        field: "threeRIInd",
                    });
                }
            }
        }

        // Cross-version field-rule checks for the optional flags that
        // change classification across versions.
        if self.spc_incomp.is_some()
            && matches!(field_rule("spcIncomp", version), FieldRule::Forbidden)
        {
            return Err(Error::ForbiddenField {
                version,
                field: "spcIncomp",
            });
        }
        if self.decoupled_auth_ind.is_some()
            && matches!(field_rule("decoupledAuthInd", version), FieldRule::Forbidden)
        {
            return Err(Error::ForbiddenField {
                version,
                field: "decoupledAuthInd",
            });
        }
        if self.delegated_auth_data.is_some()
            && matches!(field_rule("delegatedAuthData", version), FieldRule::Forbidden)
        {
            return Err(Error::ForbiddenField {
                version,
                field: "delegatedAuthData",
            });
        }
        if self.white_list_status.is_some()
            && matches!(field_rule("whiteListStatus", version), FieldRule::Forbidden)
        {
            return Err(Error::ForbiddenField {
                version,
                field: "whiteListStatus",
            });
        }

        Ok(())
    }
}

/// **ARes** — Authentication Response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ARes {
    /// Wire field: `messageVersion`.
    pub message_version: String,
    #[serde(rename = "threeDSServerTransID")]
    /// Echo of the AReq id.
    pub three_ds_server_trans_id: String,
    #[serde(rename = "dsTransID")]
    /// DS-assigned id, UUIDv4.
    pub ds_trans_id: String,
    #[serde(rename = "acsTransID")]
    /// ACS-assigned id, UUIDv4.
    pub acs_trans_id: String,
    /// One of [`crate::auth_response::TransactionStatus`].
    pub trans_status: String,
    /// ACS-issued reference number (assigned by EMVCo on registration).
    #[serde(rename = "acsReferenceNumber")]
    pub acs_reference_number: String,
    /// ACS operator id.
    #[serde(rename = "acsOperatorID")]
    pub acs_operator_id: String,
    /// URL the cardholder browser/app should POST CReq to when a
    /// challenge is required.
    #[serde(rename = "acsURL", skip_serializing_if = "Option::is_none")]
    pub acs_url: Option<String>,
    /// HTML-formatted ACS challenge mandated string. Optional unless
    /// `transStatus == "C"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acs_challenge_mandated: Option<String>,
    /// Cardholder authentication value — the cryptogram the acquirer
    /// will forward in the auth request. Base64. Optional unless
    /// `transStatus == "Y"` or `"A"`.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authentication_value: Option<String>,
    /// Electronic Commerce Indicator. Scheme-specific.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eci: Option<String>,
    /// ACS-assigned signature for the ARes (used by the SDK to verify
    /// the ARes hasn't been tampered with). JWS compact.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acs_signed_content: Option<String>,
    /// Reason transStatus == "U" (system unavailable) or "R" (rejected).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trans_status_reason: Option<String>,
    /// Forward-compat bag.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub extensions: Vec<MessageExtension>,
}

/// **CReq** — Challenge Request.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CReq {
    /// Wire field: `messageVersion`.
    pub message_version: String,
    #[serde(rename = "threeDSServerTransID")]
    /// Echo from AReq/ARes.
    pub three_ds_server_trans_id: String,
    #[serde(rename = "acsTransID")]
    /// Echo from ARes.
    pub acs_trans_id: String,
    /// Challenge window size. `"01"` 250×400, `"02"` 390×400,
    /// `"03"` 500×600, `"04"` 600×400, `"05"` full screen.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub challenge_window_size: Option<String>,
    /// SDK-side counter, increments per challenge transmission.
    #[serde(rename = "sdkCounterStoA", skip_serializing_if = "Option::is_none")]
    pub sdk_counter_s_to_a: Option<String>,
    /// Cardholder response payload (filled when re-submitting after
    /// the ACS rendered the challenge UI).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub challenge_data_entry: Option<String>,
    /// Cancel indicator: `"01"` cardholder selected cancel.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub challenge_cancel: Option<String>,
    /// Resend challenge indicator: `"Y"` resend.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub resend_challenge: Option<String>,
}

/// **CRes** — Challenge Response.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct CRes {
    /// Wire field: `messageVersion`.
    pub message_version: String,
    #[serde(rename = "threeDSServerTransID")]
    /// Echo.
    pub three_ds_server_trans_id: String,
    #[serde(rename = "acsTransID")]
    /// Echo.
    pub acs_trans_id: String,
    /// ACS-side counter, increments per challenge transmission.
    #[serde(rename = "acsCounterAtoS", skip_serializing_if = "Option::is_none")]
    pub acs_counter_a_to_s: Option<String>,
    /// HTML body, base64-encoded for HTML challenge mode.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub acs_html: Option<String>,
    /// "Y"/"N". When "Y" the challenge has completed and the
    /// cardholder may continue.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub challenge_completion_ind: Option<String>,
    /// Transaction status — same enum as on ARes. Populated on the
    /// final CRes only.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trans_status: Option<String>,
    /// Reason transStatus == "U" / "R" / "N".
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trans_status_reason: Option<String>,
    /// OOB-mode app URL the SDK should open.
    #[serde(rename = "oobAppURL", skip_serializing_if = "Option::is_none")]
    pub oob_app_url: Option<String>,
    /// OOB-mode app label (button text).
    #[serde(rename = "oobAppLabel", skip_serializing_if = "Option::is_none")]
    pub oob_app_label: Option<String>,
    /// Decoupled-mode polling URL.
    #[serde(rename = "acsDecConURL", skip_serializing_if = "Option::is_none")]
    pub acs_decoupled_url: Option<String>,
}

/// **RReq** — Results Request (ACS → DS → 3DS Server, terminal report).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RReq {
    /// Wire field: `messageVersion`.
    pub message_version: String,
    #[serde(rename = "threeDSServerTransID")]
    /// Echo.
    pub three_ds_server_trans_id: String,
    #[serde(rename = "dsTransID")]
    /// Echo.
    pub ds_trans_id: String,
    #[serde(rename = "acsTransID")]
    /// Echo.
    pub acs_trans_id: String,
    /// Final transaction status.
    pub trans_status: String,
    /// Cryptogram (base64).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authentication_value: Option<String>,
    /// Electronic Commerce Indicator.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub eci: Option<String>,
    /// Reason for non-success status.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub trans_status_reason: Option<String>,
    /// Authentication type: `"01"` static, `"02"` dynamic, `"03"` OOB,
    /// `"04"` decoupled.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authentication_type: Option<String>,
    /// Interaction counter — number of challenge cycles.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub interaction_counter: Option<String>,
    /// Date of completion in `YYYYMMDDHHMMSS` UTC.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub authentication_method: Option<String>,
}

/// **RRes** — Results Response (acknowledgement of the RReq).
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct RRes {
    /// Wire field: `messageVersion`.
    pub message_version: String,
    #[serde(rename = "threeDSServerTransID")]
    /// Echo.
    pub three_ds_server_trans_id: String,
    /// "01" Results Acknowledged.
    pub results_status: String,
}

/// **Erro** — Protocol-level error message.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ErrorMessage {
    /// Wire field: `messageVersion`.
    pub message_version: String,
    #[serde(rename = "threeDSServerTransID")]
    /// Echo (when available).
    pub three_ds_server_trans_id: String,
    /// EMVCo five-digit error code.
    pub error_code: String,
    /// Free-form description.
    pub error_description: String,
    /// Component that raised: `"A"` ACS, `"C"` 3DS SDK,
    /// `"D"` DS, `"S"` 3DS Server.
    pub error_component: String,
    /// Optional field name that triggered the error.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_detail: Option<String>,
    /// Spec version this message indicates was expected.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_message_type: Option<String>,
    /// Timestamp the error was raised at.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error_raised_at: Option<DateTime<Utc>>,
}

/// Generic scheme-specific extension envelope, attached as
/// `extensions[]` to AReq / ARes per EMVCo.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct MessageExtension {
    /// Vendor-prefixed extension name.
    pub name: String,
    /// EMVCo-assigned extension id.
    pub id: String,
    /// Whether the receiver must understand this extension or fail.
    pub criticality_indicator: bool,
    /// Free-form data block (typically a nested JSON object stringified).
    pub data: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::risk::BrowserInfo;

    fn sample_browser_areq() -> AReq {
        AReq {
            message_version: "2.2.0".into(),
            three_ds_server_trans_id: "11111111-1111-1111-1111-111111111111".into(),
            three_ds_server_ref_number: "3DS_LOA_SER_OPNP_020100_00001".into(),
            three_ds_server_url: "https://3ds.openpay.example/callback".into(),
            three_ds_requestor_id: "openpay_req_id".into(),
            three_ds_requestor_name: "OpenPay Merchant".into(),
            three_ds_requestor_challenge_ind: "01".into(),
            message_category: MessageCategory::Payment,
            device_channel: DeviceChannel::Browser,
            acct_number: "4111111111111111".into(),
            acct_type: Some("02".into()),
            merchant_name: "OpenPay Merchant".into(),
            mcc: "5411".into(),
            acquirer_bin: "400000".into(),
            acquirer_merchant_id: "MID-OPENPAY-001".into(),
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
    fn browser_areq_validates_at_2_2() {
        let a = sample_browser_areq();
        assert!(a.validate(ProtocolVersion::V2_2).is_ok());
        assert!(a.validate(ProtocolVersion::V2_3).is_ok());
        assert!(a.validate(ProtocolVersion::V2_1).is_ok());
    }

    #[test]
    fn browser_areq_missing_browser_info_fails() {
        let mut a = sample_browser_areq();
        a.browser_info = None;
        let err = a.validate(ProtocolVersion::V2_2).unwrap_err();
        assert!(matches!(
            err,
            Error::MissingField {
                field: "browserInfo",
                ..
            }
        ));
    }

    #[test]
    fn app_areq_requires_sdk_fields() {
        let mut a = sample_browser_areq();
        a.device_channel = DeviceChannel::App;
        a.browser_info = None;
        let err = a.validate(ProtocolVersion::V2_2).unwrap_err();
        assert!(matches!(err, Error::MissingField { field: "sdkAppID", .. }));
    }

    #[test]
    fn threeri_forbidden_in_2_1() {
        let mut a = sample_browser_areq();
        a.device_channel = DeviceChannel::ThreeRi;
        a.three_ri_ind = Some("01".into());
        a.browser_info = None;
        let err = a.validate(ProtocolVersion::V2_1).unwrap_err();
        assert!(matches!(
            err,
            Error::ForbiddenField {
                field: "threeRIInd",
                ..
            }
        ));
    }

    #[test]
    fn spc_forbidden_below_2_3() {
        let mut a = sample_browser_areq();
        a.spc_incomp = Some("Y".into());
        let err = a.validate(ProtocolVersion::V2_2).unwrap_err();
        assert!(matches!(
            err,
            Error::ForbiddenField {
                field: "spcIncomp",
                ..
            }
        ));
        // OK under 2.3.0.
        assert!(a.validate(ProtocolVersion::V2_3).is_ok());
    }

    #[test]
    fn decoupled_forbidden_in_2_1() {
        let mut a = sample_browser_areq();
        a.decoupled_auth_ind = Some("Y".into());
        let err = a.validate(ProtocolVersion::V2_1).unwrap_err();
        assert!(matches!(
            err,
            Error::ForbiddenField {
                field: "decoupledAuthInd",
                ..
            }
        ));
    }

    #[test]
    fn round_trip_browser_areq_json() {
        let a = sample_browser_areq();
        let json = serde_json::to_string(&a).unwrap();
        let parsed: AReq = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.acct_number, a.acct_number);
        assert_eq!(parsed.message_version, "2.2.0");
        assert!(json.contains("\"acctNumber\""));
        assert!(json.contains("\"threeDSServerTransID\""));
        assert!(json.contains("\"deviceChannel\":\"02\""));
    }

    #[test]
    fn round_trip_ares() {
        let r = ARes {
            message_version: "2.2.0".into(),
            three_ds_server_trans_id: "trans-123".into(),
            ds_trans_id: "ds-456".into(),
            acs_trans_id: "acs-789".into(),
            trans_status: "Y".into(),
            acs_reference_number: "3DS_LOA_ACS_VISA_020100_00001".into(),
            acs_operator_id: "visa-acs".into(),
            acs_url: None,
            acs_challenge_mandated: None,
            authentication_value: Some("BASE64CAVV==".into()),
            eci: Some("05".into()),
            acs_signed_content: None,
            trans_status_reason: None,
            extensions: vec![],
        };
        let json = serde_json::to_string(&r).unwrap();
        let back: ARes = serde_json::from_str(&json).unwrap();
        assert_eq!(back.eci.as_deref(), Some("05"));
        assert!(json.contains("\"transStatus\":\"Y\""));
    }

    #[test]
    fn round_trip_creq_cres() {
        let creq = CReq {
            message_version: "2.2.0".into(),
            three_ds_server_trans_id: "t".into(),
            acs_trans_id: "a".into(),
            challenge_window_size: Some("05".into()),
            sdk_counter_s_to_a: Some("001".into()),
            challenge_data_entry: None,
            challenge_cancel: None,
            resend_challenge: None,
        };
        let cres = CRes {
            message_version: "2.2.0".into(),
            three_ds_server_trans_id: "t".into(),
            acs_trans_id: "a".into(),
            acs_counter_a_to_s: Some("001".into()),
            acs_html: Some("PGh0bWw+".into()),
            challenge_completion_ind: Some("N".into()),
            trans_status: None,
            trans_status_reason: None,
            oob_app_url: None,
            oob_app_label: None,
            acs_decoupled_url: None,
        };
        assert!(serde_json::to_string(&creq).unwrap().contains("\"challengeWindowSize\""));
        assert!(serde_json::to_string(&cres).unwrap().contains("\"acsHTML\"") || serde_json::to_string(&cres).unwrap().contains("\"acsHtml\""));
    }

    #[test]
    fn round_trip_rreq_rres() {
        let rreq = RReq {
            message_version: "2.2.0".into(),
            three_ds_server_trans_id: "t".into(),
            ds_trans_id: "d".into(),
            acs_trans_id: "a".into(),
            trans_status: "Y".into(),
            authentication_value: Some("CAVV==".into()),
            eci: Some("05".into()),
            trans_status_reason: None,
            authentication_type: Some("02".into()),
            interaction_counter: Some("01".into()),
            authentication_method: None,
        };
        let rres = RRes {
            message_version: "2.2.0".into(),
            three_ds_server_trans_id: "t".into(),
            results_status: "01".into(),
        };
        let _ = serde_json::to_string(&rreq).unwrap();
        let _ = serde_json::to_string(&rres).unwrap();
    }

    #[test]
    fn error_message_round_trip() {
        let e = ErrorMessage {
            message_version: "2.2.0".into(),
            three_ds_server_trans_id: "t".into(),
            error_code: "203".into(),
            error_description: "Required field missing".into(),
            error_component: "S".into(),
            error_detail: Some("acctNumber".into()),
            error_message_type: Some("AReq".into()),
            error_raised_at: None,
        };
        let s = serde_json::to_string(&e).unwrap();
        let back: ErrorMessage = serde_json::from_str(&s).unwrap();
        assert_eq!(back.error_code, "203");
    }

    #[test]
    fn tagged_enum_round_trip() {
        let msg = ThreeDsMessage::AReq(Box::new(sample_browser_areq()));
        let s = serde_json::to_string(&msg).unwrap();
        assert!(s.contains("\"messageType\":\"AReq\""));
        let back: ThreeDsMessage = serde_json::from_str(&s).unwrap();
        match back {
            ThreeDsMessage::AReq(_) => {}
            _ => panic!("expected AReq variant"),
        }
    }
}
