//! Data-only flow (2.2.0+).
//!
//! In the data-only flow the 3DS Server sends an AReq with
//! `messageCategory == "01"` and `threeDSRequestorChallengeInd == "06"`
//! (no challenge requested, data-share only). The DS forwards the
//! risk envelope to the ACS, which returns a transaction-status of
//! `"I"` (informational only) — no authentication, no liability
//! shift, but the ACS uses the data to update its fraud model and
//! the cardholder gets no friction.
//!
//! Typical use cases:
//!
//! - Re-authentication for trusted recurring payments where SCA was
//!   completed on the initial CIT.
//! - Business-to-Account-Transfer (BAT) flows where the issuer wants
//!   risk visibility without exposing the cardholder to a challenge.
//! - Fraud-shielded card-on-file refresh.

use op_core::Money;
use serde::{Deserialize, Serialize};

use crate::message::{DeviceChannel, MessageCategory};

/// Compact data-only request the orchestrator hands to the codec.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataOnlyRequest {
    /// PAN being authenticated.
    pub pan: String,
    /// Amount of the underlying purchase (used by the ACS for risk
    /// scoring even though no authentication happens).
    pub amount: Money,
    /// Merchant name as it appears to the cardholder.
    pub merchant_name: String,
    /// Acquirer BIN.
    pub acquirer_bin: String,
    /// Channel — only browser-flow or 3RI in practice.
    pub device_channel: DeviceChannel,
}

impl DataOnlyRequest {
    /// `messageCategory` the AReq must carry.
    #[must_use]
    pub const fn message_category(&self) -> MessageCategory {
        MessageCategory::Payment
    }

    /// `threeDSRequestorChallengeInd` value for data-only flow.
    #[must_use]
    pub const fn challenge_indicator(&self) -> &'static str {
        "06"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::Currency;

    #[test]
    fn challenge_indicator_is_06_for_data_only() {
        let req = DataOnlyRequest {
            pan: "4111111111111111".into(),
            amount: Money::from_minor(2500, Currency::EUR),
            merchant_name: "OpenPay".into(),
            acquirer_bin: "400000".into(),
            device_channel: DeviceChannel::Browser,
        };
        assert_eq!(req.challenge_indicator(), "06");
        assert_eq!(req.message_category(), MessageCategory::Payment);
    }
}
