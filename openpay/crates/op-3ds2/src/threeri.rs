//! 3DS Requestor Initiated (3RI) flow.
//!
//! 3RI is the protocol category for *subsequent* merchant-initiated
//! transactions: a recurring subscription charge, an instalment, a
//! card-update probe. The cardholder is not present. The original
//! cardholder-initiated authentication is referenced by id so the
//! issuer can re-issue the cryptogram without a fresh challenge.
//!
//! Available 2.2.0+. The `threeRIInd` field carries the subcategory.

use serde::{Deserialize, Serialize};

/// EMVCo `threeRIInd` values (subcategory of the 3RI flow).
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ThreeRiCategory {
    /// `"01"` Recurring transaction.
    Recurring,
    /// `"02"` Instalment.
    Instalment,
    /// `"03"` Add card to wallet.
    AddCard,
    /// `"04"` Maintain card information.
    MaintainCard,
    /// `"05"` Account verification.
    AccountVerification,
    /// `"06"` Split / delayed shipment.
    SplitShipment,
    /// `"07"` Top-up.
    TopUp,
    /// `"08"` Mail order.
    MailOrder,
    /// `"09"` Telephone order.
    TelephoneOrder,
    /// `"10"` Whitelist status check.
    WhitelistStatusCheck,
    /// `"11"` Other payment.
    OtherPayment,
}

impl ThreeRiCategory {
    /// Two-digit wire string per EMVCo.
    #[must_use]
    pub const fn as_wire(self) -> &'static str {
        match self {
            Self::Recurring => "01",
            Self::Instalment => "02",
            Self::AddCard => "03",
            Self::MaintainCard => "04",
            Self::AccountVerification => "05",
            Self::SplitShipment => "06",
            Self::TopUp => "07",
            Self::MailOrder => "08",
            Self::TelephoneOrder => "09",
            Self::WhitelistStatusCheck => "10",
            Self::OtherPayment => "11",
        }
    }
}

/// 3RI request envelope handed by the orchestrator to the AReq codec.
#[derive(Debug, Clone)]
pub struct ThreeRiRequest {
    /// Subcategory.
    pub category: ThreeRiCategory,
    /// Reference to the original cardholder-initiated authentication
    /// (the `acsTransID` returned on the CIT). The ACS uses this to
    /// scope the subsequent cryptogram.
    pub initial_acs_trans_id: String,
    /// PAN being charged.
    pub pan: String,
    /// Acquirer BIN.
    pub acquirer_bin: String,
    /// Merchant name on the cardholder statement.
    pub merchant_name: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn three_ri_wire_values_are_two_digit() {
        for c in [
            ThreeRiCategory::Recurring,
            ThreeRiCategory::Instalment,
            ThreeRiCategory::AddCard,
            ThreeRiCategory::MaintainCard,
            ThreeRiCategory::AccountVerification,
            ThreeRiCategory::SplitShipment,
            ThreeRiCategory::TopUp,
            ThreeRiCategory::MailOrder,
            ThreeRiCategory::TelephoneOrder,
            ThreeRiCategory::WhitelistStatusCheck,
            ThreeRiCategory::OtherPayment,
        ] {
            let w = c.as_wire();
            assert_eq!(w.len(), 2);
            assert!(w.chars().all(|c| c.is_ascii_digit()));
        }
    }

    #[test]
    fn recurring_is_01() {
        assert_eq!(ThreeRiCategory::Recurring.as_wire(), "01");
    }
}
