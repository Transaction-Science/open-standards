//! ISO 20022 status codes used in `pacs.002` and `pain.014`.
//!
//! Source of truth: ISO 20022 External Code Lists, quarterly publication
//! from the ISO 20022 Registration Authority. `FedNow` Operating Procedures
//! pin the subset every participant must implement.
//!
//! ## Transaction status (`TxSts`)
//!
//! Per `FedNow` Service Operating Procedures (June 2025) and the Payments
//! Canada RTR specification (May 2025), every `pacs.002` carries one of
//! the four-letter codes below in its `TxSts` element. PDNG is special:
//! it is sent by the Federal Reserve Banks when a value message is in
//! process or has been intercepted by the Federal Reserve Banks.

use core::fmt;

use serde::{Deserialize, Serialize};

use crate::error::Error;

/// `TxSts` — transaction status code. Closed enum: any other value is
/// rejected at parse time.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TransactionStatus {
    /// `ACTC` — Accepted (technical validation passed). Sent when the
    /// receiving FI has accepted the technical content but settlement
    /// has not yet occurred.
    AcceptedTechnical,
    /// `ACSC` — Accepted, settlement completed. The funds have moved.
    AcceptedSettled,
    /// `RJCT` — Rejected. Must carry a `StatusReasonInformation` with a
    /// reason code; see [`StatusReason`].
    Rejected,
    /// `PDNG` — Pending. The Fed (or another intermediary) is holding
    /// the message in process. Not a terminal state; another pacs.002
    /// follows.
    Pending,
    /// `RCVD` — Received. Acknowledgement of receipt only; used in some
    /// schemes for pre-validation flows.
    Received,
    /// `ACCP` — Accepted customer profile (consumer-side acceptance,
    /// e.g. RFP).
    AcceptedCustomer,
}

impl TransactionStatus {
    /// The four-letter ISO code that goes on the wire.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::AcceptedTechnical => "ACTC",
            Self::AcceptedSettled => "ACSC",
            Self::Rejected => "RJCT",
            Self::Pending => "PDNG",
            Self::Received => "RCVD",
            Self::AcceptedCustomer => "ACCP",
        }
    }

    /// Parse a four-letter ISO status code.
    ///
    /// # Errors
    /// Returns `Error::InvalidField` if the code is unknown.
    pub fn from_code(code: &str) -> Result<Self, Error> {
        match code {
            "ACTC" => Ok(Self::AcceptedTechnical),
            "ACSC" => Ok(Self::AcceptedSettled),
            "RJCT" => Ok(Self::Rejected),
            "PDNG" => Ok(Self::Pending),
            "RCVD" => Ok(Self::Received),
            "ACCP" => Ok(Self::AcceptedCustomer),
            other => Err(Error::InvalidField {
                field: "TxSts",
                reason: alloc::format!("unknown status code: {other}"),
            }),
        }
    }

    /// True if this status is terminal (no further status updates expected).
    #[must_use]
    pub const fn is_terminal(self) -> bool {
        matches!(self, Self::AcceptedSettled | Self::Rejected)
    }
}

impl fmt::Display for TransactionStatus {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.code())
    }
}

/// `StatusReasonInformation.Reason.Code` — the four-letter reason that
/// accompanies an `RJCT` status. This is a subset of the ISO 20022
/// `ExternalStatusReason1Code` list; we enumerate the ones that appear
/// in production `FedNow` / RTP / SEPA Instant traffic.
///
/// The full list has ~200 codes; we'll grow this enum as we encounter
/// them in conformance vectors. Unknown codes round-trip through the
/// [`Self::Other`] variant so we never lose data.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum StatusReason {
    /// `AC01` — incorrect account number.
    IncorrectAccountNumber,
    /// `AC03` — invalid creditor account number.
    InvalidCreditorAccountNumber,
    /// `AC04` — closed account number.
    ClosedAccountNumber,
    /// `AC06` — blocked account.
    BlockedAccount,
    /// `AG01` — transaction forbidden on this type of account.
    TransactionForbidden,
    /// `AM04` — insufficient funds.
    InsufficientFunds,
    /// `AM05` — duplicate payment.
    Duplicate,
    /// `BE01` — inconsistent with end-customer.
    InconsistentCustomer,
    /// `CUST` — requested by customer.
    RequestedByCustomer,
    /// `DUPL` — duplicate payment.
    DuplicatePayment,
    /// `FRAD` — fraudulent origin.
    Fraud,
    /// `MS03` — reason not specified.
    NotSpecified,
    /// `NARR` — narrative — see additional information field.
    Narrative,
    /// `TM01` — invalid cut-off time.
    InvalidCutoff,
    /// Any other code we haven't enumerated yet. Stored as the raw
    /// four-letter string so it survives a round-trip.
    Other(String),
}

impl StatusReason {
    /// Wire code.
    #[must_use]
    pub fn code(&self) -> &str {
        match self {
            Self::IncorrectAccountNumber => "AC01",
            Self::InvalidCreditorAccountNumber => "AC03",
            Self::ClosedAccountNumber => "AC04",
            Self::BlockedAccount => "AC06",
            Self::TransactionForbidden => "AG01",
            Self::InsufficientFunds => "AM04",
            Self::Duplicate => "AM05",
            Self::InconsistentCustomer => "BE01",
            Self::RequestedByCustomer => "CUST",
            Self::DuplicatePayment => "DUPL",
            Self::Fraud => "FRAD",
            Self::NotSpecified => "MS03",
            Self::Narrative => "NARR",
            Self::InvalidCutoff => "TM01",
            Self::Other(c) => c.as_str(),
        }
    }

    /// Parse from wire code. Unknown codes are preserved in [`Self::Other`].
    #[must_use]
    pub fn from_code(code: &str) -> Self {
        match code {
            "AC01" => Self::IncorrectAccountNumber,
            "AC03" => Self::InvalidCreditorAccountNumber,
            "AC04" => Self::ClosedAccountNumber,
            "AC06" => Self::BlockedAccount,
            "AG01" => Self::TransactionForbidden,
            "AM04" => Self::InsufficientFunds,
            "AM05" => Self::Duplicate,
            "BE01" => Self::InconsistentCustomer,
            "CUST" => Self::RequestedByCustomer,
            "DUPL" => Self::DuplicatePayment,
            "FRAD" => Self::Fraud,
            "MS03" => Self::NotSpecified,
            "NARR" => Self::Narrative,
            "TM01" => Self::InvalidCutoff,
            other => Self::Other(other.to_owned()),
        }
    }
}

// Bring in alloc::format for no_std-friendly String formatting.
extern crate alloc;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn status_round_trip_known_codes() {
        for code in ["ACTC", "ACSC", "RJCT", "PDNG", "RCVD", "ACCP"] {
            let status = TransactionStatus::from_code(code).unwrap();
            assert_eq!(status.code(), code, "round trip failed for {code}");
        }
    }

    #[test]
    fn status_rejects_unknown_code() {
        assert!(TransactionStatus::from_code("XXXX").is_err());
        assert!(TransactionStatus::from_code("").is_err());
        assert!(TransactionStatus::from_code("acsc").is_err()); // case-sensitive
    }

    #[test]
    fn terminal_states_are_acsc_and_rjct_only() {
        assert!(TransactionStatus::AcceptedSettled.is_terminal());
        assert!(TransactionStatus::Rejected.is_terminal());
        assert!(!TransactionStatus::Pending.is_terminal());
        assert!(!TransactionStatus::AcceptedTechnical.is_terminal());
    }

    #[test]
    fn reason_known_codes_round_trip() {
        let cases = [
            ("AC01", StatusReason::IncorrectAccountNumber),
            ("AM04", StatusReason::InsufficientFunds),
            ("FRAD", StatusReason::Fraud),
        ];
        for (code, expected) in cases {
            let parsed = StatusReason::from_code(code);
            assert_eq!(parsed, expected);
            assert_eq!(parsed.code(), code);
        }
    }

    #[test]
    fn reason_unknown_code_preserved() {
        let parsed = StatusReason::from_code("ZZ99");
        match &parsed {
            StatusReason::Other(c) => assert_eq!(c, "ZZ99"),
            _ => panic!("expected Other variant"),
        }
        assert_eq!(parsed.code(), "ZZ99");
    }

    #[test]
    fn status_display_emits_wire_code() {
        assert_eq!(format!("{}", TransactionStatus::AcceptedSettled), "ACSC");
    }
}
