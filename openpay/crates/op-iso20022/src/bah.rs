//! Business Application Header (`head.001.001.02`).
//!
//! Required on every ISO 20022 message sent across `FedNow`, RTP, SEPA
//! Instant, and most other modern rails. The BAH carries routing,
//! identification, and timestamps that are *outside* the business
//! message payload.
//!
//! Per `FedNow` Operating Procedures: the BAH is required for all ISO
//! 20022 messages sent across the `FedNow` Service.

use serde::{Deserialize, Serialize};
use time::OffsetDateTime;
use uuid::Uuid;

use crate::error::{Error, Result};

/// A simplified Business Application Header.
///
/// We don't expose every BAH field — only the ones every `FedNow` / RTP /
/// SEPA Instant message actually populates. The full ISO 20022 `head.001`
/// has optional fields (related messages, signature, copy duplicate
/// indicator) that we add as needed.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BusinessApplicationHeader {
    /// `Fr` — sender. ABA routing number for US rails, BIC for
    /// SWIFT/SEPA. We store it opaque; profile validators check format.
    pub from: PartyIdentification,
    /// `To` — receiver.
    pub to: PartyIdentification,
    /// `BizMsgIdr` — business message identifier. Caller-generated,
    /// unique within the sender / day, must follow the rail's format
    /// rules (e.g. `FedNow` IMAD).
    pub business_message_id: String,
    /// `MsgDefIdr` — exact message definition (e.g. `pacs.008.001.08`).
    /// Profile sets this; callers don't.
    pub message_definition_id: String,
    /// `CreDt` — creation date/time. UTC, ISO 8601 with timezone.
    pub creation_datetime: OffsetDateTime,
    /// `BizPrcgDt` — business processing date, optional.
    pub business_processing_date: Option<OffsetDateTime>,
    /// `CpyDplct` — `CODU` / `COPY` / `DUPL` if the message is a copy.
    pub copy_duplicate: Option<CopyDuplicate>,
    /// `PssblDplct` — true if this might be a duplicate of a prior message.
    pub possible_duplicate: bool,
}

/// `CpyDplct` enum from ISO 20022.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum CopyDuplicate {
    /// `CODU` — copy and possible duplicate.
    CopyDuplicate,
    /// `COPY` — copy, original was already sent.
    Copy,
    /// `DUPL` — duplicate, original may not have been received.
    Duplicate,
}

/// `Fr` / `To` party. Either a BIC, an account-servicer ID (routing
/// number for US), or a structured organisation identification.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum PartyIdentification {
    /// SWIFT BIC (8 or 11 alphanumeric).
    Bic(String),
    /// Member identification — US ABA routing number for FedNow/RTP/Wire.
    /// Format: 9 digits with valid checksum.
    AbaRoutingNumber(String),
    /// Generic clearing system member id (used by non-US rails).
    ClearingSystemMemberId {
        /// Clearing system, e.g. `"USABA"`, `"CHIPS"`, `"FedNow"`.
        clearing_system: String,
        /// Member id within that clearing system.
        member_id: String,
    },
}

impl PartyIdentification {
    /// Validate format. Called by profile validators before serializing.
    ///
    /// # Errors
    /// Returns `Error::InvalidField` on format failure.
    pub fn validate(&self) -> Result<()> {
        match self {
            Self::Bic(s) => {
                let len = s.len();
                if len != 8 && len != 11 {
                    return Err(Error::InvalidField {
                        field: "BIC",
                        reason: alloc::format!("BIC must be 8 or 11 chars, got {len}"),
                    });
                }
                if !s.bytes().all(|b| b.is_ascii_alphanumeric()) {
                    return Err(Error::InvalidField {
                        field: "BIC",
                        reason: "BIC must be ASCII alphanumeric".into(),
                    });
                }
                Ok(())
            }
            Self::AbaRoutingNumber(s) => validate_aba(s),
            Self::ClearingSystemMemberId {
                clearing_system,
                member_id,
            } => {
                if clearing_system.is_empty() {
                    return Err(Error::InvalidField {
                        field: "ClrSysId",
                        reason: "clearing system must not be empty".into(),
                    });
                }
                if member_id.is_empty() {
                    return Err(Error::InvalidField {
                        field: "MmbId",
                        reason: "member id must not be empty".into(),
                    });
                }
                Ok(())
            }
        }
    }
}

/// ABA routing number checksum: 9 digits, weighted sum of (3,7,1,3,7,1,3,7,1)
/// must be a multiple of 10. This catches transposition errors that
/// would otherwise route money to the wrong bank.
///
/// Reference: Federal Reserve ABA routing number specification.
fn validate_aba(s: &str) -> Result<()> {
    if s.len() != 9 {
        return Err(Error::InvalidField {
            field: "ABA",
            reason: alloc::format!("ABA must be 9 digits, got {}", s.len()),
        });
    }
    let digits: Result<Vec<u32>> = s
        .chars()
        .map(|c| {
            c.to_digit(10).ok_or(Error::InvalidField {
                field: "ABA",
                reason: "ABA must be digits only".into(),
            })
        })
        .collect();
    let digits = digits?;
    let weights = [3u32, 7, 1, 3, 7, 1, 3, 7, 1];
    let sum: u32 = digits.iter().zip(weights.iter()).map(|(d, w)| d * w).sum();
    if !sum.is_multiple_of(10) {
        return Err(Error::InvalidField {
            field: "ABA",
            reason: alloc::format!("ABA checksum failed (sum {sum} mod 10 != 0)"),
        });
    }
    Ok(())
}

impl BusinessApplicationHeader {
    /// Construct a new BAH with a generated message id and the current
    /// UTC timestamp.
    #[must_use]
    pub fn new(
        from: PartyIdentification,
        to: PartyIdentification,
        message_definition_id: impl Into<String>,
    ) -> Self {
        Self {
            from,
            to,
            business_message_id: Uuid::now_v7().simple().to_string(),
            message_definition_id: message_definition_id.into(),
            creation_datetime: OffsetDateTime::now_utc(),
            business_processing_date: None,
            copy_duplicate: None,
            possible_duplicate: false,
        }
    }

    /// Run all format validations. Callers should invoke this before
    /// serializing or sending to a rail.
    ///
    /// # Errors
    /// First validation failure.
    pub fn validate(&self) -> Result<()> {
        self.from.validate()?;
        self.to.validate()?;
        if self.business_message_id.is_empty() {
            return Err(Error::MissingField("BizMsgIdr"));
        }
        if self.message_definition_id.is_empty() {
            return Err(Error::MissingField("MsgDefIdr"));
        }
        Ok(())
    }
}

extern crate alloc;

#[cfg(test)]
mod tests {
    use super::*;

    /// Known-good ABA routing numbers. Source: Federal Reserve public test
    /// vectors and several major US banks' published numbers.
    const VALID_ABAS: &[&str] = &[
        "021000021", // JPMorgan Chase, New York
        "026009593", // Bank of America, New York
        "111000025", // Federal Reserve Bank of Dallas
        "121000358", // Bank of America, San Francisco
        "091000019", // US Bank, Minneapolis
    ];

    #[test]
    fn known_good_abas_pass() {
        for aba in VALID_ABAS {
            assert!(
                validate_aba(aba).is_ok(),
                "expected {aba} to validate as a real ABA"
            );
        }
    }

    #[test]
    fn bad_aba_length_rejected() {
        assert!(validate_aba("12345678").is_err()); // 8
        assert!(validate_aba("1234567890").is_err()); // 10
    }

    #[test]
    fn non_digit_aba_rejected() {
        assert!(validate_aba("12345678X").is_err());
    }

    #[test]
    fn bad_checksum_rejected() {
        // Take a real ABA and bump one digit; checksum must fail.
        assert!(validate_aba("021000022").is_err());
        assert!(validate_aba("000000000").is_ok()); // edge: sum 0, divisible
        assert!(validate_aba("000000001").is_err());
    }

    #[test]
    fn bic_length_validated() {
        assert!(
            PartyIdentification::Bic("CHASUS33".into())
                .validate()
                .is_ok()
        ); // 8
        assert!(
            PartyIdentification::Bic("CHASUS33XXX".into())
                .validate()
                .is_ok()
        ); // 11
        assert!(
            PartyIdentification::Bic("CHASUS".into())
                .validate()
                .is_err()
        ); // 6
        assert!(
            PartyIdentification::Bic("CHASUS33!!!".into())
                .validate()
                .is_err()
        ); // non-alphanumeric
    }

    #[test]
    fn bah_construction_generates_msg_id() {
        let bah = BusinessApplicationHeader::new(
            PartyIdentification::AbaRoutingNumber("021000021".into()),
            PartyIdentification::AbaRoutingNumber("026009593".into()),
            "pacs.008.001.08",
        );
        assert!(!bah.business_message_id.is_empty());
        assert_eq!(bah.message_definition_id, "pacs.008.001.08");
        assert!(bah.validate().is_ok());
    }
}
