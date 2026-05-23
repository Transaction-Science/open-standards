//! `FedNow` status / reason-code mapping.
//!
//! `FedNow` uses standard ISO 20022 `pacs.002` `TransactionStatus`
//! codes verbatim. Per the `FedNow` Service Operating Procedures v3.2:
//!
//! - `ACTC` — accepted technical validation (`FedNow` received and
//!   validated; settlement starting)
//! - `ACSC` — accepted, settlement completed (funds at the receiver)
//! - `RJCT` — rejected (`reason_code` carries why)
//! - `PDNG` — pending (`FedNow` is still working on it)
//!
//! Reason codes (when status = `RJCT`) come from
//! `ExternalStatusReason1Code`. Common values: `AC03` invalid creditor
//! account, `AM04` insufficient funds, `BE05` unrecognized initiating
//! party.

use crate::acquirer::A2aStatus;
use crate::error::{Error, Result};

/// Map a `FedNow` `TransactionStatus` (from a pacs.002) to [`A2aStatus`].
///
/// # Errors
/// `Error::UnknownStatus` for codes not in the documented set.
pub fn map_transaction_status(code: &str) -> Result<A2aStatus> {
    Ok(match code {
        "ACSC" => A2aStatus::Settled,
        "ACTC" => A2aStatus::Accepted,
        "ACSP" => A2aStatus::InProgress,
        "PDNG" => A2aStatus::Pending,
        "RJCT" => A2aStatus::Rejected,
        other => return Err(Error::UnknownStatus(other.to_owned())),
    })
}

/// Reason codes that `FedNow` returns are passed through unchanged in the
/// `A2aDecision.reason_code` field. We don't try to map them — there
/// are 100+ values in `ExternalStatusReason1Code` and the caller knows
/// best what to do with each.
///
/// This function exists only to provide a stable predicate: is this
/// reason code in the documented set?
#[must_use]
pub fn is_known_reason_code(code: &str) -> bool {
    // ExternalStatusReason1Code values from ISO 20022 external code
    // lists. Sample the most-frequent codes here; the full list is
    // maintained at iso20022.org and changes quarterly.
    matches!(
        code,
        // Account issues
        "AC01" | "AC03" | "AC04" | "AC06" | "AC13" | "AC14" |
        // Amount issues
        "AM01" | "AM02" | "AM03" | "AM04" | "AM05" |
        // Party issues
        "BE01" | "BE04" | "BE05" | "BE06" | "BE07" | "BE08" |
        // Duplicate / time
        "AGNT" | "DUPL" | "DT01" | "DT02" |
        // Format / regulatory
        "FF01" | "FF02" | "FF03" | "FF04" | "FF05" |
        "FR01" | "FRAD" | "FOCR" |
        // Account closed / blocked
        "MS02" | "MS03" |
        // Generic
        "NARR" | "RR01" | "RR02" | "RR03" | "RR04"
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_documented_statuses_map() {
        for code in ["ACSC", "ACTC", "ACSP", "PDNG", "RJCT"] {
            assert!(map_transaction_status(code).is_ok(), "{code}");
        }
    }

    #[test]
    fn acsc_maps_to_settled() {
        assert_eq!(map_transaction_status("ACSC").unwrap(), A2aStatus::Settled);
    }

    #[test]
    fn actc_maps_to_accepted() {
        assert_eq!(map_transaction_status("ACTC").unwrap(), A2aStatus::Accepted);
    }

    #[test]
    fn rjct_maps_to_rejected() {
        assert_eq!(map_transaction_status("RJCT").unwrap(), A2aStatus::Rejected);
    }

    #[test]
    fn pdng_maps_to_pending() {
        assert_eq!(map_transaction_status("PDNG").unwrap(), A2aStatus::Pending);
    }

    #[test]
    fn unknown_code_errors() {
        assert!(matches!(
            map_transaction_status("XXXX"),
            Err(Error::UnknownStatus(_))
        ));
        // Case-sensitive — lowercase is rejected.
        assert!(matches!(
            map_transaction_status("acsc"),
            Err(Error::UnknownStatus(_))
        ));
    }

    #[test]
    fn reason_code_recognition_sample() {
        // Account closed
        assert!(is_known_reason_code("AC04"));
        // Insufficient funds
        assert!(is_known_reason_code("AM04"));
        // Duplicate
        assert!(is_known_reason_code("DUPL"));
        // Beneficiary deceased (BE06)
        assert!(is_known_reason_code("BE06"));
        // Unknown
        assert!(!is_known_reason_code("XX99"));
        assert!(!is_known_reason_code(""));
    }
}
