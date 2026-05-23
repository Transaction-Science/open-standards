//! SEPA Instant status / reason-code mapping.
//!
//! Per EPC SCT Inst IG 2019 v1.0 and TIPS UDFS, the pacs.002 status
//! codes used by RT1 / TIPS are: `ACCP` (positive confirmation, optional
//! subscription on RT1), `ACSC` (settled, used by TIPS), `ACSP` (in
//! progress), `RJCT` (rejected — mandatory response on failure), `PDNG`.

use crate::acquirer::A2aStatus;
use crate::error::{Error, Result};

/// Map an SCT Inst transaction status to [`A2aStatus`].
///
/// # Errors
/// `Error::UnknownStatus` for unknown codes.
pub fn map_transaction_status(code: &str) -> Result<A2aStatus> {
    Ok(match code {
        "ACSC" => A2aStatus::Settled,
        "ACCP" => A2aStatus::Accepted, // confirmed; RT1 may still finalize
        "ACSP" => A2aStatus::InProgress,
        "PDNG" => A2aStatus::Pending,
        "RJCT" => A2aStatus::Rejected,
        other => return Err(Error::UnknownStatus(other.to_owned())),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accp_is_accepted() {
        assert_eq!(map_transaction_status("ACCP").unwrap(), A2aStatus::Accepted);
    }
    #[test]
    fn acsc_is_settled() {
        assert_eq!(map_transaction_status("ACSC").unwrap(), A2aStatus::Settled);
    }
    #[test]
    fn rjct_is_rejected() {
        assert_eq!(map_transaction_status("RJCT").unwrap(), A2aStatus::Rejected);
    }
    #[test]
    fn unknown_errors() {
        assert!(matches!(
            map_transaction_status("ZZZZ"),
            Err(Error::UnknownStatus(_))
        ));
    }
}
