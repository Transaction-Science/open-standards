//! PIX status / reason-code mapping.
//!
//! Bacen's SPI uses ISO 20022 pacs.002 codes (same as `FedNow` / SEPA),
//! plus a few PIX-specific extensions in the Bacen `tipo` field on the
//! operational side. We surface the ISO codes verbatim and pass through
//! the Bacen-specific codes unchanged in `reason_code`.

use crate::acquirer::A2aStatus;
use crate::error::{Error, Result};

/// Map an ISO 20022 transaction status (`TxSts`) to [`A2aStatus`].
///
/// # Errors
/// `Error::UnknownStatus` for codes outside the documented set.
pub fn map_transaction_status(code: &str) -> Result<A2aStatus> {
    Ok(match code {
        // PIX SPI uses ACCC for "completed credit"; both are settled.
        "ACSC" | "ACCC" => A2aStatus::Settled,
        "ACSP" => A2aStatus::InProgress,
        "ACTC" => A2aStatus::Accepted,
        "PDNG" => A2aStatus::Pending,
        "RJCT" => A2aStatus::Rejected,
        other => return Err(Error::UnknownStatus(other.to_owned())),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pix_acsc_settled() {
        assert_eq!(map_transaction_status("ACSC").unwrap(), A2aStatus::Settled);
    }
    #[test]
    fn pix_accc_settled() {
        assert_eq!(map_transaction_status("ACCC").unwrap(), A2aStatus::Settled);
    }
    #[test]
    fn pix_rjct_rejected() {
        assert_eq!(map_transaction_status("RJCT").unwrap(), A2aStatus::Rejected);
    }
    #[test]
    fn pix_unknown_errors() {
        assert!(matches!(
            map_transaction_status("ZZZZ"),
            Err(Error::UnknownStatus(_))
        ));
    }
}
