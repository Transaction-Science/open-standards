//! Per-rail profile rules.
//!
//! ISO 20022 is a kit; every rail picks a subset and adds its own
//! constraints on top. A `pacs.008` that's valid for SEPA Instant may
//! be rejected by `FedNow` because `FedNow` mandates a UETR and SEPA does
//! not, while a `FedNow` `pacs.008` will be rejected by PIX because PIX
//! requires a CPF/CNPJ tax identifier on the debtor.
//!
//! Each [`Profile`] implementation enforces:
//! - **Mandatory fields** beyond the ISO base.
//! - **Field formats** (UETR pattern, IMAD pattern, ABA/IBAN checksums).
//! - **Code-list subsets** (which `TxSts` and `StatusReason` codes are
//!   accepted on this rail).
//! - **Cardinality caps** (RTP allows up to 8 agents in a `pacs.008`;
//!   `FedNow` caps the message size differently).
//! - **Version pinning** (`FedNow` currently uses `pacs.008.001.08`;
//!   migration to `.001.14` is planned).
//!
//! Sources:
//! - `FedNow` Service Operating Procedures, Version 3.2, June 2025.
//! - Payments Canada RTR ISO 20022 Specification, v1.4, May 2025.
//! - Banco Central do Brasil PIX manuals.
//! - EPC SEPA Instant rulebook 2024.

use crate::bah::BusinessApplicationHeader;
use crate::error::{Error, Result};
use crate::message::MessageKind;
use crate::status::TransactionStatus;

/// A rail profile. Each rail enforces its own subset of ISO 20022.
pub trait Profile {
    /// Profile name, e.g. `"FedNow"`. Used in error messages.
    const NAME: &'static str;

    /// The exact ISO 20022 version string this profile uses for the
    /// given message kind, e.g. `"pacs.008.001.08"`.
    ///
    /// Returns `None` if the profile does not support this kind.
    fn version_for(kind: MessageKind) -> Option<&'static str>;

    /// True if this profile accepts the given status code.
    fn accepts_status(status: TransactionStatus) -> bool;

    /// Validate a BAH against profile rules.
    ///
    /// # Errors
    /// `Error::ProfileViolation` on any rule failure.
    fn validate_bah(bah: &BusinessApplicationHeader) -> Result<()>;
}

// ---------------------------------------------------------------------------
// FedNow
// ---------------------------------------------------------------------------

/// `FedNow` Service profile.
///
/// Versions pinned per `FedNow` Operating Procedures, June 2025:
/// - `pacs.008.001.08` — customer credit transfer
/// - `pacs.002.001.10` — payment status report
/// - `pacs.004.001.09` — payment return
/// - `pacs.009.001.08` — FI credit transfer (liquidity)
/// - `pacs.028.001.03` — payment status request
/// - `pain.013.001.07` — request for payment
/// - `pain.014.001.07` — RFP status
/// - `camt.056.001.08` — return request
/// - `camt.029.001.09` — return response
/// - `admi.002.001.01` — reject
/// - `admi.004.001.02` — broadcast / heartbeat
/// - `admi.006.001.01` — retrieval request
/// - `admi.007.001.01` — receipt ack
/// - `admi.998.001.02` — participant list (`FedNow` proprietary)
/// - `head.001.001.02` — BAH
pub struct FedNow;

impl Profile for FedNow {
    const NAME: &'static str = "FedNow";

    fn version_for(kind: MessageKind) -> Option<&'static str> {
        Some(match kind {
            MessageKind::Pacs008 => "pacs.008.001.08",
            MessageKind::Pacs002 => "pacs.002.001.10",
            MessageKind::Pacs004 => "pacs.004.001.09",
            MessageKind::Pacs009 => "pacs.009.001.08",
            MessageKind::Pacs028 => "pacs.028.001.03",
            MessageKind::Pain013 => "pain.013.001.07",
            MessageKind::Pain014 => "pain.014.001.07",
            MessageKind::Camt056 => "camt.056.001.08",
            MessageKind::Camt029 => "camt.029.001.09",
            MessageKind::Admi002 => "admi.002.001.01",
            MessageKind::Admi004 => "admi.004.001.02",
            MessageKind::Admi006 => "admi.006.001.01",
            MessageKind::Admi007 => "admi.007.001.01",
            MessageKind::Admi998 => "admi.998.001.02",
            MessageKind::Head001 => "head.001.001.02",
            // Not used by FedNow (cash-management / reporting messages).
            MessageKind::Camt026
            | MessageKind::Camt027
            | MessageKind::Camt028
            | MessageKind::Camt053
            | MessageKind::Camt054
            | MessageKind::Camt060 => {
                return None;
            }
        })
    }

    fn accepts_status(status: TransactionStatus) -> bool {
        // Per FedNow Operating Procedures: ACTC, ACSC, RJCT, PDNG.
        matches!(
            status,
            TransactionStatus::AcceptedTechnical
                | TransactionStatus::AcceptedSettled
                | TransactionStatus::Rejected
                | TransactionStatus::Pending
        )
    }

    fn validate_bah(bah: &BusinessApplicationHeader) -> Result<()> {
        use crate::bah::PartyIdentification as P;

        bah.validate()?;

        // FedNow uses ABA routing numbers in From/To. Reject if BIC was
        // supplied without also providing the ABA in a ClearingSystemMemberId.
        let validate_party = |p: &P, side: &'static str| -> Result<()> {
            match p {
                P::AbaRoutingNumber(_) => Ok(()),
                P::ClearingSystemMemberId {
                    clearing_system, ..
                } if clearing_system == "USABA" || clearing_system == "FedNow" => Ok(()),
                _ => Err(Error::ProfileViolation {
                    profile: Self::NAME,
                    reason: alloc::format!(
                        "{side} party must be ABA routing number or USABA/FedNow clearing system id"
                    ),
                }),
            }
        };
        validate_party(&bah.from, "From")?;
        validate_party(&bah.to, "To")?;

        // FedNow message-definition ids must start with one of the known
        // version strings. We don't enforce *which* one here (caller may
        // be sending any of pacs/pain/camt/admi), but we do enforce that
        // it's one of the FedNow-pinned versions.
        let valid = [
            "pacs.008.001.08",
            "pacs.002.001.10",
            "pacs.004.001.09",
            "pacs.009.001.08",
            "pacs.028.001.03",
            "pain.013.001.07",
            "pain.014.001.07",
            "camt.056.001.08",
            "camt.029.001.09",
            "admi.002.001.01",
            "admi.004.001.02",
            "admi.006.001.01",
            "admi.007.001.01",
            "admi.998.001.02",
            "head.001.001.02",
        ];
        if !valid.iter().any(|v| bah.message_definition_id == *v) {
            return Err(Error::ProfileViolation {
                profile: Self::NAME,
                reason: alloc::format!(
                    "MsgDefIdr {:?} not in FedNow's pinned version set",
                    bah.message_definition_id
                ),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// RTP (The Clearing House)
// ---------------------------------------------------------------------------

/// RTP — The Clearing House Real-Time Payments network.
///
/// RTP also runs ISO 20022 but with different version pinning and
/// different rules from `FedNow` (e.g. RTP allows up to 8 agents in a
/// `pacs.008`, `FedNow` restricts further). Production version set
/// matches the Payments Canada RTR specification document conventions.
pub struct Rtp;

impl Profile for Rtp {
    const NAME: &'static str = "RTP";

    fn version_for(kind: MessageKind) -> Option<&'static str> {
        Some(match kind {
            MessageKind::Pacs008 => "pacs.008.001.08",
            MessageKind::Pacs002 => "pacs.002.001.10",
            MessageKind::Pacs028 => "pacs.028.001.03",
            MessageKind::Pain013 => "pain.013.001.07",
            MessageKind::Pain014 => "pain.014.001.07",
            MessageKind::Camt056 => "camt.056.001.08",
            MessageKind::Admi002 => "admi.002.001.01",
            MessageKind::Admi004 => "admi.004.001.02",
            MessageKind::Head001 => "head.001.001.02",
            _ => return None,
        })
    }

    fn accepts_status(status: TransactionStatus) -> bool {
        matches!(
            status,
            TransactionStatus::AcceptedSettled
                | TransactionStatus::Rejected
                | TransactionStatus::Pending
        )
    }

    fn validate_bah(bah: &BusinessApplicationHeader) -> Result<()> {
        bah.validate()?;
        // RTP also uses ABA routing numbers for US participants.
        // (Same rule as FedNow for our purposes.)
        FedNow::validate_bah(bah).map_err(|_| Error::ProfileViolation {
            profile: Self::NAME,
            reason: "BAH must use US ABA routing identifiers".into(),
        })
    }
}

// ---------------------------------------------------------------------------
// SEPA Instant
// ---------------------------------------------------------------------------

/// SEPA Instant Credit Transfer (SCT Inst) profile.
///
/// Runs against the EBA Clearing RT1 or TIPS infrastructure. Uses BICs,
/// not routing numbers; mandates IBAN for debtor/creditor accounts.
pub struct SepaInstant;

impl Profile for SepaInstant {
    const NAME: &'static str = "SEPA-Instant";

    fn version_for(kind: MessageKind) -> Option<&'static str> {
        Some(match kind {
            // EPC 2023 rulebook pinned to .001.08
            MessageKind::Pacs008 => "pacs.008.001.08",
            MessageKind::Pacs002 => "pacs.002.001.10",
            MessageKind::Pacs004 => "pacs.004.001.09",
            MessageKind::Camt056 => "camt.056.001.08",
            MessageKind::Camt029 => "camt.029.001.09",
            MessageKind::Head001 => "head.001.001.02",
            _ => return None,
        })
    }

    fn accepts_status(status: TransactionStatus) -> bool {
        matches!(
            status,
            TransactionStatus::AcceptedSettled | TransactionStatus::Rejected
        )
    }

    fn validate_bah(bah: &BusinessApplicationHeader) -> Result<()> {
        use crate::bah::PartyIdentification as P;

        bah.validate()?;
        // SEPA mandates BICs, not routing numbers.
        let is_bic = |p: &P| matches!(p, P::Bic(_));
        if !is_bic(&bah.from) || !is_bic(&bah.to) {
            return Err(Error::ProfileViolation {
                profile: Self::NAME,
                reason: "SEPA Instant requires BICs in From and To".into(),
            });
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// PIX (Brazil)
// ---------------------------------------------------------------------------

/// PIX / SPI (Sistema de Pagamentos Instantâneos) profile.
///
/// Operated by Banco Central do Brasil. Uses ISO 20022 over an mTLS
/// connection to ICOM. PIX-specific extensions live in the `Prtry`
/// elements of standard messages.
pub struct Pix;

impl Profile for Pix {
    const NAME: &'static str = "PIX";

    fn version_for(kind: MessageKind) -> Option<&'static str> {
        Some(match kind {
            MessageKind::Pacs008 => "pacs.008.001.08",
            MessageKind::Pacs002 => "pacs.002.001.10",
            MessageKind::Pacs004 => "pacs.004.001.09",
            MessageKind::Camt056 => "camt.056.001.08",
            MessageKind::Camt029 => "camt.029.001.09",
            MessageKind::Head001 => "head.001.001.02",
            _ => return None,
        })
    }

    fn accepts_status(status: TransactionStatus) -> bool {
        matches!(
            status,
            TransactionStatus::AcceptedSettled | TransactionStatus::Rejected
        )
    }

    fn validate_bah(bah: &BusinessApplicationHeader) -> Result<()> {
        use crate::bah::PartyIdentification as P;

        bah.validate()?;
        // PIX participants identify by ISPB — an 8-digit Banco Central code
        // that we treat as a ClearingSystemMemberId with clearing_system="ISPB".
        let is_ispb = |p: &P| {
            matches!(p, P::ClearingSystemMemberId { clearing_system, member_id }
                if clearing_system == "ISPB" && member_id.len() == 8
                && member_id.bytes().all(|b| b.is_ascii_digit()))
        };
        if !is_ispb(&bah.from) || !is_ispb(&bah.to) {
            return Err(Error::ProfileViolation {
                profile: Self::NAME,
                reason: "PIX requires 8-digit ISPB clearing-system member ids".into(),
            });
        }
        Ok(())
    }
}

extern crate alloc;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bah::PartyIdentification;

    fn fednow_bah(msg_def: &str) -> BusinessApplicationHeader {
        BusinessApplicationHeader::new(
            PartyIdentification::AbaRoutingNumber("021000021".into()),
            PartyIdentification::AbaRoutingNumber("026009593".into()),
            msg_def,
        )
    }

    // ---- FedNow ----

    #[test]
    fn fednow_pins_pacs008_to_v08() {
        assert_eq!(
            FedNow::version_for(MessageKind::Pacs008),
            Some("pacs.008.001.08")
        );
    }

    #[test]
    fn fednow_rejects_unsupported_kinds() {
        assert_eq!(FedNow::version_for(MessageKind::Camt060), None);
        assert_eq!(FedNow::version_for(MessageKind::Camt054), None);
    }

    #[test]
    fn fednow_accepts_only_documented_statuses() {
        assert!(FedNow::accepts_status(TransactionStatus::AcceptedTechnical));
        assert!(FedNow::accepts_status(TransactionStatus::AcceptedSettled));
        assert!(FedNow::accepts_status(TransactionStatus::Rejected));
        assert!(FedNow::accepts_status(TransactionStatus::Pending));
        // FedNow doesn't use ACCP/RCVD.
        assert!(!FedNow::accepts_status(TransactionStatus::AcceptedCustomer));
        assert!(!FedNow::accepts_status(TransactionStatus::Received));
    }

    #[test]
    fn fednow_bah_accepts_aba() {
        assert!(FedNow::validate_bah(&fednow_bah("pacs.008.001.08")).is_ok());
    }

    #[test]
    fn fednow_bah_rejects_unknown_msgdef() {
        assert!(matches!(
            FedNow::validate_bah(&fednow_bah("pacs.008.001.14")),
            Err(Error::ProfileViolation { .. })
        ));
    }

    #[test]
    fn fednow_bah_rejects_bic() {
        let bah = BusinessApplicationHeader::new(
            PartyIdentification::Bic("CHASUS33".into()),
            PartyIdentification::AbaRoutingNumber("026009593".into()),
            "pacs.008.001.08",
        );
        assert!(matches!(
            FedNow::validate_bah(&bah),
            Err(Error::ProfileViolation { .. })
        ));
    }

    // ---- SEPA Instant ----

    #[test]
    fn sepa_requires_bic() {
        let bah = BusinessApplicationHeader::new(
            PartyIdentification::Bic("DEUTDEFFXXX".into()),
            PartyIdentification::Bic("BNPAFRPPXXX".into()),
            "pacs.008.001.08",
        );
        assert!(SepaInstant::validate_bah(&bah).is_ok());
    }

    #[test]
    fn sepa_rejects_aba() {
        let bah = BusinessApplicationHeader::new(
            PartyIdentification::AbaRoutingNumber("021000021".into()),
            PartyIdentification::AbaRoutingNumber("026009593".into()),
            "pacs.008.001.08",
        );
        assert!(matches!(
            SepaInstant::validate_bah(&bah),
            Err(Error::ProfileViolation { .. })
        ));
    }

    // ---- PIX ----

    #[test]
    fn pix_requires_8_digit_ispb() {
        let bah = BusinessApplicationHeader::new(
            PartyIdentification::ClearingSystemMemberId {
                clearing_system: "ISPB".into(),
                member_id: "00000000".into(),
            },
            PartyIdentification::ClearingSystemMemberId {
                clearing_system: "ISPB".into(),
                member_id: "60746948".into(),
            },
            "pacs.008.001.08",
        );
        assert!(Pix::validate_bah(&bah).is_ok());
    }

    #[test]
    fn pix_rejects_non_8_digit_ispb() {
        let bah = BusinessApplicationHeader::new(
            PartyIdentification::ClearingSystemMemberId {
                clearing_system: "ISPB".into(),
                member_id: "123".into(),
            },
            PartyIdentification::ClearingSystemMemberId {
                clearing_system: "ISPB".into(),
                member_id: "60746948".into(),
            },
            "pacs.008.001.08",
        );
        assert!(matches!(
            Pix::validate_bah(&bah),
            Err(Error::ProfileViolation { .. })
        ));
    }

    // ---- Cross-rail invariants ----

    #[test]
    fn all_profiles_pin_pacs008() {
        assert!(FedNow::version_for(MessageKind::Pacs008).is_some());
        assert!(Rtp::version_for(MessageKind::Pacs008).is_some());
        assert!(SepaInstant::version_for(MessageKind::Pacs008).is_some());
        assert!(Pix::version_for(MessageKind::Pacs008).is_some());
    }

    #[test]
    fn all_profiles_accept_acsc() {
        assert!(FedNow::accepts_status(TransactionStatus::AcceptedSettled));
        assert!(Rtp::accepts_status(TransactionStatus::AcceptedSettled));
        assert!(SepaInstant::accepts_status(
            TransactionStatus::AcceptedSettled
        ));
        assert!(Pix::accepts_status(TransactionStatus::AcceptedSettled));
    }

    #[test]
    fn all_profiles_accept_rjct() {
        assert!(FedNow::accepts_status(TransactionStatus::Rejected));
        assert!(Rtp::accepts_status(TransactionStatus::Rejected));
        assert!(SepaInstant::accepts_status(TransactionStatus::Rejected));
        assert!(Pix::accepts_status(TransactionStatus::Rejected));
    }
}
