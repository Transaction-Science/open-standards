//! Message kinds and the version-agnostic `Message` wrapper.
//!
//! ISO 20022 messages carry both a *kind* (`pacs.008` — customer credit
//! transfer) and a *version* (`.001.08`, `.001.14`, ...). Different rails
//! mandate different versions: `FedNow` runs `pacs.008.001.08` (per the
//! readiness guide), while RTGS systems are migrating to `.001.14`. We
//! abstract over this with [`MessageKind`] and let the rail profile pick
//! the version.

use serde::{Deserialize, Serialize};

/// Top-level ISO 20022 message family code (the part before the version).
///
/// Each variant maps to one or more upstream `Document::*` variants; the
/// active rail profile decides which version is in use.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MessageKind {
    /// `pacs.008` — FI-to-FI customer credit transfer. The workhorse:
    /// `FedNow`, RTP, SEPA Instant, PIX all use it for the actual payment.
    Pacs008,
    /// `pacs.002` — payment status report (ACTC, ACSC, RJCT, PDNG).
    Pacs002,
    /// `pacs.004` — payment return.
    Pacs004,
    /// `pacs.009` — FI credit transfer (liquidity / interbank). `FedNow`
    /// uses this for liquidity management transfers.
    Pacs009,
    /// `pacs.028` — payment status request.
    Pacs028,
    /// `pain.013` — creditor payment activation request (Request for Payment).
    Pain013,
    /// `pain.014` — creditor payment activation request status report.
    Pain014,
    /// `camt.056` — FI-to-FI payment cancellation request (return request).
    Camt056,
    /// `camt.029` — resolution of investigation (return response).
    Camt029,
    /// `camt.026` — unable-to-apply / additional info request.
    Camt026,
    /// `camt.027` — claim non-receipt.
    Camt027,
    /// `camt.028` — additional payment information.
    Camt028,
    /// `camt.053` — bank-to-customer statement (end-of-day). The
    /// authoritative artifact for ledger reconciliation.
    Camt053,
    /// `camt.054` — bank-to-customer debit/credit notification.
    Camt054,
    /// `camt.060` — account reporting request.
    Camt060,
    /// `admi.002` — message reject (NAK).
    Admi002,
    /// `admi.004` — system event notification (heartbeat / broadcast).
    Admi004,
    /// `admi.006` — retrieval request.
    Admi006,
    /// `admi.007` — receipt acknowledgement.
    Admi007,
    /// `admi.998` — proprietary participant-list message (`FedNow` extension).
    Admi998,
    /// `head.001` — Business Application Header. Required on every value
    /// message across `FedNow`.
    Head001,
}

impl MessageKind {
    /// Canonical short code (e.g. `"pacs.008"`).
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Pacs008 => "pacs.008",
            Self::Pacs002 => "pacs.002",
            Self::Pacs004 => "pacs.004",
            Self::Pacs009 => "pacs.009",
            Self::Pacs028 => "pacs.028",
            Self::Pain013 => "pain.013",
            Self::Pain014 => "pain.014",
            Self::Camt056 => "camt.056",
            Self::Camt029 => "camt.029",
            Self::Camt026 => "camt.026",
            Self::Camt027 => "camt.027",
            Self::Camt028 => "camt.028",
            Self::Camt053 => "camt.053",
            Self::Camt054 => "camt.054",
            Self::Camt060 => "camt.060",
            Self::Admi002 => "admi.002",
            Self::Admi004 => "admi.004",
            Self::Admi006 => "admi.006",
            Self::Admi007 => "admi.007",
            Self::Admi998 => "admi.998",
            Self::Head001 => "head.001",
        }
    }

    /// True if this is a *value* message (moves funds). Per `FedNow`
    /// operating procedures: pacs.008, pacs.004, pacs.009.
    #[must_use]
    pub const fn is_value(self) -> bool {
        matches!(self, Self::Pacs008 | Self::Pacs004 | Self::Pacs009)
    }

    /// True if this is a *status* message (no funds movement).
    #[must_use]
    pub const fn is_status(self) -> bool {
        matches!(self, Self::Pacs002 | Self::Pacs028 | Self::Pain014)
    }

    /// True if this is a *system* / administrative message.
    #[must_use]
    pub const fn is_system(self) -> bool {
        matches!(
            self,
            Self::Admi002 | Self::Admi004 | Self::Admi006 | Self::Admi007 | Self::Admi998
        )
    }
}

/// A version-agnostic wrapper around a parsed ISO 20022 document.
///
/// We don't expose the upstream `Document` enum directly because its
/// variants are versioned and the variant set varies by feature flag.
/// Callers operate on `Message` and use kind-specific accessors.
#[derive(Debug)]
pub enum Message {
    /// `pacs.008.001.12` — `FedNow` / RTP / SEPA Instant customer credit transfer.
    /// `FedNow` originally shipped on `.001.08`; the registry moved to
    /// `.001.12` and that's what `open-payments-iso20022-pacs = 1.0.10`
    /// exposes. Profile validators are still version-aware so older
    /// rails can still be served by re-mapping at the profile layer.
    Pacs008(Box<open_payments_iso20022_pacs::pacs_008_001_12::FIToFICustomerCreditTransferV12>),
    /// `pacs.002.001.12` — payment status report.
    Pacs002(Box<open_payments_iso20022_pacs::pacs_002_001_12::FIToFIPaymentStatusReportV12>),
    /// `pacs.004.001.13` — payment return.
    Pacs004(Box<open_payments_iso20022_pacs::pacs_004_001_13::PaymentReturnV13>),
    /// `pain.013.001.11` — request for payment.
    Pain013(Box<open_payments_iso20022_pain::pain_013_001_11::CreditorPaymentActivationRequestV11>),
    /// `camt.053.001.12` — bank-to-customer statement (end-of-day).
    /// The reconciliation artifact: each `Ntry` is a settled bank
    /// line that `op-reconciliation` matches against ledger txs.
    Camt053(Box<open_payments_iso20022_camt::camt_053_001_12::BankToCustomerStatementV12>),
    /// `camt.054.001.12` — bank-to-customer debit/credit notification.
    /// Same `Ntry` shape as `camt.053`; reconciliation flattens both
    /// through the same path.
    Camt054(
        Box<open_payments_iso20022_camt::camt_054_001_12::BankToCustomerDebitCreditNotificationV12>,
    ),
    /// `camt.056.001.11` — payment cancellation / return request.
    Camt056(Box<open_payments_iso20022_camt::camt_056_001_11::FIToFIPaymentCancellationRequestV11>),
    /// `admi.002.001.01` — message reject.
    Admi002(Box<open_payments_iso20022_admi::admi_002_001_01::Admi00200101>),
    /// `admi.004.001.02` — system event notification (heartbeat / broadcast).
    Admi004(Box<open_payments_iso20022_admi::admi_004_001_02::SystemEventNotificationV02>),
}

impl Message {
    /// Which message family this is.
    #[must_use]
    pub const fn kind(&self) -> MessageKind {
        match self {
            Self::Pacs008(_) => MessageKind::Pacs008,
            Self::Pacs002(_) => MessageKind::Pacs002,
            Self::Pacs004(_) => MessageKind::Pacs004,
            Self::Pain013(_) => MessageKind::Pain013,
            Self::Camt053(_) => MessageKind::Camt053,
            Self::Camt054(_) => MessageKind::Camt054,
            Self::Camt056(_) => MessageKind::Camt056,
            Self::Admi002(_) => MessageKind::Admi002,
            Self::Admi004(_) => MessageKind::Admi004,
        }
    }

    /// Parse a `camt.053.001.12` bank-to-customer statement from XML
    /// into [`Message::Camt053`].
    ///
    /// Real-world camt.053 messages wrap the statement body in
    /// `<Document><BkToCstmrStmt>...</BkToCstmrStmt></Document>` (the
    /// ISO 20022 outer-document envelope). The upstream type is the
    /// inner body, so we deserialize through a tiny wrapper that
    /// skips past `BkToCstmrStmt`. This keeps the upstream
    /// `open-payments-iso20022-camt` dependency behind the
    /// `op-iso20022` facade — downstream crates parse statements
    /// without taking a direct dependency on the per-family ISO
    /// 20022 crates.
    ///
    /// # Errors
    /// `Error::XmlDecode` if the XML isn't a well-formed
    /// `BankToCustomerStatementV12`.
    pub fn parse_camt053(xml: &str) -> crate::error::Result<Self> {
        #[derive(serde::Deserialize)]
        struct Wrapper {
            #[serde(rename = "BkToCstmrStmt")]
            bk_to_cstmr_stmt:
                open_payments_iso20022_camt::camt_053_001_12::BankToCustomerStatementV12,
        }
        let wrapped: Wrapper = crate::codec::from_xml(xml)?;
        Ok(Self::Camt053(Box::new(wrapped.bk_to_cstmr_stmt)))
    }

    /// Borrow the inner `camt.053` statement, or `None` if this isn't
    /// a [`Message::Camt053`].
    #[must_use]
    pub fn as_camt053(
        &self,
    ) -> Option<&open_payments_iso20022_camt::camt_053_001_12::BankToCustomerStatementV12> {
        match self {
            Self::Camt053(b) => Some(b.as_ref()),
            _ => None,
        }
    }

    /// Parse a `camt.054.001.12` debit/credit notification from XML
    /// into [`Message::Camt054`]. Skips past the `<Document>
    /// <BkToCstmrDbtCdtNtfctn>` outer wrapping, same shape as
    /// [`Self::parse_camt053`].
    ///
    /// # Errors
    /// `Error::XmlDecode` if the XML isn't a well-formed
    /// `BankToCustomerDebitCreditNotificationV12`.
    pub fn parse_camt054(xml: &str) -> crate::error::Result<Self> {
        #[derive(serde::Deserialize)]
        struct Wrapper {
            #[serde(rename = "BkToCstmrDbtCdtNtfctn")]
            bk_to_cstmr_dbt_cdt_ntfctn:
                open_payments_iso20022_camt::camt_054_001_12::BankToCustomerDebitCreditNotificationV12,
        }
        let wrapped: Wrapper = crate::codec::from_xml(xml)?;
        Ok(Self::Camt054(Box::new(wrapped.bk_to_cstmr_dbt_cdt_ntfctn)))
    }

    /// Borrow the inner `camt.054` notification, or `None` if this
    /// isn't a [`Message::Camt054`].
    #[must_use]
    pub fn as_camt054(
        &self,
    ) -> Option<
        &open_payments_iso20022_camt::camt_054_001_12::BankToCustomerDebitCreditNotificationV12,
    > {
        match self {
            Self::Camt054(b) => Some(b.as_ref()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn value_classification() {
        assert!(MessageKind::Pacs008.is_value());
        assert!(MessageKind::Pacs004.is_value());
        assert!(MessageKind::Pacs009.is_value());
        assert!(!MessageKind::Pacs002.is_value());
        assert!(!MessageKind::Admi004.is_value());
    }

    #[test]
    fn status_classification() {
        assert!(MessageKind::Pacs002.is_status());
        assert!(MessageKind::Pacs028.is_status());
        assert!(!MessageKind::Pacs008.is_status());
    }

    #[test]
    fn system_classification() {
        assert!(MessageKind::Admi002.is_system());
        assert!(MessageKind::Admi004.is_system());
        assert!(!MessageKind::Pacs008.is_system());
    }

    #[test]
    fn codes_match_iso20022_format() {
        // Spot-check the four most common ones; format is `xxxx.NNN`.
        assert_eq!(MessageKind::Pacs008.code(), "pacs.008");
        assert_eq!(MessageKind::Pain013.code(), "pain.013");
        assert_eq!(MessageKind::Camt056.code(), "camt.056");
        assert_eq!(MessageKind::Admi004.code(), "admi.004");
    }

    #[test]
    fn camt053_is_a_known_kind() {
        assert_eq!(MessageKind::Camt053.code(), "camt.053");
        // Statement is neither a value, status, nor system message.
        assert!(!MessageKind::Camt053.is_value());
        assert!(!MessageKind::Camt053.is_status());
        assert!(!MessageKind::Camt053.is_system());
    }
}
