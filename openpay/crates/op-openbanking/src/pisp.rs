//! Payment Initiation Service Provider (PISP) surface.
//!
//! Implements the vendor-neutral version of:
//!
//! - UK Open Banking R/W v3.1 § Domestic / International / Standing-Order
//!   Payment Consents and Payments
//! - Berlin Group NextGenPSD2 § 5.1 (payment-initiation service)
//! - STET PSD2 § Payment Initiation
//! - SGFinDex equivalents are out of scope for v1 (data-only).
//!
//! ## Payment kinds
//!
//! The standards converge on five concrete kinds: immediate single,
//! future-dated single, recurring (standing order), bulk, and
//! international. [`PaymentKind`] is that taxonomy.

use op_core::Money;
use serde::{Deserialize, Serialize};

use crate::aisp::ConsentId;
use crate::error::Result;
use crate::fapi::OAuth2Token;

/// The five PISP payment kinds that survive across UK / Berlin / STET.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaymentKind {
    /// Immediate single payment (UK `DomesticPayment`, Berlin
    /// `payments/sepa-credit-transfers`, STET `Payment Request`).
    ImmediateSingle,
    /// Future-dated single payment, value-date in the future.
    FutureDated,
    /// Standing order — recurring payment with a schedule. The
    /// schedule is bound to the consent, not the payment.
    StandingOrder,
    /// Bulk file of payments (UK `FilePayment`, Berlin
    /// `bulk-payments`). Many ASPSPs gate this behind extra
    /// commercial agreements.
    Bulk,
    /// International payment (UK `InternationalPayment`, Berlin
    /// `cross-border-credit-transfers`). Adds correspondent-bank
    /// and FX-quote fields.
    International,
}

/// Reference returned by the ASPSP after a successful payment-initiation
/// post. UK OBIE: `Data.DomesticPaymentId`. Berlin Group:
/// `paymentId`. STET: `paymentRequestResourceId`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct PaymentRef(pub String);

/// Lifecycle status of a payment initiation.
///
/// Aligned to ISO 20022 `ExternalPaymentTransactionStatus1Code` plus
/// UK OBIE's `AcceptedSettlementInProcess` distinction.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum PaymentInitiationStatus {
    /// Consent created; PSU not yet authenticated.
    AwaitingAuthorisation,
    /// PSU authenticated; awaiting submission.
    Authorised,
    /// Submitted to the rail; awaiting clearing.
    AcceptedSettlementInProcess,
    /// Settled into the beneficiary's account.
    Settled,
    /// Consent rejected by PSU or ASPSP.
    Rejected,
    /// Consent revoked or expired before submission.
    Cancelled,
}

/// A vendor-neutral payment-initiation request.
///
/// Bindings translate this into:
/// - UK: `OBWriteDomesticConsent4` + `OBWriteDomestic2`.
/// - Berlin Group: a `PaymentInitiationXMLPart` (pain.001) for the
///   chosen `payment-product` (`sepa-credit-transfers`,
///   `instant-sepa-credit-transfers`, ...).
/// - STET: a `Payment Request Resource`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PaymentInitiation {
    /// Payment kind.
    pub kind: PaymentKind,
    /// Debtor account identifier (binding-specific format).
    pub debtor_account: String,
    /// Creditor account identifier (binding-specific format).
    pub creditor_account: String,
    /// Creditor display name (statement narrative).
    pub creditor_name: String,
    /// Amount to transfer.
    pub amount: Money,
    /// End-to-end reference (UK `EndToEndIdentification`, Berlin
    /// `endToEndIdentification`, ISO `EndToEndId`).
    pub end_to_end_id: String,
    /// Free-text remittance information (max length is
    /// binding-specific; UK = 140, SEPA = 140, FedNow = 140).
    pub remittance: Option<String>,
    /// Requested execution date for FutureDated / StandingOrder.
    /// Ignored for ImmediateSingle.
    pub requested_execution_date: Option<time::Date>,
}

impl PaymentInitiation {
    /// Basic shape validation. Each binding adds its own constraints
    /// on top (IBAN check digits, sort-code format, allowed
    /// remittance length).
    pub fn validate(&self) -> Result<()> {
        if self.amount.minor_units <= 0 {
            return Err(crate::Error::PaymentInitiationInvalid {
                reason: "amount must be strictly positive".into(),
            });
        }
        if self.creditor_account.trim().is_empty() {
            return Err(crate::Error::PaymentInitiationInvalid {
                reason: "creditor account is empty".into(),
            });
        }
        if self.end_to_end_id.is_empty() {
            return Err(crate::Error::PaymentInitiationInvalid {
                reason: "end-to-end id required (ISO 20022 EndToEndId)".into(),
            });
        }
        if matches!(
            self.kind,
            PaymentKind::FutureDated | PaymentKind::StandingOrder
        ) && self.requested_execution_date.is_none()
        {
            return Err(crate::Error::PaymentInitiationInvalid {
                reason: "future-dated / standing-order requires execution date".into(),
            });
        }
        Ok(())
    }
}

/// PISP service trait — vendor-neutral payment-initiation surface.
///
/// Bindings implement this in terms of their standard's wire format.
/// Like [`crate::AccountInfoService`], the trait is intentionally
/// synchronous; operators wrap their driver in an `async` wrapper.
pub trait PaymentInitiationService: Send + Sync {
    /// Create a payment-initiation consent. Returns the consent id
    /// the operator passes back through the PSU authorisation flow.
    fn create_consent(
        &self,
        token: &OAuth2Token,
        payment: &PaymentInitiation,
    ) -> Result<ConsentId>;

    /// Submit a previously-authorised payment for execution.
    fn submit(
        &self,
        consent: &ConsentId,
        token: &OAuth2Token,
        payment: &PaymentInitiation,
    ) -> Result<PaymentRef>;

    /// Poll the status of a submitted payment.
    fn status(
        &self,
        token: &OAuth2Token,
        payment_ref: &PaymentRef,
    ) -> Result<PaymentInitiationStatus>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::Currency;

    fn happy() -> PaymentInitiation {
        PaymentInitiation {
            kind: PaymentKind::ImmediateSingle,
            debtor_account: "GB29NWBK60161331926819".into(),
            creditor_account: "GB33BUKB20201555555555".into(),
            creditor_name: "Acme Widgets".into(),
            amount: Money::from_minor(10_000, Currency::GBP),
            end_to_end_id: "INV-2026-001".into(),
            remittance: Some("Invoice 001".into()),
            requested_execution_date: None,
        }
    }

    #[test]
    fn happy_path_validates() {
        happy().validate().expect("ok");
    }

    #[test]
    fn zero_amount_rejected() {
        let mut p = happy();
        p.amount = Money::from_minor(0, Currency::GBP);
        assert!(p.validate().is_err());
    }

    #[test]
    fn future_dated_needs_date() {
        let mut p = happy();
        p.kind = PaymentKind::FutureDated;
        assert!(p.validate().is_err());
        p.requested_execution_date =
            Some(time::Date::from_ordinal_date(2026, 1).expect("date"));
        p.validate().expect("ok with date");
    }
}
