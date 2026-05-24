//! Variable Recurring Payments (VRP).
//!
//! VRP is the UK Open Banking standard for a single long-lived consent
//! that authorises *many* subsequent payments inside operator-supplied
//! limits. Two flavours:
//!
//! - **Sweeping VRP.** Movement between the PSU's own accounts.
//!   Free of charge, mandated under CMA9 retail-banking remedies.
//! - **Non-sweeping (commercial) VRP.** Operator-to-merchant
//!   recurring payments. Commercially negotiated between TPP and
//!   ASPSP; not regulated under CMA9.
//!
//! Reference: OBIE Domestic VRP Profile v3.1.10 (2025).

use op_core::Money;
use serde::{Deserialize, Serialize};
use time::Duration;

use crate::aisp::ConsentId;
use crate::error::Result;
use crate::fapi::OAuth2Token;
use crate::pisp::{PaymentInitiationStatus, PaymentRef};

/// VRP kind: sweeping (own-account, CMA9) or commercial (non-sweeping).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VrpKind {
    /// Movement between the PSU's own accounts. CMA9 mandates this
    /// for free and bounds ASPSP discretion.
    Sweeping,
    /// Operator-to-merchant. Commercially negotiated.
    NonSweeping,
}

/// Time window over which an aggregate cap applies. UK OBIE specifies
/// `Day`, `Week`, `Fortnight`, `Month`, `HalfYear`, `Year`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum VrpWindow {
    /// 24-hour rolling window from first payment in the window.
    Day,
    /// 7-day window.
    Week,
    /// 14-day window.
    Fortnight,
    /// Calendar-month window.
    Month,
    /// 6-month window.
    HalfYear,
    /// Calendar-year window.
    Year,
}

impl VrpWindow {
    /// Duration of the window. Calendar windows are approximated as
    /// fixed durations for cap arithmetic; ASPSPs handle calendar
    /// alignment server-side.
    #[must_use]
    pub const fn approx_duration(self) -> Duration {
        match self {
            Self::Day => Duration::days(1),
            Self::Week => Duration::days(7),
            Self::Fortnight => Duration::days(14),
            Self::Month => Duration::days(30),
            Self::HalfYear => Duration::days(182),
            Self::Year => Duration::days(365),
        }
    }
}

/// Cap parameters bound to a VRP consent. The ASPSP enforces these
/// at submission time and rejects payments that exceed any limit.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VrpControlParameters {
    /// Maximum per-payment amount.
    pub max_individual_amount: Money,
    /// Maximum aggregate amount over the [`Self::window`].
    pub max_period_amount: Money,
    /// Window over which [`Self::max_period_amount`] is computed.
    pub window: VrpWindow,
    /// Optional hard expiry of the consent. After this date the
    /// consent is no longer usable regardless of caps.
    pub valid_until: Option<time::OffsetDateTime>,
}

/// VRP consent — long-lived, repeatedly-debitable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VrpConsent {
    /// Consent identifier minted by the ASPSP.
    pub id: ConsentId,
    /// Sweeping vs commercial.
    pub kind: VrpKind,
    /// Debtor account identifier.
    pub debtor_account: String,
    /// Creditor account identifier.
    pub creditor_account: String,
    /// Cap parameters the ASPSP will enforce.
    pub controls: VrpControlParameters,
}

/// A single VRP payment instruction.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct VrpExecution {
    /// Amount to transfer on this payment.
    pub amount: Money,
    /// End-to-end ID for the payment.
    pub end_to_end_id: String,
    /// Operator-side remittance reference.
    pub remittance: Option<String>,
    /// RFC 3339 timestamp the operator records as "submission time"
    /// for window arithmetic. The ASPSP's clock is authoritative.
    pub submitted_at: time::OffsetDateTime,
}

/// Sweeping helper: pairs an execution with the window context for
/// local pre-checking before submission.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VrpSweep {
    /// The consent the sweep is debited against.
    pub consent: VrpConsent,
    /// The execution about to be submitted.
    pub execution: VrpExecution,
    /// Aggregate already-spent inside the current window.
    pub already_spent_in_window: Money,
}

impl VrpSweep {
    /// Pre-check the execution against the consent's caps.
    ///
    /// This is a client-side hint; the ASPSP remains authoritative.
    /// Catching cap violations locally avoids burning interaction
    /// credits at the ASPSP for predictably-rejected payments.
    pub fn check_caps(&self) -> Result<()> {
        if self.execution.amount.currency != self.consent.controls.max_individual_amount.currency {
            return Err(crate::Error::CurrencyMismatch(format!(
                "execution currency {} != consent currency {}",
                self.execution.amount.currency, self.consent.controls.max_individual_amount.currency,
            )));
        }
        if self.execution.amount.minor_units
            > self.consent.controls.max_individual_amount.minor_units
        {
            return Err(crate::Error::VrpLimitExceeded {
                reason: "per-payment cap exceeded".into(),
            });
        }
        let projected_total = self
            .already_spent_in_window
            .checked_add(self.execution.amount)
            .map_err(|_| crate::Error::Overflow)?;
        if projected_total.minor_units > self.consent.controls.max_period_amount.minor_units {
            return Err(crate::Error::VrpLimitExceeded {
                reason: "per-period cap exceeded".into(),
            });
        }
        if let Some(expiry) = self.consent.controls.valid_until {
            if self.execution.submitted_at > expiry {
                return Err(crate::Error::ConsentStateInvalid {
                    reason: "consent expired".into(),
                });
            }
        }
        Ok(())
    }
}

/// VRP service trait.
pub trait VrpService: Send + Sync {
    /// Create a VRP consent.
    fn create_consent(&self, token: &OAuth2Token, consent: &VrpConsent) -> Result<ConsentId>;

    /// Submit a single execution against an existing consent.
    fn execute(
        &self,
        consent: &ConsentId,
        token: &OAuth2Token,
        execution: &VrpExecution,
    ) -> Result<PaymentRef>;

    /// Poll the status of a VRP execution.
    fn status(
        &self,
        token: &OAuth2Token,
        payment_ref: &PaymentRef,
    ) -> Result<PaymentInitiationStatus>;

    /// Revoke a VRP consent. No further executions accepted after
    /// the ASPSP acknowledges revocation.
    fn revoke(&self, consent: &ConsentId, token: &OAuth2Token) -> Result<()>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::Currency;

    fn consent() -> VrpConsent {
        VrpConsent {
            id: ConsentId("vrp-1".into()),
            kind: VrpKind::Sweeping,
            debtor_account: "GB29NWBK60161331926819".into(),
            creditor_account: "GB33BUKB20201555555555".into(),
            controls: VrpControlParameters {
                max_individual_amount: Money::from_minor(100_00, Currency::GBP),
                max_period_amount: Money::from_minor(500_00, Currency::GBP),
                window: VrpWindow::Month,
                valid_until: None,
            },
        }
    }

    #[test]
    fn within_caps_passes() {
        let sweep = VrpSweep {
            consent: consent(),
            execution: VrpExecution {
                amount: Money::from_minor(50_00, Currency::GBP),
                end_to_end_id: "E1".into(),
                remittance: None,
                submitted_at: time::OffsetDateTime::UNIX_EPOCH,
            },
            already_spent_in_window: Money::from_minor(200_00, Currency::GBP),
        };
        sweep.check_caps().expect("ok");
    }

    #[test]
    fn per_payment_cap_fires() {
        let sweep = VrpSweep {
            consent: consent(),
            execution: VrpExecution {
                amount: Money::from_minor(200_00, Currency::GBP),
                end_to_end_id: "E1".into(),
                remittance: None,
                submitted_at: time::OffsetDateTime::UNIX_EPOCH,
            },
            already_spent_in_window: Money::zero(Currency::GBP),
        };
        assert!(matches!(
            sweep.check_caps().unwrap_err(),
            crate::Error::VrpLimitExceeded { .. }
        ));
    }

    #[test]
    fn period_cap_fires() {
        let sweep = VrpSweep {
            consent: consent(),
            execution: VrpExecution {
                amount: Money::from_minor(50_00, Currency::GBP),
                end_to_end_id: "E1".into(),
                remittance: None,
                submitted_at: time::OffsetDateTime::UNIX_EPOCH,
            },
            already_spent_in_window: Money::from_minor(490_00, Currency::GBP),
        };
        assert!(matches!(
            sweep.check_caps().unwrap_err(),
            crate::Error::VrpLimitExceeded { .. }
        ));
    }

    #[test]
    fn windows_have_sensible_durations() {
        assert_eq!(VrpWindow::Day.approx_duration(), Duration::days(1));
        assert_eq!(VrpWindow::Year.approx_duration(), Duration::days(365));
    }
}
