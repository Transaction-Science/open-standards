//! SCA-exemption evaluator.
//!
//! PSD2 RTS Article 16-20 + the PSD3 / PSR (2026 baseline) update
//! define eight exemptions to Strong Customer Authentication. A
//! transaction that qualifies for an exemption may be authorized
//! *without* an SCA challenge, dramatically improving conversion.
//!
//! ## Exemptions implemented
//!
//! - [`EligibleExemption::LowValueTransaction`] — RTS Article 16:
//!   single payment under €30 AND cumulative under €100 over the
//!   trailing 24 hours AND at most 5 consecutive low-value
//!   transactions without SCA.
//! - [`EligibleExemption::MerchantInitiated`] — RTS Article 13:
//!   MIT outside the scope of SCA; mandate reference required.
//! - [`EligibleExemption::Recurring`] — RTS Article 14: subsequent
//!   recurring payments at the same merchant, same amount, same
//!   mandate.
//! - [`EligibleExemption::TransactionRiskAnalysis`] — RTS Article 18:
//!   TRA exemption thresholds keyed off the requestor's certified
//!   fraud-rate bracket (€100 / €250 / €500 amount caps).
//! - [`EligibleExemption::SecureCorporatePayment`] — RTS Article 17:
//!   corporate "lodge" cards on a closed-loop B2B network.
//! - [`EligibleExemption::TrustedBeneficiary`] — RTS Article 13(2):
//!   cardholder-enrolled trusted-beneficiary list maintained by the
//!   issuer.
//! - [`EligibleExemption::DelegatedAuthentication`] — PSD3 / RTS 2024
//!   extension: SCA performed by a third party (FIDO, wallet) and
//!   evidenced inside the AReq.
//! - [`EligibleExemption::OneLeg`] — RTS Recital 95: "one-leg-out"
//!   transactions where either the issuer or the acquirer is outside
//!   the EEA fall outside the SCA scope.
//!
//! ## Returns
//!
//! [`evaluate`] returns *all* eligible exemptions in priority order so
//! the caller (typically the orchestrator) can pick one per scheme
//! preference. Choosing TRA when an exemption with no fraud-rate
//! constraint is available is wasted risk budget; the priority order
//! reflects this.

use chrono::{DateTime, Utc};
use op_core::Money;
use serde::{Deserialize, Serialize};

use crate::auth_response::TransactionStatus;
use crate::error::{Error, Result};

/// EBA fraud-rate brackets that gate the TRA-exemption amount caps.
///
/// Source: PSD2 RTS Article 19. PSPs measure their fraud rate over a
/// rolling 90-day window per scheme and re-certify quarterly.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub enum FraudRateBracket {
    /// ≤ 1 bp (0.01 %) fraud rate → TRA allowed up to €500.
    UpTo1Bp,
    /// ≤ 6 bp (0.06 %) → TRA allowed up to €250.
    UpTo6Bp,
    /// ≤ 13 bp (0.13 %) → TRA allowed up to €100.
    UpTo13Bp,
    /// Above 13 bp → TRA not allowed. This is the conservative
    /// default for newly-created [`ExemptionContext`] values: it
    /// disables TRA until the PSP explicitly attests its current
    /// fraud-rate bracket.
    #[default]
    AboveTraThreshold,
}

impl FraudRateBracket {
    /// Returns the TRA cap (EUR minor units) the bracket permits.
    /// `None` means the bracket is over the regulatory threshold and
    /// TRA is unavailable.
    #[must_use]
    pub const fn tra_cap_minor_units(self) -> Option<i64> {
        match self {
            Self::UpTo1Bp => Some(500_00),
            Self::UpTo6Bp => Some(250_00),
            Self::UpTo13Bp => Some(100_00),
            Self::AboveTraThreshold => None,
        }
    }
}

/// Classification of the SCA second factor when delegated.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SecondFactorClass {
    /// Possession + inherence (FIDO authenticator).
    Fido,
    /// Possession + knowledge (wallet PIN + device).
    WalletPin,
    /// Inherence + knowledge (biometric + password).
    BiometricKnowledge,
}

/// Concrete exemption the evaluator returns as eligible.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum EligibleExemption {
    /// Low-value (RTS Article 16).
    LowValueTransaction {
        /// The amount this exemption was evaluated against.
        amount: Money,
    },
    /// Merchant-initiated transaction outside the SCA scope.
    MerchantInitiated {
        /// Mandate reference under which the cardholder pre-authorized
        /// the future merchant-initiated charge.
        mandate_ref: String,
    },
    /// Recurring (subsequent recurring payment under an existing
    /// mandate).
    Recurring {
        /// Mandate reference.
        mandate_ref: String,
        /// Subscription / plan id at the merchant.
        subscription_id: String,
    },
    /// Transaction Risk Analysis (RTS Article 18).
    TransactionRiskAnalysis {
        /// Risk score in [0.0, 1.0]; lower is better.
        score: f32,
        /// Requestor's currently-certified fraud-rate bracket.
        fraud_rate_bracket: FraudRateBracket,
    },
    /// Secure Corporate Payment (RTS Article 17).
    SecureCorporatePayment {
        /// True if the card BIN range is a known corporate/lodge BIN.
        corporate_card_indicator: bool,
    },
    /// Trusted Beneficiary, cardholder-enrolled at the issuer.
    TrustedBeneficiary {
        /// Issuer-side beneficiary id.
        beneficiary_id: String,
        /// When the cardholder enrolled the merchant as a trusted
        /// beneficiary. Issuers may require a cool-off period.
        added_at: DateTime<Utc>,
    },
    /// Delegated Authentication (PSD3 / RTS 2024).
    DelegatedAuthentication {
        /// Provider name (e.g. `"fido"`, `"applepay"`, `"googlepay"`).
        provider: String,
        /// Opaque evidence blob carried into the AReq.
        evidence: Vec<u8>,
        /// Class of the second factor used.
        factor_class: SecondFactorClass,
    },
    /// One-leg-out (issuer or acquirer outside the EEA).
    OneLeg {
        /// ISO 3166-1 alpha-2 country code of the issuer.
        issuer_country: String,
        /// ISO 3166-1 alpha-2 country code of the acquirer.
        acquirer_country: String,
    },
}

impl EligibleExemption {
    /// The `threeDSRequestorChallengeInd` value the exemption maps to.
    /// Used by the AReq codec to set the field.
    #[must_use]
    pub const fn challenge_indicator(&self) -> &'static str {
        match self {
            Self::LowValueTransaction { .. } => "08",
            Self::TransactionRiskAnalysis { .. } => "05",
            Self::SecureCorporatePayment { .. } => "07",
            Self::TrustedBeneficiary { .. } => "09",
            Self::DelegatedAuthentication { .. } => "06",
            Self::MerchantInitiated { .. } | Self::Recurring { .. } | Self::OneLeg { .. } => "02",
        }
    }

    /// `transStatus` letter this exemption produces if accepted.
    #[must_use]
    pub const fn implied_status(&self) -> TransactionStatus {
        match self {
            Self::OneLeg { .. } => TransactionStatus::InfoOnly,
            _ => TransactionStatus::AttemptedSuccessful,
        }
    }
}

/// Lightweight stand-in for the orchestrator's PaymentIntent so the
/// evaluator stays free of an `op-orchestrator` dependency (which
/// would otherwise be circular).
#[derive(Debug, Clone)]
pub struct PaymentIntent {
    /// Amount to be charged.
    pub amount: Money,
    /// PAN being charged (used only for corporate-BIN heuristic).
    pub pan: String,
    /// Cardholder-supplied indication that this is recurring.
    pub recurring: bool,
    /// Merchant-initiated indicator.
    pub merchant_initiated: bool,
    /// Mandate reference, if a recurring/MIT mandate exists.
    pub mandate_ref: Option<String>,
    /// Subscription id, if recurring.
    pub subscription_id: Option<String>,
}

/// Runtime context the evaluator combines with the intent.
#[derive(Debug, Clone, Default)]
pub struct ExemptionContext {
    /// Cumulative low-value charges in the trailing 24 h, in EUR
    /// minor units, for this cardholder.
    pub cumulative_low_value_24h_eur_minor: i64,
    /// Count of consecutive low-value-exempt payments without SCA.
    pub consecutive_low_value_count: u32,
    /// Requestor's PSP-certified fraud-rate bracket.
    pub fraud_rate_bracket: FraudRateBracket,
    /// Optional TRA risk score for this transaction.
    pub tra_score: Option<f32>,
    /// True if the BIN range is corporate / lodge per scheme tables.
    pub corporate_card_indicator: bool,
    /// Trusted-beneficiary record on file with the issuer, if any.
    pub trusted_beneficiary_id: Option<String>,
    /// Trusted-beneficiary enrolment timestamp.
    pub trusted_beneficiary_added_at: Option<DateTime<Utc>>,
    /// Delegated-authentication evidence and provider, if any.
    pub delegated_auth: Option<(String, Vec<u8>, SecondFactorClass)>,
    /// Issuer country (ISO 3166-1 alpha-2).
    pub issuer_country: Option<String>,
    /// Acquirer country (ISO 3166-1 alpha-2).
    pub acquirer_country: Option<String>,
}

/// Evaluate all eligible SCA exemptions for the intent + context, in
/// preference order (cheapest, least-risk-budget first).
///
/// The orchestrator typically picks the first element of the returned
/// vec. Callers that prefer a specific exemption can scan the vec.
#[must_use]
pub fn evaluate(intent: &PaymentIntent, ctx: &ExemptionContext) -> Vec<EligibleExemption> {
    let mut out = Vec::new();

    // 1. One-leg-out: cheapest of all, no fraud-rate constraint,
    //    no liability exposure to the issuer. Triggers when either
    //    leg sits outside the EEA.
    if let (Some(iss), Some(acq)) = (&ctx.issuer_country, &ctx.acquirer_country)
        && (!is_eea(iss) || !is_eea(acq))
    {
        out.push(EligibleExemption::OneLeg {
            issuer_country: iss.clone(),
            acquirer_country: acq.clone(),
        });
    }

    // 2. Trusted Beneficiary — high success, no challenge.
    if let (Some(id), Some(at)) = (
        ctx.trusted_beneficiary_id.as_ref(),
        ctx.trusted_beneficiary_added_at,
    ) {
        out.push(EligibleExemption::TrustedBeneficiary {
            beneficiary_id: id.clone(),
            added_at: at,
        });
    }

    // 3. Recurring (subsequent under mandate).
    if intent.recurring
        && let (Some(m), Some(sub)) = (intent.mandate_ref.clone(), intent.subscription_id.clone())
    {
        out.push(EligibleExemption::Recurring {
            mandate_ref: m,
            subscription_id: sub,
        });
    }

    // 4. Merchant-initiated.
    if intent.merchant_initiated
        && let Some(m) = intent.mandate_ref.clone()
    {
        out.push(EligibleExemption::MerchantInitiated { mandate_ref: m });
    }

    // 5. Delegated Authentication.
    if let Some((provider, evidence, factor_class)) = &ctx.delegated_auth {
        out.push(EligibleExemption::DelegatedAuthentication {
            provider: provider.clone(),
            evidence: evidence.clone(),
            factor_class: *factor_class,
        });
    }

    // 6. Secure Corporate Payment.
    if ctx.corporate_card_indicator {
        out.push(EligibleExemption::SecureCorporatePayment {
            corporate_card_indicator: true,
        });
    }

    // 7. Low-value.
    if low_value_eligible(intent, ctx) {
        out.push(EligibleExemption::LowValueTransaction {
            amount: intent.amount,
        });
    }

    // 8. TRA — only if the bracket permits and the score is supplied.
    if let (Some(score), Some(cap)) = (ctx.tra_score, ctx.fraud_rate_bracket.tra_cap_minor_units())
        && intent.amount.minor_units <= cap
        && (0.0..=1.0).contains(&score)
    {
        out.push(EligibleExemption::TransactionRiskAnalysis {
            score,
            fraud_rate_bracket: ctx.fraud_rate_bracket,
        });
    }

    out
}

/// Construct an exemption explicitly, returning an error if the
/// supplied context does not actually satisfy the regulatory rules.
/// Used by callers that want to enforce a specific exemption and fail
/// loudly when the runtime context isn't sufficient.
pub fn ensure(
    requested: &EligibleExemption,
    intent: &PaymentIntent,
    ctx: &ExemptionContext,
) -> Result<()> {
    let eligible = evaluate(intent, ctx);
    if eligible.iter().any(|e| matches_variant(e, requested)) {
        Ok(())
    } else {
        Err(Error::ExemptionIneligible(variant_name(requested)))
    }
}

fn matches_variant(a: &EligibleExemption, b: &EligibleExemption) -> bool {
    core::mem::discriminant(a) == core::mem::discriminant(b)
}

const fn variant_name(e: &EligibleExemption) -> &'static str {
    match e {
        EligibleExemption::LowValueTransaction { .. } => "LowValueTransaction",
        EligibleExemption::MerchantInitiated { .. } => "MerchantInitiated",
        EligibleExemption::Recurring { .. } => "Recurring",
        EligibleExemption::TransactionRiskAnalysis { .. } => "TransactionRiskAnalysis",
        EligibleExemption::SecureCorporatePayment { .. } => "SecureCorporatePayment",
        EligibleExemption::TrustedBeneficiary { .. } => "TrustedBeneficiary",
        EligibleExemption::DelegatedAuthentication { .. } => "DelegatedAuthentication",
        EligibleExemption::OneLeg { .. } => "OneLeg",
    }
}

fn low_value_eligible(intent: &PaymentIntent, ctx: &ExemptionContext) -> bool {
    // RTS 16 thresholds. We treat all non-EUR currencies conservatively
    // by checking the minor-unit value against EUR limits assuming a
    // 1:1 EUR equivalency; production deployments convert via op-fx.
    let single_cap = 30_00;
    let cumulative_cap = 100_00;
    let consecutive_cap = 5;
    intent.amount.minor_units <= single_cap
        && ctx.cumulative_low_value_24h_eur_minor + intent.amount.minor_units <= cumulative_cap
        && ctx.consecutive_low_value_count < consecutive_cap
}

const EEA_COUNTRIES: &[&str] = &[
    "AT", "BE", "BG", "HR", "CY", "CZ", "DK", "EE", "FI", "FR", "DE", "GR", "HU", "IS", "IE", "IT",
    "LV", "LI", "LT", "LU", "MT", "NL", "NO", "PL", "PT", "RO", "SK", "SI", "ES", "SE",
];

fn is_eea(country: &str) -> bool {
    EEA_COUNTRIES.iter().any(|c| c.eq_ignore_ascii_case(country))
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::Currency;

    fn eur(minor: i64) -> Money {
        Money::from_minor(minor, Currency::EUR)
    }

    fn base_intent(amount: i64) -> PaymentIntent {
        PaymentIntent {
            amount: eur(amount),
            pan: "4111111111111111".into(),
            recurring: false,
            merchant_initiated: false,
            mandate_ref: None,
            subscription_id: None,
        }
    }

    fn base_ctx() -> ExemptionContext {
        ExemptionContext {
            fraud_rate_bracket: FraudRateBracket::AboveTraThreshold,
            ..Default::default()
        }
    }

    #[test]
    fn low_value_25_eur_exempt() {
        let intent = base_intent(25_00);
        let ex = evaluate(&intent, &base_ctx());
        assert!(
            ex.iter()
                .any(|e| matches!(e, EligibleExemption::LowValueTransaction { .. })),
            "expected low-value exemption for 25 EUR, got: {ex:?}"
        );
    }

    #[test]
    fn low_value_100_eur_not_exempt_under_low_value() {
        let intent = base_intent(100_00);
        let ex = evaluate(&intent, &base_ctx());
        assert!(
            !ex.iter()
                .any(|e| matches!(e, EligibleExemption::LowValueTransaction { .. })),
            "100 EUR must not qualify as low-value"
        );
    }

    #[test]
    fn tra_eligible_when_bracket_and_score_supplied() {
        let intent = base_intent(80_00);
        let ctx = ExemptionContext {
            fraud_rate_bracket: FraudRateBracket::UpTo13Bp,
            tra_score: Some(0.05),
            ..Default::default()
        };
        let ex = evaluate(&intent, &ctx);
        assert!(
            ex.iter()
                .any(|e| matches!(e, EligibleExemption::TransactionRiskAnalysis { .. })),
            "expected TRA exemption to be eligible, got: {ex:?}"
        );
    }

    #[test]
    fn tra_capped_by_bracket() {
        let intent = base_intent(400_00); // 400 EUR
        let ctx = ExemptionContext {
            fraud_rate_bracket: FraudRateBracket::UpTo13Bp, // cap 100 EUR
            tra_score: Some(0.05),
            ..Default::default()
        };
        let ex = evaluate(&intent, &ctx);
        assert!(
            !ex.iter()
                .any(|e| matches!(e, EligibleExemption::TransactionRiskAnalysis { .. })),
            "400 EUR exceeds UpTo13Bp cap of 100 EUR"
        );

        // Same intent under UpTo1Bp (500 EUR cap) qualifies.
        let ctx2 = ExemptionContext {
            fraud_rate_bracket: FraudRateBracket::UpTo1Bp,
            tra_score: Some(0.05),
            ..Default::default()
        };
        let ex2 = evaluate(&intent, &ctx2);
        assert!(
            ex2.iter()
                .any(|e| matches!(e, EligibleExemption::TransactionRiskAnalysis { .. })),
            "400 EUR within UpTo1Bp 500 EUR cap"
        );
    }

    #[test]
    fn recurring_returns_recurring_exemption() {
        let intent = PaymentIntent {
            recurring: true,
            mandate_ref: Some("MNDT-1".into()),
            subscription_id: Some("SUB-1".into()),
            ..base_intent(15_00)
        };
        let ex = evaluate(&intent, &base_ctx());
        assert!(
            ex.iter()
                .any(|e| matches!(e, EligibleExemption::Recurring { .. }))
        );
    }

    #[test]
    fn one_leg_exempt_when_issuer_outside_eea() {
        let intent = base_intent(200_00);
        let ctx = ExemptionContext {
            issuer_country: Some("US".into()),
            acquirer_country: Some("DE".into()),
            ..Default::default()
        };
        let ex = evaluate(&intent, &ctx);
        assert!(
            ex.iter()
                .any(|e| matches!(e, EligibleExemption::OneLeg { .. })),
            "US issuer + DE acquirer must be one-leg-out"
        );
    }

    #[test]
    fn trusted_beneficiary_exemption() {
        let intent = base_intent(120_00);
        let ctx = ExemptionContext {
            trusted_beneficiary_id: Some("MERCH-42".into()),
            trusted_beneficiary_added_at: Some(Utc::now()),
            ..Default::default()
        };
        let ex = evaluate(&intent, &ctx);
        assert!(
            ex.iter()
                .any(|e| matches!(e, EligibleExemption::TrustedBeneficiary { .. }))
        );
    }

    #[test]
    fn delegated_auth_exemption() {
        let intent = base_intent(500_00);
        let ctx = ExemptionContext {
            delegated_auth: Some((
                "fido".into(),
                vec![0x01, 0x02, 0x03],
                SecondFactorClass::Fido,
            )),
            ..Default::default()
        };
        let ex = evaluate(&intent, &ctx);
        assert!(
            ex.iter()
                .any(|e| matches!(e, EligibleExemption::DelegatedAuthentication { .. }))
        );
    }

    #[test]
    fn ensure_passes_for_eligible_and_fails_otherwise() {
        let intent = base_intent(25_00);
        let ctx = base_ctx();
        let req = EligibleExemption::LowValueTransaction { amount: eur(25_00) };
        assert!(ensure(&req, &intent, &ctx).is_ok());

        let bad = EligibleExemption::TransactionRiskAnalysis {
            score: 0.1,
            fraud_rate_bracket: FraudRateBracket::AboveTraThreshold,
        };
        assert!(matches!(
            ensure(&bad, &intent, &ctx),
            Err(Error::ExemptionIneligible(_))
        ));
    }

    #[test]
    fn fraud_rate_caps_align_with_eba_table() {
        assert_eq!(FraudRateBracket::UpTo1Bp.tra_cap_minor_units(), Some(500_00));
        assert_eq!(FraudRateBracket::UpTo6Bp.tra_cap_minor_units(), Some(250_00));
        assert_eq!(FraudRateBracket::UpTo13Bp.tra_cap_minor_units(), Some(100_00));
        assert_eq!(FraudRateBracket::AboveTraThreshold.tra_cap_minor_units(), None);
    }
}
