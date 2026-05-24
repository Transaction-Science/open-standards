//! Pre-checkout eligibility checks.
//!
//! Each BNPL provider has its own eligibility rules:
//!
//! - **Country / region** — Afterpay is geographic by region; Klarna
//!   merchants are bound to one regional cloud; Affirm is US/CA only.
//! - **Currency** — providers only underwrite in their region's
//!   currencies.
//! - **Amount band** — there is an issuer-set min and max per region
//!   (e.g. Afterpay Pay-in-4 max $2,000 USD).
//! - **Consumer age** — minimum 18 in all provider regions.
//! - **Cart contents** — some MCC categories are excluded (e.g.
//!   firearms, gambling).
//!
//! This module is intentionally *deterministic*. No network. It runs
//! before any provider call, returning a fast pre-flight verdict so
//! the merchant doesn't fire the request only to have the provider
//! reject it after the consumer redirect.

use op_core::{Currency, Money};
use serde::{Deserialize, Serialize};

use crate::lifecycle::BnplProvider;

/// Context for the eligibility check.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EligibilityContext {
    /// Provider being considered.
    pub provider: BnplProvider,
    /// Consumer's shipping country (ISO 3166-1 alpha-2).
    pub country: String,
    /// Cart total.
    pub amount: Money,
    /// Currency.
    pub currency: Currency,
    /// Consumer's age, if known (Afterpay's flow needs it).
    pub consumer_age: Option<u8>,
}

/// Eligibility verdict.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EligibilityResult {
    /// Consumer + cart can use this provider.
    Eligible,
    /// Cannot use this provider for the given reason. Renderable to
    /// the consumer ("Try a different payment method").
    Ineligible {
        /// Stable machine-readable reason code.
        reason: IneligibilityReason,
        /// Human-friendly explanation.
        detail: String,
    },
}

/// Machine-readable reason codes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum IneligibilityReason {
    /// Provider does not operate in the consumer's country.
    CountryNotSupported,
    /// Cart currency is not underwritten by this provider in this
    /// country.
    CurrencyNotSupported,
    /// Cart total is below provider's minimum.
    AmountBelowMinimum,
    /// Cart total is above provider's maximum.
    AmountAboveMaximum,
    /// Consumer is younger than 18.
    UnderageConsumer,
}

/// Pluggable eligibility check.
///
/// Default impls below: [`StandardEligibility`] applies the providers'
/// public rule-of-thumb thresholds.
#[allow(async_fn_in_trait)] // dyn-compat is not required for this trait
pub trait EligibilityCheck {
    /// Run the check.
    async fn check(&self, ctx: &EligibilityContext) -> EligibilityResult;
}

/// Standard eligibility: deterministic rule table reflecting the
/// providers' published acceptance bands as of 2026.
///
/// The thresholds are intentionally conservative — the *provider* is
/// always authoritative on the actual decision at checkout time; this
/// check just filters out the obvious nos before the network round
/// trip.
#[derive(Copy, Clone, Debug, Default)]
pub struct StandardEligibility;

impl EligibilityCheck for StandardEligibility {
    async fn check(&self, ctx: &EligibilityContext) -> EligibilityResult {
        // Age gate: applies to all providers.
        if let Some(age) = ctx.consumer_age
            && age < 18
        {
            return EligibilityResult::Ineligible {
                reason: IneligibilityReason::UnderageConsumer,
                detail: format!("consumer is {age}, minimum is 18"),
            };
        }

        match ctx.provider {
            BnplProvider::Affirm => check_affirm(ctx),
            BnplProvider::Klarna => check_klarna(ctx),
            BnplProvider::AfterpayClearpay => check_afterpay(ctx),
        }
    }
}

/// Affirm: US + Canada, USD/CAD, $50–$30,000.
fn check_affirm(ctx: &EligibilityContext) -> EligibilityResult {
    let country = ctx.country.as_str();
    if !matches!(country, "US" | "CA") {
        return EligibilityResult::Ineligible {
            reason: IneligibilityReason::CountryNotSupported,
            detail: format!("Affirm operates in US/CA only; got {country}"),
        };
    }
    let code = ctx.currency.code();
    let supported = matches!(code, "USD" | "CAD");
    if !supported {
        return EligibilityResult::Ineligible {
            reason: IneligibilityReason::CurrencyNotSupported,
            detail: format!("Affirm does not underwrite {code}"),
        };
    }
    band_check(ctx.amount, 5_000, 3_000_000)
}

const KLARNA_COUNTRIES: &[&str] = &[
    "US", "CA", "MX", "GB", "DE", "AT", "CH", "NL", "BE", "DK", "SE", "NO", "FI", "ES", "IT",
    "FR", "IE", "PL", "PT", "AU", "NZ",
];

const AFTERPAY_COUNTRIES: &[&str] = &[
    "US", "AU", "NZ", "GB", "CA", "DE", "ES", "FR", "IT", "IE",
];

/// Klarna: EU + UK + US + AU + NZ + CA + MX; varies by region.
/// Conservative band: $1–$10,000 equivalent.
fn check_klarna(ctx: &EligibilityContext) -> EligibilityResult {
    let country = ctx.country.as_str();
    if !KLARNA_COUNTRIES.contains(&country) {
        return EligibilityResult::Ineligible {
            reason: IneligibilityReason::CountryNotSupported,
            detail: format!("Klarna does not operate in {country}"),
        };
    }
    band_check(ctx.amount, 100, 1_000_000)
}

/// Afterpay/Clearpay: US/AU/NZ/GB/CA/EU; Pay-in-4 max ~$2,000.
fn check_afterpay(ctx: &EligibilityContext) -> EligibilityResult {
    let country = ctx.country.as_str();
    if !AFTERPAY_COUNTRIES.contains(&country) {
        return EligibilityResult::Ineligible {
            reason: IneligibilityReason::CountryNotSupported,
            detail: format!("Afterpay does not operate in {country}"),
        };
    }
    let code = ctx.currency.code();
    let allowed = match country {
        "US" => code == "USD",
        "AU" => code == "AUD",
        "NZ" => code == "NZD",
        "GB" => code == "GBP",
        "CA" => code == "CAD",
        _ => code == "EUR",
    };
    if !allowed {
        return EligibilityResult::Ineligible {
            reason: IneligibilityReason::CurrencyNotSupported,
            detail: format!("Afterpay in {country} requires region-native currency, got {code}"),
        };
    }
    band_check(ctx.amount, 100, 20_0000)
}

/// Bounds check on the cart total (in the currency's own minor units).
fn band_check(amount: Money, min: i64, max: i64) -> EligibilityResult {
    if amount.minor_units < min {
        return EligibilityResult::Ineligible {
            reason: IneligibilityReason::AmountBelowMinimum,
            detail: format!("amount {} below minimum {min}", amount.minor_units),
        };
    }
    if amount.minor_units > max {
        return EligibilityResult::Ineligible {
            reason: IneligibilityReason::AmountAboveMaximum,
            detail: format!("amount {} above maximum {max}", amount.minor_units),
        };
    }
    EligibilityResult::Eligible
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ctx(provider: BnplProvider, country: &str, amount: i64, currency: Currency) -> EligibilityContext {
        EligibilityContext {
            provider,
            country: country.into(),
            amount: Money::from_minor(amount, currency),
            currency,
            consumer_age: Some(30),
        }
    }

    #[tokio::test]
    async fn affirm_us_usd_eligible() {
        let r = StandardEligibility
            .check(&ctx(BnplProvider::Affirm, "US", 10_000, Currency::USD))
            .await;
        assert_eq!(r, EligibilityResult::Eligible);
    }

    #[tokio::test]
    async fn affirm_eu_ineligible_country() {
        let r = StandardEligibility
            .check(&ctx(BnplProvider::Affirm, "DE", 10_000, Currency::EUR))
            .await;
        assert!(matches!(
            r,
            EligibilityResult::Ineligible {
                reason: IneligibilityReason::CountryNotSupported,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn underage_rejected_for_all_providers() {
        let mut c = ctx(BnplProvider::Klarna, "DE", 10_000, Currency::EUR);
        c.consumer_age = Some(16);
        let r = StandardEligibility.check(&c).await;
        assert!(matches!(
            r,
            EligibilityResult::Ineligible {
                reason: IneligibilityReason::UnderageConsumer,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn afterpay_us_amount_too_high() {
        let r = StandardEligibility
            .check(&ctx(BnplProvider::AfterpayClearpay, "US", 500_000, Currency::USD))
            .await;
        assert!(matches!(
            r,
            EligibilityResult::Ineligible {
                reason: IneligibilityReason::AmountAboveMaximum,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn afterpay_uk_eur_currency_wrong() {
        let r = StandardEligibility
            .check(&ctx(BnplProvider::AfterpayClearpay, "GB", 10_000, Currency::EUR))
            .await;
        assert!(matches!(
            r,
            EligibilityResult::Ineligible {
                reason: IneligibilityReason::CurrencyNotSupported,
                ..
            }
        ));
    }

    #[tokio::test]
    async fn klarna_eu_eligible() {
        let r = StandardEligibility
            .check(&ctx(BnplProvider::Klarna, "DE", 20_000, Currency::EUR))
            .await;
        assert_eq!(r, EligibilityResult::Eligible);
    }

    #[tokio::test]
    async fn klarna_mx_supported() {
        let r = StandardEligibility
            .check(&ctx(BnplProvider::Klarna, "MX", 20_000, Currency::USD))
            .await;
        assert_eq!(r, EligibilityResult::Eligible);
    }

    #[tokio::test]
    async fn affirm_amount_below_min() {
        let r = StandardEligibility
            .check(&ctx(BnplProvider::Affirm, "US", 1_000, Currency::USD))
            .await;
        assert!(matches!(
            r,
            EligibilityResult::Ineligible {
                reason: IneligibilityReason::AmountBelowMinimum,
                ..
            }
        ));
    }
}
