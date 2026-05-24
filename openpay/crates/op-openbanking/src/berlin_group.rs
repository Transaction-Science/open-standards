//! Berlin Group NextGenPSD2 XS2A binding.
//!
//! Reference: Berlin Group, *NextGenPSD2 XS2A Framework Implementation
//! Guidelines* v1.3.13 (Berlin Group OpenFinance API
//! Framework). Endpoints rooted at `/v1/{accounts|consents|payments|
//! funds-confirmations}`.
//!
//! Berlin Group is the most-adopted PSD2 dialect across the EEA;
//! almost every continental-European ASPSP runs an XS2A implementation
//! that follows these IGs. The wire format is ISO 20022-aligned
//! (`pain.001` payloads in `payments/sepa-credit-transfers` etc.).

use serde::{Deserialize, Serialize};

/// Berlin Group payment-product taxonomy. The `payment-product` URL
/// segment in `POST /payments/{payment-product}` is one of these.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum BerlinPaymentProduct {
    /// `sepa-credit-transfers` — standard SEPA SCT.
    SepaCreditTransfers,
    /// `instant-sepa-credit-transfers` — SEPA SCT Inst (TIPS / RT1).
    InstantSepaCreditTransfers,
    /// `target-2-payments` — TARGET2 high-value EUR.
    Target2Payments,
    /// `cross-border-credit-transfers` — non-EUR / cross-border.
    CrossBorderCreditTransfers,
}

impl BerlinPaymentProduct {
    /// URL segment used in the payment-initiation endpoint.
    #[must_use]
    pub const fn as_segment(self) -> &'static str {
        match self {
            Self::SepaCreditTransfers => "sepa-credit-transfers",
            Self::InstantSepaCreditTransfers => "instant-sepa-credit-transfers",
            Self::Target2Payments => "target-2-payments",
            Self::CrossBorderCreditTransfers => "cross-border-credit-transfers",
        }
    }
}

/// Berlin Group consent body. The IGs allow `allPsd2`, `availableAccounts`,
/// or a per-account `access` object naming IBANs and the read scopes
/// granted. We model the per-account variant; the wildcard cases are
/// modelled by populating no IBAN set and setting [`Self::all_psd2`].
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BerlinConsent {
    /// IBANs covered by the consent (empty when `all_psd2== true`).
    pub ibans: Vec<String>,
    /// True ⇒ `access.allPsd2: "allAccounts"` (account list + balances
    /// + transactions, all accounts).
    pub all_psd2: bool,
    /// True ⇒ `recurringIndicator: true` (long-lived AISP consent;
    /// allows up to 4 reads/day per IBAN under SCA-exemption rules).
    pub recurring: bool,
    /// `validUntil` per the IGs (`YYYY-MM-DD`).
    pub valid_until: time::Date,
    /// `frequencyPerDay` — count of permitted reads per day on a
    /// recurring consent. Default 4 per PSD2 RTS Article 10.
    pub frequency_per_day: u32,
}

/// Berlin Group XS2A service handle.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct BerlinService {
    /// ASPSP base URL (opaque).
    pub aspsp_base_url: String,
    /// ASPSP identifier (BIC or operator-internal label).
    pub aspsp_id: String,
}

impl BerlinService {
    /// Construct a full endpoint URL.
    #[must_use]
    pub fn endpoint(&self, path: &str) -> String {
        let trimmed_base = self.aspsp_base_url.trim_end_matches('/');
        let trimmed_path = path.trim_start_matches('/');
        format!("{}/v1/{}", trimmed_base, trimmed_path)
    }

    /// Construct the payment-initiation endpoint for a product.
    #[must_use]
    pub fn payments_endpoint(&self, product: BerlinPaymentProduct) -> String {
        self.endpoint(&format!("payments/{}", product.as_segment()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn product_segments_match_spec() {
        assert_eq!(
            BerlinPaymentProduct::SepaCreditTransfers.as_segment(),
            "sepa-credit-transfers"
        );
        assert_eq!(
            BerlinPaymentProduct::InstantSepaCreditTransfers.as_segment(),
            "instant-sepa-credit-transfers"
        );
    }

    #[test]
    fn payments_endpoint_constructs() {
        let svc = BerlinService {
            aspsp_base_url: "https://xs2a.aspsp.example".into(),
            aspsp_id: "DEUTDEFFXXX".into(),
        };
        let url = svc.payments_endpoint(BerlinPaymentProduct::SepaCreditTransfers);
        assert_eq!(
            url,
            "https://xs2a.aspsp.example/v1/payments/sepa-credit-transfers"
        );
    }
}
