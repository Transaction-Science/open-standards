//! Liability flows: who is on the hook?
//!
//! Two roles can hold the merchant of record (MoR) responsibility:
//!
//! - **Platform of record** — the platform is the merchant on the
//!   acquirer's books. Card-scheme chargebacks land on the platform,
//!   PCI scope inherits to the platform, and 1099-Ks are filed under
//!   the platform's TIN. This is the model Stripe Connect "Custom"
//!   and Adyen MarketPay sit in.
//!
//! - **Sub-merchant of record** — the sub-merchant is the merchant on
//!   the acquirer's books. Chargebacks, PCI scope, and tax reporting
//!   pass through to the sub-merchant. Stripe Connect "Standard" and
//!   Square "Sub-merchant of Record" sit here.
//!
//! - **Hybrid** — per-rail split. Card payments may be platform-of-record
//!   (for scheme compliance reasons) while A2A and crypto sit
//!   sub-merchant-of-record (because the rails don't define an MoR
//!   role). Operators converging from Stripe Connect "Express" land
//!   here.
//!
//! Each branch of the enum carries an explicit per-rail decision; we
//! reject one level of nesting (Hybrid inside Hybrid) at runtime via
//! [`LiabilityModel::validate`] because that representation produces
//! ambiguous attribution downstream.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Who holds merchant-of-record liability on each rail.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum LiabilityModel {
    /// Platform is MoR for every rail.
    PlatformOfRecord,
    /// Sub-merchant is MoR for every rail.
    SubMerchantOfRecord,
    /// Per-rail split. Each box must be a non-`Hybrid` variant.
    Hybrid {
        /// Card-rail liability (Visa / Mastercard / Amex / Discover).
        card: Box<LiabilityModel>,
        /// Account-to-account rail liability (ACH / FedNow / RTP / SEPA).
        a2a: Box<LiabilityModel>,
        /// Crypto-rail liability.
        crypto: Box<LiabilityModel>,
    },
}

impl LiabilityModel {
    /// Validate that nested branches of a `Hybrid` are themselves
    /// non-`Hybrid`. Nested hybrids produce ambiguous tax/dispute
    /// attribution.
    ///
    /// # Errors
    /// [`Error::InvalidLiabilityModel`] on nested-hybrid.
    pub fn validate(&self) -> Result<()> {
        if let Self::Hybrid { card, a2a, crypto } = self {
            for (name, branch) in [
                ("card", card.as_ref()),
                ("a2a", a2a.as_ref()),
                ("crypto", crypto.as_ref()),
            ] {
                if matches!(branch, Self::Hybrid { .. }) {
                    return Err(Error::InvalidLiabilityModel {
                        reason: format!("Hybrid nested inside Hybrid on branch `{name}`"),
                    });
                }
            }
        }
        Ok(())
    }

    /// Return the merchant-of-record for a given rail.
    #[must_use]
    pub fn for_rail(&self, rail: RailKind) -> Self {
        match self {
            Self::PlatformOfRecord => Self::PlatformOfRecord,
            Self::SubMerchantOfRecord => Self::SubMerchantOfRecord,
            Self::Hybrid { card, a2a, crypto } => match rail {
                RailKind::Card => (**card).clone(),
                RailKind::A2a => (**a2a).clone(),
                RailKind::Crypto => (**crypto).clone(),
            },
        }
    }

    /// True if the *platform* must file 1099-Ks under this model + rail.
    ///
    /// Per IRS Form 1099-K instructions (2024 revision), the Payment
    /// Settlement Entity (PSE) is the filer. The PSE is whichever
    /// party is the MoR for the underlying transaction. So
    /// PlatformOfRecord ⇒ platform files; SubMerchantOfRecord ⇒
    /// sub-merchant files (or its acquirer does).
    #[must_use]
    pub fn platform_files_1099k_for(&self, rail: RailKind) -> bool {
        matches!(self.for_rail(rail), Self::PlatformOfRecord)
    }

    /// True if the *platform* inherits PCI DSS scope under this model.
    ///
    /// Per PCI DSS v4.0.1 §A1, a service provider that processes,
    /// stores, or transmits cardholder data on behalf of others is in
    /// scope. PlatformOfRecord on the card rail ⇒ platform is the
    /// service provider ⇒ in-scope. SubMerchantOfRecord on the card
    /// rail ⇒ platform is a token-only conduit (assuming `op-vault`
    /// is wired correctly) ⇒ out-of-scope.
    #[must_use]
    pub fn platform_pci_inherits(&self) -> bool {
        matches!(self.for_rail(RailKind::Card), Self::PlatformOfRecord)
    }

    /// Dispute pass-through: who must respond to a chargeback?
    ///
    /// PlatformOfRecord ⇒ platform responds (and may pass-through to
    /// sub-merchant via internal-ledger transfer if it wishes).
    /// SubMerchantOfRecord ⇒ sub-merchant responds directly.
    #[must_use]
    pub fn dispute_responder(&self, rail: RailKind) -> DisputeResponder {
        match self.for_rail(rail) {
            Self::PlatformOfRecord => DisputeResponder::Platform,
            Self::SubMerchantOfRecord => DisputeResponder::SubMerchant,
            // Should be impossible after `validate()`; default to Platform
            // (the safer choice — they have the resources to push back).
            Self::Hybrid { .. } => DisputeResponder::Platform,
        }
    }
}

/// Which underlying rail an analysis is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RailKind {
    /// Card schemes (Visa / Mastercard / Amex / Discover).
    Card,
    /// Account-to-account (ACH / FedNow / RTP / SEPA / Pix / UPI).
    A2a,
    /// Crypto (BTC / ETH / stablecoins).
    Crypto,
}

/// Who must respond to a chargeback.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisputeResponder {
    /// Platform responds; may pass-through to sub-merchant internally.
    Platform,
    /// Sub-merchant responds directly to the acquirer.
    SubMerchant,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nested_hybrid_rejected() {
        let model = LiabilityModel::Hybrid {
            card: Box::new(LiabilityModel::Hybrid {
                card: Box::new(LiabilityModel::PlatformOfRecord),
                a2a: Box::new(LiabilityModel::PlatformOfRecord),
                crypto: Box::new(LiabilityModel::PlatformOfRecord),
            }),
            a2a: Box::new(LiabilityModel::SubMerchantOfRecord),
            crypto: Box::new(LiabilityModel::SubMerchantOfRecord),
        };
        let err = model.validate().expect_err("nested hybrid");
        assert!(matches!(err, Error::InvalidLiabilityModel { .. }));
    }

    #[test]
    fn hybrid_per_rail_resolves() {
        let model = LiabilityModel::Hybrid {
            card: Box::new(LiabilityModel::PlatformOfRecord),
            a2a: Box::new(LiabilityModel::SubMerchantOfRecord),
            crypto: Box::new(LiabilityModel::SubMerchantOfRecord),
        };
        model.validate().expect("ok");
        assert!(model.platform_pci_inherits());
        assert!(model.platform_files_1099k_for(RailKind::Card));
        assert!(!model.platform_files_1099k_for(RailKind::A2a));
        assert_eq!(model.dispute_responder(RailKind::A2a), DisputeResponder::SubMerchant);
    }

    #[test]
    fn pure_platform_inherits_pci() {
        let m = LiabilityModel::PlatformOfRecord;
        assert!(m.platform_pci_inherits());
        assert!(m.platform_files_1099k_for(RailKind::Card));
    }

    #[test]
    fn pure_sub_does_not_inherit_pci() {
        let m = LiabilityModel::SubMerchantOfRecord;
        assert!(!m.platform_pci_inherits());
        assert!(!m.platform_files_1099k_for(RailKind::Card));
    }
}
