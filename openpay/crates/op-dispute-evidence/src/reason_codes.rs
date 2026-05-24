//! Reason-code catalog and required-evidence map.
//!
//! Each card network publishes, per reason code, the bundle of
//! evidence the merchant must include in the representment for the
//! issuer to consider reversing the chargeback. This module is the
//! canonical lookup table.
//!
//! The catalog is intentionally a static lookup function rather
//! than a `HashMap` — the data is small, fixed, and benefits from
//! exhaustiveness checking at compile time (a new
//! [`crate::network::VisaReasonCode`] variant breaks the match,
//! forcing the maintainer to declare its evidence requirements).

use serde::{Deserialize, Serialize};

use crate::network::{
    AmexReasonCode, DiscoverReasonCode, MastercardReasonCode, Network, PayPalReasonCode,
    VisaReasonCode,
};

/// A specific class of evidence the network may require.
///
/// These map directly to the entries on every PSP's
/// "evidence-upload" form (Stripe, Adyen, Braintree, etc.) and
/// align with the categories called out in the directive's
/// evidence catalog (AVS/CVV, 3DS auth value, delivery, comms,
/// IP / device fingerprint).
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
pub enum EvidenceRequirement {
    /// Original receipt / invoice.
    Receipt,
    /// Address Verification Service result from auth.
    AvsResult,
    /// CVV / CVC2 verification result.
    CvvResult,
    /// 3-D Secure 2 cryptogram / auth value (CAVV / AAV / AEVV).
    ThreeDsAuthValue,
    /// Shipping carrier + tracking number + delivery confirmation
    /// (signature when available).
    ProofOfDelivery,
    /// Customer service / email / chat transcripts demonstrating
    /// the cardholder was engaged.
    CustomerCommunications,
    /// IP address used at checkout.
    CheckoutIp,
    /// Device fingerprint (User-Agent + canvas + storage hash).
    DeviceFingerprint,
    /// Terms of service / refund policy the cardholder accepted.
    TermsOfService,
    /// Prior, undisputed transactions from the same cardholder —
    /// the CE3.0 qualifying transactions.
    QualifyingHistory,
    /// Proof the cardholder cancelled (or did not cancel) a
    /// recurring subscription.
    SubscriptionCancellation,
    /// Refund / credit issued evidence (for "credit not processed"
    /// reasons).
    RefundReceipt,
    /// Authorization log entries proving an approval existed and
    /// matched the captured amount.
    AuthorizationLog,
}

/// A reason code in any network, with its network of origin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ReasonCode {
    /// Visa VCR reason code.
    Visa(VisaReasonCode),
    /// Mastercard Mastercom reason code.
    Mastercard(MastercardReasonCode),
    /// American Express reason code.
    Amex(AmexReasonCode),
    /// Discover DRR reason code.
    Discover(DiscoverReasonCode),
    /// PayPal dispute reason.
    PayPal(PayPalReasonCode),
}

impl ReasonCode {
    /// The originating network.
    #[must_use]
    pub const fn network(self) -> Network {
        match self {
            Self::Visa(_) => Network::Visa,
            Self::Mastercard(_) => Network::Mastercard,
            Self::Amex(_) => Network::Amex,
            Self::Discover(_) => Network::Discover,
            Self::PayPal(_) => Network::PayPal,
        }
    }

    /// Canonical wire string for the code.
    #[must_use]
    pub fn code(self) -> &'static str {
        match self {
            Self::Visa(c) => c.code(),
            Self::Mastercard(c) => c.code(),
            Self::Amex(c) => c.code(),
            Self::Discover(c) => c.code(),
            Self::PayPal(c) => c.code(),
        }
    }
}

/// Static catalog mapping a [`ReasonCode`] to its required evidence
/// list and a short human-readable category.
///
/// The catalog itself holds no state; it's a thin namespace.
#[derive(Debug, Clone, Copy)]
pub struct ReasonCodeCatalog;

impl ReasonCodeCatalog {
    /// Required evidence items for a given reason code.
    ///
    /// Returned slices are stable across calls and embedded in the
    /// binary — no allocation.
    #[must_use]
    pub fn required_evidence(code: ReasonCode) -> &'static [EvidenceRequirement] {
        match code {
            ReasonCode::Visa(c) => Self::visa(c),
            ReasonCode::Mastercard(c) => Self::mastercard(c),
            ReasonCode::Amex(c) => Self::amex(c),
            ReasonCode::Discover(c) => Self::discover(c),
            ReasonCode::PayPal(c) => Self::paypal(c),
        }
    }

    /// Short human label for a reason code (English).
    #[must_use]
    pub fn label(code: ReasonCode) -> &'static str {
        match code {
            ReasonCode::Visa(c) => Self::visa_label(c),
            ReasonCode::Mastercard(c) => Self::mastercard_label(c),
            ReasonCode::Amex(c) => Self::amex_label(c),
            ReasonCode::Discover(c) => Self::discover_label(c),
            ReasonCode::PayPal(c) => Self::paypal_label(c),
        }
    }

    // ---------------- Visa ---------------------------------------

    const fn visa(c: VisaReasonCode) -> &'static [EvidenceRequirement] {
        use EvidenceRequirement::{
            AuthorizationLog, AvsResult, CheckoutIp, CustomerCommunications, CvvResult,
            DeviceFingerprint, ProofOfDelivery, QualifyingHistory, Receipt, RefundReceipt,
            SubscriptionCancellation, TermsOfService, ThreeDsAuthValue,
        };
        match c {
            // Fraud: card-absent. CE3.0-eligible — qualifying
            // history is the heaviest hitter.
            VisaReasonCode::F1040 => &[
                Receipt,
                AvsResult,
                CvvResult,
                ThreeDsAuthValue,
                ProofOfDelivery,
                CheckoutIp,
                DeviceFingerprint,
                QualifyingHistory,
            ],
            // Other fraud variants — heavy on auth, light on
            // history because they're not CE3.0-targetable.
            VisaReasonCode::F1010
            | VisaReasonCode::F1020
            | VisaReasonCode::F1030
            | VisaReasonCode::F1050 => &[
                Receipt,
                AvsResult,
                CvvResult,
                ThreeDsAuthValue,
                ProofOfDelivery,
            ],
            // Authorization chapter.
            VisaReasonCode::A1110 | VisaReasonCode::A1120 | VisaReasonCode::A1130 => {
                &[Receipt, AuthorizationLog]
            }
            // Processing chapter.
            VisaReasonCode::P1210
            | VisaReasonCode::P1220
            | VisaReasonCode::P1230
            | VisaReasonCode::P1240
            | VisaReasonCode::P1250
            | VisaReasonCode::P1260
            | VisaReasonCode::P1270 => &[Receipt, AuthorizationLog],
            // Consumer disputes — services not delivered / quality.
            VisaReasonCode::C1310 => &[Receipt, ProofOfDelivery, CustomerCommunications],
            VisaReasonCode::C1320 | VisaReasonCode::C1370 => {
                &[Receipt, SubscriptionCancellation, TermsOfService]
            }
            VisaReasonCode::C1330 | VisaReasonCode::C1340 | VisaReasonCode::C1350 => &[
                Receipt,
                ProofOfDelivery,
                CustomerCommunications,
                TermsOfService,
            ],
            VisaReasonCode::C1360 => &[Receipt, RefundReceipt],
            VisaReasonCode::C1380 | VisaReasonCode::C1390 => &[Receipt, AuthorizationLog],
        }
    }

    const fn visa_label(c: VisaReasonCode) -> &'static str {
        match c {
            VisaReasonCode::F1010 => "EMV liability shift counterfeit fraud",
            VisaReasonCode::F1020 => "EMV liability shift non-counterfeit fraud",
            VisaReasonCode::F1030 => "Other fraud — card-present environment",
            VisaReasonCode::F1040 => "Other fraud — card-absent environment",
            VisaReasonCode::F1050 => "Visa fraud monitoring program",
            VisaReasonCode::A1110 => "Card recovery bulletin",
            VisaReasonCode::A1120 => "Declined authorization",
            VisaReasonCode::A1130 => "No authorization",
            VisaReasonCode::P1210 => "Late presentment",
            VisaReasonCode::P1220 => "Incorrect transaction code",
            VisaReasonCode::P1230 => "Incorrect currency",
            VisaReasonCode::P1240 => "Incorrect account number",
            VisaReasonCode::P1250 => "Incorrect amount",
            VisaReasonCode::P1260 => "Duplicate processing / paid by other means",
            VisaReasonCode::P1270 => "Invalid data",
            VisaReasonCode::C1310 => "Merchandise / services not received",
            VisaReasonCode::C1320 => "Cancelled recurring",
            VisaReasonCode::C1330 => "Not as described or defective merchandise",
            VisaReasonCode::C1340 => "Counterfeit merchandise",
            VisaReasonCode::C1350 => "Misrepresentation",
            VisaReasonCode::C1360 => "Credit not processed",
            VisaReasonCode::C1370 => "Cancelled merchandise / services",
            VisaReasonCode::C1380 => "Original credit transaction not accepted",
            VisaReasonCode::C1390 => "Non-receipt of cash or load transaction value",
        }
    }

    // ---------------- Mastercard ---------------------------------

    const fn mastercard(c: MastercardReasonCode) -> &'static [EvidenceRequirement] {
        use EvidenceRequirement::{
            AuthorizationLog, AvsResult, CheckoutIp, CustomerCommunications, CvvResult,
            DeviceFingerprint, ProofOfDelivery, Receipt, RefundReceipt, TermsOfService,
            ThreeDsAuthValue,
        };
        match c {
            // Fraud (card-absent).
            MastercardReasonCode::Mc4837 | MastercardReasonCode::Mc4863 => &[
                Receipt,
                AvsResult,
                CvvResult,
                ThreeDsAuthValue,
                ProofOfDelivery,
                CheckoutIp,
                DeviceFingerprint,
            ],
            MastercardReasonCode::Mc4840 => &[Receipt, AuthorizationLog],
            MastercardReasonCode::Mc4849 => &[Receipt, AuthorizationLog, TermsOfService],
            // Cardholder dispute.
            MastercardReasonCode::Mc4853 | MastercardReasonCode::Mc4855 => &[
                Receipt,
                ProofOfDelivery,
                CustomerCommunications,
                TermsOfService,
            ],
            MastercardReasonCode::Mc4859 => &[Receipt, ProofOfDelivery, CustomerCommunications],
            MastercardReasonCode::Mc4860 => &[Receipt, RefundReceipt],
            // EMV liability shift / chip + PIN. Card-present
            // chip-fraud cases — historically near-impossible to win
            // without proof the terminal was chip-capable.
            MastercardReasonCode::Mc4870 | MastercardReasonCode::Mc4871 => {
                &[Receipt, AuthorizationLog]
            }
        }
    }

    const fn mastercard_label(c: MastercardReasonCode) -> &'static str {
        match c {
            MastercardReasonCode::Mc4837 => "No cardholder authorization",
            MastercardReasonCode::Mc4840 => "Fraudulent processing of transactions",
            MastercardReasonCode::Mc4849 => "Questionable merchant activity",
            MastercardReasonCode::Mc4853 => "Cardholder dispute",
            MastercardReasonCode::Mc4855 => "Goods or services not provided",
            MastercardReasonCode::Mc4859 => "Services not rendered / addendum / ATM dispute",
            MastercardReasonCode::Mc4860 => "Credit not processed",
            MastercardReasonCode::Mc4863 => "Cardholder does not recognize",
            MastercardReasonCode::Mc4870 => "Chip liability shift",
            MastercardReasonCode::Mc4871 => "Chip / PIN liability shift",
        }
    }

    // ---------------- Amex ---------------------------------------

    const fn amex(c: AmexReasonCode) -> &'static [EvidenceRequirement] {
        use EvidenceRequirement::{
            AuthorizationLog, AvsResult, CheckoutIp, CustomerCommunications, CvvResult,
            DeviceFingerprint, ProofOfDelivery, Receipt, RefundReceipt,
            SubscriptionCancellation, TermsOfService, ThreeDsAuthValue,
        };
        match c {
            AmexReasonCode::F24 | AmexReasonCode::F29 => &[
                Receipt,
                AvsResult,
                CvvResult,
                ThreeDsAuthValue,
                ProofOfDelivery,
                CheckoutIp,
                DeviceFingerprint,
            ],
            AmexReasonCode::F30 | AmexReasonCode::F31 => &[Receipt, AuthorizationLog],
            AmexReasonCode::C02 => &[Receipt, RefundReceipt],
            AmexReasonCode::C04 | AmexReasonCode::C05 | AmexReasonCode::C08 => &[
                Receipt,
                ProofOfDelivery,
                CustomerCommunications,
                TermsOfService,
            ],
            AmexReasonCode::C14 => &[Receipt, AuthorizationLog],
            AmexReasonCode::C18 => &[Receipt, TermsOfService, CustomerCommunications],
            AmexReasonCode::C28 => &[Receipt, SubscriptionCancellation, TermsOfService],
            AmexReasonCode::C31 | AmexReasonCode::C32 => &[
                Receipt,
                CustomerCommunications,
                ProofOfDelivery,
                TermsOfService,
            ],
            AmexReasonCode::A01
            | AmexReasonCode::A02
            | AmexReasonCode::A08
            | AmexReasonCode::P01
            | AmexReasonCode::P03
            | AmexReasonCode::P04
            | AmexReasonCode::P05
            | AmexReasonCode::P07 => &[Receipt, AuthorizationLog],
        }
    }

    const fn amex_label(c: AmexReasonCode) -> &'static str {
        match c {
            AmexReasonCode::F24 => "No cardmember authorization",
            AmexReasonCode::F29 => "Card-not-present fraud",
            AmexReasonCode::F30 => "EMV counterfeit fraud",
            AmexReasonCode::F31 => "EMV lost / stolen / non-received fraud",
            AmexReasonCode::C02 => "Credit not processed",
            AmexReasonCode::C04 => "Goods / services returned or refused",
            AmexReasonCode::C05 => "Goods / services cancelled",
            AmexReasonCode::C08 => "Goods / services not received",
            AmexReasonCode::C14 => "Paid by other means",
            AmexReasonCode::C18 => "No show or CARDeposit cancelled",
            AmexReasonCode::C28 => "Cancelled recurring billing",
            AmexReasonCode::C31 => "Goods / services not as described",
            AmexReasonCode::C32 => "Goods / services damaged or defective",
            AmexReasonCode::A01 => "Charge amount exceeds authorization amount",
            AmexReasonCode::A02 => "No valid authorization",
            AmexReasonCode::A08 => "Authorization approval expired",
            AmexReasonCode::P01 => "Unassigned card number",
            AmexReasonCode::P03 => "Credit processed as charge",
            AmexReasonCode::P04 => "Charge processed as credit",
            AmexReasonCode::P05 => "Incorrect charge amount",
            AmexReasonCode::P07 => "Late submission",
        }
    }

    // ---------------- Discover -----------------------------------

    const fn discover(c: DiscoverReasonCode) -> &'static [EvidenceRequirement] {
        use EvidenceRequirement::{
            AuthorizationLog, AvsResult, CheckoutIp, CustomerCommunications, CvvResult,
            DeviceFingerprint, ProofOfDelivery, Receipt, RefundReceipt, TermsOfService,
            ThreeDsAuthValue,
        };
        match c {
            DiscoverReasonCode::UA01 | DiscoverReasonCode::UA02 => &[
                Receipt,
                AvsResult,
                CvvResult,
                ThreeDsAuthValue,
                ProofOfDelivery,
                CheckoutIp,
                DeviceFingerprint,
            ],
            DiscoverReasonCode::UA05 | DiscoverReasonCode::UA06 => &[Receipt, AuthorizationLog],
            DiscoverReasonCode::AT
            | DiscoverReasonCode::LP
            | DiscoverReasonCode::IC
            | DiscoverReasonCode::DA => &[Receipt, AuthorizationLog],
            DiscoverReasonCode::RG => &[Receipt, ProofOfDelivery, CustomerCommunications],
            DiscoverReasonCode::RM => &[
                Receipt,
                ProofOfDelivery,
                CustomerCommunications,
                TermsOfService,
            ],
            DiscoverReasonCode::RN1 => &[Receipt, RefundReceipt],
            DiscoverReasonCode::RN2 => &[Receipt, TermsOfService, CustomerCommunications],
        }
    }

    const fn discover_label(c: DiscoverReasonCode) -> &'static str {
        match c {
            DiscoverReasonCode::UA01 => "Fraud — card present",
            DiscoverReasonCode::UA02 => "Fraud — card not present",
            DiscoverReasonCode::UA05 => "Fraud — chip counterfeit",
            DiscoverReasonCode::UA06 => "Fraud — chip lost / stolen / non-received",
            DiscoverReasonCode::AT => "Authorization",
            DiscoverReasonCode::RG => "Non-receipt of goods or services",
            DiscoverReasonCode::RM => "Cardholder dispute — quality",
            DiscoverReasonCode::RN1 => "Credit not processed",
            DiscoverReasonCode::RN2 => "Cancelled merchandise / services",
            DiscoverReasonCode::LP => "Late presentment",
            DiscoverReasonCode::IC => "Invalid card number",
            DiscoverReasonCode::DA => "Declined authorization",
        }
    }

    // ---------------- PayPal -------------------------------------

    const fn paypal(c: PayPalReasonCode) -> &'static [EvidenceRequirement] {
        use EvidenceRequirement::{
            AvsResult, CheckoutIp, CustomerCommunications, DeviceFingerprint, ProofOfDelivery,
            Receipt, RefundReceipt, SubscriptionCancellation, TermsOfService, ThreeDsAuthValue,
        };
        match c {
            PayPalReasonCode::Unauthorized => &[
                Receipt,
                AvsResult,
                ThreeDsAuthValue,
                CheckoutIp,
                DeviceFingerprint,
                ProofOfDelivery,
            ],
            PayPalReasonCode::ItemNotReceived => {
                &[Receipt, ProofOfDelivery, CustomerCommunications]
            }
            PayPalReasonCode::SignificantlyNotAsDescribed => &[
                Receipt,
                ProofOfDelivery,
                CustomerCommunications,
                TermsOfService,
            ],
            PayPalReasonCode::Duplicate => &[Receipt, RefundReceipt],
            PayPalReasonCode::CancelledRecurring => {
                &[Receipt, SubscriptionCancellation, TermsOfService]
            }
            PayPalReasonCode::CreditNotProcessed => &[Receipt, RefundReceipt],
        }
    }

    const fn paypal_label(c: PayPalReasonCode) -> &'static str {
        match c {
            PayPalReasonCode::Unauthorized => "Unauthorized",
            PayPalReasonCode::ItemNotReceived => "Item not received",
            PayPalReasonCode::SignificantlyNotAsDescribed => "Significantly not as described",
            PayPalReasonCode::Duplicate => "Duplicate transaction",
            PayPalReasonCode::CancelledRecurring => "Cancelled recurring",
            PayPalReasonCode::CreditNotProcessed => "Credit not processed",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visa_10_4_requires_qualifying_history() {
        let req = ReasonCodeCatalog::required_evidence(ReasonCode::Visa(VisaReasonCode::F1040));
        assert!(req.contains(&EvidenceRequirement::QualifyingHistory));
        assert!(req.contains(&EvidenceRequirement::ThreeDsAuthValue));
    }

    #[test]
    fn mastercard_4853_does_not_demand_3ds() {
        // 4853 is a cardholder-dispute (quality) code; 3DS auth value
        // is not part of the representment bundle.
        let req = ReasonCodeCatalog::required_evidence(ReasonCode::Mastercard(
            MastercardReasonCode::Mc4853,
        ));
        assert!(!req.contains(&EvidenceRequirement::ThreeDsAuthValue));
    }

    #[test]
    fn paypal_inr_wants_proof_of_delivery() {
        let req =
            ReasonCodeCatalog::required_evidence(ReasonCode::PayPal(PayPalReasonCode::ItemNotReceived));
        assert!(req.contains(&EvidenceRequirement::ProofOfDelivery));
    }
}
