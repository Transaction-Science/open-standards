//! Card-network code enums.
//!
//! One file, by directive — the five major dispute-bearing networks
//! and their reason-code surfaces live side by side so a `match`
//! across them stays auditable.
//!
//! These enums model the *codes the operator sees on the wire*, not
//! the human-readable categories. The categorization (fraud /
//! consumer / processing / authorization) is attached in
//! [`crate::reason_codes::ReasonCode`].

use serde::{Deserialize, Serialize};

/// Top-level card / wallet network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Network {
    /// Visa, processed through **VCR** (Visa Claims Resolution).
    Visa,
    /// Mastercard, processed through **Mastercom**.
    Mastercard,
    /// American Express, with **SafeKey** 3-D Secure and the
    /// network's dispute-resolution flow.
    Amex,
    /// Discover, processed through **DRR** (Dispute Resolution
    /// Re-engineering).
    Discover,
    /// PayPal — wallet, not a card network, but disputes look
    /// structurally similar enough to live in the same enum.
    PayPal,
}

impl Network {
    /// Stable wire string used in OpenPay messages.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Visa => "visa",
            Self::Mastercard => "mastercard",
            Self::Amex => "amex",
            Self::Discover => "discover",
            Self::PayPal => "paypal",
        }
    }
}

// -- Visa VCR -----------------------------------------------------
//
// Visa published the VCR reason-code refactor (Apr 2018) collapsing
// the legacy two-digit codes into four chapters:
//   10.x  Fraud
//   11.x  Authorization
//   12.x  Processing errors
//   13.x  Consumer disputes
//
// We model the subset the merchant evidence flow actually cares
// about. The variants are exhaustive enough to drive the
// representment-evidence requirements in [`crate::reason_codes`].

/// Visa VCR reason code (a subset of the published list; the ones
/// that drive representment evidence flows).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum VisaReasonCode {
    /// 10.1 — EMV liability shift counterfeit fraud.
    F1010,
    /// 10.2 — EMV liability shift non-counterfeit fraud.
    F1020,
    /// 10.3 — Other fraud, card-present environment.
    F1030,
    /// 10.4 — Other fraud, card-absent environment. The most-cited
    /// card-not-present chargeback reason; the one CE3.0 targets.
    F1040,
    /// 10.5 — Visa fraud monitoring program.
    F1050,
    /// 11.1 — Card recovery bulletin.
    A1110,
    /// 11.2 — Declined authorization.
    A1120,
    /// 11.3 — No authorization.
    A1130,
    /// 12.1 — Late presentment.
    P1210,
    /// 12.2 — Incorrect transaction code.
    P1220,
    /// 12.3 — Incorrect currency.
    P1230,
    /// 12.4 — Incorrect account number.
    P1240,
    /// 12.5 — Incorrect amount.
    P1250,
    /// 12.6 — Duplicate processing / paid by other means.
    P1260,
    /// 12.7 — Invalid data.
    P1270,
    /// 13.1 — Merchandise / services not received.
    C1310,
    /// 13.2 — Cancelled recurring.
    C1320,
    /// 13.3 — Not as described or defective merchandise.
    C1330,
    /// 13.4 — Counterfeit merchandise.
    C1340,
    /// 13.5 — Misrepresentation.
    C1350,
    /// 13.6 — Credit not processed.
    C1360,
    /// 13.7 — Cancelled merchandise / services.
    C1370,
    /// 13.8 — Original credit transaction not accepted.
    C1380,
    /// 13.9 — Non-receipt of cash or load transaction value.
    C1390,
}

impl VisaReasonCode {
    /// Canonical Visa code string (e.g. `"10.4"`).
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::F1010 => "10.1",
            Self::F1020 => "10.2",
            Self::F1030 => "10.3",
            Self::F1040 => "10.4",
            Self::F1050 => "10.5",
            Self::A1110 => "11.1",
            Self::A1120 => "11.2",
            Self::A1130 => "11.3",
            Self::P1210 => "12.1",
            Self::P1220 => "12.2",
            Self::P1230 => "12.3",
            Self::P1240 => "12.4",
            Self::P1250 => "12.5",
            Self::P1260 => "12.6",
            Self::P1270 => "12.7",
            Self::C1310 => "13.1",
            Self::C1320 => "13.2",
            Self::C1330 => "13.3",
            Self::C1340 => "13.4",
            Self::C1350 => "13.5",
            Self::C1360 => "13.6",
            Self::C1370 => "13.7",
            Self::C1380 => "13.8",
            Self::C1390 => "13.9",
        }
    }

    /// Parse a canonical Visa code string into the enum.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "10.1" => Self::F1010,
            "10.2" => Self::F1020,
            "10.3" => Self::F1030,
            "10.4" => Self::F1040,
            "10.5" => Self::F1050,
            "11.1" => Self::A1110,
            "11.2" => Self::A1120,
            "11.3" => Self::A1130,
            "12.1" => Self::P1210,
            "12.2" => Self::P1220,
            "12.3" => Self::P1230,
            "12.4" => Self::P1240,
            "12.5" => Self::P1250,
            "12.6" => Self::P1260,
            "12.7" => Self::P1270,
            "13.1" => Self::C1310,
            "13.2" => Self::C1320,
            "13.3" => Self::C1330,
            "13.4" => Self::C1340,
            "13.5" => Self::C1350,
            "13.6" => Self::C1360,
            "13.7" => Self::C1370,
            "13.8" => Self::C1380,
            "13.9" => Self::C1390,
            _ => return None,
        })
    }
}

// -- Mastercard Mastercom -----------------------------------------
//
// Mastercom unified the reason codes onto MIP messages 1442 (first
// chargeback) and 1240 (first presentment). The semantic reason
// rides as a four-digit code on the message; legacy codes 4837/4853
// etc. are still in active use.

/// Mastercard Mastercom reason code (the canonical four-digit form
/// carried on MIP message 1442).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MastercardReasonCode {
    /// 4837 — No cardholder authorization (fraud, card-absent).
    Mc4837,
    /// 4840 — Fraudulent processing of transactions.
    Mc4840,
    /// 4849 — Questionable merchant activity.
    Mc4849,
    /// 4853 — Cardholder dispute (goods / services not received,
    /// not as described, defective).
    Mc4853,
    /// 4855 — Goods or services not provided.
    Mc4855,
    /// 4859 — Services not rendered / addendum / no-show /
    /// ATM dispute.
    Mc4859,
    /// 4860 — Credit not processed.
    Mc4860,
    /// 4863 — Cardholder does not recognize — potential fraud.
    Mc4863,
    /// 4870 — Chip liability shift.
    Mc4870,
    /// 4871 — Chip / PIN liability shift — lost / stolen / never
    /// received.
    Mc4871,
}

impl MastercardReasonCode {
    /// Canonical Mastercard reason-code string.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Mc4837 => "4837",
            Self::Mc4840 => "4840",
            Self::Mc4849 => "4849",
            Self::Mc4853 => "4853",
            Self::Mc4855 => "4855",
            Self::Mc4859 => "4859",
            Self::Mc4860 => "4860",
            Self::Mc4863 => "4863",
            Self::Mc4870 => "4870",
            Self::Mc4871 => "4871",
        }
    }

    /// Parse a canonical Mastercard reason-code string.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "4837" => Self::Mc4837,
            "4840" => Self::Mc4840,
            "4849" => Self::Mc4849,
            "4853" => Self::Mc4853,
            "4855" => Self::Mc4855,
            "4859" => Self::Mc4859,
            "4860" => Self::Mc4860,
            "4863" => Self::Mc4863,
            "4870" => Self::Mc4870,
            "4871" => Self::Mc4871,
            _ => return None,
        })
    }
}

/// Mastercom MIP message identifier.
///
/// The wire layer carries the reason code *inside* one of two MIP
/// envelopes; operators reconciling chargeback feeds will see these
/// as the outer message type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MastercomMessage {
    /// 1240 — First presentment (the original financial leg).
    FirstPresentment1240,
    /// 1442 — First chargeback.
    FirstChargeback1442,
    /// 1240/205 — Second presentment (representment).
    SecondPresentment,
    /// 1442/282 — Arbitration chargeback.
    ArbitrationChargeback,
}

// -- Amex SafeKey + DRR -------------------------------------------
//
// American Express runs both issuer and acquirer in-house, so the
// dispute taxonomy is shorter and SafeKey (3DS2) authentication
// directly drives liability.

/// American Express reason code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum AmexReasonCode {
    /// F24 — No cardmember authorization.
    F24,
    /// F29 — Card-not-present fraud.
    F29,
    /// F30 — EMV counterfeit fraud.
    F30,
    /// F31 — EMV lost / stolen / non-received fraud.
    F31,
    /// C02 — Credit (refund) not processed.
    C02,
    /// C04 — Goods / services returned or refused.
    C04,
    /// C05 — Goods / services cancelled.
    C05,
    /// C08 — Goods / services not received or partially received.
    C08,
    /// C14 — Paid by other means.
    C14,
    /// C18 — "No show" or CARDeposit cancelled.
    C18,
    /// C28 — Cancelled recurring billing.
    C28,
    /// C31 — Goods / services not as described.
    C31,
    /// C32 — Goods / services damaged or defective.
    C32,
    /// A01 — Charge amount exceeds authorization amount.
    A01,
    /// A02 — No valid authorization.
    A02,
    /// A08 — Authorization approval expired.
    A08,
    /// P01 — Unassigned card number.
    P01,
    /// P03 — Credit processed as charge.
    P03,
    /// P04 — Charge processed as credit.
    P04,
    /// P05 — Incorrect charge amount.
    P05,
    /// P07 — Late submission.
    P07,
}

impl AmexReasonCode {
    /// Canonical Amex code string.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::F24 => "F24",
            Self::F29 => "F29",
            Self::F30 => "F30",
            Self::F31 => "F31",
            Self::C02 => "C02",
            Self::C04 => "C04",
            Self::C05 => "C05",
            Self::C08 => "C08",
            Self::C14 => "C14",
            Self::C18 => "C18",
            Self::C28 => "C28",
            Self::C31 => "C31",
            Self::C32 => "C32",
            Self::A01 => "A01",
            Self::A02 => "A02",
            Self::A08 => "A08",
            Self::P01 => "P01",
            Self::P03 => "P03",
            Self::P04 => "P04",
            Self::P05 => "P05",
            Self::P07 => "P07",
        }
    }

    /// Parse a canonical Amex code string.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "F24" => Self::F24,
            "F29" => Self::F29,
            "F30" => Self::F30,
            "F31" => Self::F31,
            "C02" => Self::C02,
            "C04" => Self::C04,
            "C05" => Self::C05,
            "C08" => Self::C08,
            "C14" => Self::C14,
            "C18" => Self::C18,
            "C28" => Self::C28,
            "C31" => Self::C31,
            "C32" => Self::C32,
            "A01" => Self::A01,
            "A02" => Self::A02,
            "A08" => Self::A08,
            "P01" => Self::P01,
            "P03" => Self::P03,
            "P04" => Self::P04,
            "P05" => Self::P05,
            "P07" => Self::P07,
            _ => return None,
        })
    }
}

// -- Discover DRR -------------------------------------------------

/// Discover (DRR) reason code.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DiscoverReasonCode {
    /// UA01 — Fraud, card-present.
    UA01,
    /// UA02 — Fraud, card-not-present.
    UA02,
    /// UA05 — Fraud, chip counterfeit.
    UA05,
    /// UA06 — Fraud, chip lost / stolen / non-received.
    UA06,
    /// AT — Authorization (no auth / expired / declined).
    AT,
    /// RG — Non-receipt of goods or services.
    RG,
    /// RM — Cardholder dispute — quality.
    RM,
    /// RN1 — Credit not processed.
    RN1,
    /// RN2 — Cancelled merchandise / services.
    RN2,
    /// LP — Late presentment.
    LP,
    /// IC — Invalid card number.
    IC,
    /// DA — Declined authorization.
    DA,
}

impl DiscoverReasonCode {
    /// Canonical Discover DRR code string.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::UA01 => "UA01",
            Self::UA02 => "UA02",
            Self::UA05 => "UA05",
            Self::UA06 => "UA06",
            Self::AT => "AT",
            Self::RG => "RG",
            Self::RM => "RM",
            Self::RN1 => "RN1",
            Self::RN2 => "RN2",
            Self::LP => "LP",
            Self::IC => "IC",
            Self::DA => "DA",
        }
    }

    /// Parse a canonical Discover code string.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "UA01" => Self::UA01,
            "UA02" => Self::UA02,
            "UA05" => Self::UA05,
            "UA06" => Self::UA06,
            "AT" => Self::AT,
            "RG" => Self::RG,
            "RM" => Self::RM,
            "RN1" => Self::RN1,
            "RN2" => Self::RN2,
            "LP" => Self::LP,
            "IC" => Self::IC,
            "DA" => Self::DA,
            _ => return None,
        })
    }
}

// -- PayPal --------------------------------------------------------

/// PayPal dispute reason.
///
/// PayPal exposes a much smaller, plain-English taxonomy. We map it
/// to constants the OpenPay evidence-requirement table can key on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PayPalReasonCode {
    /// Unauthorized — buyer claims the transaction was not made by
    /// them.
    Unauthorized,
    /// Item not received (INR).
    ItemNotReceived,
    /// Significantly not as described (SNAD).
    SignificantlyNotAsDescribed,
    /// Duplicate transaction.
    Duplicate,
    /// Cancelled recurring transaction.
    CancelledRecurring,
    /// Credit not processed.
    CreditNotProcessed,
}

impl PayPalReasonCode {
    /// Canonical PayPal code string.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::Unauthorized => "UNAUTHORIZED",
            Self::ItemNotReceived => "ITEM_NOT_RECEIVED",
            Self::SignificantlyNotAsDescribed => "SIGNIFICANTLY_NOT_AS_DESCRIBED",
            Self::Duplicate => "DUPLICATE_TRANSACTION",
            Self::CancelledRecurring => "CANCELLED_RECURRING_BILLING",
            Self::CreditNotProcessed => "CREDIT_NOT_PROCESSED",
        }
    }

    /// Parse a canonical PayPal code string.
    #[must_use]
    pub fn parse(s: &str) -> Option<Self> {
        Some(match s {
            "UNAUTHORIZED" => Self::Unauthorized,
            "ITEM_NOT_RECEIVED" => Self::ItemNotReceived,
            "SIGNIFICANTLY_NOT_AS_DESCRIBED" => Self::SignificantlyNotAsDescribed,
            "DUPLICATE_TRANSACTION" => Self::Duplicate,
            "CANCELLED_RECURRING_BILLING" => Self::CancelledRecurring,
            "CREDIT_NOT_PROCESSED" => Self::CreditNotProcessed,
            _ => return None,
        })
    }
}
