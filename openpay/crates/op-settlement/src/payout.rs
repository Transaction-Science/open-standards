//! Payout rail taxonomy and per-entry pacs.008 construction helpers.
//!
//! A batch settles over one of a small handful of rails. The choice
//! drives which payout-file generator runs:
//!
//! - **NACHA** for US ACH — see [`crate::nacha::nacha_file`].
//! - **`pacs.008`** for SEPA / RTP / `FedNow` — operators construct one
//!   [`op_iso20022::CreditTransferBuilder`] per batch entry and feed
//!   it the rail profile + creditor party. The
//!   [`Pacs008EntryContext`] type is a small convenience struct that
//!   bundles the per-entry data points the builder needs.
//! - **Wire / internal book transfer** — operator-driven; we tag
//!   the batch.

use op_core::Money;
use op_iso20022::bah::PartyIdentification;
use op_ledger::TransactionId;
use serde::{Deserialize, Serialize};

/// Which rail will move the batch's funds to the beneficiary.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PayoutRail {
    /// US ACH batch via NACHA file. Same-day or next-day.
    AchNacha,
    /// SEPA Credit Transfer (`pacs.008.001.10`). Euro area.
    SepaCt,
    /// `FedNow` instant credit transfer (`pacs.008`).
    FedNow,
    /// The Clearing House RTP (`pacs.008`).
    Rtp,
    /// Wire transfer (Fedwire / CHIPS / SWIFT). Operator drives the
    /// message themselves; we just tag the batch.
    Wire,
    /// Operator's internal book transfer (own ledger). No external
    /// rail involved — used by wallets / closed-loop systems.
    InternalBookTransfer,
}

impl PayoutRail {
    /// Whether the rail uses the NACHA file format.
    #[must_use]
    pub const fn is_nacha(self) -> bool {
        matches!(self, Self::AchNacha)
    }

    /// Whether the rail uses ISO 20022 `pacs.008`.
    #[must_use]
    pub const fn is_iso20022_pacs008(self) -> bool {
        matches!(self, Self::SepaCt | Self::FedNow | Self::Rtp)
    }
}

/// Operator-supplied per-entry context for a `pacs.008` credit
/// transfer. The settlement engine doesn't know banking
/// coordinates — operators map from the batch's `(tx_id, amount)`
/// to this struct via their own merchant directory.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Pacs008EntryContext {
    /// The ledger transaction this entry settles.
    pub tx_id: TransactionId,
    /// Amount of this individual credit transfer.
    pub amount: Money,
    /// Debtor agent (the operator's settlement bank).
    pub debtor_agent: PartyIdentification,
    /// Creditor agent (the beneficiary's bank).
    pub creditor_agent: PartyIdentification,
    /// End-to-end id (operator's reference, ≤35 chars).
    pub end_to_end_id: String,
    /// UETR — mandatory on `FedNow`, optional elsewhere.
    pub uetr: Option<String>,
    /// Remittance info (free-form description on the wire).
    pub remittance_info: Option<String>,
}
