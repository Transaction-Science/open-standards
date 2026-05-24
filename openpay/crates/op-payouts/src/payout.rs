//! Universal payout trait and supporting types.
//!
//! Every rail driver in this crate implements [`Payout`]. The
//! orchestrator routes a [`PayoutRequest`] by inspecting
//! `request.method` and dispatching to the appropriate driver.

use op_core::Money;
use serde::{Deserialize, Serialize};

use crate::error::Result;
use crate::funding::FundingSource;

/// Identifier of the rail / scheme a payout will be pushed over.
///
/// This enum is the routing key inside the payout orchestrator and is
/// kept deliberately flat: one variant per network surface we ship a
/// driver for. New rails extend the enum and add a new module.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PayoutMethod {
    /// Visa Direct OCT (Original Credit Transaction) to a debit-card PAN
    /// or token.
    VisaDirect,
    /// Mastercard Send (also OCT-based) to a debit-card PAN or token.
    MastercardSend,
    /// NACHA ACH credit (PPD = consumer, CCD = corporate).
    AchCredit {
        /// Standard Entry Class code. `"PPD"` or `"CCD"`.
        sec_code: String,
    },
    /// Fedwire funds transfer.
    Fedwire,
    /// SWIFT MT103 single customer credit transfer.
    SwiftMt103,
    /// ISO 20022 `pacs.008.001.08` credit transfer (SWIFT MX / CBPR+).
    Pacs008,
    /// SEPA Credit Transfer (D+1).
    SepaSct,
    /// SEPA Instant Credit Transfer (<10 s).
    SepaSctInst,
    /// UK Faster Payments Service.
    UkFasterPayments,
    /// FedNow instant credit.
    FedNow,
    /// TCH Real-Time Payments instant credit.
    Rtp,
    /// PayPal Payouts API (wallet → wallet or wallet → bank).
    PayPalPayouts,
    /// Wise Platform API recipient transfer.
    Wise,
    /// On-chain crypto payout (stablecoins or BTC).
    Crypto {
        /// Asset symbol, e.g. `"USDC"`, `"USDT"`, `"BTC"`.
        asset: String,
        /// Network, e.g. `"ethereum"`, `"polygon"`, `"base"`, `"solana"`,
        /// `"bitcoin"`, `"lightning"`.
        network: String,
    },
}

/// Beneficiary account format. Mirrors [`PayoutMethod`] but carries the
/// data, not the routing key.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum BeneficiaryAccount {
    /// 13–19 digit debit card PAN. Cards-only push.
    CardPan(String),
    /// Network token referring to a vaulted PAN. Preferred over raw PAN.
    CardToken(String),
    /// US bank account: ABA routing number + account number.
    UsBank {
        /// 9-digit ABA routing & transit number.
        aba: String,
        /// Account number, free-form (NACHA `DFI Account Number`, 1–17
        /// alphanumerics).
        account: String,
        /// `"CHECKING"` or `"SAVINGS"`.
        account_type: String,
    },
    /// IBAN (SEPA, UK domestic-IBAN, etc.).
    Iban(String),
    /// UK sort code (6 digits) + account number (8 digits).
    UkSortCode {
        /// 6-digit sort code (no dashes).
        sort_code: String,
        /// 8-digit account number.
        account: String,
    },
    /// SWIFT BIC + account. Used for MT103 / pacs.008 to non-SEPA banks.
    SwiftBic {
        /// 8 or 11 character BIC.
        bic: String,
        /// Beneficiary account or IBAN.
        account: String,
    },
    /// PayPal account email or payer-id.
    PayPalEmail(String),
    /// Wise recipient id (numeric, opaque).
    WiseRecipientId(u64),
    /// EVM address (`0x` + 40 hex chars).
    EvmAddress(String),
    /// Solana base58 address.
    SolanaAddress(String),
    /// Bitcoin address (bech32 / legacy).
    BitcoinAddress(String),
    /// BOLT-11 Lightning invoice.
    LightningInvoice(String),
}

/// Counterparty receiving the funds.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Beneficiary {
    /// Legal name (rail caps vary; drivers clip).
    pub name: String,
    /// Optional address, free-form. Required by some rails (Fedwire).
    pub address: Option<String>,
    /// Account.
    pub account: BeneficiaryAccount,
    /// Opaque KYC reference assigned by `op-screening`. The driver
    /// does not re-check; presence here is the operator's assertion
    /// that screening passed.
    pub kyc_ref: Option<String>,
}

/// A payout request — the "push money out" call.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PayoutRequest {
    /// Caller idempotency key — UUID v4 lowercase. Drivers forward to
    /// the rail where the rail supports idempotency tokens.
    pub idempotency_key: String,
    /// Rail / scheme to push over.
    pub method: PayoutMethod,
    /// Amount and currency.
    pub amount: Money,
    /// Beneficiary.
    pub beneficiary: Beneficiary,
    /// Funding source.
    pub funding: FundingSource,
    /// Free-form memo / remittance text. Drivers clip to the rail max.
    pub memo: Option<String>,
}

/// Normalized payout status. Aligned to `pacs.002` `TransactionStatus`
/// plus the operational states needed for batch / instant blends.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PayoutStatus {
    /// Driver built the message; nothing has been sent.
    PreparedOffline,
    /// Submitted, awaiting rail acknowledgement.
    Submitted,
    /// Accepted by rail, not yet settled (e.g. ACH next-day window).
    Accepted,
    /// Funds at beneficiary.
    Settled,
    /// Rejected with a final reason.
    Rejected,
    /// Returned by the beneficiary bank (ACH R-codes, wire returns).
    Returned,
}

/// Result of a payout attempt — the canonical record the orchestrator
/// stores.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct PayoutResult {
    /// Echoed idempotency key.
    pub idempotency_key: String,
    /// Driver-assigned payout id, UUID v7 (time-sortable).
    pub payout_id: String,
    /// Normalized status.
    pub status: PayoutStatus,
    /// Rail-specific raw status (e.g. Visa Direct `actionCode`, NACHA
    /// return code, ISO 20022 `TxSts`).
    pub raw_status: Option<String>,
    /// Optional reason code.
    pub reason_code: Option<String>,
    /// Optional reason text.
    pub reason_text: Option<String>,
    /// Rail-side transaction identifier.
    pub rail_txn_id: Option<String>,
    /// Settled / submitted amount. Equal to request amount unless the
    /// rail applies FX or fees.
    pub settled_amount: Option<Money>,
    /// Serialized rail message bytes (NACHA flat file fragment, ISO
    /// 20022 XML, JSON-RPC body, etc.). The operator transmits these.
    pub wire_payload: Option<Vec<u8>>,
}

/// Universal payout-driver trait.
pub trait Payout {
    /// Rail name for diagnostics. Lowercase, stable across versions.
    fn rail(&self) -> &'static str;

    /// Build (and, when wired in, submit) a payout.
    fn submit(&self, req: &PayoutRequest) -> Result<PayoutResult>;

    /// Query rail-side status of a previously submitted payout.
    ///
    /// Offline drivers return a `PreparedOffline` result for unknown
    /// ids; live drivers poll the rail's status endpoint.
    fn status(&self, payout_id: &str) -> Result<PayoutResult>;
}
