//! The normalized statement line.
//!
//! Every source — CAMT.053, CAMT.054, settlement webhook, an
//! operator's bespoke CSV — is decoded into this one shape so the
//! matcher never has to know where a line came from.

use op_core::Money;

/// Direction of money movement **from the account holder's point of
/// view**, as a bank statement reports it.
///
/// This is intentionally *not* [`op_ledger::Direction`]. Ledger
/// direction is a double-entry bookkeeping primitive (which side of
/// which account); statement direction is the bank's plain-English
/// "money came in" / "money went out". Conflating them is a classic
/// reconciliation bug, so we keep them distinct types.
#[derive(Copy, Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum LineDirection {
    /// Money credited **to** the account holder (an inbound payment,
    /// a payout received). ISO 20022 `CdtDbtInd = CRDT`.
    Credit,
    /// Money debited **from** the account holder (an outbound
    /// payment, a fee, a chargeback). ISO 20022 `CdtDbtInd = DBIT`.
    Debit,
}

/// One normalized line from a statement source.
///
/// A line is the atomic unit of reconciliation: it asserts "this much
/// money moved, in this direction, around this time, and here is the
/// reference the counterparty used."
#[derive(Clone, Debug, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub struct StatementLine {
    /// The id the bank/PSP assigned to this line (CAMT `NtryRef` /
    /// `AcctSvcrRef`, or a webhook event id). Unique within a source;
    /// used to de-duplicate on re-ingest.
    pub source_id: String,

    /// The end-to-end / order reference the counterparty echoed back.
    /// This is the strong join key against a ledger transaction's
    /// `external_id`. `None` when the source didn't carry one (some
    /// fee lines, some sweeps).
    pub external_id: Option<String>,

    /// Signed-magnitude amount. The sign is carried separately in
    /// [`Self::direction`]; `amount.minor_units` is always the
    /// non-negative magnitude so currency-exponent math stays simple.
    pub amount: Money,

    /// Which way the money moved, from the account holder's view.
    pub direction: LineDirection,

    /// When the bank/PSP says the line posted (value date / booking
    /// timestamp), unix epoch seconds. Caller-supplied via the source;
    /// no clock is read here.
    pub posted_at_unix_secs: u64,

    /// Free-form passthrough the source preserved (remittance text,
    /// PSP fee codes, debtor name). Never interpreted by the matcher;
    /// surfaced in the report so an operator can eyeball a
    /// discrepancy.
    pub metadata: Vec<(String, String)>,
}

impl StatementLine {
    /// Construct a line. `amount` is normalized to its magnitude
    /// (sign lives in `direction`).
    #[must_use]
    pub fn new(
        source_id: impl Into<String>,
        amount: Money,
        direction: LineDirection,
        posted_at_unix_secs: u64,
    ) -> Self {
        let amount = Money {
            minor_units: amount.minor_units.abs(),
            currency: amount.currency,
        };
        Self {
            source_id: source_id.into(),
            external_id: None,
            amount,
            direction,
            posted_at_unix_secs,
            metadata: Vec::new(),
        }
    }

    /// Builder: attach the end-to-end / order reference.
    #[must_use]
    pub fn with_external_id(mut self, id: impl Into<String>) -> Self {
        self.external_id = Some(id.into());
        self
    }

    /// Builder: attach a metadata key/value pair.
    #[must_use]
    pub fn with_metadata(mut self, k: impl Into<String>, v: impl Into<String>) -> Self {
        self.metadata.push((k.into(), v.into()));
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::Currency;

    #[test]
    fn new_normalizes_amount_to_magnitude() {
        let line = StatementLine::new(
            "ntry-1",
            Money {
                minor_units: -4200,
                currency: Currency::USD,
            },
            LineDirection::Debit,
            1_700_000_000,
        );
        // Sign is carried by `direction`, not the amount.
        assert_eq!(line.amount.minor_units, 4200);
        assert_eq!(line.direction, LineDirection::Debit);
        assert!(line.external_id.is_none());
    }

    #[test]
    fn builders_chain() {
        let line = StatementLine::new(
            "ntry-2",
            Money::from_minor(999, Currency::EUR),
            LineDirection::Credit,
            42,
        )
        .with_external_id("ORD-9")
        .with_metadata("remittance", "invoice 9");
        assert_eq!(line.external_id.as_deref(), Some("ORD-9"));
        assert_eq!(
            line.metadata,
            vec![("remittance".to_owned(), "invoice 9".to_owned())]
        );
    }
}
