//! A neutral, flattened view of a `camt.053` / `camt.054` statement.
//!
//! ISO 20022 statement entries are deeply nested (`Stmt[] -> Ntry[] ->
//! NtryDtls[] -> TxDtls[] -> Refs.EndToEndId`). Downstream consumers
//! (reconciliation) shouldn't have to learn that tree or take a
//! direct dependency on the per-family `open-payments-iso20022-camt`
//! crate. This module walks the tree once and yields a flat
//! [`StatementEntry`] per booked line, keeping ISO 20022 knowledge
//! behind the `op-iso20022` facade.

use crate::message::Message;

/// One flattened statement line.
///
/// Dates are left as the raw ISO 20022 strings (`YYYY-MM-DD` or an
/// xsd:dateTime); converting them to a timestamp is the consumer's
/// concern (it owns the calendar/timezone policy).
#[derive(Clone, Debug, PartialEq)]
pub struct StatementEntry {
    /// `NtryRef`, falling back to `AcctSvcrRef`. The bank's own id
    /// for this booked line.
    pub reference: Option<String>,
    /// The end-to-end id echoed from the original payment, dug out of
    /// `NtryDtls -> TxDtls -> Refs.EndToEndId`. The strong join key
    /// for reconciliation; `None` for lines the bank originated
    /// (sweeps, fees) that carry no customer reference.
    pub end_to_end_id: Option<String>,
    /// Amount as it appears on the wire (always non-negative; the
    /// sign is [`Self::is_credit`]).
    pub amount_value: f64,
    /// ISO 4217 alphabetic currency code.
    pub currency: String,
    /// `true` = `CRDT` (money into the account holder), `false` =
    /// `DBIT`.
    pub is_credit: bool,
    /// `true` when the bank flagged this entry as a reversal
    /// (`RvslInd`).
    pub reversal: bool,
    /// Value date (`ValDt`) — `Dt` or `DtTm`, raw string.
    pub value_date: Option<String>,
    /// Booking date (`BookgDt`) — `Dt` or `DtTm`, raw string.
    pub booking_date: Option<String>,
    /// `AddtlNtryInf` free text, if present.
    pub additional_info: Option<String>,
}

impl Message {
    /// Flatten a [`Message::Camt053`] into one [`StatementEntry`] per
    /// booked `Ntry`. Returns an empty vec for any other message kind.
    #[must_use]
    pub fn camt053_entries(&self) -> Vec<StatementEntry> {
        let Some(stmt_doc) = self.as_camt053() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for stmt in &stmt_doc.stmt {
            let Some(entries) = &stmt.ntry else { continue };
            for ntry in entries {
                out.push(flatten_entry(ntry));
            }
        }
        out
    }

    /// Flatten a [`Message::Camt054`] into one [`StatementEntry`] per
    /// `Ntry`. Same `Ntry` shape as `camt.053`; we reuse the same
    /// flattening helper, so downstream consumers don't need a
    /// separate code path. Returns an empty vec for any other
    /// message kind.
    #[must_use]
    pub fn camt054_entries(&self) -> Vec<StatementEntry> {
        let Some(ntfctn_doc) = self.as_camt054() else {
            return Vec::new();
        };
        let mut out = Vec::new();
        for ntfctn in &ntfctn_doc.ntfctn {
            let Some(entries) = &ntfctn.ntry else {
                continue;
            };
            for ntry in entries {
                out.push(flatten_entry(ntry));
            }
        }
        out
    }
}

fn flatten_entry(ntry: &iso20022_common::common::ReportEntry14) -> StatementEntry {
    use iso20022_common::common::CreditDebitCode;

    // End-to-end id lives one level down, in the first transaction
    // detail that carries a Refs block.
    let end_to_end_id = ntry.ntry_dtls.as_ref().and_then(|dtls| {
        dtls.iter().find_map(|d| {
            d.tx_dtls.as_ref().and_then(|txs| {
                txs.iter()
                    .find_map(|t| t.refs.as_ref().and_then(|r| r.end_to_end_id.clone()))
            })
        })
    });

    let reference = ntry.ntry_ref.clone().or_else(|| ntry.acct_svcr_ref.clone());

    let is_credit = matches!(ntry.cdt_dbt_ind, CreditDebitCode::CodeCRDT);

    StatementEntry {
        reference,
        end_to_end_id,
        amount_value: ntry.amt.value.abs(),
        currency: ntry.amt.ccy.clone(),
        is_credit,
        reversal: ntry.rvsl_ind.unwrap_or(false),
        value_date: ntry
            .val_dt
            .as_ref()
            .and_then(|d| d.dt.clone().or_else(|| d.dt_tm.clone())),
        booking_date: ntry
            .bookg_dt
            .as_ref()
            .and_then(|d| d.dt.clone().or_else(|| d.dt_tm.clone())),
        additional_info: ntry.addtl_ntry_inf.clone(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use iso20022_common::common::{
        AccountStatement13, ActiveOrHistoricCurrencyAndAmount, CreditDebitCode,
        DateAndDateTime2Choice, EntryDetails13, EntryTransaction14, ReportEntry14,
        TransactionReferences6,
    };
    use open_payments_iso20022_camt::camt_053_001_12::BankToCustomerStatementV12;

    // Build the smallest tree that exercises every field flatten_entry
    // reads, via the upstream `derive_default` impls. We deliberately
    // construct the struct rather than parse XML: the flattening
    // traversal is the logic this crate owns and must test; full
    // camt.053 serde-round-trip is a separate conformance concern.
    fn one_entry_statement() -> BankToCustomerStatementV12 {
        let refs = TransactionReferences6 {
            end_to_end_id: Some("ORD-77".to_owned()),
            ..Default::default()
        };
        let txd = EntryTransaction14 {
            refs: Some(refs),
            ..Default::default()
        };
        let dtls = EntryDetails13 {
            tx_dtls: Some(vec![txd]),
            ..Default::default()
        };
        let ntry = ReportEntry14 {
            ntry_ref: Some("NTRY-1".to_owned()),
            amt: ActiveOrHistoricCurrencyAndAmount {
                ccy: "USD".to_owned(),
                value: 52.50,
            },
            cdt_dbt_ind: CreditDebitCode::CodeCRDT,
            rvsl_ind: Some(false),
            val_dt: Some(DateAndDateTime2Choice {
                dt: Some("2026-05-18".to_owned()),
                dt_tm: None,
            }),
            ntry_dtls: Some(vec![dtls]),
            addtl_ntry_inf: Some("coffee".to_owned()),
            ..Default::default()
        };
        let stmt = AccountStatement13 {
            ntry: Some(vec![ntry]),
            ..Default::default()
        };
        BankToCustomerStatementV12 {
            stmt: vec![stmt],
            ..Default::default()
        }
    }

    #[test]
    fn flattens_nested_entry() {
        let msg = Message::Camt053(Box::new(one_entry_statement()));
        let entries = msg.camt053_entries();
        assert_eq!(entries.len(), 1);
        let e = &entries[0];
        assert_eq!(e.reference.as_deref(), Some("NTRY-1"));
        assert_eq!(e.end_to_end_id.as_deref(), Some("ORD-77"));
        assert!((e.amount_value - 52.50).abs() < 1e-9);
        assert_eq!(e.currency, "USD");
        assert!(e.is_credit);
        assert!(!e.reversal);
        assert_eq!(e.value_date.as_deref(), Some("2026-05-18"));
        assert_eq!(e.additional_info.as_deref(), Some("coffee"));
    }

    #[test]
    fn non_camt053_message_yields_no_entries() {
        // Any other kind flattens to nothing rather than erroring.
        let admi = Message::Admi002(Box::default());
        assert!(admi.camt053_entries().is_empty());
    }

    #[test]
    fn camt054_flattens_through_the_same_helper() {
        use iso20022_common::common::AccountNotification22;
        use open_payments_iso20022_camt::camt_054_001_12::BankToCustomerDebitCreditNotificationV12;

        // Re-use the camt.053 fixture's single entry, wrap it as a
        // notification — the Ntry shape is identical, so the flatten
        // result must match.
        let stmt = one_entry_statement();
        let ntry = stmt.stmt.into_iter().next().unwrap().ntry.unwrap();
        let ntfctn = AccountNotification22 {
            ntry: Some(ntry),
            ..Default::default()
        };
        let doc = BankToCustomerDebitCreditNotificationV12 {
            ntfctn: vec![ntfctn],
            ..Default::default()
        };
        let msg = Message::Camt054(Box::new(doc));
        let entries = msg.camt054_entries();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].end_to_end_id.as_deref(), Some("ORD-77"));
        assert_eq!(entries[0].currency, "USD");
        assert!(entries[0].is_credit);
    }
}
