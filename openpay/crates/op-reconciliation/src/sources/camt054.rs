//! [`Camt054Source`] — ISO 20022 intra-day debit/credit notification.
//!
//! Same `Ntry` shape as `camt.053`; we reuse the flattening pipeline
//! in `op-iso20022::statement` and map each entry to a
//! [`StatementLine`] identically. Operators who poll the bank for
//! intra-day notifications instead of end-of-day statements can
//! reconcile against the freshest data.

use op_iso20022::Message;

use crate::error::{Error, Result};
use crate::source::ReconciliationSource;
use crate::sources::{currency_from_code, direction, to_money, unix_from_camt_date};
use crate::statement::StatementLine;

/// Reconciliation source backed by a parsed `camt.054`
/// bank-to-customer debit/credit notification.
pub struct Camt054Source {
    message: Message,
}

impl Camt054Source {
    /// Build from an already-parsed [`Message`]. Errors unless it is
    /// a [`Message::Camt054`].
    ///
    /// # Errors
    /// [`Error::Iso20022`] if `message` is the wrong kind.
    pub fn from_message(message: Message) -> Result<Self> {
        if message.as_camt054().is_none() {
            return Err(Error::Iso20022(format!(
                "expected camt.054, got {:?}",
                message.kind()
            )));
        }
        Ok(Self { message })
    }

    /// Build by parsing `camt.054` XML.
    ///
    /// # Errors
    /// [`Error::Iso20022`] if the XML is not a well-formed
    /// `BankToCustomerDebitCreditNotificationV12`.
    pub fn from_xml(xml: &str) -> Result<Self> {
        Self::from_message(Message::parse_camt054(xml)?)
    }
}

impl ReconciliationSource for Camt054Source {
    fn iter_lines(&self) -> Box<dyn Iterator<Item = Result<StatementLine>> + '_> {
        let entries = self.message.camt054_entries();
        Box::new(entries.into_iter().map(|e| {
            let currency = currency_from_code(&e.currency)?;
            let amount = to_money(e.amount_value, currency);
            let when = e
                .value_date
                .as_deref()
                .or(e.booking_date.as_deref())
                .ok_or_else(|| {
                    Error::MalformedLine(format!(
                        "entry {:?} has no value or booking date",
                        e.reference
                    ))
                })?;
            let posted = unix_from_camt_date(when)?;
            let source_id = e
                .reference
                .clone()
                .unwrap_or_else(|| format!("camt054:{when}:{}", e.amount_value));

            let mut line = StatementLine::new(source_id, amount, direction(e.is_credit), posted);
            if let Some(eid) = e.end_to_end_id.clone() {
                line = line.with_external_id(eid);
            }
            if e.reversal {
                line = line.with_metadata("reversal", "true");
            }
            if let Some(info) = e.additional_info.clone() {
                line = line.with_metadata("addtl_ntry_inf", info);
            }
            Ok(line)
        }))
    }
}
