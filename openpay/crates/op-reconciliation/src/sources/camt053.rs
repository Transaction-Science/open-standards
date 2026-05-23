//! [`Camt053Source`] — ISO 20022 end-of-day bank statement.

use op_iso20022::Message;

use crate::error::{Error, Result};
use crate::source::ReconciliationSource;
use crate::sources::{currency_from_code, direction, to_money, unix_from_camt_date};
use crate::statement::StatementLine;

/// Reconciliation source backed by a parsed `camt.053`
/// bank-to-customer statement.
///
/// The statement is parsed once (eagerly, via the `op-iso20022`
/// facade so this crate never touches the raw ISO 20022 crates) and
/// flattened to neutral entries; `iter_lines` then maps each entry to
/// a [`StatementLine`], deferring date/currency/amount conversion to
/// iteration so a malformed field surfaces as a per-line error rather
/// than failing the whole construction.
pub struct Camt053Source {
    message: Message,
}

impl Camt053Source {
    /// Build from an already-parsed [`Message`]. Errors unless it is
    /// a [`Message::Camt053`].
    ///
    /// # Errors
    /// [`Error::Iso20022`] if `message` is the wrong kind.
    pub fn from_message(message: Message) -> Result<Self> {
        if message.as_camt053().is_none() {
            return Err(Error::Iso20022(format!(
                "expected camt.053, got {:?}",
                message.kind()
            )));
        }
        Ok(Self { message })
    }

    /// Build by parsing `camt.053` XML.
    ///
    /// # Errors
    /// [`Error::Iso20022`] if the XML is not a well-formed
    /// `BankToCustomerStatementV12`.
    pub fn from_xml(xml: &str) -> Result<Self> {
        Self::from_message(Message::parse_camt053(xml)?)
    }
}

impl ReconciliationSource for Camt053Source {
    fn iter_lines(&self) -> Box<dyn Iterator<Item = Result<StatementLine>> + '_> {
        let entries = self.message.camt053_entries();
        Box::new(entries.into_iter().map(|e| {
            let currency = currency_from_code(&e.currency)?;
            let amount = to_money(e.amount_value, currency);

            // Prefer the value date (when funds are available); fall
            // back to booking date. A line with neither is malformed —
            // a statement entry must say when it happened.
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
                .unwrap_or_else(|| format!("camt053:{when}:{}", e.amount_value));

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
