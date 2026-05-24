//! BAI2 (Bank Administration Institute, version 2) writer.
//!
//! BAI2 is a comma-delimited record format dating to 1987, still used
//! by US banks (Wells Fargo, BofA, Chase, etc.) for cash management
//! reporting. Each record begins with a numeric type code; record
//! boundaries are `\n` (some banks use `\r\n` — operators wrap our
//! output if their pipeline needs the carriage return).
//!
//! Record types we emit:
//! - `01` File Header
//! - `02` Group Header
//! - `03` Account Identifier
//! - `16` Transaction Detail
//! - `49` Account Trailer
//! - `98` Group Trailer
//! - `99` File Trailer
//!
//! Fund-type codes follow the BAI standard for the most common cash
//! statement summary codes.

use crate::error::Result;
use crate::statement::{Statement, StatementLineKind};

/// BAI2 transmission writer.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Bai2Writer {
    /// Sender (originator) identifier — typically a bank routing or
    /// PSP-assigned id. Echoed on the `01` File Header. Defaults to
    /// "OPENPAY" if empty.
    pub sender_id: &'static str,
    /// Receiver id (the merchant or downstream pipeline). Echoed on
    /// the `01` File Header. Defaults to "MERCHANT" if empty.
    pub receiver_id: &'static str,
}

impl Bai2Writer {
    /// Serialize a [`Statement`] as a BAI2 transmission. The output is
    /// LF-terminated; CRLF-wanting consumers post-process.
    ///
    /// # Errors
    /// Reserved for future overflow / encoding errors.
    pub fn build(&self, statement: &Statement) -> Result<String> {
        let sender = if self.sender_id.is_empty() { "OPENPAY" } else { self.sender_id };
        let receiver = if self.receiver_id.is_empty() {
            "MERCHANT"
        } else {
            self.receiver_id
        };
        let mut out = String::new();
        let file_id = statement.id.replace(',', "_");
        let yymmdd = unix_to_yymmdd(statement.period.end_unix_secs);
        let hhmm = unix_to_hhmm(statement.period.end_unix_secs);

        // 01 = File Header.
        out.push_str(&format!(
            "01,{sender},{receiver},{yymmdd},{hhmm},1,,,2/\n"
        ));
        // 02 = Group Header. Status `1` = update.
        out.push_str(&format!(
            "02,{receiver},{sender},1,{yymmdd},{hhmm},{cur},/\n",
            cur = statement.primary_currency
        ));
        // 03 = Account Identifier; 010/015 = opening/closing ledger
        // balances per BAI2 type-code book.
        let primary = statement.primary_aggregate();
        out.push_str(&format!(
            "03,{merchant},{cur},010,{open},,,015,{close},,/\n",
            merchant = statement.merchant_id,
            cur = statement.primary_currency,
            open = primary.opening.minor_units,
            close = primary.ending.minor_units,
        ));
        // 16 = Transaction Detail per line.
        let mut tx_count: u32 = 0;
        let mut credits: i64 = 0;
        let mut debits: i64 = 0;
        for line in &statement.lines {
            let (type_code, is_credit) = bai_type_code(line.kind);
            let amt_abs = line.amount.minor_units.abs();
            if is_credit {
                credits = credits.saturating_add(amt_abs);
            } else {
                debits = debits.saturating_add(amt_abs);
            }
            tx_count = tx_count.saturating_add(1);
            let ext = line.external_id.as_deref().unwrap_or("");
            out.push_str(&format!(
                "16,{type_code},{amt_abs},Z,{ext},,/\n",
            ));
        }
        // 49 = Account Trailer.
        let account_control = credits.saturating_add(debits);
        let account_records = 3 + tx_count; // 03 + 16s + 49 itself
        out.push_str(&format!(
            "49,{account_control},{account_records}/\n"
        ));
        // 98 = Group Trailer.
        out.push_str(&format!(
            "98,{account_control},1,{group_records}/\n",
            group_records = account_records + 2 // 02 + 49 group seen + 98
        ));
        // 99 = File Trailer.
        out.push_str(&format!(
            "99,{account_control},1,{file_records}/\n",
            file_records = account_records + 4 // + 01, 02, 98, 99
        ));
        let _ = file_id; // reserved; some banks include this in 01
        Ok(out)
    }
}

const fn bai_type_code(kind: StatementLineKind) -> (&'static str, bool) {
    // Returns (BAI2 type code, is_credit).
    match kind {
        // 142 = ACH credit; we use it generically for inbound.
        StatementLineKind::GrossCapture => ("142", true),
        // 451 = ACH debit (refund out).
        StatementLineKind::Refund => ("451", false),
        // 506 = Chargeback debit.
        StatementLineKind::Chargeback => ("506", false),
        // 720 = Service charge / Fee.
        StatementLineKind::Fee => ("720", false),
        // 195 = Outgoing wire (payout to merchant).
        StatementLineKind::Payout => ("195", false),
        // 399 = Misc credit; 699 = misc debit. We default to credit
        // and let the caller flip via signed Adjustment.
        StatementLineKind::Adjustment => ("399", true),
    }
}

fn unix_to_yymmdd(unix_secs: u64) -> String {
    let iso = crate::iso20022::unix_to_iso8601(unix_secs);
    // "2025-01-01T00:00:00Z" -> "250101"
    let bytes = iso.as_bytes();
    if bytes.len() < 10 {
        return "000000".to_owned();
    }
    let yy = core::str::from_utf8(&bytes[2..4]).unwrap_or("00");
    let mm = core::str::from_utf8(&bytes[5..7]).unwrap_or("00");
    let dd = core::str::from_utf8(&bytes[8..10]).unwrap_or("00");
    format!("{yy}{mm}{dd}")
}

fn unix_to_hhmm(unix_secs: u64) -> String {
    let iso = crate::iso20022::unix_to_iso8601(unix_secs);
    let bytes = iso.as_bytes();
    if bytes.len() < 16 {
        return "0000".to_owned();
    }
    let hh = core::str::from_utf8(&bytes[11..13]).unwrap_or("00");
    let mm = core::str::from_utf8(&bytes[14..16]).unwrap_or("00");
    format!("{hh}{mm}")
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cadence::Period;
    use crate::statement::{Statement, StatementLine};
    use op_core::{Currency, Money};

    #[test]
    fn emits_envelope_records() {
        let mut s = Statement::new(
            "STMT-1",
            "MRCH-1",
            Period::new(0, 86_399).unwrap(),
            Currency::USD,
        )
        .unwrap()
        .with_opening(Money::from_minor(1_000, Currency::USD))
        .unwrap();
        s.push_line(StatementLine::new(
            "l1",
            StatementLineKind::GrossCapture,
            Money::from_minor(10_000, Currency::USD),
            42,
        ))
        .unwrap();
        s.aggregate().unwrap();
        let bai = Bai2Writer::default().build(&s).unwrap();
        assert!(bai.starts_with("01,OPENPAY,MERCHANT,"));
        assert!(bai.contains("02,MERCHANT,OPENPAY,"));
        assert!(bai.contains("03,MRCH-1,USD,010,1000,"));
        assert!(bai.contains("16,142,10000,Z,"));
        assert!(bai.contains("\n49,"));
        assert!(bai.contains("\n98,"));
        assert!(bai.contains("\n99,"));
    }

    #[test]
    fn fee_uses_720_code() {
        let mut s = Statement::new("S", "M", Period::new(0, 1).unwrap(), Currency::USD).unwrap();
        s.push_line(StatementLine::new(
            "f1",
            StatementLineKind::Fee,
            Money::from_minor(290, Currency::USD),
            1,
        ))
        .unwrap();
        s.aggregate().unwrap();
        let bai = Bai2Writer::default().build(&s).unwrap();
        assert!(bai.contains("16,720,290,Z,"));
    }
}
