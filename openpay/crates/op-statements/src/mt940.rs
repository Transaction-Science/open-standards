//! SWIFT MT940 customer statement writer (and MT942 intra-day report).
//!
//! MT940 is the canonical end-of-day statement in SWIFT FIN format,
//! still the workhorse for European correspondent banking and many
//! ERP integrations. MT942 is the same shape, intra-day. Both share
//! field semantics; the only difference relevant to us is the
//! statement-type indicator on field `:25:` / `:28C:`.
//!
//! ### Fields emitted
//!
//! - `:20:` Transaction reference number — `statement.id`.
//! - `:25:` Account identification — `merchant_id`.
//! - `:28C:` Statement / sequence number — `1/1`.
//! - `:60F:` Opening balance — type `C` or `D`, date YYMMDD, currency, amount.
//! - `:61:` Statement line — one per [`StatementLine`].
//! - `:86:` Information to account owner — free text (kind, ext id).
//! - `:62F:` Closing balance.

use crate::error::Result;
use crate::statement::{Statement, StatementLineKind};

/// MT940 (and MT942) writer.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Mt940Writer {
    /// If true, emits as an MT942 intra-day report; otherwise MT940.
    /// MT942 omits the `:60F:` opening balance and uses `:90D:`/`:90C:`
    /// summary counters; in our minimal implementation the only
    /// observable change is the field label set we use.
    pub intra_day: bool,
}

impl Mt940Writer {
    /// Serialize a [`Statement`] as MT940/MT942 plaintext.
    ///
    /// # Errors
    /// Reserved.
    pub fn build(&self, statement: &Statement) -> Result<String> {
        let primary = statement.primary_aggregate();
        let mut out = String::new();
        out.push_str(&format!(":20:{}\n", truncate(&statement.id, 16)));
        out.push_str(&format!(
            ":25:{}\n",
            truncate(&statement.merchant_id, 35)
        ));
        out.push_str(":28C:1/1\n");

        let yymmdd_end = unix_to_yymmdd(statement.period.end_unix_secs);
        let yymmdd_start = unix_to_yymmdd(statement.period.start_unix_secs);

        if !self.intra_day {
            // :60F: Opening balance
            let (sign, amt) = signed_for_swift(primary.opening.minor_units);
            out.push_str(&format!(
                ":60F:{sign}{yymmdd_start}{cur}{amt}\n",
                cur = primary.opening.currency,
                amt = format_swift_amount(amt, primary.opening.currency.exponent())
            ));
        }

        // :61: + :86: per line.
        for line in &statement.lines {
            let (sign, amt) = signed_for_swift_line(line.kind, line.amount.minor_units);
            let entry_date = unix_to_yymmdd(line.posted_at_unix_secs);
            let value_date = &entry_date[2..]; // MMDD on :61:
            let n_code = mt940_transaction_type(line.kind);
            let ref_field = line
                .external_id
                .as_deref()
                .map_or_else(|| line.id.clone(), str::to_owned);
            out.push_str(&format!(
                ":61:{entry_date}{value_date}{sign}{amt}{n_code}{ref_field}//{src}\n",
                amt = format_swift_amount(amt, line.amount.currency.exponent()),
                src = truncate(&line.id, 16)
            ));
            out.push_str(&format!(
                ":86:{kind:?} ext={ext}\n",
                kind = line.kind,
                ext = line.external_id.as_deref().unwrap_or("")
            ));
        }

        if !self.intra_day {
            // :62F: Closing balance
            let (sign, amt) = signed_for_swift(primary.ending.minor_units);
            out.push_str(&format!(
                ":62F:{sign}{yymmdd_end}{cur}{amt}\n",
                cur = primary.ending.currency,
                amt = format_swift_amount(amt, primary.ending.currency.exponent())
            ));
        } else {
            // :90D:/:90C: summary counters for MT942.
            let mut credits: i64 = 0;
            let mut debits: i64 = 0;
            for line in &statement.lines {
                if line.kind.is_outflow()
                    || matches!(line.kind, StatementLineKind::Adjustment)
                        && line.amount.minor_units < 0
                {
                    debits = debits.saturating_add(line.amount.minor_units.abs());
                } else {
                    credits = credits.saturating_add(line.amount.minor_units.abs());
                }
            }
            let cur = statement.primary_currency;
            let exp = cur.exponent();
            out.push_str(&format!(
                ":90D:{n}{cur}{a}\n",
                n = statement.lines.iter().filter(|l| l.kind.is_outflow()).count(),
                a = format_swift_amount(debits, exp)
            ));
            out.push_str(&format!(
                ":90C:{n}{cur}{a}\n",
                n = statement
                    .lines
                    .iter()
                    .filter(|l| matches!(l.kind, StatementLineKind::GrossCapture))
                    .count(),
                a = format_swift_amount(credits, exp)
            ));
        }
        Ok(out)
    }
}

fn signed_for_swift(minor: i64) -> (&'static str, i64) {
    if minor < 0 { ("D", minor.abs()) } else { ("C", minor) }
}

fn signed_for_swift_line(kind: StatementLineKind, minor: i64) -> (&'static str, i64) {
    let mag = minor.abs();
    match kind {
        StatementLineKind::GrossCapture => ("C", mag),
        StatementLineKind::Refund
        | StatementLineKind::Chargeback
        | StatementLineKind::Fee
        | StatementLineKind::Payout => ("D", mag),
        StatementLineKind::Adjustment => signed_for_swift(minor),
    }
}

const fn mt940_transaction_type(kind: StatementLineKind) -> &'static str {
    // SWIFT N-code prefixes per MT940 spec.
    match kind {
        StatementLineKind::GrossCapture => "NTRF", // Transfer
        StatementLineKind::Refund => "NRTI",       // Reversal
        StatementLineKind::Chargeback => "NCHG",   // Charge
        StatementLineKind::Fee => "NCOM",          // Commission
        StatementLineKind::Payout => "NTRF",       // Transfer
        StatementLineKind::Adjustment => "NMSC",   // Miscellaneous
    }
}

fn format_swift_amount(minor: i64, exponent: u8) -> String {
    let exp = u32::from(exponent);
    if exp == 0 {
        return format!("{minor},");
    }
    let divisor = 10_i64.pow(exp);
    let whole = minor / divisor;
    let frac = (minor % divisor).abs();
    // SWIFT decimal separator is ','
    format!("{whole},{frac:0width$}", width = exp as usize)
}

fn unix_to_yymmdd(unix_secs: u64) -> String {
    let iso = crate::iso20022::unix_to_iso8601(unix_secs);
    let bytes = iso.as_bytes();
    if bytes.len() < 10 {
        return "000000".to_owned();
    }
    let yy = core::str::from_utf8(&bytes[2..4]).unwrap_or("00");
    let mm = core::str::from_utf8(&bytes[5..7]).unwrap_or("00");
    let dd = core::str::from_utf8(&bytes[8..10]).unwrap_or("00");
    format!("{yy}{mm}{dd}")
}

fn truncate(s: &str, n: usize) -> String {
    if s.len() <= n { s.to_owned() } else { s[..n].to_owned() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cadence::Period;
    use crate::statement::{Statement, StatementLine};
    use op_core::{Currency, Money};

    #[test]
    fn mt940_emits_canonical_fields() {
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
        let mt = Mt940Writer::default().build(&s).unwrap();
        assert!(mt.contains(":20:STMT-1"));
        assert!(mt.contains(":25:MRCH-1"));
        assert!(mt.contains(":28C:1/1"));
        assert!(mt.contains(":60F:C700101USD"));
        assert!(mt.contains(":61:"));
        assert!(mt.contains(":62F:"));
        // The :61: amount carries no currency (it lives on :25:/:60F:).
        // For the $100.00 capture we expect a credit amount of "100,00"
        // and the N-code suffix NTRF for a gross capture.
        assert!(mt.contains("C100,00NTRF"));
        // The opening balance of $10.00 is "10,00" in MT940 decimal form.
        assert!(mt.contains(":60F:C700101USD10,00"));
    }

    #[test]
    fn mt942_uses_summary_counters_not_balance() {
        let mut s = Statement::new(
            "STMT-2",
            "MRCH-1",
            Period::new(0, 1).unwrap(),
            Currency::USD,
        )
        .unwrap();
        s.push_line(StatementLine::new(
            "l1",
            StatementLineKind::GrossCapture,
            Money::from_minor(500, Currency::USD),
            0,
        ))
        .unwrap();
        s.aggregate().unwrap();
        let mt = Mt940Writer { intra_day: true }.build(&s).unwrap();
        assert!(!mt.contains(":60F:"));
        assert!(!mt.contains(":62F:"));
        assert!(mt.contains(":90C:"));
        assert!(mt.contains(":90D:"));
    }
}
