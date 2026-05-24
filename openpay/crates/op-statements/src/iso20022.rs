//! ISO 20022 `camt.053` end-of-day customer statement builder.
//!
//! Builds the canonical `BkToCstmrStmt` envelope from a
//! [`Statement`]. The element tree we emit mirrors the camt.053.001.13
//! reference XSD; we serialize via a small hand-rolled XML writer to
//! avoid a heavyweight XML dep in this crate.
//!
//! ### What we emit
//!
//! ```xml
//! <?xml version="1.0" encoding="UTF-8"?>
//! <Document xmlns="urn:iso:std:iso:20022:tech:xsd:camt.053.001.13">
//!   <BkToCstmrStmt>
//!     <GrpHdr>
//!       <MsgId>{statement.id}</MsgId>
//!       <CreDtTm>{end-of-period iso8601}</CreDtTm>
//!     </GrpHdr>
//!     <Stmt>
//!       <Id>{statement.id}</Id>
//!       <Acct><Id><Othr><Id>{merchant_id}</Id></Othr></Id></Acct>
//!       <Bal>...opening...</Bal>
//!       <Bal>...closing...</Bal>
//!       <Ntry>...per line...</Ntry>
//!     </Stmt>
//!   </BkToCstmrStmt>
//! </Document>
//! ```
//!
//! For end-to-end XSD-canonical XML the operator pipes our output
//! through a quick-xml round-trip in op-iso20022; this builder
//! produces the structurally correct tree at zero dependency cost.

use crate::error::Result;
use crate::statement::{Statement, StatementLineKind};

const CAMT_053_NS: &str = "urn:iso:std:iso:20022:tech:xsd:camt.053.001.13";

/// Builder for `camt.053` end-of-day customer statement.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Camt053Builder;

impl Camt053Builder {
    /// Serialize a [`Statement`] as `camt.053` XML.
    ///
    /// # Errors
    /// [`Error::Xml`] on any structural failure (unreachable today;
    /// reserved for future schema validation).
    pub fn build(&self, statement: &Statement) -> Result<String> {
        let mut out = String::new();
        out.push_str("<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n");
        out.push_str(&format!("<Document xmlns=\"{CAMT_053_NS}\">\n"));
        out.push_str("  <BkToCstmrStmt>\n");
        // GrpHdr
        out.push_str("    <GrpHdr>\n");
        out.push_str(&format!("      <MsgId>{}</MsgId>\n", xml_escape(&statement.id)));
        out.push_str(&format!(
            "      <CreDtTm>{}</CreDtTm>\n",
            unix_to_iso8601(statement.period.end_unix_secs)
        ));
        out.push_str("    </GrpHdr>\n");
        // Stmt
        out.push_str("    <Stmt>\n");
        out.push_str(&format!("      <Id>{}</Id>\n", xml_escape(&statement.id)));
        out.push_str("      <CreDtTm>");
        out.push_str(&unix_to_iso8601(statement.period.end_unix_secs));
        out.push_str("</CreDtTm>\n");
        out.push_str("      <FrToDt>\n");
        out.push_str(&format!(
            "        <FrDtTm>{}</FrDtTm>\n",
            unix_to_iso8601(statement.period.start_unix_secs)
        ));
        out.push_str(&format!(
            "        <ToDtTm>{}</ToDtTm>\n",
            unix_to_iso8601(statement.period.end_unix_secs)
        ));
        out.push_str("      </FrToDt>\n");
        out.push_str("      <Acct>\n");
        out.push_str("        <Id>\n");
        out.push_str("          <Othr>\n");
        out.push_str(&format!(
            "            <Id>{}</Id>\n",
            xml_escape(&statement.merchant_id)
        ));
        out.push_str("          </Othr>\n");
        out.push_str("        </Id>\n");
        out.push_str(&format!(
            "        <Ccy>{}</Ccy>\n",
            statement.primary_currency
        ));
        out.push_str("      </Acct>\n");

        let primary = statement.primary_aggregate();
        // Opening (OPBD)
        write_balance(&mut out, "OPBD", primary.opening);
        // Closing (CLBD)
        write_balance(&mut out, "CLBD", primary.ending);

        // Entries
        for line in &statement.lines {
            let cdt_dbt = if line.kind.is_outflow()
                || matches!(line.kind, StatementLineKind::Adjustment)
                    && line.amount.minor_units < 0
            {
                "DBIT"
            } else {
                "CRDT"
            };
            out.push_str("      <Ntry>\n");
            if let Some(ext) = &line.external_id {
                out.push_str(&format!(
                    "        <NtryRef>{}</NtryRef>\n",
                    xml_escape(ext)
                ));
            }
            out.push_str(&format!(
                "        <Amt Ccy=\"{}\">{}</Amt>\n",
                line.amount.currency,
                format_minor(line.amount.minor_units.abs(), line.amount.currency.exponent())
            ));
            out.push_str(&format!("        <CdtDbtInd>{cdt_dbt}</CdtDbtInd>\n"));
            out.push_str("        <Sts><Cd>BOOK</Cd></Sts>\n");
            out.push_str(&format!(
                "        <BookgDt><DtTm>{}</DtTm></BookgDt>\n",
                unix_to_iso8601(line.posted_at_unix_secs)
            ));
            out.push_str(&format!(
                "        <ValDt><DtTm>{}</DtTm></ValDt>\n",
                unix_to_iso8601(line.posted_at_unix_secs)
            ));
            out.push_str(&format!(
                "        <AddtlNtryInf>{:?}</AddtlNtryInf>\n",
                line.kind
            ));
            out.push_str("      </Ntry>\n");
        }
        out.push_str("    </Stmt>\n");
        out.push_str("  </BkToCstmrStmt>\n");
        out.push_str("</Document>\n");
        Ok(out)
    }
}

fn write_balance(out: &mut String, type_code: &str, money: op_core::Money) {
    let cdt_dbt = if money.minor_units < 0 { "DBIT" } else { "CRDT" };
    out.push_str("      <Bal>\n");
    out.push_str("        <Tp><CdOrPrtry><Cd>");
    out.push_str(type_code);
    out.push_str("</Cd></CdOrPrtry></Tp>\n");
    out.push_str(&format!(
        "        <Amt Ccy=\"{}\">{}</Amt>\n",
        money.currency,
        format_minor(money.minor_units.abs(), money.currency.exponent())
    ));
    out.push_str(&format!("        <CdtDbtInd>{cdt_dbt}</CdtDbtInd>\n"));
    out.push_str("      </Bal>\n");
}

fn xml_escape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    for c in s.chars() {
        match c {
            '&' => out.push_str("&amp;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '"' => out.push_str("&quot;"),
            '\'' => out.push_str("&apos;"),
            other => out.push(other),
        }
    }
    out
}

/// Decimal formatting with the currency's exponent (e.g. 1234 -> "12.34").
/// Used for both balance and entry amounts.
fn format_minor(minor: i64, exponent: u8) -> String {
    let exp = u32::from(exponent);
    if exp == 0 {
        return format!("{minor}");
    }
    let divisor = 10_i64.pow(exp);
    let whole = minor / divisor;
    let frac = (minor % divisor).abs();
    format!("{whole}.{frac:0width$}", width = exp as usize)
}

/// Convert a unix epoch seconds value to an ISO 8601 date-time string
/// (UTC, second precision: `YYYY-MM-DDTHH:MM:SSZ`).
///
/// Pure-Rust, no chrono / time dependency. Handles the proleptic
/// Gregorian calendar correctly for all dates within `i64` seconds
/// representability.
pub(crate) fn unix_to_iso8601(unix_secs: u64) -> String {
    // Algorithm from Howard Hinnant's date library, "civil_from_days".
    let secs = unix_secs;
    let days = secs / 86_400;
    let hms = secs % 86_400;
    let hour = hms / 3_600;
    let minute = (hms % 3_600) / 60;
    let second = hms % 60;

    // days since 1970-01-01 -> Y-M-D.
    let z = i64::try_from(days).unwrap_or(0) + 719_468;
    let era = z.div_euclid(146_097);
    let doe = (z - era * 146_097) as u64; // [0, 146096]
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365; // [0, 399]
    let y = i64::try_from(yoe).unwrap_or(0) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100); // [0, 365]
    let mp = (5 * doy + 2) / 153; // [0, 11]
    let d = doy - (153 * mp + 2) / 5 + 1; // [1, 31]
    let m = if mp < 10 { mp + 3 } else { mp - 9 }; // [1, 12]
    let y = if m <= 2 { y + 1 } else { y };

    format!(
        "{y:04}-{m:02}-{d:02}T{hour:02}:{minute:02}:{second:02}Z"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cadence::Period;
    use crate::statement::{Statement, StatementLine};
    use op_core::{Currency, Money};

    #[test]
    fn unix_zero_is_1970_epoch() {
        assert_eq!(unix_to_iso8601(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn unix_known_date_2026_01_01() {
        // 1735689600 = 2025-01-01T00:00:00Z
        assert_eq!(unix_to_iso8601(1_735_689_600), "2025-01-01T00:00:00Z");
    }

    #[test]
    fn format_minor_two_decimals() {
        assert_eq!(format_minor(1234, 2), "12.34");
        assert_eq!(format_minor(1, 2), "0.01");
        assert_eq!(format_minor(100, 2), "1.00");
    }

    #[test]
    fn format_minor_zero_exp() {
        assert_eq!(format_minor(1000, 0), "1000");
    }

    #[test]
    fn build_emits_xml_envelope() {
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
        let xml = Camt053Builder.build(&s).unwrap();
        assert!(xml.contains("<?xml version=\"1.0\""));
        assert!(xml.contains("urn:iso:std:iso:20022:tech:xsd:camt.053.001.13"));
        assert!(xml.contains("<MsgId>STMT-1</MsgId>"));
        assert!(xml.contains("<Cd>OPBD</Cd>"));
        assert!(xml.contains("<Cd>CLBD</Cd>"));
        assert!(xml.contains("<Amt Ccy=\"USD\">100.00</Amt>"));
    }

    #[test]
    fn build_escapes_xml() {
        let s = Statement::new(
            "M&L<x>",
            "merch\"1",
            Period::new(0, 1).unwrap(),
            Currency::USD,
        )
        .unwrap();
        let xml = Camt053Builder.build(&s).unwrap();
        assert!(xml.contains("M&amp;L&lt;x&gt;"));
        assert!(xml.contains("merch&quot;1"));
    }
}
