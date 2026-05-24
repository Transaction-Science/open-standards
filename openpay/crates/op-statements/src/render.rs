//! [`RenderTarget`] — pluggable statement serializer.
//!
//! Four reference targets ship: [`Pdf`] (template-driven plain-text,
//! suitable for piping into any PDF library), [`Csv`], [`Json`], and
//! [`FixedWidth`] (NACHA-style positional). Each renders a
//! [`Statement`] to a `String`; binary outputs are produced by adapters
//! that wrap a `RenderTarget` and a writer.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::statement::{Statement, StatementLineKind};

/// Render trait. Implementations turn a [`Statement`] into a string
/// representation suitable for piping into a file, an HTTP response,
/// or a downstream binary encoder (PDF library).
pub trait RenderTarget {
    /// Render. The returned `String` is the canonical text artifact.
    ///
    /// # Errors
    /// Implementation-defined; serialization failures (JSON encoding)
    /// surface as [`Error::Json`].
    fn render(&self, statement: &Statement) -> Result<String>;
}

/// Plain-text template-driven render suitable for downstream PDF
/// encoders. Stable, line-oriented, easy to diff.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Pdf;

impl RenderTarget for Pdf {
    fn render(&self, statement: &Statement) -> Result<String> {
        let mut out = String::new();
        out.push_str("# OpenPay Statement\n");
        out.push_str(&format!("Statement: {}\n", statement.id));
        out.push_str(&format!("Merchant:  {}\n", statement.merchant_id));
        out.push_str(&format!(
            "Period:    {} - {}\n",
            statement.period.start_unix_secs, statement.period.end_unix_secs
        ));
        out.push_str(&format!("Currency:  {}\n", statement.primary_currency));
        out.push('\n');
        out.push_str("## Aggregates\n");
        for agg in &statement.aggregates {
            out.push_str(&format!(
                "  [{}] opening={} gross={} refunds={} chargebacks={} fees={} payouts={} adj={} ending={}\n",
                agg.currency,
                agg.opening,
                agg.gross_volume,
                agg.refunds,
                agg.chargebacks,
                agg.fees,
                agg.payouts,
                agg.adjustments,
                agg.ending
            ));
        }
        out.push('\n');
        out.push_str("## Lines\n");
        for line in &statement.lines {
            out.push_str(&format!(
                "  {:>12} {:?} {} @ {}\n",
                line.id, line.kind, line.amount, line.posted_at_unix_secs
            ));
        }
        Ok(out)
    }
}

/// CSV (RFC 4180) render. One header row + one row per line plus an
/// aggregate footer.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct Csv;

fn csv_escape(s: &str) -> String {
    if s.contains(',') || s.contains('"') || s.contains('\n') {
        let escaped = s.replace('"', "\"\"");
        format!("\"{escaped}\"")
    } else {
        s.to_owned()
    }
}

impl RenderTarget for Csv {
    fn render(&self, statement: &Statement) -> Result<String> {
        let mut out = String::new();
        out.push_str(
            "line_id,kind,currency,amount_minor,posted_at_unix_secs,external_id\n",
        );
        for line in &statement.lines {
            let kind = match line.kind {
                StatementLineKind::GrossCapture => "gross_capture",
                StatementLineKind::Refund => "refund",
                StatementLineKind::Chargeback => "chargeback",
                StatementLineKind::Fee => "fee",
                StatementLineKind::Payout => "payout",
                StatementLineKind::Adjustment => "adjustment",
            };
            out.push_str(&format!(
                "{},{},{},{},{},{}\n",
                csv_escape(&line.id),
                kind,
                line.amount.currency,
                line.amount.minor_units,
                line.posted_at_unix_secs,
                csv_escape(line.external_id.as_deref().unwrap_or("")),
            ));
        }
        Ok(out)
    }
}

/// JSON render — round-trips a [`Statement`] via [`serde`].
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Json;

impl RenderTarget for Json {
    fn render(&self, statement: &Statement) -> Result<String> {
        serde_json::to_string(statement).map_err(|e| Error::Json(e.to_string()))
    }
}

/// NACHA-style fixed-width render. Each line is fixed 94 characters:
///
/// ```text
///  pos  width  field
///  1    16     line id (left-padded)
///  17   2      kind code (GC/RF/CB/FE/PY/AD)
///  19   3      currency
///  22   16     amount minor units (right-aligned, zero-padded)
///  38   10     posted at unix secs
///  48   24     external id (truncated/padded)
///  72   23     reserved (spaces)
/// ```
///
/// Total = 95 chars per record + `\n`. Useful for legacy bank
/// ingestion pipelines that won't take CSV.
#[derive(Copy, Clone, Debug, Default, PartialEq, Eq)]
pub struct FixedWidth;

impl FixedWidth {
    fn kind_code(k: StatementLineKind) -> &'static str {
        match k {
            StatementLineKind::GrossCapture => "GC",
            StatementLineKind::Refund => "RF",
            StatementLineKind::Chargeback => "CB",
            StatementLineKind::Fee => "FE",
            StatementLineKind::Payout => "PY",
            StatementLineKind::Adjustment => "AD",
        }
    }
}

fn left_pad(s: &str, width: usize) -> String {
    if s.len() >= width {
        s[..width].to_owned()
    } else {
        let mut out = String::with_capacity(width);
        for _ in 0..(width - s.len()) {
            out.push(' ');
        }
        out.push_str(s);
        out
    }
}

fn right_pad(s: &str, width: usize) -> String {
    if s.len() >= width {
        s[..width].to_owned()
    } else {
        let mut out = String::with_capacity(width);
        out.push_str(s);
        for _ in 0..(width - s.len()) {
            out.push(' ');
        }
        out
    }
}

fn zero_pad_i64(n: i64, width: usize) -> String {
    let s = if n < 0 {
        format!("-{}", n.unsigned_abs())
    } else {
        format!("{n}")
    };
    if s.len() >= width {
        s[..width].to_owned()
    } else {
        let mut out = String::with_capacity(width);
        for _ in 0..(width - s.len()) {
            out.push('0');
        }
        out.push_str(&s);
        out
    }
}

impl RenderTarget for FixedWidth {
    fn render(&self, statement: &Statement) -> Result<String> {
        let mut out = String::new();
        for line in &statement.lines {
            let id_field = left_pad(&line.id, 16);
            let kind_field = Self::kind_code(line.kind);
            let cur_field = right_pad(line.amount.currency.code(), 3);
            let amt_field = zero_pad_i64(line.amount.minor_units, 16);
            let posted_field = zero_pad_i64(
                i64::try_from(line.posted_at_unix_secs).unwrap_or(i64::MAX),
                10,
            );
            let ext_field = right_pad(line.external_id.as_deref().unwrap_or(""), 24);
            let reserved = " ".repeat(23);
            let record = format!(
                "{id_field}{kind_field}{cur_field}{amt_field}{posted_field}{ext_field}{reserved}"
            );
            debug_assert_eq!(record.len(), 94);
            out.push_str(&record);
            out.push('\n');
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cadence::Period;
    use crate::statement::{Statement, StatementLine};
    use op_core::{Currency, Money};

    fn sample_statement() -> Statement {
        let mut s = Statement::new(
            "S1",
            "M1",
            Period::new(0, 86_399).unwrap(),
            Currency::USD,
        )
        .unwrap();
        s.push_line(StatementLine::new(
            "l1",
            StatementLineKind::GrossCapture,
            Money::from_minor(10_000, Currency::USD),
            42,
        ))
        .unwrap();
        s.aggregate().unwrap();
        s
    }

    #[test]
    fn pdf_renders_headers() {
        let out = Pdf.render(&sample_statement()).unwrap();
        assert!(out.contains("OpenPay Statement"));
        assert!(out.contains("Statement: S1"));
        assert!(out.contains("Merchant:  M1"));
    }

    #[test]
    fn csv_has_header_and_rows() {
        let out = Csv.render(&sample_statement()).unwrap();
        assert!(out.starts_with("line_id,kind,"));
        assert!(out.contains("l1,gross_capture,USD,10000,42,"));
    }

    #[test]
    fn json_round_trips() {
        let s = sample_statement();
        let encoded = Json.render(&s).unwrap();
        let decoded: Statement = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, s);
    }

    #[test]
    fn fixed_width_record_is_94_chars() {
        let out = FixedWidth.render(&sample_statement()).unwrap();
        for record in out.lines() {
            assert_eq!(record.len(), 94, "record `{record}` is not 94 chars");
        }
    }
}
