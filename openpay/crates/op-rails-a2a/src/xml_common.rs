//! Shared XML utilities used by every A2A rail driver.
//!
//! These functions are not profile-parameterized so they live outside
//! the per-rail modules. Centralizing them means a deployment that only
//! enables `pix` (without `fednow`) still has access to the parser and
//! the decimal formatter.
//!
//! ## What lives here
//!
//! - [`format_money`] — ISO 20022 decimal-string formatting for [`Money`].
//! - [`xml_escape`] — XML 1.0 entity escaping for the five reserved chars.
//! - [`parse_pacs002`] / [`ParsedPacs002`] — minimal pacs.002.001.10 reader.
//! - [`extract_first_tag`] — lightweight tag-scan helper.
//!
//! ## What does NOT live here
//!
//! - `emit_pacs008` — each rail emits its own profile-specific body
//!   (`FedNow` uses USABA + `Othr/Id`, SEPA uses BICFI + IBAN, PIX uses
//!   BRSPB + custom). Sharing the emitter would force a lowest-common-
//!   denominator schema, which the schemes themselves reject.

use op_core::Money;

use crate::error::{Error, Result};

/// Format a `Money` value as an ISO 20022 decimal string.
///
/// - `Money { 12345, USD (exp 2) }` → `"123.45"`
/// - `Money { 500, JPY (exp 0) }` → `"500"`
/// - `Money { 1, USD (exp 2) }` → `"0.01"`
/// - `Money { -250, USD (exp 2) }` → `"-2.50"`
#[must_use]
pub fn format_money(m: Money) -> String {
    let exp = m.currency.exponent();
    if exp == 0 {
        return m.minor_units.to_string();
    }
    let abs = m.minor_units.unsigned_abs();
    let divisor = 10u64.pow(u32::from(exp));
    let whole = abs / divisor;
    let frac = abs % divisor;
    let sign = if m.minor_units < 0 { "-" } else { "" };
    format!("{sign}{whole}.{frac:0width$}", width = usize::from(exp))
}

/// Escape the five XML reserved characters: `& < > " '`.
#[must_use]
pub fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

/// Parsed fields from a pacs.002.001.10 response.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedPacs002 {
    /// `TxSts` element — ACSC, ACTC, RJCT, PDNG, etc.
    pub transaction_status: String,
    /// `OrgnlUETR` echoed back.
    pub uetr: Option<String>,
    /// `OrgnlEndToEndId` echoed back.
    pub original_end_to_end_id: Option<String>,
    /// `StsRsnInf/Rsn/Cd` — reason code, present on rejection.
    pub reason_code: Option<String>,
    /// `StsRsnInf/AddtlInf` — free text.
    pub reason_text: Option<String>,
}

/// Parse a pacs.002.001.10 XML response.
///
/// Lightweight tag-scan parser; does NOT validate XML well-formedness
/// beyond what's needed to extract the five documented fields. This is
/// deliberate — every supported rail uses the same five fields, and
/// the parser must work the same way for all of them.
///
/// # Errors
/// `Error::Transport` if `TxSts` is missing entirely (mandatory field
/// per the ISO 20022 schema).
pub fn parse_pacs002(xml: &str) -> Result<ParsedPacs002> {
    let tx_sts = extract_first_tag(xml, "TxSts")
        .ok_or_else(|| Error::Transport("pacs.002 missing TxSts".into()))?;
    Ok(ParsedPacs002 {
        transaction_status: tx_sts,
        uetr: extract_first_tag(xml, "OrgnlUETR"),
        original_end_to_end_id: extract_first_tag(xml, "OrgnlEndToEndId"),
        reason_code: extract_first_tag(xml, "Cd"),
        reason_text: extract_first_tag(xml, "AddtlInf"),
    })
}

/// Extract the text content of the first `<tag>...</tag>` occurrence.
///
/// Note: extracts inner text verbatim, including any nested XML. For
/// the pacs.002 fields we care about (UETR, `EndToEndId`, status codes),
/// content is always plain text per the schema.
#[must_use]
pub fn extract_first_tag(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = xml.find(&open)? + open.len();
    let end_offset = xml[start..].find(&close)?;
    Some(xml[start..start + end_offset].trim().to_owned())
}

#[cfg(test)]
mod tests {
    use super::*;
    use op_core::Currency;

    #[test]
    fn format_money_two_dp() {
        assert_eq!(
            format_money(Money::from_minor(12345, Currency::USD)),
            "123.45"
        );
    }

    #[test]
    fn format_money_one_cent() {
        assert_eq!(format_money(Money::from_minor(1, Currency::USD)), "0.01");
    }

    #[test]
    fn format_money_zero_dp() {
        assert_eq!(format_money(Money::from_minor(500, Currency::JPY)), "500");
    }

    #[test]
    fn format_money_zero() {
        assert_eq!(format_money(Money::from_minor(0, Currency::USD)), "0.00");
    }

    #[test]
    fn format_money_negative() {
        assert_eq!(
            format_money(Money::from_minor(-250, Currency::USD)),
            "-2.50"
        );
    }

    #[test]
    fn xml_escape_handles_all_five_entities() {
        assert_eq!(xml_escape(r#"<&>"'"#), "&lt;&amp;&gt;&quot;&apos;");
    }

    #[test]
    fn xml_escape_preserves_plain_text() {
        assert_eq!(xml_escape("Invoice 4242"), "Invoice 4242");
    }

    #[test]
    fn extract_first_tag_basic() {
        assert_eq!(
            extract_first_tag("<a>hello</a>", "a"),
            Some("hello".to_owned())
        );
    }

    #[test]
    fn extract_first_tag_takes_first_of_many() {
        assert_eq!(
            extract_first_tag("<a>one</a><a>two</a>", "a"),
            Some("one".to_owned())
        );
    }

    #[test]
    fn extract_first_tag_trims_whitespace() {
        assert_eq!(
            extract_first_tag("<a>  hello  </a>", "a"),
            Some("hello".to_owned())
        );
    }

    #[test]
    fn extract_first_tag_returns_none_when_absent() {
        assert!(extract_first_tag("<root></root>", "missing").is_none());
    }

    #[test]
    fn parse_pacs002_acsc() {
        let xml = r"<Document><TxInfAndSts>
            <OrgnlUETR>uetr-abc</OrgnlUETR>
            <OrgnlEndToEndId>e2e-1</OrgnlEndToEndId>
            <TxSts>ACSC</TxSts>
        </TxInfAndSts></Document>";
        let p = parse_pacs002(xml).unwrap();
        assert_eq!(p.transaction_status, "ACSC");
        assert_eq!(p.uetr.as_deref(), Some("uetr-abc"));
        assert_eq!(p.original_end_to_end_id.as_deref(), Some("e2e-1"));
        assert!(p.reason_code.is_none());
    }

    #[test]
    fn parse_pacs002_rjct_with_reason() {
        let xml = r"<Document><TxInfAndSts>
            <OrgnlUETR>uetr-xyz</OrgnlUETR>
            <OrgnlEndToEndId>e2e-2</OrgnlEndToEndId>
            <TxSts>RJCT</TxSts>
            <StsRsnInf><Rsn><Cd>AC03</Cd></Rsn><AddtlInf>Invalid creditor account</AddtlInf></StsRsnInf>
        </TxInfAndSts></Document>";
        let p = parse_pacs002(xml).unwrap();
        assert_eq!(p.transaction_status, "RJCT");
        assert_eq!(p.reason_code.as_deref(), Some("AC03"));
        assert_eq!(p.reason_text.as_deref(), Some("Invalid creditor account"));
    }

    #[test]
    fn parse_pacs002_missing_status_errors() {
        let xml = r"<Document><TxInfAndSts><OrgnlUETR>x</OrgnlUETR></TxInfAndSts></Document>";
        assert!(matches!(parse_pacs002(xml), Err(Error::Transport(_))));
    }

    #[test]
    fn parse_pacs002_pdng_only() {
        let xml = r"<Document><TxSts>PDNG</TxSts></Document>";
        let p = parse_pacs002(xml).unwrap();
        assert_eq!(p.transaction_status, "PDNG");
        assert!(p.uetr.is_none());
    }
}
