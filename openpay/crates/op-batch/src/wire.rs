//! Wire transfers: Fedwire / SWIFT MT103, MT202, MT103+, CHIPS, and
//! ISO 20022 `pacs.008` / `pacs.009`.
//!
//! ## Two worlds, one module
//!
//! Two on-wire encodings cover everything an operator will need:
//!
//! 1. **MT (Message Type) text format** — the legacy SWIFT format
//!    still in active use for cross-border. Block-based:
//!    `{1:...}{2:...}{4:...:tag:value...}` etc. Each tag (`:20:`,
//!    `:32A:`, `:50K:`, `:59:`, etc.) has a fixed semantic.
//!
//! 2. **ISO 20022 XML** — Fedwire migrated to `pacs.008`/`pacs.009`
//!    in March 2025; SWIFT's coexistence period for cross-border
//!    runs through November 2025 (full cutover thereafter). For
//!    ISO 20022 messages we delegate to [`op_iso20022`] where the
//!    typed builders exist; for raw `pacs.009` (FI-to-FI) we
//!    construct minimal XML directly.
//!
//! ## CHIPS
//!
//! CHIPS uses a SWIFT-like message format with `CHIPS Universal
//! Identifier` (`{113:...}` extension and `{121:...}` UETR per the
//! 2018 MT103+ alignment). For the purposes of this crate, CHIPS is
//! an MT103 with extra header tags; we round-trip the same struct
//! and tag CHIPS via the [`WireFormat`] discriminant so operators
//! can dispatch on it.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// On-wire format selector for a wire message.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum WireFormat {
    /// SWIFT MT103 — single customer credit transfer.
    Mt103,
    /// SWIFT MT103+ — STP-conformant variant (mandatory IBAN / BIC).
    Mt103Plus,
    /// SWIFT MT202 — FI-to-FI general transfer.
    Mt202,
    /// CHIPS payment message.
    Chips,
    /// ISO 20022 `pacs.008` customer credit transfer.
    Pacs008,
    /// ISO 20022 `pacs.009` FI credit transfer.
    Pacs009,
}

/// A wire message: format-independent in-memory representation.
///
/// Operators populate the fields appropriate to their target rail
/// (Fedwire = `pacs.008` or MT103, SWIFT cross-border = MT103 or
/// `pacs.008`, CHIPS = MT103-shape with extension headers).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct WireMessage {
    /// Output format.
    pub format: WireFormat,
    /// Sender's reference (`:20:` in MT; `MsgId` in pacs.008).
    pub sender_reference: String,
    /// UETR — Unique End-to-end Transaction Reference. Mandatory on
    /// modern Fedwire / SWIFT (`{121:...}` block-3 or `<UETR>` in
    /// ISO 20022).
    pub uetr: String,
    /// Value date `YYMMDD` (MT) or ISO `YYYY-MM-DD` (pacs).
    pub value_date: String,
    /// ISO 4217 currency code (`USD`, `EUR`, ...).
    pub currency: String,
    /// Amount in minor units (cents for most currencies, units for JPY).
    pub amount_minor: u64,
    /// Number of decimal places for the currency (2 for USD, 0 for JPY).
    pub currency_exponent: u8,
    /// Ordering customer (debtor) — `:50K:` in MT103.
    pub ordering_customer: PartyRef,
    /// Beneficiary customer (creditor) — `:59:` in MT103.
    pub beneficiary_customer: PartyRef,
    /// Sender (ordering) FI BIC.
    pub sender_bic: String,
    /// Receiver (beneficiary) FI BIC.
    pub receiver_bic: String,
    /// Correspondent banking chain (intermediary BICs).
    pub correspondents: Vec<String>,
    /// Remittance info (`:70:` in MT103; `<RmtInf>` in pacs.008).
    pub remittance_info: Option<String>,
}

/// A counterparty: account number / IBAN and name.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PartyRef {
    /// Account number or IBAN.
    pub account: String,
    /// Free-text name (≤35 chars per `:50K:` line in MT103).
    pub name: String,
}

impl WireMessage {
    /// Encode as MT103 / MT202 / CHIPS / pacs.008 / pacs.009 wire bytes.
    ///
    /// # Errors
    /// [`Error::FieldRule`] for bad BIC / IBAN / amount values.
    pub fn encode(&self) -> Result<String> {
        match self.format {
            WireFormat::Mt103 | WireFormat::Mt103Plus | WireFormat::Chips => self.encode_mt103(),
            WireFormat::Mt202 => self.encode_mt202(),
            WireFormat::Pacs008 => self.encode_pacs008_xml(),
            WireFormat::Pacs009 => self.encode_pacs009_xml(),
        }
    }

    /// Decode from on-wire bytes, dispatching on the message shape.
    ///
    /// Heuristic: starts with `<?xml` or `<Document` → ISO 20022;
    /// starts with `{1:` → MT.
    ///
    /// # Errors
    /// [`Error::FieldRule`] for malformed input.
    pub fn decode(input: &str) -> Result<Self> {
        let trimmed = input.trim_start();
        if trimmed.starts_with('<') {
            if trimmed.contains("pacs.009") {
                Self::decode_pacs009_xml(input)
            } else {
                Self::decode_pacs008_xml(input)
            }
        } else if trimmed.starts_with("{1:") {
            // Distinguish MT103 from MT202 by the application id in
            // block 2 (`{2:I103...}` vs `{2:I202...}`).
            if input.contains("{2:I202") || input.contains(":202:") {
                Self::decode_mt202(input)
            } else {
                Self::decode_mt103(input)
            }
        } else {
            Err(Error::FieldRule {
                field: "wire",
                reason: "unknown wire format (expected MT or XML)".into(),
            })
        }
    }

    fn encode_mt103(&self) -> Result<String> {
        validate_bic(&self.sender_bic, "sender_bic")?;
        validate_bic(&self.receiver_bic, "receiver_bic")?;
        let block1 = format!("{{1:F01{}AXXX0000000000}}", pad_bic(&self.sender_bic));
        // Block 2: input (operator-to-network).
        let app_id = match self.format {
            WireFormat::Mt103 | WireFormat::Mt103Plus | WireFormat::Chips => "103",
            _ => "103",
        };
        let block2 = format!(
            "{{2:I{}{}N}}",
            app_id,
            pad_bic(&self.receiver_bic)
        );
        // Block 3: user header, carry UETR in `{121:...}`.
        let block3 = format!("{{3:{{121:{}}}}}", self.uetr);
        // Block 4: text — tag-value pairs.
        let amount = format_mt_amount(self.amount_minor, self.currency_exponent);
        let mut block4 = String::new();
        block4.push_str(&format!(":20:{}\n", self.sender_reference));
        block4.push_str(":23B:CRED\n");
        block4.push_str(&format!(
            ":32A:{}{}{}\n",
            self.value_date, self.currency, amount
        ));
        block4.push_str(&format!(
            ":50K:/{}\n{}\n",
            self.ordering_customer.account, self.ordering_customer.name
        ));
        for c in &self.correspondents {
            block4.push_str(&format!(":56A:{c}\n"));
        }
        block4.push_str(&format!(
            ":59:/{}\n{}\n",
            self.beneficiary_customer.account, self.beneficiary_customer.name
        ));
        if let Some(r) = &self.remittance_info {
            block4.push_str(&format!(":70:{r}\n"));
        }
        block4.push_str(":71A:SHA\n");
        Ok(format!("{block1}{block2}{block3}{{4:\n{block4}-}}"))
    }

    fn encode_mt202(&self) -> Result<String> {
        validate_bic(&self.sender_bic, "sender_bic")?;
        validate_bic(&self.receiver_bic, "receiver_bic")?;
        let block1 = format!("{{1:F01{}AXXX0000000000}}", pad_bic(&self.sender_bic));
        let block2 = format!("{{2:I202{}N}}", pad_bic(&self.receiver_bic));
        let block3 = format!("{{3:{{121:{}}}}}", self.uetr);
        let amount = format_mt_amount(self.amount_minor, self.currency_exponent);
        let mut block4 = String::new();
        block4.push_str(&format!(":20:{}\n", self.sender_reference));
        block4.push_str(&format!(":21:{}\n", self.sender_reference));
        block4.push_str(&format!(
            ":32A:{}{}{}\n",
            self.value_date, self.currency, amount
        ));
        block4.push_str(&format!(":58A:{}\n", self.receiver_bic));
        Ok(format!("{block1}{block2}{block3}{{4:\n{block4}-}}"))
    }

    fn encode_pacs008_xml(&self) -> Result<String> {
        // Minimal canonical pacs.008.001.12 — for operators who want a
        // hand-rolled message rather than going through op_iso20022's
        // typed builder (which targets specific rail profiles).
        let amt = format_iso_amount(self.amount_minor, self.currency_exponent);
        Ok(format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Document xmlns=\"urn:iso:std:iso:20022:tech:xsd:pacs.008.001.12\"><FIToFICstmrCdtTrf><GrpHdr><MsgId>{msg}</MsgId><CreDtTm>{vd}T00:00:00Z</CreDtTm><NbOfTxs>1</NbOfTxs><SttlmInf><SttlmMtd>CLRG</SttlmMtd></SttlmInf></GrpHdr><CdtTrfTxInf><PmtId><InstrId>{msg}</InstrId><EndToEndId>{msg}</EndToEndId><UETR>{uetr}</UETR></PmtId><IntrBkSttlmAmt Ccy=\"{ccy}\">{amt}</IntrBkSttlmAmt><Dbtr><Nm>{don}</Nm></Dbtr><DbtrAcct><Id><Othr><Id>{doa}</Id></Othr></Id></DbtrAcct><DbtrAgt><FinInstnId><BICFI>{sbic}</BICFI></FinInstnId></DbtrAgt><CdtrAgt><FinInstnId><BICFI>{rbic}</BICFI></FinInstnId></CdtrAgt><Cdtr><Nm>{cn}</Nm></Cdtr><CdtrAcct><Id><Othr><Id>{ca}</Id></Othr></Id></CdtrAcct></CdtTrfTxInf></FIToFICstmrCdtTrf></Document>",
            msg = self.sender_reference,
            vd = self.value_date,
            uetr = self.uetr,
            ccy = self.currency,
            amt = amt,
            don = self.ordering_customer.name,
            doa = self.ordering_customer.account,
            sbic = self.sender_bic,
            rbic = self.receiver_bic,
            cn = self.beneficiary_customer.name,
            ca = self.beneficiary_customer.account,
        ))
    }

    fn encode_pacs009_xml(&self) -> Result<String> {
        let amt = format_iso_amount(self.amount_minor, self.currency_exponent);
        Ok(format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Document xmlns=\"urn:iso:std:iso:20022:tech:xsd:pacs.009.001.10\"><FICdtTrf><GrpHdr><MsgId>{msg}</MsgId><CreDtTm>{vd}T00:00:00Z</CreDtTm><NbOfTxs>1</NbOfTxs><SttlmInf><SttlmMtd>CLRG</SttlmMtd></SttlmInf></GrpHdr><CdtTrfTxInf><PmtId><InstrId>{msg}</InstrId><EndToEndId>{msg}</EndToEndId><UETR>{uetr}</UETR></PmtId><IntrBkSttlmAmt Ccy=\"{ccy}\">{amt}</IntrBkSttlmAmt><Dbtr><FinInstnId><BICFI>{sbic}</BICFI></FinInstnId></Dbtr><Cdtr><FinInstnId><BICFI>{rbic}</BICFI></FinInstnId></Cdtr></CdtTrfTxInf></FICdtTrf></Document>",
            msg = self.sender_reference,
            vd = self.value_date,
            uetr = self.uetr,
            ccy = self.currency,
            amt = amt,
            sbic = self.sender_bic,
            rbic = self.receiver_bic,
        ))
    }

    fn decode_mt103(input: &str) -> Result<Self> {
        let uetr = mt_block3_uetr(input).unwrap_or_default();
        let sender_bic = mt_extract_bic_from_block1(input).unwrap_or_default();
        let receiver_bic = mt_extract_bic_from_block2(input).unwrap_or_default();
        let sender_reference = mt_tag(input, ":20:").unwrap_or_default();
        let (value_date, currency, amount_minor, currency_exponent) =
            mt_parse_32a(input).unwrap_or_default();
        let ordering = mt_party(input, ":50K:");
        let beneficiary = mt_party(input, ":59:");
        let remittance = mt_tag(input, ":70:");
        let correspondents = mt_all_tag(input, ":56A:");
        Ok(Self {
            format: WireFormat::Mt103,
            sender_reference,
            uetr,
            value_date,
            currency,
            amount_minor,
            currency_exponent,
            ordering_customer: ordering,
            beneficiary_customer: beneficiary,
            sender_bic,
            receiver_bic,
            correspondents,
            remittance_info: remittance,
        })
    }

    fn decode_mt202(input: &str) -> Result<Self> {
        let mut m = Self::decode_mt103(input)?;
        m.format = WireFormat::Mt202;
        Ok(m)
    }

    fn decode_pacs008_xml(xml: &str) -> Result<Self> {
        let sender_reference = xml_extract(xml, "MsgId").unwrap_or_default();
        let uetr = xml_extract(xml, "UETR").unwrap_or_default();
        let sender_bic = xml_path(xml, &["DbtrAgt", "FinInstnId", "BICFI"]).unwrap_or_default();
        let receiver_bic = xml_path(xml, &["CdtrAgt", "FinInstnId", "BICFI"]).unwrap_or_default();
        let don = xml_path(xml, &["Dbtr", "Nm"]).unwrap_or_default();
        let doa = xml_path(xml, &["DbtrAcct", "Id", "Othr", "Id"]).unwrap_or_default();
        let cn = xml_path(xml, &["Cdtr", "Nm"]).unwrap_or_default();
        let ca = xml_path(xml, &["CdtrAcct", "Id", "Othr", "Id"]).unwrap_or_default();
        let (currency, amount_minor) = xml_parse_intrbk_amt(xml).unwrap_or_default();
        let value_date = xml_extract(xml, "CreDtTm")
            .map(|s| s.split('T').next().unwrap_or("").to_string())
            .unwrap_or_default();
        Ok(Self {
            format: WireFormat::Pacs008,
            sender_reference,
            uetr,
            value_date,
            currency,
            amount_minor,
            currency_exponent: 2,
            ordering_customer: PartyRef {
                account: doa,
                name: don,
            },
            beneficiary_customer: PartyRef {
                account: ca,
                name: cn,
            },
            sender_bic,
            receiver_bic,
            correspondents: vec![],
            remittance_info: None,
        })
    }

    fn decode_pacs009_xml(xml: &str) -> Result<Self> {
        let sender_reference = xml_extract(xml, "MsgId").unwrap_or_default();
        let uetr = xml_extract(xml, "UETR").unwrap_or_default();
        let sender_bic = xml_path(xml, &["Dbtr", "FinInstnId", "BICFI"]).unwrap_or_default();
        let receiver_bic = xml_path(xml, &["Cdtr", "FinInstnId", "BICFI"]).unwrap_or_default();
        let (currency, amount_minor) = xml_parse_intrbk_amt(xml).unwrap_or_default();
        let value_date = xml_extract(xml, "CreDtTm")
            .map(|s| s.split('T').next().unwrap_or("").to_string())
            .unwrap_or_default();
        Ok(Self {
            format: WireFormat::Pacs009,
            sender_reference,
            uetr,
            value_date,
            currency,
            amount_minor,
            currency_exponent: 2,
            ordering_customer: PartyRef::default(),
            beneficiary_customer: PartyRef::default(),
            sender_bic,
            receiver_bic,
            correspondents: vec![],
            remittance_info: None,
        })
    }
}

// --- helpers ----------------------------------------------------

fn validate_bic(bic: &str, field: &'static str) -> Result<()> {
    if bic.len() != 8 && bic.len() != 11 {
        return Err(Error::FieldRule {
            field,
            reason: format!("BIC must be 8 or 11 chars, got {}", bic.len()),
        });
    }
    if !bic.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Err(Error::FieldRule {
            field,
            reason: "BIC must be ASCII alphanumeric".into(),
        });
    }
    Ok(())
}

fn pad_bic(bic: &str) -> String {
    if bic.len() == 8 {
        format!("{bic}XXX")
    } else {
        bic.to_string()
    }
}

fn format_mt_amount(minor: u64, exponent: u8) -> String {
    // MT amounts use `,` as decimal separator (no thousands sep).
    if exponent == 0 {
        format!("{minor},")
    } else {
        let pow = 10u64.pow(u32::from(exponent));
        let major = minor / pow;
        let minor_part = minor % pow;
        format!("{major},{minor_part:0width$}", width = exponent as usize)
    }
}

fn format_iso_amount(minor: u64, exponent: u8) -> String {
    if exponent == 0 {
        return format!("{minor}");
    }
    let pow = 10u64.pow(u32::from(exponent));
    let major = minor / pow;
    let minor_part = minor % pow;
    format!("{major}.{minor_part:0width$}", width = exponent as usize)
}

fn mt_tag(input: &str, tag: &str) -> Option<String> {
    let start = input.find(tag)? + tag.len();
    let rest = &input[start..];
    let end = rest.find('\n').unwrap_or(rest.len());
    Some(rest[..end].trim().to_string())
}

fn mt_all_tag(input: &str, tag: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut offset = 0;
    while let Some(pos) = input[offset..].find(tag) {
        let s = offset + pos + tag.len();
        let rest = &input[s..];
        let e = rest.find('\n').unwrap_or(rest.len());
        out.push(rest[..e].trim().to_string());
        offset = s + e;
    }
    out
}

fn mt_party(input: &str, tag: &str) -> PartyRef {
    let raw = match mt_tag(input, tag) {
        Some(s) => s,
        None => return PartyRef::default(),
    };
    let account = raw.strip_prefix('/').unwrap_or(&raw).to_string();
    // The line *after* the account holds the name.
    let start = match input.find(tag) {
        Some(p) => p + tag.len(),
        None => return PartyRef { account, name: String::new() },
    };
    let after_first_line = match input[start..].find('\n') {
        Some(p) => start + p + 1,
        None => return PartyRef { account, name: String::new() },
    };
    let name_end = match input[after_first_line..].find('\n') {
        Some(p) => after_first_line + p,
        None => input.len(),
    };
    let name = input[after_first_line..name_end].trim().to_string();
    PartyRef { account, name }
}

fn mt_parse_32a(input: &str) -> Option<(String, String, u64, u8)> {
    let raw = mt_tag(input, ":32A:")?;
    // YYMMDD (6) + CCY (3) + amount with comma decimal.
    if raw.len() < 10 {
        return None;
    }
    let date = raw[..6].to_string();
    let ccy = raw[6..9].to_string();
    let amt = &raw[9..];
    // Default exponent 2 (USD/EUR/GBP); JPY = 0 (no decimal in MT).
    let exponent: u8 = match ccy.as_str() {
        "JPY" => 0,
        _ => 2,
    };
    let (maj_s, min_s) = match amt.split_once(',') {
        Some((a, b)) => (a, b),
        None => (amt, ""),
    };
    let major: u64 = maj_s.parse().ok()?;
    let minor_str = if min_s.is_empty() {
        "0".to_string()
    } else if exponent == 0 {
        "0".into()
    } else if min_s.len() < exponent as usize {
        format!("{min_s:0<width$}", width = exponent as usize)
    } else {
        min_s[..exponent as usize].to_string()
    };
    let minor: u64 = minor_str.parse().ok()?;
    let pow = 10u64.pow(u32::from(exponent));
    Some((date, ccy, major.checked_mul(pow)?.checked_add(minor)?, exponent))
}

fn mt_block3_uetr(input: &str) -> Option<String> {
    let needle = "{121:";
    let s = input.find(needle)? + needle.len();
    let rest = &input[s..];
    let e = rest.find('}')?;
    Some(rest[..e].to_string())
}

fn mt_extract_bic_from_block1(input: &str) -> Option<String> {
    // Block 1: {1:F01XXXXXXYYAAA0000000000} → BIC is chars 4..15
    let needle = "{1:F01";
    let s = input.find(needle)? + needle.len();
    let bic = input.get(s..s + 11)?;
    Some(bic.trim_end_matches('X').to_string())
}

fn mt_extract_bic_from_block2(input: &str) -> Option<String> {
    // Block 2: {2:I103XXXXXXYYAAA N}
    let needle = "{2:I";
    let s = input.find(needle)? + needle.len();
    // 3 digits of app id, then BIC (11 chars).
    let bic = input.get(s + 3..s + 14)?;
    Some(bic.trim_end_matches('X').to_string())
}

fn xml_extract(xml: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let s = xml.find(&open)? + open.len();
    let e = xml[s..].find(&close)? + s;
    Some(xml[s..e].to_string())
}

fn xml_path(xml: &str, path: &[&str]) -> Option<String> {
    let mut cur = xml;
    for tag in path.iter().take(path.len() - 1) {
        let open = format!("<{tag}>");
        let pos = cur.find(&open)? + open.len();
        cur = &cur[pos..];
    }
    let last = path.last()?;
    let open = format!("<{last}>");
    let close = format!("</{last}>");
    let s = cur.find(&open)? + open.len();
    let e = cur[s..].find(&close)? + s;
    Some(cur[s..e].to_string())
}

fn xml_parse_intrbk_amt(xml: &str) -> Option<(String, u64)> {
    let needle = "<IntrBkSttlmAmt";
    let s = xml.find(needle)? + needle.len();
    let close = xml[s..].find('>')? + s;
    let header = &xml[s..close];
    let ccy = header
        .split("Ccy=\"")
        .nth(1)
        .and_then(|t| t.split('"').next())?
        .to_string();
    let end_tag = xml[close..].find("</IntrBkSttlmAmt>")? + close;
    let value = xml[close + 1..end_tag].trim();
    let (maj, min) = value.split_once('.').unwrap_or((value, "0"));
    let major: u64 = maj.parse().ok()?;
    let minor: u64 = min.parse().ok()?;
    Some((ccy, major.checked_mul(100)?.checked_add(minor)?))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mt() -> WireMessage {
        WireMessage {
            format: WireFormat::Mt103,
            sender_reference: "REF12345".into(),
            uetr: "8a562c67-ca16-48ba-b074-65581be6f001".into(),
            value_date: "260601".into(),
            currency: "USD".into(),
            amount_minor: 1_000_000,
            currency_exponent: 2,
            ordering_customer: PartyRef {
                account: "1234567890".into(),
                name: "OPENPAY VENDOR INC".into(),
            },
            beneficiary_customer: PartyRef {
                account: "9876543210".into(),
                name: "ACME CORP".into(),
            },
            sender_bic: "COBADEFF".into(),
            receiver_bic: "CHASUS33".into(),
            correspondents: vec!["BARCGB22".into()],
            remittance_info: Some("Invoice 12345".into()),
        }
    }

    fn pacs() -> WireMessage {
        let mut m = mt();
        m.format = WireFormat::Pacs008;
        m.value_date = "2026-06-01".into();
        m
    }

    #[test]
    fn mt103_round_trip_preserves_amount_and_uetr() {
        let m = mt();
        let wire = m.encode().unwrap();
        assert!(wire.contains(":32A:260601USD10000,00"));
        let parsed = WireMessage::decode(&wire).unwrap();
        assert_eq!(parsed.format, WireFormat::Mt103);
        assert_eq!(parsed.sender_reference, "REF12345");
        assert_eq!(parsed.amount_minor, 1_000_000);
        assert_eq!(parsed.uetr, m.uetr);
        assert_eq!(parsed.beneficiary_customer.name, "ACME CORP");
    }

    #[test]
    fn mt103_jpy_no_decimal() {
        let mut m = mt();
        m.currency = "JPY".into();
        m.currency_exponent = 0;
        m.amount_minor = 50_000;
        let wire = m.encode().unwrap();
        assert!(wire.contains(":32A:260601JPY50000,"));
    }

    #[test]
    fn mt202_carries_receiver_bic() {
        let mut m = mt();
        m.format = WireFormat::Mt202;
        let wire = m.encode().unwrap();
        assert!(wire.contains(":58A:CHASUS33"));
        let parsed = WireMessage::decode(&wire).unwrap();
        assert_eq!(parsed.format, WireFormat::Mt202);
    }

    #[test]
    fn pacs008_round_trip() {
        let m = pacs();
        let xml = m.encode().unwrap();
        assert!(xml.contains("urn:iso:std:iso:20022:tech:xsd:pacs.008.001.12"));
        let parsed = WireMessage::decode(&xml).unwrap();
        assert_eq!(parsed.format, WireFormat::Pacs008);
        assert_eq!(parsed.amount_minor, 1_000_000);
        assert_eq!(parsed.sender_bic, "COBADEFF");
        assert_eq!(parsed.receiver_bic, "CHASUS33");
    }

    #[test]
    fn pacs009_carries_fi_to_fi() {
        let mut m = pacs();
        m.format = WireFormat::Pacs009;
        let xml = m.encode().unwrap();
        assert!(xml.contains("pacs.009.001.10"));
        let parsed = WireMessage::decode(&xml).unwrap();
        assert_eq!(parsed.format, WireFormat::Pacs009);
        assert_eq!(parsed.sender_bic, "COBADEFF");
    }

    #[test]
    fn rejects_bad_bic() {
        let mut m = mt();
        m.sender_bic = "TOOSHORT".into();
        // 8-char is valid. Make it bad:
        m.sender_bic = "ABC".into();
        assert!(m.encode().is_err());
    }

    #[test]
    fn correspondent_chain_preserved_in_mt103() {
        let m = mt();
        let wire = m.encode().unwrap();
        assert!(wire.contains(":56A:BARCGB22"));
    }
}
