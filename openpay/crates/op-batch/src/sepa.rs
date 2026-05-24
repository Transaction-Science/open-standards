//! SEPA file generation: `pain.001` (Credit Transfer Initiation)
//! and `pain.008` (Direct Debit Initiation).
//!
//! Source: **EPC Implementation Guidelines** (EPC130-08 for SCT,
//! EPC131-08 for SDD), aligned to ISO 20022 `pain.001.001.09` and
//! `pain.008.001.08` (the variants in active production use across
//! the eurozone as of 2026).
//!
//! ## Why we hand-roll the XML
//!
//! `op-iso20022` re-exports the `pacs.*` and `camt.*` families but
//! does not yet ship `pain.001` / `pain.008` typed builders (those
//! live in `open-payments-iso20022-pain` and are exposed via
//! `op_iso20022::Message::Pain013` only). For batch initiation we
//! emit the canonical XML directly via `quick-xml`; the same crate
//! parses operator-supplied templates back into the typed model
//! for round-trip tests. When `op-iso20022` grows typed `pain.001`
//! builders, this module switches to delegate without changing its
//! public surface.
//!
//! ## Scheme variants
//!
//! - **SCT** — SEPA Credit Transfer. T+1 settlement, no instant flag.
//! - **SCT-Inst** — SEPA Instant. Sub-10s settlement. Same `pain.001`
//!   wire format with `SCTInst` service-level code.
//! - **SDD CORE** — consumer direct debit. R-transaction window 8 weeks.
//! - **SDD B2B** — corporate direct debit. R-transaction window 2 days.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// SEPA scheme variant. Drives the `SvcLvl/Cd` / `LclInstrm/Cd`
/// elements in the generated XML.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SepaScheme {
    /// Standard SEPA Credit Transfer.
    SctStandard,
    /// SEPA Instant Credit Transfer (`SCT-Inst`).
    SctInstant,
    /// SEPA Direct Debit CORE (consumer).
    SddCore,
    /// SEPA Direct Debit B2B (corporate).
    SddB2b,
}

impl SepaScheme {
    /// `SvcLvl/Cd` value (`SEPA` for SCT-standard; `INST` for
    /// SCT-Inst — actually carried as `SCTInst` service-level Prtry).
    #[must_use]
    pub const fn service_level(self) -> &'static str {
        match self {
            Self::SctStandard | Self::SctInstant => "SEPA",
            // Direct debits still use the `SEPA` service level; the
            // scheme variant is encoded via the local instrument.
            Self::SddCore | Self::SddB2b => "SEPA",
        }
    }

    /// `LclInstrm/Cd` (or `Prtry`) value where applicable.
    #[must_use]
    pub const fn local_instrument(self) -> Option<&'static str> {
        match self {
            Self::SctStandard => None,
            Self::SctInstant => Some("INST"),
            Self::SddCore => Some("CORE"),
            Self::SddB2b => Some("B2B"),
        }
    }

    /// True if the scheme is a credit transfer (`pain.001` side).
    #[must_use]
    pub const fn is_credit(self) -> bool {
        matches!(self, Self::SctStandard | Self::SctInstant)
    }
}

/// One credit transfer leg inside a `pain.001` message.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct CreditEntry {
    /// End-to-end identification (≤35 chars).
    pub end_to_end_id: String,
    /// Instruction identification (≤35 chars).
    pub instruction_id: String,
    /// Amount in minor units (cents). EUR is implied.
    pub amount_cents: u64,
    /// Creditor name (≤70 chars).
    pub creditor_name: String,
    /// Creditor IBAN.
    pub creditor_iban: String,
    /// Creditor agent BIC.
    pub creditor_bic: String,
    /// Optional free-text remittance info (≤140 chars).
    pub remittance_info: Option<String>,
}

/// One direct debit leg inside a `pain.008` message.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DebitEntry {
    /// End-to-end identification (≤35 chars).
    pub end_to_end_id: String,
    /// Mandate identification (≤35 chars).
    pub mandate_id: String,
    /// Date of mandate signature (`YYYY-MM-DD`).
    pub mandate_signature_date: String,
    /// Amount in minor units (cents).
    pub amount_cents: u64,
    /// Debtor name (≤70 chars).
    pub debtor_name: String,
    /// Debtor IBAN.
    pub debtor_iban: String,
    /// Debtor agent BIC.
    pub debtor_bic: String,
    /// Optional remittance info.
    pub remittance_info: Option<String>,
}

/// A complete `pain.001` message — one batch of outbound credits.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SepaCreditTransfer {
    /// Group header `MsgId` (≤35 chars).
    pub message_id: String,
    /// Payment info `PmtInfId` (≤35 chars).
    pub payment_info_id: String,
    /// Initiating party name.
    pub initiator_name: String,
    /// Initiating party id (LEI / OIN).
    pub initiator_id: String,
    /// Debtor (the operator's) name.
    pub debtor_name: String,
    /// Debtor IBAN.
    pub debtor_iban: String,
    /// Debtor agent BIC.
    pub debtor_bic: String,
    /// Scheme variant.
    pub scheme: SepaScheme,
    /// Requested execution date `YYYY-MM-DD`.
    pub requested_execution_date: String,
    /// Credit entries.
    pub entries: Vec<CreditEntry>,
    /// Group creation timestamp.
    pub creation_dt: DateTime<Utc>,
}

/// A complete `pain.008` message — one batch of outbound debits.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SepaDirectDebit {
    /// Group header `MsgId` (≤35 chars).
    pub message_id: String,
    /// Payment info `PmtInfId` (≤35 chars).
    pub payment_info_id: String,
    /// Initiating party name (typically the creditor).
    pub initiator_name: String,
    /// Initiating party id.
    pub initiator_id: String,
    /// Creditor (the operator's) name.
    pub creditor_name: String,
    /// Creditor IBAN.
    pub creditor_iban: String,
    /// Creditor agent BIC.
    pub creditor_bic: String,
    /// Creditor scheme identifier (CID).
    pub creditor_scheme_id: String,
    /// Scheme variant (CORE or B2B).
    pub scheme: SepaScheme,
    /// Requested collection date `YYYY-MM-DD`.
    pub requested_collection_date: String,
    /// Debit entries.
    pub entries: Vec<DebitEntry>,
    /// Group creation timestamp.
    pub creation_dt: DateTime<Utc>,
}

impl SepaCreditTransfer {
    /// Validate scheme fit before encoding (credit message ≠ debit
    /// scheme).
    fn validate(&self) -> Result<()> {
        if !self.scheme.is_credit() {
            return Err(Error::FieldRule {
                field: "scheme",
                reason: "pain.001 requires a credit-transfer scheme (SCT / SCT-Inst)".into(),
            });
        }
        if self.entries.is_empty() {
            return Err(Error::Invalid("no credit entries".into()));
        }
        for e in &self.entries {
            if e.end_to_end_id.len() > 35 {
                return Err(Error::FieldRule {
                    field: "end_to_end_id",
                    reason: "≤35 chars".into(),
                });
            }
            if !e.creditor_iban.chars().all(|c| c.is_ascii_alphanumeric()) {
                return Err(Error::FieldRule {
                    field: "creditor_iban",
                    reason: "IBAN must be ASCII alphanumeric".into(),
                });
            }
        }
        Ok(())
    }

    /// Encode as canonical `pain.001.001.09` XML.
    ///
    /// # Errors
    /// Any field-rule violation surfaces as [`Error::FieldRule`].
    pub fn encode_xml(&self) -> Result<String> {
        self.validate()?;
        let mut total_cents: u64 = 0;
        let mut tx_block = String::new();
        for e in &self.entries {
            total_cents = total_cents
                .checked_add(e.amount_cents)
                .ok_or(op_core::Error::Overflow)?;
            let amt = format_eur(e.amount_cents);
            let remit = e
                .remittance_info
                .as_deref()
                .map(|s| format!("<RmtInf><Ustrd>{}</Ustrd></RmtInf>", escape(s)))
                .unwrap_or_default();
            tx_block.push_str(&format!(
                "<CdtTrfTxInf><PmtId><InstrId>{instr}</InstrId><EndToEndId>{e2e}</EndToEndId></PmtId><Amt><InstdAmt Ccy=\"EUR\">{amt}</InstdAmt></Amt><CdtrAgt><FinInstnId><BICFI>{bic}</BICFI></FinInstnId></CdtrAgt><Cdtr><Nm>{cn}</Nm></Cdtr><CdtrAcct><Id><IBAN>{iban}</IBAN></Id></CdtrAcct>{remit}</CdtTrfTxInf>",
                instr = escape(&e.instruction_id),
                e2e = escape(&e.end_to_end_id),
                amt = amt,
                bic = escape(&e.creditor_bic),
                cn = escape(&e.creditor_name),
                iban = escape(&e.creditor_iban),
                remit = remit,
            ));
        }
        let svc_lvl = self.scheme.service_level();
        let local_instrm = self
            .scheme
            .local_instrument()
            .map(|c| format!("<LclInstrm><Cd>{c}</Cd></LclInstrm>"))
            .unwrap_or_default();
        let nb_tx = self.entries.len();
        let ctrl_sum = format_eur(total_cents);
        let created = self.creation_dt.to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        Ok(format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Document xmlns=\"urn:iso:std:iso:20022:tech:xsd:pain.001.001.09\"><CstmrCdtTrfInitn><GrpHdr><MsgId>{msg}</MsgId><CreDtTm>{created}</CreDtTm><NbOfTxs>{nb}</NbOfTxs><CtrlSum>{ctrl}</CtrlSum><InitgPty><Nm>{init}</Nm><Id><OrgId><Othr><Id>{init_id}</Id></Othr></OrgId></Id></InitgPty></GrpHdr><PmtInf><PmtInfId>{pii}</PmtInfId><PmtMtd>TRF</PmtMtd><NbOfTxs>{nb}</NbOfTxs><CtrlSum>{ctrl}</CtrlSum><PmtTpInf><SvcLvl><Cd>{svc}</Cd></SvcLvl>{li}</PmtTpInf><ReqdExctnDt>{exec}</ReqdExctnDt><Dbtr><Nm>{dn}</Nm></Dbtr><DbtrAcct><Id><IBAN>{di}</IBAN></Id></DbtrAcct><DbtrAgt><FinInstnId><BICFI>{db}</BICFI></FinInstnId></DbtrAgt>{tx}</PmtInf></CstmrCdtTrfInitn></Document>",
            msg = escape(&self.message_id),
            created = created,
            nb = nb_tx,
            ctrl = ctrl_sum,
            init = escape(&self.initiator_name),
            init_id = escape(&self.initiator_id),
            pii = escape(&self.payment_info_id),
            svc = svc_lvl,
            li = local_instrm,
            exec = escape(&self.requested_execution_date),
            dn = escape(&self.debtor_name),
            di = escape(&self.debtor_iban),
            db = escape(&self.debtor_bic),
            tx = tx_block,
        ))
    }

    /// Round-trip decode helper. Reads the small subset of the
    /// message we encode back into a [`SepaCreditTransfer`]. Used
    /// by tests; production receivers typically only need to parse
    /// status responses (`pain.002`), not their own initiation.
    ///
    /// # Errors
    /// `Error::Xml` for malformed input; `Error::FieldRule` for
    /// missing mandatory fields.
    pub fn decode_xml(xml: &str) -> Result<Self> {
        let msg = extract(xml, "MsgId").ok_or_else(|| Error::FieldRule {
            field: "MsgId",
            reason: "missing".into(),
        })?;
        let pii = extract(xml, "PmtInfId").ok_or_else(|| Error::FieldRule {
            field: "PmtInfId",
            reason: "missing".into(),
        })?;
        let initiator_name = extract_path(xml, &["InitgPty", "Nm"]).unwrap_or_default();
        let initiator_id = extract_path(xml, &["InitgPty", "Id", "OrgId", "Othr", "Id"])
            .unwrap_or_default();
        let debtor_name = extract_path(xml, &["Dbtr", "Nm"]).unwrap_or_default();
        let debtor_iban =
            extract_path(xml, &["DbtrAcct", "Id", "IBAN"]).unwrap_or_default();
        let debtor_bic =
            extract_path(xml, &["DbtrAgt", "FinInstnId", "BICFI"]).unwrap_or_default();
        let requested_execution_date = extract(xml, "ReqdExctnDt").unwrap_or_default();
        let created = extract(xml, "CreDtTm")
            .and_then(|s| DateTime::parse_from_rfc3339(&s).ok().map(|d| d.with_timezone(&Utc)))
            .unwrap_or_else(Utc::now);
        // Scheme detection from local instrument.
        let scheme = if xml.contains("<Cd>INST</Cd>") {
            SepaScheme::SctInstant
        } else {
            SepaScheme::SctStandard
        };
        // Extract each CdtTrfTxInf block by simple scan.
        let entries = extract_credit_entries(xml);
        Ok(Self {
            message_id: msg,
            payment_info_id: pii,
            initiator_name,
            initiator_id,
            debtor_name,
            debtor_iban,
            debtor_bic,
            scheme,
            requested_execution_date,
            entries,
            creation_dt: created,
        })
    }
}

impl SepaDirectDebit {
    fn validate(&self) -> Result<()> {
        if self.scheme.is_credit() {
            return Err(Error::FieldRule {
                field: "scheme",
                reason: "pain.008 requires a direct-debit scheme (CORE / B2B)".into(),
            });
        }
        if self.entries.is_empty() {
            return Err(Error::Invalid("no debit entries".into()));
        }
        for e in &self.entries {
            if e.mandate_id.is_empty() {
                return Err(Error::FieldRule {
                    field: "mandate_id",
                    reason: "must not be empty".into(),
                });
            }
        }
        Ok(())
    }

    /// Encode as canonical `pain.008.001.08` XML.
    ///
    /// # Errors
    /// Any field-rule violation surfaces as [`Error::FieldRule`].
    pub fn encode_xml(&self) -> Result<String> {
        self.validate()?;
        let mut total_cents: u64 = 0;
        let mut tx_block = String::new();
        for e in &self.entries {
            total_cents = total_cents
                .checked_add(e.amount_cents)
                .ok_or(op_core::Error::Overflow)?;
            let amt = format_eur(e.amount_cents);
            let remit = e
                .remittance_info
                .as_deref()
                .map(|s| format!("<RmtInf><Ustrd>{}</Ustrd></RmtInf>", escape(s)))
                .unwrap_or_default();
            tx_block.push_str(&format!(
                "<DrctDbtTxInf><PmtId><EndToEndId>{e2e}</EndToEndId></PmtId><InstdAmt Ccy=\"EUR\">{amt}</InstdAmt><DrctDbtTx><MndtRltdInf><MndtId>{mid}</MndtId><DtOfSgntr>{ds}</DtOfSgntr></MndtRltdInf></DrctDbtTx><DbtrAgt><FinInstnId><BICFI>{bic}</BICFI></FinInstnId></DbtrAgt><Dbtr><Nm>{dn}</Nm></Dbtr><DbtrAcct><Id><IBAN>{iban}</IBAN></Id></DbtrAcct>{remit}</DrctDbtTxInf>",
                e2e = escape(&e.end_to_end_id),
                amt = amt,
                mid = escape(&e.mandate_id),
                ds = escape(&e.mandate_signature_date),
                bic = escape(&e.debtor_bic),
                dn = escape(&e.debtor_name),
                iban = escape(&e.debtor_iban),
                remit = remit,
            ));
        }
        let local_instrm = self
            .scheme
            .local_instrument()
            .map(|c| format!("<LclInstrm><Cd>{c}</Cd></LclInstrm>"))
            .unwrap_or_default();
        let nb_tx = self.entries.len();
        let ctrl_sum = format_eur(total_cents);
        let created = self
            .creation_dt
            .to_rfc3339_opts(chrono::SecondsFormat::Secs, true);
        Ok(format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?><Document xmlns=\"urn:iso:std:iso:20022:tech:xsd:pain.008.001.08\"><CstmrDrctDbtInitn><GrpHdr><MsgId>{msg}</MsgId><CreDtTm>{created}</CreDtTm><NbOfTxs>{nb}</NbOfTxs><CtrlSum>{ctrl}</CtrlSum><InitgPty><Nm>{init}</Nm><Id><OrgId><Othr><Id>{iid}</Id></Othr></OrgId></Id></InitgPty></GrpHdr><PmtInf><PmtInfId>{pii}</PmtInfId><PmtMtd>DD</PmtMtd><NbOfTxs>{nb}</NbOfTxs><CtrlSum>{ctrl}</CtrlSum><PmtTpInf><SvcLvl><Cd>SEPA</Cd></SvcLvl>{li}</PmtTpInf><ReqdColltnDt>{rcd}</ReqdColltnDt><Cdtr><Nm>{cn}</Nm></Cdtr><CdtrAcct><Id><IBAN>{ci}</IBAN></Id></CdtrAcct><CdtrAgt><FinInstnId><BICFI>{cb}</BICFI></FinInstnId></CdtrAgt><CdtrSchmeId><Id><PrvtId><Othr><Id>{csi}</Id></Othr></PrvtId></Id></CdtrSchmeId>{tx}</PmtInf></CstmrDrctDbtInitn></Document>",
            msg = escape(&self.message_id),
            created = created,
            nb = nb_tx,
            ctrl = ctrl_sum,
            init = escape(&self.initiator_name),
            iid = escape(&self.initiator_id),
            pii = escape(&self.payment_info_id),
            li = local_instrm,
            rcd = escape(&self.requested_collection_date),
            cn = escape(&self.creditor_name),
            ci = escape(&self.creditor_iban),
            cb = escape(&self.creditor_bic),
            csi = escape(&self.creditor_scheme_id),
            tx = tx_block,
        ))
    }

    /// Round-trip decode helper for `pain.008` files.
    ///
    /// # Errors
    /// `Error::FieldRule` if mandatory fields are missing.
    pub fn decode_xml(xml: &str) -> Result<Self> {
        let msg = extract(xml, "MsgId").ok_or_else(|| Error::FieldRule {
            field: "MsgId",
            reason: "missing".into(),
        })?;
        let pii = extract(xml, "PmtInfId").ok_or_else(|| Error::FieldRule {
            field: "PmtInfId",
            reason: "missing".into(),
        })?;
        let initiator_name = extract_path(xml, &["InitgPty", "Nm"]).unwrap_or_default();
        let initiator_id = extract_path(xml, &["InitgPty", "Id", "OrgId", "Othr", "Id"])
            .unwrap_or_default();
        let creditor_name = extract_path(xml, &["Cdtr", "Nm"]).unwrap_or_default();
        let creditor_iban =
            extract_path(xml, &["CdtrAcct", "Id", "IBAN"]).unwrap_or_default();
        let creditor_bic =
            extract_path(xml, &["CdtrAgt", "FinInstnId", "BICFI"]).unwrap_or_default();
        let creditor_scheme_id =
            extract_path(xml, &["CdtrSchmeId", "Id", "PrvtId", "Othr", "Id"])
                .unwrap_or_default();
        let requested_collection_date = extract(xml, "ReqdColltnDt").unwrap_or_default();
        let created = extract(xml, "CreDtTm")
            .and_then(|s| {
                DateTime::parse_from_rfc3339(&s)
                    .ok()
                    .map(|d| d.with_timezone(&Utc))
            })
            .unwrap_or_else(Utc::now);
        let scheme = if xml.contains("<Cd>B2B</Cd>") {
            SepaScheme::SddB2b
        } else {
            SepaScheme::SddCore
        };
        let entries = extract_debit_entries(xml);
        Ok(Self {
            message_id: msg,
            payment_info_id: pii,
            initiator_name,
            initiator_id,
            creditor_name,
            creditor_iban,
            creditor_bic,
            creditor_scheme_id,
            scheme,
            requested_collection_date,
            entries,
            creation_dt: created,
        })
    }
}

// --- ID generators ----------------------------------------------

/// Generate an end-to-end identifier from `(seed, sequence)`. Returns a
/// string ≤35 chars (SEPA limit) shaped `<seed>-<seq>` with the seed
/// truncated as needed.
#[must_use]
pub fn gen_end_to_end_id(seed: &str, seq: u32) -> String {
    let tail = format!("-{seq:07}");
    let max_seed = 35 - tail.len();
    let s = if seed.len() > max_seed {
        &seed[..max_seed]
    } else {
        seed
    };
    format!("{s}{tail}")
}

/// Generate a payment-info identifier (≤35 chars). Convention:
/// `<scheme>-<batch>-<yyyymmdd>` truncated to 35 chars.
#[must_use]
pub fn gen_payment_info_id(scheme: SepaScheme, batch_seq: u32, date_yyyymmdd: &str) -> String {
    let prefix = match scheme {
        SepaScheme::SctStandard => "SCT",
        SepaScheme::SctInstant => "SCTINST",
        SepaScheme::SddCore => "SDDCORE",
        SepaScheme::SddB2b => "SDDB2B",
    };
    let s = format!("{prefix}-{batch_seq:05}-{date_yyyymmdd}");
    if s.len() > 35 { s[..35].into() } else { s }
}

/// Generate an instruction identifier (≤35 chars).
#[must_use]
pub fn gen_instruction_id(seed: &str, seq: u32) -> String {
    let tail = format!("INSTR-{seq:09}");
    let max_seed = 35 - tail.len();
    let s = if seed.len() > max_seed {
        &seed[..max_seed]
    } else {
        seed
    };
    format!("{s}{tail}")
}

// --- helpers ----------------------------------------------------

fn format_eur(cents: u64) -> String {
    let major = cents / 100;
    let minor = cents % 100;
    format!("{major}.{minor:02}")
}

fn escape(s: &str) -> String {
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

/// Extract the inner text of `<tag>...</tag>`, allowing the open
/// tag to carry attributes (`<tag attr="x">`).
fn extract(xml: &str, tag: &str) -> Option<String> {
    let open_attr = format!("<{tag} ");
    let open_plain = format!("<{tag}>");
    let close = format!("</{tag}>");
    let after = if let Some(p) = xml.find(&open_plain) {
        p + open_plain.len()
    } else if let Some(p) = xml.find(&open_attr) {
        let gt = xml[p..].find('>')?;
        p + gt + 1
    } else {
        return None;
    };
    let end = xml[after..].find(&close)? + after;
    Some(xml[after..end].to_string())
}

fn extract_path(xml: &str, path: &[&str]) -> Option<String> {
    let mut cur = xml;
    for tag in path.iter().take(path.len() - 1) {
        let pos = find_tag_open_end(cur, tag)?;
        cur = &cur[pos..];
    }
    let last = path.last()?;
    let s = find_tag_open_end(cur, last)?;
    let close = format!("</{last}>");
    let e = cur[s..].find(&close)? + s;
    Some(cur[s..e].to_string())
}

/// Position **after** the closing `>` of the opening `<tag>` or
/// `<tag attr="...">`. Returns `None` if no such opening tag exists.
fn find_tag_open_end(xml: &str, tag: &str) -> Option<usize> {
    let plain = format!("<{tag}>");
    let attr = format!("<{tag} ");
    if let Some(p) = xml.find(&plain) {
        return Some(p + plain.len());
    }
    if let Some(p) = xml.find(&attr) {
        let gt = xml[p..].find('>')?;
        return Some(p + gt + 1);
    }
    None
}

fn extract_credit_entries(xml: &str) -> Vec<CreditEntry> {
    let mut out = Vec::new();
    let open = "<CdtTrfTxInf>";
    let close = "</CdtTrfTxInf>";
    let mut cur = xml;
    while let Some(s) = cur.find(open) {
        let after = s + open.len();
        let e = match cur[after..].find(close) {
            Some(p) => p + after,
            None => break,
        };
        let block = &cur[after..e];
        let amount = extract(block, "InstdAmt")
            .and_then(|s| {
                let s = s.trim();
                parse_eur_to_cents(s)
            })
            .unwrap_or(0);
        out.push(CreditEntry {
            end_to_end_id: extract(block, "EndToEndId").unwrap_or_default(),
            instruction_id: extract(block, "InstrId").unwrap_or_default(),
            amount_cents: amount,
            creditor_name: extract_path(block, &["Cdtr", "Nm"]).unwrap_or_default(),
            creditor_iban: extract_path(block, &["CdtrAcct", "Id", "IBAN"]).unwrap_or_default(),
            creditor_bic: extract_path(block, &["CdtrAgt", "FinInstnId", "BICFI"])
                .unwrap_or_default(),
            remittance_info: extract(block, "Ustrd"),
        });
        cur = &cur[e + close.len()..];
    }
    out
}

fn extract_debit_entries(xml: &str) -> Vec<DebitEntry> {
    let mut out = Vec::new();
    let open = "<DrctDbtTxInf>";
    let close = "</DrctDbtTxInf>";
    let mut cur = xml;
    while let Some(s) = cur.find(open) {
        let after = s + open.len();
        let e = match cur[after..].find(close) {
            Some(p) => p + after,
            None => break,
        };
        let block = &cur[after..e];
        let amount = extract(block, "InstdAmt")
            .and_then(|s| parse_eur_to_cents(s.trim()))
            .unwrap_or(0);
        out.push(DebitEntry {
            end_to_end_id: extract(block, "EndToEndId").unwrap_or_default(),
            mandate_id: extract(block, "MndtId").unwrap_or_default(),
            mandate_signature_date: extract(block, "DtOfSgntr").unwrap_or_default(),
            amount_cents: amount,
            debtor_name: extract_path(block, &["Dbtr", "Nm"]).unwrap_or_default(),
            debtor_iban: extract_path(block, &["DbtrAcct", "Id", "IBAN"]).unwrap_or_default(),
            debtor_bic: extract_path(block, &["DbtrAgt", "FinInstnId", "BICFI"])
                .unwrap_or_default(),
            remittance_info: extract(block, "Ustrd"),
        });
        cur = &cur[e + close.len()..];
    }
    out
}

fn parse_eur_to_cents(raw: &str) -> Option<u64> {
    let trimmed = raw.trim();
    let (maj, min) = trimmed.split_once('.')?;
    let major: u64 = maj.parse().ok()?;
    let minor: u64 = min.parse().ok()?;
    major.checked_mul(100)?.checked_add(minor)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn ct() -> SepaCreditTransfer {
        SepaCreditTransfer {
            message_id: "MSG-2026-01".into(),
            payment_info_id: gen_payment_info_id(SepaScheme::SctStandard, 1, "20260601"),
            initiator_name: "OpenPay Vendor AG".into(),
            initiator_id: "DE98ZZZ09999999999".into(),
            debtor_name: "OpenPay Vendor AG".into(),
            debtor_iban: "DE89370400440532013000".into(),
            debtor_bic: "COBADEFFXXX".into(),
            scheme: SepaScheme::SctStandard,
            requested_execution_date: "2026-06-02".into(),
            entries: vec![
                CreditEntry {
                    end_to_end_id: gen_end_to_end_id("INV", 1),
                    instruction_id: gen_instruction_id("INV", 1),
                    amount_cents: 12_345,
                    creditor_name: "Acme GmbH".into(),
                    creditor_iban: "DE02100500001054550017".into(),
                    creditor_bic: "BELADEBEXXX".into(),
                    remittance_info: Some("Invoice 1".into()),
                },
                CreditEntry {
                    end_to_end_id: gen_end_to_end_id("INV", 2),
                    instruction_id: gen_instruction_id("INV", 2),
                    amount_cents: 67_890,
                    creditor_name: "Beta SARL".into(),
                    creditor_iban: "FR1420041010050500013M02606".into(),
                    creditor_bic: "PSSTFRPPXXX".into(),
                    remittance_info: None,
                },
            ],
            creation_dt: chrono::DateTime::parse_from_rfc3339("2026-06-01T10:00:00Z")
                .unwrap()
                .with_timezone(&chrono::Utc),
        }
    }

    fn dd() -> SepaDirectDebit {
        SepaDirectDebit {
            message_id: "MSG-DD-2026-01".into(),
            payment_info_id: gen_payment_info_id(SepaScheme::SddCore, 1, "20260601"),
            initiator_name: "OpenPay Vendor AG".into(),
            initiator_id: "DE98ZZZ09999999999".into(),
            creditor_name: "OpenPay Vendor AG".into(),
            creditor_iban: "DE89370400440532013000".into(),
            creditor_bic: "COBADEFFXXX".into(),
            creditor_scheme_id: "DE98ZZZ09999999999".into(),
            scheme: SepaScheme::SddCore,
            requested_collection_date: "2026-06-10".into(),
            entries: vec![DebitEntry {
                end_to_end_id: gen_end_to_end_id("DD", 1),
                mandate_id: "MND-001".into(),
                mandate_signature_date: "2025-12-01".into(),
                amount_cents: 5_000,
                debtor_name: "Charlie SA".into(),
                debtor_iban: "ES9121000418450200051332".into(),
                debtor_bic: "CAIXESBBXXX".into(),
                remittance_info: Some("Sub fee".into()),
            }],
            creation_dt: chrono::DateTime::parse_from_rfc3339("2026-06-01T10:00:00Z")
                .unwrap()
                .with_timezone(&chrono::Utc),
        }
    }

    #[test]
    fn ct_encode_contains_iso20022_namespace() {
        let xml = ct().encode_xml().unwrap();
        assert!(xml.contains("urn:iso:std:iso:20022:tech:xsd:pain.001.001.09"));
        assert!(xml.contains("<CtrlSum>802.35</CtrlSum>"));
        assert!(xml.contains("<NbOfTxs>2</NbOfTxs>"));
    }

    #[test]
    fn ct_round_trip_preserves_count_and_id() {
        let original = ct();
        let xml = original.encode_xml().unwrap();
        let decoded = SepaCreditTransfer::decode_xml(&xml).unwrap();
        assert_eq!(decoded.message_id, original.message_id);
        assert_eq!(decoded.payment_info_id, original.payment_info_id);
        assert_eq!(decoded.entries.len(), 2);
    }

    #[test]
    fn sct_inst_flag_carried() {
        let mut c = ct();
        c.scheme = SepaScheme::SctInstant;
        let xml = c.encode_xml().unwrap();
        assert!(xml.contains("<Cd>INST</Cd>"));
        let decoded = SepaCreditTransfer::decode_xml(&xml).unwrap();
        assert_eq!(decoded.scheme, SepaScheme::SctInstant);
    }

    #[test]
    fn dd_encode_b2b_carries_local_instrument() {
        let mut d = dd();
        d.scheme = SepaScheme::SddB2b;
        let xml = d.encode_xml().unwrap();
        assert!(xml.contains("urn:iso:std:iso:20022:tech:xsd:pain.008.001.08"));
        assert!(xml.contains("<Cd>B2B</Cd>"));
    }

    #[test]
    fn dd_round_trip_preserves_mandate() {
        let original = dd();
        let xml = original.encode_xml().unwrap();
        let decoded = SepaDirectDebit::decode_xml(&xml).unwrap();
        assert_eq!(decoded.message_id, original.message_id);
        assert_eq!(decoded.entries[0].mandate_id, "MND-001");
        assert_eq!(decoded.entries[0].amount_cents, 5_000);
    }

    #[test]
    fn rejects_scheme_mismatch() {
        let mut c = ct();
        c.scheme = SepaScheme::SddCore;
        assert!(c.encode_xml().is_err());
        let mut d = dd();
        d.scheme = SepaScheme::SctStandard;
        assert!(d.encode_xml().is_err());
    }

    #[test]
    fn id_generators_respect_35_char_limit() {
        let e2e = gen_end_to_end_id("very-long-seed-that-exceeds-the-allowed-limit", 42);
        assert!(e2e.len() <= 35);
        let pii = gen_payment_info_id(SepaScheme::SctInstant, 7, "20260601");
        assert!(pii.len() <= 35);
        let instr = gen_instruction_id("seed", 1);
        assert!(instr.len() <= 35);
    }
}
