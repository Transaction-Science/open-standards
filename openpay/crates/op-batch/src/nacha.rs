//! NACHA ACH file format.
//!
//! Source of truth: **NACHA Operating Rules & Guidelines, 2024**,
//! Appendix Three (ACH File Specifications).
//!
//! ## Wire layout
//!
//! A NACHA file is an ordered sequence of **94-character** ASCII
//! records, terminated `\n` (some ODFIs request `\r\n` — operators
//! post-process). Five record types appear in order:
//!
//! | Code | Name | Notes |
//! |-----|------|-------|
//! | `1` | File Header | one per file |
//! | `5` | Batch Header | one per batch |
//! | `6` | Entry Detail | one per credit / debit |
//! | `7` | Addenda | optional, follows the entry it amplifies |
//! | `8` | Batch Control | closes a batch |
//! | `9` | File Control | closes the file |
//!
//! Plus filler records (94 `9`s) padding to a multiple of 10 records.
//!
//! This module is a **superset** of the [`op_settlement::nacha`]
//! reference generator: it adds full SEC code coverage (`CCD` /
//! `PPD` / `WEB` / `TEL` / `CTX`), the `7` addenda record,
//! same-day cutoff helpers, and the return-code (`R01`..`R99`)
//! parser used by [`crate::exception`].

use chrono::{DateTime, Datelike, Timelike, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// NACHA record length. Fixed by the spec; never changes.
pub const RECORD_LEN: usize = 94;

/// Standard Entry Class. Tells the RDFI how to interpret the entry
/// and which consumer-protection rules apply.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SecCode {
    /// Prearranged Payment and Deposit — consumer.
    Ppd,
    /// Cash Concentration and Disbursement — corporate.
    Ccd,
    /// Internet-Initiated entry.
    Web,
    /// Telephone-Initiated entry.
    Tel,
    /// Corporate Trade Exchange — corporate with structured addenda.
    Ctx,
}

impl SecCode {
    /// Three-letter code as written in the batch header.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Ppd => "PPD",
            Self::Ccd => "CCD",
            Self::Web => "WEB",
            Self::Tel => "TEL",
            Self::Ctx => "CTX",
        }
    }

    /// Parse from a three-byte slice (as it appears in `5` records).
    ///
    /// # Errors
    /// `Error::FieldRule` if the code is not a known SEC.
    pub fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "PPD" => Self::Ppd,
            "CCD" => Self::Ccd,
            "WEB" => Self::Web,
            "TEL" => Self::Tel,
            "CTX" => Self::Ctx,
            other => {
                return Err(Error::FieldRule {
                    field: "sec",
                    reason: format!("unknown SEC code `{other}`"),
                });
            }
        })
    }
}

/// File header (`1` record).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct FileHeader {
    /// Immediate destination (10 chars, conventionally space + 9-digit
    /// ABA routing of the receiving point).
    pub immediate_destination: String,
    /// Immediate origin (10 chars).
    pub immediate_origin: String,
    /// File creation date `YYMMDD`.
    pub file_creation_date: String,
    /// File creation time `HHMM`.
    pub file_creation_time: String,
    /// File id modifier (`A`..`Z` / `0`..`9`).
    pub file_id_modifier: char,
    /// Free-text destination name (≤23 chars).
    pub immediate_destination_name: String,
    /// Free-text origin name (≤23 chars).
    pub immediate_origin_name: String,
    /// Reference code (≤8 chars).
    pub reference_code: String,
}

/// Batch header (`5` record).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BatchHeader {
    /// Service class code: `200` mixed, `220` credits only, `225`
    /// debits only.
    pub service_class: u16,
    /// Company name (≤16 chars) — appears on receiver bank statements.
    pub company_name: String,
    /// Company discretionary data (≤20 chars).
    pub company_discretionary: String,
    /// Company id (10 chars).
    pub company_id: String,
    /// Standard Entry Class.
    pub sec: SecCode,
    /// Company entry description (≤10 chars).
    pub company_entry_description: String,
    /// Company descriptive date `YYMMDD`.
    pub descriptive_date: String,
    /// Effective entry date `YYMMDD` — when funds should settle.
    pub effective_entry_date: String,
    /// Originator status code (almost always `1`).
    pub originator_status: char,
    /// First 8 digits of the ODFI routing number.
    pub odfi_short: String,
    /// Batch number (1-based, monotonic within a file).
    pub batch_number: u32,
}

/// Entry detail (`6` record).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct EntryDetail {
    /// Transaction code. Common values:
    /// `22` checking credit, `27` checking debit,
    /// `32` savings credit, `37` savings debit.
    pub transaction_code: u8,
    /// RDFI routing number (9 digits).
    pub rdfi_routing: String,
    /// Receiver account number (≤17 chars).
    pub account_number: String,
    /// Amount in cents (positive; sign comes from the txn code).
    pub amount_cents: u64,
    /// Individual identification (≤15 chars).
    pub individual_id: String,
    /// Receiver name (≤22 chars).
    pub receiver_name: String,
    /// Discretionary data (≤2 chars).
    pub discretionary: String,
    /// Addenda record indicator (`0` = none, `1` = follows).
    pub addenda_indicator: char,
    /// Trace number (15 chars: 8-digit ODFI + 7-digit sequence).
    pub trace_number: String,
}

/// Addenda record (`7` record). Carries free-text or structured
/// remittance info for the preceding `6` entry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Addenda {
    /// Addenda type code (`05` = remittance, `99` = return, etc.).
    pub type_code: String,
    /// Payment-related information (≤80 chars).
    pub payment_related_info: String,
    /// Addenda sequence number (1-based within an entry).
    pub sequence_number: u16,
    /// Entry detail sequence number (trace tail).
    pub entry_detail_sequence: u32,
}

/// One batch within a NACHA file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NachaBatch {
    /// The `5` record.
    pub header: BatchHeader,
    /// `6`/`7` records.
    pub entries: Vec<EntryDetail>,
    /// Addenda keyed by entry index (only entries with
    /// `addenda_indicator == '1'` get one).
    pub addenda: Vec<Option<Addenda>>,
}

/// A complete NACHA file — what gets dropped into the SFTP folder.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NachaFile {
    /// `1` record.
    pub header: FileHeader,
    /// One or more batches.
    pub batches: Vec<NachaBatch>,
}

/// Operator-static parameters for header construction.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NachaProfile {
    /// 9-digit ODFI routing number.
    pub odfi_routing: String,
    /// Immediate origin (10 chars).
    pub immediate_origin: String,
    /// Immediate destination (10 chars).
    pub immediate_destination: String,
    /// Operator's company name.
    pub company_name: String,
    /// Operator's company id (10 chars).
    pub company_id: String,
    /// Entry description (≤10 chars).
    pub company_entry_description: String,
}

impl NachaFile {
    /// Encode the in-memory model into the on-wire byte stream
    /// (94-char records + filler padding).
    ///
    /// # Errors
    /// Any field-rule violation surfaces as [`Error::FieldRule`] or
    /// [`Error::RecordLength`].
    pub fn encode(&self) -> Result<String> {
        let mut out = String::new();
        push_line(&mut out, &encode_file_header(&self.header)?)?;
        let mut total_entries: u32 = 0;
        let mut total_hash: u64 = 0;
        let mut total_debit: u64 = 0;
        let mut total_credit: u64 = 0;
        let mut records: u32 = 1;
        for (i, batch) in self.batches.iter().enumerate() {
            let batch_no = u32::try_from(i + 1).map_err(|_| Error::Invalid("too many batches".into()))?;
            push_line(&mut out, &encode_batch_header(&batch.header, batch_no)?)?;
            records += 1;
            let mut entry_hash: u64 = 0;
            let mut debit: u64 = 0;
            let mut credit: u64 = 0;
            for (idx, entry) in batch.entries.iter().enumerate() {
                validate_entry(entry)?;
                push_line(&mut out, &encode_entry(entry)?)?;
                records += 1;
                let rdfi_8 = &entry.rdfi_routing[..8];
                entry_hash = entry_hash.wrapping_add(parse_u64(rdfi_8)?);
                if is_credit(entry.transaction_code) {
                    credit = credit.checked_add(entry.amount_cents)
                        .ok_or(op_core::Error::Overflow)?;
                } else {
                    debit = debit.checked_add(entry.amount_cents)
                        .ok_or(op_core::Error::Overflow)?;
                }
                if entry.addenda_indicator == '1' {
                    if let Some(Some(add)) = batch.addenda.get(idx) {
                        push_line(&mut out, &encode_addenda(add, entry)?)?;
                        records += 1;
                    } else {
                        return Err(Error::FieldRule {
                            field: "addenda",
                            reason: "indicator=1 but no addenda record".into(),
                        });
                    }
                }
            }
            let entry_count = u32::try_from(batch.entries.len())
                .map_err(|_| Error::Invalid("batch entry count overflow".into()))?;
            push_line(
                &mut out,
                &encode_batch_control(&batch.header, batch_no, entry_count, entry_hash, debit, credit)?,
            )?;
            records += 1;
            total_entries += entry_count;
            total_hash = total_hash.wrapping_add(entry_hash);
            total_debit = total_debit.checked_add(debit)
                .ok_or(op_core::Error::Overflow)?;
            total_credit = total_credit.checked_add(credit)
                .ok_or(op_core::Error::Overflow)?;
        }
        let batch_count = u32::try_from(self.batches.len())
            .map_err(|_| Error::Invalid("batch count overflow".into()))?;
        records += 1; // file control about to be appended
        let block_count = records.div_ceil(10);
        push_line(
            &mut out,
            &encode_file_control(batch_count, block_count, total_entries, total_hash, total_debit, total_credit)?,
        )?;
        // Pad filler records to a 10-record block.
        let padding = (10 - (records % 10)) % 10;
        for _ in 0..padding {
            push_line(&mut out, &"9".repeat(RECORD_LEN))?;
        }
        Ok(out)
    }

    /// Parse a NACHA byte stream back into the typed model.
    ///
    /// Filler `9`-only records are ignored. The decoder is lenient
    /// about trailing whitespace per line but strict about record
    /// length once whitespace is trimmed.
    ///
    /// # Errors
    /// [`Error::RecordLength`] for a malformed line; [`Error::Invalid`]
    /// for structurally impossible files (no header, batch control
    /// before batch header, etc.).
    pub fn decode(input: &str) -> Result<Self> {
        let mut header: Option<FileHeader> = None;
        let mut batches: Vec<NachaBatch> = Vec::new();
        let mut cur_batch: Option<NachaBatch> = None;
        for raw in input.lines() {
            let line = raw.trim_end_matches('\r');
            if line.chars().all(|c| c == '9') && line.len() == RECORD_LEN {
                continue;
            }
            if line.is_empty() {
                continue;
            }
            if line.len() != RECORD_LEN {
                return Err(Error::RecordLength {
                    expected: RECORD_LEN,
                    got: line.len(),
                });
            }
            let kind = &line[..1];
            match kind {
                "1" => {
                    header = Some(decode_file_header(line)?);
                }
                "5" => {
                    if cur_batch.is_some() {
                        return Err(Error::Invalid("batch header without prior control".into()));
                    }
                    cur_batch = Some(NachaBatch {
                        header: decode_batch_header(line)?,
                        entries: Vec::new(),
                        addenda: Vec::new(),
                    });
                }
                "6" => {
                    let b = cur_batch.as_mut().ok_or_else(||
                        Error::Invalid("entry outside a batch".into()))?;
                    b.entries.push(decode_entry(line)?);
                    b.addenda.push(None);
                }
                "7" => {
                    let b = cur_batch.as_mut().ok_or_else(||
                        Error::Invalid("addenda outside a batch".into()))?;
                    let add = decode_addenda(line)?;
                    if let Some(slot) = b.addenda.last_mut() {
                        *slot = Some(add);
                    }
                }
                "8" => {
                    let b = cur_batch.take().ok_or_else(||
                        Error::Invalid("batch control without batch header".into()))?;
                    batches.push(b);
                }
                "9" => {
                    // File control — final.
                    break;
                }
                other => {
                    return Err(Error::Invalid(format!("unknown record kind `{other}`")));
                }
            }
        }
        Ok(Self {
            header: header.ok_or_else(|| Error::Invalid("no file header".into()))?,
            batches,
        })
    }

    /// True if this file is flagged as Same-Day ACH — the effective
    /// entry date in each batch header matches today's date.
    ///
    /// NACHA same-day windows (Eastern Time):
    /// `10:30`, `14:45`, `16:45`. Operators set the effective date
    /// to today on a Same-Day batch and to T+1 / T+2 otherwise.
    #[must_use]
    pub fn is_same_day(&self, now: DateTime<Utc>) -> bool {
        let today = format!(
            "{:02}{:02}{:02}",
            now.year() % 100,
            now.month(),
            now.day()
        );
        self.batches
            .iter()
            .all(|b| b.header.effective_entry_date == today)
    }
}

// --- Same-Day ACH cutoff helpers ---------------------------------

/// Same-Day ACH submission windows (Eastern Time hours, 24h).
/// Per NACHA Operating Rules effective March 2024.
pub const SAME_DAY_WINDOWS_ET: &[(u32, u32)] = &[(10, 30), (14, 45), (16, 45)];

/// The next Same-Day ACH cutoff strictly after `now_et`. Returns
/// `None` if no window remains today (operator should submit
/// next-day instead).
#[must_use]
pub fn next_same_day_window(now_et: chrono::NaiveTime) -> Option<chrono::NaiveTime> {
    for (h, m) in SAME_DAY_WINDOWS_ET {
        let w = chrono::NaiveTime::from_hms_opt(*h, *m, 0)?;
        if w > now_et {
            return Some(w);
        }
    }
    None
}

// --- Return codes (R01..R99) ------------------------------------

/// A NACHA return code (`R01`..`R99`) attached to a returned entry.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ReturnCode(pub u8);

impl ReturnCode {
    /// Construct from the three-character on-wire form (`"R01"`).
    ///
    /// # Errors
    /// `Error::FieldRule` for any string that isn't `R` + two digits.
    pub fn parse(s: &str) -> Result<Self> {
        if s.len() != 3 || !s.starts_with('R') {
            return Err(Error::FieldRule {
                field: "return_code",
                reason: format!("expected R## form, got `{s}`"),
            });
        }
        let n = s[1..].parse::<u8>().map_err(|_| Error::FieldRule {
            field: "return_code",
            reason: format!("non-numeric tail `{s}`"),
        })?;
        if !(1..=99).contains(&n) {
            return Err(Error::FieldRule {
                field: "return_code",
                reason: format!("R-code out of range: {n}"),
            });
        }
        Ok(Self(n))
    }

    /// On-wire form (`"R01"`).
    #[must_use]
    pub fn as_string(self) -> String {
        format!("R{:02}", self.0)
    }

    /// English description of the top 20 codes by real-world frequency.
    /// Less-common codes return `None` and operators are expected to
    /// hand-handle (they're usually fatal anyway).
    #[must_use]
    pub const fn description(self) -> Option<&'static str> {
        // Source: NACHA 2024 Operating Rules, Appendix Five (the
        // commonly cited reference; the codes themselves are stable).
        Some(match self.0 {
            1 => "Insufficient Funds",
            2 => "Account Closed",
            3 => "No Account / Unable to Locate Account",
            4 => "Invalid Account Number Structure",
            5 => "Unauthorized Debit to Consumer Account Using Corporate SEC",
            6 => "Returned per ODFI's Request",
            7 => "Authorization Revoked by Customer",
            8 => "Payment Stopped",
            9 => "Uncollected Funds",
            10 => "Customer Advises Not Authorized",
            11 => "Check Truncation Entry Returned",
            12 => "Branch Sold to Another DFI",
            13 => "RDFI Not Qualified to Participate",
            14 => "Representative Payee Deceased or Unable to Continue",
            15 => "Beneficiary or Account Holder Deceased",
            16 => "Account Frozen / Entry Returned per OFAC Instruction",
            17 => "File Record Edit Criteria",
            18 => "Improper Effective Entry Date",
            19 => "Amount Field Error",
            20 => "Non-Transaction Account",
            _ => return None,
        })
    }
}

/// A parsed NACHA return (a `7` addenda record with type `99`,
/// usually packaged in a separate return file the RDFI sends back).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NachaReturn {
    /// `R01`..`R99`.
    pub code: ReturnCode,
    /// Trace number of the original outbound entry being returned.
    pub original_trace: String,
    /// RDFI's date-of-death / addenda info, if supplied (≤44 chars).
    pub addenda_info: String,
}

impl NachaReturn {
    /// Parse one `7` addenda record (type `99`).
    ///
    /// # Errors
    /// `Error::FieldRule` if the record isn't a type-99 addenda.
    pub fn parse_addenda(line: &str) -> Result<Self> {
        if line.len() != RECORD_LEN || !line.starts_with("799") {
            return Err(Error::FieldRule {
                field: "return_addenda",
                reason: "not a type-99 addenda".into(),
            });
        }
        let code = ReturnCode::parse(&line[3..6])?;
        let original_trace = line[6..21].trim().to_string();
        let addenda_info = line[21..(21 + 44).min(line.len())].trim().to_string();
        Ok(Self {
            code,
            original_trace,
            addenda_info,
        })
    }
}

// --- Encoders ---------------------------------------------------

fn encode_file_header(h: &FileHeader) -> Result<String> {
    if h.immediate_destination.len() != 10 {
        return Err(Error::FieldRule {
            field: "immediate_destination",
            reason: "must be 10 chars".into(),
        });
    }
    if h.immediate_origin.len() != 10 {
        return Err(Error::FieldRule {
            field: "immediate_origin",
            reason: "must be 10 chars".into(),
        });
    }
    if h.file_creation_date.len() != 6 {
        return Err(Error::FieldRule {
            field: "file_creation_date",
            reason: "must be YYMMDD".into(),
        });
    }
    if h.file_creation_time.len() != 4 {
        return Err(Error::FieldRule {
            field: "file_creation_time",
            reason: "must be HHMM".into(),
        });
    }
    let priority = "01";
    let record_size = "094";
    let blocking_factor = "10";
    let format_code = "1";
    Ok(format!(
        "1{priority}{dest:<10}{orig:<10}{date}{time}{modifier}{rsize}{bf}{fmt}{dest_name:<23}{orig_name:<23}{ref_code:<8}",
        priority = priority,
        dest = h.immediate_destination,
        orig = h.immediate_origin,
        date = h.file_creation_date,
        time = h.file_creation_time,
        modifier = h.file_id_modifier,
        rsize = record_size,
        bf = blocking_factor,
        fmt = format_code,
        dest_name = pad(&h.immediate_destination_name, 23),
        orig_name = pad(&h.immediate_origin_name, 23),
        ref_code = pad(&h.reference_code, 8),
    ))
}

fn encode_batch_header(h: &BatchHeader, batch_no: u32) -> Result<String> {
    if h.odfi_short.len() != 8 {
        return Err(Error::FieldRule {
            field: "odfi_short",
            reason: "must be 8 chars".into(),
        });
    }
    Ok(format!(
        "5{class:03}{co_name:<16}{discr:<20}{co_id:<10}{sec}{desc:<10}{descd}{eff}   {status}{odfi8}{batch:07}",
        class = h.service_class,
        co_name = pad(&h.company_name, 16),
        discr = pad(&h.company_discretionary, 20),
        co_id = pad(&h.company_id, 10),
        sec = h.sec.as_str(),
        desc = pad(&h.company_entry_description, 10),
        descd = pad(&h.descriptive_date, 6),
        eff = pad(&h.effective_entry_date, 6),
        status = h.originator_status,
        odfi8 = h.odfi_short,
        batch = batch_no,
    ))
}

fn encode_entry(e: &EntryDetail) -> Result<String> {
    if e.rdfi_routing.len() != 9 || !e.rdfi_routing.chars().all(|c| c.is_ascii_digit()) {
        return Err(Error::FieldRule {
            field: "rdfi_routing",
            reason: "must be 9 ASCII digits".into(),
        });
    }
    if e.trace_number.len() != 15 {
        return Err(Error::FieldRule {
            field: "trace_number",
            reason: "must be 15 chars".into(),
        });
    }
    Ok(format!(
        "6{tx:02}{rdfi8}{check}{acct:<17}{amount:010}{ind:<15}{rcv:<22}{discr:<2}{addind}{trace}",
        tx = e.transaction_code,
        rdfi8 = &e.rdfi_routing[..8],
        check = &e.rdfi_routing[8..9],
        acct = pad(&e.account_number, 17),
        amount = e.amount_cents,
        ind = pad(&e.individual_id, 15),
        rcv = pad(&e.receiver_name, 22),
        discr = pad(&e.discretionary, 2),
        addind = e.addenda_indicator,
        trace = e.trace_number,
    ))
}

fn encode_addenda(a: &Addenda, parent: &EntryDetail) -> Result<String> {
    if a.type_code.len() != 2 {
        return Err(Error::FieldRule {
            field: "type_code",
            reason: "must be 2 chars".into(),
        });
    }
    let trace_tail = parent
        .trace_number
        .get(8..)
        .ok_or_else(|| Error::FieldRule {
            field: "trace_number",
            reason: "trace too short".into(),
        })?;
    Ok(format!(
        "7{tcode}{info:<80}{seq:04}{entry:07}",
        tcode = a.type_code,
        info = pad(&a.payment_related_info, 80),
        seq = a.sequence_number,
        entry = trace_tail.parse::<u32>().unwrap_or(a.entry_detail_sequence),
    ))
}

fn encode_batch_control(
    h: &BatchHeader,
    batch_no: u32,
    entry_count: u32,
    entry_hash: u64,
    debit: u64,
    credit: u64,
) -> Result<String> {
    let hash_trim = entry_hash % 10_u64.pow(10);
    Ok(format!(
        "8{class:03}{cnt:06}{hash:010}{debit:012}{credit:012}{co_id:<10}{auth:<19}{rsv:<6}{odfi8}{batch:07}",
        class = h.service_class,
        cnt = entry_count,
        hash = hash_trim,
        debit = debit,
        credit = credit,
        co_id = pad(&h.company_id, 10),
        auth = " ".repeat(19),
        rsv = " ".repeat(6),
        odfi8 = h.odfi_short,
        batch = batch_no,
    ))
}

fn encode_file_control(
    batch_count: u32,
    block_count: u32,
    entry_count: u32,
    entry_hash: u64,
    debit: u64,
    credit: u64,
) -> Result<String> {
    let hash_trim = entry_hash % 10_u64.pow(10);
    Ok(format!(
        "9{batches:06}{blocks:06}{cnt:08}{hash:010}{debit:012}{credit:012}{rsv:<39}",
        batches = batch_count,
        blocks = block_count,
        cnt = entry_count,
        hash = hash_trim,
        debit = debit,
        credit = credit,
        rsv = " ".repeat(39),
    ))
}

// --- Decoders ---------------------------------------------------

fn decode_file_header(line: &str) -> Result<FileHeader> {
    Ok(FileHeader {
        immediate_destination: line[3..13].trim().to_string(),
        immediate_origin: line[13..23].trim().to_string(),
        file_creation_date: line[23..29].to_string(),
        file_creation_time: line[29..33].to_string(),
        file_id_modifier: line.as_bytes()[33] as char,
        immediate_destination_name: line[40..63].trim().to_string(),
        immediate_origin_name: line[63..86].trim().to_string(),
        reference_code: line[86..94].trim().to_string(),
    })
}

fn decode_batch_header(line: &str) -> Result<BatchHeader> {
    let service_class = line[1..4].parse::<u16>().map_err(|_| Error::FieldRule {
        field: "service_class",
        reason: "not numeric".into(),
    })?;
    let sec = SecCode::from_str(&line[50..53])?;
    let originator_status = line.as_bytes()[78] as char;
    Ok(BatchHeader {
        service_class,
        company_name: line[4..20].trim().to_string(),
        company_discretionary: line[20..40].trim().to_string(),
        company_id: line[40..50].trim().to_string(),
        sec,
        company_entry_description: line[53..63].trim().to_string(),
        descriptive_date: line[63..69].to_string(),
        effective_entry_date: line[69..75].to_string(),
        originator_status,
        odfi_short: line[79..87].to_string(),
        batch_number: line[87..94].parse::<u32>().map_err(|_| Error::FieldRule {
            field: "batch_number",
            reason: "not numeric".into(),
        })?,
    })
}

fn decode_entry(line: &str) -> Result<EntryDetail> {
    let transaction_code = line[1..3].parse::<u8>().map_err(|_| Error::FieldRule {
        field: "transaction_code",
        reason: "not numeric".into(),
    })?;
    let amount_cents = line[29..39].parse::<u64>().map_err(|_| Error::FieldRule {
        field: "amount",
        reason: "not numeric".into(),
    })?;
    Ok(EntryDetail {
        transaction_code,
        rdfi_routing: line[3..12].to_string(),
        account_number: line[12..29].trim().to_string(),
        amount_cents,
        individual_id: line[39..54].trim().to_string(),
        receiver_name: line[54..76].trim().to_string(),
        discretionary: line[76..78].trim().to_string(),
        addenda_indicator: line.as_bytes()[78] as char,
        trace_number: line[79..94].to_string(),
    })
}

fn decode_addenda(line: &str) -> Result<Addenda> {
    let sequence_number = line[83..87].parse::<u16>().map_err(|_| Error::FieldRule {
        field: "seq",
        reason: "not numeric".into(),
    })?;
    let entry_detail_sequence = line[87..94].parse::<u32>().map_err(|_| Error::FieldRule {
        field: "entry_seq",
        reason: "not numeric".into(),
    })?;
    Ok(Addenda {
        type_code: line[1..3].to_string(),
        payment_related_info: line[3..83].trim().to_string(),
        sequence_number,
        entry_detail_sequence,
    })
}

// --- Helpers ----------------------------------------------------

fn pad(s: &str, width: usize) -> String {
    if s.len() >= width {
        s[..width].to_string()
    } else {
        format!("{s:<width$}")
    }
}

fn push_line(out: &mut String, line: &str) -> Result<()> {
    if line.len() != RECORD_LEN {
        return Err(Error::RecordLength {
            expected: RECORD_LEN,
            got: line.len(),
        });
    }
    out.push_str(line);
    out.push('\n');
    Ok(())
}

fn parse_u64(s: &str) -> Result<u64> {
    s.parse::<u64>().map_err(|_| Error::FieldRule {
        field: "numeric",
        reason: format!("not a u64: `{s}`"),
    })
}

const fn is_credit(tx_code: u8) -> bool {
    // Checking credit (22), prenote credit (23), savings credit (32, 33).
    matches!(tx_code, 22 | 23 | 32 | 33)
}

fn validate_entry(e: &EntryDetail) -> Result<()> {
    if e.account_number.is_empty() {
        return Err(Error::FieldRule {
            field: "account_number",
            reason: "must not be empty".into(),
        });
    }
    if e.receiver_name.is_empty() {
        return Err(Error::FieldRule {
            field: "receiver_name",
            reason: "must not be empty".into(),
        });
    }
    Ok(())
}

/// Build a NACHA file from operator profile + entries — the high-
/// level convenience entry-point used by [`crate::orchestrator`].
///
/// `effective_date` formatted `YYMMDD`; `same_day` controls only
/// the descriptive date so audit logs are clear.
///
/// # Errors
/// Propagates field-rule failures from the encoder.
pub fn build_file(
    profile: &NachaProfile,
    sec: SecCode,
    entries: Vec<EntryDetail>,
    effective_date: &str,
    now: DateTime<Utc>,
) -> Result<NachaFile> {
    if entries.is_empty() {
        return Err(Error::Invalid("no entries".into()));
    }
    let creation_date = format!(
        "{:02}{:02}{:02}",
        now.year() % 100,
        now.month(),
        now.day()
    );
    let creation_time = format!("{:02}{:02}", now.hour(), now.minute());
    let service_class: u16 = if entries.iter().all(|e| is_credit(e.transaction_code)) {
        220
    } else if entries.iter().all(|e| !is_credit(e.transaction_code)) {
        225
    } else {
        200
    };
    let odfi_short = profile.odfi_routing[..8].to_string();
    let header = FileHeader {
        immediate_destination: profile.immediate_destination.clone(),
        immediate_origin: profile.immediate_origin.clone(),
        file_creation_date: creation_date.clone(),
        file_creation_time: creation_time,
        file_id_modifier: 'A',
        immediate_destination_name: profile.company_name.clone(),
        immediate_origin_name: profile.company_name.clone(),
        reference_code: String::new(),
    };
    let batch_header = BatchHeader {
        service_class,
        company_name: profile.company_name.clone(),
        company_discretionary: String::new(),
        company_id: profile.company_id.clone(),
        sec,
        company_entry_description: profile.company_entry_description.clone(),
        descriptive_date: creation_date.clone(),
        effective_entry_date: effective_date.to_string(),
        originator_status: '1',
        odfi_short,
        batch_number: 1,
    };
    let addenda = vec![None; entries.len()];
    Ok(NachaFile {
        header,
        batches: vec![NachaBatch {
            header: batch_header,
            entries,
            addenda,
        }],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn profile() -> NachaProfile {
        NachaProfile {
            odfi_routing: "121000248".into(),
            immediate_origin: "1234567890".into(),
            immediate_destination: "0210000211".into(),
            company_name: "OPENPAY VENDOR INC".into(),
            company_id: "9876543210".into(),
            company_entry_description: "SETTLEMNT".into(),
        }
    }

    fn entry(amount: u64, suffix: u32) -> EntryDetail {
        EntryDetail {
            transaction_code: 22,
            rdfi_routing: "021000021".into(),
            account_number: format!("ACCT{suffix}"),
            amount_cents: amount,
            individual_id: format!("IND-{suffix}"),
            receiver_name: format!("RECEIVER {suffix}"),
            discretionary: String::new(),
            addenda_indicator: '0',
            trace_number: format!("12100024{suffix:07}"),
        }
    }

    #[test]
    fn round_trip_two_credits() {
        let now = Utc::now();
        let file = build_file(
            &profile(),
            SecCode::Ppd,
            vec![entry(750_000, 1), entry(250_000, 2)],
            "260601",
            now,
        )
        .unwrap();
        let wire = file.encode().unwrap();
        // Every line is exactly 94 chars.
        for line in wire.lines() {
            assert_eq!(line.len(), 94);
        }
        // Block-aligned (multiple of 10 lines).
        assert_eq!(wire.lines().count() % 10, 0);
        let parsed = NachaFile::decode(&wire).unwrap();
        assert_eq!(parsed.batches.len(), 1);
        assert_eq!(parsed.batches[0].entries.len(), 2);
        assert_eq!(parsed.batches[0].entries[0].amount_cents, 750_000);
        assert_eq!(parsed.batches[0].entries[1].amount_cents, 250_000);
        assert_eq!(parsed.batches[0].header.sec, SecCode::Ppd);
    }

    #[test]
    fn round_trip_web_sec_code() {
        let now = Utc::now();
        let mut e = entry(12_345, 7);
        e.transaction_code = 27; // checking debit
        let file = build_file(&profile(), SecCode::Web, vec![e], "260601", now).unwrap();
        let wire = file.encode().unwrap();
        let parsed = NachaFile::decode(&wire).unwrap();
        assert_eq!(parsed.batches[0].header.sec, SecCode::Web);
        assert_eq!(parsed.batches[0].entries[0].transaction_code, 27);
    }

    #[test]
    fn ctx_sec_with_addenda() {
        let now = Utc::now();
        let mut e = entry(100, 9);
        e.addenda_indicator = '1';
        let mut file = build_file(&profile(), SecCode::Ctx, vec![e], "260601", now).unwrap();
        file.batches[0].addenda[0] = Some(Addenda {
            type_code: "05".into(),
            payment_related_info: "INVOICE 12345".into(),
            sequence_number: 1,
            entry_detail_sequence: 1,
        });
        let wire = file.encode().unwrap();
        let parsed = NachaFile::decode(&wire).unwrap();
        assert_eq!(parsed.batches[0].addenda[0].as_ref().unwrap().type_code, "05");
    }

    #[test]
    fn return_code_parser_top_20() {
        for n in 1u8..=20 {
            let code = ReturnCode(n);
            let s = code.as_string();
            assert_eq!(s.len(), 3);
            assert!(s.starts_with('R'));
            assert!(code.description().is_some());
            let back = ReturnCode::parse(&s).unwrap();
            assert_eq!(back, code);
        }
    }

    #[test]
    fn return_code_rejects_garbage() {
        assert!(ReturnCode::parse("xyz").is_err());
        assert!(ReturnCode::parse("R0").is_err());
        assert!(ReturnCode::parse("R00").is_err());
    }

    #[test]
    fn same_day_window_progression() {
        let t = chrono::NaiveTime::from_hms_opt(9, 0, 0).unwrap();
        assert_eq!(
            next_same_day_window(t),
            Some(chrono::NaiveTime::from_hms_opt(10, 30, 0).unwrap())
        );
        let t = chrono::NaiveTime::from_hms_opt(11, 0, 0).unwrap();
        assert_eq!(
            next_same_day_window(t),
            Some(chrono::NaiveTime::from_hms_opt(14, 45, 0).unwrap())
        );
        let t = chrono::NaiveTime::from_hms_opt(17, 0, 0).unwrap();
        assert_eq!(next_same_day_window(t), None);
    }

    #[test]
    fn same_day_flag_matches_today() {
        let now = Utc::now();
        let today = format!(
            "{:02}{:02}{:02}",
            now.year() % 100,
            now.month(),
            now.day()
        );
        let file =
            build_file(&profile(), SecCode::Ppd, vec![entry(100, 1)], &today, now).unwrap();
        assert!(file.is_same_day(now));
    }
}
