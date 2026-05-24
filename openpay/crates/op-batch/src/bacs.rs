//! UK Bacs Direct Credit and Direct Debit.
//!
//! Bacs is the UK's three-day batch payment scheme (input day,
//! processing day, entry day). Operators submit via Bacstel-IP
//! (or modern equivalents); the file format is **fixed-width
//! 100-character records** with a fixed schema known as the
//! "Standard 18" layout for the receiver-detail records and an
//! associated VOL1/HDR1/UHL1 label set on the original
//! tape-derived format.
//!
//! ## Layout (Standard 18, per Bacs Service User's Guide)
//!
//! | Cols | Field | Notes |
//! |------|-------|-------|
//! | 1-6  | destination sort code | 6 digits |
//! | 7-14 | destination account | 8 digits |
//! | 15   | account type | usually `0` |
//! | 16-17| transaction code | `99` credit, `01` first DD, `17` recurring DD, `18` final |
//! | 18-23| originator sort code | 6 digits |
//! | 24-31| originator account | 8 digits |
//! | 32-35| originator's reference | 4 chars (RTI) |
//! | 36-46| amount in pence | 11 digits, zero-padded |
//! | 47-64| originator name | 18 chars |
//! | 65-82| destination reference | 18 chars |
//! | 83-100| destination name | 18 chars |
//!
//! Plus header / trailer records: `VOL1`, `HDR1`, `UHL1` /
//! `EOF1`, `UTL1`. We emit a minimal-but-conformant subset that
//! the bureau service indicators expect.

use chrono::{DateTime, Datelike, Utc};
use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Fixed Bacs record length.
pub const BACS_RECORD_LEN: usize = 100;

/// Bacs transaction code: which "scheme" the payment runs under.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum BacsTransactionCode {
    /// `99` — Bacs Direct Credit.
    Credit,
    /// `01` — first Direct Debit collection of a series.
    DdFirst,
    /// `17` — recurring Direct Debit collection.
    DdRecurring,
    /// `18` — final Direct Debit of a series.
    DdFinal,
    /// `19` — re-presented DD after unpaid return.
    DdRepresented,
}

impl BacsTransactionCode {
    /// Two-character on-wire code.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Credit => "99",
            Self::DdFirst => "01",
            Self::DdRecurring => "17",
            Self::DdFinal => "18",
            Self::DdRepresented => "19",
        }
    }

    /// Parse from on-wire (two chars).
    ///
    /// # Errors
    /// `Error::FieldRule` if not a known Bacs transaction code.
    pub fn from_str(s: &str) -> Result<Self> {
        Ok(match s {
            "99" => Self::Credit,
            "01" => Self::DdFirst,
            "17" => Self::DdRecurring,
            "18" => Self::DdFinal,
            "19" => Self::DdRepresented,
            other => {
                return Err(Error::FieldRule {
                    field: "bacs_tx_code",
                    reason: format!("unknown `{other}`"),
                });
            }
        })
    }
}

/// One Bacs detail line (Standard 18 record).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BacsRecord {
    /// Destination (beneficiary or payer) sort code, 6 digits.
    pub destination_sort_code: String,
    /// Destination account number, 8 digits.
    pub destination_account: String,
    /// Account type indicator (almost always `0`).
    pub account_type: char,
    /// Transaction code.
    pub transaction_code: BacsTransactionCode,
    /// Originator (operator) sort code, 6 digits.
    pub originator_sort_code: String,
    /// Originator account number, 8 digits.
    pub originator_account: String,
    /// 4-char Real-Time Information ref (HMRC / generic payment).
    pub originator_reference: String,
    /// Amount in pence (positive).
    pub amount_pence: u64,
    /// Originator name (≤18 chars).
    pub originator_name: String,
    /// Reference shown to the destination (≤18 chars).
    pub destination_reference: String,
    /// Destination name (≤18 chars).
    pub destination_name: String,
}

/// A complete Bacs submission file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct BacsFile {
    /// Service User Number (SUN), 6 digits — assigned by Bacs.
    pub service_user_number: String,
    /// Originator name displayed on the bureau header.
    pub originator_name: String,
    /// Processing day (`YYDDD` julian, the Bacs convention).
    pub processing_day_julian: String,
    /// Detail records.
    pub records: Vec<BacsRecord>,
}

impl BacsFile {
    /// Encode the file into the on-wire 100-char-per-line format.
    ///
    /// Layout: `VOL1` → `HDR1` → `UHL1` → records → `EOF1` →
    /// `UTL1`. Each record padded with spaces to 100 chars.
    ///
    /// # Errors
    /// [`Error::FieldRule`] for bad sort codes or account numbers;
    /// [`Error::RecordLength`] is impossible by construction here.
    pub fn encode(&self) -> Result<String> {
        if self.service_user_number.len() != 6
            || !self.service_user_number.chars().all(|c| c.is_ascii_digit())
        {
            return Err(Error::FieldRule {
                field: "service_user_number",
                reason: "must be 6 ASCII digits".into(),
            });
        }
        let mut out = String::new();
        // VOL1 — volume label (80 chars, padded to 100).
        out.push_str(&pad(
            &format!("VOL1{:<11}", self.service_user_number),
            BACS_RECORD_LEN,
        ));
        out.push('\n');
        // HDR1 — file header.
        out.push_str(&pad(
            &format!(
                "HDR1A{sun}{name:<14}{day}",
                sun = self.service_user_number,
                name = pad(&self.originator_name, 14),
                day = self.processing_day_julian
            ),
            BACS_RECORD_LEN,
        ));
        out.push('\n');
        // UHL1 — user header label.
        out.push_str(&pad(
            &format!("UHL1{day}", day = self.processing_day_julian),
            BACS_RECORD_LEN,
        ));
        out.push('\n');
        let mut total: u64 = 0;
        let mut count: u32 = 0;
        for r in &self.records {
            validate_record(r)?;
            total = total
                .checked_add(r.amount_pence)
                .ok_or(op_core::Error::Overflow)?;
            count += 1;
            let line = format!(
                "{dsc}{dac}{at}{tx}{osc}{oac}{oref:<4}{amt:011}{oname:<18}{dref:<18}{dname:<18}",
                dsc = r.destination_sort_code,
                dac = r.destination_account,
                at = r.account_type,
                tx = r.transaction_code.as_str(),
                osc = r.originator_sort_code,
                oac = r.originator_account,
                oref = pad(&r.originator_reference, 4),
                amt = r.amount_pence,
                oname = pad(&r.originator_name, 18),
                dref = pad(&r.destination_reference, 18),
                dname = pad(&r.destination_name, 18),
            );
            if line.len() != BACS_RECORD_LEN {
                return Err(Error::RecordLength {
                    expected: BACS_RECORD_LEN,
                    got: line.len(),
                });
            }
            out.push_str(&line);
            out.push('\n');
        }
        // EOF1 — file trailer.
        out.push_str(&pad(
            &format!("EOF1{sun}{cnt:08}{tot:013}", sun = self.service_user_number, cnt = count, tot = total),
            BACS_RECORD_LEN,
        ));
        out.push('\n');
        // UTL1 — user trailer.
        out.push_str(&pad(
            &format!("UTL1{cnt:08}{tot:013}", cnt = count, tot = total),
            BACS_RECORD_LEN,
        ));
        out.push('\n');
        Ok(out)
    }

    /// Decode a Bacs file.
    ///
    /// # Errors
    /// [`Error::RecordLength`] for any mis-sized line;
    /// [`Error::FieldRule`] for unparseable tx-codes / amounts.
    pub fn decode(input: &str) -> Result<Self> {
        let mut sun = String::new();
        let mut name = String::new();
        let mut day = String::new();
        let mut records = Vec::new();
        for line in input.lines() {
            let line = line.trim_end_matches('\r');
            if line.is_empty() {
                continue;
            }
            if line.len() != BACS_RECORD_LEN {
                return Err(Error::RecordLength {
                    expected: BACS_RECORD_LEN,
                    got: line.len(),
                });
            }
            if line.starts_with("VOL1") {
                sun = line[4..10].trim().to_string();
            } else if line.starts_with("HDR1") {
                if line.len() >= 25 {
                    name = line[10..24].trim().to_string();
                    day = line[24..29].to_string();
                }
            } else if line.starts_with("UHL1") {
                if line.len() >= 9 && day.is_empty() {
                    day = line[4..9].to_string();
                }
            } else if line.starts_with("EOF1") || line.starts_with("UTL1") {
                continue;
            } else {
                records.push(decode_record(line)?);
            }
        }
        Ok(Self {
            service_user_number: sun,
            originator_name: name,
            processing_day_julian: day,
            records,
        })
    }
}

fn decode_record(line: &str) -> Result<BacsRecord> {
    let amount_pence = line[35..46].parse::<u64>().map_err(|_| Error::FieldRule {
        field: "amount_pence",
        reason: "not numeric".into(),
    })?;
    Ok(BacsRecord {
        destination_sort_code: line[..6].to_string(),
        destination_account: line[6..14].to_string(),
        account_type: line.as_bytes()[14] as char,
        transaction_code: BacsTransactionCode::from_str(&line[15..17])?,
        originator_sort_code: line[17..23].to_string(),
        originator_account: line[23..31].to_string(),
        originator_reference: line[31..35].trim().to_string(),
        amount_pence,
        originator_name: line[46..64].trim().to_string(),
        destination_reference: line[64..82].trim().to_string(),
        destination_name: line[82..100].trim().to_string(),
    })
}

fn validate_record(r: &BacsRecord) -> Result<()> {
    if r.destination_sort_code.len() != 6
        || !r.destination_sort_code.chars().all(|c| c.is_ascii_digit())
    {
        return Err(Error::FieldRule {
            field: "destination_sort_code",
            reason: "must be 6 digits".into(),
        });
    }
    if r.destination_account.len() != 8
        || !r.destination_account.chars().all(|c| c.is_ascii_digit())
    {
        return Err(Error::FieldRule {
            field: "destination_account",
            reason: "must be 8 digits".into(),
        });
    }
    if r.originator_sort_code.len() != 6
        || !r.originator_sort_code.chars().all(|c| c.is_ascii_digit())
    {
        return Err(Error::FieldRule {
            field: "originator_sort_code",
            reason: "must be 6 digits".into(),
        });
    }
    if r.originator_account.len() != 8
        || !r.originator_account.chars().all(|c| c.is_ascii_digit())
    {
        return Err(Error::FieldRule {
            field: "originator_account",
            reason: "must be 8 digits".into(),
        });
    }
    Ok(())
}

fn pad(s: &str, width: usize) -> String {
    if s.len() >= width {
        s[..width].to_string()
    } else {
        format!("{s:<width$}")
    }
}

/// Compute the Bacs processing-day julian date (`YYDDD`) for a
/// given UTC timestamp. Bacs uses ordinal-day-of-year in the
/// header / trailer.
#[must_use]
pub fn processing_day_julian(dt: DateTime<Utc>) -> String {
    format!("{:02}{:03}", dt.year() % 100, dt.ordinal())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn file() -> BacsFile {
        BacsFile {
            service_user_number: "123456".into(),
            originator_name: "OPENPAY".into(),
            processing_day_julian: "26152".into(),
            records: vec![
                BacsRecord {
                    destination_sort_code: "200000".into(),
                    destination_account: "12345678".into(),
                    account_type: '0',
                    transaction_code: BacsTransactionCode::Credit,
                    originator_sort_code: "400000".into(),
                    originator_account: "87654321".into(),
                    originator_reference: "REF1".into(),
                    amount_pence: 12_345,
                    originator_name: "OPENPAY LTD".into(),
                    destination_reference: "INV-001".into(),
                    destination_name: "ALICE".into(),
                },
                BacsRecord {
                    destination_sort_code: "200001".into(),
                    destination_account: "12345679".into(),
                    account_type: '0',
                    transaction_code: BacsTransactionCode::DdRecurring,
                    originator_sort_code: "400000".into(),
                    originator_account: "87654321".into(),
                    originator_reference: "REF2".into(),
                    amount_pence: 6_789,
                    originator_name: "OPENPAY LTD".into(),
                    destination_reference: "MND-007".into(),
                    destination_name: "BOB".into(),
                },
            ],
        }
    }

    #[test]
    fn round_trip_two_records() {
        let f = file();
        let wire = f.encode().unwrap();
        for line in wire.lines() {
            assert_eq!(line.len(), 100);
        }
        let parsed = BacsFile::decode(&wire).unwrap();
        assert_eq!(parsed.service_user_number, "123456");
        assert_eq!(parsed.records.len(), 2);
        assert_eq!(parsed.records[0].amount_pence, 12_345);
        assert_eq!(
            parsed.records[1].transaction_code,
            BacsTransactionCode::DdRecurring
        );
    }

    #[test]
    fn rejects_bad_sort_code() {
        let mut f = file();
        f.records[0].destination_sort_code = "ABC123".into();
        assert!(f.encode().is_err());
    }

    #[test]
    fn rejects_bad_sun() {
        let mut f = file();
        f.service_user_number = "12345".into();
        assert!(f.encode().is_err());
    }

    #[test]
    fn julian_format() {
        let dt = chrono::NaiveDate::from_ymd_opt(2026, 6, 1)
            .unwrap()
            .and_hms_opt(0, 0, 0)
            .unwrap()
            .and_utc();
        assert_eq!(processing_day_julian(dt), "26152");
    }
}
