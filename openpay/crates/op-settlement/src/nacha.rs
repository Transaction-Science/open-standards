//! NACHA (US ACH) payout file generation.
//!
//! NACHA is the file format US banks use to ingest ACH credits and
//! debits. Each record is exactly **94 characters** of fixed-width
//! ASCII, line-terminated by `\n` (some banks want CRLF — operators
//! post-process if so). Five record types in the order we emit:
//!
//! | Type | Code | Purpose |
//! |------|------|---------|
//! | File Header | `1` | Identifies the ODFI, immediate origin, file id |
//! | Batch Header | `5` | One per ACH batch; class code (PPD/CCD/WEB) |
//! | Entry Detail | `6` | One per credit/debit |
//! | Batch Control | `8` | Closes the batch; entry hash + sums |
//! | File Control | `9` | Closes the file; same with batch count |
//!
//! Plus filler records (`9999...`) padding to a multiple of 10
//! records per block.
//!
//! Reference: NACHA 2024 Operating Rules, Appendix Three.
//!
//! ## Scope
//!
//! This is a **reference** generator. It produces conformant PPD
//! (consumer) and CCD (corporate) credit files good enough for most
//! ODFIs that don't need IAT/CTX/web-debit special records. Full
//! NACHA support (returns, NOC, addenda) lives outside.

use op_core::{Currency, Money};
use serde::{Deserialize, Serialize};

use crate::batch::Batch;
use crate::error::{Error, Result};
use crate::payout::PayoutRail;

/// One credit entry in a NACHA file.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NachaCredit {
    /// Receiving DFI routing number (9 digits, ABA).
    pub rdfi_routing: String,
    /// Receiver account number (≤17 chars).
    pub account_number: String,
    /// Receiver name (≤22 chars).
    pub receiver_name: String,
    /// Amount in USD cents (positive).
    pub amount_cents: u64,
    /// 15-char individual id (operator-side reference).
    pub individual_id: String,
    /// Standard Entry Class. `"PPD"` (consumer prearranged) or
    /// `"CCD"` (corporate). Default the caller passes from their
    /// merchant profile.
    pub sec: SecCode,
}

/// Standard Entry Class code supported by this generator.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum SecCode {
    /// Prearranged Payment and Deposit (consumer).
    Ppd,
    /// Cash Concentration and Disbursement (corporate).
    Ccd,
}

impl SecCode {
    fn as_str(self) -> &'static str {
        match self {
            Self::Ppd => "PPD",
            Self::Ccd => "CCD",
        }
    }
}

/// Static operator parameters required for NACHA file header / batch
/// header.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NachaProfile {
    /// Operator's ODFI routing number (9 digits) — the bank
    /// originating the ACH.
    pub odfi_routing: String,
    /// Immediate origin (10 chars; usually the operator's tax id
    /// padded with leading space or `1` prefix).
    pub immediate_origin: String,
    /// Immediate destination (10 chars; usually `1` + ODFI
    /// routing).
    pub immediate_destination: String,
    /// Company name (≤23 chars). Appears on the receiver's bank
    /// statement.
    pub company_name: String,
    /// Company id (10 chars).
    pub company_id: String,
    /// Company entry description (≤10 chars), e.g. `"PAYROLL"`,
    /// `"SETTLEMNT"`.
    pub company_entry_description: String,
    /// Effective entry date (`YYMMDD`).
    pub effective_entry_date: String,
}

/// Render a NACHA file for `batch` against `profile`.
///
/// `credits` is the per-receiver detail list; the caller maps each
/// batch entry to a `NachaCredit` (the batch knows only ledger
/// txs, not banking coordinates). Length must equal
/// `batch.entries.len()`.
///
/// # Errors
/// - [`Error::EmptyBatch`] if no credits.
/// - [`Error::Invalid`] if the profile / credits violate NACHA
///   field length rules (e.g. routing number not 9 digits).
/// - [`Error::CurrencyMismatch`] if the batch isn't USD.
/// - [`Error::Invalid`] if the batch rail isn't `AchNacha`.
#[allow(clippy::too_many_lines)]
pub fn nacha_file(
    batch: &Batch,
    profile: &NachaProfile,
    credits: &[NachaCredit],
) -> Result<String> {
    if batch.rail != PayoutRail::AchNacha {
        return Err(Error::Invalid(format!(
            "batch rail is {:?}, NACHA requires AchNacha",
            batch.rail
        )));
    }
    if batch.currency != Currency::USD {
        return Err(Error::CurrencyMismatch {
            batch: batch.currency.code().to_owned(),
            tx: "USD".to_owned(),
        });
    }
    if credits.is_empty() {
        return Err(Error::EmptyBatch);
    }
    if credits.len() != batch.entries.len() {
        return Err(Error::Invalid(format!(
            "credit count {} != batch entry count {}",
            credits.len(),
            batch.entries.len()
        )));
    }

    validate_profile(profile)?;

    let mut out = String::with_capacity(94 * (4 + credits.len()));

    // 1: File Header
    let file_id_modifier = 'A';
    let line = format!(
        "101{dest:>10}{orig:>10}{date:>6}{time:>4}{mod}094101{dest_name:<23}{orig_name:<23}{ref_code:<8}",
        dest = profile.immediate_destination,
        orig = profile.immediate_origin,
        date = &profile.effective_entry_date,
        time = "0000",
        mod = file_id_modifier,
        dest_name = pad_truncate(&profile.company_name, 23),
        orig_name = pad_truncate(&profile.company_name, 23),
        ref_code = "        ",
    );
    push_94(&mut out, &line)?;

    // 5: Batch Header (one batch per file in this reference impl).
    let service_class = "220"; // credits only
    let sec_code = credits[0].sec.as_str();
    let batch_number = 1u32;
    let descriptive_date = &profile.effective_entry_date;
    let originator_status = "1";
    let line = format!(
        "5{class}{co_name:<16}{discr:<20}{co_id:<10}{sec:<3}{desc:<10}{descd:>6}{eff:>6}   {status}{odfi_short:>8}{batch_no:0>7}",
        class = service_class,
        co_name = pad_truncate(&profile.company_name, 16),
        discr = " ".repeat(20),
        co_id = pad_truncate(&profile.company_id, 10),
        sec = sec_code,
        desc = pad_truncate(&profile.company_entry_description, 10),
        descd = pad_truncate(descriptive_date, 6),
        eff = pad_truncate(&profile.effective_entry_date, 6),
        status = originator_status,
        odfi_short = &profile.odfi_routing[..8.min(profile.odfi_routing.len())],
        batch_no = batch_number,
    );
    push_94(&mut out, &line)?;

    // 6: Entry Details
    let mut entry_hash: u64 = 0;
    let mut total_credit: u64 = 0;
    let mut seq: u32 = 0;
    for c in credits {
        validate_credit(c)?;
        seq += 1;
        let tx_code = "22"; // checking credit
        let rdfi_8 = &c.rdfi_routing[..8];
        let check_digit = &c.rdfi_routing[8..9];
        let amount = batch.amount_for_currency_in_cents(c.amount_cents)?;
        entry_hash = entry_hash.wrapping_add(parse_u64(rdfi_8)?);
        total_credit = total_credit
            .checked_add(amount)
            .ok_or(op_core::Error::Overflow)?;
        let trace = format!("{}{seq:0>7}", &profile.odfi_routing[..8], seq = seq);
        let line = format!(
            "6{tx_code}{rdfi_8}{check}{acct:<17}{amount:0>10}{ind_id:<15}{rcv_name:<22}  0{trace}",
            tx_code = tx_code,
            rdfi_8 = rdfi_8,
            check = check_digit,
            acct = pad_truncate(&c.account_number, 17),
            amount = amount,
            ind_id = pad_truncate(&c.individual_id, 15),
            rcv_name = pad_truncate(&c.receiver_name, 22),
            trace = trace,
        );
        push_94(&mut out, &line)?;
    }

    // 8: Batch Control
    let entry_count = seq;
    let entry_hash_trim = entry_hash % 10_u64.pow(10);
    let line = format!(
        "8{class}{cnt:0>6}{hash:0>10}{debit:0>12}{credit:0>12}{co_id:<10}{auth_code:<19}{rsv:<6}{odfi_short:>8}{batch_no:0>7}",
        class = service_class,
        cnt = entry_count,
        hash = entry_hash_trim,
        debit = 0u64,
        credit = total_credit,
        co_id = pad_truncate(&profile.company_id, 10),
        auth_code = " ".repeat(19),
        rsv = " ".repeat(6),
        odfi_short = &profile.odfi_routing[..8.min(profile.odfi_routing.len())],
        batch_no = batch_number,
    );
    push_94(&mut out, &line)?;

    // 9: File Control
    let batch_count: u32 = 1;
    // Block count = ceil(record_count_so_far / 10). +1 for the file
    // control record itself.
    let records_so_far = 1 + 1 + entry_count + 1 + 1;
    let block_count = records_so_far.div_ceil(10);
    let line = format!(
        "9{batches:0>6}{blocks:0>6}{cnt:0>8}{hash:0>10}{debit:0>12}{credit:0>12}{rsv:<39}",
        batches = batch_count,
        blocks = block_count,
        cnt = entry_count,
        hash = entry_hash_trim,
        debit = 0u64,
        credit = total_credit,
        rsv = " ".repeat(39),
    );
    push_94(&mut out, &line)?;

    // Pad with 9999... filler records to next block boundary.
    let total_records = records_so_far;
    let padding_needed = (10 - (total_records % 10)) % 10;
    for _ in 0..padding_needed {
        push_94(&mut out, &"9".repeat(94))?;
    }

    Ok(out)
}

fn validate_profile(p: &NachaProfile) -> Result<()> {
    if p.odfi_routing.len() != 9 || !p.odfi_routing.chars().all(|c| c.is_ascii_digit()) {
        return Err(Error::Invalid("odfi_routing must be 9 ASCII digits".into()));
    }
    if p.immediate_origin.len() != 10 {
        return Err(Error::Invalid(
            "immediate_origin must be exactly 10 chars".into(),
        ));
    }
    if p.immediate_destination.len() != 10 {
        return Err(Error::Invalid(
            "immediate_destination must be exactly 10 chars".into(),
        ));
    }
    if p.effective_entry_date.len() != 6
        || !p.effective_entry_date.chars().all(|c| c.is_ascii_digit())
    {
        return Err(Error::Invalid(
            "effective_entry_date must be YYMMDD (6 ASCII digits)".into(),
        ));
    }
    Ok(())
}

fn validate_credit(c: &NachaCredit) -> Result<()> {
    if c.rdfi_routing.len() != 9 || !c.rdfi_routing.chars().all(|d| d.is_ascii_digit()) {
        return Err(Error::Invalid("rdfi_routing must be 9 ASCII digits".into()));
    }
    if c.account_number.is_empty() || c.account_number.len() > 17 {
        return Err(Error::Invalid("account_number must be 1..=17 chars".into()));
    }
    if c.receiver_name.is_empty() {
        return Err(Error::Invalid("receiver_name required".into()));
    }
    Ok(())
}

fn parse_u64(s: &str) -> Result<u64> {
    s.parse::<u64>()
        .map_err(|_| Error::Invalid(format!("not a u64: {s}")))
}

/// Right-pad with spaces, hard-truncate if longer.
fn pad_truncate(s: &str, width: usize) -> String {
    if s.len() >= width {
        s[..width].to_owned()
    } else {
        format!("{s:<width$}")
    }
}

fn push_94(out: &mut String, line: &str) -> Result<()> {
    if line.len() != 94 {
        return Err(Error::Invalid(format!(
            "NACHA record must be 94 chars, got {}",
            line.len()
        )));
    }
    out.push_str(line);
    out.push('\n');
    Ok(())
}

impl Batch {
    /// Helper: confirm an amount in cents matches our currency
    /// (USD-only safety check used by the NACHA generator).
    fn amount_for_currency_in_cents(&self, amount_cents: u64) -> Result<u64> {
        if self.currency != Currency::USD {
            return Err(Error::CurrencyMismatch {
                batch: self.currency.code().to_owned(),
                tx: "USD".to_owned(),
            });
        }
        let _ = Money::from_minor(
            i64::try_from(amount_cents).map_err(|_| op_core::Error::Overflow)?,
            Currency::USD,
        );
        Ok(amount_cents)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::holdback::HoldbackPolicy;
    use op_core::{Currency, Money};
    use op_ledger::TransactionId;

    fn profile() -> NachaProfile {
        NachaProfile {
            odfi_routing: "121000248".into(),
            immediate_origin: "1234567890".into(),
            immediate_destination: "0210000211".into(),
            company_name: "OPENPAY VENDOR INC".into(),
            company_id: "9876543210".into(),
            company_entry_description: "SETTLEMNT".into(),
            effective_entry_date: "261122".into(),
        }
    }

    fn batch_with_two_entries() -> Batch {
        let mut b = Batch::open(Currency::USD, PayoutRail::AchNacha, 1_000);
        b.add_entry(
            TransactionId::new(),
            Money::from_minor(750_000, Currency::USD),
            Some("o-1".into()),
        )
        .unwrap();
        b.add_entry(
            TransactionId::new(),
            Money::from_minor(250_000, Currency::USD),
            Some("o-2".into()),
        )
        .unwrap();
        b.close(
            HoldbackPolicy::none()
                .compute(b.gross().unwrap(), 0)
                .unwrap(),
            2_000,
        )
        .unwrap();
        b
    }

    fn two_credits() -> Vec<NachaCredit> {
        vec![
            NachaCredit {
                rdfi_routing: "021000021".into(),
                account_number: "1111111111".into(),
                receiver_name: "ALICE EXAMPLE".into(),
                amount_cents: 750_000,
                individual_id: "ALICE-1".into(),
                sec: SecCode::Ppd,
            },
            NachaCredit {
                rdfi_routing: "121000358".into(),
                account_number: "2222222222".into(),
                receiver_name: "BOB EXAMPLE".into(),
                amount_cents: 250_000,
                individual_id: "BOB-1".into(),
                sec: SecCode::Ppd,
            },
        ]
    }

    #[test]
    fn rejects_non_nacha_rail() {
        let mut b = Batch::open(Currency::USD, PayoutRail::SepaCt, 1_000);
        b.add_entry(
            TransactionId::new(),
            Money::from_minor(100, Currency::USD),
            None,
        )
        .unwrap();
        b.close(
            HoldbackPolicy::none()
                .compute(b.gross().unwrap(), 0)
                .unwrap(),
            2_000,
        )
        .unwrap();
        let err = nacha_file(&b, &profile(), &two_credits()).unwrap_err();
        assert!(matches!(err, Error::Invalid(_)));
    }

    #[test]
    fn rejects_empty_credits() {
        let b = Batch::open(Currency::USD, PayoutRail::AchNacha, 1_000);
        let err = nacha_file(&b, &profile(), &[]).unwrap_err();
        assert!(matches!(err, Error::EmptyBatch));
    }

    #[test]
    fn happy_path_produces_block_aligned_file() {
        let b = batch_with_two_entries();
        let file = nacha_file(&b, &profile(), &two_credits()).unwrap();
        // Record count: file header + batch header + 2 details +
        // batch control + file control = 6, padded to 10.
        let lines: Vec<&str> = file.lines().collect();
        assert_eq!(lines.len(), 10);
        // All records exactly 94 chars.
        for line in &lines {
            assert_eq!(line.len(), 94, "record not 94 chars: {line}");
        }
        // First record starts with `1` (file header).
        assert!(lines[0].starts_with('1'));
        // Last padded line is all 9s.
        assert_eq!(lines[9], "9".repeat(94));
    }

    #[test]
    fn batch_control_credit_total_matches() {
        let b = batch_with_two_entries();
        let file = nacha_file(&b, &profile(), &two_credits()).unwrap();
        let lines: Vec<&str> = file.lines().collect();
        // Batch control is line index 4 (0:file 1:batch_hdr 2:e1
        // 3:e2 4:batch_ctl 5:file_ctl).
        let batch_ctl = lines[4];
        assert!(batch_ctl.starts_with('8'));
        // The credit total occupies a fixed window — easier to
        // smoke-test by substring search for the expected total.
        // 10_000.00 = 1_000_000 cents → "000001000000".
        assert!(
            batch_ctl.contains("000001000000"),
            "batch control missing expected credit total: {batch_ctl}"
        );
    }

    #[test]
    fn validates_routing_numbers() {
        let b = batch_with_two_entries();
        let mut bad = two_credits();
        bad[0].rdfi_routing = "12345".into();
        let err = nacha_file(&b, &profile(), &bad).unwrap_err();
        assert!(matches!(err, Error::Invalid(_)));
    }
}
