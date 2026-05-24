//! NACHA ACH credit-payout driver.
//!
//! Builds a single-credit NACHA file fragment:
//!
//! - File Header (`'1'`) — left to the batch builder downstream.
//! - Batch Header (`'5'`)
//! - Entry Detail (`'6'`) — credit transaction code 22 (checking) /
//!   32 (savings) for PPD; 23/33 for prenotes; we emit 22 or 32.
//! - Entry-Detail addenda are not generated (PPD credits do not require
//!   an addenda record; CCD may add one, exposed via `addenda`).
//!
//! Reference: NACHA 2026 Operating Rules, sections OR.4 and OR.5.

use uuid::Uuid;

use crate::error::{Error, Result};
use crate::payout::{
    BeneficiaryAccount, Payout, PayoutMethod, PayoutRequest, PayoutResult, PayoutStatus,
};

/// ACH credit driver.
#[derive(Clone, Debug, Default)]
pub struct AchCreditDriver {
    /// Operator's company name (16 chars max, used in batch header
    /// field `CompanyName`).
    pub company_name: String,
    /// Operator's company identifier (10 chars, IRS EIN with `'1'`
    /// prefix per NACHA).
    pub company_id: String,
    /// Originating DFI's 8-digit routing prefix.
    pub originating_dfi_id: String,
}

fn fixed_left(s: &str, n: usize) -> String {
    let trimmed: String = s.chars().take(n).collect();
    let pad = n.saturating_sub(trimmed.chars().count());
    let mut out = trimmed;
    out.extend(std::iter::repeat_n(' ', pad));
    out
}

fn fixed_right_num(value: u64, n: usize) -> String {
    let s = value.to_string();
    if s.len() >= n {
        s[s.len() - n..].to_string()
    } else {
        let pad = n - s.len();
        let mut out = String::with_capacity(n);
        out.extend(std::iter::repeat_n('0', pad));
        out.push_str(&s);
        out
    }
}

impl Payout for AchCreditDriver {
    fn rail(&self) -> &'static str {
        "ach_credit"
    }

    fn submit(&self, req: &PayoutRequest) -> Result<PayoutResult> {
        let sec_code = match &req.method {
            PayoutMethod::AchCredit { sec_code } => sec_code.as_str(),
            _ => return Err(Error::UnsupportedMethod { rail: "ach_credit" }),
        };
        if !matches!(sec_code, "PPD" | "CCD") {
            return Err(Error::DriverValidation(format!(
                "unsupported SEC code: {sec_code}"
            )));
        }
        let (aba, account, account_type) = match &req.beneficiary.account {
            BeneficiaryAccount::UsBank {
                aba,
                account,
                account_type,
            } => (aba, account, account_type),
            _ => return Err(Error::UnsupportedMethod { rail: "ach_credit" }),
        };
        if aba.len() != 9 || !aba.chars().all(|c| c.is_ascii_digit()) {
            return Err(Error::InvalidBeneficiary {
                rail: "ach_credit",
                detail: "ABA must be 9 digits".to_string(),
            });
        }
        if req.amount.currency != op_core::Currency::USD {
            return Err(Error::LimitViolation {
                rail: "ach_credit",
                detail: "ACH credits are USD-only".to_string(),
            });
        }
        if !req.amount.is_positive() {
            return Err(Error::LimitViolation {
                rail: "ach_credit",
                detail: "amount must be positive".to_string(),
            });
        }
        let txn_code = match account_type.as_str() {
            "CHECKING" => "22",
            "SAVINGS" => "32",
            other => {
                return Err(Error::DriverValidation(format!(
                    "unsupported account_type: {other}"
                )));
            }
        };
        let receiving_dfi = &aba[..8];
        let check_digit = &aba[8..];
        let amount_field = fixed_right_num(req.amount.minor_units.unsigned_abs(), 10);
        let entry = format!(
            "6{txn_code}{receiving_dfi}{check_digit}{account}{amount}{indiv_id}{name}{disc:>2}0{trace}",
            account = fixed_left(account, 17),
            amount = amount_field,
            indiv_id = fixed_left(&req.idempotency_key, 15),
            name = fixed_left(&req.beneficiary.name, 22),
            disc = "",
            trace = fixed_right_num(1, 15),
        );
        let batch_header = format!(
            "5220{company}{disc}{company_id}{sec}{entry_desc}{eff:>6}{settle:>3}1{orig:>8}0000001",
            company = fixed_left(&self.company_name, 16),
            disc = fixed_left("", 20),
            company_id = fixed_left(&self.company_id, 10),
            sec = sec_code,
            entry_desc = fixed_left("PAYOUT", 10),
            eff = "      ",
            settle = "   ",
            orig = fixed_left(&self.originating_dfi_id, 8),
        );
        let payload = format!("{batch_header}\n{entry}\n").into_bytes();
        Ok(PayoutResult {
            idempotency_key: req.idempotency_key.clone(),
            payout_id: Uuid::now_v7().to_string(),
            status: PayoutStatus::PreparedOffline,
            raw_status: None,
            reason_code: None,
            reason_text: None,
            rail_txn_id: None,
            settled_amount: Some(req.amount),
            wire_payload: Some(payload),
        })
    }

    fn status(&self, _payout_id: &str) -> Result<PayoutResult> {
        Err(Error::DriverValidation(
            "ACH status is returned out-of-band via ACH return files".to_string(),
        ))
    }
}
