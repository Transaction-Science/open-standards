//! UK Faster Payments Service (FPS) driver.
//!
//! FPS settles in seconds, GBP only, on 6-digit sort code + 8-digit
//! account number. Single-immediate-payment messages travel as Pay.UK
//! `FPS Single Immediate Payment` (SIP) — under the hood this is an
//! ISO 20022 `pacs.008.001.08` with `SvcLvl/Cd = FPSI` plus a
//! `LclInstrm/Prtry` of `SIP`.

use uuid::Uuid;

use crate::error::{Error, Result};
use crate::payout::{
    BeneficiaryAccount, Payout, PayoutRequest, PayoutResult, PayoutStatus,
};
use crate::visa_direct::format_amount;

/// UK FPS driver.
#[derive(Clone, Debug, Default)]
pub struct UkFpsDriver {
    /// Sender's sort code (6 digits).
    pub sender_sort_code: String,
    /// Sender's account (8 digits).
    pub sender_account: String,
    /// Sender's name.
    pub sender_name: String,
}

impl Payout for UkFpsDriver {
    fn rail(&self) -> &'static str {
        "uk_fps"
    }

    fn submit(&self, req: &PayoutRequest) -> Result<PayoutResult> {
        if req.amount.currency != op_core::Currency::GBP {
            return Err(Error::LimitViolation {
                rail: "uk_fps",
                detail: "FPS is GBP-only".to_string(),
            });
        }
        if !req.amount.is_positive() {
            return Err(Error::LimitViolation {
                rail: "uk_fps",
                detail: "amount must be positive".to_string(),
            });
        }
        // FPS SIP per-transaction limit as of April 2024 is GBP 1,000,000.
        if req.amount.minor_units > 1_000_000 * 100 {
            return Err(Error::LimitViolation {
                rail: "uk_fps",
                detail: "FPS per-transaction cap is GBP 1,000,000".to_string(),
            });
        }
        let (sort_code, account) = match &req.beneficiary.account {
            BeneficiaryAccount::UkSortCode { sort_code, account } => (sort_code, account),
            _ => return Err(Error::UnsupportedMethod { rail: "uk_fps" }),
        };
        if sort_code.len() != 6 || !sort_code.chars().all(|c| c.is_ascii_digit()) {
            return Err(Error::InvalidBeneficiary {
                rail: "uk_fps",
                detail: "sort code must be 6 digits".to_string(),
            });
        }
        if account.len() != 8 || !account.chars().all(|c| c.is_ascii_digit()) {
            return Err(Error::InvalidBeneficiary {
                rail: "uk_fps",
                detail: "account number must be 8 digits".to_string(),
            });
        }
        let amt = format_amount(req.amount);
        let xml = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
            <Document xmlns=\"urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08\">\
            <FIToFICstmrCdtTrf><GrpHdr><MsgId>{msg}</MsgId><NbOfTxs>1</NbOfTxs></GrpHdr>\
            <CdtTrfTxInf><PmtId><EndToEndId>{e2e}</EndToEndId><UETR>{uetr}</UETR></PmtId>\
            <PmtTpInf><SvcLvl><Cd>FPSI</Cd></SvcLvl><LclInstrm><Prtry>SIP</Prtry></LclInstrm></PmtTpInf>\
            <IntrBkSttlmAmt Ccy=\"GBP\">{amt}</IntrBkSttlmAmt>\
            <Dbtr><Nm>{dn}</Nm></Dbtr>\
            <DbtrAcct><Id><Othr><Id>{ds}{da}</Id></Othr></Id></DbtrAcct>\
            <Cdtr><Nm>{cn}</Nm></Cdtr>\
            <CdtrAcct><Id><Othr><Id>{cs}{ca}</Id></Othr></Id></CdtrAcct>\
            </CdtTrfTxInf></FIToFICstmrCdtTrf></Document>",
            msg = req.idempotency_key,
            e2e = req.idempotency_key,
            uetr = Uuid::new_v4(),
            dn = self.sender_name,
            ds = self.sender_sort_code,
            da = self.sender_account,
            cn = req.beneficiary.name,
            cs = sort_code,
            ca = account,
        );
        Ok(PayoutResult {
            idempotency_key: req.idempotency_key.clone(),
            payout_id: Uuid::now_v7().to_string(),
            status: PayoutStatus::PreparedOffline,
            raw_status: None,
            reason_code: None,
            reason_text: None,
            rail_txn_id: None,
            settled_amount: Some(req.amount),
            wire_payload: Some(xml.into_bytes()),
        })
    }

    fn status(&self, _payout_id: &str) -> Result<PayoutResult> {
        Err(Error::DriverValidation(
            "FPS status returns as pacs.002 from the Pay.UK direct-participant gateway".to_string(),
        ))
    }
}
