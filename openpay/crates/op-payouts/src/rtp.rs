//! TCH Real-Time Payments (RTP) payout driver.
//!
//! RTP is The Clearing House's instant-payments rail. Messages are
//! ISO 20022 `pacs.008.001.08` profiled by TCH. The per-payment limit
//! rose to USD 10,000,000 in February 2025.

use uuid::Uuid;

use crate::error::{Error, Result};
use crate::payout::{
    BeneficiaryAccount, Payout, PayoutRequest, PayoutResult, PayoutStatus,
};
use crate::visa_direct::format_amount;

/// RTP per-payment cap (USD, post-Feb-2025).
pub const RTP_MAX_AMOUNT_USD: i64 = 10_000_000 * 100;

/// RTP driver.
#[derive(Clone, Debug, Default)]
pub struct RtpDriver {
    /// Sender's 9-digit ABA (must be an RTP participant).
    pub sender_aba: String,
    /// Sender's account number.
    pub sender_account: String,
    /// Sender's name (RTP caps at 140 chars).
    pub sender_name: String,
}

impl Payout for RtpDriver {
    fn rail(&self) -> &'static str {
        "rtp"
    }

    fn submit(&self, req: &PayoutRequest) -> Result<PayoutResult> {
        if req.amount.currency != op_core::Currency::USD {
            return Err(Error::LimitViolation {
                rail: "rtp",
                detail: "RTP is USD-only".to_string(),
            });
        }
        if !req.amount.is_positive() {
            return Err(Error::LimitViolation {
                rail: "rtp",
                detail: "amount must be positive".to_string(),
            });
        }
        if req.amount.minor_units > RTP_MAX_AMOUNT_USD {
            return Err(Error::LimitViolation {
                rail: "rtp",
                detail: "RTP per-payment cap is USD 10,000,000".to_string(),
            });
        }
        let (aba, account) = match &req.beneficiary.account {
            BeneficiaryAccount::UsBank { aba, account, .. } => (aba, account),
            _ => return Err(Error::UnsupportedMethod { rail: "rtp" }),
        };
        if aba.len() != 9 || !aba.chars().all(|c| c.is_ascii_digit()) {
            return Err(Error::InvalidBeneficiary {
                rail: "rtp",
                detail: "ABA must be 9 digits".to_string(),
            });
        }
        let amt = format_amount(req.amount);
        let xml = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
            <Document xmlns=\"urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08\">\
            <FIToFICstmrCdtTrf><GrpHdr><MsgId>{msg}</MsgId><NbOfTxs>1</NbOfTxs>\
            <SttlmInf><SttlmMtd>CLRG</SttlmMtd><ClrSys><Prtry>TCH</Prtry></ClrSys></SttlmInf></GrpHdr>\
            <CdtTrfTxInf><PmtId><EndToEndId>{e2e}</EndToEndId><UETR>{uetr}</UETR></PmtId>\
            <IntrBkSttlmAmt Ccy=\"USD\">{amt}</IntrBkSttlmAmt>\
            <Dbtr><Nm>{dn}</Nm></Dbtr>\
            <DbtrAcct><Id><Othr><Id>{da}</Id></Othr></Id></DbtrAcct>\
            <DbtrAgt><FinInstnId><ClrSysMmbId><MmbId>{daba}</MmbId></ClrSysMmbId></FinInstnId></DbtrAgt>\
            <CdtrAgt><FinInstnId><ClrSysMmbId><MmbId>{caba}</MmbId></ClrSysMmbId></FinInstnId></CdtrAgt>\
            <Cdtr><Nm>{cn}</Nm></Cdtr>\
            <CdtrAcct><Id><Othr><Id>{ca}</Id></Othr></Id></CdtrAcct>\
            </CdtTrfTxInf></FIToFICstmrCdtTrf></Document>",
            msg = req.idempotency_key,
            e2e = req.idempotency_key,
            uetr = Uuid::new_v4(),
            dn = self.sender_name,
            da = self.sender_account,
            daba = self.sender_aba,
            caba = aba,
            cn = req.beneficiary.name,
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
            "RTP status arrives as pacs.002 from the TCH gateway".to_string(),
        ))
    }
}
