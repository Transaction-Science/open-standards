//! SEPA Credit Transfer (SCT) and SEPA Instant Credit Transfer (SCT Inst).
//!
//! Both are EUR-only credit transfers on `pacs.008.001.08`. SCT settles
//! D+1 batched through ASCT; SCT Inst settles in <10 s via RT1 / TIPS.
//! This driver emits the wire XML and lets the operator's RT1 / TIPS
//! / ASCT bridge transmit it.

use uuid::Uuid;

use crate::error::{Error, Result};
use crate::payout::{
    BeneficiaryAccount, Payout, PayoutMethod, PayoutRequest, PayoutResult, PayoutStatus,
};
use crate::visa_direct::format_amount;

/// Lightweight IBAN structural validation (length 15–34, alphanum,
/// mod-97 == 1).
fn validate_iban(raw: &str) -> Result<()> {
    let cleaned: String = raw.chars().filter(|c| !c.is_whitespace()).collect();
    if cleaned.len() < 15 || cleaned.len() > 34 {
        return Err(Error::InvalidBeneficiary {
            rail: "sepa",
            detail: "IBAN length out of range".to_string(),
        });
    }
    if !cleaned.chars().all(|c| c.is_ascii_alphanumeric()) {
        return Err(Error::InvalidBeneficiary {
            rail: "sepa",
            detail: "IBAN must be alphanumeric".to_string(),
        });
    }
    // Move first 4 chars to the end and run mod-97.
    let (head, tail) = cleaned.split_at(4);
    let rearranged = format!("{tail}{head}");
    let mut acc: u32 = 0;
    for ch in rearranged.chars() {
        let digits: String = if ch.is_ascii_digit() {
            ch.to_string()
        } else {
            (u32::from(ch.to_ascii_uppercase()) - u32::from('A') + 10).to_string()
        };
        for d in digits.chars() {
            acc = acc * 10 + d.to_digit(10).unwrap_or(0);
            acc %= 97;
        }
    }
    if acc != 1 {
        return Err(Error::InvalidBeneficiary {
            rail: "sepa",
            detail: "IBAN checksum failed mod-97".to_string(),
        });
    }
    Ok(())
}

/// SEPA driver covers SCT and SCT Inst.
#[derive(Clone, Debug, Default)]
pub struct SepaDriver {
    /// Sender's BIC.
    pub sender_bic: String,
    /// Sender's name.
    pub sender_name: String,
    /// Sender's IBAN.
    pub sender_iban: String,
}

impl SepaDriver {
    fn build_xml(&self, req: &PayoutRequest, iban: &str, service_level: &str) -> Vec<u8> {
        let amt = format_amount(req.amount);
        let ccy = req.amount.currency.code();
        let xml = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
            <Document xmlns=\"urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08\">\
            <FIToFICstmrCdtTrf><GrpHdr><MsgId>{msg}</MsgId><NbOfTxs>1</NbOfTxs>\
            <SttlmInf><SttlmMtd>CLRG</SttlmMtd></SttlmInf></GrpHdr>\
            <CdtTrfTxInf><PmtId><EndToEndId>{e2e}</EndToEndId><UETR>{uetr}</UETR></PmtId>\
            <PmtTpInf><SvcLvl><Cd>{svc}</Cd></SvcLvl></PmtTpInf>\
            <IntrBkSttlmAmt Ccy=\"{ccy}\">{amt}</IntrBkSttlmAmt>\
            <Dbtr><Nm>{dn}</Nm></Dbtr>\
            <DbtrAcct><Id><IBAN>{di}</IBAN></Id></DbtrAcct>\
            <DbtrAgt><FinInstnId><BICFI>{db}</BICFI></FinInstnId></DbtrAgt>\
            <Cdtr><Nm>{cn}</Nm></Cdtr>\
            <CdtrAcct><Id><IBAN>{ci}</IBAN></Id></CdtrAcct>\
            </CdtTrfTxInf></FIToFICstmrCdtTrf></Document>",
            msg = req.idempotency_key,
            e2e = req.idempotency_key,
            uetr = Uuid::new_v4(),
            svc = service_level,
            dn = self.sender_name,
            di = self.sender_iban,
            db = self.sender_bic,
            cn = req.beneficiary.name,
            ci = iban,
        );
        xml.into_bytes()
    }
}

impl Payout for SepaDriver {
    fn rail(&self) -> &'static str {
        "sepa"
    }

    fn submit(&self, req: &PayoutRequest) -> Result<PayoutResult> {
        if req.amount.currency != op_core::Currency::EUR {
            return Err(Error::LimitViolation {
                rail: "sepa",
                detail: "SEPA is EUR-only".to_string(),
            });
        }
        if !req.amount.is_positive() {
            return Err(Error::LimitViolation {
                rail: "sepa",
                detail: "amount must be positive".to_string(),
            });
        }
        let iban = match &req.beneficiary.account {
            BeneficiaryAccount::Iban(iban) => iban,
            _ => return Err(Error::UnsupportedMethod { rail: "sepa" }),
        };
        validate_iban(iban)?;
        let service_level = match req.method {
            PayoutMethod::SepaSct => "SEPA",
            PayoutMethod::SepaSctInst => {
                if req.amount.minor_units > 100_000 * 100 {
                    return Err(Error::LimitViolation {
                        rail: "sepa",
                        detail: "SCT Inst per-transaction cap is EUR 100k".to_string(),
                    });
                }
                "SEPA"
            }
            _ => return Err(Error::UnsupportedMethod { rail: "sepa" }),
        };
        let payload = self.build_xml(req, iban, service_level);
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
            "SEPA status arrives via pacs.002 from the operator's RT1/TIPS bridge".to_string(),
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::validate_iban;

    #[test]
    fn iban_de_valid() {
        // DE89 3704 0044 0532 0130 00 — standard textbook valid IBAN.
        assert!(validate_iban("DE89370400440532013000").is_ok());
    }

    #[test]
    fn iban_rejects_bad_checksum() {
        assert!(validate_iban("DE00370400440532013000").is_err());
    }
}
