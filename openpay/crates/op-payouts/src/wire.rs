//! Wire-transfer driver: Fedwire, SWIFT MT103, ISO 20022 `pacs.008`.
//!
//! Three message families share a driver because they share a routing
//! shape: BIC + account, free-form remittance, single high-value
//! payment. The driver selects the format from
//! [`PayoutMethod`](crate::PayoutMethod):
//!
//! - `Fedwire` → Fedwire Format 1000 / 1500 type/subtype encoding.
//! - `SwiftMt103` → MT103 with `:50K:` / `:59:` blocks.
//! - `Pacs008` → ISO 20022 `pacs.008.001.08` XML stub.

use uuid::Uuid;

use crate::error::{Error, Result};
use crate::payout::{
    BeneficiaryAccount, Payout, PayoutMethod, PayoutRequest, PayoutResult, PayoutStatus,
};
use crate::visa_direct::format_amount;

/// Combined wire driver.
#[derive(Clone, Debug, Default)]
pub struct WireDriver {
    /// Sender's BIC (for MT103 / pacs.008) or ABA (for Fedwire).
    pub sender_id: String,
    /// Sender's name as it should appear in the wire.
    pub sender_name: String,
}

impl WireDriver {
    fn build_fedwire(&self, req: &PayoutRequest, account: &str, beneficiary_id: &str) -> Vec<u8> {
        // Fedwire Format: simplified placeholder showing the type/subtype,
        // amount, beneficiary, originator, OBI tags. Real Fedwire is
        // tag-based ASCII; here we emit a deterministic representation
        // operators can post-process.
        let mut s = String::new();
        s.push_str("{1500}30");
        s.push_str(&format!("{{2000}}{}", format_amount(req.amount)));
        s.push_str(&format!("{{3400}}{}", self.sender_id));
        s.push_str(&format!("{{3600}}{beneficiary_id}"));
        s.push_str(&format!("{{4200}}{}/{}", req.beneficiary.name, account));
        s.push_str(&format!(
            "{{5000}}{}/{}",
            self.sender_name, req.idempotency_key
        ));
        if let Some(memo) = &req.memo {
            s.push_str(&format!("{{6000}}{memo}"));
        }
        s.into_bytes()
    }

    fn build_mt103(&self, req: &PayoutRequest, account: &str, beneficiary_bic: &str) -> Vec<u8> {
        let mut s = String::new();
        s.push_str(":20:");
        s.push_str(&req.idempotency_key);
        s.push('\n');
        s.push_str(":23B:CRED\n");
        s.push_str(":32A:000000");
        s.push_str(req.amount.currency.code());
        s.push_str(&format_amount(req.amount));
        s.push('\n');
        s.push_str(":50K:/");
        s.push_str(&self.sender_id);
        s.push('\n');
        s.push_str(&self.sender_name);
        s.push('\n');
        s.push_str(":57A:");
        s.push_str(beneficiary_bic);
        s.push('\n');
        s.push_str(":59:/");
        s.push_str(account);
        s.push('\n');
        s.push_str(&req.beneficiary.name);
        s.push('\n');
        if let Some(memo) = &req.memo {
            s.push_str(":70:");
            s.push_str(memo);
            s.push('\n');
        }
        s.push_str(":71A:OUR\n");
        s.into_bytes()
    }

    fn build_pacs008(
        &self,
        req: &PayoutRequest,
        account: &str,
        beneficiary_bic: &str,
    ) -> Vec<u8> {
        let amt = format_amount(req.amount);
        let ccy = req.amount.currency.code();
        let xml = format!(
            "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\
            <Document xmlns=\"urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08\">\
            <FIToFICstmrCdtTrf><GrpHdr><MsgId>{msg}</MsgId><NbOfTxs>1</NbOfTxs>\
            <SttlmInf><SttlmMtd>CLRG</SttlmMtd></SttlmInf></GrpHdr>\
            <CdtTrfTxInf><PmtId><EndToEndId>{e2e}</EndToEndId><UETR>{uetr}</UETR></PmtId>\
            <IntrBkSttlmAmt Ccy=\"{ccy}\">{amt}</IntrBkSttlmAmt>\
            <Dbtr><Nm>{dn}</Nm></Dbtr>\
            <DbtrAgt><FinInstnId><BICFI>{sb}</BICFI></FinInstnId></DbtrAgt>\
            <CdtrAgt><FinInstnId><BICFI>{cb}</BICFI></FinInstnId></CdtrAgt>\
            <Cdtr><Nm>{cn}</Nm></Cdtr>\
            <CdtrAcct><Id><Othr><Id>{acct}</Id></Othr></Id></CdtrAcct>\
            </CdtTrfTxInf></FIToFICstmrCdtTrf></Document>",
            msg = req.idempotency_key,
            e2e = req.idempotency_key,
            uetr = Uuid::new_v4(),
            dn = self.sender_name,
            sb = self.sender_id,
            cb = beneficiary_bic,
            cn = req.beneficiary.name,
            acct = account,
        );
        xml.into_bytes()
    }
}

impl Payout for WireDriver {
    fn rail(&self) -> &'static str {
        "wire"
    }

    fn submit(&self, req: &PayoutRequest) -> Result<PayoutResult> {
        if !req.amount.is_positive() {
            return Err(Error::LimitViolation {
                rail: "wire",
                detail: "amount must be positive".to_string(),
            });
        }
        let payload = match (&req.method, &req.beneficiary.account) {
            (PayoutMethod::Fedwire, BeneficiaryAccount::UsBank { aba, account, .. }) => {
                self.build_fedwire(req, account, aba)
            }
            (PayoutMethod::SwiftMt103, BeneficiaryAccount::SwiftBic { bic, account }) => {
                self.build_mt103(req, account, bic)
            }
            (PayoutMethod::Pacs008, BeneficiaryAccount::SwiftBic { bic, account }) => {
                self.build_pacs008(req, account, bic)
            }
            (PayoutMethod::Pacs008, BeneficiaryAccount::Iban(iban)) => {
                self.build_pacs008(req, iban, "NOTPROVIDED")
            }
            _ => return Err(Error::UnsupportedMethod { rail: "wire" }),
        };
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
            "wire status is delivered via MT199 / pacs.002 out-of-band".to_string(),
        ))
    }
}
