//! SEPA Instant unified client.
//!
//! One driver, two backends. The message shape is identical per EPC SCT
//! Inst IG 2019 v1.0; only the endpoint URL and a few BIC routing
//! conventions differ.

use std::time::Duration;

use op_core::{A2aKey, PaymentMethod};
use op_iso20022::CreditTransferBuilder;
use op_iso20022::bah::PartyIdentification;
use op_iso20022::profile::SepaInstant;

use crate::acquirer::{
    A2aAcquirer, A2aDecision, A2aStatus, CreditTransferReq, ParticipantId, StatusQueryReq,
};
use crate::error::{Error, Result};

use super::status_map;

/// Which SEPA Instant backend the client targets.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum SepaInstantBackend {
    /// EBA Clearing RT1.
    Rt1,
    /// ECB TIPS.
    Tips,
}

impl SepaInstantBackend {
    /// Driver name for telemetry.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Rt1 => "sepa-rt1",
            Self::Tips => "sepa-tips",
        }
    }
}

/// SEPA SCT Inst HTTPS client.
#[derive(Clone)]
pub struct SepaInstantClient {
    backend: SepaInstantBackend,
    base_url: String,
    agent: ureq::Agent,
    sender_bic: String,
}

impl SepaInstantClient {
    /// Construct with operator-provided agent (must have mTLS cert).
    #[must_use]
    pub fn new(
        backend: SepaInstantBackend,
        base_url: impl Into<String>,
        agent: ureq::Agent,
        sender_bic: impl Into<String>,
    ) -> Self {
        Self {
            backend,
            base_url: base_url.into(),
            agent,
            sender_bic: sender_bic.into(),
        }
    }

    /// Construct with a default agent (tests only — no mTLS).
    #[must_use]
    pub fn new_unsecured(
        backend: SepaInstantBackend,
        base_url: impl Into<String>,
        sender_bic: impl Into<String>,
    ) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(15))
            .build();
        Self::new(backend, base_url, agent, sender_bic)
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url.trim_end_matches('/'), path)
    }

    fn validate(req: &CreditTransferReq) -> Result<()> {
        if req.amount.currency != op_core::Currency::EUR {
            return Err(Error::CurrencyMismatch {
                rail: "sepa-instant",
                expected: "EUR",
                got: req.amount.currency.code().to_owned(),
            });
        }
        match (&req.debtor_agent, &req.creditor_agent) {
            (ParticipantId::Bic(d), ParticipantId::Bic(c)) => {
                Self::validate_bic(d)?;
                Self::validate_bic(c)?;
                Ok(())
            }
            _ => Err(Error::UnsupportedA2aKey {
                rail: "sepa-instant",
            }),
        }
    }

    /// BIC must be 8 or 11 alphanumeric uppercase chars.
    fn validate_bic(s: &str) -> Result<()> {
        let len = s.len();
        if len != 8 && len != 11 {
            return Err(Error::DriverValidation(format!(
                "BIC must be 8 or 11 chars, got {len}"
            )));
        }
        if !s
            .bytes()
            .all(|b| b.is_ascii_uppercase() || b.is_ascii_digit())
        {
            return Err(Error::DriverValidation(
                "BIC must be uppercase ASCII alphanumeric".into(),
            ));
        }
        Ok(())
    }
}

impl A2aAcquirer for SepaInstantClient {
    fn name(&self) -> &'static str {
        self.backend.name()
    }

    fn submit_credit_transfer(&self, req: &CreditTransferReq) -> Result<A2aDecision> {
        Self::validate(req)?;

        let debtor = PartyIdentification::Bic(req.debtor_agent.as_str().to_owned());
        let creditor = PartyIdentification::Bic(req.creditor_agent.as_str().to_owned());
        let debtor_method = PaymentMethod::A2a(A2aKey::Iban(req.debtor_account.clone()));
        let creditor_method = PaymentMethod::A2a(A2aKey::Iban(req.creditor_account.clone()));

        let mut b = CreditTransferBuilder::<SepaInstant>::new()
            .amount(req.amount)
            .debtor(debtor_method)
            .creditor(creditor_method)
            .debtor_agent(debtor)
            .creditor_agent(creditor)
            .end_to_end_id(&req.end_to_end_id)
            .uetr(&req.uetr);
        if let Some(r) = &req.remittance {
            b = b.remittance(r);
        }
        let built = b.build()?;

        // SCT Inst pacs.008 requires `LclInstrm.Cd = INST` and `SttlmMtd = CLRG`.
        // We emit a SEPA-specific XML body directly here.
        let xml = emit_sepa_pacs008(&built, req, &self.sender_bic);

        let url = self.url("/v1/sct-inst/pacs008");
        let resp = self
            .agent
            .post(&url)
            .set("Content-Type", "application/xml; charset=utf-8")
            .set("x-sender-bic", &self.sender_bic)
            .set("x-idempotency-key", &req.idempotency_key)
            .send_string(&xml);

        let resp_xml = match resp {
            Ok(r) => r
                .into_string()
                .map_err(|e| Error::Transport(e.to_string()))?,
            Err(ureq::Error::Status(status, r)) => {
                return Err(Error::RailRejected {
                    status,
                    code: "http".into(),
                    message: r.into_string().unwrap_or_default(),
                });
            }
            Err(ureq::Error::Transport(t)) => return Err(Error::Transport(t.to_string())),
        };

        let parsed = crate::xml_common::parse_pacs002(&resp_xml)?;
        let status = status_map::map_transaction_status(&parsed.transaction_status)?;

        Ok(A2aDecision {
            status,
            raw_status: parsed.transaction_status,
            reason_code: parsed.reason_code,
            reason_text: parsed.reason_text,
            uetr: parsed.uetr.or_else(|| Some(req.uetr.clone())),
            rail_txn_id: parsed.original_end_to_end_id,
            settled_amount: if matches!(status, A2aStatus::Settled | A2aStatus::Accepted) {
                Some(req.amount)
            } else {
                None
            },
        })
    }

    fn query_status(&self, req: &StatusQueryReq) -> Result<A2aDecision> {
        let url = self.url(&format!("/v1/sct-inst/payments/{}/status", req.uetr));
        let resp = self.agent.get(&url).call().map_err(|e| match e {
            ureq::Error::Status(s, r) => Error::RailRejected {
                status: s,
                code: "http".into(),
                message: r.into_string().unwrap_or_default(),
            },
            ureq::Error::Transport(t) => Error::Transport(t.to_string()),
        })?;
        let xml = resp
            .into_string()
            .map_err(|e| Error::Transport(e.to_string()))?;
        let parsed = crate::xml_common::parse_pacs002(&xml)?;
        let status = status_map::map_transaction_status(&parsed.transaction_status)?;
        Ok(A2aDecision {
            status,
            raw_status: parsed.transaction_status,
            reason_code: parsed.reason_code,
            reason_text: parsed.reason_text,
            uetr: parsed.uetr.or_else(|| Some(req.uetr.clone())),
            rail_txn_id: parsed.original_end_to_end_id,
            settled_amount: None,
        })
    }
}

/// Emit the SEPA Instant pacs.008.001.08 body with the scheme-mandated
/// `LclInstrm.Cd = INST` and `SttlmMtd = CLRG`.
fn emit_sepa_pacs008(
    built: &op_iso20022::BuiltCreditTransfer<SepaInstant>,
    req: &CreditTransferReq,
    sender_bic: &str,
) -> String {
    let amount = crate::xml_common::format_money(built.amount);
    let ccy = built.amount.currency.code();
    let creation_dt = built
        .bah
        .creation_datetime
        .format(&time::format_description::well_known::Rfc3339)
        .unwrap_or_else(|_| "2026-01-01T00:00:00Z".into());

    let remittance = req
        .remittance
        .as_deref()
        .map(|r| format!("<RmtInf><Ustrd>{}</Ustrd></RmtInf>", xml_escape(r)))
        .unwrap_or_default();

    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pacs.008.001.08">
  <FIToFICstmrCdtTrf>
    <GrpHdr>
      <MsgId>{msg_id}</MsgId>
      <CreDtTm>{creation_dt}</CreDtTm>
      <NbOfTxs>1</NbOfTxs>
      <SttlmInf><SttlmMtd>CLRG</SttlmMtd></SttlmInf>
      <InstgAgt><FinInstnId><BICFI>{sender_bic}</BICFI></FinInstnId></InstgAgt>
    </GrpHdr>
    <CdtTrfTxInf>
      <PmtId>
        <InstrId>{end_to_end_id}</InstrId>
        <EndToEndId>{end_to_end_id}</EndToEndId>
        <UETR>{uetr}</UETR>
      </PmtId>
      <PmtTpInf><LclInstrm><Cd>INST</Cd></LclInstrm><SvcLvl><Cd>SEPA</Cd></SvcLvl></PmtTpInf>
      <IntrBkSttlmAmt Ccy="{ccy}">{amount}</IntrBkSttlmAmt>
      <ChrgBr>SLEV</ChrgBr>
      <Dbtr><Nm>{debtor_name}</Nm></Dbtr>
      <DbtrAcct><Id><IBAN>{debtor_iban}</IBAN></Id></DbtrAcct>
      <DbtrAgt><FinInstnId><BICFI>{debtor_bic}</BICFI></FinInstnId></DbtrAgt>
      <CdtrAgt><FinInstnId><BICFI>{creditor_bic}</BICFI></FinInstnId></CdtrAgt>
      <Cdtr><Nm>{creditor_name}</Nm></Cdtr>
      <CdtrAcct><Id><IBAN>{creditor_iban}</IBAN></Id></CdtrAcct>
      {remittance}
    </CdtTrfTxInf>
  </FIToFICstmrCdtTrf>
</Document>"#,
        msg_id = built.bah.business_message_id,
        creation_dt = creation_dt,
        end_to_end_id = built.end_to_end_id,
        uetr = built.uetr,
        ccy = ccy,
        amount = amount,
        sender_bic = xml_escape(sender_bic),
        debtor_name = xml_escape(&req.debtor_name),
        debtor_iban = xml_escape(&req.debtor_account),
        debtor_bic = xml_escape(req.debtor_agent.as_str()),
        creditor_bic = xml_escape(req.creditor_agent.as_str()),
        creditor_name = xml_escape(&req.creditor_name),
        creditor_iban = xml_escape(&req.creditor_account),
        remittance = remittance,
    )
}

fn xml_escape(s: &str) -> String {
    s.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
        .replace('\'', "&apos;")
}

#[cfg(test)]
mod tests {
    use super::*;
    use httpmock::prelude::*;
    use op_core::{Currency, Money};

    fn sample_req() -> CreditTransferReq {
        CreditTransferReq {
            uetr: "12a345b6-7c89-4d01-23e4-567890abcdef".into(),
            end_to_end_id: "E2E001".into(),
            amount: Money::from_minor(2500, Currency::EUR),
            debtor_agent: ParticipantId::Bic("DEUTDEFFXXX".into()),
            creditor_agent: ParticipantId::Bic("BNPAFRPPXXX".into()),
            debtor_account: "DE89370400440532013000".into(),
            creditor_account: "FR1420041010050500013M02606".into(),
            debtor_name: "Hans Müller".into(),
            creditor_name: "Marie Dupont".into(),
            remittance: Some("Rechnung 4242".into()),
            idempotency_key: "idem_sepa_001".into(),
        }
    }

    fn pacs002_accp(uetr: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<Document><FIToFIPmtStsRpt><TxInfAndSts>
  <OrgnlUETR>{uetr}</OrgnlUETR>
  <OrgnlEndToEndId>E2E001</OrgnlEndToEndId>
  <TxSts>ACCP</TxSts>
</TxInfAndSts></FIToFIPmtStsRpt></Document>"#
        )
    }

    #[test]
    fn rt1_submit_accp_returns_accepted() {
        let server = MockServer::start();
        let req = sample_req();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/sct-inst/pacs008")
                .header_exists("x-sender-bic");
            then.status(200).body(pacs002_accp(&req.uetr));
        });
        let client = SepaInstantClient::new_unsecured(
            SepaInstantBackend::Rt1,
            server.base_url(),
            "DEUTDEFFXXX",
        );
        let decision = client.submit_credit_transfer(&req).unwrap();
        mock.assert();
        assert_eq!(decision.status, A2aStatus::Accepted);
        assert_eq!(decision.raw_status, "ACCP");
    }

    #[test]
    fn tips_submit_acsc_returns_settled() {
        let server = MockServer::start();
        let req = sample_req();
        server.mock(|when, then| {
            when.method(POST).path("/v1/sct-inst/pacs008");
            then.status(200).body(format!(
                r"<Document><FIToFIPmtStsRpt><TxInfAndSts>
                <OrgnlUETR>{}</OrgnlUETR><OrgnlEndToEndId>E2E001</OrgnlEndToEndId>
                <TxSts>ACSC</TxSts></TxInfAndSts></FIToFIPmtStsRpt></Document>",
                req.uetr
            ));
        });
        let client = SepaInstantClient::new_unsecured(
            SepaInstantBackend::Tips,
            server.base_url(),
            "DEUTDEFFXXX",
        );
        let decision = client.submit_credit_transfer(&req).unwrap();
        assert_eq!(decision.status, A2aStatus::Settled);
        assert_eq!(client.name(), "sepa-tips");
    }

    #[test]
    fn rejects_non_eur() {
        let server = MockServer::start();
        let mut req = sample_req();
        req.amount = Money::from_minor(100, Currency::USD);
        let client = SepaInstantClient::new_unsecured(
            SepaInstantBackend::Rt1,
            server.base_url(),
            "DEUTDEFFXXX",
        );
        let err = client.submit_credit_transfer(&req).unwrap_err();
        assert!(matches!(
            err,
            Error::CurrencyMismatch {
                rail: "sepa-instant",
                expected: "EUR",
                ..
            }
        ));
    }

    #[test]
    fn rejects_non_bic_agents() {
        let server = MockServer::start();
        let mut req = sample_req();
        req.debtor_agent = ParticipantId::Aba("021000021".into());
        let client = SepaInstantClient::new_unsecured(
            SepaInstantBackend::Rt1,
            server.base_url(),
            "DEUTDEFFXXX",
        );
        let err = client.submit_credit_transfer(&req).unwrap_err();
        assert!(matches!(
            err,
            Error::UnsupportedA2aKey {
                rail: "sepa-instant"
            }
        ));
    }

    #[test]
    fn rejects_short_bic() {
        let server = MockServer::start();
        let mut req = sample_req();
        req.debtor_agent = ParticipantId::Bic("DEUT".into()); // 4 chars
        let client = SepaInstantClient::new_unsecured(
            SepaInstantBackend::Rt1,
            server.base_url(),
            "DEUTDEFFXXX",
        );
        let err = client.submit_credit_transfer(&req).unwrap_err();
        assert!(matches!(err, Error::DriverValidation(_)));
    }

    #[test]
    fn payload_carries_inst_local_instrument() {
        let server = MockServer::start();
        let req = sample_req();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/v1/sct-inst/pacs008")
                .body_contains("<Cd>INST</Cd>")
                .body_contains("<SvcLvl><Cd>SEPA</Cd></SvcLvl>")
                .body_contains("DE89370400440532013000");
            then.status(200).body(pacs002_accp(&req.uetr));
        });
        let client = SepaInstantClient::new_unsecured(
            SepaInstantBackend::Rt1,
            server.base_url(),
            "DEUTDEFFXXX",
        );
        client.submit_credit_transfer(&req).unwrap();
        mock.assert();
    }

    #[test]
    fn bic_validation() {
        assert!(SepaInstantClient::validate_bic("DEUTDEFFXXX").is_ok()); // 11
        assert!(SepaInstantClient::validate_bic("DEUTDEFF").is_ok()); // 8
        assert!(SepaInstantClient::validate_bic("DEUT").is_err()); // too short
        assert!(SepaInstantClient::validate_bic("DEUTDEFFXXXX").is_err()); // too long
        assert!(SepaInstantClient::validate_bic("deutdeffxxx").is_err()); // lowercase
        assert!(SepaInstantClient::validate_bic("DEUTDEFFXX!").is_err()); // special
    }

    #[test]
    fn backend_names() {
        assert_eq!(SepaInstantBackend::Rt1.name(), "sepa-rt1");
        assert_eq!(SepaInstantBackend::Tips.name(), "sepa-tips");
    }
}
