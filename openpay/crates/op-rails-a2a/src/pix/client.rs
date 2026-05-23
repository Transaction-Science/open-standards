//! PIX HTTPS client.
//!
//! Uses the OAuth 2.0 client-credentials flow with mTLS-bound access
//! tokens (RFC 8705). All XML message bodies are passed through the
//! operator's [`Signer`] before submission.
//!
//! ## What this driver expects of the operator
//!
//! - A `Signer` impl that wraps their HSM
//! - A pre-fetched OAuth access token, or a `TokenSource` that fetches
//!   one on demand
//! - mTLS certs configured on the `ureq::Agent` (operators construct
//!   the agent and inject it via [`PixClient::with_agent`])
//!
//! We deliberately keep mTLS configuration outside this crate because
//! cert provisioning is operator-specific (some use AWS `CloudHSM`, some
//! HSM-as-a-Service, some on-prem nCipher) and the `ureq` builder for
//! client certs varies by platform.

use std::sync::Arc;
use std::time::Duration;

use op_core::{A2aKey, PaymentMethod};
use op_iso20022::CreditTransferBuilder;
use op_iso20022::bah::PartyIdentification;
use op_iso20022::profile::Pix as PixProfile;

use crate::acquirer::{
    A2aAcquirer, A2aDecision, A2aStatus, CreditTransferReq, ParticipantId, StatusQueryReq,
};
use crate::error::{Error, Result};
use crate::signer::Signer;

use super::status_map;

/// Source of OAuth access tokens. Operators implement this to wrap
/// their Authorization Server. Returning a stale token will produce a
/// 401 from Bacen which the client surfaces as [`Error::RailRejected`].
pub trait TokenSource: Send + Sync {
    /// Return a currently-valid bearer token. The PIX client doesn't
    /// cache; cache inside your implementation.
    fn token(&self) -> Result<String>;
}

/// Static token holder. Useful for tests and short-lived deployments
/// that refresh out-of-band.
pub struct StaticToken(pub String);

impl TokenSource for StaticToken {
    fn token(&self) -> Result<String> {
        Ok(self.0.clone())
    }
}

/// PIX HTTPS client.
#[derive(Clone)]
pub struct PixClient {
    base_url: String,
    agent: ureq::Agent,
    token_source: Arc<dyn TokenSource>,
    signer: Arc<dyn Signer>,
    /// Operator's own ISPB (the sender).
    sender_ispb: String,
}

impl PixClient {
    /// Construct with operator-provided agent (must have mTLS cert),
    /// token source, signer, and sender ISPB.
    #[must_use]
    pub fn new(
        base_url: impl Into<String>,
        agent: ureq::Agent,
        token_source: Arc<dyn TokenSource>,
        signer: Arc<dyn Signer>,
        sender_ispb: impl Into<String>,
    ) -> Self {
        Self {
            base_url: base_url.into(),
            agent,
            token_source,
            signer,
            sender_ispb: sender_ispb.into(),
        }
    }

    /// Construct with a default agent (no mTLS — TESTS ONLY).
    /// Production must pass an agent with client certs configured.
    #[must_use]
    pub fn new_unsecured(
        base_url: impl Into<String>,
        token_source: Arc<dyn TokenSource>,
        signer: Arc<dyn Signer>,
        sender_ispb: impl Into<String>,
    ) -> Self {
        let agent = ureq::AgentBuilder::new()
            .timeout(Duration::from_secs(15))
            .build();
        Self::new(base_url, agent, token_source, signer, sender_ispb)
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url.trim_end_matches('/'), path)
    }

    fn validate(req: &CreditTransferReq) -> Result<()> {
        if req.amount.currency != op_core::Currency::BRL {
            return Err(Error::CurrencyMismatch {
                rail: "pix",
                expected: "BRL",
                got: req.amount.currency.code().to_owned(),
            });
        }
        match (&req.debtor_agent, &req.creditor_agent) {
            (ParticipantId::Ispb(d), ParticipantId::Ispb(c)) => {
                Self::validate_ispb(d)?;
                Self::validate_ispb(c)?;
                Ok(())
            }
            _ => Err(Error::UnsupportedA2aKey { rail: "pix" }),
        }
    }

    /// ISPB validation: 8 digits per Bacen.
    fn validate_ispb(s: &str) -> Result<()> {
        if s.len() != 8 || !s.chars().all(|c| c.is_ascii_digit()) {
            return Err(Error::DriverValidation(format!(
                "ISPB must be 8 digits, got {s:?}"
            )));
        }
        Ok(())
    }
}

impl A2aAcquirer for PixClient {
    fn name(&self) -> &'static str {
        "pix"
    }

    fn submit_credit_transfer(&self, req: &CreditTransferReq) -> Result<A2aDecision> {
        Self::validate(req)?;

        // Build the typed transfer via op-iso20022 (validates UETR, EndToEndId, BAH).
        // The internal clearing-system identifier is "ISPB" (the Bacen
        // participant-id scheme the PixProfile validates against); the
        // on-the-wire `<ClrSysId><Cd>` code is emitted separately by
        // `emit_pix_pacs008`.
        let debtor = PartyIdentification::ClearingSystemMemberId {
            clearing_system: "ISPB".into(),
            member_id: req.debtor_agent.as_str().to_owned(),
        };
        let creditor = PartyIdentification::ClearingSystemMemberId {
            clearing_system: "ISPB".into(),
            member_id: req.creditor_agent.as_str().to_owned(),
        };
        let debtor_method = PaymentMethod::A2a(A2aKey::Pix(req.debtor_account.clone()));
        let creditor_method = PaymentMethod::A2a(A2aKey::Pix(req.creditor_account.clone()));

        let mut b = CreditTransferBuilder::<PixProfile>::new()
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

        // Emit PIX-profile XML directly (Bacen SPI subset of pacs.008.001.08
        // with USABA replaced by BRSPB and IBAN replaced by /Othr/Id).
        let xml = emit_pix_pacs008(&built, req, &self.sender_ispb);

        // Sign.
        let signature = self
            .signer
            .sign(xml.as_bytes())
            .map_err(|e| Error::Signing(format!("{e}")))?;

        // Auth.
        let token = self.token_source.token()?;

        // POST to /pix/v2/pacs008.
        let url = self.url("/pix/v2/pacs008");
        let resp = self
            .agent
            .post(&url)
            .set("Authorization", &format!("Bearer {token}"))
            .set("Content-Type", "application/xml; charset=utf-8")
            .set("x-signature", &base64_encode(&signature))
            .set("x-signature-key-id", self.signer.key_id())
            .set("x-idempotency-key", &req.idempotency_key)
            .set("x-sender-ispb", &self.sender_ispb)
            .send_string(&xml);

        let resp_xml = match resp {
            Ok(r) => r
                .into_string()
                .map_err(|e| Error::Transport(e.to_string()))?,
            Err(ureq::Error::Status(status, r)) => {
                let body = r.into_string().unwrap_or_default();
                return Err(Error::RailRejected {
                    status,
                    code: "http".into(),
                    message: body,
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
        let token = self.token_source.token()?;
        let url = self.url(&format!("/pix/v2/payments/{}/status", req.uetr));
        let resp = self
            .agent
            .get(&url)
            .set("Authorization", &format!("Bearer {token}"))
            .call()
            .map_err(|e| match e {
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

/// Emit a Bacen-SPI-profile pacs.008.001.08 body.
///
/// Same schema namespace as `FedNow` / SEPA, but uses `BRSPB` as the
/// clearing system code and identifies accounts via `<Othr><Id>` (PIX
/// account holder is identified by `ChaveDict` or account number, not
/// IBAN).
fn emit_pix_pacs008(
    built: &op_iso20022::BuiltCreditTransfer<PixProfile>,
    req: &CreditTransferReq,
    _sender_ispb: &str,
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
    </GrpHdr>
    <CdtTrfTxInf>
      <PmtId>
        <InstrId>{end_to_end_id}</InstrId>
        <EndToEndId>{end_to_end_id}</EndToEndId>
        <UETR>{uetr}</UETR>
      </PmtId>
      <IntrBkSttlmAmt Ccy="{ccy}">{amount}</IntrBkSttlmAmt>
      <ChrgBr>SLEV</ChrgBr>
      <Dbtr><Nm>{debtor_name}</Nm></Dbtr>
      <DbtrAcct><Id><Othr><Id>{debtor_acct}</Id></Othr></Id></DbtrAcct>
      <DbtrAgt><FinInstnId><ClrSysMmbId>
        <ClrSysId><Cd>BRSPB</Cd></ClrSysId>
        <MmbId>{debtor_ispb}</MmbId>
      </ClrSysMmbId></FinInstnId></DbtrAgt>
      <CdtrAgt><FinInstnId><ClrSysMmbId>
        <ClrSysId><Cd>BRSPB</Cd></ClrSysId>
        <MmbId>{creditor_ispb}</MmbId>
      </ClrSysMmbId></FinInstnId></CdtrAgt>
      <Cdtr><Nm>{creditor_name}</Nm></Cdtr>
      <CdtrAcct><Id><Othr><Id>{creditor_acct}</Id></Othr></Id></CdtrAcct>
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
        debtor_name = xml_escape(&req.debtor_name),
        debtor_acct = xml_escape(&req.debtor_account),
        debtor_ispb = xml_escape(req.debtor_agent.as_str()),
        creditor_ispb = xml_escape(req.creditor_agent.as_str()),
        creditor_name = xml_escape(&req.creditor_name),
        creditor_acct = xml_escape(&req.creditor_account),
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

fn base64_encode(bytes: &[u8]) -> String {
    use base64::Engine;
    base64::engine::general_purpose::STANDARD.encode(bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::signer::NoOpSigner;
    use httpmock::prelude::*;
    use op_core::{Currency, Money};

    fn sample_req() -> CreditTransferReq {
        CreditTransferReq {
            uetr: "12a345b6-7c89-4d01-23e4-567890abcdef".into(),
            end_to_end_id: "E2E001".into(),
            amount: Money::from_minor(50_00, Currency::BRL),
            debtor_agent: ParticipantId::Ispb("00038166".into()), // Banco Central do Brasil ISPB sample
            creditor_agent: ParticipantId::Ispb("60746948".into()), // Bradesco ISPB sample
            debtor_account: "11122233".into(),
            creditor_account: "+5511999999999".into(),
            debtor_name: "João Silva".into(),
            creditor_name: "Maria Souza".into(),
            remittance: Some("Pix 001".into()),
            idempotency_key: "idem_pix_001".into(),
        }
    }

    fn make_client(server: &MockServer) -> PixClient {
        PixClient::new_unsecured(
            server.base_url(),
            Arc::new(StaticToken("test-token-123".into())),
            Arc::new(NoOpSigner::new("test-key")),
            "00038166",
        )
    }

    fn pacs002_acsc(uetr: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<Document><FIToFIPmtStsRpt><TxInfAndSts>
  <OrgnlUETR>{uetr}</OrgnlUETR>
  <OrgnlEndToEndId>E2E001</OrgnlEndToEndId>
  <TxSts>ACSC</TxSts>
</TxInfAndSts></FIToFIPmtStsRpt></Document>"#
        )
    }

    #[test]
    fn submit_acsc_returns_settled() {
        let server = MockServer::start();
        let req = sample_req();
        let mock = server.mock(|when, then| {
            when.method(POST)
                .path("/pix/v2/pacs008")
                .header("Authorization", "Bearer test-token-123")
                .header_exists("x-signature")
                .header_exists("x-idempotency-key");
            then.status(200)
                .header("content-type", "application/xml")
                .body(pacs002_acsc(&req.uetr));
        });
        let client = make_client(&server);
        let decision = client.submit_credit_transfer(&req).unwrap();
        mock.assert();
        assert_eq!(decision.status, A2aStatus::Settled);
        assert_eq!(decision.raw_status, "ACSC");
    }

    #[test]
    fn submit_rejects_non_brl() {
        let server = MockServer::start();
        let mut req = sample_req();
        req.amount = Money::from_minor(100, Currency::USD);
        let client = make_client(&server);
        let err = client.submit_credit_transfer(&req).unwrap_err();
        assert!(matches!(
            err,
            Error::CurrencyMismatch {
                rail: "pix",
                expected: "BRL",
                ..
            }
        ));
    }

    #[test]
    fn submit_rejects_non_ispb_agents() {
        let server = MockServer::start();
        let mut req = sample_req();
        req.debtor_agent = ParticipantId::Aba("021000021".into());
        let client = make_client(&server);
        let err = client.submit_credit_transfer(&req).unwrap_err();
        assert!(matches!(err, Error::UnsupportedA2aKey { rail: "pix" }));
    }

    #[test]
    fn submit_rejects_malformed_ispb() {
        let server = MockServer::start();
        let mut req = sample_req();
        // 7 digits instead of 8.
        req.debtor_agent = ParticipantId::Ispb("1234567".into());
        let client = make_client(&server);
        let err = client.submit_credit_transfer(&req).unwrap_err();
        assert!(matches!(err, Error::DriverValidation(_)));
    }

    #[test]
    fn submit_401_returns_rail_rejected() {
        let server = MockServer::start();
        server.mock(|when, then| {
            when.method(POST).path("/pix/v2/pacs008");
            then.status(401).body("{\"error\":\"invalid_token\"}");
        });
        let req = sample_req();
        let client = make_client(&server);
        let err = client.submit_credit_transfer(&req).unwrap_err();
        match err {
            Error::RailRejected { status, .. } => assert_eq!(status, 401),
            other => panic!("expected RailRejected 401, got {other:?}"),
        }
    }

    #[test]
    fn query_status_via_get() {
        let server = MockServer::start();
        let req = sample_req();
        let mock = server.mock(|when, then| {
            when.method(GET)
                .path(format!("/pix/v2/payments/{}/status", req.uetr));
            then.status(200).body(pacs002_acsc(&req.uetr));
        });
        let client = make_client(&server);
        let decision = client
            .query_status(&StatusQueryReq {
                uetr: req.uetr.clone(),
                end_to_end_id: req.end_to_end_id.clone(),
            })
            .unwrap();
        mock.assert();
        assert_eq!(decision.status, A2aStatus::Settled);
    }

    #[test]
    fn ispb_validation() {
        assert!(PixClient::validate_ispb("12345678").is_ok());
        assert!(PixClient::validate_ispb("1234567").is_err()); // too short
        assert!(PixClient::validate_ispb("123456789").is_err()); // too long
        assert!(PixClient::validate_ispb("1234567a").is_err()); // non-digit
        assert!(PixClient::validate_ispb("").is_err()); // empty
    }

    #[test]
    fn base64_encode_matches_rfc4648() {
        // RFC 4648 §10 test vector
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }
}
