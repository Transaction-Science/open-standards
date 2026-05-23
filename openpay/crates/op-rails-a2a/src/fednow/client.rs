//! `FedNow` MQ client and REST client.
//!
//! Builds pacs.008.001.08 via `op-iso20022::CreditTransferBuilder<FedNow>`,
//! serializes to the `FedNow` XML profile, submits via the operator's
//! MQ channel, and parses the pacs.002 response.

use std::sync::Arc;

use op_core::{A2aKey, PaymentMethod};
use op_iso20022::bah::PartyIdentification;
use op_iso20022::profile::FedNow;
use op_iso20022::{BuiltCreditTransfer, CreditTransferBuilder};

use crate::acquirer::{
    A2aAcquirer, A2aDecision, A2aStatus, CreditTransferReq, ParticipantId, StatusQueryReq,
};
use crate::error::{Error, Result};

use super::mq::{MqChannel, MqMessage};
use super::status_map;

/// `FedNow` client that submits via an operator-supplied MQ channel.
#[derive(Clone)]
pub struct FedNowMqClient {
    channel: Arc<dyn MqChannel>,
    sender_rtn: String,
    pacs008_queue: String,
}

impl FedNowMqClient {
    /// `FedNow`'s standard inbound credit-transfer queue name.
    pub const DEFAULT_PACS008_QUEUE: &'static str = "FRB.FEDNOW.PACS008.IN";

    /// Construct with an operator-supplied MQ channel and the
    /// operator's own RTN (ABA routing number).
    #[must_use]
    pub fn new(channel: Arc<dyn MqChannel>, sender_rtn: impl Into<String>) -> Self {
        Self {
            channel,
            sender_rtn: sender_rtn.into(),
            pacs008_queue: Self::DEFAULT_PACS008_QUEUE.to_owned(),
        }
    }

    /// Override the default queue (rarely needed in production).
    #[must_use]
    pub fn with_queue(mut self, queue: impl Into<String>) -> Self {
        self.pacs008_queue = queue.into();
        self
    }

    /// Validate rail-specific constraints before building anything.
    fn validate(req: &CreditTransferReq) -> Result<()> {
        if req.amount.currency != op_core::Currency::USD {
            return Err(Error::CurrencyMismatch {
                rail: "fednow",
                expected: "USD",
                got: req.amount.currency.code().to_owned(),
            });
        }
        match (&req.debtor_agent, &req.creditor_agent) {
            (ParticipantId::Aba(_), ParticipantId::Aba(_)) => Ok(()),
            _ => Err(Error::UnsupportedA2aKey { rail: "fednow" }),
        }
    }

    /// Translate a rail-neutral request into a typed builder result.
    fn build(req: &CreditTransferReq) -> Result<BuiltCreditTransfer<FedNow>> {
        let debtor_party =
            PartyIdentification::AbaRoutingNumber(req.debtor_agent.as_str().to_owned());
        let creditor_party =
            PartyIdentification::AbaRoutingNumber(req.creditor_agent.as_str().to_owned());

        let debtor_method = PaymentMethod::A2a(A2aKey::UsAch {
            routing: req.debtor_agent.as_str().to_owned(),
            account: req.debtor_account.clone(),
        });
        let creditor_method = PaymentMethod::A2a(A2aKey::UsAch {
            routing: req.creditor_agent.as_str().to_owned(),
            account: req.creditor_account.clone(),
        });

        let mut b = CreditTransferBuilder::<FedNow>::new()
            .amount(req.amount)
            .debtor(debtor_method)
            .creditor(creditor_method)
            .debtor_agent(debtor_party)
            .creditor_agent(creditor_party)
            .end_to_end_id(&req.end_to_end_id)
            .uetr(&req.uetr);
        if let Some(r) = &req.remittance {
            b = b.remittance(r);
        }
        Ok(b.build()?)
    }
}

impl A2aAcquirer for FedNowMqClient {
    fn name(&self) -> &'static str {
        "fednow"
    }

    fn submit_credit_transfer(&self, req: &CreditTransferReq) -> Result<A2aDecision> {
        Self::validate(req)?;
        let built = Self::build(req)?;
        let xml = super::xml::emit_pacs008(
            &built,
            &req.debtor_name,
            &req.creditor_name,
            &req.debtor_account,
            &req.creditor_account,
        );

        let message = MqMessage {
            queue: self.pacs008_queue.clone(),
            payload: xml.into_bytes(),
            correlation_id: req.uetr.clone(),
            properties: vec![
                ("BAH.From".into(), self.sender_rtn.clone()),
                ("BAH.To".into(), req.creditor_agent.as_str().to_owned()),
                ("MsgDefIdr".into(), "pacs.008.001.08".into()),
                ("BizSvc".into(), "frb.fednow.01".into()),
            ],
        };

        let response = self.channel.request(&message)?;
        let xml = String::from_utf8(response.payload)
            .map_err(|e| Error::Transport(format!("response not UTF-8: {e}")))?;
        let parsed = super::xml::parse_pacs002(&xml)?;
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

    fn query_status(&self, _req: &StatusQueryReq) -> Result<A2aDecision> {
        Err(Error::DriverValidation(
            "FedNow MQ status query (pacs.028) deferred — use FedNowApiClient".into(),
        ))
    }
}

/// FedLine-VPN REST client. Used for status lookups.
///
/// Authentication is TLS client certificate. Operator provides cert/key.
#[derive(Clone)]
pub struct FedNowApiClient {
    base_url: String,
    agent: ureq::Agent,
}

impl FedNowApiClient {
    /// Construct with the operator's `FedLine` VPN endpoint.
    #[must_use]
    pub fn new(base_url: impl Into<String>) -> Self {
        Self {
            base_url: base_url.into(),
            agent: ureq::AgentBuilder::new()
                .timeout(std::time::Duration::from_secs(20))
                .build(),
        }
    }

    fn url(&self, path: &str) -> String {
        format!("{}{}", self.base_url.trim_end_matches('/'), path)
    }

    /// GET the status of a transfer by UETR.
    pub fn get_status(&self, uetr: &str) -> Result<A2aDecision> {
        let url = self.url(&format!("/v1/payments/{uetr}/status"));
        let resp = self.agent.get(&url).call().map_err(|e| match e {
            ureq::Error::Status(s, r) => Error::RailRejected {
                status: s,
                code: "http".into(),
                message: r.into_string().unwrap_or_default(),
            },
            ureq::Error::Transport(t) => Error::Transport(t.to_string()),
        })?;

        let body: serde_json::Value = resp
            .into_json()
            .map_err(|e| Error::Transport(e.to_string()))?;
        let status_code = body["transactionStatus"]
            .as_str()
            .ok_or_else(|| Error::Transport("missing transactionStatus".into()))?;
        let status = status_map::map_transaction_status(status_code)?;
        Ok(A2aDecision {
            status,
            raw_status: status_code.to_owned(),
            reason_code: body["reasonCode"].as_str().map(String::from),
            reason_text: body["reasonText"].as_str().map(String::from),
            uetr: Some(uetr.to_owned()),
            rail_txn_id: body["endToEndId"].as_str().map(String::from),
            settled_amount: None,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fednow::mq::MqResponse;
    use op_core::{Currency, Money};
    use std::sync::Mutex;

    fn sample_req() -> CreditTransferReq {
        CreditTransferReq {
            uetr: "12a345b6-7c89-4d01-23e4-567890abcdef".into(),
            end_to_end_id: "e2e_001".into(),
            amount: Money::from_minor(12345, Currency::USD),
            debtor_agent: ParticipantId::Aba("021000021".into()),
            creditor_agent: ParticipantId::Aba("026009593".into()),
            debtor_account: "1234567890".into(),
            creditor_account: "9876543210".into(),
            debtor_name: "Alice Sender".into(),
            creditor_name: "Bob Recipient".into(),
            remittance: Some("Invoice 4242".into()),
            idempotency_key: "idem_001".into(),
        }
    }

    struct CannedChannel {
        captured: Mutex<Option<MqMessage>>,
        response: MqResponse,
    }
    impl MqChannel for CannedChannel {
        fn request(&self, m: &MqMessage) -> Result<MqResponse> {
            *self.captured.lock().unwrap() = Some(m.clone());
            Ok(self.response.clone())
        }
    }

    fn pacs002_acsc(uetr: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pacs.002.001.10">
  <FIToFIPmtStsRpt>
    <GrpHdr><MsgId>m1</MsgId><CreDtTm>2026-05-17T10:00:00Z</CreDtTm></GrpHdr>
    <TxInfAndSts>
      <OrgnlUETR>{uetr}</OrgnlUETR>
      <OrgnlEndToEndId>e2e_001</OrgnlEndToEndId>
      <TxSts>ACSC</TxSts>
    </TxInfAndSts>
  </FIToFIPmtStsRpt>
</Document>"#
        )
    }

    fn pacs002_rjct(uetr: &str, reason: &str) -> String {
        format!(
            r#"<?xml version="1.0" encoding="UTF-8"?>
<Document xmlns="urn:iso:std:iso:20022:tech:xsd:pacs.002.001.10">
  <FIToFIPmtStsRpt>
    <GrpHdr><MsgId>m1</MsgId><CreDtTm>2026-05-17T10:00:00Z</CreDtTm></GrpHdr>
    <TxInfAndSts>
      <OrgnlUETR>{uetr}</OrgnlUETR>
      <OrgnlEndToEndId>e2e_001</OrgnlEndToEndId>
      <TxSts>RJCT</TxSts>
      <StsRsnInf><Rsn><Cd>{reason}</Cd></Rsn></StsRsnInf>
    </TxInfAndSts>
  </FIToFIPmtStsRpt>
</Document>"#
        )
    }

    #[test]
    fn submit_acsc_returns_settled() {
        let req = sample_req();
        let channel = Arc::new(CannedChannel {
            captured: Mutex::new(None),
            response: MqResponse {
                payload: pacs002_acsc(&req.uetr).into_bytes(),
                correlation_id: req.uetr.clone(),
                mq_ack_code: None,
            },
        });
        let client = FedNowMqClient::new(channel, "021000021");
        let decision = client.submit_credit_transfer(&req).unwrap();
        assert_eq!(decision.status, A2aStatus::Settled);
        assert_eq!(decision.raw_status, "ACSC");
        assert!(decision.settled_amount.is_some());
    }

    #[test]
    fn submit_rjct_returns_rejected_with_reason() {
        let req = sample_req();
        let channel = Arc::new(CannedChannel {
            captured: Mutex::new(None),
            response: MqResponse {
                payload: pacs002_rjct(&req.uetr, "AC03").into_bytes(),
                correlation_id: req.uetr.clone(),
                mq_ack_code: None,
            },
        });
        let client = FedNowMqClient::new(channel, "021000021");
        let decision = client.submit_credit_transfer(&req).unwrap();
        assert_eq!(decision.status, A2aStatus::Rejected);
        assert_eq!(decision.reason_code.as_deref(), Some("AC03"));
        assert!(decision.settled_amount.is_none());
    }

    #[test]
    fn submit_rejects_non_usd() {
        let mut req = sample_req();
        req.amount = Money::from_minor(1000, Currency::EUR);
        let channel = Arc::new(CannedChannel {
            captured: Mutex::new(None),
            response: MqResponse {
                payload: vec![],
                correlation_id: String::new(),
                mq_ack_code: None,
            },
        });
        let client = FedNowMqClient::new(channel, "021000021");
        let err = client.submit_credit_transfer(&req).unwrap_err();
        assert!(matches!(
            err,
            Error::CurrencyMismatch { rail: "fednow", .. }
        ));
    }

    #[test]
    fn submit_rejects_non_aba_agents() {
        let mut req = sample_req();
        req.debtor_agent = ParticipantId::Bic("DEUTDEFFXXX".into());
        let channel = Arc::new(CannedChannel {
            captured: Mutex::new(None),
            response: MqResponse {
                payload: vec![],
                correlation_id: String::new(),
                mq_ack_code: None,
            },
        });
        let client = FedNowMqClient::new(channel, "021000021");
        let err = client.submit_credit_transfer(&req).unwrap_err();
        assert!(matches!(err, Error::UnsupportedA2aKey { rail: "fednow" }));
    }

    #[test]
    fn mq_envelope_carries_correct_metadata() {
        let req = sample_req();
        let channel = Arc::new(CannedChannel {
            captured: Mutex::new(None),
            response: MqResponse {
                payload: pacs002_acsc(&req.uetr).into_bytes(),
                correlation_id: req.uetr.clone(),
                mq_ack_code: None,
            },
        });
        let client = FedNowMqClient::new(channel.clone(), "021000021");
        client.submit_credit_transfer(&req).unwrap();

        let captured = channel.captured.lock().unwrap();
        let m = captured.as_ref().unwrap();
        assert_eq!(m.queue, "FRB.FEDNOW.PACS008.IN");
        assert_eq!(m.correlation_id, req.uetr);
        let from = m
            .properties
            .iter()
            .find(|(k, _)| k == "BAH.From")
            .map(|(_, v)| v.as_str());
        assert_eq!(from, Some("021000021"));
        let msg_def = m
            .properties
            .iter()
            .find(|(k, _)| k == "MsgDefIdr")
            .map(|(_, v)| v.as_str());
        assert_eq!(msg_def, Some("pacs.008.001.08"));
    }

    #[test]
    fn payload_contains_uetr_amount_names_remittance() {
        let req = sample_req();
        let channel = Arc::new(CannedChannel {
            captured: Mutex::new(None),
            response: MqResponse {
                payload: pacs002_acsc(&req.uetr).into_bytes(),
                correlation_id: req.uetr.clone(),
                mq_ack_code: None,
            },
        });
        let client = FedNowMqClient::new(channel.clone(), "021000021");
        client.submit_credit_transfer(&req).unwrap();

        let captured = channel.captured.lock().unwrap();
        let m = captured.as_ref().unwrap();
        let xml = std::str::from_utf8(&m.payload).unwrap();
        assert!(xml.contains(&req.uetr), "carries UETR");
        assert!(xml.contains("123.45"), "carries amount");
        assert!(xml.contains("Alice Sender"), "carries debtor name");
        assert!(xml.contains("Bob Recipient"), "carries creditor name");
        assert!(xml.contains("Invoice 4242"), "carries remittance");
    }

    #[test]
    fn query_status_via_mq_is_deferred() {
        let channel = Arc::new(CannedChannel {
            captured: Mutex::new(None),
            response: MqResponse {
                payload: vec![],
                correlation_id: String::new(),
                mq_ack_code: None,
            },
        });
        let client = FedNowMqClient::new(channel, "021000021");
        let result = client.query_status(&StatusQueryReq {
            uetr: "u".into(),
            end_to_end_id: "e".into(),
        });
        assert!(matches!(result, Err(Error::DriverValidation(_))));
    }

    #[test]
    fn rest_client_constructs_with_endpoint() {
        let c = FedNowApiClient::new("https://fedline-vpn.internal/");
        assert_eq!(
            c.url("/v1/payments/abc/status"),
            "https://fedline-vpn.internal/v1/payments/abc/status"
        );
    }
}
