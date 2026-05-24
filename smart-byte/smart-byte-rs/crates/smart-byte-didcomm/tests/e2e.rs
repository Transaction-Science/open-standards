//! End-to-end tests for DIDComm v2 application protocols, using the
//! in-memory connection from `state.rs` to wire two synthetic agents
//! together.

use smart_byte_didcomm::message::DidcommMessage;
use smart_byte_didcomm::protocol::ProtocolMessage;
use smart_byte_didcomm::protocols::issue_credential::{
    AckBody as IssueAck, IssueCredentialBody, IssueCredentialKind,
    OfferCredentialBody, RequestCredentialBody, State as IssueState,
    StateMachine as IssueStateMachine,
};
use smart_byte_didcomm::protocols::present_proof::{
    AckBody as ProofAck, PresentProofKind, PresentationFormat,
    RequestPresentationBody, State as ProofState,
    StateMachine as ProofStateMachine, make_presentation_with_sd_jwt,
};
use smart_byte_didcomm::protocols::trust_ping::{
    PingBody, TrustPingKind, respond_to,
};
use smart_byte_didcomm::state::{Connection, InMemoryConnection};
use smart_byte_did::Did;

fn alice() -> Did {
    "did:example:alice".parse().unwrap()
}
fn bob() -> Did {
    "did:example:bob".parse().unwrap()
}

#[tokio::test]
async fn e2e_issue_credential_v3_happy_path() {
    let (issuer_side, holder_side) =
        InMemoryConnection::pair(alice(), bob());

    let mut issuer_sm = IssueStateMachine::new();
    let mut holder_sm = IssueStateMachine::new();

    // 1) Issuer sends offer.
    let mut offer =
        IssueCredentialKind::Offer(OfferCredentialBody::default()).to_message();
    offer.from = Some(alice());
    offer.to = vec![bob()];
    issuer_sm.step(&offer).unwrap();
    issuer_side.send(offer.clone()).await.unwrap();

    let recv_offer = holder_side.receive().await.unwrap();
    holder_sm.step(&recv_offer).unwrap();
    assert_eq!(holder_sm.state, IssueState::Offered);

    // 2) Holder sends request.
    let mut req = IssueCredentialKind::Request(RequestCredentialBody::default())
        .to_message();
    req.from = Some(bob());
    req.to = vec![alice()];
    req.thid = holder_sm.thid.clone();
    holder_sm.step(&req).unwrap();
    holder_side.send(req).await.unwrap();

    let recv_req = issuer_side.receive().await.unwrap();
    issuer_sm.step(&recv_req).unwrap();
    assert_eq!(issuer_sm.state, IssueState::Requested);

    // 3) Issuer sends issue-credential.
    let mut issue = IssueCredentialKind::Issue(IssueCredentialBody::default())
        .to_message();
    issue.from = Some(alice());
    issue.to = vec![bob()];
    issue.thid = issuer_sm.thid.clone();
    issuer_sm.step(&issue).unwrap();
    issuer_side.send(issue).await.unwrap();

    let recv_issue = holder_side.receive().await.unwrap();
    holder_sm.step(&recv_issue).unwrap();
    assert_eq!(holder_sm.state, IssueState::Issued);

    // 4) Holder acks.
    let mut ack = IssueCredentialKind::Ack(IssueAck { status: "OK".into() })
        .to_message();
    ack.from = Some(bob());
    ack.to = vec![alice()];
    ack.thid = holder_sm.thid.clone();
    holder_sm.step(&ack).unwrap();
    holder_side.send(ack).await.unwrap();

    let recv_ack = issuer_side.receive().await.unwrap();
    issuer_sm.step(&recv_ack).unwrap();
    assert_eq!(issuer_sm.state, IssueState::Done);
    assert_eq!(holder_sm.state, IssueState::Done);
}

#[tokio::test]
async fn e2e_present_proof_v3_selective_disclosure() {
    let (verifier_side, holder_side) =
        InMemoryConnection::pair(alice(), bob());

    let mut verifier_sm = ProofStateMachine::new();
    let mut holder_sm = ProofStateMachine::new();

    // 1) Verifier sends request-presentation.
    let mut req = PresentProofKind::Request(RequestPresentationBody {
        will_confirm: true,
        formats: vec![PresentationFormat {
            attach_id: "req-1".into(),
            format: "sd-jwt".into(),
        }],
        ..Default::default()
    })
    .to_message();
    req.from = Some(alice());
    req.to = vec![bob()];
    verifier_sm.step(&req).unwrap();
    verifier_side.send(req).await.unwrap();

    let recv_req = holder_side.receive().await.unwrap();
    holder_sm.step(&recv_req).unwrap();
    assert_eq!(holder_sm.state, ProofState::Requested);

    // 2) Holder builds an SD-JWT presentation (mocked here as a string).
    // The bridge to smart-byte-vc::sd_jwt would assemble disclosures
    // selecting only the claims requested by the verifier. We use a
    // canned token to verify the message envelope round-trips.
    let mock_sd_jwt_presentation =
        "eyJhbGciOiJFZERTQSJ9.eyJ2YyI6e319.SIG~disclosure-1~disclosure-2~";
    let mut pres =
        make_presentation_with_sd_jwt(mock_sd_jwt_presentation, "pres-1");
    pres.from = Some(bob());
    pres.to = vec![alice()];
    pres.thid = holder_sm.thid.clone();
    holder_sm.step(&pres).unwrap();
    holder_side.send(pres).await.unwrap();

    let recv_pres = verifier_side.receive().await.unwrap();
    let kind = PresentProofKind::from_message(&recv_pres).unwrap();
    match kind {
        PresentProofKind::Presentation(b) => {
            assert_eq!(b.formats.len(), 1);
            assert_eq!(b.formats[0].format, "sd-jwt");
        }
        _ => panic!("expected presentation"),
    }
    assert!(!recv_pres.attachments.is_empty());
    verifier_sm.step(&recv_pres).unwrap();
    assert_eq!(verifier_sm.state, ProofState::Presented);

    // 3) Verifier acks.
    let mut ack = PresentProofKind::Ack(ProofAck { status: "OK".into() })
        .to_message();
    ack.from = Some(alice());
    ack.to = vec![bob()];
    ack.thid = verifier_sm.thid.clone();
    verifier_sm.step(&ack).unwrap();
    verifier_side.send(ack).await.unwrap();

    let recv_ack = holder_side.receive().await.unwrap();
    holder_sm.step(&recv_ack).unwrap();
    assert_eq!(verifier_sm.state, ProofState::Done);
    assert_eq!(holder_sm.state, ProofState::Done);
}

#[tokio::test]
async fn e2e_trust_ping_response() {
    let (a, b) = InMemoryConnection::pair(alice(), bob());
    let ping = TrustPingKind::Ping(PingBody {
        response_requested: true,
        comment: None,
    })
    .to_message()
    .from_did(alice())
    .to_dids(vec![bob()]);
    let ping_id = ping.id.clone();
    a.send(ping).await.unwrap();
    let recv = b.receive().await.unwrap();
    let resp = respond_to(&recv);
    b.send(resp).await.unwrap();
    let pong = a.receive().await.unwrap();
    match TrustPingKind::from_message(&pong).unwrap() {
        TrustPingKind::PingResponse(_) => {}
        _ => panic!("expected ping-response"),
    }
    assert_eq!(pong.thid.as_deref(), Some(ping_id.as_str()));
}

#[tokio::test]
async fn e2e_basic_message() {
    use smart_byte_didcomm::protocols::basic_message::{
        BasicMessageBody, BasicMessageKind,
    };
    let (a, b) = InMemoryConnection::pair(alice(), bob());
    let m = BasicMessageKind::Message(BasicMessageBody {
        content: "hello bob".into(),
        sent_time: None,
    })
    .to_message()
    .from_did(alice())
    .to_dids(vec![bob()]);
    a.send(m).await.unwrap();
    let recv = b.receive().await.unwrap();
    match BasicMessageKind::from_message(&recv).unwrap() {
        BasicMessageKind::Message(body) => {
            assert_eq!(body.content, "hello bob");
        }
    }
}

#[test]
fn message_serialises_with_required_envelope_fields() {
    let m = DidcommMessage::new("https://didcomm.org/x/1.0/y")
        .from_did(alice())
        .to_dids(vec![bob()]);
    let j: serde_json::Value = serde_json::to_value(&m).unwrap();
    assert!(j.get("id").is_some());
    assert!(j.get("type").is_some());
    assert!(j.get("from").is_some());
    assert!(j.get("to").is_some());
}
