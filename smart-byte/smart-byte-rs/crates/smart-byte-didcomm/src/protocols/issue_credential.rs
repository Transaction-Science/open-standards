//! Aries RFC 0453: Issue Credential Protocol v3 (DIDComm v2).
//!
//! Protocol URI: `https://didcomm.org/issue-credential/3.0`.
//!
//! State machine (happy path, holder-initiated or issuer-initiated):
//!
//! ```text
//! propose-credential? -> offer-credential -> request-credential
//!                                        -> issue-credential -> ack
//! ```

use serde::{Deserialize, Serialize};

use crate::error::DidcommError;
use crate::message::{Attachment, AttachmentData, DidcommMessage};
use crate::protocol::{Protocol, ProtocolMessage};

/// Base URI for this protocol.
pub const PROTOCOL_URI: &str = "https://didcomm.org/issue-credential";
/// Protocol version.
pub const VERSION: &str = "3.0";

/// Singleton handle.
pub struct IssueCredential;
impl Protocol for IssueCredential {
    fn protocol_uri(&self) -> &str {
        PROTOCOL_URI
    }
    fn version(&self) -> &str {
        VERSION
    }
}

/// Body of `propose-credential`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProposeCredentialBody {
    /// Optional goal code.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub goal_code: Option<String>,
    /// Optional human comment.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub comment: Option<String>,
    /// Credential preview JSON.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub credential_preview: Option<serde_json::Value>,
}

/// Body of `offer-credential`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct OfferCredentialBody {
    /// Optional goal code.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub goal_code: Option<String>,
    /// Optional comment.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub comment: Option<String>,
    /// Whether to replace any existing credential.
    #[serde(default)]
    pub replacement_id: Option<String>,
    /// Credential preview.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub credential_preview: Option<serde_json::Value>,
    /// Formats array — declares the credential format for each attachment.
    #[serde(default)]
    pub formats: Vec<CredentialFormat>,
}

/// Body of `request-credential`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct RequestCredentialBody {
    /// Optional comment.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub comment: Option<String>,
    /// Formats array.
    #[serde(default)]
    pub formats: Vec<CredentialFormat>,
}

/// Body of `issue-credential`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct IssueCredentialBody {
    /// Optional comment.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub comment: Option<String>,
    /// Formats array.
    #[serde(default)]
    pub formats: Vec<CredentialFormat>,
}

/// Body of `ack`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AckBody {
    /// Status. RFC 0023 ack uses `OK`, `PENDING`, `FAIL`.
    pub status: String,
}

/// Maps an attachment id to a concrete credential format identifier (e.g.
/// `aries/ld-proof-vc@v1.0`, `hlindy/cred-filter@v2.0`, `jwt_vc`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CredentialFormat {
    /// Attachment id this format refers to.
    pub attach_id: String,
    /// Format identifier.
    pub format: String,
}

/// Typed enum of issue-credential v3 message types.
#[derive(Debug, Clone)]
pub enum IssueCredentialKind {
    /// `propose-credential`.
    Propose(ProposeCredentialBody),
    /// `offer-credential`.
    Offer(OfferCredentialBody),
    /// `request-credential`.
    Request(RequestCredentialBody),
    /// `issue-credential`.
    Issue(IssueCredentialBody),
    /// `ack`.
    Ack(AckBody),
}

impl ProtocolMessage for IssueCredentialKind {
    fn from_message(msg: &DidcommMessage) -> Option<Self> {
        let base = format!("{PROTOCOL_URI}/{VERSION}");
        let suffix = msg.type_.strip_prefix(&format!("{base}/"))?;
        match suffix {
            "propose-credential" => Some(Self::Propose(
                serde_json::from_value(msg.body.clone()).unwrap_or_default(),
            )),
            "offer-credential" => Some(Self::Offer(
                serde_json::from_value(msg.body.clone()).unwrap_or_default(),
            )),
            "request-credential" => Some(Self::Request(
                serde_json::from_value(msg.body.clone()).unwrap_or_default(),
            )),
            "issue-credential" => Some(Self::Issue(
                serde_json::from_value(msg.body.clone()).unwrap_or_default(),
            )),
            "ack" => Some(Self::Ack(
                serde_json::from_value(msg.body.clone()).unwrap_or_default(),
            )),
            _ => None,
        }
    }

    fn to_message(&self) -> DidcommMessage {
        let (suffix, body) = match self {
            Self::Propose(b) => (
                "propose-credential",
                serde_json::to_value(b).expect("body serialisable"),
            ),
            Self::Offer(b) => (
                "offer-credential",
                serde_json::to_value(b).expect("body serialisable"),
            ),
            Self::Request(b) => (
                "request-credential",
                serde_json::to_value(b).expect("body serialisable"),
            ),
            Self::Issue(b) => (
                "issue-credential",
                serde_json::to_value(b).expect("body serialisable"),
            ),
            Self::Ack(b) => (
                "ack",
                serde_json::to_value(b).expect("body serialisable"),
            ),
        };
        DidcommMessage::new(format!("{PROTOCOL_URI}/{VERSION}/{suffix}"))
            .body(body)
    }
}

/// State of an issue-credential v3 thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// No messages exchanged yet.
    Initial,
    /// `propose-credential` sent / received.
    Proposed,
    /// `offer-credential` sent / received.
    Offered,
    /// `request-credential` sent / received.
    Requested,
    /// `issue-credential` sent / received.
    Issued,
    /// `ack` sent / received.
    Done,
}

/// State machine over a single issue-credential thread.
#[derive(Debug, Clone)]
pub struct StateMachine {
    /// Current state.
    pub state: State,
    /// Thread id.
    pub thid: Option<String>,
}

impl Default for StateMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl StateMachine {
    /// Fresh state machine in [`State::Initial`].
    pub fn new() -> Self {
        Self {
            state: State::Initial,
            thid: None,
        }
    }

    /// Apply an inbound or outbound message, advancing state.
    pub fn step(&mut self, msg: &DidcommMessage) -> Result<(), DidcommError> {
        let kind = IssueCredentialKind::from_message(msg).ok_or_else(|| {
            DidcommError::Protocol(format!(
                "not an issue-credential v3 message: {}",
                msg.type_
            ))
        })?;
        let next = match (self.state, &kind) {
            (State::Initial, IssueCredentialKind::Propose(_)) => State::Proposed,
            (State::Initial, IssueCredentialKind::Offer(_)) => State::Offered,
            (State::Proposed, IssueCredentialKind::Offer(_)) => State::Offered,
            (State::Offered, IssueCredentialKind::Request(_)) => State::Requested,
            (State::Requested, IssueCredentialKind::Issue(_)) => State::Issued,
            (State::Issued, IssueCredentialKind::Ack(_)) => State::Done,
            (s, k) => {
                return Err(DidcommError::Protocol(format!(
                    "illegal transition: {:?} on {:?}",
                    s,
                    std::mem::discriminant(k)
                )));
            }
        };
        if self.thid.is_none() {
            self.thid = Some(msg.thid.clone().unwrap_or_else(|| msg.id.clone()));
        }
        self.state = next;
        Ok(())
    }
}

/// Build an `offer-credential` message carrying a VC-JWT attachment.
pub fn make_offer_with_vc_jwt(
    vc_jwt: &str,
    attach_id: &str,
) -> DidcommMessage {
    let body = OfferCredentialBody {
        formats: vec![CredentialFormat {
            attach_id: attach_id.into(),
            format: "jwt_vc".into(),
        }],
        ..Default::default()
    };
    let mut msg = IssueCredentialKind::Offer(body).to_message();
    msg.attachments = vec![Attachment {
        id: Some(attach_id.into()),
        media_type: Some("application/jwt".into()),
        format: Some("jwt_vc".into()),
        data: AttachmentData::from_base64(
            base64::Engine::encode(
                &base64::engine::general_purpose::URL_SAFE_NO_PAD,
                vc_jwt.as_bytes(),
            ),
        ),
        description: None,
        filename: None,
        lastmod_time: None,
        byte_count: None,
    }];
    msg
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn happy_path_state_machine() {
        let mut sm = StateMachine::new();
        let offer = IssueCredentialKind::Offer(OfferCredentialBody::default())
            .to_message();
        sm.step(&offer).unwrap();
        assert_eq!(sm.state, State::Offered);

        let mut request = IssueCredentialKind::Request(
            RequestCredentialBody::default(),
        )
        .to_message();
        request.thid = sm.thid.clone();
        sm.step(&request).unwrap();
        assert_eq!(sm.state, State::Requested);

        let mut issue =
            IssueCredentialKind::Issue(IssueCredentialBody::default())
                .to_message();
        issue.thid = sm.thid.clone();
        sm.step(&issue).unwrap();
        assert_eq!(sm.state, State::Issued);

        let mut ack = IssueCredentialKind::Ack(AckBody {
            status: "OK".into(),
        })
        .to_message();
        ack.thid = sm.thid.clone();
        sm.step(&ack).unwrap();
        assert_eq!(sm.state, State::Done);
    }

    #[test]
    fn illegal_transition_rejected() {
        let mut sm = StateMachine::new();
        let ack = IssueCredentialKind::Ack(AckBody {
            status: "OK".into(),
        })
        .to_message();
        assert!(sm.step(&ack).is_err());
    }
}
