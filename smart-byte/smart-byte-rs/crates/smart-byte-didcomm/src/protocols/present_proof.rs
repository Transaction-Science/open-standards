//! Aries RFC 0454: Present Proof Protocol v3 (DIDComm v2).
//!
//! Protocol URI: `https://didcomm.org/present-proof/3.0`.
//!
//! State machine:
//!
//! ```text
//! propose-presentation? -> request-presentation -> presentation -> ack
//! ```

use serde::{Deserialize, Serialize};

use crate::error::DidcommError;
use crate::message::{Attachment, AttachmentData, DidcommMessage};
use crate::protocol::{Protocol, ProtocolMessage};

/// Base URI for this protocol.
pub const PROTOCOL_URI: &str = "https://didcomm.org/present-proof";
/// Protocol version.
pub const VERSION: &str = "3.0";

/// Singleton handle.
pub struct PresentProof;
impl Protocol for PresentProof {
    fn protocol_uri(&self) -> &str {
        PROTOCOL_URI
    }
    fn version(&self) -> &str {
        VERSION
    }
}

/// Body of `propose-presentation`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProposePresentationBody {
    /// Optional comment.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub comment: Option<String>,
    /// Formats array.
    #[serde(default)]
    pub formats: Vec<PresentationFormat>,
}

/// Body of `request-presentation`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct RequestPresentationBody {
    /// Optional comment.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub comment: Option<String>,
    /// Whether the recipient must respond.
    #[serde(default)]
    pub will_confirm: bool,
    /// Formats array.
    #[serde(default)]
    pub formats: Vec<PresentationFormat>,
}

/// Body of `presentation`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct PresentationBody {
    /// Optional comment.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub comment: Option<String>,
    /// Formats array.
    #[serde(default)]
    pub formats: Vec<PresentationFormat>,
}

/// Body of `ack`.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AckBody {
    /// `OK` / `PENDING` / `FAIL`.
    pub status: String,
}

/// Maps an attachment id to a concrete presentation format identifier
/// (e.g. `dif/presentation-exchange/definitions@v2.0`, `sd-jwt`, `hlindy/proof@v2.0`).
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PresentationFormat {
    /// Attachment id.
    pub attach_id: String,
    /// Format identifier.
    pub format: String,
}

/// Typed enum of present-proof v3 message types.
#[derive(Debug, Clone)]
pub enum PresentProofKind {
    /// `propose-presentation`.
    Propose(ProposePresentationBody),
    /// `request-presentation`.
    Request(RequestPresentationBody),
    /// `presentation`.
    Presentation(PresentationBody),
    /// `ack`.
    Ack(AckBody),
}

impl ProtocolMessage for PresentProofKind {
    fn from_message(msg: &DidcommMessage) -> Option<Self> {
        let base = format!("{PROTOCOL_URI}/{VERSION}");
        let suffix = msg.type_.strip_prefix(&format!("{base}/"))?;
        match suffix {
            "propose-presentation" => Some(Self::Propose(
                serde_json::from_value(msg.body.clone()).unwrap_or_default(),
            )),
            "request-presentation" => Some(Self::Request(
                serde_json::from_value(msg.body.clone()).unwrap_or_default(),
            )),
            "presentation" => Some(Self::Presentation(
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
                "propose-presentation",
                serde_json::to_value(b).expect("body serialisable"),
            ),
            Self::Request(b) => (
                "request-presentation",
                serde_json::to_value(b).expect("body serialisable"),
            ),
            Self::Presentation(b) => (
                "presentation",
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

/// State of a present-proof v3 thread.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    /// No messages exchanged yet.
    Initial,
    /// `propose-presentation` sent / received.
    Proposed,
    /// `request-presentation` sent / received.
    Requested,
    /// `presentation` sent / received.
    Presented,
    /// `ack` sent / received.
    Done,
}

/// State machine over a single present-proof thread.
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

    /// Apply a message, advancing state.
    pub fn step(&mut self, msg: &DidcommMessage) -> Result<(), DidcommError> {
        let kind = PresentProofKind::from_message(msg).ok_or_else(|| {
            DidcommError::Protocol(format!(
                "not a present-proof v3 message: {}",
                msg.type_
            ))
        })?;
        let next = match (self.state, &kind) {
            (State::Initial, PresentProofKind::Propose(_)) => State::Proposed,
            (State::Initial, PresentProofKind::Request(_)) => State::Requested,
            (State::Proposed, PresentProofKind::Request(_)) => State::Requested,
            (State::Requested, PresentProofKind::Presentation(_)) => State::Presented,
            (State::Presented, PresentProofKind::Ack(_)) => State::Done,
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

/// Build a `presentation` message carrying an SD-JWT presentation
/// attachment. The presentation envelope bridges to
/// [`smart_byte_vc::sd_jwt`] for selective disclosure.
pub fn make_presentation_with_sd_jwt(
    sd_jwt_presentation: &str,
    attach_id: &str,
) -> DidcommMessage {
    let body = PresentationBody {
        formats: vec![PresentationFormat {
            attach_id: attach_id.into(),
            format: "sd-jwt".into(),
        }],
        ..Default::default()
    };
    let mut msg = PresentProofKind::Presentation(body).to_message();
    msg.attachments = vec![Attachment {
        id: Some(attach_id.into()),
        media_type: Some("application/json".into()),
        format: Some("sd-jwt".into()),
        data: AttachmentData::from_base64(base64::Engine::encode(
            &base64::engine::general_purpose::URL_SAFE_NO_PAD,
            sd_jwt_presentation.as_bytes(),
        )),
        description: None,
        filename: None,
        lastmod_time: None,
        byte_count: None,
    }];
    msg
}
