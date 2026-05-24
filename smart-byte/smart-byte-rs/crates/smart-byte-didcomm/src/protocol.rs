//! Application-protocol abstraction.
//!
//! Aries-style application protocols (issue-credential, present-proof,
//! etc.) are conventionally described by a URI of the form
//! `https://didcomm.org/<name>/<major>.<minor>` and a small set of message
//! types under that URI. Each protocol is a Mealy machine over the
//! message types: state transitions on each received / emitted message.

use crate::message::DidcommMessage;

/// Identifies an Aries-style application protocol.
pub trait Protocol {
    /// The base protocol URI (e.g. `https://didcomm.org/issue-credential`).
    fn protocol_uri(&self) -> &str;
    /// The protocol version (e.g. `3.0`).
    fn version(&self) -> &str;
}

/// A typed message belonging to a protocol. Each protocol module defines
/// an enum implementing this trait whose variants map 1:1 to the message
/// types defined by the protocol RFC.
pub trait ProtocolMessage: Sized {
    /// Decode from a generic [`DidcommMessage`]. Returns `None` if the
    /// `type_` is not one this protocol recognises.
    fn from_message(msg: &DidcommMessage) -> Option<Self>;
    /// Encode back to a [`DidcommMessage`]. The caller is expected to set
    /// `from` / `to` / `thid` etc.
    fn to_message(&self) -> DidcommMessage;
}
