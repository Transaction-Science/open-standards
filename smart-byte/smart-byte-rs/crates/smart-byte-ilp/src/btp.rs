//! BTP — Bilateral Transfer Protocol — message types.
//!
//! BTP is the framing two ILP connectors speak to each other over a
//! single persistent WebSocket (or other reliable bidirectional stream).
//! Each BTP message is a `(type, request_id, sub_protocols[])` triple
//! plus a type-specific body.
//!
//! See IL-RFC-23.

use crate::error::{IlpError, Result};
use crate::oer::{
    decode_length, decode_u32, decode_var_octet_string, encode_length, encode_u32,
    encode_var_octet_string,
};

/// BTP packet-type discriminators.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BtpPacketType {
    /// `1` — Response (success).
    Response = 1,
    /// `2` — Error response.
    ErrorResponse = 2,
    /// `6` — Message (request).
    Message = 6,
    /// `7` — Transfer (request carrying an ILP `Prepare`).
    Transfer = 7,
}

impl BtpPacketType {
    /// Map a wire byte to a BTP packet type.
    pub fn from_u8(b: u8) -> Result<Self> {
        Ok(match b {
            1 => BtpPacketType::Response,
            2 => BtpPacketType::ErrorResponse,
            6 => BtpPacketType::Message,
            7 => BtpPacketType::Transfer,
            other => return Err(IlpError::UnknownBtpPacketType(other)),
        })
    }
}

/// Content-type marker for a BTP sub-protocol payload.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ContentType {
    /// `0` — application/octet-stream.
    OctetStream = 0,
    /// `1` — text/plain-utf8.
    TextPlain = 1,
    /// `2` — application/json.
    Json = 2,
}

impl ContentType {
    /// Lift a byte into a `ContentType`. Unknown values fall back to
    /// `OctetStream`.
    pub fn from_u8(b: u8) -> Self {
        match b {
            1 => ContentType::TextPlain,
            2 => ContentType::Json,
            _ => ContentType::OctetStream,
        }
    }
}

/// A single BTP sub-protocol entry. The reference sub-protocol is
/// `"ilp"`, which carries an ILP packet in `data`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BtpSubProtocol {
    /// Protocol name (e.g. `"ilp"`).
    pub name: String,
    /// Content-type of the payload.
    pub content_type: ContentType,
    /// Payload bytes.
    pub data: Vec<u8>,
}

/// A complete BTP message envelope.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BtpMessage {
    /// Packet type discriminator.
    pub packet_type: BtpPacketType,
    /// Request id matching responses to requests.
    pub request_id: u32,
    /// Ordered list of sub-protocol entries.
    pub sub_protocols: Vec<BtpSubProtocol>,
}

impl BtpMessage {
    /// Encode the message to its on-wire byte representation.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::new();
        out.push(self.packet_type as u8);
        encode_u32(&mut out, self.request_id);
        let mut body = Vec::new();
        encode_length(&mut body, self.sub_protocols.len());
        for sp in &self.sub_protocols {
            encode_var_octet_string(&mut body, sp.name.as_bytes());
            body.push(sp.content_type as u8);
            encode_var_octet_string(&mut body, &sp.data);
        }
        // BTP wraps the sub-protocol list in a length-prefixed body.
        crate::oer::encode_length(&mut out, body.len());
        out.extend_from_slice(&body);
        out
    }

    /// Decode the message from its on-wire bytes.
    pub fn decode(input: &[u8]) -> Result<Self> {
        if input.is_empty() {
            return Err(IlpError::InvalidPacket("btp empty".into()));
        }
        let packet_type = BtpPacketType::from_u8(input[0])?;
        let (request_id, used) = decode_u32(&input[1..])?;
        let mut cursor = 1 + used;
        let (body_len, used) = decode_length(&input[cursor..])?;
        cursor += used;
        let body_end = cursor
            .checked_add(body_len)
            .ok_or_else(|| IlpError::InvalidPacket("btp body overflow".into()))?;
        if input.len() < body_end {
            return Err(IlpError::InvalidPacket("btp body truncated".into()));
        }
        let body = &input[cursor..body_end];
        let (n_sp, used) = decode_length(body)?;
        let mut bcur = used;
        let mut sub_protocols = Vec::with_capacity(n_sp);
        for _ in 0..n_sp {
            let (name_bytes, used) = decode_var_octet_string(&body[bcur..])?;
            bcur += used;
            let name = core::str::from_utf8(name_bytes)
                .map_err(|e| IlpError::InvalidPacket(format!("btp sp name utf-8: {e}")))?
                .to_string();
            if bcur >= body.len() {
                return Err(IlpError::InvalidPacket("btp sp content_type missing".into()));
            }
            let content_type = ContentType::from_u8(body[bcur]);
            bcur += 1;
            let (data_bytes, used) = decode_var_octet_string(&body[bcur..])?;
            bcur += used;
            sub_protocols.push(BtpSubProtocol {
                name,
                content_type,
                data: data_bytes.to_vec(),
            });
        }
        Ok(BtpMessage {
            packet_type,
            request_id,
            sub_protocols,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn btp_roundtrip() {
        let msg = BtpMessage {
            packet_type: BtpPacketType::Message,
            request_id: 42,
            sub_protocols: vec![
                BtpSubProtocol {
                    name: "ilp".into(),
                    content_type: ContentType::OctetStream,
                    data: vec![1, 2, 3, 4],
                },
                BtpSubProtocol {
                    name: "auth".into(),
                    content_type: ContentType::TextPlain,
                    data: b"token".to_vec(),
                },
            ],
        };
        let wire = msg.encode();
        let back = BtpMessage::decode(&wire).unwrap();
        assert_eq!(msg, back);
    }
}
