//! ILPv4 wire packets — `Prepare`, `Fulfill`, `Reject`.
//!
//! Every ILP packet is a single discriminator byte followed by an OER
//! length-prefixed payload:
//!
//! ```text
//!   +---------+----------------+----------------------+
//!   | type    | length-deter.  | payload              |
//!   +---------+----------------+----------------------+
//! ```
//!
//! Type bytes:
//!
//! * `12` — `Prepare`
//! * `13` — `Fulfill`
//! * `14` — `Reject`
//!
//! All multi-byte integers are big-endian. Expiry timestamps are
//! 17 ASCII bytes formatted as `YYYYMMDDHHMMSSmmm` in UTC.

use crate::address::Address;
use crate::condition::{Condition, Fulfillment};
use crate::error::{IlpError, Result};
use crate::oer::{
    decode_length, decode_u64, decode_var_octet_string, encode_length, encode_u64,
    encode_var_octet_string,
};

/// Packet-type discriminators used on the wire.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PacketType {
    /// `Prepare` (12).
    Prepare = 12,
    /// `Fulfill` (13).
    Fulfill = 13,
    /// `Reject` (14).
    Reject = 14,
}

impl PacketType {
    fn from_u8(b: u8) -> Result<Self> {
        match b {
            12 => Ok(PacketType::Prepare),
            13 => Ok(PacketType::Fulfill),
            14 => Ok(PacketType::Reject),
            other => Err(IlpError::UnknownPacketType(other)),
        }
    }
}

/// Canonical ILP reject codes from IL-RFC-27. Three-character ASCII.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RejectCode {
    /// `F00` — Bad request.
    F00BadRequest,
    /// `F01` — Invalid packet.
    F01InvalidPacket,
    /// `F02` — Unreachable.
    F02Unreachable,
    /// `F03` — Invalid amount.
    F03InvalidAmount,
    /// `F04` — Insufficient destination amount.
    F04InsufficientDestinationAmount,
    /// `F05` — Wrong condition.
    F05WrongCondition,
    /// `F06` — Unexpected payment.
    F06UnexpectedPayment,
    /// `F07` — Cannot receive.
    F07CannotReceive,
    /// `F08` — Amount too large.
    F08AmountTooLarge,
    /// `F99` — Application error.
    F99ApplicationError,
    /// `T00` — Internal error.
    T00InternalError,
    /// `T01` — Peer unreachable.
    T01PeerUnreachable,
    /// `T02` — Peer busy.
    T02PeerBusy,
    /// `T03` — Connector busy.
    T03ConnectorBusy,
    /// `T04` — Insufficient liquidity.
    T04InsufficientLiquidity,
    /// `T05` — Rate limited.
    T05RateLimited,
    /// `R00` — Transfer timed out.
    R00TransferTimedOut,
    /// `R01` — Insufficient source amount.
    R01InsufficientSourceAmount,
    /// `R99` — Application reject.
    R99ApplicationReject,
}

impl RejectCode {
    /// Return the on-wire three-character ASCII code.
    pub fn as_str(self) -> &'static str {
        match self {
            RejectCode::F00BadRequest => "F00",
            RejectCode::F01InvalidPacket => "F01",
            RejectCode::F02Unreachable => "F02",
            RejectCode::F03InvalidAmount => "F03",
            RejectCode::F04InsufficientDestinationAmount => "F04",
            RejectCode::F05WrongCondition => "F05",
            RejectCode::F06UnexpectedPayment => "F06",
            RejectCode::F07CannotReceive => "F07",
            RejectCode::F08AmountTooLarge => "F08",
            RejectCode::F99ApplicationError => "F99",
            RejectCode::T00InternalError => "T00",
            RejectCode::T01PeerUnreachable => "T01",
            RejectCode::T02PeerBusy => "T02",
            RejectCode::T03ConnectorBusy => "T03",
            RejectCode::T04InsufficientLiquidity => "T04",
            RejectCode::T05RateLimited => "T05",
            RejectCode::R00TransferTimedOut => "R00",
            RejectCode::R01InsufficientSourceAmount => "R01",
            RejectCode::R99ApplicationReject => "R99",
        }
    }

    /// Parse a three-character ASCII code into a `RejectCode`. Unknown
    /// codes collapse to `F00`.
    pub fn from_str(s: &str) -> Self {
        match s {
            "F00" => RejectCode::F00BadRequest,
            "F01" => RejectCode::F01InvalidPacket,
            "F02" => RejectCode::F02Unreachable,
            "F03" => RejectCode::F03InvalidAmount,
            "F04" => RejectCode::F04InsufficientDestinationAmount,
            "F05" => RejectCode::F05WrongCondition,
            "F06" => RejectCode::F06UnexpectedPayment,
            "F07" => RejectCode::F07CannotReceive,
            "F08" => RejectCode::F08AmountTooLarge,
            "F99" => RejectCode::F99ApplicationError,
            "T00" => RejectCode::T00InternalError,
            "T01" => RejectCode::T01PeerUnreachable,
            "T02" => RejectCode::T02PeerBusy,
            "T03" => RejectCode::T03ConnectorBusy,
            "T04" => RejectCode::T04InsufficientLiquidity,
            "T05" => RejectCode::T05RateLimited,
            "R00" => RejectCode::R00TransferTimedOut,
            "R01" => RejectCode::R01InsufficientSourceAmount,
            "R99" => RejectCode::R99ApplicationReject,
            _ => RejectCode::F00BadRequest,
        }
    }
}

/// `Prepare` packet payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Prepare {
    /// Amount to be delivered to the next hop, in that hop's smallest
    /// unit. Big-endian uint64 on the wire.
    pub amount: u64,
    /// Expiry timestamp as 17 ASCII bytes `YYYYMMDDHHMMSSmmm` (UTC).
    pub expires_at: [u8; 17],
    /// 32-byte SHA-256 condition.
    pub condition: Condition,
    /// Destination ILP address.
    pub destination: Address,
    /// Application data (STREAM frames, SPSP payload, etc.) up to 32 767 bytes.
    pub data: Vec<u8>,
}

/// `Fulfill` packet payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Fulfill {
    /// 32-byte preimage of the corresponding `Prepare`'s condition.
    pub fulfillment: Fulfillment,
    /// Application data (STREAM acknowledgements, etc.).
    pub data: Vec<u8>,
}

/// `Reject` packet payload.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Reject {
    /// Three-character reject code.
    pub code: RejectCode,
    /// ILP address that triggered the reject (may be empty).
    pub triggered_by: String,
    /// Human-readable message (UTF-8, may be empty).
    pub message: String,
    /// Application data.
    pub data: Vec<u8>,
}

/// Sum type over all three ILPv4 wire packets.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum IlpPacket {
    /// `Prepare` (type 12).
    Prepare(Prepare),
    /// `Fulfill` (type 13).
    Fulfill(Fulfill),
    /// `Reject` (type 14).
    Reject(Reject),
}

impl IlpPacket {
    /// Encode a packet to its on-wire byte representation.
    pub fn encode(&self) -> Vec<u8> {
        match self {
            IlpPacket::Prepare(p) => encode_with_header(PacketType::Prepare, |body| {
                encode_prepare_body(p, body);
            }),
            IlpPacket::Fulfill(f) => encode_with_header(PacketType::Fulfill, |body| {
                encode_fulfill_body(f, body);
            }),
            IlpPacket::Reject(r) => encode_with_header(PacketType::Reject, |body| {
                encode_reject_body(r, body);
            }),
        }
    }

    /// Decode a packet from its on-wire bytes.
    pub fn decode(input: &[u8]) -> Result<Self> {
        let type_byte = *input
            .first()
            .ok_or_else(|| IlpError::InvalidPacket("empty input".into()))?;
        let ty = PacketType::from_u8(type_byte)?;
        let (len, prefix) = decode_length(&input[1..])?;
        let body_start = 1 + prefix;
        let body_end = body_start
            .checked_add(len)
            .ok_or_else(|| IlpError::InvalidPacket("body length overflow".into()))?;
        if input.len() < body_end {
            return Err(IlpError::InvalidPacket("packet truncated".into()));
        }
        let body = &input[body_start..body_end];
        match ty {
            PacketType::Prepare => Ok(IlpPacket::Prepare(decode_prepare_body(body)?)),
            PacketType::Fulfill => Ok(IlpPacket::Fulfill(decode_fulfill_body(body)?)),
            PacketType::Reject => Ok(IlpPacket::Reject(decode_reject_body(body)?)),
        }
    }
}

fn encode_with_header<F: FnOnce(&mut Vec<u8>)>(ty: PacketType, body_writer: F) -> Vec<u8> {
    let mut body = Vec::new();
    body_writer(&mut body);
    let mut out = Vec::with_capacity(body.len() + 9);
    out.push(ty as u8);
    encode_length(&mut out, body.len());
    out.extend_from_slice(&body);
    out
}

fn encode_prepare_body(p: &Prepare, out: &mut Vec<u8>) {
    encode_u64(out, p.amount);
    out.extend_from_slice(&p.expires_at);
    out.extend_from_slice(p.condition.as_bytes());
    encode_var_octet_string(out, p.destination.as_str().as_bytes());
    encode_var_octet_string(out, &p.data);
}

fn decode_prepare_body(body: &[u8]) -> Result<Prepare> {
    let mut cursor = 0;
    let (amount, used) = decode_u64(&body[cursor..])?;
    cursor += used;
    if body.len() < cursor + 17 {
        return Err(IlpError::InvalidPacket("expires_at truncated".into()));
    }
    let mut expires_at = [0u8; 17];
    expires_at.copy_from_slice(&body[cursor..cursor + 17]);
    cursor += 17;
    if body.len() < cursor + 32 {
        return Err(IlpError::InvalidPacket("condition truncated".into()));
    }
    let mut cond_bytes = [0u8; 32];
    cond_bytes.copy_from_slice(&body[cursor..cursor + 32]);
    cursor += 32;
    let (dest_bytes, used) = decode_var_octet_string(&body[cursor..])?;
    let dest = core::str::from_utf8(dest_bytes)
        .map_err(|e| IlpError::InvalidPacket(format!("destination utf-8: {e}")))?;
    let destination = Address::parse(dest)?;
    cursor += used;
    let (data_bytes, _) = decode_var_octet_string(&body[cursor..])?;
    Ok(Prepare {
        amount,
        expires_at,
        condition: Condition::new(cond_bytes),
        destination,
        data: data_bytes.to_vec(),
    })
}

fn encode_fulfill_body(f: &Fulfill, out: &mut Vec<u8>) {
    out.extend_from_slice(f.fulfillment.as_bytes());
    encode_var_octet_string(out, &f.data);
}

fn decode_fulfill_body(body: &[u8]) -> Result<Fulfill> {
    if body.len() < 32 {
        return Err(IlpError::InvalidPacket("fulfillment truncated".into()));
    }
    let mut bytes = [0u8; 32];
    bytes.copy_from_slice(&body[..32]);
    let (data_bytes, _) = decode_var_octet_string(&body[32..])?;
    Ok(Fulfill {
        fulfillment: Fulfillment::new(bytes),
        data: data_bytes.to_vec(),
    })
}

fn encode_reject_body(r: &Reject, out: &mut Vec<u8>) {
    let code = r.code.as_str().as_bytes();
    // Code is fixed three ASCII bytes — no length prefix per ILPv4 wire.
    out.extend_from_slice(code);
    encode_var_octet_string(out, r.triggered_by.as_bytes());
    encode_var_octet_string(out, r.message.as_bytes());
    encode_var_octet_string(out, &r.data);
}

fn decode_reject_body(body: &[u8]) -> Result<Reject> {
    if body.len() < 3 {
        return Err(IlpError::InvalidPacket("reject code truncated".into()));
    }
    let code_str = core::str::from_utf8(&body[..3])
        .map_err(|e| IlpError::InvalidPacket(format!("reject code utf-8: {e}")))?;
    let code = RejectCode::from_str(code_str);
    let mut cursor = 3;
    let (triggered_by_bytes, used) = decode_var_octet_string(&body[cursor..])?;
    let triggered_by = core::str::from_utf8(triggered_by_bytes)
        .map_err(|e| IlpError::InvalidPacket(format!("triggered_by utf-8: {e}")))?
        .to_string();
    cursor += used;
    let (message_bytes, used) = decode_var_octet_string(&body[cursor..])?;
    let message = core::str::from_utf8(message_bytes)
        .map_err(|e| IlpError::InvalidPacket(format!("message utf-8: {e}")))?
        .to_string();
    cursor += used;
    let (data_bytes, _) = decode_var_octet_string(&body[cursor..])?;
    Ok(Reject {
        code,
        triggered_by,
        message,
        data: data_bytes.to_vec(),
    })
}

/// Format a UTC datetime as the 17-byte ILP expiry string.
/// Convenience for constructing `Prepare::expires_at` from a
/// `chrono::DateTime<Utc>`.
pub fn format_expiry(dt: chrono::DateTime<chrono::Utc>) -> [u8; 17] {
    let formatted = dt.format("%Y%m%d%H%M%S%3f").to_string();
    let mut out = [b'0'; 17];
    let bytes = formatted.as_bytes();
    let n = bytes.len().min(17);
    out[..n].copy_from_slice(&bytes[..n]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::condition::Fulfillment;

    #[test]
    fn prepare_roundtrip() {
        let f = Fulfillment::new([3u8; 32]);
        let pkt = IlpPacket::Prepare(Prepare {
            amount: 1_000,
            expires_at: *b"20260524120000000",
            condition: f.condition(),
            destination: Address::parse("g.us.bank.alice").unwrap(),
            data: b"hello".to_vec(),
        });
        let wire = pkt.encode();
        assert_eq!(wire[0], 12);
        let back = IlpPacket::decode(&wire).unwrap();
        assert_eq!(pkt, back);
    }

    #[test]
    fn fulfill_roundtrip() {
        let pkt = IlpPacket::Fulfill(Fulfill {
            fulfillment: Fulfillment::new([9u8; 32]),
            data: vec![1, 2, 3],
        });
        let wire = pkt.encode();
        assert_eq!(wire[0], 13);
        let back = IlpPacket::decode(&wire).unwrap();
        assert_eq!(pkt, back);
    }

    #[test]
    fn reject_roundtrip() {
        let pkt = IlpPacket::Reject(Reject {
            code: RejectCode::F02Unreachable,
            triggered_by: "g.us.bank".to_string(),
            message: "no route".to_string(),
            data: vec![],
        });
        let wire = pkt.encode();
        assert_eq!(wire[0], 14);
        let back = IlpPacket::decode(&wire).unwrap();
        assert_eq!(pkt, back);
    }
}
