//! STREAM — Send Transports for Real-time Exchange of Assets and
//! Messages — frame types.
//!
//! STREAM multiplexes typed frames over the application-data field of
//! ILP `Prepare` / `Fulfill` / `Reject` packets. A frame is a
//! `(type, var-octet-string)` pair; multiple frames can ride in one
//! packet.
//!
//! This module covers the frame layer only: callers AEAD-encrypt the
//! frame bytes themselves before placing them in `Prepare::data`.

use crate::error::{IlpError, Result};
use crate::oer::{
    decode_length, decode_u64, decode_var_octet_string, encode_length, encode_u64,
    encode_var_octet_string,
};

/// STREAM frame-type discriminators per IL-RFC-29.
#[repr(u8)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FrameType {
    /// `01` — Connection close.
    ConnectionClose = 0x01,
    /// `02` — Connection new address.
    ConnectionNewAddress = 0x02,
    /// `03` — Connection asset details.
    ConnectionAssetDetails = 0x03,
    /// `04` — Connection max data.
    ConnectionMaxData = 0x04,
    /// `05` — Connection data blocked.
    ConnectionDataBlocked = 0x05,
    /// `06` — Connection max stream id.
    ConnectionMaxStreamId = 0x06,
    /// `07` — Connection stream id blocked.
    ConnectionStreamIdBlocked = 0x07,
    /// `10` — Stream close.
    StreamClose = 0x10,
    /// `11` — Stream money.
    StreamMoney = 0x11,
    /// `12` — Stream max money.
    StreamMaxMoney = 0x12,
    /// `13` — Stream money blocked.
    StreamMoneyBlocked = 0x13,
    /// `14` — Stream data.
    StreamData = 0x14,
    /// `15` — Stream max data.
    StreamMaxData = 0x15,
    /// `16` — Stream data blocked.
    StreamDataBlocked = 0x16,
    /// `17` — Stream receipt.
    StreamReceipt = 0x17,
}

impl FrameType {
    /// Map a wire byte to a frame type.
    pub fn from_u8(b: u8) -> Result<Self> {
        Ok(match b {
            0x01 => FrameType::ConnectionClose,
            0x02 => FrameType::ConnectionNewAddress,
            0x03 => FrameType::ConnectionAssetDetails,
            0x04 => FrameType::ConnectionMaxData,
            0x05 => FrameType::ConnectionDataBlocked,
            0x06 => FrameType::ConnectionMaxStreamId,
            0x07 => FrameType::ConnectionStreamIdBlocked,
            0x10 => FrameType::StreamClose,
            0x11 => FrameType::StreamMoney,
            0x12 => FrameType::StreamMaxMoney,
            0x13 => FrameType::StreamMoneyBlocked,
            0x14 => FrameType::StreamData,
            0x15 => FrameType::StreamMaxData,
            0x16 => FrameType::StreamDataBlocked,
            0x17 => FrameType::StreamReceipt,
            other => return Err(IlpError::UnknownStreamFrame(other)),
        })
    }
}

/// A single STREAM frame. The `payload` is the opaque
/// type-specific body; callers shape it per IL-RFC-29.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Frame {
    /// Frame discriminator.
    pub frame_type: FrameType,
    /// Frame-specific payload bytes (already serialized).
    pub payload: Vec<u8>,
}

impl Frame {
    /// Construct a `StreamMoney` frame for `stream_id` carrying `shares`
    /// units of value.
    pub fn stream_money(stream_id: u64, shares: u64) -> Self {
        let mut body = Vec::new();
        encode_u64(&mut body, stream_id);
        encode_u64(&mut body, shares);
        Self {
            frame_type: FrameType::StreamMoney,
            payload: body,
        }
    }

    /// Construct a `StreamData` frame for `stream_id` at `offset`.
    pub fn stream_data(stream_id: u64, offset: u64, data: &[u8]) -> Self {
        let mut body = Vec::new();
        encode_u64(&mut body, stream_id);
        encode_u64(&mut body, offset);
        encode_var_octet_string(&mut body, data);
        Self {
            frame_type: FrameType::StreamData,
            payload: body,
        }
    }
}

/// A STREAM packet — sequence number, ILP packet type echo, prepare
/// amount, and an ordered list of frames.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StreamPacket {
    /// 4-byte protocol version (IL-RFC-29 v1 == 1).
    pub version: u8,
    /// Echo of the carrying ILP packet type (12 / 13 / 14).
    pub ilp_packet_type: u8,
    /// Monotonic sequence number assigned by the sender.
    pub sequence: u64,
    /// Amount the carrying `Prepare` is expected to deliver (used for
    /// minimum-acceptable-amount checks).
    pub prepare_amount: u64,
    /// Ordered list of frames.
    pub frames: Vec<Frame>,
}

impl StreamPacket {
    /// Encode the packet to its on-wire byte representation.
    pub fn encode(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(32 + self.frames.iter().map(|f| f.payload.len() + 6).sum::<usize>());
        out.push(self.version);
        out.push(self.ilp_packet_type);
        encode_u64(&mut out, self.sequence);
        encode_u64(&mut out, self.prepare_amount);
        encode_length(&mut out, self.frames.len());
        for frame in &self.frames {
            out.push(frame.frame_type as u8);
            encode_var_octet_string(&mut out, &frame.payload);
        }
        out
    }

    /// Decode the packet from its on-wire bytes.
    pub fn decode(input: &[u8]) -> Result<Self> {
        if input.len() < 2 {
            return Err(IlpError::InvalidPacket("stream packet header".into()));
        }
        let version = input[0];
        let ilp_packet_type = input[1];
        let mut cursor = 2;
        let (sequence, used) = decode_u64(&input[cursor..])?;
        cursor += used;
        let (prepare_amount, used) = decode_u64(&input[cursor..])?;
        cursor += used;
        let (n_frames, used) = decode_length(&input[cursor..])?;
        cursor += used;
        let mut frames = Vec::with_capacity(n_frames);
        for _ in 0..n_frames {
            if cursor >= input.len() {
                return Err(IlpError::InvalidPacket("stream frame header truncated".into()));
            }
            let frame_type = FrameType::from_u8(input[cursor])?;
            cursor += 1;
            let (payload, used) = decode_var_octet_string(&input[cursor..])?;
            cursor += used;
            frames.push(Frame {
                frame_type,
                payload: payload.to_vec(),
            });
        }
        Ok(StreamPacket {
            version,
            ilp_packet_type,
            sequence,
            prepare_amount,
            frames,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn money_frame_roundtrip() {
        let pkt = StreamPacket {
            version: 1,
            ilp_packet_type: 12,
            sequence: 7,
            prepare_amount: 100,
            frames: vec![Frame::stream_money(1, 100), Frame::stream_data(1, 0, b"hi")],
        };
        let wire = pkt.encode();
        let back = StreamPacket::decode(&wire).unwrap();
        assert_eq!(pkt, back);
    }
}
