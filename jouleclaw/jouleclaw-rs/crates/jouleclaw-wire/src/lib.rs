//! Joule wire protocol (R9).
//!
//! A binary, length-prefixed protocol for inter-process / inter-node
//! cascade traffic. Every message carries its energy cost, origin
//! tier, confidence, and expiry — the receiver knows what it's paying
//! for and whether to trust it.
//!
//! # Frame layout
//!
//! ```text
//!  +0:   8 bytes   "JOULEW01"           magic + version
//!  +8:   1 byte    kind                  Request=0, Quote=1, Response=2, Error=3
//!  +9:   8 bytes   payload_len (BE u64)
//! +17:   N bytes   payload
//! +N+17: 64 bytes  signature placeholder (all zeros for R9; reserved)
//! ```
//!
//! Every message ends with the signature field. R9 leaves it as zeros;
//! a future crypto layer fills it. The receiver verifies length but
//! not signature contents in R9.
//!
//! # Payloads
//!
//! ## Request
//! ```text
//! +0:    32 bytes  query_key (content-addressed hash of the query)
//! +32:   8 bytes   max_joules (f64 BE) — caller's budget for this lookup
//! +40:   8 bytes   request_id (u64 BE) — opaque tag for matching responses
//! ```
//!
//! ## Quote
//! ```text
//! +0:    32 bytes  query_key
//! +32:   8 bytes   request_id
//! +40:   8 bytes   joules_charged (f64 BE) — what serving the request will cost
//! +48:   4 bytes   confidence (f32 BE)
//! +52:   1 byte    origin_tier_family (0=L0, 1=L1, 2=L2, 3=L3, 4=L4)
//! +53:   4 bytes   origin_tier_data (BE)
//! +57:   8 bytes   expiry_secs (u64 BE) — Unix epoch; 0 means no expiry
//! ```
//!
//! ## Response
//! ```text
//! +0:    32 bytes  query_key
//! +32:   8 bytes   request_id
//! +40:   8 bytes   joules_charged (f64 BE)
//! +48:   4 bytes   confidence (f32 BE)
//! +52:   1 byte    origin_tier_family
//! +53:   4 bytes   origin_tier_data
//! +57:   8 bytes   expiry_secs (u64 BE)
//! +65:   1 byte    output_kind (0=Text, 1=Structured, 2=Refused)
//! +66:   4 bytes   payload_len (BE u32)
//! +70:   payload_len bytes  output payload
//! ```
//!
//! Output payload for Text is UTF-8 bytes; for Structured is raw bytes.
//! Refused output uses a 1-byte reason variant (matching the disk
//! format's refusal encoding from R3).
//!
//! ## Error
//! ```text
//! +0:    8 bytes   request_id
//! +8:    1 byte    code (0=NotFound, 1=BudgetExceeded, 2=Malformed, 3=Other)
//! +9:    4 bytes   message_len (BE u32)
//! +13:   message_len bytes  UTF-8 message
//! ```

use jouleclaw_cascade::*;

pub mod rpc;
pub use rpc::{RpcTier, Transport, serve_request};

pub const MAGIC: &[u8; 8] = b"JOULEW01";
pub const SIGNATURE_LEN: usize = 64;

#[derive(Debug, Clone)]
pub enum WireMessage {
    Request(WireRequest),
    Quote(WireQuote),
    Response(WireResponse),
    Error(WireError),
}

#[derive(Debug, Clone)]
pub struct WireRequest {
    pub query_key: [u8; 32],
    pub max_joules: f64,
    pub request_id: u64,
}

#[derive(Debug, Clone)]
pub struct WireQuote {
    pub query_key: [u8; 32],
    pub request_id: u64,
    pub joules_charged: f64,
    pub confidence: f32,
    pub origin_tier: TierId,
    pub expiry_secs: u64,
}

#[derive(Debug, Clone)]
pub struct WireResponse {
    pub query_key: [u8; 32],
    pub request_id: u64,
    pub joules_charged: f64,
    pub confidence: f32,
    pub origin_tier: TierId,
    pub expiry_secs: u64,
    pub output: AnswerOutput,
}

#[derive(Debug, Clone)]
pub struct WireError {
    pub request_id: u64,
    pub code: ErrorCode,
    pub message: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ErrorCode {
    NotFound = 0,
    BudgetExceeded = 1,
    Malformed = 2,
    Other = 3,
}

impl ErrorCode {
    fn from_u8(b: u8) -> Option<Self> {
        match b {
            0 => Some(Self::NotFound),
            1 => Some(Self::BudgetExceeded),
            2 => Some(Self::Malformed),
            3 => Some(Self::Other),
            _ => None,
        }
    }
}

#[derive(Debug)]
pub enum DecodeError {
    Truncated { needed: usize, have: usize },
    BadMagic,
    UnsupportedVersion(u32),
    UnknownKind(u8),
    BadTier(u8, u32),
    BadOutput(u8),
    BadString(std::string::FromUtf8Error),
    BadErrorCode(u8),
}

impl std::fmt::Display for DecodeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Truncated { needed, have } =>
                write!(f, "truncated: needed {} bytes, have {}", needed, have),
            Self::BadMagic => write!(f, "bad magic — not a Joule wire message"),
            Self::UnsupportedVersion(v) => write!(f, "unsupported version: {}", v),
            Self::UnknownKind(b) => write!(f, "unknown message kind {}", b),
            Self::BadTier(family, data) =>
                write!(f, "bad tier family={} data={}", family, data),
            Self::BadOutput(b) => write!(f, "bad output kind {}", b),
            Self::BadString(e) => write!(f, "non-utf8 string: {}", e),
            Self::BadErrorCode(b) => write!(f, "bad error code {}", b),
        }
    }
}

impl std::error::Error for DecodeError {}

// ============================================================
// Encoding
// ============================================================

pub fn encode(msg: &WireMessage) -> Vec<u8> {
    let mut out = Vec::with_capacity(128);
    // Magic.
    out.extend_from_slice(MAGIC);

    // Kind.
    let kind: u8 = match msg {
        WireMessage::Request(_) => 0,
        WireMessage::Quote(_) => 1,
        WireMessage::Response(_) => 2,
        WireMessage::Error(_) => 3,
    };
    out.push(kind);

    // Payload (will rewrite the length prefix after building).
    let payload = encode_payload(msg);
    out.extend_from_slice(&(payload.len() as u64).to_be_bytes());
    out.extend_from_slice(&payload);

    // Signature placeholder (zeros).
    out.extend_from_slice(&[0u8; SIGNATURE_LEN]);
    out
}

fn encode_payload(msg: &WireMessage) -> Vec<u8> {
    let mut p = Vec::with_capacity(128);
    match msg {
        WireMessage::Request(r) => {
            p.extend_from_slice(&r.query_key);
            p.extend_from_slice(&r.max_joules.to_be_bytes());
            p.extend_from_slice(&r.request_id.to_be_bytes());
        }
        WireMessage::Quote(q) => {
            p.extend_from_slice(&q.query_key);
            p.extend_from_slice(&q.request_id.to_be_bytes());
            p.extend_from_slice(&q.joules_charged.to_be_bytes());
            p.extend_from_slice(&q.confidence.to_be_bytes());
            let (family, data) = encode_tier(&q.origin_tier);
            p.push(family);
            p.extend_from_slice(&data.to_be_bytes());
            p.extend_from_slice(&q.expiry_secs.to_be_bytes());
        }
        WireMessage::Response(r) => {
            p.extend_from_slice(&r.query_key);
            p.extend_from_slice(&r.request_id.to_be_bytes());
            p.extend_from_slice(&r.joules_charged.to_be_bytes());
            p.extend_from_slice(&r.confidence.to_be_bytes());
            let (family, data) = encode_tier(&r.origin_tier);
            p.push(family);
            p.extend_from_slice(&data.to_be_bytes());
            p.extend_from_slice(&r.expiry_secs.to_be_bytes());
            encode_output(&r.output, &mut p);
        }
        WireMessage::Error(e) => {
            p.extend_from_slice(&e.request_id.to_be_bytes());
            p.push(e.code as u8);
            let bytes = e.message.as_bytes();
            p.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
            p.extend_from_slice(bytes);
        }
    }
    p
}

fn encode_tier(tier: &TierId) -> (u8, u32) {
    // L0-L4 wire encoding is byte-stable for receipts (SPEC §7). The L0-L10
    // fractional tiers added in v0.2 collapse onto their coarse class for
    // wire transport; tier precision is preserved in receipt.wire_tag.
    use jouleclaw_cascade::JouleClass;
    match tier {
        TierId::L0 => (0, 0),
        TierId::L1(prim) => (1, *prim as u32),
        TierId::L2(m) => (2, m.0),
        TierId::L3(m) => (3, m.0),
        TierId::L4(m) => (4, m.0),
        other => match other.joule_class() {
            JouleClass::Cache => (0, 0),
            JouleClass::Lawful => (1, 0),
            JouleClass::Embed => (2, 0),
            JouleClass::Model => (3, 0),
            JouleClass::Wire => (4, 0),
            JouleClass::Meta => (5, 0),
        },
    }
}

fn encode_output(o: &AnswerOutput, p: &mut Vec<u8>) {
    match o {
        AnswerOutput::Text(s) => {
            p.push(0);
            let b = s.as_bytes();
            p.extend_from_slice(&(b.len() as u32).to_be_bytes());
            p.extend_from_slice(b);
        }
        AnswerOutput::Structured(b) => {
            p.push(1);
            p.extend_from_slice(&(b.len() as u32).to_be_bytes());
            p.extend_from_slice(b);
        }
        AnswerOutput::Refused(reason) => {
            p.push(2);
            // Use the same refusal encoding as the disk history format.
            let mut payload = Vec::new();
            match reason {
                RefusalReason::Inapplicable => {
                    payload.push(0);
                }
                RefusalReason::LowConfidence(c) => {
                    payload.push(1);
                    payload.extend_from_slice(&c.to_be_bytes());
                }
                RefusalReason::TierSpecific(s) => {
                    payload.push(2);
                    payload.extend_from_slice(&(s.len() as u32).to_be_bytes());
                    payload.extend_from_slice(s.as_bytes());
                }
            }
            p.extend_from_slice(&(payload.len() as u32).to_be_bytes());
            p.extend_from_slice(&payload);
        }
    }
}

// ============================================================
// Decoding
// ============================================================

pub fn decode(bytes: &[u8]) -> Result<WireMessage, DecodeError> {
    let mut c = Cursor::new(bytes);

    // Header.
    let magic = c.read_n(8)?;
    if magic != MAGIC {
        return Err(DecodeError::BadMagic);
    }

    let kind = c.read_u8()?;
    let payload_len = c.read_u64()? as usize;
    let payload = c.read_n(payload_len)?.to_vec();

    // Signature (verified for length only in R9; not for contents).
    let _sig = c.read_n(SIGNATURE_LEN)?;

    let mut pc = Cursor::new(&payload);
    let msg = match kind {
        0 => WireMessage::Request(decode_request(&mut pc)?),
        1 => WireMessage::Quote(decode_quote(&mut pc)?),
        2 => WireMessage::Response(decode_response(&mut pc)?),
        3 => WireMessage::Error(decode_error(&mut pc)?),
        k => return Err(DecodeError::UnknownKind(k)),
    };
    Ok(msg)
}

fn decode_request(c: &mut Cursor) -> Result<WireRequest, DecodeError> {
    let mut query_key = [0u8; 32];
    query_key.copy_from_slice(c.read_n(32)?);
    let max_joules = c.read_f64()?;
    let request_id = c.read_u64()?;
    Ok(WireRequest { query_key, max_joules, request_id })
}

fn decode_quote(c: &mut Cursor) -> Result<WireQuote, DecodeError> {
    let mut query_key = [0u8; 32];
    query_key.copy_from_slice(c.read_n(32)?);
    let request_id = c.read_u64()?;
    let joules_charged = c.read_f64()?;
    let confidence = c.read_f32()?;
    let family = c.read_u8()?;
    let data = c.read_u32()?;
    let origin_tier = decode_tier(family, data)?;
    let expiry_secs = c.read_u64()?;
    Ok(WireQuote {
        query_key, request_id, joules_charged, confidence,
        origin_tier, expiry_secs,
    })
}

fn decode_response(c: &mut Cursor) -> Result<WireResponse, DecodeError> {
    let mut query_key = [0u8; 32];
    query_key.copy_from_slice(c.read_n(32)?);
    let request_id = c.read_u64()?;
    let joules_charged = c.read_f64()?;
    let confidence = c.read_f32()?;
    let family = c.read_u8()?;
    let data = c.read_u32()?;
    let origin_tier = decode_tier(family, data)?;
    let expiry_secs = c.read_u64()?;
    let output = decode_output(c)?;
    Ok(WireResponse {
        query_key, request_id, joules_charged, confidence,
        origin_tier, expiry_secs, output,
    })
}

fn decode_error(c: &mut Cursor) -> Result<WireError, DecodeError> {
    let request_id = c.read_u64()?;
    let code_byte = c.read_u8()?;
    let code = ErrorCode::from_u8(code_byte)
        .ok_or(DecodeError::BadErrorCode(code_byte))?;
    let n = c.read_u32()? as usize;
    let bytes = c.read_n(n)?.to_vec();
    let message = String::from_utf8(bytes).map_err(DecodeError::BadString)?;
    Ok(WireError { request_id, code, message })
}

fn decode_tier(family: u8, data: u32) -> Result<TierId, DecodeError> {
    match family {
        0 => Ok(TierId::L0),
        1 => {
            let prim = match data {
                0 => L1Primitive::CacheLookup,
                1 => L1Primitive::Tokenize,
                2 => L1Primitive::Detokenize,
                3 => L1Primitive::Regex,
                4 => L1Primitive::Parse,
                5 => L1Primitive::TemplateFill,
                6 => L1Primitive::Retrieve,
                7 => L1Primitive::Execute,
                _ => return Err(DecodeError::BadTier(family, data)),
            };
            Ok(TierId::L1(prim))
        }
        2 => Ok(TierId::L2(L2ModelId(data))),
        3 => Ok(TierId::L3(L3ModelId(data))),
        4 => Ok(TierId::L4(L4ModelId(data))),
        _ => Err(DecodeError::BadTier(family, data)),
    }
}

fn decode_output(c: &mut Cursor) -> Result<AnswerOutput, DecodeError> {
    let kind = c.read_u8()?;
    let n = c.read_u32()? as usize;
    let payload = c.read_n(n)?.to_vec();
    match kind {
        0 => Ok(AnswerOutput::Text(
            String::from_utf8(payload).map_err(DecodeError::BadString)?
        )),
        1 => Ok(AnswerOutput::Structured(payload)),
        2 => {
            let mut sub = Cursor::new(&payload);
            let r_kind = sub.read_u8()?;
            match r_kind {
                0 => Ok(AnswerOutput::Refused(RefusalReason::Inapplicable)),
                1 => {
                    let c_val = sub.read_u32()?;
                    Ok(AnswerOutput::Refused(RefusalReason::LowConfidence(c_val)))
                }
                2 => {
                    let s_len = sub.read_u32()? as usize;
                    let s_bytes = sub.read_n(s_len)?.to_vec();
                    let s = String::from_utf8(s_bytes).map_err(DecodeError::BadString)?;
                    Ok(AnswerOutput::Refused(RefusalReason::TierSpecific(s)))
                }
                k => Err(DecodeError::BadOutput(k)),
            }
        }
        k => Err(DecodeError::BadOutput(k)),
    }
}

struct Cursor<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Cursor<'a> {
    fn new(bytes: &'a [u8]) -> Self { Self { bytes, pos: 0 } }
    fn read_n(&mut self, n: usize) -> Result<&'a [u8], DecodeError> {
        if self.pos + n > self.bytes.len() {
            return Err(DecodeError::Truncated {
                needed: self.pos + n, have: self.bytes.len(),
            });
        }
        let out = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }
    fn read_u8(&mut self) -> Result<u8, DecodeError> { Ok(self.read_n(1)?[0]) }
    fn read_u32(&mut self) -> Result<u32, DecodeError> {
        let b = self.read_n(4)?;
        Ok(u32::from_be_bytes([b[0], b[1], b[2], b[3]]))
    }
    fn read_u64(&mut self) -> Result<u64, DecodeError> {
        let b = self.read_n(8)?;
        Ok(u64::from_be_bytes(b.try_into().unwrap()))
    }
    fn read_f32(&mut self) -> Result<f32, DecodeError> {
        let b = self.read_n(4)?;
        Ok(f32::from_be_bytes(b.try_into().unwrap()))
    }
    fn read_f64(&mut self) -> Result<f64, DecodeError> {
        let b = self.read_n(8)?;
        Ok(f64::from_be_bytes(b.try_into().unwrap()))
    }
}
