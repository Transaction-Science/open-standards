//! Content-addressable archive (CAR) v1 reader and writer.
//!
//! CAR is the IPLD-native serialisation used by AT Protocol to ship a
//! repository over the wire. A CARv1 file is:
//!
//! ```text
//! <varint headerLen> <headerCbor>
//! (<varint blockLen> <cid> <blockBytes>)*
//! ```
//!
//! The header is a CBOR map with two fields: `roots` (an array of CIDs)
//! and `version` (`1`).
//!
//! Block CIDs in AT Protocol are CIDv1 with codec `dag-cbor` (`0x71`) and
//! SHA-256 multihash (`0x12 0x20 …`). This module only supports that
//! shape; foreign codecs are rejected with [`AtprotoError::InvalidCid`].

use std::io::{Read, Write};

use serde::{Deserialize, Serialize};
use serde_bytes::ByteBuf;
use sha2::{Digest, Sha256};

use crate::error::AtprotoError;

/// IPLD codec code for `dag-cbor`.
pub const CODEC_DAG_CBOR: u64 = 0x71;
/// IPLD codec code for `raw`.
pub const CODEC_RAW: u64 = 0x55;
/// Multihash code for SHA-256.
pub const MH_SHA256: u64 = 0x12;
/// CIDv1 version byte.
pub const CID_V1: u64 = 0x01;

/// A CIDv1 carrying an `0x12` (SHA-256) 32-byte multihash.
///
/// This is the only CID shape used by AT Protocol repositories in
/// practice.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Cid {
    /// IPLD codec (`dag-cbor` or `raw`).
    pub codec: u64,
    /// 32-byte SHA-256 digest of the encoded block.
    pub digest: [u8; 32],
}

impl Cid {
    /// Compute the CID for a `dag-cbor` block: SHA-256 of `bytes` with
    /// codec `0x71`.
    pub fn dag_cbor(bytes: &[u8]) -> Self {
        let mut h = Sha256::new();
        h.update(bytes);
        let out = h.finalize();
        let mut digest = [0u8; 32];
        digest.copy_from_slice(&out);
        Self {
            codec: CODEC_DAG_CBOR,
            digest,
        }
    }

    /// Serialize the CID to its binary form (`<version><codec><mh-code><mh-len><digest>`).
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(36);
        write_varint(&mut out, CID_V1);
        write_varint(&mut out, self.codec);
        write_varint(&mut out, MH_SHA256);
        write_varint(&mut out, 32);
        out.extend_from_slice(&self.digest);
        out
    }

    /// Parse a CID from a byte slice. Returns the consumed length.
    pub fn read_from(src: &[u8]) -> Result<(Self, usize), AtprotoError> {
        let mut off = 0;
        let (version, n) = read_varint(&src[off..])?;
        off += n;
        if version != CID_V1 {
            return Err(AtprotoError::InvalidCid(format!(
                "expected CIDv1, got v{version}"
            )));
        }
        let (codec, n) = read_varint(&src[off..])?;
        off += n;
        if codec != CODEC_DAG_CBOR && codec != CODEC_RAW {
            return Err(AtprotoError::InvalidCid(format!(
                "unsupported codec 0x{codec:x}"
            )));
        }
        let (mh_code, n) = read_varint(&src[off..])?;
        off += n;
        if mh_code != MH_SHA256 {
            return Err(AtprotoError::InvalidCid(format!(
                "unsupported multihash 0x{mh_code:x}"
            )));
        }
        let (mh_len, n) = read_varint(&src[off..])?;
        off += n;
        if mh_len != 32 {
            return Err(AtprotoError::InvalidCid(format!(
                "unexpected digest length {mh_len}"
            )));
        }
        if src.len() < off + 32 {
            return Err(AtprotoError::InvalidCid(
                "truncated CID digest".into(),
            ));
        }
        let mut digest = [0u8; 32];
        digest.copy_from_slice(&src[off..off + 32]);
        Ok((Self { codec, digest }, off + 32))
    }

    /// Hex-encoded digest (lower-case), useful for logging and tests.
    pub fn to_hex(&self) -> String {
        hex_encode(&self.digest)
    }
}

impl Serialize for Cid {
    fn serialize<S: serde::Serializer>(
        &self,
        s: S,
    ) -> Result<S::Ok, S::Error> {
        // Serialize as CBOR tag-42 byte string per dag-cbor convention.
        // For our internal use we encode as a base32 string with a "b"
        // multibase prefix.
        s.collect_str(&format!(
            "b{}{}",
            hex_encode_lower_varint(self.codec),
            hex_encode(&self.digest)
        ))
    }
}

impl<'de> Deserialize<'de> for Cid {
    fn deserialize<D: serde::Deserializer<'de>>(
        d: D,
    ) -> Result<Self, D::Error> {
        let raw = String::deserialize(d)?;
        // Parse `b<codec-hex><digest-hex>`. Codec is a fixed two-hex-digit
        // value here for the dag-cbor / raw codecs we accept.
        let s = raw.strip_prefix('b').ok_or_else(|| {
            serde::de::Error::custom(format!("missing b prefix: {raw}"))
        })?;
        if s.len() != 2 + 64 {
            return Err(serde::de::Error::custom(format!(
                "bad cid length {}",
                s.len()
            )));
        }
        let codec_hex = &s[..2];
        let digest_hex = &s[2..];
        let codec = u64::from_str_radix(codec_hex, 16)
            .map_err(serde::de::Error::custom)?;
        let mut digest = [0u8; 32];
        let bytes = hex_decode(digest_hex).map_err(|e| {
            serde::de::Error::custom(format!("bad cid digest: {e}"))
        })?;
        if bytes.len() != 32 {
            return Err(serde::de::Error::custom("digest length not 32"));
        }
        digest.copy_from_slice(&bytes);
        Ok(Cid { codec, digest })
    }
}

/// A single block inside a CAR file: its CID plus the encoded bytes.
#[derive(Debug, Clone)]
pub struct CarBlock {
    /// The block's CID.
    pub cid: Cid,
    /// The encoded block bytes (DAG-CBOR for `dag-cbor` codec).
    pub data: Vec<u8>,
}

impl CarBlock {
    /// Build a `dag-cbor` block from `data`. The CID is computed from the
    /// block bytes.
    pub fn dag_cbor(data: Vec<u8>) -> Self {
        let cid = Cid::dag_cbor(&data);
        Self { cid, data }
    }
}

/// A CARv1 file: a list of root CIDs and a list of blocks.
#[derive(Debug, Clone, Default)]
pub struct CarFile {
    /// Root CIDs (typically a single commit CID).
    pub roots: Vec<Cid>,
    /// Encoded blocks, in storage order.
    pub blocks: Vec<CarBlock>,
}

impl CarFile {
    /// Construct a fresh CAR with the given roots and no blocks.
    pub fn new(roots: Vec<Cid>) -> Self {
        Self {
            roots,
            blocks: Vec::new(),
        }
    }

    /// Append a block.
    pub fn push(&mut self, block: CarBlock) {
        self.blocks.push(block);
    }

    /// Look up a block by CID.
    pub fn get(&self, cid: &Cid) -> Option<&CarBlock> {
        self.blocks.iter().find(|b| &b.cid == cid)
    }

    /// Encode the CAR file to bytes.
    pub fn encode(&self) -> Result<Vec<u8>, AtprotoError> {
        let mut out = Vec::with_capacity(64 + self.blocks.len() * 256);
        // Header is a CBOR map { roots: [CID...], version: 1 }.
        let header = CarHeader {
            version: 1,
            roots: self
                .roots
                .iter()
                .map(|c| ByteBuf::from(c.to_bytes()))
                .collect(),
        };
        let header_bytes = serde_cbor::to_vec(&header)?;
        write_varint(&mut out, header_bytes.len() as u64);
        out.extend_from_slice(&header_bytes);
        for block in &self.blocks {
            let cid_bytes = block.cid.to_bytes();
            let total = cid_bytes.len() + block.data.len();
            write_varint(&mut out, total as u64);
            out.extend_from_slice(&cid_bytes);
            out.extend_from_slice(&block.data);
        }
        Ok(out)
    }

    /// Decode a CAR file from bytes.
    pub fn decode(mut src: &[u8]) -> Result<Self, AtprotoError> {
        let (header_len, n) = read_varint(src)?;
        src = &src[n..];
        if (src.len() as u64) < header_len {
            return Err(AtprotoError::InvalidCar(
                "truncated header".into(),
            ));
        }
        let header_bytes = &src[..header_len as usize];
        let header: CarHeader = serde_cbor::from_slice(header_bytes)?;
        if header.version != 1 {
            return Err(AtprotoError::InvalidCar(format!(
                "unsupported CAR version {}",
                header.version
            )));
        }
        src = &src[header_len as usize..];
        let mut roots = Vec::with_capacity(header.roots.len());
        for r in &header.roots {
            let (cid, _) = Cid::read_from(r.as_ref())?;
            roots.push(cid);
        }
        let mut blocks = Vec::new();
        while !src.is_empty() {
            let (block_len, n) = read_varint(src)?;
            src = &src[n..];
            if (src.len() as u64) < block_len {
                return Err(AtprotoError::InvalidCar(
                    "truncated block".into(),
                ));
            }
            let chunk = &src[..block_len as usize];
            let (cid, cid_n) = Cid::read_from(chunk)?;
            let data = chunk[cid_n..].to_vec();
            blocks.push(CarBlock { cid, data });
            src = &src[block_len as usize..];
        }
        Ok(Self { roots, blocks })
    }

    /// Encode + write the CAR to a `Write` sink.
    pub fn write_to<W: Write>(&self, w: &mut W) -> Result<(), AtprotoError> {
        let buf = self.encode()?;
        w.write_all(&buf)
            .map_err(|e| AtprotoError::InvalidCar(e.to_string()))
    }

    /// Read a CAR from a `Read` source. Reads the source to end first.
    pub fn read_from<R: Read>(r: &mut R) -> Result<Self, AtprotoError> {
        let mut buf = Vec::new();
        r.read_to_end(&mut buf)
            .map_err(|e| AtprotoError::InvalidCar(e.to_string()))?;
        Self::decode(&buf)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct CarHeader {
    version: u64,
    roots: Vec<ByteBuf>,
}

/// Encode an unsigned varint (MSB continuation, little-endian groups).
pub fn write_varint(out: &mut Vec<u8>, mut value: u64) {
    while value >= 0x80 {
        out.push(((value as u8) & 0x7f) | 0x80);
        value >>= 7;
    }
    out.push(value as u8);
}

/// Decode an unsigned varint. Returns (value, consumed bytes).
pub fn read_varint(src: &[u8]) -> Result<(u64, usize), AtprotoError> {
    let mut value: u64 = 0;
    let mut shift = 0u32;
    for (i, b) in src.iter().enumerate() {
        if shift >= 64 {
            return Err(AtprotoError::InvalidCar("varint overflow".into()));
        }
        value |= ((*b & 0x7f) as u64) << shift;
        if *b & 0x80 == 0 {
            return Ok((value, i + 1));
        }
        shift += 7;
    }
    Err(AtprotoError::InvalidCar("truncated varint".into()))
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        s.push(nibble_to_hex(b >> 4));
        s.push(nibble_to_hex(b & 0x0f));
    }
    s
}

fn hex_encode_lower_varint(codec: u64) -> String {
    // Render the codec as a fixed two hex digits when it fits in one byte.
    format!("{:02x}", codec & 0xff)
}

fn hex_decode(s: &str) -> Result<Vec<u8>, String> {
    if !s.len().is_multiple_of(2) {
        return Err("odd length".into());
    }
    let bytes = s.as_bytes();
    let mut out = Vec::with_capacity(s.len() / 2);
    for chunk in bytes.chunks(2) {
        let hi = hex_to_nibble(chunk[0])?;
        let lo = hex_to_nibble(chunk[1])?;
        out.push((hi << 4) | lo);
    }
    Ok(out)
}

fn nibble_to_hex(n: u8) -> char {
    match n {
        0..=9 => (b'0' + n) as char,
        10..=15 => (b'a' + (n - 10)) as char,
        _ => '?',
    }
}

fn hex_to_nibble(c: u8) -> Result<u8, String> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(10 + c - b'a'),
        b'A'..=b'F' => Ok(10 + c - b'A'),
        _ => Err(format!("non-hex byte {c:#x}")),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn varint_roundtrip() {
        for v in [0u64, 1, 127, 128, 300, 16_384, 1 << 30, u64::MAX / 2] {
            let mut buf = Vec::new();
            write_varint(&mut buf, v);
            let (got, n) = read_varint(&buf).unwrap();
            assert_eq!(got, v);
            assert_eq!(n, buf.len());
        }
    }

    #[test]
    fn cid_roundtrip() {
        let cid = Cid::dag_cbor(b"hello");
        let bytes = cid.to_bytes();
        let (got, n) = Cid::read_from(&bytes).unwrap();
        assert_eq!(got, cid);
        assert_eq!(n, bytes.len());
    }
}
