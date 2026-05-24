//! GGUF v3 read sketch.
//!
//! GGUF (llama.cpp) v3 file layout:
//!
//! ```text
//!   u32 magic       = 'G','G','U','F' (0x46554747 little-endian)
//!   u32 version     = 3
//!   u64 tensor_count
//!   u64 metadata_kv_count
//!   metadata_kv[metadata_kv_count]
//!   tensor_info[tensor_count]
//!   alignment padding
//!   tensor data
//! ```
//!
//! This module implements the *header* parse (magic + version + counts)
//! and a minimal string reader. It is deliberately not a full
//! GGUF reader — the goal is to demonstrate the wire format and give
//! downstream crates a starting point.

use crate::error::QuantError;

/// GGUF magic bytes ("GGUF", little-endian).
pub const GGUF_MAGIC: u32 = 0x4655_4747;
/// Supported GGUF version.
pub const GGUF_VERSION_V3: u32 = 3;

/// Parsed GGUF header.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GgufHeader {
    /// Format version.
    pub version: u32,
    /// Number of tensors recorded after the header.
    pub tensor_count: u64,
    /// Number of metadata key/value pairs.
    pub metadata_kv_count: u64,
}

/// Parse a GGUF header from a byte slice.
pub fn parse_header(bytes: &[u8]) -> Result<GgufHeader, QuantError> {
    if bytes.len() < 24 {
        return Err(QuantError::Truncated {
            needed: 24,
            got: bytes.len(),
        });
    }
    let magic = read_u32(&bytes[0..4]);
    if magic != GGUF_MAGIC {
        return Err(QuantError::InvalidFormat("bad GGUF magic"));
    }
    let version = read_u32(&bytes[4..8]);
    if version != GGUF_VERSION_V3 {
        return Err(QuantError::InvalidFormat("unsupported GGUF version"));
    }
    let tensor_count = read_u64(&bytes[8..16]);
    let metadata_kv_count = read_u64(&bytes[16..24]);
    Ok(GgufHeader {
        version,
        tensor_count,
        metadata_kv_count,
    })
}

/// Write a fresh GGUF header (helper for tests + tooling).
pub fn write_header(h: &GgufHeader) -> Vec<u8> {
    let mut out = Vec::with_capacity(24);
    out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    out.extend_from_slice(&h.version.to_le_bytes());
    out.extend_from_slice(&h.tensor_count.to_le_bytes());
    out.extend_from_slice(&h.metadata_kv_count.to_le_bytes());
    out
}

/// Read a GGUF length-prefixed string (u64 len + utf8 bytes).
pub fn read_gguf_string(bytes: &[u8]) -> Result<(String, usize), QuantError> {
    if bytes.len() < 8 {
        return Err(QuantError::Truncated {
            needed: 8,
            got: bytes.len(),
        });
    }
    let len = read_u64(&bytes[0..8]) as usize;
    let end = 8 + len;
    if bytes.len() < end {
        return Err(QuantError::Truncated {
            needed: end,
            got: bytes.len(),
        });
    }
    let s = String::from_utf8(bytes[8..end].to_vec())
        .map_err(|_| QuantError::InvalidFormat("non-utf8 GGUF string"))?;
    Ok((s, end))
}

fn read_u32(b: &[u8]) -> u32 {
    let mut a = [0u8; 4];
    a.copy_from_slice(&b[0..4]);
    u32::from_le_bytes(a)
}

fn read_u64(b: &[u8]) -> u64 {
    let mut a = [0u8; 8];
    a.copy_from_slice(&b[0..8]);
    u64::from_le_bytes(a)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn write_then_parse_header_roundtrip() {
        let h = GgufHeader {
            version: GGUF_VERSION_V3,
            tensor_count: 42,
            metadata_kv_count: 7,
        };
        let bytes = write_header(&h);
        let parsed = parse_header(&bytes).expect("parse");
        assert_eq!(parsed, h);
    }

    #[test]
    fn parse_rejects_bad_magic() {
        let bytes = vec![0u8; 24];
        let err = parse_header(&bytes);
        assert!(matches!(err, Err(QuantError::InvalidFormat(_))));
    }

    #[test]
    fn parse_rejects_truncated() {
        let bytes = vec![0u8; 10];
        let err = parse_header(&bytes);
        assert!(matches!(err, Err(QuantError::Truncated { .. })));
    }
}
