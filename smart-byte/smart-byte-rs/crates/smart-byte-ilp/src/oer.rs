//! Octet Encoding Rules — the minimal subset ILPv4 actually uses.
//!
//! ILPv4 packets do not use full ASN.1 OER. They use a deliberately
//! tiny subset:
//!
//! * A **length-determinant** that is either a single byte `< 128` or a
//!   "long form" byte `0x80 | n` followed by `n` big-endian bytes of
//!   length.
//! * A **variable-length octet string** which is a length-determinant
//!   followed by that many bytes.
//! * Fixed-size big-endian integers (the amount in `Prepare` is exactly
//!   8 bytes).
//! * A fixed 17-byte expiry timestamp formatted as
//!   `YYYYMMDDHHMMSSmmm` (UTC).
//!
//! This module exposes just those helpers; everything else is hand-coded
//! in [`crate::packet`].

use crate::error::{IlpError, Result};

/// Encode a length-determinant per ILPv4 conventions.
///
/// Lengths `< 128` are written as a single byte. Larger lengths use the
/// long form: `0x80 | n` where `n` is the number of length bytes that
/// follow, in big-endian order.
pub fn encode_length(out: &mut Vec<u8>, len: usize) {
    if len < 128 {
        out.push(len as u8);
        return;
    }
    let mut tmp = Vec::with_capacity(8);
    let mut v = len;
    while v > 0 {
        tmp.push((v & 0xff) as u8);
        v >>= 8;
    }
    tmp.reverse();
    out.push(0x80 | tmp.len() as u8);
    out.extend_from_slice(&tmp);
}

/// Decode a length-determinant and return `(length, bytes_consumed)`.
pub fn decode_length(input: &[u8]) -> Result<(usize, usize)> {
    let first = *input
        .first()
        .ok_or_else(|| IlpError::Oer("length determinant: empty input".into()))?;
    if first < 128 {
        return Ok((first as usize, 1));
    }
    let n = (first & 0x7f) as usize;
    if n == 0 {
        return Err(IlpError::Oer("indefinite length not supported".into()));
    }
    if input.len() < 1 + n {
        return Err(IlpError::Oer("length determinant truncated".into()));
    }
    if n > core::mem::size_of::<usize>() {
        return Err(IlpError::Oer("length determinant too large".into()));
    }
    let mut len: usize = 0;
    for &b in &input[1..1 + n] {
        len = (len << 8) | (b as usize);
    }
    Ok((len, 1 + n))
}

/// Encode an octet string with its length determinant prefix.
pub fn encode_var_octet_string(out: &mut Vec<u8>, bytes: &[u8]) {
    encode_length(out, bytes.len());
    out.extend_from_slice(bytes);
}

/// Decode an octet string and return `(bytes, bytes_consumed)`.
pub fn decode_var_octet_string(input: &[u8]) -> Result<(&[u8], usize)> {
    let (len, prefix) = decode_length(input)?;
    let end = prefix
        .checked_add(len)
        .ok_or_else(|| IlpError::Oer("octet string length overflow".into()))?;
    if input.len() < end {
        return Err(IlpError::Oer("octet string truncated".into()));
    }
    Ok((&input[prefix..end], end))
}

/// Encode a big-endian unsigned 64-bit integer (used for ILP amounts).
pub fn encode_u64(out: &mut Vec<u8>, value: u64) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// Decode a big-endian unsigned 64-bit integer.
pub fn decode_u64(input: &[u8]) -> Result<(u64, usize)> {
    if input.len() < 8 {
        return Err(IlpError::Oer("u64 truncated".into()));
    }
    let mut buf = [0u8; 8];
    buf.copy_from_slice(&input[..8]);
    Ok((u64::from_be_bytes(buf), 8))
}

/// Encode a big-endian unsigned 32-bit integer.
pub fn encode_u32(out: &mut Vec<u8>, value: u32) {
    out.extend_from_slice(&value.to_be_bytes());
}

/// Decode a big-endian unsigned 32-bit integer.
pub fn decode_u32(input: &[u8]) -> Result<(u32, usize)> {
    if input.len() < 4 {
        return Err(IlpError::Oer("u32 truncated".into()));
    }
    let mut buf = [0u8; 4];
    buf.copy_from_slice(&input[..4]);
    Ok((u32::from_be_bytes(buf), 4))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_short_form_roundtrip() {
        for n in [0_usize, 1, 64, 127] {
            let mut buf = Vec::new();
            encode_length(&mut buf, n);
            assert_eq!(buf.len(), 1);
            let (got, used) = decode_length(&buf).unwrap();
            assert_eq!(got, n);
            assert_eq!(used, 1);
        }
    }

    #[test]
    fn length_long_form_roundtrip() {
        for n in [128_usize, 256, 65_535, 1_000_000] {
            let mut buf = Vec::new();
            encode_length(&mut buf, n);
            let (got, used) = decode_length(&buf).unwrap();
            assert_eq!(got, n);
            assert_eq!(used, buf.len());
        }
    }

    #[test]
    fn octet_string_roundtrip() {
        let payload = b"hello world";
        let mut buf = Vec::new();
        encode_var_octet_string(&mut buf, payload);
        let (got, used) = decode_var_octet_string(&buf).unwrap();
        assert_eq!(got, payload);
        assert_eq!(used, buf.len());
    }

    #[test]
    fn u64_roundtrip() {
        let mut buf = Vec::new();
        encode_u64(&mut buf, 0x0102_0304_0506_0708);
        let (got, used) = decode_u64(&buf).unwrap();
        assert_eq!(got, 0x0102_0304_0506_0708);
        assert_eq!(used, 8);
    }
}
