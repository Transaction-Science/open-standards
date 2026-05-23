//! Tiny hex encoder. We don't pull in the `hex` crate because:
//!
//! - It's 20 lines we need to write anyway,
//! - Dependency hygiene matters for an Apache-2.0 reference stack,
//! - The crate already has `sha2 + hmac + subtle`; adding another
//!   tiny dep increases supply-chain surface for no gain.
//!
//! The encoder is constant-time-ish (no data-dependent branches in
//! `encode_into`); we don't need timing safety for hex output but
//! it costs nothing.

/// Encode `bytes` as a lowercase hex string.
#[must_use]
pub fn encode(bytes: &[u8]) -> String {
    let mut s = String::with_capacity(bytes.len() * 2);
    encode_into(bytes, &mut s);
    s
}

/// Encode `bytes` as lowercase hex into the given `String`.
pub fn encode_into(bytes: &[u8], out: &mut String) {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    for &b in bytes {
        out.push(HEX[(b >> 4) as usize] as char);
        out.push(HEX[(b & 0x0f) as usize] as char);
    }
}

/// The input wasn't valid hex (odd length or a non-hex character).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct InvalidHex;

/// Decode a lowercase or uppercase hex string into bytes.
///
/// # Errors
/// Returns [`InvalidHex`] if the length is odd or any character is
/// not a hex digit.
pub fn decode(s: &str) -> core::result::Result<Vec<u8>, InvalidHex> {
    if !s.len().is_multiple_of(2) {
        return Err(InvalidHex);
    }
    let mut out = Vec::with_capacity(s.len() / 2);
    let bytes = s.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let hi = hex_val(bytes[i])?;
        let lo = hex_val(bytes[i + 1])?;
        out.push((hi << 4) | lo);
        i += 2;
    }
    Ok(out)
}

fn hex_val(c: u8) -> core::result::Result<u8, InvalidHex> {
    match c {
        b'0'..=b'9' => Ok(c - b'0'),
        b'a'..=b'f' => Ok(c - b'a' + 10),
        b'A'..=b'F' => Ok(c - b'A' + 10),
        _ => Err(InvalidHex),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn encode_empty() {
        assert_eq!(encode(&[]), "");
    }

    #[test]
    fn encode_one_byte() {
        assert_eq!(encode(&[0x00]), "00");
        assert_eq!(encode(&[0xff]), "ff");
        assert_eq!(encode(&[0xab]), "ab");
    }

    #[test]
    fn encode_multibyte() {
        assert_eq!(encode(&[0xde, 0xad, 0xbe, 0xef]), "deadbeef");
    }

    #[test]
    fn round_trip_random_bytes() {
        let inputs: &[&[u8]] = &[
            &[],
            &[0],
            &[255],
            b"hello",
            &[0xde, 0xad, 0xbe, 0xef, 0x00, 0x10, 0x20],
        ];
        for input in inputs {
            let encoded = encode(input);
            let decoded = decode(&encoded).unwrap();
            assert_eq!(decoded.as_slice(), *input);
        }
    }

    #[test]
    fn decode_uppercase() {
        assert_eq!(decode("DEADBEEF").unwrap(), vec![0xde, 0xad, 0xbe, 0xef]);
    }

    #[test]
    fn decode_odd_length_fails() {
        assert!(decode("abc").is_err());
    }

    #[test]
    fn decode_invalid_char_fails() {
        assert!(decode("zz").is_err());
        assert!(decode("ag").is_err());
    }
}
