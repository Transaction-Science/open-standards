//! Wire-format codecs used by ISO 8583 data elements.
//!
//! ISO 8583 is a "kitchen sink" of encodings — each network picks a
//! subset and applies it inconsistently across data elements. The five
//! that cover essentially every production deployment:
//!
//! - **BCD (binary-coded decimal)** — two decimal digits packed per byte,
//!   high nibble first. `1234` → `[0x12, 0x34]`. Used for numerics on
//!   Visa Base I, Mastercard MDS, and most legacy mainframe rails.
//! - **ASCII** — one byte per digit. `1234` → `[0x31, 0x32, 0x33, 0x34]`.
//!   Common on JCB and some Discover variants.
//! - **EBCDIC** — IBM mainframe encoding. Still seen on legacy Amex GNS
//!   front-ends. We implement the IBM-037 subset (the only one Amex
//!   GNS uses for ISO 8583 message bodies).
//! - **Binary** — opaque byte buffer (DE 52 PIN block, DE 55 EMV TLV,
//!   DE 64 MAC, bitmap itself).
//! - **LL / LLL variable-length** — a 2- or 3-digit length prefix
//!   (BCD or ASCII per dialect) followed by that many bytes of payload.
//!   `LLVAR("ABC")` BCD-prefix → `[0x00, 0x03, 'A', 'B', 'C']` (in
//!   practice the prefix is two BCD digits: `[0x03, 'A', 'B', 'C']`).
//!
//! Every function in this module is total: it returns a `Result` and
//! never panics, never allocates beyond the documented case.

use crate::error::{Error, Result};

// ---------- BCD ----------

/// Encode an ASCII decimal string into packed BCD.
///
/// Odd-length strings are left-padded with a `0` nibble so the high
/// nibble of the first output byte is the pad. This matches ISO 8583
/// "n" right-justified semantics for numeric data elements.
///
/// # Errors
/// Returns [`Error::InvalidAscii`] if any character is not `0..=9`.
pub fn bcd_encode(digits: &str) -> Result<Vec<u8>> {
    for (i, b) in digits.bytes().enumerate() {
        if !b.is_ascii_digit() {
            return Err(Error::InvalidAscii { byte: b, offset: i });
        }
    }
    // Left-pad odd lengths with a `0` so the resulting byte stream is
    // right-justified, ISO 8583 "n" semantics.
    let normalized: String = if digits.len() % 2 == 1 {
        let mut s = String::with_capacity(digits.len() + 1);
        s.push('0');
        s.push_str(digits);
        s
    } else {
        digits.to_owned()
    };
    let bytes = normalized.as_bytes();
    let mut out = Vec::with_capacity(bytes.len() / 2);
    let mut i = 0;
    while i + 1 < bytes.len() {
        let hi = bytes[i] - b'0';
        let lo = bytes[i + 1] - b'0';
        out.push((hi << 4) | lo);
        i += 2;
    }
    Ok(out)
}

/// Decode `n_digits` decimal digits from packed BCD.
///
/// Consumes `n_digits.div_ceil(2)` bytes starting at `offset`. Returns
/// the decoded string plus the new cursor.
///
/// # Errors
/// [`Error::UnexpectedEof`] if the input is too short.
/// [`Error::InvalidBcd`] if any nibble is `> 0x9`.
pub fn bcd_decode(input: &[u8], offset: usize, n_digits: usize) -> Result<(String, usize)> {
    let n_bytes = n_digits.div_ceil(2);
    if offset + n_bytes > input.len() {
        return Err(Error::UnexpectedEof {
            offset,
            needed: n_bytes,
        });
    }
    let mut out = String::with_capacity(n_digits);
    let start_skip = n_bytes * 2 - n_digits; // 0 or 1 leading pad nibble
    for (idx, byte) in input[offset..offset + n_bytes].iter().enumerate() {
        let hi = byte >> 4;
        let lo = byte & 0x0F;
        if hi > 9 {
            return Err(Error::InvalidBcd {
                nibble: hi,
                offset: offset + idx,
            });
        }
        if lo > 9 {
            return Err(Error::InvalidBcd {
                nibble: lo,
                offset: offset + idx,
            });
        }
        // Skip the leading pad nibble on the very first byte if any.
        if idx == 0 && start_skip == 1 {
            out.push(char::from(b'0' + lo));
        } else {
            out.push(char::from(b'0' + hi));
            out.push(char::from(b'0' + lo));
        }
    }
    Ok((out, offset + n_bytes))
}

// ---------- ASCII ----------

/// Encode an ASCII numeric or alphanumeric string as raw ASCII bytes.
///
/// # Errors
/// [`Error::InvalidAscii`] if any byte falls outside `0x20..=0x7E`.
pub fn ascii_encode(s: &str) -> Result<Vec<u8>> {
    for (i, b) in s.bytes().enumerate() {
        if !(0x20..=0x7E).contains(&b) {
            return Err(Error::InvalidAscii { byte: b, offset: i });
        }
    }
    Ok(s.as_bytes().to_vec())
}

/// Decode `n` bytes of ASCII starting at `offset`.
///
/// # Errors
/// [`Error::UnexpectedEof`] if input too short.
/// [`Error::InvalidAscii`] if any byte is outside printable ASCII.
pub fn ascii_decode(input: &[u8], offset: usize, n: usize) -> Result<(String, usize)> {
    if offset + n > input.len() {
        return Err(Error::UnexpectedEof { offset, needed: n });
    }
    let mut out = String::with_capacity(n);
    for (i, b) in input[offset..offset + n].iter().enumerate() {
        if !(0x20..=0x7E).contains(b) {
            return Err(Error::InvalidAscii {
                byte: *b,
                offset: offset + i,
            });
        }
        out.push(char::from(*b));
    }
    Ok((out, offset + n))
}

// ---------- EBCDIC (IBM-037, the only subset Amex GNS / legacy
//                   mainframe ISO 8583 stacks emit) ----------

const ASCII_TO_EBCDIC_037: [u8; 128] = {
    let mut t = [0x6F_u8; 128]; // '?' in EBCDIC for anything we don't map
    // Digits 0..=9
    t[b'0' as usize] = 0xF0;
    t[b'1' as usize] = 0xF1;
    t[b'2' as usize] = 0xF2;
    t[b'3' as usize] = 0xF3;
    t[b'4' as usize] = 0xF4;
    t[b'5' as usize] = 0xF5;
    t[b'6' as usize] = 0xF6;
    t[b'7' as usize] = 0xF7;
    t[b'8' as usize] = 0xF8;
    t[b'9' as usize] = 0xF9;
    // Uppercase letters
    t[b'A' as usize] = 0xC1;
    t[b'B' as usize] = 0xC2;
    t[b'C' as usize] = 0xC3;
    t[b'D' as usize] = 0xC4;
    t[b'E' as usize] = 0xC5;
    t[b'F' as usize] = 0xC6;
    t[b'G' as usize] = 0xC7;
    t[b'H' as usize] = 0xC8;
    t[b'I' as usize] = 0xC9;
    t[b'J' as usize] = 0xD1;
    t[b'K' as usize] = 0xD2;
    t[b'L' as usize] = 0xD3;
    t[b'M' as usize] = 0xD4;
    t[b'N' as usize] = 0xD5;
    t[b'O' as usize] = 0xD6;
    t[b'P' as usize] = 0xD7;
    t[b'Q' as usize] = 0xD8;
    t[b'R' as usize] = 0xD9;
    t[b'S' as usize] = 0xE2;
    t[b'T' as usize] = 0xE3;
    t[b'U' as usize] = 0xE4;
    t[b'V' as usize] = 0xE5;
    t[b'W' as usize] = 0xE6;
    t[b'X' as usize] = 0xE7;
    t[b'Y' as usize] = 0xE8;
    t[b'Z' as usize] = 0xE9;
    // Lowercase letters
    t[b'a' as usize] = 0x81;
    t[b'b' as usize] = 0x82;
    t[b'c' as usize] = 0x83;
    t[b'd' as usize] = 0x84;
    t[b'e' as usize] = 0x85;
    t[b'f' as usize] = 0x86;
    t[b'g' as usize] = 0x87;
    t[b'h' as usize] = 0x88;
    t[b'i' as usize] = 0x89;
    t[b'j' as usize] = 0x91;
    t[b'k' as usize] = 0x92;
    t[b'l' as usize] = 0x93;
    t[b'm' as usize] = 0x94;
    t[b'n' as usize] = 0x95;
    t[b'o' as usize] = 0x96;
    t[b'p' as usize] = 0x97;
    t[b'q' as usize] = 0x98;
    t[b'r' as usize] = 0x99;
    t[b's' as usize] = 0xA2;
    t[b't' as usize] = 0xA3;
    t[b'u' as usize] = 0xA4;
    t[b'v' as usize] = 0xA5;
    t[b'w' as usize] = 0xA6;
    t[b'x' as usize] = 0xA7;
    t[b'y' as usize] = 0xA8;
    t[b'z' as usize] = 0xA9;
    // Common punctuation used in DE 43 merchant name/location lines.
    t[b' ' as usize] = 0x40;
    t[b'.' as usize] = 0x4B;
    t[b',' as usize] = 0x6B;
    t[b'-' as usize] = 0x60;
    t[b'/' as usize] = 0x61;
    t[b'*' as usize] = 0x5C;
    t[b'(' as usize] = 0x4D;
    t[b')' as usize] = 0x5D;
    t[b'\'' as usize] = 0x7D;
    t[b'"' as usize] = 0x7F;
    t[b'&' as usize] = 0x50;
    t[b'+' as usize] = 0x4E;
    t[b':' as usize] = 0x7A;
    t[b';' as usize] = 0x5E;
    t[b'=' as usize] = 0x7E;
    t[b'?' as usize] = 0x6F;
    t[b'#' as usize] = 0x7B;
    t[b'@' as usize] = 0x7C;
    t
};

/// Encode an ASCII string into IBM-037 EBCDIC.
///
/// # Errors
/// [`Error::InvalidAscii`] if any byte is outside 7-bit ASCII.
pub fn ebcdic_encode(s: &str) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(s.len());
    for (i, b) in s.bytes().enumerate() {
        if b > 0x7F {
            return Err(Error::InvalidAscii { byte: b, offset: i });
        }
        out.push(ASCII_TO_EBCDIC_037[b as usize]);
    }
    Ok(out)
}

/// Decode `n` IBM-037 EBCDIC bytes into ASCII.
///
/// Characters that have no IBM-037 → ASCII mapping decode to `?`. This
/// is intentional: ISO 8583 EBCDIC payloads in the Amex GNS subset never
/// contain unmappable bytes; if they do, the message is malformed
/// upstream and we don't want a parse failure to hide that.
///
/// # Errors
/// [`Error::UnexpectedEof`] if input too short.
pub fn ebcdic_decode(input: &[u8], offset: usize, n: usize) -> Result<(String, usize)> {
    if offset + n > input.len() {
        return Err(Error::UnexpectedEof { offset, needed: n });
    }
    let mut out = String::with_capacity(n);
    for byte in &input[offset..offset + n] {
        out.push(ebcdic_to_ascii(*byte));
    }
    Ok((out, offset + n))
}

fn ebcdic_to_ascii(byte: u8) -> char {
    // Reverse mapping of the table above. Linear search is fine for a
    // 256-byte alphabet on a payment-ops fast path.
    for (i, ebc) in ASCII_TO_EBCDIC_037.iter().enumerate().take(128) {
        if *ebc == byte {
            return char::from_u32(i as u32).unwrap_or('?');
        }
    }
    '?'
}

// ---------- Binary (opaque) ----------

/// Copy `n` raw bytes out of `input` starting at `offset`.
///
/// # Errors
/// [`Error::UnexpectedEof`] if `offset + n > input.len()`.
pub fn binary_decode(input: &[u8], offset: usize, n: usize) -> Result<(Vec<u8>, usize)> {
    if offset + n > input.len() {
        return Err(Error::UnexpectedEof { offset, needed: n });
    }
    Ok((input[offset..offset + n].to_vec(), offset + n))
}

// ---------- LLVAR / LLLVAR ----------

/// Length-prefix encoding for variable data elements.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum VarLenKind {
    /// Two-digit length prefix in BCD (max 99 bytes/digits of payload).
    LlBcd,
    /// Three-digit length prefix in BCD (max 999).
    LllBcd,
    /// Two-digit length prefix in ASCII (max 99).
    LlAscii,
    /// Three-digit length prefix in ASCII (max 999).
    LllAscii,
}

impl VarLenKind {
    const fn max(self) -> usize {
        match self {
            Self::LlBcd | Self::LlAscii => 99,
            Self::LllBcd | Self::LllAscii => 999,
        }
    }
}

/// Encode an LLVAR or LLLVAR length-prefixed field.
///
/// The payload is written verbatim after the length prefix; callers
/// have already encoded the payload bytes (ASCII, BCD, binary, ...)
/// however the data element requires.
///
/// # Errors
/// [`Error::InvalidVarLength`] if the payload exceeds the prefix's
/// maximum encodable length.
pub fn var_encode(kind: VarLenKind, payload: &[u8]) -> Result<Vec<u8>> {
    if payload.len() > kind.max() {
        return Err(Error::InvalidVarLength {
            prefix: payload.len(),
            max: kind.max(),
        });
    }
    let mut out = Vec::new();
    match kind {
        VarLenKind::LlBcd => {
            let s = format!("{:02}", payload.len());
            out.extend(bcd_encode(&s)?);
        }
        VarLenKind::LllBcd => {
            let s = format!("{:03}", payload.len());
            out.extend(bcd_encode(&s)?);
        }
        VarLenKind::LlAscii => {
            out.extend(format!("{:02}", payload.len()).into_bytes());
        }
        VarLenKind::LllAscii => {
            out.extend(format!("{:03}", payload.len()).into_bytes());
        }
    }
    out.extend_from_slice(payload);
    Ok(out)
}

/// Decode an LL/LLL prefix and return `(payload, new_cursor)`.
///
/// The payload is returned as the raw bytes following the prefix;
/// the caller is responsible for further decoding (BCD digits, ASCII
/// text, binary blob).
///
/// # Errors
/// [`Error::UnexpectedEof`] if input is too short.
/// [`Error::InvalidVarLength`] if the prefix is outside its kind's range.
pub fn var_decode(
    kind: VarLenKind,
    input: &[u8],
    offset: usize,
) -> Result<(Vec<u8>, usize)> {
    let (len, new_offset) = match kind {
        VarLenKind::LlBcd => {
            let (s, o) = bcd_decode(input, offset, 2)?;
            (
                s.parse::<usize>()
                    .map_err(|_| Error::InvalidVarLength { prefix: 0, max: 99 })?,
                o,
            )
        }
        VarLenKind::LllBcd => {
            let (s, o) = bcd_decode(input, offset, 3)?;
            (
                s.parse::<usize>()
                    .map_err(|_| Error::InvalidVarLength { prefix: 0, max: 999 })?,
                o,
            )
        }
        VarLenKind::LlAscii => {
            let (s, o) = ascii_decode(input, offset, 2)?;
            (
                s.parse::<usize>()
                    .map_err(|_| Error::InvalidVarLength { prefix: 0, max: 99 })?,
                o,
            )
        }
        VarLenKind::LllAscii => {
            let (s, o) = ascii_decode(input, offset, 3)?;
            (
                s.parse::<usize>()
                    .map_err(|_| Error::InvalidVarLength { prefix: 0, max: 999 })?,
                o,
            )
        }
    };
    if len > kind.max() {
        return Err(Error::InvalidVarLength {
            prefix: len,
            max: kind.max(),
        });
    }
    if new_offset + len > input.len() {
        return Err(Error::UnexpectedEof {
            offset: new_offset,
            needed: len,
        });
    }
    Ok((input[new_offset..new_offset + len].to_vec(), new_offset + len))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bcd_encode_even_length() {
        let out = bcd_encode("1234").unwrap();
        assert_eq!(out, vec![0x12, 0x34]);
    }

    #[test]
    fn bcd_encode_odd_length_left_pads() {
        let out = bcd_encode("123").unwrap();
        // "0123" -> [0x01, 0x23]
        assert_eq!(out, vec![0x01, 0x23]);
    }

    #[test]
    fn bcd_round_trip() {
        let cases = ["0", "9", "01", "99", "1234567890", "000000000100"];
        for c in cases {
            let n_digits = c.len();
            let enc = bcd_encode(c).unwrap();
            let (dec, off) = bcd_decode(&enc, 0, n_digits).unwrap();
            assert_eq!(dec, c, "round trip {c}");
            assert_eq!(off, enc.len());
        }
    }

    #[test]
    fn bcd_decode_rejects_non_decimal_nibble() {
        // 0xAB is high nibble = A (>9), low nibble = B (>9).
        let err = bcd_decode(&[0xAB], 0, 2).unwrap_err();
        assert!(matches!(err, Error::InvalidBcd { .. }));
    }

    #[test]
    fn bcd_decode_eof() {
        let err = bcd_decode(&[0x12], 0, 4).unwrap_err();
        assert!(matches!(err, Error::UnexpectedEof { .. }));
    }

    #[test]
    fn ascii_round_trip() {
        let s = "MERCHANT NAME";
        let enc = ascii_encode(s).unwrap();
        let (dec, _) = ascii_decode(&enc, 0, s.len()).unwrap();
        assert_eq!(dec, s);
    }

    #[test]
    fn ascii_rejects_control_chars() {
        let err = ascii_encode("\x01").unwrap_err();
        assert!(matches!(err, Error::InvalidAscii { .. }));
    }

    #[test]
    fn llvar_encode_abc() {
        // "ABC" with LL ASCII prefix -> "03ABC" -> 0x30 0x33 'A' 'B' 'C'
        let payload = b"ABC";
        let enc = var_encode(VarLenKind::LlAscii, payload).unwrap();
        assert_eq!(enc, vec![b'0', b'3', b'A', b'B', b'C']);
    }

    #[test]
    fn llvar_round_trip_bcd_prefix() {
        let payload = b"ABC";
        let enc = var_encode(VarLenKind::LlBcd, payload).unwrap();
        // BCD prefix "03" -> 0x03
        assert_eq!(enc, vec![0x03, b'A', b'B', b'C']);
        let (dec, off) = var_decode(VarLenKind::LlBcd, &enc, 0).unwrap();
        assert_eq!(dec, payload);
        assert_eq!(off, enc.len());
    }

    #[test]
    fn lllvar_encode_long_payload() {
        let payload = vec![0xAA_u8; 256];
        let enc = var_encode(VarLenKind::LllAscii, &payload).unwrap();
        assert_eq!(&enc[..3], b"256");
        assert_eq!(&enc[3..], &payload[..]);
    }

    #[test]
    fn ebcdic_round_trip_digits_and_letters() {
        for s in ["123", "ABC", "merchant", "AMEX 1234"] {
            let enc = ebcdic_encode(s).unwrap();
            let (dec, _) = ebcdic_decode(&enc, 0, enc.len()).unwrap();
            assert_eq!(dec, s, "ebcdic round trip {s}");
        }
    }

    #[test]
    fn binary_decode_eof() {
        let err = binary_decode(&[0; 4], 0, 5).unwrap_err();
        assert!(matches!(err, Error::UnexpectedEof { .. }));
    }
}
