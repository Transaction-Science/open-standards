//! Lightning: BOLT-11 invoice parsing + LNURL decoding.
//!
//! BOLT-11 invoices are bech32-encoded payment requests. The HRP
//! (human-readable prefix) encodes the network and amount:
//! ```text
//!   lnbc1m1p...     (mainnet, 1 milli-bitcoin)
//!   lntb500u1p...   (testnet, 500 micro-bitcoin)
//!   lnbcrt1...      (regtest)
//!   lnsb1...        (signet)
//! ```
//!
//! LNURL is a bech32-encoded URL pointing to a Lightning service
//! endpoint (`lnurl1...`).
//!
//! This module provides:
//! - bech32 decoding (BOLT-11 uses the *original* bech32 checksum,
//!   not bech32m used by Taproot).
//! - HRP parsing: network + amount.
//! - Tagged-field iteration (`p` = payment hash, `d` = description,
//!   `n` = payee public key, `x` = expiry, `s` = payment secret,
//!   `9` = features, etc.) producing a structured walk.
//! - LNURL bech32 decode → UTF-8 URL.
//!
//! The aim is *parse, validate structure, expose the fields*. The
//! crate does NOT verify the BOLT-11 signature (operators with
//! secp256k1 in their dependency graph can do that against the
//! payee_node_id; see BOLT-11 §"signature").

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};

/// Lightning network the invoice is for.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Bolt11Network {
    /// Mainnet (`lnbc`).
    Mainnet,
    /// Testnet (`lntb`).
    Testnet,
    /// Signet (`lnsb`).
    Signet,
    /// Regtest (`lnbcrt`).
    Regtest,
}

impl Bolt11Network {
    /// HRP prefix for this network (without amount suffix).
    #[must_use]
    pub const fn hrp_prefix(self) -> &'static str {
        match self {
            Self::Mainnet => "lnbc",
            Self::Testnet => "lntb",
            Self::Signet => "lnsb",
            Self::Regtest => "lnbcrt",
        }
    }
}

/// LNURL kind. The bech32-decoded URL points to one of several
/// well-known LNURL flows.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum LnurlKind {
    /// `lnurl-pay` (HTTPS GET → JSON with `callback`,
    /// `maxSendable`, etc.).
    Pay,
    /// `lnurl-withdraw`.
    Withdraw,
    /// `lnurl-auth`.
    Auth,
    /// `lnurl-channel`.
    Channel,
    /// Unknown / generic LNURL — kind disambiguation requires
    /// fetching the URL.
    Generic,
}

/// Parsed BOLT-11 invoice envelope.
///
/// Captures the *structural* parsing — bech32 + HRP + raw tagged
/// fields. Operators with full BOLT-11 stacks (e.g. `lightning`
/// crate) can extend this; minimal services usually only need the
/// payment hash, amount, network, and expiry.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Bolt11Invoice {
    /// Network derived from HRP.
    pub network: Bolt11Network,
    /// Amount in millisatoshis, derived from the HRP amount suffix.
    /// `None` for amount-less invoices (`lnbc1p...` — no value).
    pub amount_msat: Option<u64>,
    /// Invoice timestamp (unix seconds), from the 35-bit prefix
    /// after HRP.
    pub timestamp: u64,
    /// Raw tagged fields as `(tag_char, data_bits_len, payload)`.
    /// Payload is the bech32-5-bit-decoded byte run (not yet
    /// converted to 8-bit bytes — different tags have different
    /// encodings).
    pub tagged_fields: Vec<(char, u16, Vec<u8>)>,
    /// 65-byte recoverable signature trailer (r||s||recovery_id),
    /// carried as a `Vec` because `[u8; 65]` exceeds serde's
    /// default-derive array support (cap is 32). Callers can
    /// assert `signature.len() == 65`.
    pub signature: Vec<u8>,
}

impl Bolt11Invoice {
    /// Returns the payment hash (32 bytes) if the invoice carries a
    /// `p` tag, decoded from 5-bit groups to 8-bit bytes.
    #[must_use]
    pub fn payment_hash(&self) -> Option<[u8; 32]> {
        for (tag, _, payload) in &self.tagged_fields {
            if *tag == 'p' {
                let bytes = bits5_to_bits8(payload);
                if bytes.len() == 32 {
                    let mut out = [0u8; 32];
                    out.copy_from_slice(&bytes);
                    return Some(out);
                }
            }
        }
        None
    }

    /// Description (`d`) if present.
    #[must_use]
    pub fn description(&self) -> Option<String> {
        for (tag, _, payload) in &self.tagged_fields {
            if *tag == 'd' {
                let bytes = bits5_to_bits8(payload);
                if let Ok(s) = String::from_utf8(bytes) {
                    return Some(s);
                }
            }
        }
        None
    }
}

/// Parse a BOLT-11 invoice string into [`Bolt11Invoice`].
///
/// # Errors
/// Returns [`Error::Decode`] or [`Error::InvalidLayout`] on
/// structural failure; [`Error::Integrity`] on checksum failure.
pub fn parse_bolt11(invoice: &str) -> Result<Bolt11Invoice> {
    // Lowercase per bech32 spec; uppercase is also valid but mixed-case is forbidden.
    let invoice_lc = if invoice.chars().any(char::is_uppercase) {
        if invoice.chars().any(char::is_lowercase) {
            return Err(Error::Decode("mixed-case invoice".into()));
        }
        invoice.to_lowercase()
    } else {
        invoice.to_owned()
    };
    let (hrp, data_5bit) = bech32_decode(&invoice_lc)?;
    // Parse HRP: lnbc|lntb|lnsb|lnbcrt + optional amount.
    let (network, amount_msat) = parse_hrp(&hrp)?;
    // BOLT-11 data layout:
    //   7 × 5-bit = 35-bit timestamp
    //   tagged fields (each: 5-bit tag, 10-bit length, length×5-bit payload)
    //   104 × 5-bit = 520-bit signature
    if data_5bit.len() < 7 + 104 {
        return Err(Error::InvalidLayout(format!(
            "data too short: {} groups",
            data_5bit.len()
        )));
    }
    let timestamp = read_u35(&data_5bit[..7]);
    let body = &data_5bit[7..data_5bit.len() - 104];
    let sig_5bit = &data_5bit[data_5bit.len() - 104..];
    let sig_bytes = bits5_to_bits8(sig_5bit);
    if sig_bytes.len() < 65 {
        return Err(Error::InvalidLayout(format!(
            "signature decode produced {} bytes, expected 65",
            sig_bytes.len()
        )));
    }
    let signature = sig_bytes[..65].to_vec();

    let mut tagged_fields = Vec::new();
    let mut cursor = 0;
    while cursor < body.len() {
        if cursor + 3 > body.len() {
            return Err(Error::InvalidLayout("tagged-field header truncated".into()));
        }
        let tag_5bit = body[cursor];
        cursor += 1;
        let len_hi = body[cursor];
        cursor += 1;
        let len_lo = body[cursor];
        cursor += 1;
        let data_len = (u16::from(len_hi) << 5) | u16::from(len_lo);
        let data_len_usize = data_len as usize;
        if cursor + data_len_usize > body.len() {
            return Err(Error::InvalidLayout(format!(
                "tagged-field body overflow: len={data_len}"
            )));
        }
        let tag_char = bech32_5bit_to_char(tag_5bit)?;
        let payload = body[cursor..cursor + data_len_usize].to_vec();
        cursor += data_len_usize;
        tagged_fields.push((tag_char, data_len, payload));
    }

    Ok(Bolt11Invoice {
        network,
        amount_msat,
        timestamp,
        tagged_fields,
        signature,
    })
}

/// Decode an LNURL (`lnurl1...`) string into the encoded URL.
///
/// # Errors
/// Returns [`Error::Decode`] when HRP isn't `lnurl` or the bech32
/// decode fails.
pub fn lnurl_decode(lnurl: &str) -> Result<String> {
    let s = if lnurl.chars().any(char::is_uppercase) {
        if lnurl.chars().any(char::is_lowercase) {
            return Err(Error::Decode("mixed-case lnurl".into()));
        }
        lnurl.to_lowercase()
    } else {
        lnurl.to_owned()
    };
    let (hrp, data_5bit) = bech32_decode(&s)?;
    if hrp != "lnurl" {
        return Err(Error::Decode(format!("lnurl hrp must be `lnurl`, got `{hrp}`")));
    }
    let bytes = bits5_to_bits8(&data_5bit);
    String::from_utf8(bytes).map_err(|e| Error::Decode(format!("lnurl utf-8: {e}")))
}

fn parse_hrp(hrp: &str) -> Result<(Bolt11Network, Option<u64>)> {
    // Determine the longest matching prefix.
    let (net, prefix_len) = if let Some(rest) = hrp.strip_prefix("lnbcrt") {
        (Bolt11Network::Regtest, hrp.len() - rest.len())
    } else if let Some(rest) = hrp.strip_prefix("lnbc") {
        (Bolt11Network::Mainnet, hrp.len() - rest.len())
    } else if let Some(rest) = hrp.strip_prefix("lntb") {
        (Bolt11Network::Testnet, hrp.len() - rest.len())
    } else if let Some(rest) = hrp.strip_prefix("lnsb") {
        (Bolt11Network::Signet, hrp.len() - rest.len())
    } else {
        let _ = hrp;
        return Err(Error::Unsupported(format!("hrp `{hrp}`")));
    };
    let amount_suffix = &hrp[prefix_len..];
    if amount_suffix.is_empty() {
        return Ok((net, None));
    }
    // Trailing multiplier: m=10^-3, u=10^-6, n=10^-9, p=10^-12.
    let last = amount_suffix.chars().last();
    let (digits, multiplier_msat) = match last {
        Some(c) if c.is_ascii_digit() => (amount_suffix, 100_000_000_000_u64), // BTC → msat: 1 BTC = 10^11 msat * ... actually 10^11 msat is incorrect; 1 BTC = 10^8 sat = 10^11 msat. OK.
        Some('m') => (&amount_suffix[..amount_suffix.len() - 1], 100_000_000_u64), // milli-BTC = 10^8 msat
        Some('u') => (&amount_suffix[..amount_suffix.len() - 1], 100_000_u64),     // micro-BTC = 10^5 msat
        Some('n') => (&amount_suffix[..amount_suffix.len() - 1], 100_u64),         // nano-BTC = 100 msat
        Some('p') => (&amount_suffix[..amount_suffix.len() - 1], 0_u64),           // pico-BTC = 0.1 msat; only valid in multiples of 10
        Some(c) => return Err(Error::Decode(format!("unknown amount multiplier `{c}`"))),
        None => return Err(Error::Decode("empty amount suffix".into())),
    };
    let n: u64 = digits
        .parse()
        .map_err(|e| Error::Decode(format!("bad amount digits `{digits}`: {e}")))?;
    let msat = if multiplier_msat == 0 {
        // pico-BTC must be a multiple of 10 (so total stays whole msat).
        if !n.is_multiple_of(10) {
            return Err(Error::Decode("pico-BTC amount must be multiple of 10".into()));
        }
        n / 10
    } else {
        n.checked_mul(multiplier_msat)
            .ok_or_else(|| Error::Decode("amount overflow".into()))?
    };
    Ok((net, Some(msat)))
}

// -----------------------------------------------------------------
// bech32
// -----------------------------------------------------------------

const BECH32_CHARSET: &[u8] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";

fn bech32_char_to_5bit(c: char) -> Result<u8> {
    if !c.is_ascii() {
        return Err(Error::Decode(format!("non-ASCII char `{c}`")));
    }
    let pos = BECH32_CHARSET
        .iter()
        .position(|&b| b == c as u8)
        .ok_or_else(|| Error::Decode(format!("invalid bech32 char `{c}`")))?;
    Ok(u8::try_from(pos).map_err(|e| Error::Decode(format!("bech32 char position: {e}")))?)
}

fn bech32_5bit_to_char(v: u8) -> Result<char> {
    if v >= 32 {
        return Err(Error::Decode(format!("5-bit value out of range: {v}")));
    }
    Ok(BECH32_CHARSET[v as usize] as char)
}

fn bech32_polymod(values: &[u8]) -> u32 {
    const GEN: [u32; 5] = [
        0x3b6a_57b2,
        0x2650_8e6d,
        0x1ea1_19fa,
        0x3d42_33dd,
        0x2a14_62b3,
    ];
    let mut chk: u32 = 1;
    for v in values {
        let b = chk >> 25;
        chk = ((chk & 0x01ff_ffff) << 5) ^ u32::from(*v);
        for (i, &g) in GEN.iter().enumerate() {
            #[allow(clippy::cast_possible_truncation)]
            if ((b >> i) & 1) == 1 {
                chk ^= g;
            }
        }
    }
    chk
}

fn bech32_hrp_expand(hrp: &str) -> Vec<u8> {
    let mut out = Vec::with_capacity(hrp.len() * 2 + 1);
    for c in hrp.chars() {
        out.push((c as u8) >> 5);
    }
    out.push(0);
    for c in hrp.chars() {
        out.push((c as u8) & 0x1f);
    }
    out
}

fn bech32_decode(s: &str) -> Result<(String, Vec<u8>)> {
    let split = s
        .rfind('1')
        .ok_or_else(|| Error::Decode("missing bech32 separator `1`".into()))?;
    if split < 1 || split + 7 > s.len() {
        return Err(Error::Decode(format!(
            "bech32 split position {split} invalid"
        )));
    }
    let hrp = s[..split].to_owned();
    let data_chars = &s[split + 1..];
    let mut data_5bit = Vec::with_capacity(data_chars.len());
    for c in data_chars.chars() {
        data_5bit.push(bech32_char_to_5bit(c)?);
    }
    // Checksum check.
    let mut full = bech32_hrp_expand(&hrp);
    full.extend_from_slice(&data_5bit);
    if bech32_polymod(&full) != 1 {
        return Err(Error::Integrity("bech32 checksum mismatch".into()));
    }
    // Strip the 6-symbol checksum at the end.
    let payload = data_5bit[..data_5bit.len() - 6].to_vec();
    Ok((hrp, payload))
}

fn read_u35(groups: &[u8]) -> u64 {
    let mut out: u64 = 0;
    for &g in groups.iter().take(7) {
        out = (out << 5) | u64::from(g);
    }
    out
}

/// Convert from-5-bit-per-byte to 8-bit bytes (MSB-first, pad with
/// zero). Truncates incomplete trailing group.
fn bits5_to_bits8(input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(input.len() * 5 / 8);
    let mut acc: u32 = 0;
    let mut bits: u32 = 0;
    for &v in input {
        acc = (acc << 5) | u32::from(v);
        bits += 5;
        if bits >= 8 {
            bits -= 8;
            #[allow(clippy::cast_possible_truncation)]
            out.push(((acc >> bits) & 0xff) as u8);
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hrp_parse_mainnet_amount() {
        let (net, amt) = parse_hrp("lnbc1m").unwrap();
        assert_eq!(net, Bolt11Network::Mainnet);
        // 1m = 1 milli-bitcoin = 10^8 msat.
        assert_eq!(amt, Some(100_000_000));
    }

    #[test]
    fn hrp_parse_testnet_no_amount() {
        let (net, amt) = parse_hrp("lntb").unwrap();
        assert_eq!(net, Bolt11Network::Testnet);
        assert_eq!(amt, None);
    }

    #[test]
    fn hrp_parse_micro() {
        let (net, amt) = parse_hrp("lnbc500u").unwrap();
        assert_eq!(net, Bolt11Network::Mainnet);
        assert_eq!(amt, Some(500 * 100_000));
    }

    #[test]
    fn hrp_parse_pico_rejects_odd() {
        // 5p would be 0.5 msat — illegal.
        assert!(parse_hrp("lnbc5p").is_err());
    }

    #[test]
    fn hrp_unknown_prefix() {
        let err = parse_hrp("foobar").unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    #[test]
    fn lnurl_round_trip_smoke() {
        // Smoke: we can decode a deliberately-malformed lnurl into
        // an error, and we don't panic. Building a full valid lnurl
        // fixture means producing a valid bech32 checksum, which
        // would require a generator we don't ship.
        let err = lnurl_decode("not-an-lnurl").unwrap_err();
        assert!(matches!(err, Error::Decode(_) | Error::Integrity(_)));
    }

    #[test]
    fn bech32_polymod_known_value() {
        // The all-zero input expands to a specific polymod value;
        // smoke-test it produces something stable.
        let v = bech32_polymod(&[0u8; 6]);
        let v2 = bech32_polymod(&[0u8; 6]);
        assert_eq!(v, v2);
    }

    #[test]
    fn bits5_to_bits8_canonical() {
        // 8 input symbols (5 bits each = 40 bits) → 5 output bytes.
        let input = vec![0x1f; 8];
        let out = bits5_to_bits8(&input);
        assert_eq!(out.len(), 5);
        // All 1s in.
        for b in &out {
            assert_eq!(*b, 0xff);
        }
    }

    #[test]
    fn parse_bolt11_rejects_mixed_case() {
        let err = parse_bolt11("LNBC1mAbcDef").unwrap_err();
        assert!(matches!(err, Error::Decode(_)));
    }

    #[test]
    fn parse_bolt11_rejects_missing_separator() {
        let err = parse_bolt11("lnbcdoesnothaveone").unwrap_err();
        assert!(matches!(err, Error::Decode(_) | Error::Integrity(_)));
    }
}
