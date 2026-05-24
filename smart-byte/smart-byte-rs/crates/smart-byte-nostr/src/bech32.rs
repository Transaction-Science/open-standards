//! NIP-19 bech32 encodings: `npub`, `nsec`, `note`, `nprofile`, `nevent`,
//! `naddr`, `nrelay`.
//!
//! Implementation is a from-scratch BIP-173 bech32 encoder/decoder plus
//! the NIP-19 TLV (`type:length:value`) payloads. We support strings up
//! to a fairly generous size — relays publish their list metadata via
//! plain events, so the encoded length budget is well within bech32's
//! standard 90-character limit for the bare `npub`/`nsec`/`note` forms,
//! and intentionally exceeds it for TLV forms (`nprofile`, `nevent`,
//! `naddr`) per NIP-19. We therefore do not enforce the 90-char limit;
//! NIP-19 explicitly waives it for TLV forms.

use crate::error::NostrError;

const CHARSET: &[u8; 32] = b"qpzry9x8gf2tvdw0s3jn54khce6mua7l";

const GEN: [u32; 5] = [
    0x3b6a_57b2, 0x2650_8e6d, 0x1ea1_19fa, 0x3d42_33dd, 0x2a14_62b3,
];

fn charset_rev(c: u8) -> Option<u8> {
    // Find the index of `c` in CHARSET, or None.
    CHARSET.iter().position(|&x| x == c).map(|i| i as u8)
}

fn polymod(values: &[u8]) -> u32 {
    let mut chk: u32 = 1;
    for &v in values {
        let top = (chk >> 25) as u8;
        chk = ((chk & 0x1ff_ffff) << 5) ^ (v as u32);
        for (i, g) in GEN.iter().enumerate() {
            if (top >> i) & 1 == 1 {
                chk ^= *g;
            }
        }
    }
    chk
}

fn hrp_expand(hrp: &str) -> Vec<u8> {
    let mut v = Vec::with_capacity(hrp.len() * 2 + 1);
    for c in hrp.bytes() {
        v.push(c >> 5);
    }
    v.push(0);
    for c in hrp.bytes() {
        v.push(c & 0x1f);
    }
    v
}

fn create_checksum(hrp: &str, data: &[u8]) -> [u8; 6] {
    let mut values = hrp_expand(hrp);
    values.extend_from_slice(data);
    values.extend_from_slice(&[0u8; 6]);
    let pmod = polymod(&values) ^ 1;
    let mut out = [0u8; 6];
    for (i, slot) in out.iter_mut().enumerate() {
        *slot = ((pmod >> (5 * (5 - i))) & 0x1f) as u8;
    }
    out
}

fn verify_checksum(hrp: &str, data: &[u8]) -> bool {
    let mut values = hrp_expand(hrp);
    values.extend_from_slice(data);
    polymod(&values) == 1
}

/// Encode raw bytes into a bech32 string under the given HRP.
pub fn bech32_encode(hrp: &str, bytes: &[u8]) -> Result<String, NostrError> {
    if hrp.is_empty() || !hrp.bytes().all(|b| (33..=126).contains(&b)) {
        return Err(NostrError::Bech32("invalid hrp".into()));
    }
    let data5 = convert_bits(bytes, 8, 5, true)?;
    let checksum = create_checksum(hrp, &data5);
    let mut out = String::with_capacity(hrp.len() + 1 + data5.len() + 6);
    out.push_str(hrp);
    out.push('1');
    for v in data5.iter().chain(checksum.iter()) {
        out.push(CHARSET[*v as usize] as char);
    }
    Ok(out)
}

/// Decode a bech32 string into `(hrp, bytes)`.
pub fn bech32_decode(s: &str) -> Result<(String, Vec<u8>), NostrError> {
    if s.bytes().any(|b| !(33..=126).contains(&b)) {
        return Err(NostrError::Bech32("non-printable byte".into()));
    }
    let lower = s.to_ascii_lowercase();
    let upper = s.to_ascii_uppercase();
    if s != lower && s != upper {
        return Err(NostrError::Bech32("mixed case".into()));
    }
    let s = lower;
    let sep = s
        .rfind('1')
        .ok_or_else(|| NostrError::Bech32("missing separator".into()))?;
    if sep == 0 {
        return Err(NostrError::Bech32("empty hrp".into()));
    }
    if s.len() - sep - 1 < 6 {
        return Err(NostrError::Bech32("checksum too short".into()));
    }
    let hrp = &s[..sep];
    let mut data5 = Vec::with_capacity(s.len() - sep - 1);
    for c in s[sep + 1..].bytes() {
        let v = charset_rev(c)
            .ok_or_else(|| NostrError::Bech32(format!("bad char {c:?}")))?;
        data5.push(v);
    }
    if !verify_checksum(hrp, &data5) {
        return Err(NostrError::Bech32("bad checksum".into()));
    }
    let payload5 = &data5[..data5.len() - 6];
    let bytes = convert_bits(payload5, 5, 8, false)?;
    Ok((hrp.to_string(), bytes))
}

fn convert_bits(data: &[u8], from: u32, to: u32, pad: bool) -> Result<Vec<u8>, NostrError> {
    let mut acc: u32 = 0;
    let mut bits: u32 = 0;
    let mut out = Vec::with_capacity(data.len() * from as usize / to as usize + 1);
    let maxv = (1u32 << to) - 1;
    let max_acc = (1u32 << (from + to - 1)) - 1;
    for &v in data {
        let v = v as u32;
        if v >> from != 0 {
            return Err(NostrError::Bech32("value out of range".into()));
        }
        acc = ((acc << from) | v) & max_acc;
        bits += from;
        while bits >= to {
            bits -= to;
            out.push(((acc >> bits) & maxv) as u8);
        }
    }
    if pad {
        if bits > 0 {
            out.push(((acc << (to - bits)) & maxv) as u8);
        }
    } else if bits >= from || ((acc << (to - bits)) & maxv) != 0 {
        return Err(NostrError::Bech32("invalid padding".into()));
    }
    Ok(out)
}

// ----- NIP-19 HRPs -----

/// HRP for `npub` (32-byte x-only public key).
pub const HRP_NPUB: &str = "npub";
/// HRP for `nsec` (32-byte secret key).
pub const HRP_NSEC: &str = "nsec";
/// HRP for `note` (32-byte event id).
pub const HRP_NOTE: &str = "note";
/// HRP for `nprofile` (TLV).
pub const HRP_NPROFILE: &str = "nprofile";
/// HRP for `nevent` (TLV).
pub const HRP_NEVENT: &str = "nevent";
/// HRP for `naddr` (TLV).
pub const HRP_NADDR: &str = "naddr";
/// HRP for `nrelay` (TLV).
pub const HRP_NRELAY: &str = "nrelay";

const TLV_SPECIAL: u8 = 0;
const TLV_RELAY: u8 = 1;
const TLV_AUTHOR: u8 = 2;
const TLV_KIND: u8 = 3;

/// Encode a 32-byte x-only public key as `npub`.
pub fn encode_npub(pubkey: &[u8; 32]) -> Result<String, NostrError> {
    bech32_encode(HRP_NPUB, pubkey)
}

/// Decode an `npub` to a 32-byte x-only public key.
pub fn decode_npub(s: &str) -> Result<[u8; 32], NostrError> {
    decode_fixed32(s, HRP_NPUB)
}

/// Encode a 32-byte secret key as `nsec`.
pub fn encode_nsec(seckey: &[u8; 32]) -> Result<String, NostrError> {
    bech32_encode(HRP_NSEC, seckey)
}

/// Decode an `nsec` to a 32-byte secret key.
pub fn decode_nsec(s: &str) -> Result<[u8; 32], NostrError> {
    decode_fixed32(s, HRP_NSEC)
}

/// Encode a 32-byte event id as `note`.
pub fn encode_note(event_id: &[u8; 32]) -> Result<String, NostrError> {
    bech32_encode(HRP_NOTE, event_id)
}

/// Decode a `note` to a 32-byte event id.
pub fn decode_note(s: &str) -> Result<[u8; 32], NostrError> {
    decode_fixed32(s, HRP_NOTE)
}

fn decode_fixed32(s: &str, expected_hrp: &str) -> Result<[u8; 32], NostrError> {
    let (hrp, bytes) = bech32_decode(s)?;
    if hrp != expected_hrp {
        return Err(NostrError::Bech32(format!(
            "expected hrp {expected_hrp}, got {hrp}"
        )));
    }
    if bytes.len() != 32 {
        return Err(NostrError::Bech32("expected 32 bytes".into()));
    }
    let mut out = [0u8; 32];
    out.copy_from_slice(&bytes);
    Ok(out)
}

/// NIP-19 `nprofile` payload: pubkey + optional relay hints.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NProfile {
    /// 32-byte x-only public key.
    pub pubkey: [u8; 32],
    /// Suggested relays.
    pub relays: Vec<String>,
}

/// NIP-19 `nevent` payload: event id + optional relays/author/kind.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NEvent {
    /// 32-byte event id.
    pub event_id: [u8; 32],
    /// Suggested relays.
    pub relays: Vec<String>,
    /// Author public key.
    pub author: Option<[u8; 32]>,
    /// Event kind.
    pub kind: Option<u32>,
}

/// NIP-19 `nrelay` payload: one relay URL.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NRelay {
    /// Relay URL.
    pub url: String,
}

/// Encode an `nprofile`.
pub fn encode_nprofile(p: &NProfile) -> Result<String, NostrError> {
    let mut tlv = Vec::new();
    tlv_push(&mut tlv, TLV_SPECIAL, &p.pubkey);
    for r in &p.relays {
        tlv_push(&mut tlv, TLV_RELAY, r.as_bytes());
    }
    bech32_encode(HRP_NPROFILE, &tlv)
}

/// Decode an `nprofile`.
pub fn decode_nprofile(s: &str) -> Result<NProfile, NostrError> {
    let (hrp, bytes) = bech32_decode(s)?;
    if hrp != HRP_NPROFILE {
        return Err(NostrError::Bech32("expected nprofile".into()));
    }
    let entries = tlv_parse(&bytes)?;
    let mut pubkey: Option<[u8; 32]> = None;
    let mut relays = Vec::new();
    for (t, v) in entries {
        match t {
            TLV_SPECIAL => {
                if v.len() != 32 {
                    return Err(NostrError::Tlv("nprofile pubkey not 32 bytes".into()));
                }
                let mut a = [0u8; 32];
                a.copy_from_slice(&v);
                pubkey = Some(a);
            }
            TLV_RELAY => relays.push(
                String::from_utf8(v).map_err(|e| NostrError::Tlv(e.to_string()))?,
            ),
            _ => {}
        }
    }
    Ok(NProfile {
        pubkey: pubkey.ok_or_else(|| NostrError::Tlv("missing nprofile pubkey".into()))?,
        relays,
    })
}

/// Encode an `nevent`.
pub fn encode_nevent(e: &NEvent) -> Result<String, NostrError> {
    let mut tlv = Vec::new();
    tlv_push(&mut tlv, TLV_SPECIAL, &e.event_id);
    for r in &e.relays {
        tlv_push(&mut tlv, TLV_RELAY, r.as_bytes());
    }
    if let Some(a) = e.author {
        tlv_push(&mut tlv, TLV_AUTHOR, &a);
    }
    if let Some(k) = e.kind {
        let kb = k.to_be_bytes();
        tlv_push(&mut tlv, TLV_KIND, &kb);
    }
    bech32_encode(HRP_NEVENT, &tlv)
}

/// Decode an `nevent`.
pub fn decode_nevent(s: &str) -> Result<NEvent, NostrError> {
    let (hrp, bytes) = bech32_decode(s)?;
    if hrp != HRP_NEVENT {
        return Err(NostrError::Bech32("expected nevent".into()));
    }
    let entries = tlv_parse(&bytes)?;
    let mut event_id: Option<[u8; 32]> = None;
    let mut relays = Vec::new();
    let mut author: Option<[u8; 32]> = None;
    let mut kind: Option<u32> = None;
    for (t, v) in entries {
        match t {
            TLV_SPECIAL => {
                if v.len() != 32 {
                    return Err(NostrError::Tlv("nevent id not 32 bytes".into()));
                }
                let mut a = [0u8; 32];
                a.copy_from_slice(&v);
                event_id = Some(a);
            }
            TLV_RELAY => relays.push(
                String::from_utf8(v).map_err(|e| NostrError::Tlv(e.to_string()))?,
            ),
            TLV_AUTHOR => {
                if v.len() != 32 {
                    return Err(NostrError::Tlv("nevent author not 32 bytes".into()));
                }
                let mut a = [0u8; 32];
                a.copy_from_slice(&v);
                author = Some(a);
            }
            TLV_KIND => {
                if v.len() != 4 {
                    return Err(NostrError::Tlv("nevent kind not 4 bytes".into()));
                }
                let mut k = [0u8; 4];
                k.copy_from_slice(&v);
                kind = Some(u32::from_be_bytes(k));
            }
            _ => {}
        }
    }
    Ok(NEvent {
        event_id: event_id.ok_or_else(|| NostrError::Tlv("missing nevent id".into()))?,
        relays,
        author,
        kind,
    })
}

/// Encode an `nrelay`.
pub fn encode_nrelay(r: &NRelay) -> Result<String, NostrError> {
    let mut tlv = Vec::new();
    tlv_push(&mut tlv, TLV_SPECIAL, r.url.as_bytes());
    bech32_encode(HRP_NRELAY, &tlv)
}

/// Decode an `nrelay`.
pub fn decode_nrelay(s: &str) -> Result<NRelay, NostrError> {
    let (hrp, bytes) = bech32_decode(s)?;
    if hrp != HRP_NRELAY {
        return Err(NostrError::Bech32("expected nrelay".into()));
    }
    let entries = tlv_parse(&bytes)?;
    let mut url: Option<String> = None;
    for (t, v) in entries {
        if t == TLV_SPECIAL {
            url = Some(String::from_utf8(v).map_err(|e| NostrError::Tlv(e.to_string()))?);
        }
    }
    Ok(NRelay {
        url: url.ok_or_else(|| NostrError::Tlv("missing nrelay url".into()))?,
    })
}

fn tlv_push(out: &mut Vec<u8>, t: u8, v: &[u8]) {
    out.push(t);
    out.push(v.len() as u8);
    out.extend_from_slice(v);
}

fn tlv_parse(bytes: &[u8]) -> Result<Vec<(u8, Vec<u8>)>, NostrError> {
    let mut out = Vec::new();
    let mut i = 0;
    while i < bytes.len() {
        if i + 2 > bytes.len() {
            return Err(NostrError::Tlv("truncated tlv header".into()));
        }
        let t = bytes[i];
        let len = bytes[i + 1] as usize;
        i += 2;
        if i + len > bytes.len() {
            return Err(NostrError::Tlv("truncated tlv value".into()));
        }
        out.push((t, bytes[i..i + len].to_vec()));
        i += len;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bech32_roundtrip_short() {
        let s = bech32_encode("abc", b"hi").expect("enc");
        let (hrp, bytes) = bech32_decode(&s).expect("dec");
        assert_eq!(hrp, "abc");
        assert_eq!(bytes, b"hi");
    }

    #[test]
    fn npub_roundtrip() {
        let pk = [7u8; 32];
        let s = encode_npub(&pk).expect("enc");
        assert!(s.starts_with("npub1"));
        assert_eq!(decode_npub(&s).expect("dec"), pk);
    }

    #[test]
    fn nevent_with_author_and_kind() {
        let e = NEvent {
            event_id: [1u8; 32],
            relays: vec!["wss://relay.example".into()],
            author: Some([2u8; 32]),
            kind: Some(1),
        };
        let s = encode_nevent(&e).expect("enc");
        assert!(s.starts_with("nevent1"));
        let d = decode_nevent(&s).expect("dec");
        assert_eq!(d, e);
    }
}
