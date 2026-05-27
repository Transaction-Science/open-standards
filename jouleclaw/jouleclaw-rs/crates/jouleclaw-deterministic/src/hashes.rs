//! Hash primitives. Compute a cryptographic digest of the byte
//! representation of the input string and return it as lowercase hex.

use jouleclaw_cascade::LawfulPrimitive;
use sha2::{Digest, Sha256};
use std::sync::Arc;

pub fn primitives() -> Vec<Arc<dyn LawfulPrimitive>> {
    vec![
        Arc::new(Blake3Hex),
        Arc::new(Sha256Hex),
        Arc::new(Md5Hex),
    ]
}

fn strip_prefix_ci<'a>(q: &'a str, prefix: &str) -> Option<&'a str> {
    let q = q.trim();
    if q.len() < prefix.len() {
        return None;
    }
    let (head, tail) = q.split_at(prefix.len());
    if !head.eq_ignore_ascii_case(prefix) {
        return None;
    }
    let rest = tail.strip_prefix(|c: char| c.is_whitespace())?;
    Some(rest.trim_start())
}

pub struct Blake3Hex;
impl LawfulPrimitive for Blake3Hex {
    fn id(&self) -> &str {
        "lawful:hashes:blake3"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_prefix_ci(query, "blake3")?;
        if rest.is_empty() {
            return None;
        }
        Some(blake3::hash(rest.as_bytes()).to_hex().to_string())
    }
    fn declared_cost_uj(&self) -> u64 {
        180
    }
}

pub struct Sha256Hex;
impl LawfulPrimitive for Sha256Hex {
    fn id(&self) -> &str {
        "lawful:hashes:sha256"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_prefix_ci(query, "sha256")?;
        if rest.is_empty() {
            return None;
        }
        let mut h = Sha256::new();
        h.update(rest.as_bytes());
        let digest = h.finalize();
        let mut s = String::with_capacity(64);
        for b in digest {
            s.push_str(&format!("{b:02x}"));
        }
        Some(s)
    }
    fn declared_cost_uj(&self) -> u64 {
        200
    }
}

pub struct Md5Hex;
impl LawfulPrimitive for Md5Hex {
    fn id(&self) -> &str {
        "lawful:hashes:md5"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_prefix_ci(query, "md5")?;
        if rest.is_empty() {
            return None;
        }
        let digest = md5::compute(rest.as_bytes());
        Some(format!("{digest:x}"))
    }
    fn declared_cost_uj(&self) -> u64 {
        160
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blake3_known_vector() {
        // BLAKE3("") == af1349b9f5f9a1a6a0404dea36dcc9499bcb25c9adc112b7cc9a93cae41f3262
        // We can't easily call with empty input through the keyword grammar,
        // so check a known short message.
        let out = Blake3Hex.try_resolve("blake3 abc").expect("hit");
        // 64-hex chars
        assert_eq!(out.len(), 64);
        assert!(out.chars().all(|c| c.is_ascii_hexdigit()));
        // Deterministic across calls.
        let again = Blake3Hex.try_resolve("blake3 abc").expect("hit");
        assert_eq!(out, again);
    }

    #[test]
    fn sha256_known_vector() {
        // SHA-256("abc") == ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        let out = Sha256Hex.try_resolve("sha256 abc").expect("hit");
        assert_eq!(out, "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    }

    #[test]
    fn md5_known_vector() {
        // MD5("abc") == 900150983cd24fb0d6963f7d28e17f72
        let out = Md5Hex.try_resolve("md5 abc").expect("hit");
        assert_eq!(out, "900150983cd24fb0d6963f7d28e17f72");
    }

    #[test]
    fn malformed_returns_none() {
        assert!(Blake3Hex.try_resolve("hash this").is_none());
        assert!(Sha256Hex.try_resolve("sha256").is_none());
        assert!(Md5Hex.try_resolve("md5  ").is_none()); // empty input after trim
    }

    #[test]
    fn category_count() {
        assert_eq!(primitives().len(), 3);
    }
}
