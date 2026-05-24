//! HTTP Signatures — draft-cavage-http-signatures-12 + RFC 9421 helpers.
//!
//! ActivityPub federation rides on signed HTTP requests. The deployed
//! footprint is overwhelmingly draft-cavage-12 (Mastodon, Pleroma,
//! Misskey), with newer servers also accepting RFC 9421.
//!
//! This module provides:
//!
//! * [`SigningString`] — canonical concatenation of `(request-target)`,
//!   `host`, `date`, `digest`, and other listed headers.
//! * [`SignatureParams`] — the parsed `Signature:` header (keyId,
//!   algorithm, headers, signature).
//! * [`Digest`] — `SHA-256=BASE64(SHA256(body))` helpers.
//! * [`sign_ed25519`] / [`verify_ed25519`] — convenience signers.
//!
//! We treat the cavage and RFC 9421 modes as variants of the same
//! canonicalisation; an RFC-9421 caller picks `@method` / `@authority`
//! / `@path` instead of `(request-target)`, but the signing-string
//! shape is identical.

use crate::error::{ActivityPubError, Result};
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use ed25519_dalek::{
    Signature, Signer, SigningKey, Verifier, VerifyingKey, SIGNATURE_LENGTH,
};
use sha2::{Digest as _, Sha256};

/// Body digest header value (`SHA-256=BASE64(SHA256(body))`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Digest(pub String);

impl Digest {
    /// Compute the digest of a body.
    pub fn sha256(body: &[u8]) -> Self {
        let mut h = Sha256::new();
        h.update(body);
        let out = h.finalize();
        Digest(format!("SHA-256={}", B64.encode(out)))
    }

    /// Render as the header value.
    pub fn header_value(&self) -> &str {
        &self.0
    }

    /// Verify a digest against a body.
    pub fn verify(&self, body: &[u8]) -> bool {
        let expected = Self::sha256(body);
        expected == *self
    }
}

/// The headers selected for inclusion in the signing string and the
/// canonicalised material itself.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SigningString {
    /// Header names in lower-case, in the order they appear in the
    /// signing-string.
    pub headers: Vec<String>,
    /// Final canonical bytes that are fed to the signer.
    pub bytes: String,
}

impl SigningString {
    /// Build the canonical signing string from a request.
    ///
    /// * `method` is upper- or lower-case; the cavage form lowercases it.
    /// * `path` is the request target path + optional `?query`.
    /// * `headers` is a slice of `(name, value)` tuples. Names are
    ///   case-insensitive and lowered here. Order of `header_list` is
    ///   preserved.
    pub fn build(
        method: &str,
        path: &str,
        headers: &[(&str, &str)],
        header_list: &[&str],
    ) -> Result<Self> {
        let mut lines = Vec::with_capacity(header_list.len());
        let mut names = Vec::with_capacity(header_list.len());
        for h in header_list {
            let h_lower = h.to_ascii_lowercase();
            let value = match h_lower.as_str() {
                "(request-target)" => format!("{} {}", method.to_ascii_lowercase(), path),
                "@method" => method.to_ascii_uppercase(),
                "@path" => path.to_string(),
                _ => header_lookup(headers, &h_lower).ok_or_else(|| {
                    ActivityPubError::HttpSig(format!("header {h_lower} missing for signing"))
                })?,
            };
            lines.push(format!("{h_lower}: {value}"));
            names.push(h_lower);
        }
        Ok(SigningString {
            headers: names,
            bytes: lines.join("\n"),
        })
    }
}

fn header_lookup(headers: &[(&str, &str)], name_lower: &str) -> Option<String> {
    headers
        .iter()
        .find(|(k, _)| k.eq_ignore_ascii_case(name_lower))
        .map(|(_, v)| (*v).to_string())
}

/// Parsed `Signature:` header parameters.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SignatureParams {
    /// `keyId` — the IRI of the verifying public key.
    pub key_id: String,
    /// `algorithm` — e.g. `"rsa-sha256"`, `"ed25519"`, `"hs2019"`.
    pub algorithm: String,
    /// `headers` — space-separated header list that was signed.
    pub headers: Vec<String>,
    /// Base64-encoded signature bytes.
    pub signature: String,
}

impl SignatureParams {
    /// Serialise to the cavage `Signature:` header value form:
    ///
    /// ```text
    /// keyId="…",algorithm="…",headers="…",signature="…"
    /// ```
    pub fn to_header(&self) -> String {
        format!(
            "keyId=\"{}\",algorithm=\"{}\",headers=\"{}\",signature=\"{}\"",
            self.key_id,
            self.algorithm,
            self.headers.join(" "),
            self.signature
        )
    }

    /// Parse a cavage `Signature:` header value.
    ///
    /// Handles the quoted-string form used by Mastodon. Whitespace and
    /// optional `Signature ` prefix are tolerated.
    pub fn parse(header_value: &str) -> Result<Self> {
        let raw = header_value
            .trim()
            .strip_prefix("Signature ")
            .or_else(|| header_value.trim().strip_prefix("signature "))
            .unwrap_or(header_value.trim());
        let mut key_id = None;
        let mut algorithm = None;
        let mut headers = None;
        let mut signature = None;
        for part in split_params(raw) {
            let (name, value) = part
                .split_once('=')
                .ok_or_else(|| ActivityPubError::HttpSig(format!("bad param {part}")))?;
            let value = value.trim_matches('"');
            match name.trim() {
                "keyId" => key_id = Some(value.to_string()),
                "algorithm" => algorithm = Some(value.to_string()),
                "headers" => {
                    headers = Some(
                        value
                            .split(' ')
                            .filter(|s| !s.is_empty())
                            .map(|s| s.to_string())
                            .collect(),
                    )
                }
                "signature" => signature = Some(value.to_string()),
                _ => {}
            }
        }
        Ok(SignatureParams {
            key_id: key_id
                .ok_or_else(|| ActivityPubError::HttpSig("missing keyId".into()))?,
            algorithm: algorithm.unwrap_or_else(|| "hs2019".to_string()),
            headers: headers.unwrap_or_else(|| vec!["(created)".to_string()]),
            signature: signature
                .ok_or_else(|| ActivityPubError::HttpSig("missing signature".into()))?,
        })
    }
}

/// Split a cavage parameter string respecting double-quoted values.
fn split_params(s: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut buf = String::new();
    let mut in_quotes = false;
    for c in s.chars() {
        match c {
            '"' => {
                in_quotes = !in_quotes;
                buf.push(c);
            }
            ',' if !in_quotes => {
                if !buf.trim().is_empty() {
                    out.push(buf.trim().to_string());
                }
                buf.clear();
            }
            _ => buf.push(c),
        }
    }
    if !buf.trim().is_empty() {
        out.push(buf.trim().to_string());
    }
    out
}

/// Sign a [`SigningString`] with an Ed25519 key.
///
/// Returns the base64-encoded signature suitable for the `signature=`
/// parameter on a `Signature:` header.
pub fn sign_ed25519(key: &SigningKey, signing_string: &SigningString) -> String {
    let sig: Signature = key.sign(signing_string.bytes.as_bytes());
    B64.encode(sig.to_bytes())
}

/// Verify an Ed25519 signature against a canonical signing string.
pub fn verify_ed25519(
    key: &VerifyingKey,
    signing_string: &SigningString,
    signature_b64: &str,
) -> Result<()> {
    let sig_bytes = B64
        .decode(signature_b64)
        .map_err(|e| ActivityPubError::HttpSig(format!("base64 decode: {e}")))?;
    if sig_bytes.len() != SIGNATURE_LENGTH {
        return Err(ActivityPubError::HttpSig(format!(
            "ed25519 signature must be {SIGNATURE_LENGTH} bytes, got {}",
            sig_bytes.len()
        )));
    }
    let mut arr = [0u8; SIGNATURE_LENGTH];
    arr.copy_from_slice(&sig_bytes);
    let sig = Signature::from_bytes(&arr);
    key.verify(signing_string.bytes.as_bytes(), &sig)
        .map_err(|e| ActivityPubError::HttpSig(format!("ed25519 verify: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    #[test]
    fn digest_roundtrip() {
        let d = Digest::sha256(b"hello");
        assert!(d.header_value().starts_with("SHA-256="));
        assert!(d.verify(b"hello"));
        assert!(!d.verify(b"world"));
    }

    #[test]
    fn signing_string_shape() -> Result<()> {
        let s = SigningString::build(
            "POST",
            "/inbox",
            &[
                ("Host", "example.test"),
                ("Date", "Tue, 20 May 2025 14:00:00 GMT"),
                ("Digest", "SHA-256=abcdef"),
            ],
            &["(request-target)", "host", "date", "digest"],
        )?;
        let expected = "(request-target): post /inbox\nhost: example.test\ndate: Tue, 20 May 2025 14:00:00 GMT\ndigest: SHA-256=abcdef";
        assert_eq!(s.bytes, expected);
        Ok(())
    }

    #[test]
    fn header_parse_roundtrip() -> Result<()> {
        let sp = SignatureParams {
            key_id: "https://example.test/users/alice#main-key".to_string(),
            algorithm: "ed25519".to_string(),
            headers: vec![
                "(request-target)".to_string(),
                "host".to_string(),
                "date".to_string(),
            ],
            signature: "AAAA".to_string(),
        };
        let header = sp.to_header();
        let back = SignatureParams::parse(&header)?;
        assert_eq!(back, sp);
        Ok(())
    }

    #[test]
    fn ed25519_sign_and_verify() -> Result<()> {
        let mut csprng = OsRng;
        let key = SigningKey::generate(&mut csprng);
        let s = SigningString::build(
            "POST",
            "/inbox",
            &[("Host", "example.test")],
            &["(request-target)", "host"],
        )?;
        let sig = sign_ed25519(&key, &s);
        verify_ed25519(&key.verifying_key(), &s, &sig)?;
        Ok(())
    }
}
