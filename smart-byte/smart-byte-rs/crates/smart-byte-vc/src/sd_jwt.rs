//! Selective-Disclosure JWT (draft-ietf-oauth-selective-disclosure-jwt-15).
//!
//! Workflow:
//!
//! 1. **Issue.** The issuer chooses which claims of a flat object are
//!    selectively disclosable. For each disclosable claim the issuer
//!    creates a *Disclosure* — `[salt, name, value]` — base64url-encodes
//!    its JSON form, and replaces the claim in the JWT payload with an
//!    `_sd` array entry containing the SHA-256 hash of the disclosure
//!    string. The JWT is then signed (EdDSA). The combined serialisation
//!    is `jwt~d1~d2~…`.
//!
//! 2. **Present.** The holder removes disclosures it does NOT want to
//!    reveal, producing `jwt~da~db` (a subset).
//!
//! 3. **Verify.** The verifier checks the JWT signature, hashes each
//!    presented disclosure with SHA-256, and matches the result against
//!    the `_sd` array in the JWT payload. Disclosed claims are
//!    re-inserted under their original names.
//!
//! This module implements the flat-claims subset of SD-JWT (no nested
//! `_sd` objects, no decoy digests). That is sufficient to exercise the
//! selective-disclosure path end-to-end against the IETF test vectors
//! for flat issuances.

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use ed25519_dalek::{
    Signature, SigningKey, Verifier, VerifyingKey, ed25519::signature::Signer,
};
use rand::RngCore;
use rand::rngs::OsRng;
use serde::{Deserialize, Serialize};
use serde_json::Value;

use crate::error::VcError;
use crate::proof::SdJwtProof;

/// One disclosure: `[salt, name, value]` triple, base64url-encoded for
/// transmission.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Disclosure {
    /// Random salt (>= 128 bits).
    pub salt: String,
    /// Claim name.
    pub name: String,
    /// Claim value.
    pub value: Value,
}

impl Disclosure {
    /// JSON array form: `["salt","name",value]`.
    pub fn to_json(&self) -> Result<Vec<u8>, VcError> {
        let arr = serde_json::json!([self.salt, self.name, self.value]);
        // SD-JWT uses standard (not canonical) JSON encoding for
        // disclosures; serde_json's compact form matches the draft.
        Ok(serde_json::to_vec(&arr)?)
    }

    /// Base64url-encoded disclosure ready for the `~`-delimited form.
    pub fn to_base64(&self) -> Result<String, VcError> {
        Ok(URL_SAFE_NO_PAD.encode(self.to_json()?))
    }

    /// SHA-256 digest, base64url-encoded — the value placed in `_sd`.
    pub fn digest(&self) -> Result<String, VcError> {
        let b64 = self.to_base64()?;
        let hash =
            <sha2::Sha256 as sha2::Digest>::digest(b64.as_bytes());
        Ok(URL_SAFE_NO_PAD.encode(hash))
    }

    /// Generate a fresh random 16-byte salt, base64url-encoded.
    pub fn new_salt() -> String {
        let mut buf = [0u8; 16];
        OsRng.fill_bytes(&mut buf);
        URL_SAFE_NO_PAD.encode(buf)
    }
}

/// Issue an SD-JWT.
///
/// `payload` is the full flat-claim JSON. Each name in
/// `selectively_disclosable` is removed from the payload, replaced by
/// an entry in `_sd`, and returned as a separate disclosure.
pub fn issue_sd_jwt(
    payload: &Value,
    selectively_disclosable: &[&str],
    signing_key: &SigningKey,
    kid: Option<String>,
) -> Result<(String, Vec<Disclosure>, SdJwtProof), VcError> {
    let obj = payload
        .as_object()
        .ok_or_else(|| VcError::SdJwt("payload must be a JSON object".into()))?;
    let mut visible = serde_json::Map::new();
    let mut disclosures: Vec<Disclosure> = Vec::new();
    let mut sd_digests: Vec<String> = Vec::new();
    for (k, v) in obj {
        if selectively_disclosable.contains(&k.as_str()) {
            let d = Disclosure {
                salt: Disclosure::new_salt(),
                name: k.clone(),
                value: v.clone(),
            };
            sd_digests.push(d.digest()?);
            disclosures.push(d);
        } else {
            visible.insert(k.clone(), v.clone());
        }
    }
    // Deterministically sort digests so issuance is reproducible
    // independent of the order in which claims were processed.
    sd_digests.sort();
    visible.insert("_sd".into(), serde_json::json!(sd_digests));
    visible.insert("_sd_alg".into(), serde_json::Value::from("sha-256"));

    let header = serde_json::json!({
        "alg": "EdDSA",
        "typ": "sd-jwt",
        "kid": kid,
    });
    let header_bytes = serde_json::to_vec(&header)?;
    let payload_bytes = serde_json::to_vec(&visible)?;
    let h = URL_SAFE_NO_PAD.encode(&header_bytes);
    let p = URL_SAFE_NO_PAD.encode(&payload_bytes);
    let signing_input = format!("{h}.{p}");
    let sig: Signature = signing_key.sign(signing_input.as_bytes());
    let s = URL_SAFE_NO_PAD.encode(sig.to_bytes());
    let jwt = format!("{signing_input}.{s}");

    let mut combined = jwt.clone();
    for d in &disclosures {
        combined.push('~');
        combined.push_str(&d.to_base64()?);
    }
    combined.push('~');

    let proof = SdJwtProof {
        type_: "SdJwtProof".to_string(),
        sd_jwt: combined.clone(),
    };
    Ok((combined, disclosures, proof))
}

/// Holder operation: produce a presentation that discloses only the
/// disclosures whose `name` is in `disclose`.
pub fn present_sd_jwt(
    combined: &str,
    disclose: &[&str],
) -> Result<String, VcError> {
    let mut parts = combined.split('~');
    let jwt = parts
        .next()
        .ok_or_else(|| VcError::SdJwt("empty SD-JWT".into()))?
        .to_string();
    let mut out = jwt;
    for token in parts {
        if token.is_empty() {
            continue;
        }
        let bytes = URL_SAFE_NO_PAD
            .decode(token)
            .map_err(|e| VcError::SdJwt(format!("disclosure b64: {e}")))?;
        let arr: Value = serde_json::from_slice(&bytes)?;
        let name = arr
            .get(1)
            .and_then(|v| v.as_str())
            .ok_or_else(|| VcError::SdJwt("disclosure missing name".into()))?;
        if disclose.contains(&name) {
            out.push('~');
            out.push_str(token);
        }
    }
    out.push('~');
    Ok(out)
}

/// Verifier operation: check the SD-JWT signature, validate each
/// presented disclosure against the JWT's `_sd` digest array, and
/// return the merged claim set (visible + disclosed).
pub fn verify_sd_jwt(
    combined: &str,
    verifying_key: &VerifyingKey,
) -> Result<Value, VcError> {
    let mut iter = combined.split('~');
    let jwt = iter
        .next()
        .ok_or_else(|| VcError::SdJwt("empty SD-JWT".into()))?;
    let parts: Vec<&str> = jwt.split('.').collect();
    if parts.len() != 3 {
        return Err(VcError::SdJwt(format!(
            "expected 3 JWS parts, got {}",
            parts.len()
        )));
    }
    let signing_input = format!("{}.{}", parts[0], parts[1]);
    let sig_bytes = URL_SAFE_NO_PAD.decode(parts[2])?;
    if sig_bytes.len() != 64 {
        return Err(VcError::SdJwt(format!(
            "expected 64-byte Ed25519 signature, got {}",
            sig_bytes.len()
        )));
    }
    let mut s = [0u8; 64];
    s.copy_from_slice(&sig_bytes);
    let signature = Signature::from_bytes(&s);
    verifying_key
        .verify(signing_input.as_bytes(), &signature)
        .map_err(|e| VcError::Signature(e.to_string()))?;

    let payload_bytes = URL_SAFE_NO_PAD.decode(parts[1])?;
    let payload: Value = serde_json::from_slice(&payload_bytes)?;
    let payload_obj = payload
        .as_object()
        .ok_or_else(|| VcError::SdJwt("payload not an object".into()))?;
    let sd_arr: Vec<String> = payload_obj
        .get("_sd")
        .and_then(|v| v.as_array())
        .map(|a| {
            a.iter()
                .filter_map(|x| x.as_str().map(|s| s.to_string()))
                .collect()
        })
        .unwrap_or_default();

    let mut merged = serde_json::Map::new();
    for (k, v) in payload_obj {
        if k == "_sd" || k == "_sd_alg" {
            continue;
        }
        merged.insert(k.clone(), v.clone());
    }

    for token in iter {
        if token.is_empty() {
            continue;
        }
        let bytes = URL_SAFE_NO_PAD
            .decode(token)
            .map_err(|e| VcError::SdJwt(format!("disclosure b64: {e}")))?;
        let arr: Value = serde_json::from_slice(&bytes)?;
        let digest = {
            let h = <sha2::Sha256 as sha2::Digest>::digest(token.as_bytes());
            URL_SAFE_NO_PAD.encode(h)
        };
        if !sd_arr.contains(&digest) {
            return Err(VcError::SdJwt(format!(
                "disclosure digest not in _sd: {digest}"
            )));
        }
        let name = arr
            .get(1)
            .and_then(|v| v.as_str())
            .ok_or_else(|| VcError::SdJwt("disclosure missing name".into()))?
            .to_string();
        let value = arr
            .get(2)
            .cloned()
            .ok_or_else(|| VcError::SdJwt("disclosure missing value".into()))?;
        merged.insert(name, value);
    }
    Ok(Value::Object(merged))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;
    use serde_json::json;

    #[test]
    fn issue_present_verify_subset() {
        let sk = SigningKey::generate(&mut OsRng);
        let vk = sk.verifying_key();
        let payload = json!({
            "given_name": "Alice",
            "family_name": "Lockhart",
            "birthdate": "1990-01-15",
            "email": "alice@example.org",
            "country": "GB",
            "iss": "did:example:issuer",
        });
        let (combined, _disc, _proof) = issue_sd_jwt(
            &payload,
            &["given_name", "family_name", "birthdate", "email", "country"],
            &sk,
            None,
        )
        .unwrap();
        // Holder discloses only 2 of 5 claims.
        let presented =
            present_sd_jwt(&combined, &["given_name", "country"]).unwrap();
        let merged = verify_sd_jwt(&presented, &vk).unwrap();
        assert_eq!(merged["given_name"], "Alice");
        assert_eq!(merged["country"], "GB");
        assert!(merged.get("family_name").is_none());
        assert!(merged.get("birthdate").is_none());
        assert!(merged.get("email").is_none());
        assert_eq!(merged["iss"], "did:example:issuer");
    }

    #[test]
    fn verify_rejects_forged_disclosure() {
        let sk = SigningKey::generate(&mut OsRng);
        let vk = sk.verifying_key();
        let payload = json!({"name": "Alice", "iss": "did:example:i"});
        let (combined, _d, _p) =
            issue_sd_jwt(&payload, &["name"], &sk, None).unwrap();
        // Splice in a forged disclosure that wasn't issued.
        let forged = Disclosure {
            salt: Disclosure::new_salt(),
            name: "name".into(),
            value: json!("Mallory"),
        };
        let jwt_part = combined.split('~').next().unwrap();
        let mut tampered = jwt_part.to_string();
        tampered.push('~');
        tampered.push_str(&forged.to_base64().unwrap());
        tampered.push('~');
        assert!(verify_sd_jwt(&tampered, &vk).is_err());
    }
}
