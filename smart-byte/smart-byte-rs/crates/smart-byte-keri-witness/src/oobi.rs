//! Out-Of-Band Introduction (OOBI).
//!
//! An OOBI is a signed envelope distributed off-band that says "I claim
//! to be controller X, here are my witnesses, here's how to reach them."
//! Verifiers use OOBIs to bootstrap a controller's identity: from an
//! OOBI plus a witness reachable per the OOBI, a verifier can fetch
//! the full key-event log and replay it.

use chrono::{DateTime, Utc};
use ed25519_dalek::{Signature, Signer as EdSigner, Verifier, VerifyingKey};
use serde::{Deserialize, Serialize};
use smart_byte_core::Said;

use crate::controller::KeyPair;
use crate::error::{KeriError, Result};
use crate::events::{ControllerAid, WitnessAid};

/// Network endpoint advertised by an OOBI.
///
/// The `url` schema is opaque to this crate — it can be an HTTPS URL,
/// an Iroh node-id, a `tor:` onion address, or anything callers agree
/// on.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct OobiEndpoint {
    /// Witness AID the endpoint serves.
    pub witness: WitnessAid,
    /// Free-form network address.
    pub url: String,
}

/// Out-Of-Band Introduction payload.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Oobi {
    /// Controller AID being introduced.
    pub controller: ControllerAid,
    /// SAID of the controller's inception event — pinning the
    /// introduction to one specific identity epoch.
    pub inception_said: Said,
    /// Endpoints for the controller's witnesses.
    pub endpoints: Vec<OobiEndpoint>,
    /// When the OOBI was minted.
    pub issued_at: DateTime<Utc>,
    /// Issuer's Ed25519 verifying key, base32-encoded.
    pub issuer_pubkey_b32: String,
    /// Ed25519 signature over the canonical CBOR of the OOBI with the
    /// signature field cleared, base32-encoded.
    pub signature_b32: String,
}

impl Oobi {
    /// Mint a fresh OOBI signed by `signer`.
    pub fn issue(
        controller: ControllerAid,
        inception_said: Said,
        endpoints: Vec<OobiEndpoint>,
        signer: &KeyPair,
    ) -> Result<Self> {
        let mut oobi = Self {
            controller,
            inception_said,
            endpoints,
            issued_at: Utc::now(),
            issuer_pubkey_b32: data_encoding_b32(signer.verifying.as_bytes()),
            signature_b32: String::new(),
        };
        let body = serde_cbor::to_vec(&oobi)?;
        let sig: Signature = signer.signing.sign(&body);
        oobi.signature_b32 = data_encoding_b32(&sig.to_bytes());
        Ok(oobi)
    }

    /// Verify the OOBI's signature against the embedded issuer public key.
    pub fn verify(&self) -> Result<()> {
        let pk_bytes = decode_b32_fixed::<32>(&self.issuer_pubkey_b32)?;
        let vk = VerifyingKey::from_bytes(&pk_bytes)
            .map_err(|e| KeriError::MalformedKey(e.to_string()))?;
        let sig_bytes = decode_b32_fixed::<64>(&self.signature_b32)?;
        let sig = Signature::from_bytes(&sig_bytes);

        let mut stripped = self.clone();
        stripped.signature_b32 = String::new();
        let body = serde_cbor::to_vec(&stripped)?;
        vk.verify(&body, &sig).map_err(|_| KeriError::BadSignature)
    }
}

fn data_encoding_b32(bytes: &[u8]) -> String {
    data_encoding::BASE32_NOPAD.encode(bytes)
}

fn decode_b32_fixed<const N: usize>(s: &str) -> Result<[u8; N]> {
    let upper = s.to_ascii_uppercase();
    let bytes = data_encoding::BASE32_NOPAD
        .decode(upper.as_bytes())
        .map_err(|e| KeriError::MalformedKey(e.to_string()))?;
    if bytes.len() != N {
        return Err(KeriError::MalformedKey(format!(
            "expected {N}-byte base32 payload, got {}",
            bytes.len()
        )));
    }
    let mut out = [0u8; N];
    out.copy_from_slice(&bytes);
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::rngs::OsRng;

    #[test]
    fn oobi_round_trip() {
        let mut rng = OsRng;
        let kp = KeyPair::generate(&mut rng);
        let said = Said::hash(b"inception");
        let aid = ControllerAid::from_inception_said(&said);
        let oobi = Oobi::issue(
            aid,
            said,
            vec![OobiEndpoint {
                witness: WitnessAid("W1".into()),
                url: "https://example.test/witness1".into(),
            }],
            &kp,
        )
        .expect("issue");
        oobi.verify().expect("verify");
    }
}
