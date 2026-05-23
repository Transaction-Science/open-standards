//! W3C Data Integrity proofs and the discriminated [`Proof`] union.
//!
//! ## Cryptosuites
//!
//! * `eddsa-jcs-2022` — Ed25519 over the JCS-canonical JSON of the
//!   credential with `proof` removed. Implemented natively in this
//!   crate.
//! * `eddsa-rdfc-2022`, `ecdsa-rdfc-2019`, `ecdsa-jcs-2019`, `bbs-2023`
//!   — accepted in the [`DataIntegrityProof::proof_type`] field for
//!   round-trip parity with test vectors but only `eddsa-jcs-2022` can
//!   be issued or verified by this crate. RDFC variants require
//!   URDNA2015 dataset canonicalisation, which is gated behind the
//!   `rdf-canon` feature (no-op stub here).

use chrono::{DateTime, Utc};
use ed25519_dalek::{SigningKey, Verifier, VerifyingKey, ed25519::signature::Signer};
use iref::IriBuf;
use serde::{Deserialize, Serialize};

use crate::credential::VerifiableCredential;
use crate::error::VcError;

/// Proof purpose. The W3C registry includes `assertionMethod`,
/// `authentication`, `keyAgreement`, `capabilityInvocation`,
/// `capabilityDelegation`. We store as a typed wrapper around a string
/// so unknown purposes round-trip.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct ProofPurpose(pub String);

impl ProofPurpose {
    /// `assertionMethod` (the default for VC issuance).
    pub fn assertion_method() -> Self {
        Self("assertionMethod".to_string())
    }
    /// `authentication` (the default for VP proof binding).
    pub fn authentication() -> Self {
        Self("authentication".to_string())
    }
}

/// Embedded W3C Data Integrity proof.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct DataIntegrityProof {
    /// `type` (always `"DataIntegrityProof"` in v2).
    #[serde(rename = "type")]
    pub type_: String,
    /// Cryptosuite name.
    pub cryptosuite: String,
    /// Issuance timestamp.
    pub created: DateTime<Utc>,
    /// Verification-method IRI (typically a DID URL with fragment).
    #[serde(rename = "verificationMethod")]
    pub verification_method: IriBuf,
    /// Proof purpose.
    #[serde(rename = "proofPurpose")]
    pub proof_purpose: ProofPurpose,
    /// Multibase-encoded signature value.
    #[serde(rename = "proofValue")]
    pub proof_value: String,
}

/// External JWT proof (VC-JWT). The JWT itself carries the canonical
/// VC payload; we keep only the compact-form string here so the
/// embedding VC's `proof` array can reference it.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct JwtProof {
    /// `type` discriminator inside the `proof` array.
    #[serde(rename = "type")]
    pub type_: String,
    /// Compact JWS (`header.payload.signature`).
    pub jwt: String,
}

/// External Selective-Disclosure JWT proof.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SdJwtProof {
    /// `type` discriminator.
    #[serde(rename = "type")]
    pub type_: String,
    /// Combined SD-JWT serialisation: `jwt~d1~d2~…[~kb]`.
    pub sd_jwt: String,
}

/// Union of supported proof representations.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum Proof {
    /// Embedded Data Integrity proof.
    DataIntegrity(DataIntegrityProof),
    /// External VC-JWT.
    Jwt(JwtProof),
    /// External Selective-Disclosure JWT.
    SdJwt(SdJwtProof),
}

/// Cryptosuite name for the implemented suite.
pub const CRYPTOSUITE_EDDSA_JCS_2022: &str = "eddsa-jcs-2022";

/// Issue a Data Integrity proof over `vc` using cryptosuite
/// `eddsa-jcs-2022`. The proof is appended to `vc.proof` and the same
/// VC is returned for fluent use.
pub fn issue_data_integrity(
    mut vc: VerifiableCredential,
    verification_method: IriBuf,
    created: DateTime<Utc>,
    signing_key: &SigningKey,
) -> Result<VerifiableCredential, VcError> {
    let proof_meta = DataIntegrityProof {
        type_: "DataIntegrityProof".to_string(),
        cryptosuite: CRYPTOSUITE_EDDSA_JCS_2022.to_string(),
        created,
        verification_method,
        proof_purpose: ProofPurpose::assertion_method(),
        // Placeholder; replaced below.
        proof_value: String::new(),
    };
    // The W3C eddsa-jcs-2022 cryptosuite signs the concatenation of
    // hash(canonical_proof_config) || hash(canonical_credential).
    let proof_config = serde_json::json!({
        "type": proof_meta.type_,
        "cryptosuite": proof_meta.cryptosuite,
        "created": proof_meta.created,
        "verificationMethod": proof_meta.verification_method,
        "proofPurpose": proof_meta.proof_purpose,
    });
    let config_jcs = serde_jcs::to_vec(&proof_config)
        .map_err(|e| VcError::Jcs(e.to_string()))?;
    let cred_jcs = vc.to_jcs_without_proof()?;
    let config_hash = <sha2::Sha256 as sha2::Digest>::digest(&config_jcs);
    let cred_hash = <sha2::Sha256 as sha2::Digest>::digest(&cred_jcs);
    let mut tbs = Vec::with_capacity(64);
    tbs.extend_from_slice(&config_hash);
    tbs.extend_from_slice(&cred_hash);
    let sig = signing_key.sign(&tbs);
    let proof_value =
        multibase::encode(multibase::Base::Base58Btc, sig.to_bytes());
    let finished = DataIntegrityProof {
        proof_value,
        ..proof_meta
    };
    vc.proof.push(Proof::DataIntegrity(finished));
    Ok(vc)
}

/// Verify the embedded `eddsa-jcs-2022` Data Integrity proof on `vc`
/// using `verifying_key`. If multiple proofs are attached, all
/// `DataIntegrity` proofs must verify.
pub fn verify_data_integrity(
    vc: &VerifiableCredential,
    verifying_key: &VerifyingKey,
) -> Result<(), VcError> {
    let proofs: Vec<&DataIntegrityProof> = vc
        .proof
        .iter()
        .filter_map(|p| match p {
            Proof::DataIntegrity(d) => Some(d),
            _ => None,
        })
        .collect();
    if proofs.is_empty() {
        return Err(VcError::Signature(
            "no Data Integrity proof present".into(),
        ));
    }
    for di in proofs {
        if di.cryptosuite != CRYPTOSUITE_EDDSA_JCS_2022 {
            return Err(VcError::UnsupportedCryptosuite(di.cryptosuite.clone()));
        }
        let proof_config = serde_json::json!({
            "type": di.type_,
            "cryptosuite": di.cryptosuite,
            "created": di.created,
            "verificationMethod": di.verification_method,
            "proofPurpose": di.proof_purpose,
        });
        let config_jcs = serde_jcs::to_vec(&proof_config)
            .map_err(|e| VcError::Jcs(e.to_string()))?;
        let cred_jcs = vc.to_jcs_without_proof()?;
        let config_hash = <sha2::Sha256 as sha2::Digest>::digest(&config_jcs);
        let cred_hash = <sha2::Sha256 as sha2::Digest>::digest(&cred_jcs);
        let mut tbs = Vec::with_capacity(64);
        tbs.extend_from_slice(&config_hash);
        tbs.extend_from_slice(&cred_hash);
        let (_base, sig_bytes) = multibase::decode(&di.proof_value)
            .map_err(|e| VcError::Multibase(e.to_string()))?;
        if sig_bytes.len() != 64 {
            return Err(VcError::Signature(format!(
                "expected 64-byte Ed25519 signature, got {}",
                sig_bytes.len()
            )));
        }
        let mut s = [0u8; 64];
        s.copy_from_slice(&sig_bytes);
        let signature = ed25519_dalek::Signature::from_bytes(&s);
        verifying_key
            .verify(&tbs, &signature)
            .map_err(|e| VcError::Signature(e.to_string()))?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credential::{CredentialSubject, VcBuilder};
    use crate::issuer::Issuer;
    use chrono::TimeZone;
    use ed25519_dalek::SigningKey;
    use rand::rngs::OsRng;

    fn fixture_vc() -> VerifiableCredential {
        let subj = CredentialSubject {
            id: Some("did:example:alice".parse().unwrap()),
            claims: serde_json::Map::new(),
        };
        VcBuilder::new()
            .issuer(Issuer::Uri("did:example:issuer".parse().unwrap()))
            .subject(subj)
            .valid_from(chrono::Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap())
            .build()
            .unwrap()
    }

    #[test]
    fn issue_and_verify_data_integrity() {
        let sk = SigningKey::generate(&mut OsRng);
        let vk = sk.verifying_key();
        let vc = fixture_vc();
        let vm: IriBuf = "did:example:issuer#keys-1".parse().unwrap();
        let signed = issue_data_integrity(
            vc,
            vm,
            chrono::Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
            &sk,
        )
        .unwrap();
        verify_data_integrity(&signed, &vk).unwrap();
    }

    #[test]
    fn tamper_breaks_proof() {
        let sk = SigningKey::generate(&mut OsRng);
        let vk = sk.verifying_key();
        let vc = fixture_vc();
        let vm: IriBuf = "did:example:issuer#keys-1".parse().unwrap();
        let mut signed = issue_data_integrity(
            vc,
            vm,
            chrono::Utc.with_ymd_and_hms(2026, 1, 1, 0, 0, 0).unwrap(),
            &sk,
        )
        .unwrap();
        // Mutate the subject.
        signed.credential_subject[0]
            .claims
            .insert("x".into(), serde_json::Value::from(1));
        assert!(verify_data_integrity(&signed, &vk).is_err());
    }
}
