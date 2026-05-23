//! Bitstring Status List 2021 / W3C BitstringStatusList.
//!
//! Each credential subject in a status-list credential carries an
//! `encodedList` field: a base64url string whose underlying bytes are
//! the GZIP-compressed bitstring. Bit *i* of that bitstring records
//! the status of the credential whose `credentialStatus.statusListIndex`
//! equals `i`. A bit value of `1` means the status purpose is active
//! (revoked, suspended, …); `0` means inactive.
//!
//! This module provides:
//! * [`BitstringStatusList`] — a mutable bitstring that can be set,
//!   cleared, queried, and compressed to/from `encodedList` form.
//! * [`StatusListCredential`] — a thin wrapper around a [`crate::credential::VerifiableCredential`]
//!   whose subject is the status list.
//! * [`check_status`] — look up the status bit for a given index.

use std::io::{Read, Write};

use base64::Engine;
use base64::engine::general_purpose::URL_SAFE_NO_PAD;
use flate2::Compression;
use flate2::read::GzDecoder;
use flate2::write::GzEncoder;
use serde::{Deserialize, Serialize};

use crate::credential::VerifiableCredential;
use crate::error::VcError;

/// Status purpose. The W3C registry defines `revocation`, `suspension`,
/// and `message` (status with a message); custom values round-trip.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(transparent)]
pub struct StatusPurpose(pub String);

impl StatusPurpose {
    /// `revocation` — credential is permanently revoked.
    pub fn revocation() -> Self {
        Self("revocation".to_string())
    }
    /// `suspension` — credential is temporarily inactive.
    pub fn suspension() -> Self {
        Self("suspension".to_string())
    }
}

/// In-memory bitstring with `len` bits.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BitstringStatusList {
    bits: Vec<u8>,
    len: usize,
}

impl BitstringStatusList {
    /// Create a status list with `bit_len` bits, all zero.
    pub fn new(bit_len: usize) -> Self {
        let byte_len = bit_len.div_ceil(8);
        Self {
            bits: vec![0u8; byte_len],
            len: bit_len,
        }
    }

    /// Number of indexable bits.
    pub fn len(&self) -> usize {
        self.len
    }

    /// True if the list has no indexable bits.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Set bit at `index` to `value`. Returns an error if out of bounds.
    pub fn set(&mut self, index: usize, value: bool) -> Result<(), VcError> {
        if index >= self.len {
            return Err(VcError::StatusList(format!(
                "index {index} out of bounds (len={})",
                self.len
            )));
        }
        let byte = index / 8;
        let bit = 7 - (index % 8); // bit 0 of byte = MSB per the spec.
        if value {
            self.bits[byte] |= 1 << bit;
        } else {
            self.bits[byte] &= !(1 << bit);
        }
        Ok(())
    }

    /// Query bit at `index`.
    pub fn get(&self, index: usize) -> Result<bool, VcError> {
        if index >= self.len {
            return Err(VcError::StatusList(format!(
                "index {index} out of bounds (len={})",
                self.len
            )));
        }
        let byte = index / 8;
        let bit = 7 - (index % 8);
        Ok((self.bits[byte] >> bit) & 1 == 1)
    }

    /// Compress the bitstring with GZIP and encode as base64url. This
    /// is the form placed in `encodedList`.
    pub fn to_encoded(&self) -> Result<String, VcError> {
        let mut encoder = GzEncoder::new(Vec::new(), Compression::default());
        encoder
            .write_all(&self.bits)
            .map_err(|e| VcError::Io(e.to_string()))?;
        let gz = encoder
            .finish()
            .map_err(|e| VcError::Io(e.to_string()))?;
        Ok(URL_SAFE_NO_PAD.encode(gz))
    }

    /// Decode an `encodedList` string. `bit_len` is the declared bit
    /// length; the decoded byte array MUST be `ceil(bit_len / 8)` bytes
    /// after gunzip.
    pub fn from_encoded(encoded: &str, bit_len: usize) -> Result<Self, VcError> {
        let gz = URL_SAFE_NO_PAD.decode(encoded)?;
        let mut gunzip = GzDecoder::new(gz.as_slice());
        let mut out = Vec::new();
        gunzip
            .read_to_end(&mut out)
            .map_err(|e| VcError::Io(e.to_string()))?;
        let expected = bit_len.div_ceil(8);
        if out.len() != expected {
            return Err(VcError::StatusList(format!(
                "decoded list is {} bytes, expected {expected}",
                out.len()
            )));
        }
        Ok(Self {
            bits: out,
            len: bit_len,
        })
    }
}

/// A credential whose subject is a status list. Wraps a generic
/// [`VerifiableCredential`] for proof + DI handling, while exposing
/// helpers for the list-specific fields.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StatusListCredential {
    /// Wrapped credential.
    pub vc: VerifiableCredential,
    /// Status purpose ("revocation", "suspension", …).
    pub purpose: StatusPurpose,
    /// In-memory bitstring.
    pub bitstring: BitstringStatusList,
}

impl StatusListCredential {
    /// Embed the current bitstring into the wrapped credential's
    /// `credentialSubject[0]` claims as `statusPurpose`, `encodedList`,
    /// and `type`: `BitstringStatusList`.
    pub fn refresh_subject(&mut self) -> Result<(), VcError> {
        if self.vc.credential_subject.is_empty() {
            return Err(VcError::StatusList(
                "status list credential has no subject".into(),
            ));
        }
        let encoded = self.bitstring.to_encoded()?;
        let subj = &mut self.vc.credential_subject[0];
        subj.claims
            .insert("type".into(), serde_json::Value::from("BitstringStatusList"));
        subj.claims.insert(
            "statusPurpose".into(),
            serde_json::Value::from(self.purpose.0.clone()),
        );
        subj.claims
            .insert("encodedList".into(), serde_json::Value::from(encoded));
        Ok(())
    }
}

/// Convenience: look up the status of an index inside an `encodedList`
/// string. Returns `true` if the bit is set.
pub fn check_status(
    encoded_list: &str,
    bit_len: usize,
    index: usize,
) -> Result<bool, VcError> {
    let list = BitstringStatusList::from_encoded(encoded_list, bit_len)?;
    list.get(index)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::credential::{CredentialSubject, VcBuilder};
    use crate::issuer::Issuer;

    #[test]
    fn bit_roundtrip() {
        let mut bs = BitstringStatusList::new(1024);
        bs.set(0, true).unwrap();
        bs.set(42, true).unwrap();
        bs.set(1023, true).unwrap();
        let encoded = bs.to_encoded().unwrap();
        let back = BitstringStatusList::from_encoded(&encoded, 1024).unwrap();
        assert!(back.get(0).unwrap());
        assert!(back.get(42).unwrap());
        assert!(back.get(1023).unwrap());
        assert!(!back.get(1).unwrap());
        assert!(!back.get(500).unwrap());
    }

    #[test]
    fn out_of_bounds_errs() {
        let mut bs = BitstringStatusList::new(8);
        assert!(bs.set(8, true).is_err());
        assert!(bs.get(8).is_err());
    }

    #[test]
    fn status_list_credential_workflow() {
        let subj = CredentialSubject {
            id: Some("https://status.example/1".parse().unwrap()),
            claims: serde_json::Map::new(),
        };
        let vc = VcBuilder::new()
            .type_tag("BitstringStatusListCredential")
            .issuer(Issuer::Uri("did:example:issuer".parse().unwrap()))
            .subject(subj)
            .build()
            .unwrap();
        let mut slc = StatusListCredential {
            vc,
            purpose: StatusPurpose::revocation(),
            bitstring: BitstringStatusList::new(1024),
        };
        slc.bitstring.set(42, true).unwrap();
        slc.refresh_subject().unwrap();
        let encoded = slc.vc.credential_subject[0]
            .claims
            .get("encodedList")
            .and_then(|v| v.as_str())
            .unwrap()
            .to_string();
        assert!(check_status(&encoded, 1024, 42).unwrap());
        assert!(!check_status(&encoded, 1024, 43).unwrap());
    }
}
