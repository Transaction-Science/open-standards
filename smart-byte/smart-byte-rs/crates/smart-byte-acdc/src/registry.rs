//! Credential registry trait + in-memory implementation.
//!
//! A registry ties together (a) a credential body, (b) the TEL events
//! anchoring its lifecycle, and (c) the issuer/holder bookkeeping a
//! verifier needs. The [`CredentialRegistry`] trait is the abstract
//! surface; [`InMemoryRegistry`] is the reference implementation used
//! by tests and by downstream callers that do not need persistence.

use std::collections::HashMap;

use smart_byte_core::Said;

use crate::acdc::Acdc;
use crate::error::{AcdcError, Result};
use crate::tel::{CredentialTelState, Tel};

/// Combined runtime state for a credential held in a registry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum RegistryState {
    /// Issued and currently valid.
    Active,
    /// Issued and later revoked.
    Revoked,
    /// Not present in this registry.
    Absent,
}

/// Storage and lifecycle operations for a credential registry.
pub trait CredentialRegistry {
    /// SAID of the registry (the TEL's inception event).
    fn registry_said(&self) -> Said;

    /// Issuer AID owning this registry.
    fn issuer(&self) -> &str;

    /// Issue a credential. The credential must already be SAID-sealed
    /// (built via [`crate::acdc::AcdcBuilder`]). The registry both
    /// stores the body and appends an `iss` event to the TEL.
    fn issue(&mut self, credential: Acdc) -> Result<()>;

    /// Revoke a previously issued credential by SAID.
    fn revoke(&mut self, credential: &Said) -> Result<()>;

    /// Look up the runtime state of a credential.
    fn state(&self, credential: &Said) -> RegistryState;

    /// Retrieve the credential body if present.
    fn get(&self, credential: &Said) -> Option<&Acdc>;
}

/// In-memory registry combining a TEL with a SAID->ACDC map.
#[derive(Clone, Debug)]
pub struct InMemoryRegistry {
    tel: Tel,
    credentials: HashMap<Said, Acdc>,
}

impl InMemoryRegistry {
    /// Open a new registry rooted at a freshly created TEL.
    pub fn open(issuer: impl Into<String>) -> Result<Self> {
        let tel = Tel::open(issuer)?;
        Ok(Self {
            tel,
            credentials: HashMap::new(),
        })
    }

    /// Borrow the underlying TEL.
    pub fn tel(&self) -> &Tel {
        &self.tel
    }
}

impl CredentialRegistry for InMemoryRegistry {
    fn registry_said(&self) -> Said {
        self.tel.registry()
    }

    fn issuer(&self) -> &str {
        // The first event of any TEL is the RIP, which carries the
        // issuer AID. Borrow it from there.
        self.tel
            .events()
            .first()
            .map(|e| e.i.as_str())
            .unwrap_or("")
    }

    fn issue(&mut self, credential: Acdc) -> Result<()> {
        credential.verify_said()?;
        // Reject if the credential's `ri` field, when present,
        // disagrees with this registry.
        if let Some(ri) = credential.ri {
            if ri != self.registry_said() {
                return Err(AcdcError::Registry(format!(
                    "credential anchored to registry {ri}, not {}",
                    self.registry_said()
                )));
            }
        }
        if self.credentials.contains_key(&credential.d) {
            return Err(AcdcError::Registry(format!(
                "credential {} already issued",
                credential.d
            )));
        }
        self.tel.issue(credential.d)?;
        self.credentials.insert(credential.d, credential);
        Ok(())
    }

    fn revoke(&mut self, credential: &Said) -> Result<()> {
        if !self.credentials.contains_key(credential) {
            return Err(AcdcError::Registry(format!(
                "credential {credential} not present in registry"
            )));
        }
        self.tel.revoke(*credential)?;
        Ok(())
    }

    fn state(&self, credential: &Said) -> RegistryState {
        match self.tel.credential_state(credential) {
            CredentialTelState::Issued => RegistryState::Active,
            CredentialTelState::Revoked => RegistryState::Revoked,
            CredentialTelState::Unknown => RegistryState::Absent,
        }
    }

    fn get(&self, credential: &Said) -> Option<&Acdc> {
        self.credentials.get(credential)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acdc::{AcdcBuilder, AttributeSection, SchemaSection};
    use serde_json::json;

    fn cred(reg: Said) -> Acdc {
        let mut s = serde_json::Map::new();
        s.insert("$id".into(), json!("schema"));
        let mut a = serde_json::Map::new();
        a.insert("name".into(), json!("Alice"));
        AcdcBuilder::new()
            .issuer("Bissuer")
            .registry(reg)
            .schema(SchemaSection::Inline(s))
            .attributes(AttributeSection::Inline(a))
            .build()
            .expect("build")
    }

    #[test]
    fn lifecycle() {
        let mut r = InMemoryRegistry::open("Bissuer").expect("open");
        let c = cred(r.registry_said());
        let id = c.d;
        r.issue(c).expect("issue");
        assert_eq!(r.state(&id), RegistryState::Active);
        r.revoke(&id).expect("revoke");
        assert_eq!(r.state(&id), RegistryState::Revoked);
    }

    #[test]
    fn rejects_credential_for_other_registry() {
        let mut r = InMemoryRegistry::open("Bissuer").expect("open");
        let other = Said([0xAB; 32]);
        let c = cred(other);
        assert!(matches!(r.issue(c), Err(AcdcError::Registry(_))));
    }
}
