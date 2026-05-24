//! Transaction Event Log (TEL).
//!
//! A TEL is a hash-chained log of credential lifecycle events anchored
//! by a registry. Spec §6 defines four event types:
//!
//! * `RIP` — Registry Inception. Brings a registry into existence and
//!   anchors it to the issuer's KEL.
//! * `VRT` — Registry roTation. Updates registry parameters (witnesses,
//!   backers, configuration) without changing identity.
//! * `ISS` — Credential Issuance. Records that an ACDC SAID has been
//!   issued under this registry.
//! * `REV` — Credential Revocation. Marks a previously issued
//!   credential as revoked.
//!
//! Each event is SAID-chained: every event's `p` field carries the
//! prior event's SAID, and the event's own `d` field is the SAID of
//! the canonical body with `d` placeheld. This is the same procedure
//! used by [`crate::acdc::Acdc`] and KERI's KEL.

use serde::{Deserialize, Serialize};
use smart_byte_core::Said;

use crate::error::{AcdcError, Result};

/// The four TEL event types.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TelEventKind {
    /// Registry Inception.
    Rip,
    /// Registry Rotation.
    Vrt,
    /// Credential Issuance.
    Iss,
    /// Credential Revocation.
    Rev,
}

impl TelEventKind {
    fn tag(self) -> &'static str {
        match self {
            Self::Rip => "rip",
            Self::Vrt => "vrt",
            Self::Iss => "iss",
            Self::Rev => "rev",
        }
    }
}

/// A single TEL event. Distinct event kinds populate different optional
/// fields; we keep them on one struct for cheap serialisation.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct TelEvent {
    /// Version string.
    pub v: String,
    /// Event SAID (`d`).
    pub d: Said,
    /// Event kind tag.
    pub t: String,
    /// Sequence number within the log (per-registry counter).
    pub s: u64,
    /// Prior event SAID (zero for the inception event).
    pub p: Said,
    /// Issuer AID (controller).
    pub i: String,
    /// Registry SAID (`ri`) — present on all events except RIP, which
    /// derives it from its own `d`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ri: Option<Said>,
    /// Anchored credential SAID for `iss`/`rev` events; `None` for
    /// registry events.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub a: Option<Said>,
    /// Anchor digest for `vrt` — config hash, etc. Opaque to the TEL.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub cfg: Option<Said>,
}

impl TelEvent {
    /// Recompute the SAID over this event's body and verify it matches.
    pub fn verify_said(&self) -> Result<()> {
        let computed = self.compute_said()?;
        if computed != self.d {
            return Err(AcdcError::SaidMismatch {
                asserted: self.d,
                computed,
            });
        }
        Ok(())
    }

    /// Compute the SAID with `d` placeheld.
    pub fn compute_said(&self) -> Result<Said> {
        let mut tmp = self.clone();
        tmp.d = Said([0u8; 32]);
        let bytes =
            serde_jcs::to_vec(&tmp).map_err(|e| AcdcError::Jcs(e.to_string()))?;
        Ok(Said::hash(&bytes))
    }
}

/// An in-memory append-only TEL for a single registry.
#[derive(Clone, Debug)]
pub struct Tel {
    issuer: String,
    registry: Said,
    events: Vec<TelEvent>,
}

impl Tel {
    /// Open a fresh TEL with a Registry Inception (`RIP`) event. The
    /// registry's SAID is derived from the inception event's SAID and
    /// is also returned for callers that want to anchor credentials.
    pub fn open(issuer: impl Into<String>) -> Result<Self> {
        let issuer = issuer.into();
        let mut event = TelEvent {
            v: crate::VERSION_STRING.to_string(),
            d: Said([0u8; 32]),
            t: TelEventKind::Rip.tag().into(),
            s: 0,
            p: Said([0u8; 32]),
            i: issuer.clone(),
            ri: None,
            a: None,
            cfg: None,
        };
        let said = event.compute_said()?;
        event.d = said;
        Ok(Self {
            issuer,
            registry: said,
            events: vec![event],
        })
    }

    /// Registry SAID — the SAID of the RIP event.
    pub fn registry(&self) -> Said {
        self.registry
    }

    /// Borrow all events in order.
    pub fn events(&self) -> &[TelEvent] {
        &self.events
    }

    /// Append a Registry Rotation event with a config-anchor SAID.
    pub fn rotate(&mut self, cfg: Said) -> Result<&TelEvent> {
        self.append(TelEventKind::Vrt, None, Some(cfg))
    }

    /// Append an Issuance event for a credential SAID.
    pub fn issue(&mut self, credential: Said) -> Result<&TelEvent> {
        self.append(TelEventKind::Iss, Some(credential), None)
    }

    /// Append a Revocation event for a credential SAID. Errors if the
    /// credential was never issued in this TEL or is already revoked.
    pub fn revoke(&mut self, credential: Said) -> Result<&TelEvent> {
        let state = self.credential_state(&credential);
        match state {
            CredentialTelState::Issued => {}
            CredentialTelState::Unknown => {
                return Err(AcdcError::Tel(format!(
                    "cannot revoke unknown credential {credential}"
                )));
            }
            CredentialTelState::Revoked => {
                return Err(AcdcError::Tel(format!(
                    "credential {credential} already revoked"
                )));
            }
        }
        self.append(TelEventKind::Rev, Some(credential), None)
    }

    /// Inspect a credential's TEL state.
    pub fn credential_state(&self, credential: &Said) -> CredentialTelState {
        let mut seen = false;
        for e in &self.events {
            if e.a.as_ref() == Some(credential) {
                match e.t.as_str() {
                    "iss" => seen = true,
                    "rev" => return CredentialTelState::Revoked,
                    _ => {}
                }
            }
        }
        if seen {
            CredentialTelState::Issued
        } else {
            CredentialTelState::Unknown
        }
    }

    fn append(
        &mut self,
        kind: TelEventKind,
        anchored: Option<Said>,
        cfg: Option<Said>,
    ) -> Result<&TelEvent> {
        let last = self
            .events
            .last()
            .ok_or_else(|| AcdcError::Tel("empty TEL".into()))?;
        let s = last.s + 1;
        let p = last.d;
        let mut event = TelEvent {
            v: crate::VERSION_STRING.to_string(),
            d: Said([0u8; 32]),
            t: kind.tag().into(),
            s,
            p,
            i: self.issuer.clone(),
            ri: Some(self.registry),
            a: anchored,
            cfg,
        };
        let said = event.compute_said()?;
        event.d = said;
        self.events.push(event);
        Ok(self
            .events
            .last()
            .expect("event just pushed must be present"))
    }

    /// Verify chain integrity: every event's `p` matches the previous
    /// event's `d`, every event's `d` matches its computed SAID, and
    /// sequence numbers are contiguous starting at 0.
    pub fn verify_chain(&self) -> Result<()> {
        let mut prev: Option<&TelEvent> = None;
        for (i, e) in self.events.iter().enumerate() {
            e.verify_said()?;
            if e.s as usize != i {
                return Err(AcdcError::Tel(format!(
                    "sequence gap at index {i}: event s={}",
                    e.s
                )));
            }
            if let Some(p) = prev {
                if e.p != p.d {
                    return Err(AcdcError::Tel(format!(
                        "prior mismatch at seq {}: expected {} got {}",
                        e.s, p.d, e.p
                    )));
                }
            } else if e.p != Said([0u8; 32]) {
                return Err(AcdcError::Tel(
                    "inception event must have zero prior".into(),
                ));
            }
            prev = Some(e);
        }
        Ok(())
    }
}

/// TEL-derived state for a credential.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CredentialTelState {
    /// Credential SAID has never been issued in this TEL.
    Unknown,
    /// Credential SAID was issued and is currently valid.
    Issued,
    /// Credential SAID was issued and later revoked.
    Revoked,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cred(byte: u8) -> Said {
        Said([byte; 32])
    }

    #[test]
    fn open_issues_and_revokes() {
        let mut tel = Tel::open("Bissuer").expect("open");
        let c = cred(1);
        tel.issue(c).expect("issue");
        assert_eq!(tel.credential_state(&c), CredentialTelState::Issued);
        tel.revoke(c).expect("revoke");
        assert_eq!(tel.credential_state(&c), CredentialTelState::Revoked);
        tel.verify_chain().expect("chain");
    }

    #[test]
    fn double_revoke_fails() {
        let mut tel = Tel::open("Bissuer").expect("open");
        let c = cred(2);
        tel.issue(c).expect("issue");
        tel.revoke(c).expect("revoke");
        assert!(matches!(tel.revoke(c), Err(AcdcError::Tel(_))));
    }

    #[test]
    fn revoke_unknown_fails() {
        let mut tel = Tel::open("Bissuer").expect("open");
        assert!(matches!(tel.revoke(cred(9)), Err(AcdcError::Tel(_))));
    }

    #[test]
    fn rotation_event_chains() {
        let mut tel = Tel::open("Bissuer").expect("open");
        tel.rotate(cred(7)).expect("rotate");
        tel.verify_chain().expect("chain");
    }
}
