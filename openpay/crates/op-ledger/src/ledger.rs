//! Ledger container.
//!
//! A [`Ledger`] is a top-level scope. All accounts in a ledger share
//! a scope. Transactions can only post entries against accounts in
//! the same ledger.
//!
//! In practice a single OpenPay deployment usually has one ledger
//! per merchant (or per merchant per fiscal year, depending on the
//! audit story). Cross-ledger movements are modeled as two paired
//! transactions in two ledgers with a shared `external_id`
//! convention.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

/// Opaque ledger id.
#[derive(Copy, Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct LedgerId(pub Uuid);

impl LedgerId {
    /// Generate a fresh id.
    #[must_use]
    pub fn new() -> Self {
        Self(Uuid::new_v4())
    }

    /// Wrap an existing UUID.
    #[must_use]
    pub fn from_uuid(u: Uuid) -> Self {
        Self(u)
    }

    /// The underlying UUID.
    #[must_use]
    pub fn as_uuid(&self) -> Uuid {
        self.0
    }
}

impl Default for LedgerId {
    fn default() -> Self {
        Self::new()
    }
}

impl core::fmt::Display for LedgerId {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        self.0.fmt(f)
    }
}

/// A ledger.
///
/// Metadata only — the actual accounts and transactions live in a
/// [`LedgerStore`](crate::LedgerStore).
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Ledger {
    /// Stable id.
    pub id: LedgerId,
    /// Human-readable name (e.g. `"Acme Coffee LLC 2026"`).
    pub name: String,
    /// Optional description.
    pub description: Option<String>,
}

impl Ledger {
    /// Construct a fresh ledger.
    ///
    /// # Errors
    /// [`Error::InvalidInput`](crate::Error::InvalidInput) if `name`
    /// is empty.
    pub fn new(name: impl Into<String>) -> crate::Result<Self> {
        let name = name.into();
        if name.is_empty() {
            return Err(crate::Error::InvalidInput("ledger name empty".into()));
        }
        Ok(Self {
            id: LedgerId::new(),
            name,
            description: None,
        })
    }

    /// Builder: set a description.
    #[must_use]
    pub fn with_description(mut self, d: impl Into<String>) -> Self {
        self.description = Some(d.into());
        self
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn new_assigns_id() {
        let a = Ledger::new("acme").unwrap();
        let b = Ledger::new("acme").unwrap();
        assert_ne!(a.id, b.id);
    }

    #[test]
    fn empty_name_rejected() {
        let r = Ledger::new("");
        assert!(matches!(r, Err(crate::Error::InvalidInput(_))));
    }

    #[test]
    fn description_builder() {
        let l = Ledger::new("acme").unwrap().with_description("FY 2026");
        assert_eq!(l.description.as_deref(), Some("FY 2026"));
    }

    #[test]
    fn ledger_id_round_trip() {
        let u = Uuid::new_v4();
        assert_eq!(LedgerId::from_uuid(u).as_uuid(), u);
    }

    #[test]
    fn ledger_id_display() {
        let u = Uuid::new_v4();
        let id = LedgerId::from_uuid(u);
        assert_eq!(format!("{id}"), u.to_string());
    }
}
