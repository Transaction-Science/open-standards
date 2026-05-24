//! ZCAP-LD capability chains (W3C CCG / DIF EDV v0.10 § 6.3 — Authorization).
//!
//! ZCAP-LD ("authorization capabilities for linked data") is the
//! delegation model used by EDV: a vault controller mints a *root
//! capability* over their vault, then delegates restricted slices of that
//! capability to other DIDs by signing a chain of capability documents.
//!
//! This module is a deliberately small, in-memory model of the data
//! shape — enough to express delegation, enforce caveat (action) lists,
//! and validate chains in tests and reference deployments. Full HTTP
//! Signatures / capability-invocation proofs are out of scope for this
//! reference build; the [`verify_chain`] function takes a `signer` lookup
//! callback so callers can wire in any signature scheme they like.

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::error::EdvError;

/// A single capability document. Capabilities form a chain via
/// [`parent_capability`](Capability::parent_capability); the root has
/// `parent_capability == None` and is signed by the resource controller.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Capability {
    /// `@context` — typically `https://w3id.org/zcap/v1`.
    #[serde(rename = "@context")]
    pub context: Vec<String>,
    /// Capability identifier (URN).
    pub id: String,
    /// Invocation target — the URL or URN the capability authorises
    /// action upon (e.g. the vault root URL or a specific document URL).
    pub invocation_target: String,
    /// DID of the controller of this capability (the delegate).
    pub controller: String,
    /// DID of the party authorised to invoke this capability.
    pub invoker: String,
    /// Allowed actions — a non-empty list of action strings such as
    /// `read`, `write`, `delete`. An empty list means "all actions".
    #[serde(default)]
    pub allowed_action: Vec<String>,
    /// Reference to the parent capability id, if any.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub parent_capability: Option<String>,
    /// Optional expiry — capabilities are rejected after this time.
    #[serde(skip_serializing_if = "Option::is_none", default)]
    pub expires: Option<DateTime<Utc>>,
}

impl Capability {
    /// Build a root capability owned by `controller`, granting `invoker`
    /// the listed `actions` on `target`.
    pub fn root(
        id: impl Into<String>,
        target: impl Into<String>,
        controller: impl Into<String>,
        invoker: impl Into<String>,
        actions: Vec<String>,
    ) -> Self {
        Self {
            context: vec!["https://w3id.org/zcap/v1".into()],
            id: id.into(),
            invocation_target: target.into(),
            controller: controller.into(),
            invoker: invoker.into(),
            allowed_action: actions,
            parent_capability: None,
            expires: None,
        }
    }

    /// Delegate this capability to `new_invoker` with `actions`. The
    /// delegate's actions MUST be a subset of `self.allowed_action`
    /// (unless `self.allowed_action` is empty, in which case any action
    /// is allowed and the delegate can pick a non-empty subset).
    pub fn delegate(
        &self,
        id: impl Into<String>,
        new_controller: impl Into<String>,
        new_invoker: impl Into<String>,
        actions: Vec<String>,
    ) -> Result<Self, EdvError> {
        if !self.allowed_action.is_empty() {
            for a in &actions {
                if !self.allowed_action.contains(a) {
                    return Err(EdvError::Capability(format!(
                        "delegated action {a} not in parent's allowed_action"
                    )));
                }
            }
        }
        Ok(Self {
            context: self.context.clone(),
            id: id.into(),
            invocation_target: self.invocation_target.clone(),
            controller: new_controller.into(),
            invoker: new_invoker.into(),
            allowed_action: actions,
            parent_capability: Some(self.id.clone()),
            expires: self.expires,
        })
    }

    /// Does this capability authorise `action`?
    pub fn allows(&self, action: &str) -> bool {
        self.allowed_action.is_empty()
            || self.allowed_action.iter().any(|a| a == action)
    }
}

/// An invocation of a capability — the runtime credential the caller
/// presents to the vault. Carries the action the caller wants to perform.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Invocation {
    /// The capability being invoked (by id).
    pub capability: String,
    /// The action the caller wants to perform.
    pub action: String,
    /// The DID of the caller.
    pub invoker: String,
    /// When this invocation was created — used for clock-skew checks.
    #[serde(default = "Utc::now")]
    pub created: DateTime<Utc>,
}

impl Invocation {
    /// Build an invocation for `action` against `cap`.
    pub fn new(cap: &Capability, action: impl Into<String>) -> Self {
        Self {
            capability: cap.id.clone(),
            action: action.into(),
            invoker: cap.invoker.clone(),
            created: Utc::now(),
        }
    }
}

/// Validate a chain of capabilities rooted at `chain[0]`.
///
/// Invariants enforced:
///
/// 1. `chain[0].parent_capability` is `None` (it's a root).
/// 2. The root's controller is `expected_root_controller`.
/// 3. For each adjacent pair `(parent, child)`:
///    * `child.parent_capability == Some(parent.id)`
///    * `child.invocation_target == parent.invocation_target`
///    * Each `child.allowed_action` is in `parent.allowed_action`
///      (or `parent.allowed_action` is empty).
///    * `child.controller == parent.invoker` (the delegate must be the
///      previously-named invoker).
/// 4. No capability in the chain has expired.
/// 5. `invocation.invoker == chain.last().invoker`.
/// 6. `invocation.action` is in `chain.last().allowed_action` (or list
///    is empty).
pub fn verify_chain(
    chain: &[Capability],
    expected_root_controller: &str,
    invocation: &Invocation,
) -> Result<(), EdvError> {
    let root = chain
        .first()
        .ok_or_else(|| EdvError::Capability("empty chain".into()))?;
    if root.parent_capability.is_some() {
        return Err(EdvError::Capability(
            "chain[0] is not a root capability".into(),
        ));
    }
    if root.controller != expected_root_controller {
        return Err(EdvError::Capability(format!(
            "root controller mismatch: expected {expected_root_controller}, got {}",
            root.controller
        )));
    }

    let now = Utc::now();
    for cap in chain {
        if let Some(exp) = cap.expires {
            if exp < now {
                return Err(EdvError::Capability(format!(
                    "capability {} expired at {}",
                    cap.id, exp
                )));
            }
        }
    }

    for pair in chain.windows(2) {
        let parent = &pair[0];
        let child = &pair[1];
        if child.parent_capability.as_deref() != Some(parent.id.as_str()) {
            return Err(EdvError::Capability(format!(
                "child {} does not name parent {}",
                child.id, parent.id
            )));
        }
        if child.invocation_target != parent.invocation_target {
            return Err(EdvError::Capability(
                "delegated invocation_target diverges from parent".into(),
            ));
        }
        if child.controller != parent.invoker {
            return Err(EdvError::Capability(format!(
                "delegate controller {} is not the parent's invoker {}",
                child.controller, parent.invoker
            )));
        }
        if !parent.allowed_action.is_empty() {
            for a in &child.allowed_action {
                if !parent.allowed_action.contains(a) {
                    return Err(EdvError::Capability(format!(
                        "delegated action {a} not allowed by parent {}",
                        parent.id
                    )));
                }
            }
        }
    }

    let leaf = chain
        .last()
        .ok_or_else(|| EdvError::Capability("empty chain".into()))?;
    if invocation.invoker != leaf.invoker {
        return Err(EdvError::Unauthorized(
            invocation.action.clone(),
            leaf.invocation_target.clone(),
        ));
    }
    if !leaf.allows(&invocation.action) {
        return Err(EdvError::Unauthorized(
            invocation.action.clone(),
            leaf.invocation_target.clone(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn root_allows_listed_action() {
        let cap = Capability::root(
            "urn:cap:root",
            "https://vault/example",
            "did:example:alice",
            "did:example:alice",
            vec!["read".into(), "write".into()],
        );
        assert!(cap.allows("read"));
        assert!(!cap.allows("delete"));
    }

    #[test]
    fn delegation_restricts_actions() {
        let root = Capability::root(
            "urn:cap:root",
            "https://vault/example",
            "did:example:alice",
            "did:example:alice",
            vec!["read".into(), "write".into()],
        );
        let child = root
            .delegate(
                "urn:cap:child",
                "did:example:alice",
                "did:example:bob",
                vec!["read".into()],
            )
            .expect("delegate");
        assert_eq!(child.parent_capability, Some(root.id.clone()));
        assert!(child.allows("read"));
        assert!(!child.allows("write"));
    }

    #[test]
    fn delegation_cannot_widen() {
        let root = Capability::root(
            "urn:cap:root",
            "https://vault/example",
            "did:example:alice",
            "did:example:alice",
            vec!["read".into()],
        );
        let res = root.delegate(
            "urn:cap:child",
            "did:example:alice",
            "did:example:bob",
            vec!["read".into(), "delete".into()],
        );
        assert!(res.is_err());
    }

    #[test]
    fn chain_verifies_for_authorised_action() {
        let root = Capability::root(
            "urn:cap:root",
            "https://vault/example",
            "did:example:alice",
            "did:example:alice",
            vec!["read".into(), "write".into()],
        );
        let child = root
            .delegate(
                "urn:cap:child",
                "did:example:alice",
                "did:example:bob",
                vec!["read".into()],
            )
            .expect("delegate");
        let inv = Invocation::new(&child, "read");
        verify_chain(&[root, child], "did:example:alice", &inv).expect("verify");
    }

    #[test]
    fn chain_rejects_unauthorised_action() {
        let root = Capability::root(
            "urn:cap:root",
            "https://vault/example",
            "did:example:alice",
            "did:example:alice",
            vec!["read".into(), "write".into()],
        );
        let child = root
            .delegate(
                "urn:cap:child",
                "did:example:alice",
                "did:example:bob",
                vec!["read".into()],
            )
            .expect("delegate");
        let inv = Invocation::new(&child, "delete");
        let res = verify_chain(&[root, child], "did:example:alice", &inv);
        assert!(matches!(res, Err(EdvError::Unauthorized(_, _))));
    }

    #[test]
    fn chain_rejects_wrong_invoker() {
        let root = Capability::root(
            "urn:cap:root",
            "https://vault/example",
            "did:example:alice",
            "did:example:alice",
            vec!["read".into()],
        );
        let child = root
            .delegate(
                "urn:cap:child",
                "did:example:alice",
                "did:example:bob",
                vec!["read".into()],
            )
            .expect("delegate");
        let mut inv = Invocation::new(&child, "read");
        inv.invoker = "did:example:mallory".into();
        let res = verify_chain(&[root, child], "did:example:alice", &inv);
        assert!(matches!(res, Err(EdvError::Unauthorized(_, _))));
    }

    #[test]
    fn chain_rejects_expired_capability() {
        let mut root = Capability::root(
            "urn:cap:root",
            "https://vault/example",
            "did:example:alice",
            "did:example:alice",
            vec!["read".into()],
        );
        root.expires = Some(Utc::now() - chrono::Duration::seconds(60));
        let inv = Invocation::new(&root, "read");
        let res = verify_chain(&[root], "did:example:alice", &inv);
        assert!(matches!(res, Err(EdvError::Capability(_))));
    }
}
