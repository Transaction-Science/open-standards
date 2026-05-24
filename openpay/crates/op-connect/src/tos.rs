//! Terms-of-service acceptance tracking.
//!
//! Every Connect / MarketPay flow requires the sub-merchant to
//! affirmatively accept the platform's terms before activation. That
//! acceptance must be evidentiary: regulators (and the platform's
//! acquirer) want a record of *who* accepted *what version* *from
//! where* and *when*. The four-tuple below covers the standard
//! evidentiary bar:
//!
//! - **IP** of the accepting client at acceptance time;
//! - **User-Agent** of the accepting browser / app;
//! - **Timestamp** in UTC;
//! - **Version hash** — the SHA-256 (or any stable hash) of the
//!   accepted ToS text, so the platform can prove what was on screen.
//!
//! Idempotency: re-submitting the same `(version_hash, ip, ua,
//! accepted_at)` tuple is a no-op — operators replaying a webhook
//! shouldn't pollute the audit trail. [`AcceptanceStore::record`]
//! returns `Inserted` on first submission and `Duplicate` on replay.

use std::collections::{BTreeMap, HashSet};

use chrono::{DateTime, Utc};
use serde::{Deserialize, Serialize};

use crate::account::AccountId;
use crate::error::{Error, Result};

/// A single ToS acceptance record.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TosAcceptance {
    /// IP address of the accepting client (IPv4 or IPv6 textual form).
    pub ip: String,
    /// User-Agent header value verbatim.
    pub user_agent: String,
    /// UTC timestamp at which acceptance was captured.
    pub accepted_at: DateTime<Utc>,
    /// Stable hash of the ToS version that was on screen.
    pub version_hash: String,
}

impl TosAcceptance {
    /// Validate basic field shape (non-empty hash + UA + IP).
    ///
    /// # Errors
    /// [`Error::TosInvalid`].
    pub fn validate(&self) -> Result<()> {
        if self.version_hash.is_empty() {
            return Err(Error::TosInvalid {
                reason: "empty version_hash".into(),
            });
        }
        if self.ip.is_empty() {
            return Err(Error::TosInvalid {
                reason: "empty ip".into(),
            });
        }
        if self.user_agent.is_empty() {
            return Err(Error::TosInvalid {
                reason: "empty user_agent".into(),
            });
        }
        Ok(())
    }
}

/// Outcome of [`AcceptanceStore::record`].
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RecordOutcome {
    /// First time we saw this acceptance for this account.
    Inserted,
    /// Same tuple has already been recorded — no-op.
    Duplicate,
}

/// Append-only store of ToS acceptances per account.
#[derive(Debug, Clone, Default)]
pub struct AcceptanceStore {
    by_account: BTreeMap<AccountId, Vec<TosAcceptance>>,
    /// Hash of (acct, version, ip, ua, accepted_at) to detect replays
    /// without scanning the per-account vec.
    seen: HashSet<String>,
}

impl AcceptanceStore {
    /// Fresh empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Record an acceptance, idempotent on the full tuple.
    ///
    /// # Errors
    /// [`Error::TosInvalid`] on any shape failure.
    pub fn record(
        &mut self,
        acct: &AccountId,
        acceptance: TosAcceptance,
    ) -> Result<RecordOutcome> {
        acceptance.validate()?;
        let key = format!(
            "{}|{}|{}|{}|{}",
            acct.0,
            acceptance.version_hash,
            acceptance.ip,
            acceptance.user_agent,
            acceptance.accepted_at.timestamp_micros()
        );
        if !self.seen.insert(key) {
            return Ok(RecordOutcome::Duplicate);
        }
        self.by_account
            .entry(acct.clone())
            .or_default()
            .push(acceptance);
        Ok(RecordOutcome::Inserted)
    }

    /// Most-recent acceptance for an account, if any.
    #[must_use]
    pub fn latest(&self, acct: &AccountId) -> Option<&TosAcceptance> {
        self.by_account.get(acct).and_then(|v| v.last())
    }

    /// Full acceptance history for an account.
    #[must_use]
    pub fn history(&self, acct: &AccountId) -> &[TosAcceptance] {
        self.by_account
            .get(acct)
            .map(Vec::as_slice)
            .unwrap_or(&[])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample(version: &str, ts: DateTime<Utc>) -> TosAcceptance {
        TosAcceptance {
            ip: "203.0.113.42".into(),
            user_agent: "Mozilla/5.0 (Macintosh)".into(),
            accepted_at: ts,
            version_hash: version.into(),
        }
    }

    #[test]
    fn first_record_inserts() {
        let mut store = AcceptanceStore::new();
        let acct = AccountId("acct_x".into());
        let out = store
            .record(&acct, sample("v1-hash", Utc::now()))
            .expect("ok");
        assert_eq!(out, RecordOutcome::Inserted);
        assert!(store.latest(&acct).is_some());
    }

    #[test]
    fn replay_is_idempotent() {
        let mut store = AcceptanceStore::new();
        let acct = AccountId("acct_y".into());
        let ts = Utc::now();
        let a = sample("v1-hash", ts);
        store.record(&acct, a.clone()).expect("ok");
        let out = store.record(&acct, a).expect("ok");
        assert_eq!(out, RecordOutcome::Duplicate);
        assert_eq!(store.history(&acct).len(), 1);
    }

    #[test]
    fn different_version_records_separately() {
        let mut store = AcceptanceStore::new();
        let acct = AccountId("acct_z".into());
        let ts = Utc::now();
        store.record(&acct, sample("v1", ts)).expect("ok");
        store.record(&acct, sample("v2", ts)).expect("ok");
        assert_eq!(store.history(&acct).len(), 2);
    }

    #[test]
    fn empty_hash_rejected() {
        let mut store = AcceptanceStore::new();
        let acct = AccountId("acct_w".into());
        let err = store
            .record(&acct, sample("", Utc::now()))
            .expect_err("empty hash");
        assert!(matches!(err, Error::TosInvalid { .. }));
    }
}
