//! Challenge flow: CReq → ACS-hosted UI → CRes → RReq/RRes.
//!
//! When the ARes returns `transStatus == "C"` the cardholder must
//! complete a challenge before the transaction can authorize. The
//! ACS hosts the challenge UI at the URL returned in `ARes.acsURL`;
//! the 3DS Server POSTs the CReq there and the cardholder
//! browser/app renders the result.
//!
//! EMVCo defines three challenge modes:
//!
//! - [`ChallengeMode::Html`] — ACS returns base64-encoded HTML, the
//!   browser renders it inline (or inside a 3DS iframe).
//! - [`ChallengeMode::NativeApp`] — ACS returns SDK-side challenge
//!   data; the 3DS SDK renders a native UI.
//! - [`ChallengeMode::OutOfBand`] — ACS asks the cardholder to
//!   confirm on a separate channel (banking app push, OTP, hardware
//!   token); the merchant polls for completion (see
//!   [`crate::decoupled`]).
//!
//! ## Timeout
//!
//! EMVCo recommends a 5-minute total window for HTML and native
//! challenges. Decoupled challenges may run longer (default 10 min,
//! up to 24 h depending on issuer).

use chrono::{DateTime, Duration, Utc};
use serde::{Deserialize, Serialize};

use crate::auth_response::TransactionStatus;
use crate::error::{Error, Result};
use crate::message::{CReq, CRes};

/// Challenge mode the ACS indicated in the ARes.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChallengeMode {
    /// HTML challenge — ACS-hosted page rendered in the cardholder
    /// browser.
    Html,
    /// Native-app challenge — SDK renders a native widget.
    NativeApp,
    /// Out-of-band — cardholder confirms on a separate channel.
    OutOfBand,
}

/// A challenge request the 3DS Server is about to send.
///
/// Wraps a [`CReq`] together with the URL it must POST to and the
/// mode-specific UX context.
#[derive(Debug, Clone)]
pub struct ChallengeRequest {
    /// Where to POST.
    pub acs_url: String,
    /// Selected mode.
    pub mode: ChallengeMode,
    /// CReq payload.
    pub creq: CReq,
}

/// Outcome of a challenge cycle.
#[derive(Debug, Clone)]
pub struct ChallengeResult {
    /// Final transaction status.
    pub trans_status: TransactionStatus,
    /// Optional reason code for non-success.
    pub reason: Option<String>,
    /// Final CRes payload, if the cycle produced one.
    pub final_cres: Option<CRes>,
}

/// Server-side state for one in-flight challenge.
///
/// The orchestrator constructs one of these when an ARes returns
/// `transStatus == "C"` and tears it down once `submit_result` is
/// called.
#[derive(Debug, Clone)]
pub struct ChallengeSession {
    /// Echo of the 3DS Server transaction id.
    pub three_ds_server_trans_id: String,
    /// Echo of the ACS transaction id.
    pub acs_trans_id: String,
    /// Mode the ACS chose.
    pub mode: ChallengeMode,
    /// When the challenge started; used for timeout enforcement.
    pub started_at: DateTime<Utc>,
    /// Maximum wall-clock duration.
    pub timeout: Duration,
}

impl ChallengeSession {
    /// Construct a session with the EMVCo-recommended 5-minute default
    /// timeout.
    #[must_use]
    pub fn new(three_ds_server_trans_id: String, acs_trans_id: String, mode: ChallengeMode) -> Self {
        Self {
            three_ds_server_trans_id,
            acs_trans_id,
            mode,
            started_at: Utc::now(),
            timeout: Duration::minutes(5),
        }
    }

    /// Override the default 5-minute timeout (used for decoupled and
    /// long-running OOB).
    #[must_use]
    pub fn with_timeout(mut self, timeout: Duration) -> Self {
        self.timeout = timeout;
        self
    }

    /// True if the session has exceeded its timeout.
    #[must_use]
    pub fn is_expired(&self, now: DateTime<Utc>) -> bool {
        now - self.started_at > self.timeout
    }

    /// Build the initial [`CReq`] payload for this session.
    #[must_use]
    pub fn initial_creq(&self, message_version: &str, window_size: &str) -> CReq {
        CReq {
            message_version: message_version.to_owned(),
            three_ds_server_trans_id: self.three_ds_server_trans_id.clone(),
            acs_trans_id: self.acs_trans_id.clone(),
            challenge_window_size: Some(window_size.to_owned()),
            sdk_counter_s_to_a: Some("001".into()),
            challenge_data_entry: None,
            challenge_cancel: None,
            resend_challenge: None,
        }
    }

    /// Settle the session from a terminal CRes (`challengeCompletionInd
    /// == "Y"`). Returns [`Error::ChallengeTimeout`] if the deadline
    /// has already passed.
    pub fn settle(self, final_cres: CRes, now: DateTime<Utc>) -> Result<ChallengeResult> {
        if self.is_expired(now) {
            return Err(Error::ChallengeTimeout);
        }
        let trans_status = final_cres
            .trans_status
            .as_deref()
            .and_then(TransactionStatus::from_letter)
            .unwrap_or(TransactionStatus::NotAuthenticated);
        Ok(ChallengeResult {
            trans_status,
            reason: final_cres.trans_status_reason.clone(),
            final_cres: Some(final_cres),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn session() -> ChallengeSession {
        ChallengeSession::new("t-1".into(), "a-1".into(), ChallengeMode::Html)
    }

    #[test]
    fn default_timeout_is_five_minutes() {
        let s = session();
        assert_eq!(s.timeout, Duration::minutes(5));
    }

    #[test]
    fn initial_creq_contains_required_ids() {
        let s = session();
        let c = s.initial_creq("2.2.0", "05");
        assert_eq!(c.three_ds_server_trans_id, "t-1");
        assert_eq!(c.acs_trans_id, "a-1");
        assert_eq!(c.challenge_window_size.as_deref(), Some("05"));
        assert_eq!(c.sdk_counter_s_to_a.as_deref(), Some("001"));
    }

    #[test]
    fn settle_records_trans_status_from_cres() {
        let s = session();
        let cres = CRes {
            message_version: "2.2.0".into(),
            three_ds_server_trans_id: "t-1".into(),
            acs_trans_id: "a-1".into(),
            acs_counter_a_to_s: Some("001".into()),
            acs_html: None,
            challenge_completion_ind: Some("Y".into()),
            trans_status: Some("Y".into()),
            trans_status_reason: None,
            oob_app_url: None,
            oob_app_label: None,
            acs_decoupled_url: None,
        };
        let r = s.settle(cres, Utc::now()).unwrap();
        assert_eq!(r.trans_status, TransactionStatus::Authenticated);
    }

    #[test]
    fn settle_after_timeout_errs() {
        let mut s = session();
        s.started_at = Utc::now() - Duration::minutes(10);
        let cres = CRes {
            message_version: "2.2.0".into(),
            three_ds_server_trans_id: "t-1".into(),
            acs_trans_id: "a-1".into(),
            acs_counter_a_to_s: None,
            acs_html: None,
            challenge_completion_ind: Some("Y".into()),
            trans_status: Some("Y".into()),
            trans_status_reason: None,
            oob_app_url: None,
            oob_app_label: None,
            acs_decoupled_url: None,
        };
        assert!(matches!(s.settle(cres, Utc::now()), Err(Error::ChallengeTimeout)));
    }
}
