//! Decoupled authentication (RTS Article 4(c)).
//!
//! In decoupled mode the ACS returns a CRes with
//! `transStatus == "D"` and a `acsDecConURL` polling endpoint. The
//! merchant (3DS Server) polls the URL until the cardholder
//! completes authentication on a separate channel — typically a
//! banking-app push notification confirmed via biometric.
//!
//! Polling cadence and budget are configurable; defaults are aligned
//! with the EMVCo recommendation of 5 s minimum interval and the
//! issuer-declared `decoupledAuthMaxTime` (in minutes).

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::auth_response::TransactionStatus;
use crate::error::{Error, Result};

/// Per-poll result returned by the ACS decoupled polling endpoint.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum DecoupledPollResult {
    /// Cardholder has not responded yet; keep polling.
    Pending,
    /// Cardholder approved.
    Approved {
        /// Cryptogram for the acquirer auth message.
        authentication_value: String,
        /// ECI.
        eci: String,
    },
    /// Cardholder declined or timed out at the ACS.
    Declined {
        /// Optional reason supplied by the ACS.
        reason: Option<String>,
    },
}

/// Active decoupled-authentication session.
#[derive(Debug, Clone)]
pub struct DecoupledSession {
    /// ACS-supplied polling URL.
    pub polling_url: String,
    /// Echo of the 3DS Server transaction id (for log correlation).
    pub three_ds_server_trans_id: String,
    /// Cardholder window (minutes) the ACS committed to.
    pub decoupled_auth_max_time: u32,
    /// Polling cadence.
    pub poll_interval: Duration,
    /// Maximum number of polls before giving up. Computed from
    /// `decoupled_auth_max_time * 60 / poll_interval` by [`Self::new`].
    pub max_polls: u32,
}

impl DecoupledSession {
    /// Construct with the EMVCo-recommended 5 s interval.
    #[must_use]
    pub fn new(polling_url: String, three_ds_server_trans_id: String, max_minutes: u32) -> Self {
        let interval = Duration::from_secs(5);
        let max_polls = (u64::from(max_minutes) * 60 / interval.as_secs()) as u32;
        Self {
            polling_url,
            three_ds_server_trans_id,
            decoupled_auth_max_time: max_minutes,
            poll_interval: interval,
            max_polls,
        }
    }

    /// Drive the polling loop with a caller-supplied async poller. The
    /// poller is called up to `self.max_polls` times; the loop returns
    /// as soon as a non-`Pending` result arrives.
    pub async fn run<F, Fut>(&self, mut poll: F) -> Result<DecoupledPollResult>
    where
        F: FnMut(&str) -> Fut + Send,
        Fut: std::future::Future<Output = Result<DecoupledPollResult>> + Send,
    {
        for _ in 0..self.max_polls {
            match poll(&self.polling_url).await? {
                DecoupledPollResult::Pending => {
                    tokio::time::sleep(self.poll_interval).await;
                }
                other => return Ok(other),
            }
        }
        Err(Error::DecoupledTimeout {
            polls: self.max_polls,
        })
    }

    /// Map an [`DecoupledPollResult::Approved`] into the matching
    /// [`TransactionStatus`] for downstream rail emission.
    #[must_use]
    pub const fn status_for(r: &DecoupledPollResult) -> TransactionStatus {
        match r {
            DecoupledPollResult::Approved { .. } => TransactionStatus::Authenticated,
            DecoupledPollResult::Declined { .. } => TransactionStatus::NotAuthenticated,
            DecoupledPollResult::Pending => TransactionStatus::ChallengeRequiredDecoupled,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{Arc, Mutex};

    fn session(max_minutes: u32) -> DecoupledSession {
        DecoupledSession::new(
            "https://acs.example/poll/abc".into(),
            "t-decoupled".into(),
            max_minutes,
        )
    }

    #[tokio::test]
    async fn poll_returns_after_three_pending_then_approved() {
        let s = DecoupledSession {
            poll_interval: Duration::from_millis(1),
            max_polls: 10,
            ..session(1)
        };
        let count = Arc::new(Mutex::new(0_u32));
        let result = s
            .run(|_url| {
                let count = Arc::clone(&count);
                async move {
                    let mut c = count.lock().unwrap();
                    *c += 1;
                    if *c < 3 {
                        Ok(DecoupledPollResult::Pending)
                    } else {
                        Ok(DecoupledPollResult::Approved {
                            authentication_value: "CAVV==".into(),
                            eci: "05".into(),
                        })
                    }
                }
            })
            .await
            .unwrap();
        assert_eq!(
            result,
            DecoupledPollResult::Approved {
                authentication_value: "CAVV==".into(),
                eci: "05".into(),
            }
        );
        assert_eq!(*count.lock().unwrap(), 3);
    }

    #[tokio::test]
    async fn poll_times_out_after_budget() {
        let s = DecoupledSession {
            poll_interval: Duration::from_millis(1),
            max_polls: 2,
            ..session(1)
        };
        let result = s.run(|_| async { Ok(DecoupledPollResult::Pending) }).await;
        assert!(matches!(result, Err(Error::DecoupledTimeout { polls: 2 })));
    }

    #[test]
    fn status_for_maps_correctly() {
        assert_eq!(
            DecoupledSession::status_for(&DecoupledPollResult::Approved {
                authentication_value: "x".into(),
                eci: "05".into(),
            }),
            TransactionStatus::Authenticated
        );
        assert_eq!(
            DecoupledSession::status_for(&DecoupledPollResult::Declined { reason: None }),
            TransactionStatus::NotAuthenticated
        );
        assert_eq!(
            DecoupledSession::status_for(&DecoupledPollResult::Pending),
            TransactionStatus::ChallengeRequiredDecoupled
        );
    }
}
