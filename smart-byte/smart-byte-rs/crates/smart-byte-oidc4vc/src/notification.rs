//! Notification endpoint (OID4VCI draft 13 §10).
//!
//! After a successful credential issuance, the wallet MAY POST a
//! [`NotificationRequest`] to the issuer's `notification_endpoint` to
//! signal one of the standard events:
//!
//! * `credential_accepted` — the wallet successfully stored the credential.
//! * `credential_failure` — the wallet rejected or could not store it.
//! * `credential_deleted` — the wallet deleted a previously-stored credential.

use serde::{Deserialize, Serialize};

use crate::error::OidcError;

/// Standard `event` values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NotificationEvent {
    /// Credential was accepted by the wallet.
    CredentialAccepted,
    /// Credential issuance failed wallet-side.
    CredentialFailure,
    /// Credential was deleted from the wallet.
    CredentialDeleted,
}

impl NotificationEvent {
    /// Wire-form identifier.
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::CredentialAccepted => "credential_accepted",
            Self::CredentialFailure => "credential_failure",
            Self::CredentialDeleted => "credential_deleted",
        }
    }

    /// Parse from wire form.
    pub fn parse(s: &str) -> Result<Self, OidcError> {
        match s {
            "credential_accepted" => Ok(Self::CredentialAccepted),
            "credential_failure" => Ok(Self::CredentialFailure),
            "credential_deleted" => Ok(Self::CredentialDeleted),
            other => Err(OidcError::Notification(format!(
                "unknown event: {other}"
            ))),
        }
    }
}

/// Notification request body.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NotificationRequest {
    /// The notification id returned in [`crate::credential::CredentialResponse`].
    pub notification_id: String,
    /// Event identifier (`credential_accepted`, `credential_failure`,
    /// `credential_deleted`).
    pub event: String,
    /// Optional free-form description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub event_description: Option<String>,
}

impl NotificationRequest {
    /// Build a notification request.
    pub fn new(
        notification_id: impl Into<String>,
        event: NotificationEvent,
    ) -> Self {
        Self {
            notification_id: notification_id.into(),
            event: event.as_str().to_string(),
            event_description: None,
        }
    }

    /// Validate that `notification_id` is non-empty and `event` is one
    /// of the standard values.
    pub fn validate(&self) -> Result<(), OidcError> {
        if self.notification_id.is_empty() {
            return Err(OidcError::Notification(
                "notification_id is required".into(),
            ));
        }
        NotificationEvent::parse(&self.event)?;
        Ok(())
    }

    /// Parsed event.
    pub fn parsed_event(&self) -> Result<NotificationEvent, OidcError> {
        NotificationEvent::parse(&self.event)
    }
}

/// Notification error response body.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct NotificationErrorResponse {
    /// Error code (`invalid_notification_id`, `invalid_notification_request`).
    pub error: String,
    /// Optional description.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_description: Option<String>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn event_roundtrip() {
        for ev in [
            NotificationEvent::CredentialAccepted,
            NotificationEvent::CredentialFailure,
            NotificationEvent::CredentialDeleted,
        ] {
            assert_eq!(NotificationEvent::parse(ev.as_str()).unwrap(), ev);
        }
    }

    #[test]
    fn rejects_unknown_event() {
        assert!(NotificationEvent::parse("nope").is_err());
    }

    #[test]
    fn request_validates() {
        let r = NotificationRequest::new(
            "notif-1",
            NotificationEvent::CredentialAccepted,
        );
        r.validate().unwrap();
        assert_eq!(
            r.parsed_event().unwrap(),
            NotificationEvent::CredentialAccepted
        );
    }

    #[test]
    fn rejects_empty_id() {
        let r = NotificationRequest {
            notification_id: "".into(),
            event: NotificationEvent::CredentialAccepted.as_str().into(),
            event_description: None,
        };
        assert!(r.validate().is_err());
    }
}
