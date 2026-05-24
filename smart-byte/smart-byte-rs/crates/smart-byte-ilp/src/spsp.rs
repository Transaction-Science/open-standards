//! SPSP — Simple Payment Setup Protocol.
//!
//! SPSP is the HTTPS-based receiver-discovery step that turns a
//! human-friendly payment pointer like `$wallet.example/alice` into the
//! `(destination_address, shared_secret)` pair a STREAM sender needs.
//!
//! Wire format: the receiver's URL is fetched with
//! `Accept: application/spsp4+json`, and the response is a small JSON
//! document with the destination ILP address (base64-encoded shared
//! secret optional).

use crate::address::Address;
use crate::error::{IlpError, Result};
use serde::{Deserialize, Serialize};

/// A resolved payment pointer. Implementations turn the human-friendly
/// `$wallet.example/alice` into an HTTPS URL by stripping the leading
/// `$`, splitting on the first `/`, and producing
/// `https://wallet.example/.well-known/pay` or
/// `https://wallet.example/alice` respectively.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SpspQuery {
    /// The HTTPS URL the receiver will be polled at.
    pub url: String,
}

impl SpspQuery {
    /// Parse a `$pointer` into an HTTPS URL.
    pub fn from_payment_pointer(pointer: &str) -> Result<Self> {
        let p = pointer
            .strip_prefix('$')
            .ok_or_else(|| IlpError::Spsp("payment pointer must start with $".into()))?;
        let (host, path) = match p.split_once('/') {
            Some((h, rest)) => (h, format!("/{rest}")),
            None => (p, "/.well-known/pay".to_string()),
        };
        if host.is_empty() {
            return Err(IlpError::Spsp("payment pointer host empty".into()));
        }
        Ok(Self {
            url: format!("https://{host}{path}"),
        })
    }
}

/// SPSP receiver response document.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct SpspResponse {
    /// Destination ILP address for the receiver.
    pub destination_account: String,
    /// 32-byte STREAM shared secret, base64-encoded.
    pub shared_secret: String,
    /// Asset-detail block describing what the receiver is denominated in.
    pub receiver_info: Option<ReceiverInfo>,
}

/// Receiver-side asset information.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ReceiverInfo {
    /// ISO-4217-style asset code (`USD`, `XRP`, `EUR`).
    pub asset_code: String,
    /// Asset scale — 10^-scale is one unit (e.g. cents have scale 2).
    pub asset_scale: u8,
}

impl SpspResponse {
    /// Parse and validate the receiver's destination ILP address.
    pub fn validated_destination(&self) -> Result<Address> {
        Address::parse(&self.destination_account)
    }

    /// Encode as canonical JSON.
    pub fn to_json(&self) -> Result<String> {
        Ok(serde_json::to_string(self)?)
    }

    /// Parse from JSON.
    pub fn from_json(s: &str) -> Result<Self> {
        Ok(serde_json::from_str(s)?)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pointer_with_no_path() {
        let q = SpspQuery::from_payment_pointer("$wallet.example").unwrap();
        assert_eq!(q.url, "https://wallet.example/.well-known/pay");
    }

    #[test]
    fn pointer_with_path() {
        let q = SpspQuery::from_payment_pointer("$wallet.example/alice").unwrap();
        assert_eq!(q.url, "https://wallet.example/alice");
    }

    #[test]
    fn pointer_must_have_dollar() {
        assert!(SpspQuery::from_payment_pointer("wallet.example").is_err());
    }

    #[test]
    fn response_roundtrip() {
        let resp = SpspResponse {
            destination_account: "g.us.bank.alice".into(),
            shared_secret: "AAAA".into(),
            receiver_info: Some(ReceiverInfo {
                asset_code: "USD".into(),
                asset_scale: 6,
            }),
        };
        let json = resp.to_json().unwrap();
        let back = SpspResponse::from_json(&json).unwrap();
        assert_eq!(resp, back);
        assert!(back.validated_destination().is_ok());
    }
}
