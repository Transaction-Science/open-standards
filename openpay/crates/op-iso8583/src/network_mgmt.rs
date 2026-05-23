//! ISO 8583 0800/0810 network-management messages.
//!
//! Card networks gate authorization traffic on a separately-tracked
//! network-management session. Before an acquirer can post 0100s, it
//! must:
//!
//! 1. **Sign on**: an `0800` with [`NetworkMgmtCode::SignOn`] establishes
//!    the session and supplies the working keys (PIN, MAC, encipherment).
//! 2. **Echo**: periodically (every 30–60 seconds, network-specific)
//!    send an `0800` [`NetworkMgmtCode::EchoTest`] and wait for the
//!    matching `0810` response. Failure to receive a response within
//!    the timeout triggers a re-sign-on.
//! 3. **Key change**: the network may push (or the acquirer may
//!    request) a `0800` [`NetworkMgmtCode::KeyChange`] mid-session to
//!    rotate MAC / PIN keys without dropping the link.
//! 4. **Cutover**: end-of-day batch boundary marker. `0800`
//!    [`NetworkMgmtCode::Cutover`].
//! 5. **Sign off**: graceful disconnect.
//!
//! All of these messages share the same on-the-wire shape: MTI 0800/0810,
//! DE 7 transmission date/time, DE 11 STAN, DE 70 network-management
//! information code (3 digits), plus DE 39 response code in the
//! response direction.
//!
//! This module provides typed constructors so the wire format never
//! gets a free-form numeric.

use crate::error::Result;
use crate::fields::FieldValue;
use crate::message::{Iso8583Message, Mti};

/// Network-management information code (DE 70).
///
/// The numeric values are the ISO 8583 standard codes used by all five
/// dialects this crate supports.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
#[repr(u16)]
pub enum NetworkMgmtCode {
    /// Sign on. Establishes the session and triggers key exchange.
    SignOn = 1,
    /// Sign off. Graceful disconnect.
    SignOff = 2,
    /// Cutover. End-of-day batch boundary.
    Cutover = 201,
    /// Echo test. Liveness check.
    EchoTest = 301,
    /// Key change. Working-key rotation (MAC / PIN encipherment).
    KeyChange = 101,
}

impl NetworkMgmtCode {
    /// Numeric form as a zero-padded 3-digit string (DE 70 is `n3`).
    #[must_use]
    pub fn as_str3(self) -> String {
        format!("{:03}", self as u16)
    }
}

/// Build a network-management request (`0800`) for the given code.
///
/// The caller supplies DE 7 (transmission date/time) and DE 11 (STAN);
/// these are message-identification fields the dialect doesn't fill in
/// for us (they come from the local link counter).
///
/// # Errors
/// None today; reserved for future field validation.
pub fn build_request(
    code: NetworkMgmtCode,
    transmission_date_time: &str,
    stan: &str,
) -> Result<Iso8583Message> {
    let mut m = Iso8583Message::new(Mti::NETWORK_REQUEST);
    m.set(7, FieldValue::Numeric(transmission_date_time.to_owned()));
    m.set(11, FieldValue::Numeric(stan.to_owned()));
    m.set(70, FieldValue::Numeric(code.as_str3()));
    Ok(m)
}

/// Build a network-management response (`0810`) to the given request.
///
/// Mirrors DE 7, DE 11, and DE 70 from the request and adds DE 39
/// with the response code (`"00"` = accepted).
///
/// # Errors
/// None today; reserved for future field validation.
pub fn build_response(
    request: &Iso8583Message,
    response_code: &str,
) -> Result<Iso8583Message> {
    let mut m = Iso8583Message::new(Mti::NETWORK_RESPONSE);
    if let Some(v) = request.get(7) {
        m.set(7, v.clone());
    }
    if let Some(v) = request.get(11) {
        m.set(11, v.clone());
    }
    if let Some(v) = request.get(70) {
        m.set(70, v.clone());
    }
    m.set(39, FieldValue::Text(response_code.to_owned()));
    Ok(m)
}

/// Convenience: sign-on request with the given date-time and STAN.
///
/// # Errors
/// As [`build_request`].
pub fn sign_on(date_time: &str, stan: &str) -> Result<Iso8583Message> {
    build_request(NetworkMgmtCode::SignOn, date_time, stan)
}

/// Convenience: sign-off request.
///
/// # Errors
/// As [`build_request`].
pub fn sign_off(date_time: &str, stan: &str) -> Result<Iso8583Message> {
    build_request(NetworkMgmtCode::SignOff, date_time, stan)
}

/// Convenience: echo test request.
///
/// # Errors
/// As [`build_request`].
pub fn echo_test(date_time: &str, stan: &str) -> Result<Iso8583Message> {
    build_request(NetworkMgmtCode::EchoTest, date_time, stan)
}

/// Convenience: key-change request.
///
/// # Errors
/// As [`build_request`].
pub fn key_change(date_time: &str, stan: &str) -> Result<Iso8583Message> {
    build_request(NetworkMgmtCode::KeyChange, date_time, stan)
}

/// Convenience: cutover request.
///
/// # Errors
/// As [`build_request`].
pub fn cutover(date_time: &str, stan: &str) -> Result<Iso8583Message> {
    build_request(NetworkMgmtCode::Cutover, date_time, stan)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fields::{Encoding, FieldSpec, LengthRule, default_catalog};

    fn catalog_with_de70() -> Vec<Option<FieldSpec>> {
        // Default catalog doesn't include DE 70; add it here for the tests.
        let mut c = default_catalog();
        c[70] = Some(FieldSpec {
            de: 70,
            label: "Network management information code",
            encoding: Encoding::Bcd,
            length: LengthRule::Fixed(3),
        });
        c
    }

    #[test]
    fn sign_on_round_trip() {
        let req = sign_on("0101120000", "000001").unwrap();
        assert_eq!(req.mti, Mti::NETWORK_REQUEST);
        let bytes = req.encode_with_catalog(&catalog_with_de70()).unwrap();
        let back = Iso8583Message::decode_with_catalog(&bytes, &catalog_with_de70()).unwrap();
        assert_eq!(back.mti, Mti::NETWORK_REQUEST);
        assert_eq!(back.get(70), req.get(70));
    }

    #[test]
    fn echo_test_code_is_301() {
        let m = echo_test("0101120000", "000002").unwrap();
        assert_eq!(m.get(70).and_then(FieldValue::as_numeric), Some("301"));
    }

    #[test]
    fn response_mirrors_request_fields() {
        let req = sign_on("0101120000", "000003").unwrap();
        let resp = build_response(&req, "00").unwrap();
        assert_eq!(resp.mti, Mti::NETWORK_RESPONSE);
        assert_eq!(resp.get(7), req.get(7));
        assert_eq!(resp.get(11), req.get(11));
        assert_eq!(resp.get(70), req.get(70));
        assert_eq!(resp.response_code(), Some("00"));
    }

    #[test]
    fn key_change_uses_101() {
        let m = key_change("0101120000", "000004").unwrap();
        assert_eq!(m.get(70).and_then(FieldValue::as_numeric), Some("101"));
    }

    #[test]
    fn cutover_uses_201() {
        let m = cutover("0101120000", "000005").unwrap();
        assert_eq!(m.get(70).and_then(FieldValue::as_numeric), Some("201"));
    }
}
