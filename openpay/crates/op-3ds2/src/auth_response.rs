//! Authentication-response primitives: `transStatus`, ECI, CAVV.
//!
//! `transStatus` is the EMVCo-coded outcome a 3DS authentication
//! produces. Once the issuer ACS settles the cardholder challenge, the
//! ARes (or final CRes for challenge flows) carries one of the
//! letters below. The ECI and CAVV that the acquirer forwards in the
//! ISO 8583 / ISO 20022 authorization request derive from this
//! letter plus the card scheme.
//!
//! ## `transStatus` values (EMVCo Table A.6)
//!
//! - `Y` — Authentication / Account Verification Successful.
//! - `N` — Not Authenticated / Account Not Verified.
//! - `U` — Authentication / Account Verification could not be performed.
//! - `A` — Attempts processing performed; not authenticated.
//! - `C` — Challenge Required (cardholder must complete a challenge).
//! - `D` — Challenge Required; decoupled authentication.
//! - `R` — Authentication / Account Verification Rejected.
//! - `I` — Informational only; 3RI / data-only.

use base64::Engine;
use serde::{Deserialize, Serialize};

use crate::directory_server::DsRoute;
use crate::error::{Error, Result};

/// `transStatus` enum.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum TransactionStatus {
    /// `Y` — authenticated. ECI 05 / 02 (scheme-dependent), CAVV
    /// present.
    Authenticated,
    /// `N` — not authenticated. No CAVV.
    NotAuthenticated,
    /// `A` — attempts (stand-in). ECI 06 / 01, CAVV present (proxy).
    AttemptStanin,
    /// `R` — rejected by the issuer. No CAVV.
    Rejected,
    /// `C` — challenge required (in-band).
    ChallengeRequired,
    /// `D` — challenge required (decoupled).
    ChallengeRequiredDecoupled,
    /// `A` after a successful attempt cycle. Synonym for `AttemptStanin`
    /// used by some integrators.
    AttemptedSuccessful,
    /// `I` — informational only (3RI / data-only). No ECI, no CAVV.
    InfoOnly,
}

impl TransactionStatus {
    /// Wire-format single-letter code.
    #[must_use]
    pub const fn as_letter(self) -> &'static str {
        match self {
            Self::Authenticated => "Y",
            Self::NotAuthenticated => "N",
            Self::AttemptStanin | Self::AttemptedSuccessful => "A",
            Self::Rejected => "R",
            Self::ChallengeRequired => "C",
            Self::ChallengeRequiredDecoupled => "D",
            Self::InfoOnly => "I",
        }
    }

    /// Parse a wire-letter back into a [`TransactionStatus`]. Returns
    /// `None` for unknown letters.
    #[must_use]
    pub fn from_letter(s: &str) -> Option<Self> {
        match s {
            "Y" => Some(Self::Authenticated),
            "N" => Some(Self::NotAuthenticated),
            "A" => Some(Self::AttemptStanin),
            "R" => Some(Self::Rejected),
            "C" => Some(Self::ChallengeRequired),
            "D" => Some(Self::ChallengeRequiredDecoupled),
            "I" => Some(Self::InfoOnly),
            _ => None,
        }
    }
}

/// Electronic Commerce Indicator newtype.
///
/// Carried in the acquirer auth request as a two-digit field. Different
/// schemes encode the same semantics with different values; see
/// [`eci_for`] for the per-scheme mapping.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Eci(pub String);

/// Cardholder Authentication Verification Value (CAVV) /
/// Accountholder Authentication Value (AAV) / Amex Verification
/// Value (AEVV).
///
/// All three scheme names map to the same conceptual bag: a base64
/// blob the acquirer forwards to the issuer for cryptographic
/// verification. We model it as a thin newtype over `Vec<u8>` so the
/// codec is unambiguous about whether we hold raw or base64 bytes.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Cavv(Vec<u8>);

impl Cavv {
    /// Construct from raw bytes.
    #[must_use]
    pub fn from_bytes(b: Vec<u8>) -> Self {
        Self(b)
    }

    /// Decode from the standard base64 string that travels on the wire.
    pub fn decode_base64(s: &str) -> Result<Self> {
        base64::engine::general_purpose::STANDARD
            .decode(s)
            .map(Self)
            .map_err(|_| Error::InvalidCryptogram)
    }

    /// Encode back to the wire base64 form.
    #[must_use]
    pub fn encode_base64(&self) -> String {
        base64::engine::general_purpose::STANDARD.encode(&self.0)
    }

    /// Borrow the raw byte slice.
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

/// Per-scheme ECI mapping for a given `transStatus` letter.
///
/// Visa / Amex / Discover / JCB use the `05` / `06` coding;
/// Mastercard uses the `02` / `01` coding. All other letters map to
/// `"07"` (non-3DS / no liability shift) by convention.
#[must_use]
pub const fn eci_for(scheme: DsRoute, trans_status: &str) -> &'static str {
    let bytes = trans_status.as_bytes();
    let first = if bytes.is_empty() { 0 } else { bytes[0] };
    match scheme {
        DsRoute::Mastercard => match first {
            b'Y' => "02",
            b'A' => "01",
            _ => "00",
        },
        // Visa / Amex / Discover / JCB / UnionPay / Mir all use 05/06.
        _ => match first {
            b'Y' => "05",
            b'A' => "06",
            _ => "07",
        },
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trans_status_round_trips_via_letter() {
        for ts in [
            TransactionStatus::Authenticated,
            TransactionStatus::NotAuthenticated,
            TransactionStatus::AttemptStanin,
            TransactionStatus::Rejected,
            TransactionStatus::ChallengeRequired,
            TransactionStatus::ChallengeRequiredDecoupled,
            TransactionStatus::InfoOnly,
        ] {
            let letter = ts.as_letter();
            let back = TransactionStatus::from_letter(letter).unwrap();
            assert_eq!(back.as_letter(), letter);
        }
    }

    #[test]
    fn visa_eci_05_on_success_06_on_attempts() {
        assert_eq!(eci_for(DsRoute::Visa, "Y"), "05");
        assert_eq!(eci_for(DsRoute::Visa, "A"), "06");
        assert_eq!(eci_for(DsRoute::Visa, "N"), "07");
    }

    #[test]
    fn mastercard_eci_02_on_success_01_on_attempts() {
        assert_eq!(eci_for(DsRoute::Mastercard, "Y"), "02");
        assert_eq!(eci_for(DsRoute::Mastercard, "A"), "01");
        assert_eq!(eci_for(DsRoute::Mastercard, "N"), "00");
    }

    #[test]
    fn amex_discover_jcb_match_visa_coding() {
        assert_eq!(eci_for(DsRoute::Amex, "Y"), "05");
        assert_eq!(eci_for(DsRoute::Discover, "Y"), "05");
        assert_eq!(eci_for(DsRoute::Jcb, "Y"), "05");
    }

    #[test]
    fn cavv_round_trip() {
        let raw = vec![0xDE, 0xAD, 0xBE, 0xEF];
        let c = Cavv::from_bytes(raw.clone());
        let s = c.encode_base64();
        let back = Cavv::decode_base64(&s).unwrap();
        assert_eq!(back.as_bytes(), &raw);
    }

    #[test]
    fn cavv_rejects_invalid_base64() {
        assert!(matches!(
            Cavv::decode_base64("!!!not-base-64!!!"),
            Err(Error::InvalidCryptogram)
        ));
    }
}
