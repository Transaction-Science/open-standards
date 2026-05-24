//! Directory-Server abstraction and BIN-based routing.
//!
//! The DS is the scheme-operated middlebox that sits between the 3DS
//! Server (merchant side) and the ACS (issuer side). It performs
//! version-discovery on behalf of an issuer's BIN range, relays the
//! AReq/ARes to/from the ACS, and forwards the terminal RReq back to
//! the 3DS Server.
//!
//! Each card scheme runs its own DS:
//!
//! - **Visa** — Directory Server hosted by Visa.
//! - **Mastercard** — Identity Check / DS.
//! - **American Express** — SafeKey DS.
//! - **Discover / Diners** — ProtectBuy DS.
//! - **JCB** — J/Secure DS.
//! - **UnionPay** — UPI Secure Plus DS.
//! - **Mir** — Mir Accept DS.
//!
//! The [`DsRoute`] enum captures which DS we should route to based on
//! a PAN's IIN/BIN.

use serde::{Deserialize, Serialize};

use crate::error::{Error, Result};
use crate::message::{ARes, AReq, RRes, RReq};
use crate::version::ProtocolVersion;

/// Per-scheme DS-routing tag. Set by the [`route_for_pan`] helper from
/// the leading IIN digits.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DsRoute {
    /// Visa Directory Server.
    Visa,
    /// Mastercard DS.
    Mastercard,
    /// American Express SafeKey DS.
    Amex,
    /// Discover / Diners ProtectBuy DS.
    Discover,
    /// JCB J/Secure DS.
    Jcb,
    /// UnionPay UPI Secure Plus DS.
    UnionPay,
    /// Russian Mir Accept DS.
    Mir,
}

/// Result of a DS version-check call.
///
/// The DS responds with the protocol versions the issuer ACS supports
/// for the given PAN range. The 3DS Server then picks the highest
/// version it shares with the issuer (see
/// [`ProtocolVersion::preference_order`]).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VersionCheckResponse {
    /// All versions the issuer's ACS supports for this BIN range.
    pub supported_versions: Vec<ProtocolVersion>,
    /// ACS-side reference number for the range.
    pub acs_reference_number: String,
    /// URL the 3DS Method invocation should POST device data to,
    /// before the AReq is sent. Optional — older ACSes don't expose
    /// a 3DS-Method-URL.
    pub three_ds_method_url: Option<String>,
}

impl VersionCheckResponse {
    /// Returns the highest version both ends speak, or
    /// [`Error::NoCommonVersion`] if there's no overlap.
    pub fn negotiate(&self, requestor: &[ProtocolVersion]) -> Result<ProtocolVersion> {
        for pref in ProtocolVersion::preference_order() {
            if requestor.contains(&pref) && self.supported_versions.contains(&pref) {
                return Ok(pref);
            }
        }
        Err(Error::NoCommonVersion)
    }
}

/// Trait every DS adapter implements. The default scheme adapters live
/// in [`crate::visa_ds`], [`crate::mc_ds`], [`crate::amex_ds`],
/// [`crate::discover_ds`], [`crate::jcb_ds`].
#[async_trait::async_trait]
pub trait DirectoryServer: Send + Sync {
    /// Ask the DS which 3DS protocol versions the issuer's ACS
    /// supports for the given PAN range. The 3DS Server typically
    /// caches the result for ~24 hours per the spec's recommendation.
    async fn version_check(&self, card_range_pan: &str) -> Result<VersionCheckResponse>;

    /// Send the [`AReq`] and receive the issuer ACS's [`ARes`].
    async fn auth_request(&self, areq: &AReq) -> Result<ARes>;

    /// Forward the terminal [`RReq`] from the ACS back to the 3DS
    /// Server. Returns the [`RRes`] acknowledgement.
    async fn results_request(&self, rreq: &RReq) -> Result<RRes>;
}

// We use `async_trait` (the macro crate) to keep the async trait
// surface ergonomic on stable 1.95. RFC 3185-style native async-fn-in-
// trait is stabilised in 1.75+, but the dyn-compatible form still
// needs the macro for object safety.
pub use async_trait::async_trait;

/// Route a PAN to the appropriate DS based on its IIN/BIN.
///
/// The official scheme ranges are large and continuously updated; this
/// function implements the broad-stroke routing that covers > 99% of
/// production traffic. Operators who need scheme-issued routing tables
/// override [`DirectoryServerRouter`].
///
/// # Errors
/// Returns [`Error::InvalidPan`] if the PAN is empty or non-numeric.
/// Returns [`Error::NoDsRoute`] if no scheme matches.
pub fn route_for_pan(pan: &str) -> Result<DsRoute> {
    if pan.is_empty() || !pan.chars().all(|c| c.is_ascii_digit()) {
        return Err(Error::InvalidPan);
    }
    // IIN-prefix routing. Order matters: longer prefixes first.
    let p1 = pan.as_bytes().first().copied().unwrap_or(0);
    let p2 = pan.as_bytes().get(..2).and_then(|s| std::str::from_utf8(s).ok())
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);
    let p4 = pan.as_bytes().get(..4).and_then(|s| std::str::from_utf8(s).ok())
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);
    let p6 = pan.as_bytes().get(..6).and_then(|s| std::str::from_utf8(s).ok())
        .and_then(|s| s.parse::<u32>().ok())
        .unwrap_or(0);

    // Visa: 4XX...
    if p1 == b'4' {
        return Ok(DsRoute::Visa);
    }
    // Mastercard: 51-55, 2221-2720
    if (51..=55).contains(&p2) || (2221..=2720).contains(&p4) {
        return Ok(DsRoute::Mastercard);
    }
    // Amex: 34, 37
    if p2 == 34 || p2 == 37 {
        return Ok(DsRoute::Amex);
    }
    // Discover: 6011, 622126-622925, 644-649, 65
    if p4 == 6011
        || (622_126..=622_925).contains(&p6)
        || (644..=649).contains(&pan[..3].parse::<u32>().unwrap_or(0))
        || p2 == 65
    {
        return Ok(DsRoute::Discover);
    }
    // JCB: 3528-3589
    if (3528..=3589).contains(&p4) {
        return Ok(DsRoute::Jcb);
    }
    // UnionPay: 62, 81
    if p2 == 62 || p2 == 81 {
        return Ok(DsRoute::UnionPay);
    }
    // Mir: 2200-2204
    if (2200..=2204).contains(&p4) {
        return Ok(DsRoute::Mir);
    }

    Err(Error::NoDsRoute {
        bin: pan.chars().take(6).collect(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn visa_routes_to_visa() {
        assert_eq!(route_for_pan("4111111111111111").unwrap(), DsRoute::Visa);
    }

    #[test]
    fn mastercard_classic_routes_to_mc() {
        assert_eq!(
            route_for_pan("5500000000000004").unwrap(),
            DsRoute::Mastercard
        );
    }

    #[test]
    fn mastercard_two_series_routes_to_mc() {
        assert_eq!(
            route_for_pan("2221000000000009").unwrap(),
            DsRoute::Mastercard
        );
    }

    #[test]
    fn amex_routes_to_amex() {
        assert_eq!(route_for_pan("378282246310005").unwrap(), DsRoute::Amex);
    }

    #[test]
    fn discover_routes_to_discover() {
        assert_eq!(route_for_pan("6011111111111117").unwrap(), DsRoute::Discover);
    }

    #[test]
    fn jcb_routes_to_jcb() {
        assert_eq!(route_for_pan("3530111333300000").unwrap(), DsRoute::Jcb);
    }

    #[test]
    fn unionpay_routes_to_upi() {
        assert_eq!(route_for_pan("6200000000000005").unwrap(), DsRoute::UnionPay);
    }

    #[test]
    fn invalid_pan_errs() {
        assert!(matches!(route_for_pan(""), Err(Error::InvalidPan)));
        assert!(matches!(route_for_pan("abcd"), Err(Error::InvalidPan)));
    }

    #[test]
    fn version_negotiation_picks_highest() {
        let r = VersionCheckResponse {
            supported_versions: vec![ProtocolVersion::V2_1, ProtocolVersion::V2_2],
            acs_reference_number: "x".into(),
            three_ds_method_url: None,
        };
        let chosen = r
            .negotiate(&[ProtocolVersion::V2_3, ProtocolVersion::V2_2, ProtocolVersion::V2_1])
            .unwrap();
        assert_eq!(chosen, ProtocolVersion::V2_2);
    }

    #[test]
    fn version_negotiation_no_overlap_errs() {
        let r = VersionCheckResponse {
            supported_versions: vec![ProtocolVersion::V2_1],
            acs_reference_number: "x".into(),
            three_ds_method_url: None,
        };
        assert!(matches!(
            r.negotiate(&[ProtocolVersion::V2_3]),
            Err(Error::NoCommonVersion)
        ));
    }
}
