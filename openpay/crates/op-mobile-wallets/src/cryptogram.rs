//! Common cryptogram + ECI primitives shared by all three wallets.
//!
//! Each wallet emits a per-transaction cryptogram bound to the DPAN and
//! the transaction amount. The wire format differs by network (Visa
//! TAVV, Mastercard AAV/UCAF, Amex AEVV), but the **role** is the same:
//! the cryptogram authenticates the *use* of the network token at
//! authorization time.
//!
//! We reuse [`op_rails_card::network_token::Cryptogram`] as the
//! normalized over-the-wire shape so a decryption pipeline can feed
//! straight into the network-token rail without re-shaping. The
//! types below are the *wallet-side* metadata that doesn't survive
//! into the normalized cryptogram — the ECI string and the 3DS2
//! indicator — but which the rail driver needs to read off the
//! decrypted token before forwarding.

use serde::{Deserialize, Serialize};

pub use op_rails_card::network_token::Cryptogram;

/// Electronic Commerce Indicator (ECI).
///
/// Wallets emit network-specific ECI values that downstream
/// authorization logic forwards verbatim to the acquirer:
///
/// - Visa wallet-authenticated: `"05"`
/// - Visa attempted-authentication: `"06"`
/// - Mastercard wallet-authenticated: `"02"`
/// - Mastercard attempted-authentication: `"01"`
/// - Amex wallet-authenticated: `"05"`
///
/// The decryptor preserves the string the wallet emitted — we do not
/// re-derive ECI from cryptogram material. Translation between
/// network-specific and rail-specific ECI ranges happens at the
/// rail driver, not here.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct EciIndicator(String);

impl EciIndicator {
    /// Wrap a wallet-emitted ECI string.
    #[must_use]
    pub fn new(eci: impl Into<String>) -> Self {
        Self(eci.into())
    }

    /// Borrow the ECI string.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }

    /// True if this ECI corresponds to the network's "fully
    /// authenticated" code (Visa `"05"` / Mastercard `"02"` /
    /// Amex `"05"`). Used by rail drivers to assert the
    /// network-token liability-shift posture before forwarding.
    #[must_use]
    pub fn is_fully_authenticated(&self) -> bool {
        matches!(self.0.as_str(), "05" | "02")
    }
}

/// 3-D Secure 2 indicator extracted from the wallet payload.
///
/// Apple Pay sets this implicitly via the `paymentDataType` field
/// (`"3DSecure"` vs `"EMV"`); Google Pay surfaces it as a
/// `messageId`-adjacent field; Samsung Pay matches Apple Pay's shape.
/// We normalize across them into a small enum so rail code branches
/// on the type system instead of free-form strings.
#[derive(Copy, Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ThreeDsIndicator {
    /// The wallet has performed 3-D Secure 2 authentication for this
    /// transaction; the cryptogram includes the device-attested AAV
    /// (Mastercard) / CAVV (Visa) material.
    Authenticated,
    /// The wallet has *attempted* 3-D Secure but the issuer's ACS
    /// returned `attempts` rather than full authentication. Liability
    /// shift is still in effect on most networks; surfaced separately
    /// because some rail-side risk policies treat it differently.
    Attempted,
    /// Wallet-authenticated via EMV cryptogram path (Apple Pay
    /// `"EMV"`), not via 3-D Secure. Predominant for in-store
    /// (Tap-to-Pay) and most in-app transactions.
    EmvCryptogram,
}
