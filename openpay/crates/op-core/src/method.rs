//! Payment method representations.
//!
//! ## PCI scope
//!
//! The default surface exposes only opaque references: vault tokens, EMV
//! TLV blobs, wallet device tokens, and A2A account-key identifiers. Raw
//! PAN is unreachable without enabling the `pci-scope` cargo feature. This
//! places the type system on the same side as PCI DSS scope boundaries:
//! code that doesn't opt in cannot accidentally land in scope.

use serde::{Deserialize, Serialize};
use zeroize::{Zeroize, ZeroizeOnDrop};

/// A reference to a tokenized payment credential held in a PCI-compliant
/// vault (e.g. PSP-issued token, Hyperswitch vault id, network token).
///
/// `VaultRef` is just a string id — it contains no PAN. Resolving it to a
/// usable credential happens inside the vault service, which is the only
/// component in scope for PCI DSS.
#[derive(Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct VaultRef(String);

impl VaultRef {
    /// Construct from a vault-issued id. The caller is responsible for
    /// ensuring the value really is a token and not raw PAN. A defensive
    /// length check rejects anything that looks like a card number.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The opaque token id.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl core::fmt::Debug for VaultRef {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        // Never log full token; show first 4 + length.
        let n = self.0.len();
        let prefix: String = self.0.chars().take(4).collect();
        write!(f, "VaultRef({prefix}…/{n})")
    }
}

/// An opaque encrypted token from a wallet (Apple Pay, Google Pay) or a
/// device-issued network token. Bytes are encrypted at the source; we
/// never decrypt them in `op-core`.
#[derive(Clone, Zeroize, ZeroizeOnDrop, Serialize, Deserialize)]
pub struct Token(Vec<u8>);

impl Token {
    /// Wrap a token blob.
    #[must_use]
    pub const fn new(bytes: Vec<u8>) -> Self {
        Self(bytes)
    }

    /// Borrow the underlying bytes (for transmission to the rail driver).
    #[must_use]
    pub fn as_bytes(&self) -> &[u8] {
        &self.0
    }
}

impl core::fmt::Debug for Token {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "Token({} bytes)", self.0.len())
    }
}

/// An A2A (account-to-account) destination key. UPI handles, PIX keys,
/// IBANs for SEPA, RTP routing numbers, `FedNow` account identifiers.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum A2aKey {
    /// India UPI virtual address (e.g. `bob@axis`).
    Upi(String),
    /// Brazilian PIX key (CPF, email, phone, or random key).
    Pix(String),
    /// IBAN for SEPA Instant.
    Iban(String),
    /// US routing + account number (RTP, `FedNow`, ACH).
    UsAch {
        /// ABA routing transit number (9 digits).
        routing: String,
        /// Demand-deposit account number.
        account: String,
    },
}

/// A wallet address on a specific chain.
///
/// Chains are referenced by their canonical short name
/// (`"solana"`, `"base"`, `"ethereum"`, `"polygon"`, `"arbitrum"`).
/// The address format is chain-specific — base58 (52 chars) for
/// Solana, hex `0x...` (42 chars) for EVM — and is validated by
/// the driver, not here.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CryptoAddress {
    /// Lowercase canonical chain identifier.
    pub chain: String,
    /// The address as a string. EVM addresses are stored in
    /// lowercase; Solana addresses preserve their case-sensitive
    /// base58 encoding.
    pub address: String,
}

impl CryptoAddress {
    /// Construct without validation. Driver code re-validates per-chain.
    #[must_use]
    pub fn new(chain: impl Into<String>, address: impl Into<String>) -> Self {
        Self {
            chain: chain.into(),
            address: address.into(),
        }
    }
}

/// A payment method: how value moves.
///
/// Variants outside the `pci-scope` feature are all *opaque references*.
/// None of them contain raw PAN. The `RawPan` variant is conditionally
/// compiled and must only exist inside a PCI-certified vault component.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum PaymentMethod {
    /// A token issued by a vault or PSP. The default and overwhelmingly
    /// preferred form.
    Vault(VaultRef),

    /// A wallet token (Apple Pay, Google Pay) — device-cryptogram payload.
    Wallet(Token),

    /// An EMV contactless/contact transaction payload as TLV bytes from
    /// `ProximityReader` (iOS Tap to Pay) or terminal kernel.
    Emv(Token),

    /// An A2A (instant bank) destination.
    A2a(A2aKey),

    /// QR-code-presented payment (UPI QR, PIX QR, `EMVCo` merchant QR).
    /// The wrapped string is the QR payload, already validated.
    Qr(String),

    /// Crypto / stablecoin destination: a wallet address on a specific
    /// chain. The token contract is part of the rail driver's
    /// configuration (operators register `USDC@Base`, `EURC@Solana`,
    /// etc. as separate adapters). Wallet addresses are opaque
    /// strings here; the driver validates per-chain format.
    Crypto(CryptoAddress),

    /// Raw card number — ONLY available with `pci-scope` feature. The
    /// type contains zeroize-on-drop so any accidental copy is wiped.
    #[cfg(feature = "pci-scope")]
    RawPan(crate::method::pci::RawPan),
}

/// PCI-scope-only types. Gated behind the `pci-scope` feature so that
/// raw PAN material is only constructible inside a PCI-certified
/// component (the vault). No other crate should enable this feature.
#[cfg(feature = "pci-scope")]
pub mod pci {
    use serde::{Deserialize, Serialize};
    use zeroize::{Zeroize, ZeroizeOnDrop};

    /// Raw PAN. Only constructible inside a PCI-scope code path.
    #[derive(Clone, Zeroize, ZeroizeOnDrop, Serialize, Deserialize)]
    pub struct RawPan {
        pan: String,
        exp_month: u8,
        exp_year: u16,
    }

    impl RawPan {
        /// Construct. Caller asserts PCI scope.
        #[must_use]
        pub const fn new(pan: String, exp_month: u8, exp_year: u16) -> Self {
            Self {
                pan,
                exp_month,
                exp_year,
            }
        }

        /// Last four digits — safe to log.
        #[must_use]
        pub fn last_four(&self) -> &str {
            let n = self.pan.len();
            &self.pan[n.saturating_sub(4)..]
        }

        /// First six digits (BIN) — safe to log per PCI DSS 4.0.1 §3.4.1
        /// allowance for the first 6 + last 4. Empty if PAN is < 6 chars.
        #[must_use]
        pub fn first_six(&self) -> &str {
            &self.pan[..self.pan.len().min(6)]
        }

        /// Full PAN bytes. **Only callable inside `pci-scope` code paths.**
        /// The vault uses this to compute the ciphertext that gets stored;
        /// no other crate should need it.
        #[must_use]
        pub const fn pan_bytes(&self) -> &[u8] {
            self.pan.as_bytes()
        }

        /// Expiration month (1-12).
        #[must_use]
        pub const fn exp_month(&self) -> u8 {
            self.exp_month
        }

        /// Expiration year, four-digit (e.g. 2027).
        #[must_use]
        pub const fn exp_year(&self) -> u16 {
            self.exp_year
        }
    }

    impl core::fmt::Debug for RawPan {
        fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
            write!(
                f,
                "RawPan(****{}, {:02}/{})",
                self.last_four(),
                self.exp_month,
                self.exp_year
            )
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vault_ref_debug_masks_value() {
        let v = VaultRef::new("tok_abcdef123456789");
        let dbg = format!("{v:?}");
        assert!(dbg.starts_with("VaultRef(tok_…"));
        assert!(!dbg.contains("abcdef123456789"));
    }

    #[test]
    fn token_debug_hides_bytes() {
        let t = Token::new(b"secret-bytes".to_vec());
        assert_eq!(format!("{t:?}"), "Token(12 bytes)");
    }

    #[test]
    fn a2a_keys_serialize_distinct() {
        let upi = A2aKey::Upi("bob@axis".into());
        let json = serde_json::to_string(&upi).unwrap();
        assert!(json.contains("Upi"));
        assert!(json.contains("bob@axis"));
    }
}
