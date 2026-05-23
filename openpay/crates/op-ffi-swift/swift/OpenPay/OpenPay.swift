// OpenPay.swift — idiomatic Swift wrapper over the op-ffi-swift bridge.
//
// This file is hand-written and lives next to the swift-bridge
// generated OpenPay.swift (which is copied in by the build script).
// The split is deliberate: swift-bridge's output is mechanical and
// hews to the Rust function names; this wrapper layers Swift idioms
// (throwing functions, value-typed enums, async APIs) on top.
//
// What's covered:
//   - `OpenPayError`: a typed error enum that mirrors `FfiError`.
//   - `OpenPay.Vault`: a class wrapping `RustVault` with throwing
//     methods. Tokenize/detokenize throw `OpenPayError` rather than
//     returning Optional.
//   - `OpenPay.CardData`: a class wrapping `RustCardData` with safe
//     initializer that throws on invalid input.
//   - `TokenizationPolicy`: a Swift-native struct that converts to
//     the bridged FFI struct.
//   - `FraudDecision`: re-exported as a Swift enum.
//
// What's NOT covered yet (deferred to consumer code or a later phase):
//   - `KeychainVault`: a Swift implementation of `Vault` backed by
//     `SecItemAdd` / `SecItemCopyMatching`. The op-ffi-swift crate
//     does NOT ship one; consumers write their own because the
//     accessibility class and access-control flags are deployment
//     decisions (background access, biometric gating, etc.). The
//     PCI DSS Tokenization Guidelines § 3.3 distinguishability rule
//     is already satisfied by Rust-side token format.
//   - Network token bridging (Visa VTS, Mastercard MDES).
//   - Apple Pay PKPayment → CardData conversion.

import Foundation

// MARK: - Error type

/// Errors raised by OpenPay Swift APIs. Mirrors the FFI enum.
public enum OpenPayError: Error, Equatable {
    /// Input data is malformed (PAN, expiration, token format).
    case invalidInput
    /// Token lookup failed (unknown, auth failed, or invalid format).
    /// Collapsed for oracle discipline — never distinguish.
    case vaultLookupFailed
    /// Token has expired per policy.
    case tokenExpired
    /// Single-use token was already consumed.
    case tokenAlreadyConsumed
    /// Fraud scorer rejected the request.
    case fraudDeclined
    /// Fraud scorer flagged for human review.
    case fraudReviewRequired
    /// Backend (vault, rail, scorer) reported an opaque failure.
    case backend
    /// FFI-internal failure.
    case `internal`
    /// Rate limit or capacity exhaustion.
    case capacity

    /// Construct from the raw FFI discriminant.
    public init(rawValue: Int32) {
        switch rawValue {
        case 0: self = .internal // Ok should never become an error
        case 1: self = .invalidInput
        case 2: self = .vaultLookupFailed
        case 3: self = .tokenExpired
        case 4: self = .tokenAlreadyConsumed
        case 5: self = .fraudDeclined
        case 6: self = .fraudReviewRequired
        case 7: self = .backend
        case 8: self = .internal
        case 9: self = .capacity
        default: self = .internal
        }
    }
}

extension OpenPayError: LocalizedError {
    public var errorDescription: String? {
        switch self {
        case .invalidInput:           return "Invalid input"
        case .vaultLookupFailed:      return "Vault lookup failed"
        case .tokenExpired:           return "Token expired"
        case .tokenAlreadyConsumed:   return "Token already consumed"
        case .fraudDeclined:          return "Fraud declined"
        case .fraudReviewRequired:    return "Fraud review required"
        case .backend:                return "Backend error"
        case .internal:               return "Internal error"
        case .capacity:               return "Rate limit or capacity"
        }
    }
}

// MARK: - Tokenization policy

/// Tokenization policy controlling how the vault mints a token.
public struct TokenizationPolicy {
    public enum Format {
        case random
        case deterministic
    }
    public enum Lifetime {
        case reusable
        case singleUse
    }

    public var format: Format
    public var lifetime: Lifetime
    /// `nil` = no TTL. Otherwise number of seconds before the token
    /// becomes invalid for detokenize.
    public var ttlSeconds: UInt64?

    public init(
        format: Format = .random,
        lifetime: Lifetime = .reusable,
        ttlSeconds: UInt64? = nil
    ) {
        self.format = format
        self.lifetime = lifetime
        self.ttlSeconds = ttlSeconds
    }

    /// Short-lived single-use token, appropriate for 3DS auth.
    public static func singleUse(ttlSeconds: UInt64 = 120) -> TokenizationPolicy {
        TokenizationPolicy(format: .random, lifetime: .singleUse, ttlSeconds: ttlSeconds)
    }

    /// Long-lived random reusable token for card-on-file.
    public static func cardOnFile() -> TokenizationPolicy {
        TokenizationPolicy(format: .random, lifetime: .reusable, ttlSeconds: nil)
    }

    /// Convert to the bridged FFI struct.
    internal func toFFI() -> TokenizationPolicyFfi {
        TokenizationPolicyFfi(
            format: format == .random ? .Random : .Deterministic,
            lifetime: lifetime == .reusable ? .Reusable : .SingleUse,
            ttl_seconds: ttlSeconds ?? 0
        )
    }
}

// MARK: - OpenPay namespace

/// Namespace for the OpenPay Swift surface.
public enum OpenPay {

    /// A card-data handle. Construction validates Luhn + length +
    /// expiration sanity. The PAN is held inside the Rust core,
    /// zeroized on drop.
    public final class CardData {
        internal let inner: RustCardData

        /// Construct from PAN + expiration. Throws `OpenPayError.invalidInput`
        /// if the input fails validation.
        public init(pan: String, expMonth: UInt8, expYear: UInt16) throws {
            guard let card = RustCardData.new(pan: pan, exp_month: expMonth, exp_year: expYear) else {
                throw OpenPayError(rawValue: last_error_card())
            }
            self.inner = card
        }

        internal init(wrapping inner: RustCardData) {
            self.inner = inner
        }

        public var firstSix: String { inner.first_six().toString() }
        public var lastFour: String { inner.last_four().toString() }
        public var expMonth: UInt8 { inner.exp_month() }
        public var expYear: UInt16 { inner.exp_year() }
    }

    /// An opaque vault token reference. Safe to log, store, or transmit.
    public final class VaultRef {
        internal let inner: RustVaultRef

        internal init(wrapping inner: RustVaultRef) {
            self.inner = inner
        }

        /// String form of the token.
        public var asString: String { inner.as_string().toString() }
    }

    /// A vault. Holds an Arc to the underlying Rust implementation.
    public final class Vault {
        internal let inner: RustVault

        /// Construct an ephemeral in-memory vault. Useful for tests
        /// and development; production deploys plug in a platform
        /// vault implementation.
        public static func ephemeral(name: String) -> Vault {
            Vault(wrapping: RustVault.ephemeral(name: name))
        }

        internal init(wrapping inner: RustVault) {
            self.inner = inner
        }

        /// Tokenize a card under the given policy. Consumes `card`.
        public func tokenize(card: CardData, policy: TokenizationPolicy) throws -> VaultRef {
            guard let vref = inner.tokenize(card: card.inner, policy: policy.toFFI()) else {
                throw OpenPayError(rawValue: last_error_vault(v: inner))
            }
            return VaultRef(wrapping: vref)
        }

        /// Detokenize. Throws `vaultLookupFailed` (unknown / auth / malformed),
        /// `tokenExpired`, or `tokenAlreadyConsumed`.
        public func detokenize(token: VaultRef) throws -> CardData {
            guard let card = inner.detokenize(token: token.inner) else {
                throw OpenPayError(rawValue: last_error_vault(v: inner))
            }
            return CardData(wrapping: card)
        }

        /// Probe existence.
        public func exists(token: VaultRef) -> Bool {
            inner.exists(token: token.inner)
        }

        /// Delete a token. Returns `true` if a mapping was removed.
        /// Idempotent.
        @discardableResult
        public func delete(token: VaultRef) -> Bool {
            inner.delete(token: token.inner)
        }
    }

    /// Heuristic fraud scorer.
    public final class HeuristicScorer {
        internal let inner: RustHeuristicScorer

        public init() {
            self.inner = RustHeuristicScorer.default()
        }

        public var name: String { inner.name().toString() }
    }
}

// MARK: - Swift string helper

private extension RustString {
    func toString() -> String {
        // swift-bridge gives RustString a `toString()` extension already,
        // but we redeclare for clarity in case downstream users see this
        // file before the generated glue.
        return String(describing: self)
    }
}
