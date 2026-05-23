package dev.openpay

/**
 * Base class for all OpenPay errors raised across the JNI boundary.
 *
 * Subclasses match the [FfiError] discriminants from the Rust side
 * one-to-one. Idiomatic Kotlin callers catch the base class and
 * `when` on the subtype.
 *
 * ## Oracle discipline
 *
 * [VaultLookupFailed] collapses three distinct Rust-side errors —
 * `NotFound`, `AuthFailed`, and `InvalidToken` — into a single
 * exception subclass. Distinguishing these would let an attacker
 * probe the vault to learn which tokens existed. The compiler
 * cannot enforce this contract; the Rust side does the collapse,
 * and the Kotlin side preserves the single subclass.
 */
public sealed class OpenPayException(message: String) : Exception(message) {

    /** Input data is malformed (PAN, expiration, token format). */
    public class InvalidInput(message: String = "invalid input") : OpenPayException(message)

    /**
     * Vault could not resolve a token. Encompasses unknown token,
     * authentication failure, and malformed token format —
     * **deliberately not distinguished** to avoid leaking oracle
     * information.
     */
    public class VaultLookupFailed(message: String = "vault lookup failed") :
        OpenPayException(message)

    /** Token has expired per its tokenization policy. */
    public class TokenExpired(message: String = "token expired") : OpenPayException(message)

    /** Single-use token was already consumed. */
    public class TokenAlreadyConsumed(message: String = "token already consumed") :
        OpenPayException(message)

    /** Fraud scorer rejected the request. */
    public class FraudDeclined(message: String = "fraud declined") : OpenPayException(message)

    /** Fraud scorer flagged for human review. */
    public class FraudReviewRequired(message: String = "fraud review required") :
        OpenPayException(message)

    /** Backend (vault, rail, scorer) returned an opaque failure. */
    public class Backend(message: String = "backend error") : OpenPayException(message)

    /** Rate-limit or capacity exhaustion. */
    public class Capacity(message: String = "rate limit or capacity") : OpenPayException(message)
}
