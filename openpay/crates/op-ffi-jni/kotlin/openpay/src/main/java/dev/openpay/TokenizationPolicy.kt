package dev.openpay

/**
 * Tokenization format. Random is the PCI DSS-aligned default.
 *
 * @see TokenizationPolicy
 */
public enum class TokenFormat(internal val nativeCode: Int) {
    /**
     * Each call to [Vault.tokenize] returns a fresh random token,
     * even for the same PAN. Matches PCI DSS guidance that tokens
     * have no value for PAN recovery.
     */
    RANDOM(0),

    /**
     * Same PAN → same token. Useful for deduplication but creates
     * a query oracle. Only use if the analytics value is documented
     * and the risk is accepted.
     */
    DETERMINISTIC(1),
}

/**
 * Token lifetime.
 */
public enum class TokenLifetime(internal val nativeCode: Int) {
    /** Token survives until explicitly deleted or expired. */
    REUSABLE(0),

    /**
     * Token is consumed on first successful [Vault.detokenize].
     * Subsequent detokenize attempts throw
     * [OpenPayException.TokenAlreadyConsumed].
     */
    SINGLE_USE(1),
}

/**
 * Operator-tunable tokenization rules.
 *
 * Three orthogonal axes; two pre-built configurations cover the
 * common cases.
 *
 * @property format see [TokenFormat]
 * @property lifetime see [TokenLifetime]
 * @property ttlSeconds optional TTL in seconds, or `null` for no
 *   expiration. The PCI DSS recommendation for ephemeral auth tokens
 *   (3DS, network token cryptograms) is a short TTL bounding the
 *   replay window.
 */
public data class TokenizationPolicy(
    val format: TokenFormat = TokenFormat.RANDOM,
    val lifetime: TokenLifetime = TokenLifetime.REUSABLE,
    val ttlSeconds: Long? = null,
) {
    /** TTL as `Long` for the JNI signature (0 means "no TTL"). */
    internal fun nativeTtl(): Long = ttlSeconds ?: 0L

    public companion object {
        /**
         * Short-lived single-use token. Appropriate for 3DS
         * authentication and other one-shot flows.
         */
        @JvmStatic
        public fun singleUse(ttlSeconds: Long = 120L): TokenizationPolicy =
            TokenizationPolicy(
                format = TokenFormat.RANDOM,
                lifetime = TokenLifetime.SINGLE_USE,
                ttlSeconds = ttlSeconds,
            )

        /** Long-lived random reusable token for card-on-file. */
        @JvmStatic
        public fun cardOnFile(): TokenizationPolicy =
            TokenizationPolicy(
                format = TokenFormat.RANDOM,
                lifetime = TokenLifetime.REUSABLE,
                ttlSeconds = null,
            )
    }
}
