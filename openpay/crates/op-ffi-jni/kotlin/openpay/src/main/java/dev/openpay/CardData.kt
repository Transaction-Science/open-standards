package dev.openpay

import java.util.concurrent.atomic.AtomicLong

/**
 * Validated card data. Wraps a Rust heap allocation that holds the
 * raw PAN, expiration month, and expiration year. The PAN is
 * zeroized when [close] is called or when the object is collected.
 *
 * ## Security
 *
 * The PAN string passed to the constructor is read by the JNI
 * boundary and stored inside the Rust [Vault] crate. **The original
 * Kotlin string remains in JVM heap until garbage collection** — if
 * you need defense in depth, overwrite it manually after this
 * constructor returns. The JVM offers no guarantee.
 *
 * Once constructed, the only accessors are [firstSix] and [lastFour]
 * (safe to log per PCI DSS 4.0.1 §3.4.1) plus [expMonth] / [expYear].
 *
 * ## Lifecycle
 *
 * `CardData` implements [AutoCloseable]. Use `use { }` or call
 * [close] explicitly when done. After [close], the handle is zero
 * and subsequent method calls throw [OpenPayException.InvalidInput].
 *
 * @throws OpenPayException.InvalidInput if Luhn / length / expiration
 *   validation fails.
 */
public class CardData private constructor(handleValue: Long) : AutoCloseable {

    /** Native handle. Wrapped in AtomicLong so concurrent [close]
     *  calls don't race. */
    private val handle = AtomicLong(handleValue)

    /**
     * Public constructor. Validates Luhn + length + expiration.
     *
     * @throws OpenPayException.InvalidInput on validation failure.
     */
    @Throws(OpenPayException::class)
    public constructor(pan: String, expMonth: Byte, expYear: Short) :
        this(nativeNew(pan, expMonth, expYear))

    /** Internal: expose the handle for [Vault.tokenize] to consume. */
    internal fun consumeHandle(): Long {
        val h = handle.getAndSet(0L)
        if (h == 0L) {
            throw OpenPayException.InvalidInput("CardData has been consumed or closed")
        }
        return h
    }

    private fun requireHandle(): Long {
        val h = handle.get()
        if (h == 0L) {
            throw OpenPayException.InvalidInput("CardData has been consumed or closed")
        }
        return h
    }

    /** First six digits (BIN). Safe to log. */
    public val firstSix: String
        get() = nativeFirstSix(requireHandle())

    /** Last four digits. Safe to log. */
    public val lastFour: String
        get() = nativeLastFour(requireHandle())

    /** Expiration month (1-12). */
    public val expMonth: Byte
        get() = nativeExpMonth(requireHandle())

    /** Expiration year (e.g. 2030). */
    public val expYear: Short
        get() = nativeExpYear(requireHandle())

    override fun close() {
        val h = handle.getAndSet(0L)
        if (h != 0L) {
            nativeFree(h)
        }
    }

    /**
     * Finalizer as a defense-in-depth backstop. Kotlin guidance is
     * that [close] should always be called explicitly, but the
     * finalizer ensures the Rust allocation is freed even if the
     * caller forgets.
     */
    @Suppress("removal", "deprecation")
    protected fun finalize() {
        close()
    }

    public companion object {
        init {
            System.loadLibrary("op_ffi_jni")
        }

        /**
         * Internal: wrap a handle returned from a Rust function
         * (e.g. [Vault.detokenize]) without re-running PAN
         * validation. The handle MUST come from a Rust function that
         * produces valid CardData.
         */
        @JvmSynthetic
        internal fun fromHandle(handle: Long): CardData = CardData(handle)

        @JvmStatic
        @Throws(OpenPayException::class)
        private external fun nativeNew(pan: String, expMonth: Byte, expYear: Short): Long

        @JvmStatic
        private external fun nativeFree(handle: Long)

        @JvmStatic
        private external fun nativeFirstSix(handle: Long): String

        @JvmStatic
        private external fun nativeLastFour(handle: Long): String

        @JvmStatic
        private external fun nativeExpMonth(handle: Long): Byte

        @JvmStatic
        private external fun nativeExpYear(handle: Long): Short
    }
}
