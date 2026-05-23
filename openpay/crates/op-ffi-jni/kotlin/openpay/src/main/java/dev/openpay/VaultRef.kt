package dev.openpay

import java.util.concurrent.atomic.AtomicLong

/**
 * Opaque vault token reference. Safe to log, persist, or transmit —
 * it carries no PAN information.
 *
 * Construct from a stored string via [VaultRef.fromString] when
 * recovering a token from durable storage. The vault returns
 * [VaultRef] instances from [Vault.tokenize]; callers typically just
 * pass them around without inspecting the string form.
 *
 * Implements [AutoCloseable]. Call [close] when done, or wrap in
 * `use { }`.
 */
public class VaultRef internal constructor(handleValue: Long) : AutoCloseable {

    private val handle = AtomicLong(handleValue)

    /** Internal: read the live handle for native calls. */
    internal fun nativeHandle(): Long {
        val h = handle.get()
        if (h == 0L) {
            throw OpenPayException.InvalidInput("VaultRef has been closed")
        }
        return h
    }

    /**
     * The token string. Safe to persist to durable storage (Room,
     * EncryptedSharedPreferences, etc.) for later recovery via
     * [fromString].
     */
    public val asString: String
        get() = nativeAsString(nativeHandle())

    override fun close() {
        val h = handle.getAndSet(0L)
        if (h != 0L) {
            nativeFree(h)
        }
    }

    @Suppress("removal", "deprecation")
    protected fun finalize() {
        close()
    }

    public companion object {
        init {
            System.loadLibrary("op_ffi_jni")
        }

        /**
         * Reconstruct a [VaultRef] from its string form. Use when
         * loading a saved token from storage.
         *
         * No validation: malformed tokens are detected by the vault
         * on detokenize and surface as
         * [OpenPayException.VaultLookupFailed].
         */
        @JvmStatic
        public fun fromString(token: String): VaultRef = VaultRef(nativeFromString(token))

        @JvmStatic
        private external fun nativeFromString(token: String): Long

        @JvmStatic
        private external fun nativeFree(handle: Long)

        @JvmStatic
        private external fun nativeAsString(handle: Long): String
    }
}
