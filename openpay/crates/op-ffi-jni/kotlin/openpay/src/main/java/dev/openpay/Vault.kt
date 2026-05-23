package dev.openpay

import java.util.concurrent.atomic.AtomicLong

/**
 * The vault interface. Operators implement this against their
 * preferred backend:
 *
 * - [RustVault] — Rust in-memory reference vault (development).
 * - [KeystoreVault] — Android Keystore + EncryptedSharedPreferences
 *   (production).
 * - Custom implementations against AWS KMS, HashiCorp Vault, a
 *   remote vault service, etc.
 *
 * All methods are [Throws] [OpenPayException]; callers `when` on
 * the subtype.
 *
 * Thread-safe: implementations must support concurrent calls from
 * multiple threads.
 */
public interface Vault : AutoCloseable {

    /** Vault name for telemetry. */
    public val name: String

    /**
     * Tokenize a card. **Consumes** [card] — the [CardData]
     * handle is invalid after this call returns, success or
     * failure.
     */
    @Throws(OpenPayException::class)
    public fun tokenize(card: CardData, policy: TokenizationPolicy = TokenizationPolicy()): VaultRef

    /**
     * Detokenize. Throws:
     * - [OpenPayException.VaultLookupFailed] for unknown / auth-failed
     *   / malformed tokens (collapsed for oracle discipline).
     * - [OpenPayException.TokenExpired] if past TTL.
     * - [OpenPayException.TokenAlreadyConsumed] for single-use tokens.
     */
    @Throws(OpenPayException::class)
    public fun detokenize(token: VaultRef): CardData

    /** Probe existence. */
    @Throws(OpenPayException::class)
    public fun exists(token: VaultRef): Boolean

    /** Delete a token. Returns `true` if a mapping was removed.
     *  Idempotent. */
    @Throws(OpenPayException::class)
    public fun delete(token: VaultRef): Boolean
}

/**
 * The Rust-side in-memory reference vault, exposed via JNI. Suitable
 * for tests and development. Production deployments use
 * [KeystoreVault] or a custom implementation.
 *
 * State is in RAM only; restarting the app loses every token. Use
 * `RustVault` only when this is what you want.
 */
public class RustVault(name: String) : Vault {

    override val name: String = name

    private val handle = AtomicLong(nativeNewEphemeral(name))

    private fun requireHandle(): Long {
        val h = handle.get()
        if (h == 0L) {
            throw OpenPayException.InvalidInput("RustVault has been closed")
        }
        return h
    }

    @Throws(OpenPayException::class)
    override fun tokenize(card: CardData, policy: TokenizationPolicy): VaultRef {
        val vh = requireHandle()
        val ch = card.consumeHandle()
        val tokenHandle = nativeTokenize(
            vh,
            ch,
            policy.format.nativeCode,
            policy.lifetime.nativeCode,
            policy.nativeTtl(),
        )
        return VaultRef(tokenHandle)
    }

    @Throws(OpenPayException::class)
    override fun detokenize(token: VaultRef): CardData {
        val vh = requireHandle()
        val th = token.nativeHandle()
        val cardHandle = nativeDetokenize(vh, th)
        // We need to wrap the returned handle into a CardData. The
        // CardData public constructor runs validation against the PAN
        // string, which we can't do here because there's no PAN; we
        // already have a valid handle from the vault. Use the
        // internal constructor.
        return CardData.fromHandle(cardHandle)
    }

    @Throws(OpenPayException::class)
    override fun exists(token: VaultRef): Boolean {
        val vh = requireHandle()
        val th = token.nativeHandle()
        return nativeExists(vh, th)
    }

    @Throws(OpenPayException::class)
    override fun delete(token: VaultRef): Boolean {
        val vh = requireHandle()
        val th = token.nativeHandle()
        return nativeDelete(vh, th)
    }

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

    private companion object {
        init {
            System.loadLibrary("op_ffi_jni")
        }

        @JvmStatic
        private external fun nativeNewEphemeral(name: String): Long

        @JvmStatic
        private external fun nativeFree(handle: Long)

        @JvmStatic
        @Throws(OpenPayException::class)
        private external fun nativeTokenize(
            vault: Long,
            card: Long,
            format: Int,
            lifetime: Int,
            ttlSeconds: Long,
        ): Long

        @JvmStatic
        @Throws(OpenPayException::class)
        private external fun nativeDetokenize(vault: Long, token: Long): Long

        @JvmStatic
        @Throws(OpenPayException::class)
        private external fun nativeExists(vault: Long, token: Long): Boolean

        @JvmStatic
        @Throws(OpenPayException::class)
        private external fun nativeDelete(vault: Long, token: Long): Boolean
    }
}
