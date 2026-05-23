package dev.openpay

import java.util.concurrent.atomic.AtomicLong

/**
 * Heuristic fraud scorer. Rule-based, pure Rust, no model load.
 *
 * Implements [AutoCloseable]. The Rust handle is freed when [close]
 * is called.
 */
public class HeuristicScorer : AutoCloseable {

    private val handle = AtomicLong(nativeNew())

    private fun requireHandle(): Long {
        val h = handle.get()
        if (h == 0L) {
            throw OpenPayException.InvalidInput("HeuristicScorer has been closed")
        }
        return h
    }

    /** Scorer name for telemetry. */
    public val name: String
        get() = nativeName(requireHandle())

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
        private external fun nativeNew(): Long

        @JvmStatic
        private external fun nativeFree(handle: Long)

        @JvmStatic
        private external fun nativeName(handle: Long): String
    }
}
