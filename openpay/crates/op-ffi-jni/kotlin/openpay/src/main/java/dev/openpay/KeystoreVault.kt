package dev.openpay

import android.content.Context
import android.content.SharedPreferences
import androidx.security.crypto.EncryptedSharedPreferences
import androidx.security.crypto.MasterKey
import java.util.UUID
import org.json.JSONObject

/**
 * Production Android vault backed by [EncryptedSharedPreferences] +
 * [MasterKey] (AndroidX Security 1.1.x), themselves backed by the
 * Android Keystore via Tink. On devices with TEE / StrongBox, the
 * master key never leaves secure hardware.
 *
 * ## Architectural note: not a plain [Vault]
 *
 * `KeystoreVault` does **not** implement the [Vault] interface
 * directly. The interface requires `tokenize(CardData)`, but
 * [CardData] deliberately hides its PAN bytes — they live inside
 * the Rust heap and cannot be read by Kotlin. That's the PCI scope
 * boundary by design.
 *
 * `KeystoreVault` therefore takes PAN strings directly via
 * [tokenizeFromString]. The PAN goes into
 * [EncryptedSharedPreferences], which encrypts it under the
 * Keystore-resident master key. This is the correct production
 * pattern on Android — the encrypted store IS the Keystore — and
 * matches what PSP SDKs like Stripe and Adyen do in their native
 * Android integrations.
 *
 * If you want the [Vault] interface (for code that's polymorphic
 * over vault types), use [RustVault] for development and write a
 * platform-bridge wrapper for production.
 *
 * ## Why this isn't [RustVault]
 *
 * [RustVault] is an in-memory reference vault suitable for tests
 * but unsuitable for production because state is lost on app
 * restart. `KeystoreVault` persists ciphertext to disk under a
 * Keystore-backed master key.
 *
 * ## Crypto details
 *
 * - **Master key**: AES-256-GCM, alias managed by [MasterKey].
 * - **Per-record encryption**: handled by [EncryptedSharedPreferences]
 *   (AES-256-SIV for keys, AES-256-GCM for values; both via Tink).
 * - **Token format**: `tok_v7_` + a UUID v4. Tokens are visually
 *   distinguishable from PANs per PCI §3.3.
 *
 * ## Backup discipline
 *
 * **The encrypted file must not be included in Auto Backup.** When
 * restoring to a different device, the new device has a different
 * Keystore-resident master key, and decryption will fail. Configure
 * your app's `backup_rules.xml`:
 *
 * ```xml
 * <full-backup-content>
 *     <exclude domain="sharedpref" path="openpay-vault.xml" />
 * </full-backup-content>
 * ```
 *
 * This matches Apple's `kSecAttrAccessibleAfterFirstUnlockThisDeviceOnly`
 * stance for iOS.
 *
 * ## Library deprecation note
 *
 * `androidx.security:security-crypto` was deprecated at 1.1.0-alpha07
 * in April 2025. The API surface remains stable; the deprecation is
 * about future maintenance. Two options:
 *
 * 1. Continue on `androidx.security:security-crypto:1.1.0-alpha06` (or
 *    newer). The classes still work.
 * 2. Switch to the community fork `dev.spght:encryptedprefs-ktx`,
 *    which preserves the API and continues maintenance.
 *
 * This class uses the AndroidX class names; swapping to the fork is
 * a one-line import change.
 *
 * @param context Android [Context]. Use `applicationContext` to avoid
 *   holding an activity reference.
 * @param fileName Name of the encrypted SharedPreferences file. The
 *   default is `openpay-vault` which becomes `openpay-vault.xml` on
 *   disk.
 */
public class KeystoreVault(
    context: Context,
    fileName: String = "openpay-vault",
    public val name: String = "android-keystore",
) : AutoCloseable {

    private val masterKey: MasterKey = MasterKey.Builder(context)
        .setKeyScheme(MasterKey.KeyScheme.AES256_GCM)
        .build()

    private val prefs: SharedPreferences = EncryptedSharedPreferences.create(
        context,
        fileName,
        masterKey,
        EncryptedSharedPreferences.PrefKeyEncryptionScheme.AES256_SIV,
        EncryptedSharedPreferences.PrefValueEncryptionScheme.AES256_GCM,
    )

    /**
     * Tokenize a PAN. The PAN string is validated (Luhn + length +
     * expiration sanity) then written into EncryptedSharedPreferences
     * under a fresh token id.
     *
     * @throws OpenPayException.InvalidInput if validation fails.
     */
    @Throws(OpenPayException::class)
    public fun tokenizeFromString(
        pan: String,
        expMonth: Byte,
        expYear: Short,
        policy: TokenizationPolicy = TokenizationPolicy(),
    ): VaultRef {
        // Validate via Rust constructor (Luhn + length + expiration).
        // The handle is immediately closed; we only needed the
        // validation side-effect.
        CardData(pan, expMonth, expYear).close()

        val tokenId = "tok_v7_${UUID.randomUUID().toString().replace("-", "")}"

        val record = JSONObject().apply {
            put("pan", pan)
            put("exp_month", expMonth.toInt())
            put("exp_year", expYear.toInt())
            put("format", policy.format.nativeCode)
            put("lifetime", policy.lifetime.nativeCode)
            put("ttl_seconds", policy.nativeTtl())
            put("issued_at", System.currentTimeMillis() / 1000L)
            put("consumed", false)
        }

        prefs.edit().putString(tokenId, record.toString()).apply()

        return VaultRef.fromString(tokenId)
    }

    /**
     * Detokenize. Throws [OpenPayException.VaultLookupFailed] for
     * unknown / malformed / corrupted tokens (oracle-discipline
     * collapsing). Throws [OpenPayException.TokenExpired] /
     * [OpenPayException.TokenAlreadyConsumed] for policy violations.
     */
    @Throws(OpenPayException::class)
    public fun detokenize(token: VaultRef): CardData {
        val tokenId = token.asString

        if (!tokenId.startsWith("tok_v7_")) {
            throw OpenPayException.VaultLookupFailed()
        }

        val json = prefs.getString(tokenId, null)
            ?: throw OpenPayException.VaultLookupFailed()

        val record = try {
            JSONObject(json)
        } catch (_: Throwable) {
            throw OpenPayException.VaultLookupFailed()
        }

        // Check expiration.
        val ttl = record.optLong("ttl_seconds", 0L)
        if (ttl > 0L) {
            val issuedAt = record.optLong("issued_at", 0L)
            val now = System.currentTimeMillis() / 1000L
            val age = now - issuedAt
            if (age >= 0L && age >= ttl) {
                throw OpenPayException.TokenExpired()
            }
        }

        // Check single-use.
        val lifetime = record.optInt("lifetime", 0)
        val consumed = record.optBoolean("consumed", false)
        if (lifetime == TokenLifetime.SINGLE_USE.nativeCode && consumed) {
            throw OpenPayException.TokenAlreadyConsumed()
        }

        val pan = record.getString("pan")
        val expMonth = record.getInt("exp_month").toByte()
        val expYear = record.getInt("exp_year").toShort()

        // Mark single-use as consumed after successful read.
        if (lifetime == TokenLifetime.SINGLE_USE.nativeCode) {
            record.put("consumed", true)
            prefs.edit().putString(tokenId, record.toString()).apply()
        }

        // Reconstruct via the validated constructor — runs Luhn again
        // as defense in depth against tampering with the stored record.
        return CardData(pan, expMonth, expYear)
    }

    /** Probe existence. */
    public fun exists(token: VaultRef): Boolean {
        val tokenId = token.asString
        if (!tokenId.startsWith("tok_v7_")) return false
        return prefs.contains(tokenId)
    }

    /**
     * Delete a token. Returns `true` if a mapping was removed.
     * Idempotent.
     */
    public fun delete(token: VaultRef): Boolean {
        val tokenId = token.asString
        if (!tokenId.startsWith("tok_v7_")) return false
        if (!prefs.contains(tokenId)) return false
        prefs.edit().remove(tokenId).apply()
        return true
    }

    /** Number of tokens currently held (for diagnostics, not PCI logs). */
    public val size: Int
        get() = prefs.all.size

    override fun close() {
        // EncryptedSharedPreferences has no explicit close; the
        // Keystore-resident master key persists across app launches.
    }
}
