package dev.openpay

import androidx.test.core.app.ApplicationProvider
import androidx.test.ext.junit.runners.AndroidJUnit4
import org.junit.After
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Before
import org.junit.Test
import org.junit.runner.RunWith

/**
 * Instrumented tests for [KeystoreVault].
 *
 * These run on a connected Android device or emulator because
 * [KeystoreVault] requires an Android [Context] and the
 * Keystore-resident master key, neither of which exists on a plain
 * JVM. Trigger via `./gradlew :openpay:connectedAndroidTest`.
 *
 * Each test uses a per-test file name and clears it before/after, so
 * tests are independent.
 */
@RunWith(AndroidJUnit4::class)
class KeystoreVaultInstrumentedTest {

    private val context get() = ApplicationProvider.getApplicationContext<android.content.Context>()
    private val testFileName = "openpay-vault-test"
    private val VALID_VISA = "4242424242424242"
    private val VALID_MC = "5555555555554444"

    @Before
    fun setUp() {
        // Clear the test file before each run. We can't go through
        // EncryptedSharedPreferences.edit().clear() here because that
        // requires constructing the vault, which we test separately.
        // Use a plain SharedPreferences deletion at the underlying
        // file level.
        context.deleteSharedPreferences(testFileName)
    }

    @After
    fun tearDown() {
        context.deleteSharedPreferences(testFileName)
    }

    @Test
    fun keystoreVault_constructs() {
        KeystoreVault(context, fileName = testFileName).use { vault ->
            assertEquals("android-keystore", vault.name)
            assertEquals(0, vault.size)
        }
    }

    @Test
    fun keystoreVault_tokenizeAndDetokenize_roundTrip() {
        KeystoreVault(context, fileName = testFileName).use { vault ->
            val token = vault.tokenizeFromString(VALID_VISA, 12, 2030)
            assertTrue(token.asString.startsWith("tok_v7_"))

            val recovered = vault.detokenize(token)
            assertEquals("424242", recovered.firstSix)
            assertEquals("4242", recovered.lastFour)
            assertEquals(12.toByte(), recovered.expMonth)
            assertEquals(2030.toShort(), recovered.expYear)
            recovered.close()
            token.close()
        }
    }

    @Test
    fun keystoreVault_invalidPan_throwsBeforePersisting() {
        KeystoreVault(context, fileName = testFileName).use { vault ->
            assertThrows(OpenPayException.InvalidInput::class.java) {
                vault.tokenizeFromString("1111111111111111", 12, 2030)
            }
            // Nothing was persisted.
            assertEquals(0, vault.size)
        }
    }

    @Test
    fun keystoreVault_unknownToken_throwsVaultLookupFailed() {
        KeystoreVault(context, fileName = testFileName).use { vault ->
            val fake = VaultRef.fromString("tok_v7_doesnotexist")
            assertThrows(OpenPayException.VaultLookupFailed::class.java) {
                vault.detokenize(fake)
            }
            fake.close()
        }
    }

    @Test
    fun keystoreVault_malformedToken_collapsesToLookupFailed() {
        // Oracle discipline: malformed and unknown share the same
        // exception type so an attacker can't distinguish them.
        KeystoreVault(context, fileName = testFileName).use { vault ->
            val bad = VaultRef.fromString("not-a-real-token")
            assertThrows(OpenPayException.VaultLookupFailed::class.java) {
                vault.detokenize(bad)
            }
            bad.close()
        }
    }

    @Test
    fun keystoreVault_singleUse_consumesOnFirstDetokenize() {
        KeystoreVault(context, fileName = testFileName).use { vault ->
            val token = vault.tokenizeFromString(
                VALID_VISA, 12, 2030,
                TokenizationPolicy.singleUse(120L),
            )

            val first = vault.detokenize(token)
            first.close()

            assertThrows(OpenPayException.TokenAlreadyConsumed::class.java) {
                vault.detokenize(token)
            }
            token.close()
        }
    }

    @Test
    fun keystoreVault_reusable_survivesMultipleDetokenizes() {
        KeystoreVault(context, fileName = testFileName).use { vault ->
            val token = vault.tokenizeFromString(VALID_VISA, 12, 2030)
            repeat(5) {
                vault.detokenize(token).close()
            }
            token.close()
        }
    }

    @Test
    fun keystoreVault_existsAndDelete() {
        KeystoreVault(context, fileName = testFileName).use { vault ->
            val token = vault.tokenizeFromString(VALID_VISA, 12, 2030)
            assertTrue(vault.exists(token))
            assertEquals(1, vault.size)

            assertTrue(vault.delete(token))
            assertFalse(vault.exists(token))
            assertEquals(0, vault.size)

            // Idempotent.
            assertFalse(vault.delete(token))
            token.close()
        }
    }

    @Test
    fun keystoreVault_distinctTokensForSamePan() {
        KeystoreVault(context, fileName = testFileName).use { vault ->
            val t1 = vault.tokenizeFromString(VALID_VISA, 12, 2030)
            val t2 = vault.tokenizeFromString(VALID_VISA, 12, 2030)
            assertNotEquals(t1.asString, t2.asString)
            assertEquals(2, vault.size)
            t1.close()
            t2.close()
        }
    }

    @Test
    fun keystoreVault_multipleCardsCoexist() {
        KeystoreVault(context, fileName = testFileName).use { vault ->
            val visa = vault.tokenizeFromString(VALID_VISA, 12, 2030)
            val mc = vault.tokenizeFromString(VALID_MC, 11, 2028)

            val visaRecovered = vault.detokenize(visa)
            assertEquals("4242", visaRecovered.lastFour)
            visaRecovered.close()

            val mcRecovered = vault.detokenize(mc)
            assertEquals("4444", mcRecovered.lastFour)
            assertEquals(11.toByte(), mcRecovered.expMonth)
            mcRecovered.close()

            visa.close()
            mc.close()
        }
    }

    @Test
    fun keystoreVault_panNeverAppearsInTokenString() {
        // Same opacity guarantee as RustVault.
        KeystoreVault(context, fileName = testFileName).use { vault ->
            val token = vault.tokenizeFromString(VALID_VISA, 12, 2030)
            assertFalse(token.asString.contains(VALID_VISA))
            assertFalse(token.asString.contains("424242"))
            assertFalse(token.asString.contains("4242"))
            token.close()
        }
    }

    @Test
    fun keystoreVault_persistsAcrossInstanceReconstruction() {
        // Tokenize on one instance, detokenize on a fresh instance
        // pointing at the same file. This is the durability property
        // that distinguishes KeystoreVault from RustVault.
        val tokenString: String
        KeystoreVault(context, fileName = testFileName).use { v1 ->
            val t = v1.tokenizeFromString(VALID_VISA, 12, 2030)
            tokenString = t.asString
            t.close()
        }
        KeystoreVault(context, fileName = testFileName).use { v2 ->
            VaultRef.fromString(tokenString).use { recoveredRef ->
                v2.detokenize(recoveredRef).use { card ->
                    assertEquals("4242", card.lastFour)
                }
            }
        }
    }

    @Test
    fun keystoreVault_isolatesAcrossFileNames() {
        val tokenInVaultA: String
        KeystoreVault(context, fileName = "${testFileName}-a").use { a ->
            val t = a.tokenizeFromString(VALID_VISA, 12, 2030)
            tokenInVaultA = t.asString
            t.close()
        }
        try {
            KeystoreVault(context, fileName = "${testFileName}-b").use { b ->
                VaultRef.fromString(tokenInVaultA).use { ref ->
                    assertFalse(b.exists(ref))
                    assertThrows(OpenPayException.VaultLookupFailed::class.java) {
                        b.detokenize(ref)
                    }
                }
            }
        } finally {
            context.deleteSharedPreferences("${testFileName}-a")
            context.deleteSharedPreferences("${testFileName}-b")
        }
    }
}
