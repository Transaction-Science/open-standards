package dev.openpay

import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertNotEquals
import org.junit.Assert.assertThrows
import org.junit.Assert.assertTrue
import org.junit.Test

/**
 * JVM unit tests for the OpenPay Kotlin layer.
 *
 * These tests require the JNI library to be available on the host
 * JVM. Run via `./gradlew :openpay:test` after building the .so for
 * the host architecture (or after staging a desktop libop_ffi_jni.so
 * on `java.library.path`).
 *
 * Tests that require an Android Context (e.g. [KeystoreVault]) live
 * in `src/androidTest/` and run on a connected device or emulator.
 */
class CoreTest {

    private val VALID_VISA = "4242424242424242"

    @Test
    fun cardData_validInput_succeeds() {
        CardData(VALID_VISA, 12, 2030).use { card ->
            assertEquals("424242", card.firstSix)
            assertEquals("4242", card.lastFour)
            assertEquals(12.toByte(), card.expMonth)
            assertEquals(2030.toShort(), card.expYear)
        }
    }

    @Test
    fun cardData_invalidPan_throwsInvalidInput() {
        assertThrows(OpenPayException.InvalidInput::class.java) {
            CardData("1111111111111111", 12, 2030)
        }
    }

    @Test
    fun cardData_badExpiration_throwsInvalidInput() {
        assertThrows(OpenPayException.InvalidInput::class.java) {
            CardData(VALID_VISA, 13, 2030)
        }
        assertThrows(OpenPayException.InvalidInput::class.java) {
            CardData(VALID_VISA, 0, 2030)
        }
    }

    @Test
    fun cardData_afterClose_methodsThrow() {
        val card = CardData(VALID_VISA, 12, 2030)
        card.close()
        assertThrows(OpenPayException.InvalidInput::class.java) {
            card.firstSix
        }
    }

    @Test
    fun cardData_doubleClose_isSafe() {
        val card = CardData(VALID_VISA, 12, 2030)
        card.close()
        // Second close is a no-op, not an error.
        card.close()
    }

    @Test
    fun rustVault_roundTrip() {
        RustVault("test").use { vault ->
            val card = CardData(VALID_VISA, 12, 2030)
            val token = vault.tokenize(card)
            assertTrue(token.asString.startsWith("tok_v7_"))

            val recovered = vault.detokenize(token)
            assertEquals("4242", recovered.lastFour)
            recovered.close()
            token.close()
        }
    }

    @Test
    fun rustVault_unknownToken_throwsVaultLookupFailed() {
        RustVault("err-test").use { vault ->
            val fake = VaultRef.fromString("tok_v7_doesnotexist")
            assertThrows(OpenPayException.VaultLookupFailed::class.java) {
                vault.detokenize(fake)
            }
            fake.close()
        }
    }

    @Test
    fun rustVault_malformedToken_alsoCollapsesToLookupFailed() {
        // Per oracle discipline: malformed and unknown both throw the
        // same exception type.
        RustVault("malformed-test").use { vault ->
            val bad = VaultRef.fromString("not-a-token")
            assertThrows(OpenPayException.VaultLookupFailed::class.java) {
                vault.detokenize(bad)
            }
            bad.close()
        }
    }

    @Test
    fun rustVault_singleUse_consumesOnFirstDetokenize() {
        RustVault("single-use").use { vault ->
            val card = CardData(VALID_VISA, 12, 2030)
            val token = vault.tokenize(card, TokenizationPolicy.singleUse(120))

            val recovered = vault.detokenize(token)
            recovered.close()

            assertThrows(OpenPayException.TokenAlreadyConsumed::class.java) {
                vault.detokenize(token)
            }
            token.close()
        }
    }

    @Test
    fun rustVault_reusable_survivesMultipleDetokenizes() {
        RustVault("reusable").use { vault ->
            val card = CardData(VALID_VISA, 12, 2030)
            val token = vault.tokenize(card)
            repeat(5) {
                vault.detokenize(token).close()
            }
            token.close()
        }
    }

    @Test
    fun rustVault_existsAndDelete() {
        RustVault("delete-test").use { vault ->
            val card = CardData(VALID_VISA, 12, 2030)
            val token = vault.tokenize(card)
            assertTrue(vault.exists(token))
            assertTrue(vault.delete(token))
            assertFalse(vault.exists(token))
            // Idempotent.
            assertFalse(vault.delete(token))
            token.close()
        }
    }

    @Test
    fun rustVault_distinctTokensForSamePan() {
        // Random format: same PAN → different tokens.
        RustVault("uniq").use { vault ->
            val t1 = vault.tokenize(CardData(VALID_VISA, 12, 2030))
            val t2 = vault.tokenize(CardData(VALID_VISA, 12, 2030))
            assertNotEquals(t1.asString, t2.asString)
            t1.close()
            t2.close()
        }
    }

    @Test
    fun heuristicScorer_nameIsStable() {
        HeuristicScorer().use { scorer ->
            assertEquals("heuristic-v1", scorer.name)
        }
    }

    @Test
    fun tokenizationPolicy_helpers() {
        val s = TokenizationPolicy.singleUse(60L)
        assertEquals(TokenLifetime.SINGLE_USE, s.lifetime)
        assertEquals(60L, s.ttlSeconds)

        val c = TokenizationPolicy.cardOnFile()
        assertEquals(TokenLifetime.REUSABLE, c.lifetime)
        assertEquals(null, c.ttlSeconds)
    }

    @Test
    fun vaultRef_fromStringAndBack() {
        val original = "tok_v7_abc123"
        val vref = VaultRef.fromString(original)
        assertEquals(original, vref.asString)
        vref.close()
    }

    @Test
    fun panNeverAppearsInTokenString() {
        RustVault("opacity").use { vault ->
            val card = CardData(VALID_VISA, 12, 2030)
            val token = vault.tokenize(card)
            assertFalse(token.asString.contains(VALID_VISA))
            assertFalse(token.asString.contains("424242"))
            assertFalse(token.asString.contains("4242"))
            token.close()
        }
    }
}
