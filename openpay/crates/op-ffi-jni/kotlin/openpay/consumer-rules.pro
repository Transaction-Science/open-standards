# Consumer ProGuard rules. Applied to apps that depend on this
# library so they don't have to copy/paste keep-rules in their own
# project.

# 1. Keep all JNI entry points. R8 / ProGuard would otherwise strip
#    them as unreachable from Kotlin, then the JVM can't dispatch
#    `native` calls.
-keepclasseswithmembernames class dev.openpay.** {
    native <methods>;
}

# 2. Keep the AutoCloseable wrapper classes and their public surface.
#    These are part of the consumer-facing API; aggressive R8
#    repackaging would break Kotlin-level reflection on the package.
-keep public class dev.openpay.CardData { *; }
-keep public class dev.openpay.VaultRef { *; }
-keep public class dev.openpay.Vault { *; }
-keep public class dev.openpay.RustVault { *; }
-keep public class dev.openpay.KeystoreVault { *; }
-keep public class dev.openpay.HeuristicScorer { *; }
-keep public class dev.openpay.TokenizationPolicy { *; }
-keep public enum dev.openpay.TokenFormat { *; }
-keep public enum dev.openpay.TokenLifetime { *; }

# 3. Keep the OpenPayException class hierarchy. The native side
#    throws by class name; R8 renaming would break that lookup.
-keep public class dev.openpay.OpenPayException { *; }
-keep public class dev.openpay.OpenPayException$* { *; }
