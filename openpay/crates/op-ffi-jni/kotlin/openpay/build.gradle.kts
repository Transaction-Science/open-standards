// OpenPay Android library module.
//
// Produces an AAR consumable as a Gradle subproject or via mavenLocal()
// publication. The cdylib shipped under src/main/jniLibs/<ABI>/ is
// packaged automatically by AGP.

plugins {
    id("com.android.library")
}

android {
    namespace = "dev.openpay"
    compileSdk = 36

    defaultConfig {
        // 23 = Android 6.0 Marshmallow. Set by Tink 1.18+ which is
        // pulled in by androidx.security:security-crypto 1.1.x.
        // Lower minSdk is not supported by the encrypted-prefs stack.
        minSdk = 23
        testInstrumentationRunner = "androidx.test.runner.AndroidJUnitRunner"
        consumerProguardFiles("consumer-rules.pro")

        // ABI filters: ship all four standard ABIs so apps consuming
        // this library don't need to add their own ABI filters.
        ndk {
            abiFilters += setOf("arm64-v8a", "armeabi-v7a", "x86_64", "x86")
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlin {
        compilerOptions {
            jvmTarget.set(org.jetbrains.kotlin.gradle.dsl.JvmTarget.JVM_17)
        }
    }

    // The .so files live under src/main/jniLibs/<ABI>/libop_ffi_jni.so.
    // We don't compile native code from within Gradle — cargo-ndk does
    // that out of band via scripts/build-android.sh.
    sourceSets {
        getByName("main") {
            jniLibs.srcDirs("src/main/jniLibs")
        }
    }

    packaging {
        // The .so files are pre-built and already stripped by Rust's
        // release profile (strip = true). Leave packaging alone.
        jniLibs {
            useLegacyPackaging = false
        }
    }

    publishing {
        singleVariant("release") {
            withSourcesJar()
        }
    }
}

dependencies {
    // EncryptedSharedPreferences + MasterKey. Deprecated at 1.1.0-alpha07
    // (April 2025) but the API surface is stable and the implementation
    // is still maintained at the level needed for production. Apps may
    // optionally swap in `dev.spght:encryptedprefs-ktx` as a drop-in
    // replacement; the import paths are compatible.
    //
    // 1.1.0-alpha06 was the last version published before deprecation
    // and is what most production apps standardize on today.
    implementation("androidx.security:security-crypto:1.1.0-alpha06")

    // Unit tests (run on host JVM via `./gradlew :openpay:test`).
    testImplementation("junit:junit:4.13.2")

    // Instrumented tests (run on device / emulator via
    // `./gradlew :openpay:connectedAndroidTest`).
    androidTestImplementation("androidx.test.ext:junit:1.2.1")
    androidTestImplementation("androidx.test:runner:1.6.2")
    androidTestImplementation("androidx.test:core-ktx:1.6.1")
}
