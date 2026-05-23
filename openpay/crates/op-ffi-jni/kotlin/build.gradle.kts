// Top-level build file. Plugin versions live here and are applied per-module.
//
// Pinned to AGP 9.1.0 (Feb 2026) which has built-in Kotlin support;
// the `org.jetbrains.kotlin.android` plugin is no longer applied
// separately in module build files.
//
// Requirements (verified against developer.android.com release notes):
//   - Gradle 9.1+ (set in gradle/wrapper/gradle-wrapper.properties)
//   - JDK 17+
//   - Kotlin Gradle Plugin 2.2.x (bundled with AGP 9.1)
//   - compileSdk up to 37 supported

plugins {
    id("com.android.library") version "9.1.0" apply false
}
