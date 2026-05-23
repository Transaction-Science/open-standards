//! Build script that invokes swift-bridge's parser on our bridge
//! module and emits the corresponding Swift and C glue.
//!
//! The generated artifacts land in `OUT_DIR/swift-bridge`:
//!
//! - `OpenPay.swift` — the idiomatic Swift surface
//! - `openpay-swift-bridge.h` — the C header that Swift imports
//! - `module.modulemap` — declares the C header as a Clang module
//!   that Swift's importer can find
//!
//! The downstream Xcode build copies these into the consuming Swift
//! package. See `scripts/build-xcframework.sh` for the full workflow.

fn main() {
    // Skip the codegen entirely if the consumer wants the C ABI only.
    if std::env::var("CARGO_FEATURE_C_ONLY").is_ok() {
        println!("cargo:warning=op-ffi-swift: c-only feature set; skipping swift-bridge codegen");
        return;
    }

    let out_dir =
        std::path::PathBuf::from(std::env::var_os("OUT_DIR").expect("cargo always sets OUT_DIR"));

    let bridges = vec!["src/bridge.rs"];

    // Tell cargo to re-run if any of our bridge sources change.
    for path in &bridges {
        println!("cargo:rerun-if-changed={path}");
    }
    println!("cargo:rerun-if-changed=build.rs");

    // Parse the bridges and emit Swift + C glue.
    swift_bridge_build::parse_bridges(bridges).write_all_concatenated(&out_dir, "OpenPay");

    // Expose the generated Swift dir to downstream consumers via cargo
    // metadata, so Xcode build scripts can find it without parsing
    // OUT_DIR conventions.
    println!("cargo:swift-bridge-out-dir={}", out_dir.display());
}
