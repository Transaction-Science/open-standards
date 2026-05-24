//! Cross-platform linker paths for the C libraries we wrap.
//!
//! Most of our codec deps (`opus`, `jpegxl-rs`, `dav1d`, `zstd`, `xz2`,
//! `libavif-image`) already have build.rs files that probe pkg-config
//! themselves. `flac-bound` does NOT — it just links `-lflac` and
//! expects the linker to find it. On Homebrew macOS that lives under
//! `/opt/homebrew/opt/flac/lib`, which Cargo doesn't probe. We use the
//! `pkg-config` crate to discover the right path on whatever system
//! the user is on (Linux distros, macOS Homebrew, BSD ports, etc.).
//!
//! This replaces the macOS-hardcoded `.cargo/config.toml` rustflags.

fn main() {
    generate_c_header();
    link_libflac();
}

/// Regenerate `include/wai.h` from the FFI surface so SDK consumers can
/// always grab a typed header that matches the shipped binary. Only
/// regenerates when source files actually change (cargo handles this
/// via `rerun-if-changed` declarations).
fn generate_c_header() {
    println!("cargo:rerun-if-changed=src/ffi.rs");
    println!("cargo:rerun-if-changed=src/container.rs");
    println!("cargo:rerun-if-changed=cbindgen.toml");
    let crate_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    let out_path = std::path::Path::new(&crate_dir).join("include/wai.h");
    if let Err(e) = std::fs::create_dir_all(out_path.parent().unwrap()) {
        println!("cargo:warning=cbindgen: could not create include/ dir: {e}");
        return;
    }
    match cbindgen::Builder::new()
        .with_crate(&crate_dir)
        .with_config(cbindgen::Config::from_file(
            std::path::Path::new(&crate_dir).join("cbindgen.toml")).unwrap_or_default())
        .generate()
    {
        Ok(bindings) => {
            bindings.write_to_file(&out_path);
        }
        Err(e) => {
            println!("cargo:warning=cbindgen failed: {e}");
        }
    }
}

fn link_libflac() {
    if let Ok(lib) = pkg_config::Config::new()
        .atleast_version("1.3")
        .probe("flac")
    {
        for path in &lib.link_paths {
            println!("cargo:rustc-link-search=native={}", path.display());
        }
        // pkg-config also emits the library names; flac-bound links its
        // own `-lflac` directive so we only need the search path.
    } else {
        // Fallback: try the canonical Homebrew location on macOS so the
        // common case still works without pkg-config installed.
        #[cfg(target_os = "macos")]
        {
            let candidates = [
                "/opt/homebrew/opt/flac/lib",            // Apple Silicon
                "/usr/local/opt/flac/lib",                // Intel macOS
            ];
            for p in candidates {
                if std::path::Path::new(p).exists() {
                    println!("cargo:rustc-link-search=native={p}");
                    return;
                }
            }
        }
        // Linux/other: hope that the system linker finds -lflac on its own.
        // If not, the link error will be obvious ("cannot find -lflac")
        // and the user needs to install libflac-dev / equivalent.
    }
}
