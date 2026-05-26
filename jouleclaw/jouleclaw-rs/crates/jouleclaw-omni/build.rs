fn main() {
    // Link Apple Accelerate framework on macOS for AMX-accelerated BLAS (cblas_sgemm).
    // This gives ~10x speedup over naive matmul for any remaining CPU matrix operations.
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
