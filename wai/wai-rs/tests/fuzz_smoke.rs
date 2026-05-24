//! Lightweight fuzz-smoke for every decoder. Feeds adversarial random
//! byte sequences to each codec's decode entry point and asserts that
//! none of them panic, segfault, or hang. This catches the bug class
//! that the corpus bench can't (the corpus is all VALID input by
//! construction; fuzz hits the malformed-input path the wrappers'
//! `catch_unwind`s and bounds-checks are supposed to handle).
//!
//! Stays on stable Rust (no libFuzzer/AFL needed). For deeper coverage
//! drop in `cargo-fuzz` targets later — this catches the cheap ones.

use std::panic;

const N: usize = 1024;          // tries per codec; bumps if a target needs more
const MAX_LEN: usize = 16_384;  // ceiling — large enough for header chases

fn lcg(state: &mut u64) -> u8 {
    // xorshift64* — deterministic, dependency-free; not cryptographic.
    *state ^= *state >> 12;
    *state ^= *state << 25;
    *state ^= *state >> 27;
    (state.wrapping_mul(0x2545F4914F6CDD1D) >> 32) as u8
}

fn rand_bytes(seed: u64, len: usize) -> Vec<u8> {
    let mut s = seed.max(1);
    (0..len).map(|_| lcg(&mut s)).collect()
}

/// Run `n` adversarial inputs through `decode` and assert it never
/// panics. Failures of the codec on invalid input are expected — they
/// should return `Err`, NOT unwind.
fn fuzz<F>(name: &str, n: usize, decode: F)
where F: Fn(&[u8]) -> bool + std::panic::RefUnwindSafe
{
    for i in 0..n {
        let seed = (name.bytes().map(|b| b as u64).sum::<u64>() ^ i as u64).wrapping_mul(0x9E37);
        // Vary length: empty, tiny, medium, large
        let len = match i % 4 {
            0 => 0,
            1 => 1 + (i % 31),
            2 => 64 + (i % 256),
            _ => 1 + (seed as usize % MAX_LEN),
        };
        let buf = rand_bytes(seed, len);
        let r = panic::catch_unwind(|| decode(&buf));
        assert!(r.is_ok(),
                "{} decoder panicked on adversarial input (seed={seed}, len={len})",
                name);
    }
}

// silence the "should never succeed" lint — decode result is intentionally ignored
fn ok<T, E>(_: Result<T, E>) -> bool { true }

#[test] fn fuzz_png_decode() {
    fuzz("png", N, |b| ok(wai::codecs::image::png_decode(b)));
}
#[test] fn fuzz_jpeg_decode() {
    fuzz("jpeg", N, |b| ok(wai::codecs::image::jpeg_decode(b)));
}
#[test] fn fuzz_avif_decode() {
    fuzz("avif", N, |b| ok(wai::codecs::image::avif_decode(b)));
}
#[test] fn fuzz_jxl_decode() {
    fuzz("jxl", N, |b| ok(wai::codecs::image::jxl_decode(b)));
}
#[test] fn fuzz_flac_decode() {
    fuzz("flac", N, |b| ok(wai::codecs::audio::flac_decode(b)));
}
#[test] fn fuzz_opus_decode() {
    fuzz("opus", N, |b| ok(wai::codecs::audio::opus_decode(b)));
}
#[test] fn fuzz_zstd_decode() {
    fuzz("zstd", N, |b| ok(wai::codecs::text::zstd_decode(b)));
}
#[test] fn fuzz_xz_decode() {
    fuzz("xz", N, |b| ok(wai::codecs::text::xz_decode(b)));
}
#[test] fn fuzz_av1_decode() {
    // av1_decode has explicit bounds checks (the av1_decode_garbage
    // robustness test caught the dav1d-abort path); confirm under random
    // load too.
    fuzz("av1", N, |b| ok(wai::codecs::video::av1_decode(b)));
}
#[test] fn fuzz_envelope_unpack() {
    fuzz("envelope", N, |b| ok(wai::container::Wai::from_bytes(b)));
}
