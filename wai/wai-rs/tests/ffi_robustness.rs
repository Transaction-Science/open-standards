//! Robustness tests for the C FFI surface. These exercise exactly the
//! mistakes an SDK consumer is most likely to make when calling
//! `libwai.dylib` from Python/Node/Swift/etc.: null pointers, zero
//! lengths, mismatched dimensions, truncated payloads, malformed magic.
//!
//! Every wai_* function MUST return a non-zero status code (never panic
//! into C — that's undefined behavior) and leave its output `WaiBuffer*`
//! cleared on any error.

use std::ptr;

use wai::ffi::*;

// ---- helpers --------------------------------------------------------

fn empty_buf() -> WaiBuffer {
    WaiBuffer { data: ptr::null_mut(), len: 0, cap: 0 }
}

fn synth_rgb(h: u32, w: u32) -> Vec<u8> {
    let mut v = vec![0u8; (h * w * 3) as usize];
    for i in 0..h {
        for j in 0..w {
            let p = ((i * w + j) * 3) as usize;
            v[p] = ((i + j) as u8).wrapping_mul(4);
            v[p + 1] = (i as u8).wrapping_mul(8);
            v[p + 2] = (j as u8).wrapping_mul(8);
        }
    }
    v
}

// ---- null / mismatched inputs --------------------------------------

#[test]
fn png_encode_null_out() {
    let rgb = synth_rgb(8, 8);
    let rc = unsafe {
        wai_image_png_encode(rgb.as_ptr(), rgb.len(), 8, 8, ptr::null_mut())
    };
    assert_eq!(rc, WAI_ERR_INVALID_ARG, "null out must reject");
}

#[test]
fn png_encode_len_mismatch() {
    let rgb = synth_rgb(8, 8);     // 192 bytes
    let mut out = empty_buf();
    let rc = unsafe {
        // claim 16×16 but pass 8×8 buffer
        wai_image_png_encode(rgb.as_ptr(), rgb.len(), 16, 16, &mut out)
    };
    assert_eq!(rc, WAI_ERR_INVALID_ARG, "len mismatch must reject");
    assert!(out.data.is_null(), "out must be empty on error");
}

#[test]
fn png_encode_null_data_with_nonzero_len() {
    let mut out = empty_buf();
    let rc = unsafe {
        wai_image_png_encode(ptr::null(), 100, 8, 8, &mut out)
    };
    assert_eq!(rc, WAI_ERR_INVALID_ARG);
    assert!(out.data.is_null());
}

#[test]
fn png_encode_zero_dimensions_overflow_safe() {
    let mut out = empty_buf();
    // h*w*3 of u32::MAX should overflow detection, not produce UB
    let rc = unsafe {
        wai_image_png_encode(ptr::null(), 0, u32::MAX, u32::MAX, &mut out)
    };
    assert_eq!(rc, WAI_ERR_INVALID_ARG, "huge dims must be caught");
}

#[test]
fn png_decode_garbage_bytes() {
    let garbage = b"not a png file at all".to_vec();
    let mut out = empty_buf();
    let mut h = 0u32; let mut w = 0u32;
    let rc = unsafe {
        wai_image_png_decode(garbage.as_ptr(), garbage.len(),
                             &mut out, &mut h, &mut w)
    };
    assert_ne!(rc, WAI_OK, "garbage must not decode as PNG");
    assert!(out.data.is_null());
    assert_eq!((h, w), (0, 0), "dimensions must be zeroed on error");
}

#[test]
fn png_decode_truncated_png() {
    // start of a valid PNG signature but cut off
    let trunc = vec![0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A, 0x00];
    let mut out = empty_buf();
    let mut h = 0u32; let mut w = 0u32;
    let rc = unsafe {
        wai_image_png_decode(trunc.as_ptr(), trunc.len(),
                             &mut out, &mut h, &mut w)
    };
    assert_ne!(rc, WAI_OK);
}

#[test]
fn png_decode_null_size_out() {
    let mut out = empty_buf();
    let rc = unsafe {
        wai_image_png_decode(b"x".as_ptr(), 1, &mut out, ptr::null_mut(), ptr::null_mut())
    };
    assert_eq!(rc, WAI_ERR_INVALID_ARG);
}

// ---- audio: FLAC / Opus -------------------------------------------

#[test]
fn flac_decode_garbage() {
    let garbage = vec![0u8; 64];
    let mut out = empty_buf();
    let mut sr = 0u32;
    let rc = unsafe {
        wai_audio_flac_decode(garbage.as_ptr(), garbage.len(), &mut out, &mut sr)
    };
    assert_ne!(rc, WAI_OK);
    assert_eq!(sr, 0);
}

#[test]
fn opus_encode_invalid_sr() {
    let samples = vec![0f32; 1000];
    let mut out = empty_buf();
    // 44100 isn't an opus sr; must fail cleanly (not panic)
    let rc = unsafe {
        wai_audio_opus_encode(samples.as_ptr(), samples.len(),
                              44100, 64_000, &mut out)
    };
    assert_eq!(rc, WAI_ERR_CODEC);
    assert!(out.data.is_null());
}

#[test]
fn opus_decode_garbage() {
    let garbage = vec![0u8; 16];
    let mut out = empty_buf();
    let mut sr = 0u32;
    let rc = unsafe {
        wai_audio_opus_decode(garbage.as_ptr(), garbage.len(), &mut out, &mut sr)
    };
    assert_ne!(rc, WAI_OK);
}

// ---- text: zstd / xz ----------------------------------------------

#[test]
fn zstd_decode_garbage() {
    let garbage = b"this is not a zstd stream".to_vec();
    let mut out = empty_buf();
    let rc = unsafe {
        wai_text_zstd_decode(garbage.as_ptr(), garbage.len(), &mut out)
    };
    assert_ne!(rc, WAI_OK);
}

#[test]
fn xz_decode_garbage() {
    let garbage = vec![0u8; 32];
    let mut out = empty_buf();
    let rc = unsafe {
        wai_text_xz_decode(garbage.as_ptr(), garbage.len(), &mut out)
    };
    assert_ne!(rc, WAI_OK);
}

#[test]
fn zstd_round_trip_empty_input() {
    let mut enc = empty_buf();
    let rc = unsafe { wai_text_zstd_encode(ptr::null(), 0, 19, &mut enc) };
    assert_eq!(rc, WAI_OK, "encoding zero bytes is valid");
    let mut dec = empty_buf();
    let rc = unsafe { wai_text_zstd_decode(enc.data, enc.len, &mut dec) };
    assert_eq!(rc, WAI_OK);
    assert_eq!(dec.len, 0);
    unsafe { wai_buffer_free(enc); wai_buffer_free(dec); }
}

// ---- envelope ------------------------------------------------------

#[test]
fn envelope_unpack_bad_magic() {
    let bad = b"NOTWAI...".to_vec();
    let mut m = empty_buf();
    let mut p = empty_buf();
    let rc = unsafe {
        wai_envelope_unpack(bad.as_ptr(), bad.len(), &mut m, &mut p)
    };
    assert_eq!(rc, WAI_ERR_CONTAINER);
    assert!(m.data.is_null() && p.data.is_null());
}

#[test]
fn envelope_unpack_truncated() {
    // just the magic, nothing else
    let trunc = b"WAI1".to_vec();
    let mut m = empty_buf();
    let mut p = empty_buf();
    let rc = unsafe {
        wai_envelope_unpack(trunc.as_ptr(), trunc.len(), &mut m, &mut p)
    };
    assert_ne!(rc, WAI_OK);
}

#[test]
fn envelope_pack_bad_manifest_json() {
    let bad_json = std::ffi::CString::new("{not a valid json}").unwrap();
    let mut out = empty_buf();
    let rc = unsafe {
        wai_envelope_pack(bad_json.as_ptr(), ptr::null(), 0, &mut out)
    };
    assert_eq!(rc, WAI_ERR_CONTAINER);
    assert!(out.data.is_null());
}

#[test]
fn envelope_pack_null_manifest() {
    let mut out = empty_buf();
    let rc = unsafe {
        wai_envelope_pack(ptr::null(), ptr::null(), 0, &mut out)
    };
    assert_eq!(rc, WAI_ERR_INVALID_ARG);
}

// ---- video ---------------------------------------------------------

#[test]
fn av1_encode_zero_fps() {
    let rgb = synth_rgb(16, 16);
    let mut out = empty_buf();
    let rc = unsafe {
        wai_video_av1_encode(rgb.as_ptr(), rgb.len(),
                             1, 16, 16, 0, 1, 0, 50, &mut out)
    };
    assert_eq!(rc, WAI_ERR_INVALID_ARG, "zero fps_num must be rejected");
}

#[test]
fn av1_decode_garbage() {
    let garbage = vec![0u8; 64];
    let mut out = empty_buf();
    let mut n = 0u32; let mut h = 0u32; let mut w = 0u32;
    let mut fn_ = 0u32; let mut fd_ = 0u32;
    let rc = unsafe {
        wai_video_av1_decode(garbage.as_ptr(), garbage.len(),
                             &mut out, &mut n, &mut h, &mut w, &mut fn_, &mut fd_)
    };
    assert_ne!(rc, WAI_OK);
}

// ---- happy-path sanity (1 per codec to confirm we didn't break it) -

#[test]
fn png_round_trip_happy() {
    let rgb = synth_rgb(16, 16);
    let mut enc = empty_buf();
    let rc = unsafe { wai_image_png_encode(rgb.as_ptr(), rgb.len(), 16, 16, &mut enc) };
    assert_eq!(rc, WAI_OK);
    let mut dec = empty_buf();
    let mut h = 0u32; let mut w = 0u32;
    let rc = unsafe { wai_image_png_decode(enc.data, enc.len, &mut dec, &mut h, &mut w) };
    assert_eq!(rc, WAI_OK);
    assert_eq!((h, w), (16, 16));
    let got = unsafe { std::slice::from_raw_parts(dec.data, dec.len) };
    assert_eq!(got, rgb.as_slice(), "PNG happy-path must be bit-exact");
    unsafe { wai_buffer_free(enc); wai_buffer_free(dec); }
}

#[test]
fn last_error_returns_something_after_failure() {
    let mut out = empty_buf();
    // force an error
    let _ = unsafe { wai_image_png_decode(b"x".as_ptr(), 1, &mut out,
                                          std::ptr::null_mut(), std::ptr::null_mut()) };
    let p = wai_last_error();
    assert!(!p.is_null(), "wai_last_error must be set after a failed call");
}
