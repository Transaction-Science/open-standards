//! C ABI for the native neural sink (behind `feature = "neural"`).
//!
//! Three entry points, one per media class:
//!
//!   wai_neural_decode_audio(envelope, len, model_path, *samples, *sr)
//!   wai_neural_decode_image(envelope, len, model_path, *rgb, *w, *h)
//!   wai_neural_decode_video(envelope, len, model_path,
//!                            *frames_concat, *n_frames, *w, *h, *fps_x_1000)
//!
//! Each caller passes the model path as a NUL-terminated C string;
//! that lets a C consumer point at any deployer-installed ONNX file
//! without re-linking. The Rust side picks the right decoder by parsing
//! the envelope's capability and validating media class matches.
//!
//! Output buffers are heap-allocated here; free with `wai_buffer_free`.

use std::ffi::{c_char, c_int, CStr};
use std::panic;
use std::path::Path;
use std::slice;

use crate::container::Wai;
use crate::ffi::{WaiBuffer, WAI_ERR_CODEC, WAI_ERR_CONTAINER, WAI_ERR_INVALID_ARG,
                 WAI_ERR_PANIC, WAI_OK};
use crate::neural::{decode_envelope, Decoded, ModelRegistry};

// Neural FFI currently returns only status codes (no chained
// `wai_last_error` payload). Rationale: `set_err` in the main FFI module
// is private to that module, and forwarding the rich Rust-side error
// would require either making it `pub(crate)` or duplicating the
// thread-local. Status codes carry the essential branch info
// (invalid arg / container / codec / panic); for richer diagnostics the
// caller can drop into the Rust-side `wai::neural` API directly.

unsafe fn input_envelope<'a>(bytes: *const u8, len: usize) -> Option<&'a [u8]> {
    if bytes.is_null() && len > 0 { return None; }
    if bytes.is_null() { return Some(&[]); }
    Some(unsafe { slice::from_raw_parts(bytes, len) })
}

unsafe fn input_cstr<'a>(s: *const c_char) -> Option<&'a Path> {
    if s.is_null() { return None; }
    let cstr = unsafe { CStr::from_ptr(s) };
    cstr.to_str().ok().map(Path::new)
}

fn run<F: FnOnce() -> Result<(), c_int> + panic::UnwindSafe>(f: F) -> c_int {
    match panic::catch_unwind(f) {
        Ok(Ok(())) => WAI_OK,
        Ok(Err(code)) => code,
        Err(_) => WAI_ERR_PANIC,
    }
}

/// Decode a `wai.neural.<audio>` envelope (encodec32 / dac / mimi /
/// wavtokenizer). Writes mono `f32` samples into `out_samples`; the
/// caller reads them as `float*` of length `out_samples.len / 4`.
///
/// # Safety
/// `envelope` valid for `envelope_len` bytes. `model_path` NUL-terminated
/// UTF-8 path to the decoder.onnx. `out_samples` writable `WaiBuffer*`.
/// `out_sample_rate` writable `uint32_t*` (or null to skip).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_neural_decode_audio(
    envelope: *const u8, envelope_len: usize, model_path: *const c_char,
    out_samples: *mut WaiBuffer, out_sample_rate: *mut u32,
) -> c_int {
    if out_samples.is_null() { return WAI_ERR_INVALID_ARG; }
    run(|| {
        let bytes = unsafe { input_envelope(envelope, envelope_len) }
            .ok_or(WAI_ERR_INVALID_ARG)?;
        let path = unsafe { input_cstr(model_path) }.ok_or(WAI_ERR_INVALID_ARG)?;
        let env = Wai::from_bytes(bytes).map_err(|_| WAI_ERR_CONTAINER)?;
        let reg = ModelRegistry::new()
            .register(env.manifest.model_requirement.capability.clone(), path);
        let dec = decode_envelope(&env, &reg).map_err(|_| WAI_ERR_CODEC)?;
        let Decoded::Audio(audio) = dec else { return Err(WAI_ERR_CONTAINER); };
        // f32 samples → bytes (LE on supported targets; native byte order).
        let byte_len = audio.samples.len() * std::mem::size_of::<f32>();
        let mut buf = Vec::<u8>::with_capacity(byte_len);
        unsafe {
            std::ptr::copy_nonoverlapping(
                audio.samples.as_ptr() as *const u8, buf.as_mut_ptr(), byte_len);
            buf.set_len(byte_len);
        }
        unsafe { *out_samples = WaiBuffer::from_vec(buf); }
        if !out_sample_rate.is_null() { unsafe { *out_sample_rate = audio.sample_rate; } }
        Ok(())
    })
}

/// Decode a `wai.neural.bmshj2018` envelope. Writes packed RGB
/// (row-major, 3 bytes/pixel) into `out_rgb`; writes width/height
/// into `*out_width` / `*out_height`.
///
/// # Safety
/// See `wai_neural_decode_audio`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_neural_decode_image(
    envelope: *const u8, envelope_len: usize, model_path: *const c_char,
    out_rgb: *mut WaiBuffer, out_width: *mut u32, out_height: *mut u32,
) -> c_int {
    if out_rgb.is_null() { return WAI_ERR_INVALID_ARG; }
    run(|| {
        let bytes = unsafe { input_envelope(envelope, envelope_len) }
            .ok_or(WAI_ERR_INVALID_ARG)?;
        let path = unsafe { input_cstr(model_path) }.ok_or(WAI_ERR_INVALID_ARG)?;
        let env = Wai::from_bytes(bytes).map_err(|_| WAI_ERR_CONTAINER)?;
        let reg = ModelRegistry::new()
            .register(env.manifest.model_requirement.capability.clone(), path);
        let dec = decode_envelope(&env, &reg).map_err(|_| WAI_ERR_CODEC)?;
        let Decoded::Image(img) = dec else { return Err(WAI_ERR_CONTAINER); };
        unsafe {
            *out_rgb = WaiBuffer::from_vec(img.rgb);
            if !out_width.is_null()  { *out_width  = img.width;  }
            if !out_height.is_null() { *out_height = img.height; }
        }
        Ok(())
    })
}

/// Decode a `wai.neural.video_bmshj2018` envelope. Writes all frames
/// concatenated (frame 0 RGB, then frame 1 RGB, …) into
/// `out_frames_concat`; the caller splits at `width*height*3` strides.
///
/// # Safety
/// See `wai_neural_decode_audio`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_neural_decode_video(
    envelope: *const u8, envelope_len: usize, model_path: *const c_char,
    out_frames_concat: *mut WaiBuffer,
    out_n_frames: *mut u32, out_width: *mut u32, out_height: *mut u32,
    out_fps_x_1000: *mut u32,
) -> c_int {
    if out_frames_concat.is_null() { return WAI_ERR_INVALID_ARG; }
    run(|| {
        let bytes = unsafe { input_envelope(envelope, envelope_len) }
            .ok_or(WAI_ERR_INVALID_ARG)?;
        let path = unsafe { input_cstr(model_path) }.ok_or(WAI_ERR_INVALID_ARG)?;
        let env = Wai::from_bytes(bytes).map_err(|_| WAI_ERR_CONTAINER)?;
        let reg = ModelRegistry::new()
            .register(env.manifest.model_requirement.capability.clone(), path);
        let dec = decode_envelope(&env, &reg).map_err(|_| WAI_ERR_CODEC)?;
        let Decoded::Video(v) = dec else { return Err(WAI_ERR_CONTAINER); };
        let frame_bytes = (v.width as usize) * (v.height as usize) * 3;
        let mut concat = Vec::with_capacity(frame_bytes * v.frames_rgb.len());
        for f in &v.frames_rgb { concat.extend_from_slice(f); }
        unsafe {
            *out_frames_concat = WaiBuffer::from_vec(concat);
            if !out_n_frames.is_null()    { *out_n_frames    = v.frames_rgb.len() as u32; }
            if !out_width.is_null()       { *out_width       = v.width;  }
            if !out_height.is_null()      { *out_height      = v.height; }
            if !out_fps_x_1000.is_null()  { *out_fps_x_1000  = (v.fps * 1000.0).round() as u32; }
        }
        Ok(())
    })
}
