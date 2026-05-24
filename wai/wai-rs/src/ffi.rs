//! C ABI for SDK consumers. Stable, language-agnostic surface — every
//! other-language binding (Python ctypes, Node N-API, Swift, JNI, etc.)
//! talks to this. The Rust crate is built as `cdylib`/`staticlib` so it
//! drops in as `libwai.so`/`libwai.dylib`/`wai.dll`.
//!
//! ## Status codes (returned by every wai_* function)
//! - `0` = ok
//! - `1` = invalid argument (null pointer where one shouldn't be, etc.)
//! - `2` = encode/decode failed (codec-level error; details in
//!   `wai_last_error`)
//! - `3` = container/envelope error (bad magic, truncated, bad JSON)
//! - `4` = internal panic (a bug in the wrapper — we catch it instead
//!   of unwinding into C, which would be UB)
//!
//! ## Memory model
//! All byte buffers crossing the boundary are heap-allocated here and
//! must be freed with `wai_buffer_free`. Errors are returned as a
//! non-zero status code; `wai_last_error` returns a null-terminated
//! UTF-8 string of the most recent error per thread.

use std::cell::RefCell;
use std::ffi::{c_char, c_int, CString};
use std::panic;
use std::slice;

use crate::codecs;

pub const WAI_OK: c_int = 0;
pub const WAI_ERR_INVALID_ARG: c_int = 1;
pub const WAI_ERR_CODEC: c_int = 2;
pub const WAI_ERR_CONTAINER: c_int = 3;
pub const WAI_ERR_PANIC: c_int = 4;

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_err(e: impl ToString) {
    let cs = CString::new(e.to_string())
        .unwrap_or_else(|_| CString::new("invalid utf-8 error").unwrap());
    LAST_ERROR.with(|c| *c.borrow_mut() = Some(cs));
}

#[repr(C)]
pub struct WaiBuffer {
    pub data: *mut u8,
    pub len: usize,
    pub cap: usize,
}

impl WaiBuffer {
    pub(crate) fn from_vec(mut v: Vec<u8>) -> Self {
        let data = v.as_mut_ptr();
        let len = v.len();
        let cap = v.capacity();
        std::mem::forget(v);
        WaiBuffer { data, len, cap }
    }
    fn empty() -> Self { WaiBuffer { data: std::ptr::null_mut(), len: 0, cap: 0 } }
}

/// Free a buffer previously returned by any `wai_*_encode` /
/// `wai_*_decode` / `wai_envelope_*` call. Safe to call with a null
/// or zero-cap buffer.
///
/// # Safety
/// `buf.data` must have been produced by this crate.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_buffer_free(buf: WaiBuffer) {
    if !buf.data.is_null() && buf.cap > 0 {
        drop(unsafe { Vec::from_raw_parts(buf.data, buf.len, buf.cap) });
    }
}

/// Returns a pointer to the most recent error message for this thread,
/// or null if there is none. The pointer is valid until the next FFI
/// call on this thread that sets an error.
#[unsafe(no_mangle)]
pub extern "C" fn wai_last_error() -> *const c_char {
    LAST_ERROR.with(|c| match &*c.borrow() {
        Some(cs) => cs.as_ptr(),
        None => std::ptr::null(),
    })
}

// ---- generic helpers ------------------------------------------------
unsafe fn input_slice<'a>(data: *const u8, len: usize) -> Option<&'a [u8]> {
    match (data.is_null(), len) {
        (true, 0) => Some(&[]),
        (true, _) => None,                          // null but non-zero len
        (false, _) => Some(unsafe { slice::from_raw_parts(data, len) }),
    }
}

unsafe fn write_buf(out: *mut WaiBuffer, v: Vec<u8>) -> bool {
    if out.is_null() { return false; }
    unsafe { *out = WaiBuffer::from_vec(v); }
    true
}

unsafe fn clear_buf(out: *mut WaiBuffer) {
    if !out.is_null() {
        unsafe { *out = WaiBuffer::empty(); }
    }
}

/// Wraps an FFI body. Validates the `out` buffer pointer is non-null,
/// runs the closure inside `catch_unwind` (so a Rust panic returns
/// `WAI_ERR_PANIC` instead of unwinding into C — which would be UB),
/// and clears `out` to an empty buffer on any error path.
fn ffi_buf_call<F>(out: *mut WaiBuffer, f: F) -> c_int
where F: FnOnce() -> std::result::Result<Vec<u8>, (c_int, String)> + panic::UnwindSafe
{
    if out.is_null() {
        set_err("output buffer pointer is null");
        return WAI_ERR_INVALID_ARG;
    }
    match panic::catch_unwind(f) {
        Ok(Ok(v)) => unsafe {
            if write_buf(out, v) { WAI_OK }
            else { set_err("write to out failed"); WAI_ERR_INVALID_ARG }
        }
        Ok(Err((code, msg))) => {
            set_err(msg);
            unsafe { clear_buf(out); }
            code
        }
        Err(p) => {
            let msg = if let Some(s) = p.downcast_ref::<&str>() { (*s).to_string() }
                      else if let Some(s) = p.downcast_ref::<String>() { s.clone() }
                      else { "panic with non-string payload".into() };
            set_err(format!("internal panic: {msg}"));
            unsafe { clear_buf(out); }
            WAI_ERR_PANIC
        }
    }
}

fn ffi_codec<T, F>(f: F) -> std::result::Result<Vec<u8>, (c_int, String)>
where F: FnOnce() -> std::result::Result<T, String>,
      T: Into<Vec<u8>>,
{
    f().map(Into::into).map_err(|e| (WAI_ERR_CODEC, e))
}

// ---- image: PNG -----------------------------------------------------
/// Encode RGB pixels (row-major, 3 bytes per pixel) as PNG.
///
/// # Safety
/// `rgb` must point to a valid RGB buffer of at least `len` bytes;
/// `out` must be a writable `WaiBuffer*`. Pass `(null, 0)` for empty.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_image_png_encode(
    rgb: *const u8, len: usize, h: u32, w: u32, out: *mut WaiBuffer,
) -> c_int {
    ffi_buf_call(out, || {
        let s = unsafe { input_slice(rgb, len) }
            .ok_or((WAI_ERR_INVALID_ARG, "rgb pointer null with non-zero len".into()))?;
        let expected = (h as usize)
            .checked_mul(w as usize)
            .and_then(|n| n.checked_mul(3))
            .ok_or((WAI_ERR_INVALID_ARG, "h*w*3 overflows".into()))?;
        if s.len() != expected {
            return Err((WAI_ERR_INVALID_ARG,
                format!("rgb len {} != h*w*3 = {}", s.len(), expected)));
        }
        ffi_codec(|| codecs::image::png_encode(s, h, w)
            .map_err(|e| format!("{e:?}")))
    })
}

/// Decode a PNG byte stream to RGB. Returns `(rgb, h, w)`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_image_png_decode(
    bytes: *const u8, len: usize, out: *mut WaiBuffer,
    out_h: *mut u32, out_w: *mut u32,
) -> c_int {
    if out_h.is_null() || out_w.is_null() {
        set_err("out_h/out_w null");
        return WAI_ERR_INVALID_ARG;
    }
    decode_to_rgb_size(out, out_h, out_w, || {
        let s = unsafe { input_slice(bytes, len) }
            .ok_or((WAI_ERR_INVALID_ARG, "bytes pointer null with non-zero len".into()))?;
        codecs::image::png_decode(s)
            .map_err(|e| (WAI_ERR_CODEC, format!("{e:?}")))
    })
}

fn decode_to_rgb_size<F>(out: *mut WaiBuffer, out_h: *mut u32, out_w: *mut u32, f: F) -> c_int
where F: FnOnce() -> std::result::Result<(Vec<u8>, u32, u32), (c_int, String)> + panic::UnwindSafe
{
    if out.is_null() {
        set_err("out null"); return WAI_ERR_INVALID_ARG;
    }
    match panic::catch_unwind(f) {
        Ok(Ok((v, h, w))) => unsafe {
            *out = WaiBuffer::from_vec(v); *out_h = h; *out_w = w; WAI_OK
        }
        Ok(Err((code, msg))) => {
            set_err(msg);
            unsafe { clear_buf(out); *out_h = 0; *out_w = 0; }
            code
        }
        Err(p) => {
            let msg = if let Some(s) = p.downcast_ref::<&str>() { (*s).to_string() }
                      else if let Some(s) = p.downcast_ref::<String>() { s.clone() }
                      else { "panic with non-string payload".into() };
            set_err(format!("internal panic: {msg}"));
            unsafe { clear_buf(out); *out_h = 0; *out_w = 0; }
            WAI_ERR_PANIC
        }
    }
}

// ---- image: AVIF ----------------------------------------------------
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_image_avif_encode(
    rgb: *const u8, len: usize, h: u32, w: u32, quality: u8,
    out: *mut WaiBuffer,
) -> c_int {
    ffi_buf_call(out, || {
        let s = unsafe { input_slice(rgb, len) }
            .ok_or((WAI_ERR_INVALID_ARG, "rgb null + len>0".into()))?;
        let expected = (h as usize).checked_mul(w as usize)
            .and_then(|n| n.checked_mul(3))
            .ok_or((WAI_ERR_INVALID_ARG, "h*w*3 overflows".into()))?;
        if s.len() != expected {
            return Err((WAI_ERR_INVALID_ARG,
                format!("rgb len {} != h*w*3 = {}", s.len(), expected)));
        }
        codecs::image::avif_encode(s, h, w, quality)
            .map_err(|e| (WAI_ERR_CODEC, format!("{e:?}")))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_image_avif_decode(
    bytes: *const u8, len: usize, out: *mut WaiBuffer,
    out_h: *mut u32, out_w: *mut u32,
) -> c_int {
    if out_h.is_null() || out_w.is_null() {
        set_err("out_h/out_w null"); return WAI_ERR_INVALID_ARG;
    }
    decode_to_rgb_size(out, out_h, out_w, || {
        let s = unsafe { input_slice(bytes, len) }
            .ok_or((WAI_ERR_INVALID_ARG, "bytes null + len>0".into()))?;
        codecs::image::avif_decode(s)
            .map_err(|e| (WAI_ERR_CODEC, format!("{e:?}")))
    })
}

// ---- image: JPEG-XL -------------------------------------------------
/// `quality = 0` ⇒ lossless. Otherwise butteraugli distance proxy
/// (higher quality = smaller distance internally).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_image_jxl_encode(
    rgb: *const u8, len: usize, h: u32, w: u32, quality: u8,
    out: *mut WaiBuffer,
) -> c_int {
    ffi_buf_call(out, || {
        let s = unsafe { input_slice(rgb, len) }
            .ok_or((WAI_ERR_INVALID_ARG, "rgb null + len>0".into()))?;
        let expected = (h as usize).checked_mul(w as usize)
            .and_then(|n| n.checked_mul(3))
            .ok_or((WAI_ERR_INVALID_ARG, "h*w*3 overflows".into()))?;
        if s.len() != expected {
            return Err((WAI_ERR_INVALID_ARG,
                format!("rgb len {} != h*w*3 = {}", s.len(), expected)));
        }
        let q = if quality == 0 { None } else { Some(quality as f32) };
        codecs::image::jxl_encode(s, h, w, q)
            .map_err(|e| (WAI_ERR_CODEC, format!("{e:?}")))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_image_jxl_decode(
    bytes: *const u8, len: usize, out: *mut WaiBuffer,
    out_h: *mut u32, out_w: *mut u32,
) -> c_int {
    if out_h.is_null() || out_w.is_null() {
        set_err("out_h/out_w null"); return WAI_ERR_INVALID_ARG;
    }
    decode_to_rgb_size(out, out_h, out_w, || {
        let s = unsafe { input_slice(bytes, len) }
            .ok_or((WAI_ERR_INVALID_ARG, "bytes null + len>0".into()))?;
        codecs::image::jxl_decode(s)
            .map_err(|e| (WAI_ERR_CODEC, format!("{e:?}")))
    })
}

// ---- audio: Opus ----------------------------------------------------
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_audio_opus_encode(
    samples: *const f32, n: usize, sr: u32, bitrate_bps: i32,
    out: *mut WaiBuffer,
) -> c_int {
    ffi_buf_call(out, || {
        if samples.is_null() && n != 0 {
            return Err((WAI_ERR_INVALID_ARG, "samples null + n>0".into()));
        }
        let s = if n == 0 { &[][..] }
                else { unsafe { slice::from_raw_parts(samples, n) } };
        codecs::audio::opus_encode(s, sr, bitrate_bps)
            .map_err(|e| (WAI_ERR_CODEC, format!("{e:?}")))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_audio_opus_decode(
    bytes: *const u8, len: usize, out: *mut WaiBuffer, out_sr: *mut u32,
) -> c_int {
    if out_sr.is_null() {
        set_err("out_sr null"); return WAI_ERR_INVALID_ARG;
    }
    if out.is_null() {
        set_err("out null"); return WAI_ERR_INVALID_ARG;
    }
    match panic::catch_unwind(|| -> std::result::Result<(Vec<f32>, u32), (c_int, String)> {
        let s = unsafe { input_slice(bytes, len) }
            .ok_or((WAI_ERR_INVALID_ARG, "bytes null + len>0".into()))?;
        codecs::audio::opus_decode(s)
            .map_err(|e| (WAI_ERR_CODEC, format!("{e:?}")))
    }) {
        Ok(Ok((samples, sr))) => {
            let bytes: Vec<u8> = samples.iter().flat_map(|f| f.to_le_bytes()).collect();
            unsafe { *out = WaiBuffer::from_vec(bytes); *out_sr = sr; }
            WAI_OK
        }
        Ok(Err((code, msg))) => {
            set_err(msg);
            unsafe { clear_buf(out); *out_sr = 0; }
            code
        }
        Err(_) => { set_err("internal panic in opus_decode");
            unsafe { clear_buf(out); *out_sr = 0; } WAI_ERR_PANIC }
    }
}

// ---- audio: FLAC ----------------------------------------------------
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_audio_flac_encode(
    samples: *const f32, n: usize, sr: u32, out: *mut WaiBuffer,
) -> c_int {
    ffi_buf_call(out, || {
        if samples.is_null() && n != 0 {
            return Err((WAI_ERR_INVALID_ARG, "samples null + n>0".into()));
        }
        let s = if n == 0 { &[][..] }
                else { unsafe { slice::from_raw_parts(samples, n) } };
        codecs::audio::flac_encode(s, sr)
            .map_err(|e| (WAI_ERR_CODEC, format!("{e:?}")))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_audio_flac_decode(
    bytes: *const u8, len: usize, out: *mut WaiBuffer, out_sr: *mut u32,
) -> c_int {
    if out_sr.is_null() || out.is_null() {
        set_err("out/out_sr null"); return WAI_ERR_INVALID_ARG;
    }
    match panic::catch_unwind(|| -> std::result::Result<(Vec<f32>, u32), (c_int, String)> {
        let s = unsafe { input_slice(bytes, len) }
            .ok_or((WAI_ERR_INVALID_ARG, "bytes null + len>0".into()))?;
        codecs::audio::flac_decode(s)
            .map_err(|e| (WAI_ERR_CODEC, format!("{e:?}")))
    }) {
        Ok(Ok((samples, sr))) => {
            let bytes: Vec<u8> = samples.iter().flat_map(|f| f.to_le_bytes()).collect();
            unsafe { *out = WaiBuffer::from_vec(bytes); *out_sr = sr; }
            WAI_OK
        }
        Ok(Err((code, msg))) => {
            set_err(msg);
            unsafe { clear_buf(out); *out_sr = 0; }
            code
        }
        Err(_) => { set_err("internal panic in flac_decode");
            unsafe { clear_buf(out); *out_sr = 0; } WAI_ERR_PANIC }
    }
}

// ---- text: zstd / xz ------------------------------------------------
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_text_zstd_encode(
    data: *const u8, len: usize, level: i32, out: *mut WaiBuffer,
) -> c_int {
    ffi_buf_call(out, || {
        let s = unsafe { input_slice(data, len) }
            .ok_or((WAI_ERR_INVALID_ARG, "data null + len>0".into()))?;
        codecs::text::zstd_encode(s, level)
            .map_err(|e| (WAI_ERR_CODEC, format!("{e:?}")))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_text_zstd_decode(
    bytes: *const u8, len: usize, out: *mut WaiBuffer,
) -> c_int {
    ffi_buf_call(out, || {
        let s = unsafe { input_slice(bytes, len) }
            .ok_or((WAI_ERR_INVALID_ARG, "bytes null + len>0".into()))?;
        codecs::text::zstd_decode(s)
            .map_err(|e| (WAI_ERR_CODEC, format!("{e:?}")))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_text_xz_encode(
    data: *const u8, len: usize, level: u32, out: *mut WaiBuffer,
) -> c_int {
    ffi_buf_call(out, || {
        let s = unsafe { input_slice(data, len) }
            .ok_or((WAI_ERR_INVALID_ARG, "data null + len>0".into()))?;
        codecs::text::xz_encode(s, level)
            .map_err(|e| (WAI_ERR_CODEC, format!("{e:?}")))
    })
}

#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_text_xz_decode(
    bytes: *const u8, len: usize, out: *mut WaiBuffer,
) -> c_int {
    ffi_buf_call(out, || {
        let s = unsafe { input_slice(bytes, len) }
            .ok_or((WAI_ERR_INVALID_ARG, "bytes null + len>0".into()))?;
        codecs::text::xz_decode(s)
            .map_err(|e| (WAI_ERR_CODEC, format!("{e:?}")))
    })
}

// ---- video: AV1 -----------------------------------------------------
/// Encode a sequence of `n` RGB frames as AV1.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_video_av1_encode(
    rgb: *const u8, len: usize,
    n: u32, h: u32, w: u32, fps_num: u32, fps_den: u32,
    lossless: u8, quality: u8,
    out: *mut WaiBuffer,
) -> c_int {
    ffi_buf_call(out, || {
        let s = unsafe { input_slice(rgb, len) }
            .ok_or((WAI_ERR_INVALID_ARG, "rgb null + len>0".into()))?;
        let per_frame = (h as usize).checked_mul(w as usize)
            .and_then(|x| x.checked_mul(3))
            .ok_or((WAI_ERR_INVALID_ARG, "h*w*3 overflows".into()))?;
        let total = (n as usize).checked_mul(per_frame)
            .ok_or((WAI_ERR_INVALID_ARG, "n*h*w*3 overflows".into()))?;
        if s.len() != total {
            return Err((WAI_ERR_INVALID_ARG,
                format!("video rgb len {} != n*h*w*3 = {}", s.len(), total)));
        }
        if fps_num == 0 || fps_den == 0 {
            return Err((WAI_ERR_INVALID_ARG, "fps_num/fps_den must be non-zero".into()));
        }
        let frames: Vec<Vec<u8>> = (0..n as usize)
            .map(|i| s[i * per_frame..(i + 1) * per_frame].to_vec())
            .collect();
        crate::codecs::video::av1_encode(&frames, h, w, fps_num, fps_den,
                                          lossless != 0, quality)
            .map_err(|e| (WAI_ERR_CODEC, format!("{e:?}")))
    })
}

/// Decode an AV1 WAI payload to a packed RGB buffer `n*h*w*3` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_video_av1_decode(
    bytes: *const u8, len: usize, out: *mut WaiBuffer,
    out_n: *mut u32, out_h: *mut u32, out_w: *mut u32,
    out_fps_num: *mut u32, out_fps_den: *mut u32,
) -> c_int {
    if out.is_null() || out_n.is_null() || out_h.is_null() || out_w.is_null()
        || out_fps_num.is_null() || out_fps_den.is_null() {
        set_err("video decode: one of the out pointers is null");
        return WAI_ERR_INVALID_ARG;
    }
    match panic::catch_unwind(|| -> std::result::Result<(Vec<Vec<u8>>, u32, u32, u32, u32), (c_int, String)> {
        let s = unsafe { input_slice(bytes, len) }
            .ok_or((WAI_ERR_INVALID_ARG, "bytes null + len>0".into()))?;
        crate::codecs::video::av1_decode(s)
            .map_err(|e| (WAI_ERR_CODEC, format!("{e:?}")))
    }) {
        Ok(Ok((frames, h, w, fps_num, fps_den))) => {
            let n = frames.len() as u32;
            let mut packed = Vec::with_capacity(frames.len() * (h * w * 3) as usize);
            for f in &frames { packed.extend_from_slice(f); }
            unsafe {
                *out = WaiBuffer::from_vec(packed);
                *out_n = n; *out_h = h; *out_w = w;
                *out_fps_num = fps_num; *out_fps_den = fps_den;
            }
            WAI_OK
        }
        Ok(Err((code, msg))) => {
            set_err(msg);
            unsafe {
                clear_buf(out);
                *out_n = 0; *out_h = 0; *out_w = 0;
                *out_fps_num = 0; *out_fps_den = 0;
            }
            code
        }
        Err(_) => {
            set_err("internal panic in av1_decode");
            unsafe {
                clear_buf(out);
                *out_n = 0; *out_h = 0; *out_w = 0;
                *out_fps_num = 0; *out_fps_den = 0;
            }
            WAI_ERR_PANIC
        }
    }
}

// ---- envelope: wrap an existing payload in a WAI container ----------
/// Builds a complete WAI file from a codec-produced payload + manifest
/// fields. `manifest_json` is a UTF-8 JSON object overriding/supplying
/// any fields; `payload` is the bytes from one of the codec encoders.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_envelope_pack(
    manifest_json: *const c_char,
    payload: *const u8, payload_len: usize,
    out: *mut WaiBuffer,
) -> c_int {
    ffi_buf_call(out, || {
        if manifest_json.is_null() {
            return Err((WAI_ERR_INVALID_ARG, "null manifest_json".into()));
        }
        let s = unsafe { std::ffi::CStr::from_ptr(manifest_json) }
            .to_string_lossy().to_string();
        let manifest: crate::container::Manifest = serde_json::from_str(&s)
            .map_err(|e| (WAI_ERR_CONTAINER, format!("manifest parse: {e}")))?;
        let pl = unsafe { input_slice(payload, payload_len) }
            .ok_or((WAI_ERR_INVALID_ARG, "payload null + len>0".into()))?
            .to_vec();
        let wai = crate::container::Wai::new(manifest, pl);
        wai.to_bytes()
            .map_err(|e| (WAI_ERR_CONTAINER, format!("envelope serialize: {e}")))
    })
}

// ---- v1.1: multi-rendition envelope FFI ------------------------
// Multi-rendition is a separate magic ("WAI2") with multiple payloads
// in one envelope. The C ABI keeps it consumer-friendly by serializing
// the rendition metadata (capability + kind + payload offset/length)
// as a JSON sidecar and packing every rendition payload contiguously
// into a single buffer — the caller indexes into the buffer using the
// JSON. No array-of-structs ownership games.

/// Detect WAI version from a byte stream. Returns 1 (WAI1), 2 (WAI2),
/// or 0 with an error message in `wai_last_error` for unknown magic.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_envelope_detect_version(
    bytes: *const u8, len: usize,
) -> c_int {
    let s = match unsafe { input_slice(bytes, len) } {
        Some(s) => s,
        None => { set_err("bytes null + len>0"); return 0; }
    };
    match crate::container::detect_version(s) {
        Ok(v) => v as c_int,
        Err(e) => { set_err(e); 0 }
    }
}

/// Pack a WAI2 multi-rendition envelope.
///
/// - `manifest_json` must declare a `renditions` array of length `n`.
/// - `payload_data` is an array of `n` byte-buffer pointers.
/// - `payload_lens` is the matching array of `n` lengths.
/// - The i-th payload pairs with `manifest.renditions[i]`.
///
/// # Safety
/// All array pointers must point to valid memory of the declared length.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_envelope_pack_multi(
    manifest_json: *const c_char,
    payload_data: *const *const u8,
    payload_lens: *const usize,
    n_payloads: usize,
    out: *mut WaiBuffer,
) -> c_int {
    ffi_buf_call(out, || {
        if manifest_json.is_null() {
            return Err((WAI_ERR_INVALID_ARG, "null manifest_json".into()));
        }
        let mj = unsafe { std::ffi::CStr::from_ptr(manifest_json) }
            .to_string_lossy().to_string();
        let manifest: crate::container::ManifestV2 = serde_json::from_str(&mj)
            .map_err(|e| (WAI_ERR_CONTAINER, format!("manifest parse: {e}")))?;
        if manifest.renditions.len() != n_payloads {
            return Err((WAI_ERR_INVALID_ARG, format!(
                "manifest has {} renditions but n_payloads={n_payloads}",
                manifest.renditions.len())));
        }
        if n_payloads > 0 && (payload_data.is_null() || payload_lens.is_null()) {
            return Err((WAI_ERR_INVALID_ARG,
                "payload_data/payload_lens null with n_payloads>0".into()));
        }
        // Collect rendition payloads. The native WaiMulti API takes
        // owned Vec<u8>, so copy each input buffer once here.
        let mut renditions: Vec<crate::container::Rendition> =
            Vec::with_capacity(n_payloads);
        for i in 0..n_payloads {
            let ptr = unsafe { *payload_data.add(i) };
            let len = unsafe { *payload_lens.add(i) };
            let s = unsafe { input_slice(ptr, len) }
                .ok_or((WAI_ERR_INVALID_ARG,
                        format!("payload {i} null with len>0")))?;
            renditions.push(crate::container::Rendition {
                capability: manifest.renditions[i].capability.clone(),
                kind: manifest.renditions[i].kind.clone(),
                payload: s.to_vec(),
            });
        }
        let wm = crate::container::WaiMulti::new(
            manifest.media, manifest.intent, manifest.target, renditions);
        wm.to_bytes()
            .map_err(|e| (WAI_ERR_CONTAINER, format!("envelope serialize: {e}")))
    })
}

/// Parse a WAI2 envelope. Writes three outputs:
///
/// - `out_manifest` — the original manifest JSON (UTF-8 bytes, no NUL)
/// - `out_renditions_json` — a JSON array of `{capability, kind, offset,
///   length}` entries. `offset` is into `out_payload_block`, `length`
///   is the rendition's payload byte count.
/// - `out_payload_block` — every rendition's payload bytes, packed
///   contiguously. The caller indexes via the JSON above.
///
/// This avoids exposing a C struct array (and the ownership pain that
/// comes with it). Any JSON parser the consumer already has decodes
/// the rendition table in two lines.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_envelope_unpack_multi(
    bytes: *const u8, len: usize,
    out_manifest: *mut WaiBuffer,
    out_renditions_json: *mut WaiBuffer,
    out_payload_block: *mut WaiBuffer,
) -> c_int {
    if out_manifest.is_null() || out_renditions_json.is_null()
        || out_payload_block.is_null()
    {
        set_err("unpack_multi: one of the out pointers is null");
        return WAI_ERR_INVALID_ARG;
    }
    match panic::catch_unwind(|| -> std::result::Result<(Vec<u8>, Vec<u8>, Vec<u8>), (c_int, String)> {
        let s = unsafe { input_slice(bytes, len) }
            .ok_or((WAI_ERR_INVALID_ARG, "bytes null + len>0".into()))?;
        let m = crate::container::WaiMulti::from_bytes(s)
            .map_err(|e| (WAI_ERR_CONTAINER, format!("envelope parse: {e}")))?;
        let mb = serde_json::to_vec(&m.manifest)
            .map_err(|e| (WAI_ERR_CONTAINER, format!("manifest serialize: {e}")))?;
        // build rendition table JSON + concatenated payload block
        let mut table: Vec<serde_json::Value> = Vec::with_capacity(m.renditions.len());
        let mut block: Vec<u8> = Vec::new();
        for (meta, payload) in m.manifest.renditions.iter().zip(m.renditions.iter()) {
            table.push(serde_json::json!({
                "capability": meta.capability,
                "kind":       meta.kind,
                "offset":     block.len(),
                "length":     payload.len(),
            }));
            block.extend_from_slice(payload);
        }
        let tj = serde_json::to_vec(&serde_json::Value::Array(table))
            .map_err(|e| (WAI_ERR_CONTAINER, format!("table serialize: {e}")))?;
        Ok((mb, tj, block))
    }) {
        Ok(Ok((mb, tj, block))) => unsafe {
            *out_manifest = WaiBuffer::from_vec(mb);
            *out_renditions_json = WaiBuffer::from_vec(tj);
            *out_payload_block = WaiBuffer::from_vec(block);
            WAI_OK
        }
        Ok(Err((code, msg))) => {
            set_err(msg);
            unsafe {
                clear_buf(out_manifest);
                clear_buf(out_renditions_json);
                clear_buf(out_payload_block);
            }
            code
        }
        Err(_) => {
            set_err("internal panic in envelope_unpack_multi");
            unsafe {
                clear_buf(out_manifest);
                clear_buf(out_renditions_json);
                clear_buf(out_payload_block);
            }
            WAI_ERR_PANIC
        }
    }
}

/// Parses a WAI envelope. Writes the manifest as JSON to `out_manifest`
/// and the raw payload to `out_payload`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn wai_envelope_unpack(
    bytes: *const u8, len: usize,
    out_manifest: *mut WaiBuffer, out_payload: *mut WaiBuffer,
) -> c_int {
    if out_manifest.is_null() || out_payload.is_null() {
        set_err("envelope_unpack: out_manifest/out_payload null");
        return WAI_ERR_INVALID_ARG;
    }
    match panic::catch_unwind(|| -> std::result::Result<(Vec<u8>, Vec<u8>), (c_int, String)> {
        let s = unsafe { input_slice(bytes, len) }
            .ok_or((WAI_ERR_INVALID_ARG, "bytes null + len>0".into()))?;
        let wai = crate::container::Wai::from_bytes(s)
            .map_err(|e| (WAI_ERR_CONTAINER, format!("envelope parse: {e}")))?;
        let mj = serde_json::to_vec(&wai.manifest)
            .map_err(|e| (WAI_ERR_CONTAINER, format!("manifest serialize: {e}")))?;
        Ok((mj, wai.payload))
    }) {
        Ok(Ok((mj, pl))) => unsafe {
            *out_manifest = WaiBuffer::from_vec(mj);
            *out_payload = WaiBuffer::from_vec(pl);
            WAI_OK
        }
        Ok(Err((code, msg))) => {
            set_err(msg);
            unsafe { clear_buf(out_manifest); clear_buf(out_payload); }
            code
        }
        Err(_) => {
            set_err("internal panic in envelope_unpack");
            unsafe { clear_buf(out_manifest); clear_buf(out_payload); }
            WAI_ERR_PANIC
        }
    }
}
