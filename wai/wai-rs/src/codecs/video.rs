//! Video codec wrapper: AV1 (encode via rav1e, decode via dav1d).
//! Both lossy and lossless via rav1e's `still_picture`/`bitrate`/`quantizer`
//! knobs.
//!
//! Wire format: a minimal frame-sequence container (NOT raw IVF — we
//! emit length-prefixed AV1 OBUs so the .wai payload is self-contained
//! and language-agnostic). The bitstream within each frame is standard
//! AV1, so any AV1 decoder reads it.

use byteorder::{LittleEndian, ReadBytesExt, WriteBytesExt};
use rav1e::config::SpeedSettings;
use rav1e::{Config, EncoderConfig, EncoderStatus};
use std::io::{Cursor, Read};

#[derive(Debug)]
pub struct VideoError(pub String);

impl<E: std::fmt::Display> From<E> for VideoError {
    fn from(e: E) -> Self { VideoError(e.to_string()) }
}

pub type Result<T> = std::result::Result<T, VideoError>;

/// Encode a sequence of RGB frames as AV1.
///
/// `frames`: each is row-major H×W×3 u8 RGB; all the same dimensions.
/// `lossless`: bypass rate-distortion and target lossless YUV
///   reconstruction. NOTE: WAI's RGB→YUV 4:2:0 conversion is itself
///   lossy (chroma subsampling + BT.601 rounding), so the RGB
///   round-trip is *near-lossless* (PSNR ≫ 40 dB) even with
///   `lossless=true`. For mathematically-lossless RGB use `wai.image.png`
///   per frame.
/// `quality`: 1..=100 (lossy mode only); 100 = highest, 1 = smallest.
///   Internally maps to rav1e's 0..=255 quantizer (lower = better
///   quality there).
pub fn av1_encode(frames: &[Vec<u8>], h: u32, w: u32, fps_num: u32,
                  fps_den: u32, lossless: bool, quality: u8) -> Result<Vec<u8>> {
    let mut enc_cfg = EncoderConfig::with_speed_preset(6);
    enc_cfg.width = w as usize;
    enc_cfg.height = h as usize;
    enc_cfg.time_base = rav1e::data::Rational { num: fps_den as u64,
                                                den: fps_num as u64 };
    enc_cfg.chroma_sampling = rav1e::color::ChromaSampling::Cs420;
    enc_cfg.bit_depth = 8;
    enc_cfg.still_picture = frames.len() == 1;
    enc_cfg.speed_settings = SpeedSettings::from_preset(6);
    if lossless {
        enc_cfg.quantizer = 0;
        enc_cfg.min_quantizer = 0;
    } else {
        // Map user quality 1..=100 → rav1e quantizer 255..=0 (inverted).
        let q = quality.clamp(1, 100) as u32;
        let qz = 255u32.saturating_sub((q * 255) / 100);
        enc_cfg.quantizer = qz as usize;
        enc_cfg.min_quantizer = qz.saturating_sub(8).min(255) as u8;
    }
    let cfg = Config::new().with_encoder_config(enc_cfg);
    let mut ctx: rav1e::Context<u8> = cfg.new_context()
        .map_err(|e| VideoError(format!("rav1e ctx: {e:?}")))?;

    // Outer container: <u32 n_frames><u32 h><u32 w><u32 fps_num><u32 fps_den>
    // followed by a sequence of <u32 pkt_len><pkt bytes>. The packet bytes
    // are AV1 OBUs (the standard bitstream).
    let mut out = Vec::new();
    out.write_u32::<LittleEndian>(frames.len() as u32)?;
    out.write_u32::<LittleEndian>(h)?;
    out.write_u32::<LittleEndian>(w)?;
    out.write_u32::<LittleEndian>(fps_num)?;
    out.write_u32::<LittleEndian>(fps_den)?;

    for rgb in frames {
        let mut frame = ctx.new_frame();
        rgb_to_yuv420_into(rgb, h as usize, w as usize, &mut frame);
        ctx.send_frame(frame)
           .map_err(|e| VideoError(format!("send_frame: {e:?}")))?;
        drain_packets(&mut ctx, &mut out)?;
    }
    ctx.flush();
    drain_packets(&mut ctx, &mut out)?;
    Ok(out)
}

fn drain_packets(ctx: &mut rav1e::Context<u8>, out: &mut Vec<u8>) -> Result<()> {
    loop {
        match ctx.receive_packet() {
            Ok(pkt) => {
                out.write_u32::<LittleEndian>(pkt.data.len() as u32)?;
                out.extend_from_slice(&pkt.data);
            }
            Err(EncoderStatus::Encoded) => continue,
            Err(EncoderStatus::NeedMoreData) => return Ok(()),
            Err(EncoderStatus::LimitReached) => return Ok(()),
            Err(EncoderStatus::EnoughData) => return Ok(()),
            Err(e) => return Err(VideoError(format!("recv: {e:?}"))),
        }
    }
}

fn rgb_to_yuv420_into(rgb: &[u8], h: usize, w: usize, frame: &mut rav1e::Frame<u8>) {
    // BT.601 conversion to Y'CbCr full-range. rav1e indexes planes as a
    // fixed-size array so we can't take three independent mutable borrows
    // in one expression — convert each plane in its own scope.
    {
        let yp = &mut frame.planes[0];
        let y_stride = yp.cfg.stride;
        let y_data = yp.data_origin_mut();
        for i in 0..h {
            for j in 0..w {
                let p = (i * w + j) * 3;
                let r = rgb[p] as f32;
                let g = rgb[p + 1] as f32;
                let b = rgb[p + 2] as f32;
                let y = 0.299 * r + 0.587 * g + 0.114 * b;
                y_data[i * y_stride + j] = y.clamp(0.0, 255.0).round() as u8;
            }
        }
    }
    // chroma: 4:2:0 nearest-sample downsample (production-correct would
    // 2×2 average; deferred — AV1 picks its own internal chroma quality).
    let h2 = h / 2;
    let w2 = w / 2;
    {
        let up = &mut frame.planes[1];
        let u_stride = up.cfg.stride;
        let u_data = up.data_origin_mut();
        for i in 0..h2 {
            for j in 0..w2 {
                let p = (2 * i * w + 2 * j) * 3;
                let r = rgb[p] as f32;
                let g = rgb[p + 1] as f32;
                let b = rgb[p + 2] as f32;
                let cb = -0.168736 * r - 0.331264 * g + 0.5 * b + 128.0;
                u_data[i * u_stride + j] = cb.clamp(0.0, 255.0).round() as u8;
            }
        }
    }
    {
        let vp = &mut frame.planes[2];
        let v_stride = vp.cfg.stride;
        let v_data = vp.data_origin_mut();
        for i in 0..h2 {
            for j in 0..w2 {
                let p = (2 * i * w + 2 * j) * 3;
                let r = rgb[p] as f32;
                let g = rgb[p + 1] as f32;
                let b = rgb[p + 2] as f32;
                let cr = 0.5 * r - 0.418688 * g - 0.081312 * b + 128.0;
                v_data[i * v_stride + j] = cr.clamp(0.0, 255.0).round() as u8;
            }
        }
    }
}

/// Decode a WAI AV1 payload back to a sequence of RGB frames.
///
/// Defensive: every value read from the header is sanity-checked before
/// being used so malformed input can't trigger dav1d's C-level abort()
/// (which would bypass Rust's catch_unwind). All errors return a clean
/// `VideoError`; we never feed a zero-length or out-of-bounds packet to
/// dav1d_send_data.
pub fn av1_decode(bytes: &[u8]) -> Result<(Vec<Vec<u8>>, u32, u32, u32, u32)> {
    // The fixed header is 20 bytes (5×u32).
    if bytes.len() < 20 {
        return Err(VideoError(format!(
            "payload too short for AV1 header: {} < 20", bytes.len())));
    }
    let mut rd = Cursor::new(bytes);
    let nframes = rd.read_u32::<LittleEndian>()? as usize;
    let h = rd.read_u32::<LittleEndian>()?;
    let w = rd.read_u32::<LittleEndian>()?;
    let fps_num = rd.read_u32::<LittleEndian>()?;
    let fps_den = rd.read_u32::<LittleEndian>()?;

    // sanity bounds — reject malformed/adversarial headers.
    if nframes == 0 || nframes > 1_000_000 {
        return Err(VideoError(format!("nframes out of range: {nframes}")));
    }
    if h == 0 || w == 0 || h > 32_768 || w > 32_768 {
        return Err(VideoError(format!("dimensions out of range: {w}×{h}")));
    }
    if fps_num == 0 || fps_den == 0 {
        return Err(VideoError("fps_num/fps_den must be non-zero".into()));
    }

    let mut dec = dav1d::Decoder::new()
        .map_err(|e| VideoError(format!("dav1d new: {e:?}")))?;
    let mut frames = Vec::with_capacity(nframes);

    loop {
        if rd.position() as usize >= bytes.len() {
            break;
        }
        // need 4 bytes for the packet length
        if (bytes.len() as u64).saturating_sub(rd.position()) < 4 {
            return Err(VideoError("truncated packet length field".into()));
        }
        let sz = rd.read_u32::<LittleEndian>()? as usize;
        // dav1d aborts on sz == 0 or sz > SIZE_MAX/2 — catch here.
        if sz == 0 || sz > 1 << 30 {
            return Err(VideoError(format!("invalid packet size {sz}")));
        }
        // must have `sz` bytes remaining
        let remaining = bytes.len().saturating_sub(rd.position() as usize);
        if sz > remaining {
            return Err(VideoError(format!(
                "packet size {sz} exceeds {remaining} remaining bytes")));
        }
        let mut pkt = vec![0u8; sz];
        rd.read_exact(&mut pkt)?;
        dec.send_data(pkt, None, None, None)
           .map_err(|e| VideoError(format!("dav1d send: {e:?}")))?;
        while let Ok(pic) = dec.get_picture() {
            frames.push(picture_to_rgb(&pic));
        }
    }
    dec.send_pending_data().ok();
    while let Ok(pic) = dec.get_picture() {
        frames.push(picture_to_rgb(&pic));
    }
    frames.truncate(nframes);
    Ok((frames, h, w, fps_num, fps_den))
}

fn picture_to_rgb(pic: &dav1d::Picture) -> Vec<u8> {
    let (w, h) = (pic.width() as usize, pic.height() as usize);
    let y_plane = pic.plane(dav1d::PlanarImageComponent::Y);
    let u_plane = pic.plane(dav1d::PlanarImageComponent::U);
    let v_plane = pic.plane(dav1d::PlanarImageComponent::V);
    let y_stride = pic.stride(dav1d::PlanarImageComponent::Y) as usize;
    let u_stride = pic.stride(dav1d::PlanarImageComponent::U) as usize;
    let v_stride = pic.stride(dav1d::PlanarImageComponent::V) as usize;
    let mut out = vec![0u8; w * h * 3];
    for i in 0..h {
        for j in 0..w {
            let y = y_plane.as_ref()[i * y_stride + j] as f32;
            let ci = i / 2; let cj = j / 2;
            let cb = u_plane.as_ref()[ci * u_stride + cj] as f32 - 128.0;
            let cr = v_plane.as_ref()[ci * v_stride + cj] as f32 - 128.0;
            let r = y + 1.402 * cr;
            let g = y - 0.344136 * cb - 0.714136 * cr;
            let b = y + 1.772 * cb;
            let p = (i * w + j) * 3;
            out[p]     = r.clamp(0.0, 255.0).round() as u8;
            out[p + 1] = g.clamp(0.0, 255.0).round() as u8;
            out[p + 2] = b.clamp(0.0, 255.0).round() as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth_clip(n: usize, h: u32, w: u32) -> Vec<Vec<u8>> {
        (0..n).map(|f| {
            let mut img = vec![0u8; (h * w * 3) as usize];
            for i in 0..h {
                for j in 0..w {
                    let p = ((i * w + j) * 3) as usize;
                    img[p]     = (((j + f as u32) * 4) % 256) as u8;
                    img[p + 1] = ((i * 4) % 256) as u8;
                    img[p + 2] = (((i + j + f as u32) * 3) % 256) as u8;
                }
            }
            img
        }).collect()
    }

    fn rgb_psnr(a: &[u8], b: &[u8]) -> f64 {
        let n = a.len();
        let mse: f64 = a.iter().zip(b.iter())
            .map(|(&x, &y)| (x as f64 - y as f64).powi(2))
            .sum::<f64>() / n as f64;
        if mse <= 1e-12 { 99.0 } else { 10.0 * (255.0_f64.powi(2) / mse).log10() }
    }

    #[test]
    fn av1_round_trip_lossy() {
        let (h, w) = (64u32, 64u32);
        let clip = synth_clip(4, h, w);
        let bytes = av1_encode(&clip, h, w, 30, 1, false, 80).unwrap();
        let (rec, rh, rw, _, _) = av1_decode(&bytes).unwrap();
        assert_eq!((rh, rw), (h, w));
        assert_eq!(rec.len(), clip.len());
    }

    #[test]
    fn av1_quality_knob_monotonic() {
        // Higher `quality` should produce larger files at the same content.
        let (h, w) = (64u32, 64u32);
        let clip = synth_clip(3, h, w);
        let low = av1_encode(&clip, h, w, 30, 1, false, 20).unwrap();
        let high = av1_encode(&clip, h, w, 30, 1, false, 90).unwrap();
        assert!(high.len() > low.len(),
                "quality knob direction: q=90 ({} B) should be larger than q=20 ({} B)",
                high.len(), low.len());
    }

    /// Smooth low-frequency clip — 4:2:0 chroma subsample doesn't destroy
    /// content that varies slowly across 2×2 neighborhoods, so the
    /// "lossless" AV1 path can be measured fairly here.
    fn smooth_clip(n: usize, h: u32, w: u32) -> Vec<Vec<u8>> {
        (0..n).map(|f| {
            let mut img = vec![0u8; (h * w * 3) as usize];
            for i in 0..h {
                for j in 0..w {
                    let p = ((i * w + j) * 3) as usize;
                    // 8-pixel-period gradients — well within Nyquist for 4:2:0
                    img[p]     = (((i / 8) + (j / 8) + f as u32) as u8).wrapping_mul(8);
                    img[p + 1] = ((i / 8) as u8).wrapping_mul(8);
                    img[p + 2] = ((j / 8) as u8).wrapping_mul(8);
                }
            }
            img
        }).collect()
    }

    #[test]
    fn av1_lossless_rgb_near_lossless() {
        // AV1 with quantizer=0 reconstructs YUV bit-exactly. Our RGB→YUV
        // 4:2:0 → AV1 → YUV → RGB path is therefore *near-lossless* on
        // content that doesn't violate Nyquist on the chroma plane — the
        // only error is BT.601 conversion rounding + the 4:2:0 subsample.
        let (h, w) = (64u32, 64u32);
        let clip = smooth_clip(2, h, w);
        let bytes = av1_encode(&clip, h, w, 30, 1, true, 0).unwrap();
        let (rec, _, _, _, _) = av1_decode(&bytes).unwrap();
        assert_eq!(rec.len(), clip.len());
        for (orig, dec) in clip.iter().zip(rec.iter()) {
            let p = rgb_psnr(orig, dec);
            assert!(p > 40.0,
                    "lossless AV1 on chroma-Nyquist-safe content: PSNR \
                     {p:.1} dB (expected ≫ 40 dB; lower means AV1 isn't \
                     actually running lossless)");
        }
    }
}
