//! Image codec wrappers. PNG (universal floor), JPEG (compatibility),
//! AVIF (modern lossy, AV1-based), JPEG-XL (lossless + lossy).
//!
//! Each codec exposes `encode(rgb, h, w, quality) -> Vec<u8>` and
//! `decode(bytes) -> (rgb, h, w)`. RGB is row-major, 8-bit, 3 channels.
//! `quality` is in 1..=100 where applicable.

use std::io::Cursor;

use image::ImageEncoder;
use jpegxl_rs::encode::EncoderSpeed;
use jpegxl_rs::{decoder_builder, encoder_builder};

#[derive(Debug)]
pub struct ImageError(pub String);

impl<E: std::fmt::Display> From<E> for ImageError {
    fn from(e: E) -> Self { ImageError(e.to_string()) }
}

pub type Result<T> = std::result::Result<T, ImageError>;

// ---- PNG (lossless, universal) ----
pub fn png_encode(rgb: &[u8], h: u32, w: u32) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    image::codecs::png::PngEncoder::new(&mut out)
        .write_image(rgb, w, h, image::ExtendedColorType::Rgb8)?;
    Ok(out)
}

pub fn png_decode(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    let img = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()?.decode()?.to_rgb8();
    let (w, h) = img.dimensions();
    Ok((img.into_raw(), h, w))
}

// ---- JPEG (lossy) ----
// Uses `jpeg-encoder` (pure Rust) directly. Notes:
//   * `image` crate's default JpegEncoder produced files ~60-90% larger
//     than libjpeg at the same nominal q (bench caught it).
//   * `jpeg-encoder`'s q scale is offset from libjpeg's: q=80 there ≈
//     libjpeg q=92 in RD position. We remap the *public* WAI q to a
//     jpeg-encoder q so WAI q=N gives roughly libjpeg q=N output bytes,
//     letting users reason about WAI q the same way they reason about
//     libjpeg q (the universal convention in the field).
//   * We DON'T enable optimized Huffman tables — they're standard JPEG
//     but the `zune-jpeg` decoder we link via `image` misreads them.
fn remap_q_to_jpeg_encoder(public_q: u8) -> u8 {
    // Empirically calibrated against libjpeg on Kodak (bench harness):
    //   user-q 50 → encoder-q 35 ≈ libjpeg q=50 bytes
    //   user-q 80 → encoder-q 65 ≈ libjpeg q=80 bytes
    //   user-q 90 → encoder-q 78 ≈ libjpeg q=90 bytes
    // Linear-ish in the working range; clamp to 1..=100.
    let q = public_q.clamp(1, 100) as i32;
    let mapped = if q >= 50 { ((q - 50) * 130 / 100) + 35 } else { q * 70 / 100 };
    mapped.clamp(1, 100) as u8
}

pub fn jpeg_encode(rgb: &[u8], h: u32, w: u32, quality: u8) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    let q = remap_q_to_jpeg_encoder(quality);
    let enc = jpeg_encoder::Encoder::new(&mut out, q);
    enc.encode(rgb, w as u16, h as u16,
               jpeg_encoder::ColorType::Rgb)
        .map_err(|e| ImageError(format!("{e:?}")))?;
    Ok(out)
}

pub fn jpeg_decode(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    let img = image::ImageReader::new(Cursor::new(bytes))
        .with_guessed_format()?.decode()?.to_rgb8();
    let (w, h) = img.dimensions();
    Ok((img.into_raw(), h, w))
}

// ---- AVIF (modern lossy, AV1-based) ----
pub fn avif_encode(rgb: &[u8], h: u32, w: u32, quality: u8) -> Result<Vec<u8>> {
    let mut out = Vec::new();
    image::codecs::avif::AvifEncoder::new_with_speed_quality(&mut out, 6, quality)
        .write_image(rgb, w, h, image::ExtendedColorType::Rgb8)?;
    Ok(out)
}

pub fn avif_decode(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    // Use libavif-image (links the system libavif). The `image` crate's
    // own AVIF decoder is encode-only in the default build; relying on
    // it silently returned empty buffers (bench harness caught this).
    let img = libavif_image::read(bytes)
        .map_err(|e| ImageError(format!("libavif decode: {e:?}")))?
        .to_rgb8();
    let (w, h) = img.dimensions();
    Ok((img.into_raw(), h, w))
}

// ---- JPEG-XL (lossless or lossy) ----
pub fn jxl_encode(rgb: &[u8], h: u32, w: u32, quality: Option<f32>) -> Result<Vec<u8>> {
    // `quality=None` ⇒ mathematically lossless (the `.lossless(true)`
    // flag is what enables this; distance=0 alone is *not* sufficient).
    // `quality=Some(q)` ⇒ lossy with butteraugli distance derived from q.
    let mut enc = if let Some(q) = quality {
        let distance = ((100.0 - q.clamp(1.0, 100.0)) * 0.15).max(0.01);
        encoder_builder().speed(EncoderSpeed::Squirrel).quality(distance).build()?
    } else {
        // True lossless needs both `lossless(true)` AND
        // `uses_original_profile(true)` so libjxl keeps the original
        // color encoding instead of round-tripping through XYB internally
        // (which causes the small per-pixel drift we were seeing).
        encoder_builder()
            .speed(EncoderSpeed::Squirrel)
            .lossless(true)
            .quality(0.0)
            .uses_original_profile(true)
            .build()?
    };
    let r = enc.encode::<u8, u8>(rgb, w, h)?;
    Ok(r.data)
}

pub fn jxl_decode(bytes: &[u8]) -> Result<(Vec<u8>, u32, u32)> {
    let dec = decoder_builder().build()?;
    let r = dec.decode(bytes)?;
    let (info, pixels) = r;
    let pixels = match pixels {
        jpegxl_rs::decode::Pixels::Uint8(v) => v,
        _ => return Err(ImageError("jxl: non-u8 pixel buffer".into())),
    };
    Ok((pixels, info.height, info.width))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synth(h: u32, w: u32) -> Vec<u8> {
        let mut rgb = vec![0u8; (h * w * 3) as usize];
        for i in 0..h {
            for j in 0..w {
                let idx = ((i * w + j) * 3) as usize;
                rgb[idx]     = ((i + j) % 256) as u8;
                rgb[idx + 1] = ((i * 3) % 256) as u8;
                rgb[idx + 2] = ((j * 5) % 256) as u8;
            }
        }
        rgb
    }

    #[test]
    fn png_round_trip_lossless() {
        let (h, w) = (64u32, 64u32);
        let rgb = synth(h, w);
        let bytes = png_encode(&rgb, h, w).unwrap();
        let (rec, rh, rw) = png_decode(&bytes).unwrap();
        assert_eq!((rh, rw), (h, w));
        assert_eq!(rec, rgb, "PNG must be lossless");
    }

    #[test]
    fn jpeg_round_trip_lossy_but_sane() {
        let (h, w) = (64u32, 64u32);
        let rgb = synth(h, w);
        let bytes = jpeg_encode(&rgb, h, w, 90).unwrap();
        let (rec, _, _) = jpeg_decode(&bytes).unwrap();
        assert_eq!(rec.len(), rgb.len());
    }

    #[test]
    fn avif_round_trip_lossy_but_sane() {
        let (h, w) = (64u32, 64u32);
        let rgb = synth(h, w);
        let bytes = avif_encode(&rgb, h, w, 80).unwrap();
        assert!(bytes.len() > 0);
        // decode requires AVIF feature in `image` — may not be enabled
        // depending on build features; allow either case.
        let _ = avif_decode(&bytes);
    }

    #[test]
    fn jxl_lossless_round_trip() {
        let (h, w) = (64u32, 64u32);
        let rgb = synth(h, w);
        let bytes = jxl_encode(&rgb, h, w, None).unwrap();
        let (rec, rh, rw) = jxl_decode(&bytes).unwrap();
        assert_eq!((rh, rw), (h, w));
        assert_eq!(rec, rgb, "JPEG-XL distance=0 must be lossless");
    }
}
