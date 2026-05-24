//! Image preprocessing for vision embedders.
//!
//! Each model expects a specific input geometry and normalisation:
//!
//! | model         | resolution | mean (RGB)              | std (RGB)               |
//! |---------------|------------|-------------------------|-------------------------|
//! | CLIP ViT-B/32 | 224×224    | 0.4815, 0.4578, 0.4082  | 0.2686, 0.2613, 0.2758  |
//! | SigLIP        | 384×384    | 0.5, 0.5, 0.5           | 0.5, 0.5, 0.5           |
//! | LLaVA         | 336×336    | 0.4815, 0.4578, 0.4082  | 0.2686, 0.2613, 0.2758  |
//!
//! The preprocess pipeline is: decode → convert to RGB8 → resize (preserving
//! aspect ratio) → center crop → normalise to f32. Output is a flat
//! `Vec<f32>` in `CHW` layout.

use image::imageops::FilterType;
use image::{DynamicImage, GenericImageView};

use crate::error::{MultimodalError, MultimodalResult};

/// CLIP ViT-B/32 input geometry.
pub const CLIP_SIZE: u32 = 224;
/// CLIP normalisation mean (RGB).
pub const CLIP_MEAN: [f32; 3] = [0.4815, 0.4578, 0.4082];
/// CLIP normalisation std (RGB).
pub const CLIP_STD: [f32; 3] = [0.2686, 0.2613, 0.2758];

/// SigLIP input geometry.
pub const SIGLIP_SIZE: u32 = 384;
/// SigLIP normalisation mean.
pub const SIGLIP_MEAN: [f32; 3] = [0.5, 0.5, 0.5];
/// SigLIP normalisation std.
pub const SIGLIP_STD: [f32; 3] = [0.5, 0.5, 0.5];

/// LLaVA input geometry.
pub const LLAVA_SIZE: u32 = 336;

/// Decode image bytes into an `image::DynamicImage`.
pub fn decode(bytes: &[u8]) -> MultimodalResult<DynamicImage> {
    let img = image::load_from_memory(bytes)?;
    Ok(img)
}

/// Resize so the *short* side equals `size`, then center-crop to
/// `size × size`. This is the canonical CLIP / SigLIP pipeline.
pub fn resize_and_center_crop(img: &DynamicImage, size: u32) -> DynamicImage {
    let (w, h) = img.dimensions();
    if w == 0 || h == 0 {
        return img.clone();
    }
    let scale = size as f32 / (w.min(h) as f32);
    let new_w = ((w as f32) * scale).round() as u32;
    let new_h = ((h as f32) * scale).round() as u32;
    let resized = img.resize_exact(new_w.max(size), new_h.max(size), FilterType::Triangle);
    let x = (resized.width().saturating_sub(size)) / 2;
    let y = (resized.height().saturating_sub(size)) / 2;
    resized.crop_imm(x, y, size, size)
}

/// Convert to RGB f32 CHW layout, normalised by `(pixel/255 - mean) / std`.
pub fn to_chw_normalized(img: &DynamicImage, mean: [f32; 3], std: [f32; 3]) -> Vec<f32> {
    let rgb = img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let mut out = vec![0.0f32; 3 * w * h];
    for y in 0..h {
        for x in 0..w {
            let p = rgb.get_pixel(x as u32, y as u32);
            for c in 0..3 {
                let v = (p[c] as f32) / 255.0;
                let n = (v - mean[c]) / std[c];
                out[c * w * h + y * w + x] = n;
            }
        }
    }
    out
}

/// Full CLIP preprocess: decode → resize+crop to 224×224 → normalise.
pub fn preprocess_clip(bytes: &[u8]) -> MultimodalResult<Vec<f32>> {
    let img = decode(bytes)?;
    let cropped = resize_and_center_crop(&img, CLIP_SIZE);
    Ok(to_chw_normalized(&cropped, CLIP_MEAN, CLIP_STD))
}

/// Full SigLIP preprocess: decode → resize+crop to 384×384 → normalise.
pub fn preprocess_siglip(bytes: &[u8]) -> MultimodalResult<Vec<f32>> {
    let img = decode(bytes)?;
    let cropped = resize_and_center_crop(&img, SIGLIP_SIZE);
    Ok(to_chw_normalized(&cropped, SIGLIP_MEAN, SIGLIP_STD))
}

/// Full LLaVA preprocess: decode → resize+crop to 336×336 → normalise (CLIP
/// mean/std).
pub fn preprocess_llava(bytes: &[u8]) -> MultimodalResult<Vec<f32>> {
    let img = decode(bytes)?;
    let cropped = resize_and_center_crop(&img, LLAVA_SIZE);
    Ok(to_chw_normalized(&cropped, CLIP_MEAN, CLIP_STD))
}

/// Convenience: convert any input image (decoded) to RGB-uint8 bytes.
pub fn to_rgb_u8(img: &DynamicImage) -> Vec<u8> {
    img.to_rgb8().into_raw()
}

/// Build a synthetic solid-colour PNG of the given size, useful for tests.
pub fn solid_png(width: u32, height: u32, rgb: [u8; 3]) -> MultimodalResult<Vec<u8>> {
    let mut buf = image::RgbImage::new(width, height);
    for p in buf.pixels_mut() {
        *p = image::Rgb(rgb);
    }
    let mut out = Vec::<u8>::new();
    image::DynamicImage::ImageRgb8(buf)
        .write_to(&mut std::io::Cursor::new(&mut out), image::ImageFormat::Png)?;
    Ok(out)
}

/// Pre-flight check that `bytes` decodes as an image (does not allocate the
/// full pixel buffer).
pub fn validate(bytes: &[u8]) -> MultimodalResult<()> {
    image::guess_format(bytes).map_err(|e| MultimodalError::Decode(e.to_string()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn preprocess_clip_known_dimensions() {
        let png = solid_png(64, 32, [128, 128, 128]).expect("png");
        let v = preprocess_clip(&png).expect("ok");
        // CHW: 3 * 224 * 224.
        assert_eq!(v.len(), 3 * CLIP_SIZE as usize * CLIP_SIZE as usize);
    }

    #[test]
    fn preprocess_siglip_known_dimensions() {
        let png = solid_png(64, 32, [255, 0, 0]).expect("png");
        let v = preprocess_siglip(&png).expect("ok");
        assert_eq!(v.len(), 3 * SIGLIP_SIZE as usize * SIGLIP_SIZE as usize);
    }

    #[test]
    fn normalisation_means_match_expectation() {
        // Solid gray (128/255 ≈ 0.502) → normalised mean ≈ (0.502 - mean)/std.
        let png = solid_png(32, 32, [128, 128, 128]).expect("png");
        let v = preprocess_clip(&png).expect("ok");
        let chan_size = (CLIP_SIZE * CLIP_SIZE) as usize;
        let r_mean: f32 = v[..chan_size].iter().copied().sum::<f32>() / (chan_size as f32);
        let expected_r = (128.0_f32 / 255.0 - CLIP_MEAN[0]) / CLIP_STD[0];
        assert!(
            (r_mean - expected_r).abs() < 1e-2,
            "r_mean {r_mean} expected {expected_r}"
        );
    }

    #[test]
    fn validate_accepts_png_rejects_garbage() {
        let png = solid_png(8, 8, [0, 0, 0]).expect("png");
        validate(&png).expect("png is valid");
        let bad = b"not an image";
        assert!(validate(bad).is_err());
    }
}
