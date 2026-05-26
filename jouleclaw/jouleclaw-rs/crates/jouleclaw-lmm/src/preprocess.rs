//! Image preprocessing: resize, normalize, patchify.
//!
//! Operates on `RgbImage` (an already-decoded f32 RGB tensor in
//! `[height, width, 3]` layout, channel-last, values in [0, 1]). Byte
//! decoding (PNG/JPEG → RGB) is R31.1 — it requires either an external
//! crate or a hand-rolled decoder, both larger than this revision's
//! scope. Once decoded, the steps here are the same ones used by every
//! vision-language model: resize to the model's expected resolution,
//! normalize against ImageNet (or model-specific) mean/std, then split
//! into ViT-style patches.

use std::fmt;

#[derive(Debug, Clone, PartialEq)]
pub enum ImageError {
    BadDimensions { what: &'static str, value: usize },
    SizeMismatch { expected: usize, got: usize },
    NonDivisible { dim: usize, patch_size: usize },
    Decode(String),
}

/// Decode encoded image bytes (JPEG/PNG/…) into an [`RgbImage`] with
/// pixel values scaled to `[0, 1]` f32, channel-last. This is the
/// raw-byte → RGB step the LFM2.5-VL path needs before resize +
/// normalize + patchify.
pub fn decode_image_bytes(bytes: &[u8]) -> Result<RgbImage, ImageError> {
    let dyn_img = image::load_from_memory(bytes)
        .map_err(|e| ImageError::Decode(format!("{e}")))?;
    let rgb = dyn_img.to_rgb8();
    let (w, h) = (rgb.width() as usize, rgb.height() as usize);
    let mut px = Vec::with_capacity(w * h * 3);
    for p in rgb.pixels() {
        px.push(p[0] as f32 / 255.0);
        px.push(p[1] as f32 / 255.0);
        px.push(p[2] as f32 / 255.0);
    }
    RgbImage::new(w, h, px)
}

impl fmt::Display for ImageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadDimensions { what, value } => {
                write!(f, "bad image dimension {} = {}", what, value)
            }
            Self::SizeMismatch { expected, got } => {
                write!(f, "buffer size mismatch: expected {}, got {}", expected, got)
            }
            Self::NonDivisible { dim, patch_size } => {
                write!(f, "image dim {} not divisible by patch size {}", dim, patch_size)
            }
            Self::Decode(s) => write!(f, "image decode failed: {}", s),
        }
    }
}

impl std::error::Error for ImageError {}

/// Channel-last RGB image: `pixels[(y * width + x) * 3 + c]`.
#[derive(Debug, Clone)]
pub struct RgbImage {
    pub width: usize,
    pub height: usize,
    pub pixels: Vec<f32>,
}

impl RgbImage {
    pub fn new(width: usize, height: usize, pixels: Vec<f32>) -> Result<Self, ImageError> {
        if width == 0 {
            return Err(ImageError::BadDimensions { what: "width", value: 0 });
        }
        if height == 0 {
            return Err(ImageError::BadDimensions { what: "height", value: 0 });
        }
        let expected = width * height * 3;
        if pixels.len() != expected {
            return Err(ImageError::SizeMismatch {
                expected,
                got: pixels.len(),
            });
        }
        Ok(Self { width, height, pixels })
    }

    #[inline]
    pub fn at(&self, x: usize, y: usize, c: usize) -> f32 {
        self.pixels[(y * self.width + x) * 3 + c]
    }
}

/// Settings for one preprocess pass.
#[derive(Debug, Clone)]
pub struct PreprocessConfig {
    pub target_width: usize,
    pub target_height: usize,
    /// Per-channel mean (RGB). ImageNet defaults: [0.485, 0.456, 0.406].
    pub mean: [f32; 3],
    /// Per-channel std (RGB). ImageNet defaults: [0.229, 0.224, 0.225].
    pub std: [f32; 3],
    /// Square patch side, in pixels of the resized image. ViT uses 14 or 16.
    pub patch_size: usize,
}

impl PreprocessConfig {
    /// ViT-B/16 defaults: 224×224, ImageNet normalization, patch size 16.
    pub fn vit_b16() -> Self {
        Self {
            target_width: 224,
            target_height: 224,
            mean: [0.485, 0.456, 0.406],
            std: [0.229, 0.224, 0.225],
            patch_size: 16,
        }
    }
}

/// Bilinear resize to `(target_w, target_h)`. Channel-last, f32 in/out.
/// Determinism: pure f32 arithmetic with stable iteration order — same
/// input gives bit-identical output across calls.
pub fn bilinear_resize(
    img: &RgbImage,
    target_w: usize,
    target_h: usize,
) -> Result<RgbImage, ImageError> {
    if target_w == 0 {
        return Err(ImageError::BadDimensions { what: "target_w", value: 0 });
    }
    if target_h == 0 {
        return Err(ImageError::BadDimensions { what: "target_h", value: 0 });
    }
    let mut out = vec![0.0_f32; target_w * target_h * 3];
    // Map output pixel center to source coordinate. Using corner-aligned
    // sampling (matches PIL's "PIL.Image.BILINEAR" with align_corners=False).
    let scale_x = (img.width as f32) / (target_w as f32);
    let scale_y = (img.height as f32) / (target_h as f32);
    for ty in 0..target_h {
        let sy = ((ty as f32) + 0.5) * scale_y - 0.5;
        let y0 = sy.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(img.height - 1);
        let fy = (sy - (y0 as f32)).clamp(0.0, 1.0);
        for tx in 0..target_w {
            let sx = ((tx as f32) + 0.5) * scale_x - 0.5;
            let x0 = sx.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(img.width - 1);
            let fx = (sx - (x0 as f32)).clamp(0.0, 1.0);
            for c in 0..3 {
                let v00 = img.at(x0, y0, c);
                let v10 = img.at(x1, y0, c);
                let v01 = img.at(x0, y1, c);
                let v11 = img.at(x1, y1, c);
                let top = v00 + fx * (v10 - v00);
                let bot = v01 + fx * (v11 - v01);
                let v = top + fy * (bot - top);
                out[(ty * target_w + tx) * 3 + c] = v;
            }
        }
    }
    RgbImage::new(target_w, target_h, out)
}

/// In-place per-channel mean/std normalization: `pixels[c] = (pixels[c] - mean[c]) / std[c]`.
pub fn normalize(img: &mut RgbImage, mean: [f32; 3], std: [f32; 3]) {
    for px in img.pixels.chunks_exact_mut(3) {
        for c in 0..3 {
            px[c] = (px[c] - mean[c]) / std[c];
        }
    }
}

/// Patchify into ViT-style sequence: each patch is a flat row of
/// `patch_size * patch_size * 3` values, in row-major (within-patch) order.
/// The output is `n_patches × patch_dim`. `n_patches = (H/P) * (W/P)`.
pub fn patchify(img: &RgbImage, patch_size: usize) -> Result<Vec<Vec<f32>>, ImageError> {
    if patch_size == 0 {
        return Err(ImageError::BadDimensions { what: "patch_size", value: 0 });
    }
    if img.width % patch_size != 0 {
        return Err(ImageError::NonDivisible {
            dim: img.width,
            patch_size,
        });
    }
    if img.height % patch_size != 0 {
        return Err(ImageError::NonDivisible {
            dim: img.height,
            patch_size,
        });
    }
    let n_y = img.height / patch_size;
    let n_x = img.width / patch_size;
    let patch_dim = patch_size * patch_size * 3;
    let mut patches = Vec::with_capacity(n_y * n_x);
    for py in 0..n_y {
        for px in 0..n_x {
            let mut buf = Vec::with_capacity(patch_dim);
            for iy in 0..patch_size {
                for ix in 0..patch_size {
                    for c in 0..3 {
                        buf.push(img.at(px * patch_size + ix, py * patch_size + iy, c));
                    }
                }
            }
            patches.push(buf);
        }
    }
    Ok(patches)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn solid(w: usize, h: usize, rgb: [f32; 3]) -> RgbImage {
        let mut px = Vec::with_capacity(w * h * 3);
        for _ in 0..(w * h) {
            px.extend_from_slice(&rgb);
        }
        RgbImage::new(w, h, px).unwrap()
    }

    #[test]
    fn rgb_image_validates_buffer_size() {
        assert!(matches!(
            RgbImage::new(4, 4, vec![0.0; 10]),
            Err(ImageError::SizeMismatch { .. })
        ));
        assert!(matches!(
            RgbImage::new(0, 4, vec![0.0; 0]),
            Err(ImageError::BadDimensions { .. })
        ));
    }

    #[test]
    fn bilinear_resize_preserves_solid_color() {
        // Resizing a constant image should yield a constant image of the
        // same color, regardless of scale factor.
        let img = solid(8, 8, [0.3, 0.6, 0.9]);
        let up = bilinear_resize(&img, 32, 32).unwrap();
        for px in up.pixels.chunks_exact(3) {
            assert!((px[0] - 0.3).abs() < 1e-6);
            assert!((px[1] - 0.6).abs() < 1e-6);
            assert!((px[2] - 0.9).abs() < 1e-6);
        }
        let down = bilinear_resize(&img, 2, 2).unwrap();
        for px in down.pixels.chunks_exact(3) {
            assert!((px[0] - 0.3).abs() < 1e-6);
        }
    }

    #[test]
    fn bilinear_resize_is_deterministic() {
        // Two different gradient pixels so interpolation actually runs.
        let mut px = Vec::new();
        for y in 0..4 {
            for x in 0..4 {
                px.push((x as f32) / 3.0);
                px.push((y as f32) / 3.0);
                px.push(0.5);
            }
        }
        let img = RgbImage::new(4, 4, px).unwrap();
        let a = bilinear_resize(&img, 7, 9).unwrap();
        let b = bilinear_resize(&img, 7, 9).unwrap();
        assert_eq!(a.pixels, b.pixels);
    }

    #[test]
    fn normalize_subtracts_mean_and_divides_by_std() {
        let mut img = solid(2, 2, [0.5, 0.5, 0.5]);
        normalize(&mut img, [0.5, 0.5, 0.5], [0.1, 0.2, 0.5]);
        for px in img.pixels.chunks_exact(3) {
            assert!(px[0].abs() < 1e-6);
            assert!(px[1].abs() < 1e-6);
            assert!(px[2].abs() < 1e-6);
        }
        // Non-zero offset.
        let mut img2 = solid(1, 1, [0.6, 0.3, 1.0]);
        normalize(&mut img2, [0.5, 0.5, 0.5], [0.1, 0.1, 0.1]);
        assert!((img2.pixels[0] - 1.0).abs() < 1e-6);
        assert!((img2.pixels[1] - (-2.0)).abs() < 1e-6);
        assert!((img2.pixels[2] - 5.0).abs() < 1e-6);
    }

    #[test]
    fn patchify_count_and_dim_match_expectations() {
        let img = solid(32, 16, [0.1, 0.2, 0.3]);
        let patches = patchify(&img, 8).unwrap();
        // 4 across × 2 down = 8 patches; each of size 8*8*3 = 192.
        assert_eq!(patches.len(), 8);
        for p in &patches {
            assert_eq!(p.len(), 192);
            // Solid color: every value in every patch is the same channel value.
            for chunk in p.chunks_exact(3) {
                assert!((chunk[0] - 0.1).abs() < 1e-6);
                assert!((chunk[1] - 0.2).abs() < 1e-6);
                assert!((chunk[2] - 0.3).abs() < 1e-6);
            }
        }
    }

    #[test]
    fn patchify_rejects_non_divisible_dims() {
        let img = solid(15, 16, [0.0; 3]);
        assert!(matches!(
            patchify(&img, 8),
            Err(ImageError::NonDivisible { .. })
        ));
    }

    #[test]
    fn end_to_end_preprocess_yields_expected_shape() {
        let img = solid(100, 50, [0.5, 0.5, 0.5]);
        let cfg = PreprocessConfig::vit_b16();
        let mut resized = bilinear_resize(&img, cfg.target_width, cfg.target_height).unwrap();
        normalize(&mut resized, cfg.mean, cfg.std);
        let patches = patchify(&resized, cfg.patch_size).unwrap();
        let expected_n = (cfg.target_width / cfg.patch_size) * (cfg.target_height / cfg.patch_size);
        assert_eq!(patches.len(), expected_n); // 14*14 = 196
        let expected_dim = cfg.patch_size * cfg.patch_size * 3;
        assert_eq!(patches[0].len(), expected_dim); // 768
    }
}
