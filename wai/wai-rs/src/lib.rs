//! WAI — Web AI media transport + execution.
//!
//! What this crate IS:
//!   - the **envelope** (length-prefixed magic + JSON manifest + payload)
//!   - the **capability dispatch** model (capability is named, not shipped)
//!   - thin, well-tested **wrappers around SOTA standard codecs** for
//!     the zeroth condition (AVIF / JPEG-XL / PNG / Opus / FLAC / AV1 /
//!     zstd / XZ — all royalty-free, all production-grade, all the same
//!     libraries every modern media stack uses)
//!   - the **FFI surface** so this Rust crate is the SDK other languages
//!     consume (C ABI via `extern "C"`, cdylib/staticlib outputs)
//!
//! What this crate is NOT:
//!   - a from-scratch codec implementation. The zeroth-condition codecs
//!     are wrappers around the field's SOTA libraries. The standard's
//!     value is the envelope + capability model, not "rewrite JPEG".
//!
//! Neural capabilities (`wai.neural.*`) are declared here so the
//! dispatch table knows them; the actual neural model runs in whatever
//! ML runtime the sink has installed (PyTorch / Core ML / ONNX / etc.).

pub mod codecs;
pub mod container;
pub mod ffi;

#[cfg(feature = "neural")]
pub mod neural;
#[cfg(feature = "neural")]
pub mod ffi_neural;

pub use codecs::{
    sink_capabilities,
    CAP_AUDIO_FLAC, CAP_AUDIO_OPUS,
    CAP_IMAGE_AVIF, CAP_IMAGE_JPEG, CAP_IMAGE_JXL, CAP_IMAGE_PNG,
    CAP_NEURAL_BMSHJ2018, CAP_NEURAL_DAC, CAP_NEURAL_DCVC_RT, CAP_NEURAL_ENCODEC32,
    CAP_NEURAL_GLC, CAP_NEURAL_MIMI, CAP_NEURAL_VIDEO_BMSHJ2018, CAP_NEURAL_WAVTOKENIZER,
    CAP_TEXT_XZ, CAP_TEXT_ZSTD,
    CAP_VIDEO_AV1, CAP_VIDEO_AV1_LOSSLESS,
};
pub use container::{Conditioning, Manifest, ModelRequirement, Wai};

#[cfg(test)]
mod integration_tests {
    use super::*;

    #[test]
    fn end_to_end_image_jxl_lossless() {
        // build an image, encode as JXL lossless, wrap in a WAI envelope,
        // round-trip the envelope, decode, byte-compare.
        let h = 64u32;
        let w = 64u32;
        let mut rgb = vec![0u8; (h * w * 3) as usize];
        for i in 0..h as usize {
            for j in 0..w as usize {
                let p = (i * w as usize + j) * 3;
                rgb[p] = (i * 4) as u8;
                rgb[p + 1] = (j * 4) as u8;
                rgb[p + 2] = ((i + j) * 2) as u8;
            }
        }
        let payload = codecs::image::jxl_encode(&rgb, h, w, None).unwrap();
        let manifest = Manifest {
            wai: "1.0".into(),
            media: "image".into(),
            intent: "replicate".into(),
            model_requirement: ModelRequirement {
                capability: CAP_IMAGE_JXL.into(),
                fallback: Some(CAP_IMAGE_PNG.into()),
            },
            conditioning: Conditioning { kind: "jxl".into() },
            target: serde_json::json!({"h": h, "w": w}),
        };
        let wai = Wai::new(manifest, payload);
        let bytes = wai.to_bytes().unwrap();
        let recovered = Wai::from_bytes(&bytes).unwrap();
        assert_eq!(recovered.capability(), CAP_IMAGE_JXL);
        let (rec_rgb, rh, rw) = codecs::image::jxl_decode(&recovered.payload).unwrap();
        assert_eq!((rh, rw), (h, w));
        assert_eq!(rec_rgb, rgb, "JXL lossless via WAI envelope must round-trip exactly");
    }
}
