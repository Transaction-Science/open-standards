//! Registered zeroth-condition codec capabilities.
//!
//! The WAI thesis: the **envelope** + **capability dispatch** is the
//! standard; the codecs themselves are the SOTA libraries every modern
//! media system already uses. The zeroth condition is a *menu* of
//! always-implementable standard capabilities — pick any one and a sink
//! with the corresponding library can open the file. No hand-rolled
//! codec lives in this crate.

pub mod audio;
pub mod image;
pub mod text;
pub mod video;

// ---- Capability strings (the values that go in manifest.model_requirement.capability)
// Image
pub const CAP_IMAGE_PNG:  &str = "wai.image.png";       // universal floor
pub const CAP_IMAGE_JPEG: &str = "wai.image.jpeg";      // most-supported lossy
pub const CAP_IMAGE_AVIF: &str = "wai.image.avif";      // modern lossy (AV1-based)
pub const CAP_IMAGE_JXL:  &str = "wai.image.jxl";       // JPEG-XL (lossless + lossy)
// Audio
pub const CAP_AUDIO_OPUS: &str = "wai.audio.opus";      // modern lossy
pub const CAP_AUDIO_FLAC: &str = "wai.audio.flac";      // lossless
// Video
pub const CAP_VIDEO_AV1:          &str = "wai.video.av1";          // lossy AV1
pub const CAP_VIDEO_AV1_LOSSLESS: &str = "wai.video.av1.lossless"; // lossless AV1
// Text
pub const CAP_TEXT_ZSTD: &str = "wai.text.zstd";        // general-purpose
pub const CAP_TEXT_XZ:   &str = "wai.text.xz";          // maximum classical ratio
// Neural (declared here so the dispatcher sees them; impls live at the sink).
// See SPEC.md §5 "Neural capabilities" for the SOTA selection rationale.
pub const CAP_NEURAL_ENCODEC32:    &str = "wai.neural.encodec32";    // Meta EnCodec, 32 kHz
pub const CAP_NEURAL_DAC:          &str = "wai.neural.dac";          // Descript Audio Codec, 44.1 kHz
pub const CAP_NEURAL_MIMI:         &str = "wai.neural.mimi";         // Kyutai Mimi, real-time speech
pub const CAP_NEURAL_WAVTOKENIZER: &str = "wai.neural.wavtokenizer"; // ultra-low bitrate audio
pub const CAP_NEURAL_BMSHJ2018:       &str = "wai.neural.bmshj2018";       // bmshj2018-factorized image codec
pub const CAP_NEURAL_VIDEO_BMSHJ2018: &str = "wai.neural.video_bmshj2018"; // per-frame bmshj2018 video
pub const CAP_NEURAL_GLC:             &str = "wai.neural.glc";             // (future) GLC ultra-low bpp images
pub const CAP_NEURAL_DCVC_RT:         &str = "wai.neural.dcvc_rt";         // DCVC-RT (native-sink only, requires CUDA)

/// Returns the set of capabilities this sink (the WAI Rust crate)
/// supports natively. A wrapping application is free to advertise
/// additional neural capabilities backed by its own ML runtime.
pub fn sink_capabilities() -> &'static [&'static str] {
    &[
        CAP_IMAGE_PNG,
        CAP_IMAGE_JPEG,
        CAP_IMAGE_AVIF,
        CAP_IMAGE_JXL,
        CAP_AUDIO_OPUS,
        CAP_AUDIO_FLAC,
        CAP_VIDEO_AV1,
        CAP_VIDEO_AV1_LOSSLESS,
        CAP_TEXT_ZSTD,
        CAP_TEXT_XZ,
    ]
}
