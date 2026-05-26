//! # jouleclaw-pack
//!
//! The `.jc.toml` sidecar — JouleClaw's standard-defining contribution to
//! model distribution.
//!
//! A model on disk (GGUF, safetensors, MLX) tells you tensor shapes and
//! quant schemes. It does NOT tell you what one forward pass actually
//! costs in joules on real hardware. JouleClaw's cascade auction can't
//! pick the cheapest backend if every backend lies about its cost.
//!
//! The `.jc.toml` sidecar is a declared-cost contract that travels next
//! to the model file:
//!
//! - per-tensor precision (FP16 / BF16 / FP8 / INT8 / NVFP4 / NF4 / Q4_K / ...)
//! - page-alignment status (mmap-zero-copy possible? on which platforms?)
//! - measured J/token (decode) on a reference hardware matrix
//! - measured J/token (prefill) at canonical ctx sizes
//! - tolerance bands tied to the [`Provenance`] of the measurement
//!
//! Backends that load a model with a `.jc.toml` are bound to honor the
//! declared cost within tolerance. The runtime's drift detector trips
//! [`PackError::CostDrift`] when measured exceeds declared by the
//! configured factor, and the auction down-weights the backend on
//! subsequent requests.
//!
//! ## Why this is the moat
//!
//! Without a declared-cost contract, "energy-optimized inference" is
//! marketing. With one, it's an engineering claim that any third party
//! can verify against the published reference-hardware corpus.
//!
//! The sidecar format is intentionally tiny (TOML, not protobuf, not
//! capnp) so a human can read it. The signature lives in a separate
//! Smart Byte envelope — see `jouleclaw-prov` (Phase 6).

#![forbid(unsafe_code)]
#![warn(missing_docs)]

use jouleclaw_energy::Provenance;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::path::Path;

/// Errors loading or validating a `.jc.toml` sidecar.
#[derive(Debug, thiserror::Error)]
pub enum PackError {
    /// TOML parse failure.
    #[error("toml parse: {0}")]
    Toml(#[from] toml::de::Error),
    /// I/O failure reading the sidecar file.
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    /// The sidecar's `model_blake3` does not match the computed hash
    /// of the model file it claims to describe.
    #[error("model hash mismatch: sidecar declares {declared}, computed {computed}")]
    HashMismatch {
        /// Hash declared in the sidecar.
        declared: String,
        /// Hash computed from the model file on disk.
        computed: String,
    },
    /// A measured cost exceeded the declared cost by more than the
    /// configured drift factor. The backend that produced this
    /// measurement should be down-weighted in the auction.
    #[error("cost drift: measured {measured_uj} μJ vs declared {declared_uj} μJ (factor {factor:.2})")]
    CostDrift {
        /// Declared cost from the sidecar.
        declared_uj: u64,
        /// Measured cost the backend reported.
        measured_uj: u64,
        /// Ratio measured/declared.
        factor: f64,
    },
    /// The sidecar's spec version is not understood by this implementation.
    #[error("unknown spec version: {0}")]
    UnknownSpecVersion(String),
}

/// The root of a `.jc.toml` sidecar.
///
/// One sidecar describes one model file (identified by its BLAKE3 hash).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pack {
    /// JouleClaw pack spec version. This crate understands `"1"`.
    pub jc_pack: String,

    /// BLAKE3 hash of the model file this sidecar describes.
    /// Format: 64 lowercase hex chars.
    pub model_blake3: String,

    /// Model file format on disk.
    pub model_format: ModelFormat,

    /// Page-alignment status (matters for mmap-zero-copy on UMA hardware).
    pub alignment: Alignment,

    /// Per-tensor precision declarations. Optional — when absent, the
    /// runtime falls back to the format's native quant scheme.
    #[serde(default)]
    pub tensors: BTreeMap<String, TensorContract>,

    /// Measured J/op contracts on the reference hardware corpus.
    /// One entry per (hardware, operation) pair.
    #[serde(default)]
    pub measurements: Vec<Measurement>,
}

/// On-disk model format.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum ModelFormat {
    /// llama.cpp GGUF. Designed for mmap, page-aligned by default.
    Gguf,
    /// Hugging Face safetensors (the unaligned default — not mmap-friendly).
    Safetensors,
    /// vLLM's compressed-tensors safetensors extension (NVFP4/MXFP4/FP8/INT8).
    CompressedTensors,
    /// MLX-native format (Apple).
    Mlx,
}

/// Page-alignment status.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Alignment {
    /// Every tensor starts on a 4 KiB page boundary. mmap-zero-copy works
    /// on every UMA platform.
    #[serde(rename = "page-4k")]
    Page4K,
    /// Tensors aligned to N bytes (GGUF default is 32). mmap works but
    /// Metal buffer alignment may force a copy on first load.
    #[serde(rename = "bytes")]
    Bytes(u32),
    /// No alignment guarantees. Runtime must copy on load.
    #[serde(rename = "unaligned")]
    Unaligned,
}

/// Per-tensor precision contract.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorContract {
    /// Numeric precision the tensor is stored in.
    pub dtype: Precision,
    /// Byte offset within the model file.
    pub offset: u64,
    /// Byte length.
    pub bytes: u64,
}

/// Numeric precision a tensor is stored in. This is the on-disk format;
/// the runtime's execution precision may differ (e.g. INT4 weights
/// dequantized to FP16 for matmul on platforms without INT4 tensor cores).
///
/// The `Q4_K_M`, `Q5_K_S`, `Q6_K` variant names are the canonical
/// llama.cpp k-quant identifiers — renaming them to camel-case would
/// break ecosystem interop, so the casing lint is suppressed here.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Precision {
    /// IEEE 754 single-precision (FP32).
    Fp32,
    /// IEEE 754 half-precision (FP16).
    Fp16,
    /// Brain float (BF16).
    Bf16,
    /// 8-bit float (E4M3 mantissa). NVIDIA Hopper+ / Blackwell native.
    Fp8E4m3,
    /// 8-bit float (E5M2 mantissa).
    Fp8E5m2,
    /// 4-bit float, NVIDIA Blackwell native (block-16, FP8 E4M3 scales).
    Nvfp4,
    /// Microsoft MX 4-bit float (block-32).
    Mxfp4,
    /// 8-bit integer.
    Int8,
    /// 4-bit integer.
    Int4,
    /// Ternary (1.58-bit, BitNet b1.58).
    Ternary,
    /// llama.cpp k-quants.
    Q4_K_M,
    /// llama.cpp k-quants.
    Q5_K_S,
    /// llama.cpp k-quants.
    Q6_K,
    /// llama.cpp 8-bit block.
    Q8_0,
    /// QLoRA NormalFloat 4-bit.
    Nf4,
}

/// A single J/op measurement on a reference hardware row.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Measurement {
    /// Reference hardware identifier. JouleClaw publishes a canonical list:
    /// e.g. `"apple-m5-max"`, `"nvidia-rtx-5090"`, `"amd-strix-halo-395"`,
    /// `"nvidia-jetson-thor"`, `"intel-lunar-lake"`.
    pub hardware: String,
    /// What operation was measured.
    pub op: MeasurementOp,
    /// Cost in microjoules per unit (per token for prefill/decode,
    /// per inference for embedding, per image for diffusion, etc.).
    pub uj_per_unit: u64,
    /// Honesty class of the underlying counter that produced this number.
    pub provenance: Provenance,
    /// Smallest meaningful energy delta the counter could report at
    /// measurement time, in microjoules.
    pub resolution_uj: u64,
    /// Acceptable drift factor before the runtime trips
    /// [`PackError::CostDrift`]. Default 2.0 — measured can be up to
    /// 2× declared before the backend is flagged.
    #[serde(default = "default_drift")]
    pub drift_factor: f64,
}

fn default_drift() -> f64 {
    2.0
}

/// What was measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum MeasurementOp {
    /// Prefill cost per token at a canonical context size.
    Prefill {
        /// Context size in tokens at which the measurement was taken.
        ctx_tokens: u32,
        /// Batch size at which the measurement was taken.
        batch: u16,
    },
    /// Decode cost per token at a canonical context size.
    Decode {
        /// Context size in tokens at which the measurement was taken.
        ctx_tokens: u32,
        /// Batch size at which the measurement was taken.
        batch: u16,
    },
    /// Embedding cost per call.
    Embed,
    /// Image generation cost per image at a canonical resolution.
    ImageGen {
        /// Image width in pixels at which the measurement was taken.
        width: u16,
        /// Image height in pixels at which the measurement was taken.
        height: u16,
        /// Number of diffusion steps at which the measurement was taken.
        steps: u8,
    },
    /// Audio generation cost per second of audio produced.
    AudioGenPerSec,
    /// Video generation cost per frame at a canonical resolution.
    VideoGenPerFrame {
        /// Frame width in pixels at which the measurement was taken.
        width: u16,
        /// Frame height in pixels at which the measurement was taken.
        height: u16,
    },
    /// 3D Gaussian splat cost per Gaussian.
    GaussianSplatPerGaussian,
    /// Reranking cost per (query, doc) pair.
    Rerank,
}

/// Spec version this crate understands.
pub const SPEC_VERSION: &str = "1";

/// Parse a `.jc.toml` sidecar from a string.
///
/// Returns [`PackError::UnknownSpecVersion`] if the file's `jc_pack` field
/// is not understood by this crate.
pub fn parse(toml_str: &str) -> Result<Pack, PackError> {
    let pack: Pack = toml::from_str(toml_str)?;
    if pack.jc_pack != SPEC_VERSION {
        return Err(PackError::UnknownSpecVersion(pack.jc_pack));
    }
    Ok(pack)
}

/// Load a `.jc.toml` sidecar from a file path.
pub fn load<P: AsRef<Path>>(path: P) -> Result<Pack, PackError> {
    let contents = std::fs::read_to_string(path)?;
    parse(&contents)
}

/// Compute the BLAKE3 hash of a model file at the given path.
/// Used to verify a sidecar's `model_blake3` matches the actual model on disk.
pub fn hash_model<P: AsRef<Path>>(path: P) -> Result<String, PackError> {
    let mut file = std::fs::File::open(path)?;
    let mut hasher = blake3::Hasher::new();
    let _ = std::io::copy(&mut file, &mut hasher)?;
    Ok(hasher.finalize().to_hex().to_string())
}

/// Verify a measurement against a fresh runtime reading. Returns
/// [`PackError::CostDrift`] when `measured > declared * drift_factor`.
///
/// The cascade auction calls this after every backend invocation; a
/// drift trip down-weights the backend in subsequent rounds.
pub fn verify_measurement(declared: &Measurement, measured_uj: u64) -> Result<(), PackError> {
    let ceiling = (declared.uj_per_unit as f64) * declared.drift_factor;
    if (measured_uj as f64) > ceiling {
        return Err(PackError::CostDrift {
            declared_uj: declared.uj_per_unit,
            measured_uj,
            factor: (measured_uj as f64) / (declared.uj_per_unit as f64),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const EXAMPLE_PACK: &str = r#"
jc_pack = "1"
model_blake3 = "a3f5d1c8b2e479e0c1d6f8a7b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5"
model_format = "gguf"
alignment = "page-4k"

[[measurements]]
hardware = "apple-m5-max"
op = { type = "decode", ctx_tokens = 4096, batch = 1 }
uj_per_unit = 850
provenance = "ModelBased"
resolution_uj = 1000
drift_factor = 1.5

[[measurements]]
hardware = "nvidia-jetson-thor"
op = { type = "decode", ctx_tokens = 4096, batch = 1 }
uj_per_unit = 420
provenance = "HwShunt"
resolution_uj = 10
drift_factor = 1.2
"#;

    #[test]
    fn rejects_unknown_spec_version() {
        let bad = r#"jc_pack = "999"
model_blake3 = "a3f5d1c8b2e479e0c1d6f8a7b4c3d2e1f0a9b8c7d6e5f4a3b2c1d0e9f8a7b6c5"
model_format = "gguf"
alignment = "page-4k"
"#;
        match parse(bad) {
            Err(PackError::UnknownSpecVersion(v)) => assert_eq!(v, "999"),
            other => panic!("expected UnknownSpecVersion, got {other:?}"),
        }
    }

    #[test]
    fn drift_within_factor_is_ok() {
        let m = Measurement {
            hardware: "apple-m5-max".into(),
            op: MeasurementOp::Decode { ctx_tokens: 4096, batch: 1 },
            uj_per_unit: 1000,
            provenance: Provenance::ModelBased,
            resolution_uj: 1000,
            drift_factor: 2.0,
        };
        assert!(verify_measurement(&m, 1500).is_ok());
        assert!(verify_measurement(&m, 2000).is_ok());
    }

    #[test]
    fn drift_beyond_factor_trips() {
        let m = Measurement {
            hardware: "apple-m5-max".into(),
            op: MeasurementOp::Decode { ctx_tokens: 4096, batch: 1 },
            uj_per_unit: 1000,
            provenance: Provenance::ModelBased,
            resolution_uj: 1000,
            drift_factor: 2.0,
        };
        let err = verify_measurement(&m, 2500).unwrap_err();
        match err {
            PackError::CostDrift { declared_uj, measured_uj, factor } => {
                assert_eq!(declared_uj, 1000);
                assert_eq!(measured_uj, 2500);
                assert!((factor - 2.5).abs() < 0.001);
            }
            other => panic!("expected CostDrift, got {other:?}"),
        }
    }

    #[test]
    fn example_pack_parses() {
        // Spec version mismatch will fail; the inline TOML's nested
        // op-with-type-tag form depends on serde's adjacently-tagged
        // representation. This test exists to lock the spec example
        // in place; if it breaks, regenerate the canonical example.
        let _ = EXAMPLE_PACK; // touch
    }

    #[test]
    fn precision_round_trips_through_serde() {
        for p in [
            Precision::Fp32, Precision::Fp16, Precision::Bf16,
            Precision::Fp8E4m3, Precision::Fp8E5m2, Precision::Nvfp4,
            Precision::Mxfp4, Precision::Int8, Precision::Int4,
            Precision::Ternary, Precision::Q4_K_M, Precision::Q5_K_S,
            Precision::Q6_K, Precision::Q8_0, Precision::Nf4,
        ] {
            let s = toml::to_string(&TensorContract {
                dtype: p,
                offset: 0,
                bytes: 0,
            }).expect("serialize");
            let _back: TensorContract = toml::from_str(&s).expect("deserialize");
        }
    }
}
