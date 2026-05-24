//! Quantization scheme trait + the catalogue of numeric kinds.

use serde::{Deserialize, Serialize};

/// The numeric kinds known to EOC. Bits-per-weight is the energy axis;
/// these values are what we route on.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Numeric {
    /// IEEE 754 binary32.
    Fp32,
    /// IEEE 754 binary16.
    Fp16,
    /// bfloat16.
    Bf16,
    /// OFP8 E4M3 (Nvidia Hopper training/inference).
    Fp8E4m3,
    /// OFP8 E5M2 (Nvidia Hopper training).
    Fp8E5m2,
    /// 8-bit integer, symmetric or asymmetric.
    Int8,
    /// 4-bit grouped integer (GPTQ / AWQ family).
    Int4,
    /// NormalFloat4 (QLoRA).
    Nf4,
    /// 2-bit ternary / sign-magnitude (BitNet family).
    Int2,
}

impl Numeric {
    /// Bits per weight stored on disk (excluding metadata / scales).
    pub fn bits_per_weight(self) -> u32 {
        match self {
            Self::Fp32 => 32,
            Self::Fp16 | Self::Bf16 => 16,
            Self::Fp8E4m3 | Self::Fp8E5m2 | Self::Int8 => 8,
            Self::Int4 | Self::Nf4 => 4,
            Self::Int2 => 2,
        }
    }

    /// Short label.
    pub fn label(self) -> &'static str {
        match self {
            Self::Fp32 => "fp32",
            Self::Fp16 => "fp16",
            Self::Bf16 => "bf16",
            Self::Fp8E4m3 => "fp8-e4m3",
            Self::Fp8E5m2 => "fp8-e5m2",
            Self::Int8 => "int8",
            Self::Int4 => "int4",
            Self::Nf4 => "nf4",
            Self::Int2 => "int2",
        }
    }
}

/// Abstract quantization scheme: round-trip between dense `f32` weights
/// and a packed encoded form, with associated metadata (scales, zero
/// points, etc.).
pub trait QuantizationScheme {
    /// Encoded representation produced by `quantize`.
    type Encoded;

    /// The numeric kind this scheme produces.
    fn numeric(&self) -> Numeric;

    /// Encode a slice of fp32 weights into the packed form.
    fn quantize(&self, weights: &[f32]) -> Self::Encoded;

    /// Decode the packed form back to fp32.
    fn dequantize(&self, encoded: &Self::Encoded) -> Vec<f32>;
}
