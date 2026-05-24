//! KV-cache quantization (int8 K / int8 V).
//!
//! For long-context inference the KV cache dominates memory and
//! bandwidth, so storing K/V in int8 (or int4) is a big win. This
//! module provides per-head, per-token symmetric int8 quantization for
//! K and V tensors and a small `QuantKvCache` container.

use crate::error::QuantError;
use crate::int8::{Int8SymEncoded, Int8Symmetric};
use crate::scheme::QuantizationScheme;

/// One quantized K (or V) entry: a tensor slice with its scale.
#[derive(Debug, Clone)]
pub struct QuantKvEntry {
    /// Per-slice scale.
    pub scale: f32,
    /// Int8 values.
    pub q: Vec<i8>,
}

impl From<Int8SymEncoded> for QuantKvEntry {
    fn from(e: Int8SymEncoded) -> Self {
        Self {
            scale: e.scale,
            q: e.q,
        }
    }
}

/// Container of int8-quantized K and V tensors, one entry per
/// (layer, head) pair indexed externally.
#[derive(Debug, Default, Clone)]
pub struct QuantKvCache {
    /// K entries in insertion order.
    pub keys: Vec<QuantKvEntry>,
    /// V entries in insertion order.
    pub values: Vec<QuantKvEntry>,
}

impl QuantKvCache {
    /// Empty cache.
    pub fn new() -> Self {
        Self::default()
    }

    /// Push a K tensor (quantized symmetric int8).
    pub fn push_key(&mut self, k_fp: &[f32]) -> Result<(), QuantError> {
        let enc = Int8Symmetric::new().try_quantize(k_fp)?;
        self.keys.push(enc.into());
        Ok(())
    }

    /// Push a V tensor (quantized symmetric int8).
    pub fn push_value(&mut self, v_fp: &[f32]) -> Result<(), QuantError> {
        let enc = Int8Symmetric::new().try_quantize(v_fp)?;
        self.values.push(enc.into());
        Ok(())
    }

    /// Dequantize the i-th K back to fp32.
    pub fn key_fp32(&self, i: usize) -> Vec<f32> {
        self.keys
            .get(i)
            .map(|e| {
                let enc = Int8SymEncoded {
                    scale: e.scale,
                    q: e.q.clone(),
                };
                Int8Symmetric::new().dequantize(&enc)
            })
            .unwrap_or_default()
    }

    /// Dequantize the i-th V back to fp32.
    pub fn value_fp32(&self, i: usize) -> Vec<f32> {
        self.values
            .get(i)
            .map(|e| {
                let enc = Int8SymEncoded {
                    scale: e.scale,
                    q: e.q.clone(),
                };
                Int8Symmetric::new().dequantize(&enc)
            })
            .unwrap_or_default()
    }

    /// Total stored bytes (excluding scales) — useful for the meter.
    pub fn bytes(&self) -> usize {
        self.keys.iter().map(|e| e.q.len()).sum::<usize>()
            + self.values.iter().map(|e| e.q.len()).sum::<usize>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn push_and_dequant_kv_pair() {
        let mut cache = QuantKvCache::new();
        let k: Vec<f32> = (0..16).map(|i| (i as f32 - 8.0) * 0.1).collect();
        let v: Vec<f32> = (0..16).map(|i| (i as f32) * 0.05).collect();
        cache.push_key(&k).expect("push k");
        cache.push_value(&v).expect("push v");
        let kr = cache.key_fp32(0);
        let vr = cache.value_fp32(0);
        assert_eq!(kr.len(), k.len());
        assert_eq!(vr.len(), v.len());
        assert!(cache.bytes() == 32);
    }
}
